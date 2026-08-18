"""Minimal paged KV cache, block_size=64 (= one verification tree, spec §3.5).

Layout: k/v of shape [num_blocks, block_size, num_kv_heads, head_dim] per layer.
Each sequence owns `block_table` (list of block ids). The last, partially filled
block always reserves the *tree scratch region*: verification writes tree-node
KV into a dedicated scratch block; op4's commit remaps only accepted slots into
the sequence's tail block (index remap, no bulk copy).
"""

from __future__ import annotations

import torch

from treemoe.model.config import MixtralConfig

BLOCK_SIZE = 64


class PagedKVCache:
    def __init__(
        self,
        config: MixtralConfig,
        num_blocks: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.block_size = BLOCK_SIZE
        shape = (
            config.num_layers,
            num_blocks,
            BLOCK_SIZE,
            config.num_kv_heads,
            config.head_dim,
        )
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.block_table: list[int] = []
        self.seq_len = 0
        self._free = list(range(num_blocks - 1, 0, -1))  # block 0 = tree scratch
        self.tree_block = 0

    # ---------------- allocation ----------------

    def _ensure_capacity(self, new_len: int) -> None:
        while len(self.block_table) * self.block_size < new_len:
            if not self._free:
                raise RuntimeError("KV cache out of blocks")
            self.block_table.append(self._free.pop())

    def slot_of(self, pos: int) -> tuple[int, int]:
        return self.block_table[pos // self.block_size], pos % self.block_size

    # ---------------- prefill / AR append ----------------

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """Write [T, kv_heads, head_dim] at positions [start_pos, start_pos+T)."""
        t = k.shape[0]
        self._ensure_capacity(start_pos + t)
        for i in range(t):  # reference impl; kernelized later with a scatter
            blk, off = self.slot_of(start_pos + i)
            self.k[layer, blk, off] = k[i]
            self.v[layer, blk, off] = v[i]
        self.seq_len = max(self.seq_len, start_pos + t)

    # ---------------- tree verification path ----------------

    def write_tree(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write all N tree-node KV into the scratch block (positions 0..N-1)."""
        n = k.shape[0]
        assert n <= self.block_size
        self.k[layer, self.tree_block, :n] = k
        self.v[layer, self.tree_block, :n] = v

    def commit_tree(self, accepted_slots: torch.Tensor) -> None:
        """op4 step 5 (reference): remap accepted tree slots into the main cache.

        accepted_slots: LongTensor[m] of tree-node indices along the accepted
        path, in sequence order. GPU-side only; no CPU readback of m — callers
        pass a fixed-size buffer with -1 padding.
        """
        valid = accepted_slots[accepted_slots >= 0]
        m = int(valid.numel())  # graph-safe variant uses masked scatter instead
        if m == 0:
            return
        self._ensure_capacity(self.seq_len + m)
        for j in range(m):
            blk, off = self.slot_of(self.seq_len + j)
            self.k[:, blk, off] = self.k[:, self.tree_block, valid[j]]
            self.v[:, blk, off] = self.v[:, self.tree_block, valid[j]]
        self.seq_len += m

    # ---------------- gather for attention ----------------

    def gather(self, layer: int, upto: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize contiguous [T, kv_heads, head_dim] (reference attention path)."""
        t = self.seq_len if upto is None else upto
        ks, vs = [], []
        for pos in range(t):
            blk, off = self.slot_of(pos)
            ks.append(self.k[layer, blk, off])
            vs.append(self.v[layer, blk, off])
        if not ks:
            empty = self.k.new_zeros((0, self.config.num_kv_heads, self.config.head_dim))
            return empty, empty.clone()
        return torch.stack(ks), torch.stack(vs)

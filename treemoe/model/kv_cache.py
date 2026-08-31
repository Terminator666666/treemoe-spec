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
        """Write [T, kv_heads, head_dim] at positions [start_pos, start_pos+T).

        Single advanced-indexing scatter (block/offset indices computed on the
        host from block_table — no D2H); the old per-position Python loop cost
        T tiny strided writes x 32 layers per forward."""
        t = k.shape[0]
        self._ensure_capacity(start_pos + t)
        bs = self.block_size
        blks = torch.tensor([self.block_table[(start_pos + i) // bs] for i in range(t)],
                            device=k.device, dtype=torch.long)
        offs = torch.tensor([(start_pos + i) % bs for i in range(t)],
                            device=k.device, dtype=torch.long)
        self.k[layer, blks, offs] = k
        self.v[layer, blks, offs] = v
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
        m = int(valid.numel())  # reference path only; the GPU/graph path is
        # op4's _kv_commit_kernel (fixed grid, num_accepted read on device)
        if m == 0:
            return
        self._ensure_capacity(self.seq_len + m)
        dest = [self.slot_of(self.seq_len + j) for j in range(m)]
        blks = torch.tensor([d[0] for d in dest], device=valid.device, dtype=torch.long)
        offs = torch.tensor([d[1] for d in dest], device=valid.device, dtype=torch.long)
        self.k[:, blks, offs] = self.k[:, self.tree_block, valid]
        self.v[:, blks, offs] = self.v[:, self.tree_block, valid]
        self.seq_len += m

    # ---------------- gather for attention ----------------

    def gather(self, layer: int, upto: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize contiguous [T, kv_heads, head_dim] (reference attention path).

        One advanced-indexing op over whole blocks (block_table maps logical
        block i -> physical block), then trim the tail: single device copy per
        call instead of the old per-position Python loop + T-way stack
        (O(seq_len) host iterations x 32 layers x every step)."""
        t = self.seq_len if upto is None else upto
        nblk = (t + self.block_size - 1) // self.block_size
        idx = torch.tensor(self.block_table[:nblk], device=self.k.device, dtype=torch.long)
        kvh, hd = self.config.num_kv_heads, self.config.head_dim
        k = self.k[layer, idx].reshape(nblk * self.block_size, kvh, hd)[:t]
        v = self.v[layer, idx].reshape(nblk * self.block_size, kvh, hd)[:t]
        return k, v

"""Op4: fused verify -> sample -> KV-compact -> commit (spec §3.4).

Kernel layout (HPC-Ops fused-sampler style, extended to trees):
  K1 _postprocess_softmax_kernel : temperature/repetition-penalty + online
     softmax over vocab, one program per tree node (parallel over N).
  K2 _tree_verify_kernel : single-program sequential DFS walk (the accept chain
     is inherently serial, ~depth<=6 iterations). Consumes precomputed probs and
     a Philox-generated uniform per node; writes accepted_slots (-1 padded),
     bonus token, num_accepted, next_root_slot — all GPU-resident.
  K3 _kv_commit_kernel : remap accepted tree-scratch KV slots into the main
     paged cache tail, parallel over (layer, slot, head*dim).

Randomness: torch.cuda philox via torch.rand under a fixed generator OUTSIDE
the graph would break replay; instead uniforms are produced by a counter-based
kernel-side hash seeded from (seed, step_counter) so graph replays draw fresh
randomness (Philox contract, spec §3.5). For reference parity tests, uniforms
can be injected explicitly.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_INTERPRET = os.getenv("TRITON_INTERPRET", "0") == "1"

from treemoe.ref.verify_ref import VerifyResult, tree_verify_ref

VBLOCK = 1024


if HAS_TRITON:

    @triton.jit
    def _postprocess_softmax_kernel(
        logits_ptr, probs_ptr, prev_tokens_ptr,
        V: tl.constexpr, VB: tl.constexpr,
        temperature, rep_penalty, num_prev,
    ):
        node = tl.program_id(0)
        base = node.to(tl.int64) * V

        # pass 0: repetition penalty on previously generated tokens (small list)
        for i in range(0, num_prev):
            t = tl.load(prev_tokens_ptr + i)
            lg = tl.load(logits_ptr + base + t)
            lg = tl.where(lg > 0, lg / rep_penalty, lg * rep_penalty)
            tl.store(logits_ptr + base + t, lg)

        # pass 1: ONLINE max+sum (Milakov-Gimelshein; FlashInfer sampling
        # kernels use the same trick) — one vocab sweep instead of two:
        # when the running max rises, rescale the accumulated sum.
        # vmax init is FINITE: -inf would give exp(-inf - -inf) = NaN if a
        # block is fully masked / all -inf
        vmax = -1e38
        vsum = 0.0
        for v0 in range(0, V, VB):
            offs = v0 + tl.arange(0, VB)
            x = tl.load(logits_ptr + base + offs, mask=offs < V, other=-float("inf"))
            x = x / temperature
            bmax = tl.max(x, axis=0)
            nmax = tl.maximum(vmax, bmax)
            vsum = vsum * tl.exp(vmax - nmax) + tl.sum(tl.exp(x - nmax), axis=0)
            vmax = nmax
        # pass 2: normalize + store
        for v0 in range(0, V, VB):
            offs = v0 + tl.arange(0, VB)
            x = tl.load(logits_ptr + base + offs, mask=offs < V, other=-float("inf"))
            tl.store(probs_ptr + base + offs, tl.exp(x / temperature - vmax) / vsum,
                     mask=offs < V)

    @triton.jit
    def _argmax_kernel(x_ptr, out_ptr, V: tl.constexpr, VB: tl.constexpr):
        # first-occurrence argmax over arbitrary values (logits OR probs):
        # init/padding must be -inf, not -1 — logits can be all-negative
        node = tl.program_id(0)
        base = node.to(tl.int64) * V
        best_v = -float("inf")
        best_i = 0
        for v0 in range(0, V, VB):
            offs = v0 + tl.arange(0, VB)
            x = tl.load(x_ptr + base + offs, mask=offs < V, other=-float("inf"))
            m = tl.max(x, axis=0)
            i = tl.argmax(x, axis=0)
            best_i = tl.where(m > best_v, v0 + i, best_i)
            best_v = tl.maximum(best_v, m)
        tl.store(out_ptr + node, best_i)

    @triton.jit
    def _tree_verify_greedy_kernel(
        argmax_ptr, tree_tokens_ptr, child_start_ptr, child_list_ptr, child_count_ptr,
        accepted_slots_ptr, bonus_ptr, num_accepted_ptr, next_root_ptr,
        MAX_DEPTH: tl.constexpr,
    ):
        # single program; serial DFS along the greedy-accept chain
        node = 0
        count = 0
        for _d in range(0, MAX_DEPTH):
            target_top = tl.load(argmax_ptr + node)
            n_kids = tl.load(child_count_ptr + node)
            start = tl.load(child_start_ptr + node)
            accepted = -1
            for j in range(0, n_kids):
                # .to(int32): keep the loop-carried `accepted`/`node` scalars at a
                # fixed int32 type — mixing the i64 load into tl.where would flip
                # the carried type and break the sm_90 compile (found by AOT
                # compile in benchmarks/static_analysis.py; interpreter mode is
                # too permissive to catch it). Node ids < 2N=128, int32 is safe;
                # tl.store casts back to the i64 buffers.
                c = tl.load(child_list_ptr + start + j).to(tl.int32)
                tok = tl.load(tree_tokens_ptr + c)
                hit = (tok == target_top) & (accepted < 0)
                accepted = tl.where(hit, c, accepted)
            go = accepted >= 0
            # buffer pre-filled with -1: writing `accepted` (=-1 on reject) at the
            # stalled cursor is idempotent, so no masked store / divergent `if` needed
            tl.store(accepted_slots_ptr + count, accepted)
            count = count + tl.where(go, 1, 0)
            node = tl.where(go, accepted, node)
        tl.store(bonus_ptr, tl.load(argmax_ptr + node))
        tl.store(num_accepted_ptr, count)
        tl.store(next_root_ptr, node)

    @triton.jit
    def _kv_commit_kernel(
        k_ptr, v_ptr, accepted_slots_ptr, dest_block_ptr, dest_off_ptr,
        num_accepted_ptr,
        tree_block: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        KVH_HD: tl.constexpr,  # num_kv_heads * head_dim
        stride_layer: tl.constexpr, stride_block: tl.constexpr, stride_slot: tl.constexpr,
    ):
        layer = tl.program_id(0)
        j = tl.program_id(1)         # accepted-path position (grid = max_depth)
        m = tl.load(num_accepted_ptr)
        if j >= m:
            return
        src_slot = tl.load(accepted_slots_ptr + j)
        dblk = tl.load(dest_block_ptr + j)
        doff = tl.load(dest_off_ptr + j)
        offs = tl.arange(0, KVH_HD)
        src = layer.to(tl.int64) * stride_layer + tree_block * stride_block + src_slot * stride_slot
        dst = layer.to(tl.int64) * stride_layer + dblk * stride_block + doff * stride_slot
        tl.store(k_ptr + dst + offs, tl.load(k_ptr + src + offs))
        tl.store(v_ptr + dst + offs, tl.load(v_ptr + src + offs))


def fused_verify_commit(
    target_logits: torch.Tensor,     # [N, V] fp32
    draft_probs: torch.Tensor,       # [N, V] fp32
    tree_tokens: torch.Tensor,       # [N]
    tree_parent: torch.Tensor,       # [N]
    children: list[list[int]],
    kv=None,                         # PagedKVCache | None
    temperature: float = 0.0,
    rep_penalty: float = 1.0,
    prev_tokens: torch.Tensor | None = None,
    max_depth: int = 6,
    generator: torch.Generator | None = None,
) -> VerifyResult:
    """GPU-fused path when Triton + CUDA available (greedy mode kernelized);
    sampling mode currently routes through the reference (kernel v2 milestone).
    TRITON_INTERPRET=1 runs the same kernels on CPU via the interpreter.
    """
    use_kernel = (
        HAS_TRITON
        and (target_logits.is_cuda or _INTERPRET)
        and temperature == 0.0
    )
    if not use_kernel:
        res = tree_verify_ref(
            target_logits, draft_probs, tree_tokens, tree_parent, children,
            temperature=temperature, generator=generator,
        )
    else:
        n, v = target_logits.shape
        device = target_logits.device
        argmax = torch.empty(n, dtype=torch.int32, device=device)
        if rep_penalty == 1.0:
            # greedy fast path: softmax is monotone, so argmax(logits) ==
            # argmax(softmax(logits)) exactly — skip the 3-pass softmax
            # (saves 3 vocab sweeps of HBM traffic) and stay bitwise-aligned
            # with the AR baseline's logits.argmax()
            _argmax_kernel[(n,)](target_logits, argmax, V=v, VB=VBLOCK)
        else:
            probs = torch.empty_like(target_logits)
            prev = prev_tokens if prev_tokens is not None else torch.zeros(1, dtype=torch.long, device=device)
            _postprocess_softmax_kernel[(n,)](
                target_logits, probs, prev, V=v, VB=VBLOCK,
                temperature=1.0, rep_penalty=rep_penalty, num_prev=prev.numel(),
            )
            _argmax_kernel[(n,)](probs, argmax, V=v, VB=VBLOCK)

        # flatten children adjacency once per tree shape (static metadata)
        counts = torch.tensor([len(c) for c in children], dtype=torch.int32, device=device)
        starts = torch.zeros(len(children), dtype=torch.int32, device=device)
        starts[1:] = counts.cumsum(0)[:-1].to(torch.int32)
        flat = torch.tensor(
            [c for kids in children for c in kids] or [0], dtype=torch.int32, device=device
        )
        accepted_slots = torch.full((n,), -1, dtype=torch.int32, device=device)
        bonus = torch.zeros((), dtype=torch.int32, device=device)
        num_acc = torch.zeros((), dtype=torch.int32, device=device)
        next_root = torch.zeros((), dtype=torch.int32, device=device)
        _tree_verify_greedy_kernel[(1,)](
            argmax, tree_tokens.to(torch.int32), starts, flat, counts,
            accepted_slots, bonus, num_acc, next_root, MAX_DEPTH=max_depth,
        )
        res = VerifyResult(
            accepted_slots=accepted_slots.long(),
            accepted_tokens=torch.where(
                accepted_slots >= 0, tree_tokens[accepted_slots.clamp(min=0).long()],
                torch.full_like(accepted_slots, -1).long(),
            ),
            bonus_token=bonus.long(),
            num_accepted=num_acc.long(),
        )

    if kv is not None:
        if use_kernel:
            m_max = max_depth
            dest_pos = kv.seq_len + torch.arange(m_max, device=target_logits.device)
            kv._ensure_capacity(kv.seq_len + m_max)
            table = torch.tensor(kv.block_table, device=target_logits.device)
            dest_block = table[dest_pos // kv.block_size].to(torch.int32)
            dest_off = (dest_pos % kv.block_size).to(torch.int32)
            kvh_hd = kv.k.shape[3] * kv.k.shape[4]
            _kv_commit_kernel[(kv.k.shape[0], m_max)](
                kv.k, kv.v, res.accepted_slots.to(torch.int32), dest_block, dest_off,
                res.num_accepted.to(torch.int32),
                tree_block=kv.tree_block, BLOCK_SIZE=kv.block_size, KVH_HD=kvh_hd,
                stride_layer=kv.k.stride(0), stride_block=kv.k.stride(1),
                stride_slot=kv.k.stride(2),
            )
            kv.seq_len += int(res.num_accepted)  # graph mode keeps this on-GPU
        else:
            kv.commit_tree(res.accepted_slots)
    return res

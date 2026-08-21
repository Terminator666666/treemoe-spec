"""Bisect the op1 GPU parity failure (first run on real silicon).

Two paths ran for the first time on hardware in tier 1b:
  (a) fused Kernel A routing  (single-CTA, 32 warps)   -> TREEMOE_FUSED_A=0 disables
  (b) u32-packed weight loads (PACK_W=1, bf16)         -> TREEMOE_PACK_W=0 disables

This script compares the fused routing against the torch route_and_bucket
path buffer-by-buffer, then runs the full forward in all 4 on/off combos
against the reference. Run from anywhere:  python benchmarks/diag_op1.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tests.conftest import make_moe_inputs
from treemoe.ref.tree_moe_ref import tree_moe_forward_ref

N, E, H, I, BUDGET, SEED = 32, 8, 4096, 14336, 4, 7
ATOL, RTOL = 1e-2, 1e-3


def fwd(op1, inputs, fused: bool, pack: bool):
    os.environ["TREEMOE_FUSED_A"] = "1" if fused else "0"
    os.environ["TREEMOE_PACK_W"] = "1" if pack else "0"
    x, w1, w2, w3, router, accept = inputs
    return op1.tree_moe_forward(x, w1, w2, w3, router, accept, BUDGET)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    from treemoe.kernels import op1_tree_moe as op1

    g = torch.Generator().manual_seed(SEED)
    x, w1, w2, w3, router, accept = make_moe_inputs(N, E, H, I, g, dtype=torch.bfloat16)
    x, w1, w2, w3 = (t.cuda() for t in (x, w1, w2, w3))
    router, accept = router.cuda(), accept.cuda()
    inputs = (x, w1, w2, w3, router, accept)
    ref = tree_moe_forward_ref(x, w1, w2, w3, router, accept, BUDGET).float()

    print(f"device={torch.cuda.get_device_name()}  torch={torch.__version__}")

    # ---- 1. routing buffers: fused Kernel A vs torch path ----
    fwd(op1, inputs, fused=True, pack=True)  # populates ws.* via Kernel A
    key = (N, E, H, I, x.device.index, x.dtype)
    ws = op1._ws_cache[key]
    tids, tgates, tslots, tblk, ts2r, _ = op1.route_and_bucket(x, router, accept, BUDGET)
    tgates_flat = tgates.reshape(-1).float()

    print("\n== fused Kernel A vs torch route_and_bucket ==")
    id_eq = torch.equal(ws.topk_flat, tids.reshape(-1))
    print(f"topk ids equal:        {id_eq}"
          + ("" if id_eq else f"   ({(ws.topk_flat != tids.reshape(-1)).sum().item()}/{2*N} slots differ)"))
    gd = (ws.gates_flat - tgates_flat).abs().max().item()
    print(f"gates max|d|:          {gd:.3e}")
    for name, a, b in (("padded_slots", ws.padded_slots, tslots),
                       ("block_expert_ids", ws.block_expert_ids, tblk),
                       ("slot_to_row", ws.slot_to_row, ts2r)):
        print(f"{name:22s} equal: {torch.equal(a, b)}")

    # ---- 2. forward in all 4 combos vs reference ----
    print(f"\n== forward vs ref (n={N} budget={BUDGET} bf16, atol={ATOL} rtol={RTOL}) ==")
    for fused in (True, False):
        for pack in (True, False):
            out = fwd(op1, inputs, fused, pack).float()
            d = (out - ref).abs()
            bad = d > (ATOL + RTOL * ref.abs())
            bad_rows = bad.any(dim=1).sum().item()
            print(f"fused_a={int(fused)} pack_w={int(pack)}: max|d|={d.max().item():.4f}  "
                  f"mismatch={bad.sum().item()}/{ref.numel()}  bad_rows={bad_rows}/{N}")

    os.environ.pop("TREEMOE_FUSED_A", None)
    os.environ.pop("TREEMOE_PACK_W", None)


if __name__ == "__main__":
    main()

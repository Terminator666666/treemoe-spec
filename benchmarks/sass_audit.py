"""Static SASS audit of the op1/op4 kernels (GPU-less, AsmEvo-inspired).

AsmEvo (arXiv:2608.20711) optimizes *compiled* GPU kernels by recovering
assembly, localizing hot instruction windows, and editing under a
differential-correctness gate.  Its NVIDIA-transferable insight: the
categories that actually paid off on production MoE/GEMM kernels were
  * cache-hint / load-width variants
  * scheduling around long-latency memory ops (visible as spill + narrow loads)
  * reduced conversion / address-generation work
All of these are *visible in static SASS* -- no GPU needed to audit them.

This tool AOT-compiles every kernel for sm_89 (the 4090 we actually run on),
disassembles the cubin with nvdisasm, and reports the static signals that
correspond to AsmEvo's edit classes:

  * opcode histogram             -> conversion/address-gen overhead (I2F, CVT,
                                    IMAD-as-adressing vs FFMA/HMMA ratio)
  * LDG/STG width breakdown      -> narrow global loads waste bandwidth; the
                                    streaming GEMMs must be .128 (u32-packing
                                    regression guard!)
  * LDL/STL count                -> register spills reaching local memory
  * eviction-hint survival       -> tl.load(eviction_policy=...) can be
                                    silently dropped by ptxas on some archs
  * BAR.SYNC / branch counts     -> sync and control overhead

SASS is dumped to build/sass/ for manual (or agent) hot-window review.
Timing-based hot-window *selection* still needs silicon (torch.profiler);
this is the static half of the loop.

Usage:  python benchmarks/sass_audit.py [--arch 89] [--kernel <substr>]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_NV_ROOT = "/usr/local/lib/python3.12/dist-packages/nvidia"
_PTXAS_CANDIDATES = glob.glob(f"{_NV_ROOT}/cuda_nvcc/bin/ptxas") + glob.glob(
    os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cuda_nvcc/bin/ptxas"))
if _PTXAS_CANDIDATES:
    os.environ.setdefault("TRITON_PTXAS_PATH", _PTXAS_CANDIDATES[0])
_NVDISASM_CANDIDATES = glob.glob(f"{_NV_ROOT}/cu13/bin/nvdisasm") + glob.glob(
    f"{_NV_ROOT}/cuda_nvcc/bin/nvdisasm") + glob.glob(
    os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/*/bin/nvdisasm"))

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

from static_analysis import _mixtral_specs  # noqa: E402  (reuse kernel specs)

# SASS line:  /*0120*/  LDG.E.128 R4, desc[UR4][R2.64] ;
_SASS_INST = re.compile(r"/\*[0-9a-f]{4,}\*/\s+@?!?P?\d*\s*([A-Z][A-Z0-9._]*)")


def disassemble(spec, arch: int, build_dir: str) -> str | None:
    src = triton.compiler.ASTSource(
        fn=spec.fn, signature=spec.signature, constexprs=spec.constexprs)
    k = triton.compile(
        src, target=GPUTarget("cuda", arch, 32),
        options={"num_warps": spec.num_warps, "num_stages": spec.num_stages})
    cubin = k.asm.get("cubin")
    if not cubin:
        return None
    safe = re.sub(r"[^A-Za-z0-9]+", "_", spec.name).strip("_")
    cubin_path = os.path.join(build_dir, safe + ".cubin")
    with open(cubin_path, "wb") as f:
        f.write(cubin)
    sass = subprocess.run(
        [_NVDISASM_CANDIDATES[0], "-c", cubin_path],
        capture_output=True, text=True).stdout
    with open(os.path.join(build_dir, safe + ".sass"), "w") as f:
        f.write(sass)
    return sass


def audit(name: str, sass: str) -> None:
    ops = [m.group(1) for m in _SASS_INST.finditer(sass)]
    hist = Counter(op.split(".")[0] for op in ops)
    total = len(ops)

    def n(*keys: str) -> int:
        return sum(hist[k] for k in keys)

    # global-load width breakdown (streaming GEMMs must be 128-bit)
    ldg = Counter()
    for op in ops:
        if op.startswith("LDG"):
            m = re.search(r"\.(128|64|32|16|8)(?:\.|$)", op + ".")
            ldg[m.group(1) if m else "32"] += 1
    spill = n("LDL", "STL")
    evict = len(re.findall(r"EF|EL|LU|createpolicy|POLICY", sass))

    print(f"\n== {name} ==  ({total} SASS instructions)")
    top = ", ".join(f"{k}:{v}" for k, v in hist.most_common(8))
    print(f"   top ops : {top}")
    if ldg:
        widths = "  ".join(f".{w}b x{c}" for w, c in sorted(
            ldg.items(), key=lambda t: -int(t[0])))
        narrow = sum(c for w, c in ldg.items() if int(w) < 128)
        flag = "  <-- narrow loads!" if narrow > ldg.get("128", 0) else ""
        print(f"   LDG     : {widths}{flag}")
    print(f"   spill   : {'LDL/STL x%d  <-- local-memory traffic' % spill if spill else 'none'}")
    print(f"   sync    : BAR x{n('BAR')}  branches x{n('BRA', 'BRX')}  "
          f"convert x{n('I2F', 'F2I', 'F2F', 'CVT')}")
    if "evict" in sass or "EF" in sass or evict:
        print(f"   evict   : {evict} policy-marked accesses (hints survived to SASS)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=int, default=89, help="sm arch (default 89 = 4090)")
    ap.add_argument("--kernel", type=str, default="", help="substring filter")
    args = ap.parse_args()
    if not _NVDISASM_CANDIDATES:
        sys.exit("nvdisasm not found (pip install nvidia-cuda-nvcc / cu13 wheel)")

    build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "sass")
    os.makedirs(build_dir, exist_ok=True)
    print(f"arch sm_{args.arch}  nvdisasm: {_NVDISASM_CANDIDATES[0]}")
    for spec in _mixtral_specs():
        if args.kernel and args.kernel.lower() not in spec.name.lower():
            continue
        try:
            sass = disassemble(spec, args.arch, build_dir)
        except Exception as exc:
            print(f"\n== {spec.name} ==  COMPILE FAILED: {type(exc).__name__}: {exc}")
            continue
        if not sass:
            print(f"\n== {spec.name} ==  no cubin (ptxas missing?)")
            continue
        audit(spec.name, sass)
    print(f"\nfull SASS in {os.path.relpath(build_dir)}/ -- grep LDG to inspect "
          f"load widths, STL for spill sites, and instruction gaps for stalls")


if __name__ == "__main__":
    main()

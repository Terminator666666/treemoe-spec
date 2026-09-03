"""PTX-level regression guards (no GPU needed).

Locks in facts established by the static-analysis rounds so future edits
can't silently regress them:
  * all kernels AOT-compile for sm_90a (caught a real int32/i64 bug once)
  * zero register spill at production launch configs
  * L2 eviction hints survive into PTX (Triton may drop them silently)
    * Kernel A consumes exact-HF gates and must not contain a second router GEMM

Skipped when ptxas is unavailable or under TRITON_INTERPRET=1.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_INTERPRET", "0") == "1",
    reason="AOT compile paths differ under the interpreter",
)

_BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")


@pytest.fixture(scope="module")
def sa():
    """Import benchmarks/static_analysis.py (sets TRITON_PTXAS_PATH)."""
    spec = importlib.util.spec_from_file_location(
        "static_analysis", os.path.join(_BENCH, "static_analysis.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["static_analysis"] = mod   # @dataclass needs sys.modules entry
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:  # no triton on this box
        pytest.skip(f"triton unavailable: {exc}")
    ptxas = os.environ.get("TRITON_PTXAS_PATH")
    if not ptxas or not os.path.exists(ptxas):
        pytest.skip("ptxas not found (pip install nvidia-cuda-nvcc-cu12)")
    return mod


@pytest.fixture(scope="module")
def compiled(sa, tmp_path_factory):
    build = str(tmp_path_factory.mktemp("ptx"))
    specs = sa._mixtral_specs()
    for s in specs:
        sa.compile_and_measure(s, build)   # raises on compile failure
    return {s.name: (s, build) for s in specs}


def _ptx(build: str, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    with open(os.path.join(build, safe + ".ptx")) as f:
        return f.read()


def test_all_kernels_compile_sm90a(compiled):
    assert len(compiled) == 10


def test_zero_spill_everywhere(compiled):
    spills = {n: (s.spill_st, s.spill_ld) for n, (s, _) in compiled.items()
              if s.spill_st or s.spill_ld}
    assert not spills, f"register spills reappeared: {spills}"


def test_memory_bound_kernels_meet_occupancy_floor(compiled):
    for name in ("op1 GEMM1 (w1/w3 + SiLU)",
                 "op1 GEMM2 det (split-K partials)",
                 "op1 GEMM2 atomic (fast path)"):
        s, _ = compiled[name]
        assert s.occupancy >= 0.25, f"{name}: occupancy {s.occupancy:.0%} < 25%"


def test_eviction_hints_survive_to_ptx(compiled):
    # Note: the u32-packed weight loads (SASS vectorization round) drop
    # evict_first — an accepted trade (8x fewer load instructions beats an L2
    # hint on a 23.8GB/step stream that could never fit 50MB L2 anyway). The
    # evict_last hints on reused x/h tiles are the ones that matter; guard them.
    for name in ("op1 GEMM1 (w1/w3 + SiLU)",
                 "op1 GEMM2 det (split-K partials)"):
        s, build = compiled[name]
        ptx = _ptx(build, name)
        assert "evict_last" in ptx, f"{name}: reuse hint dropped"


def test_kernel_a_does_not_duplicate_router_gemm(compiled):
    for name in ("op1 Kernel A (fused budget+bucket)",
                 "op1 Kernel A (critical-path)"):
        s, build = compiled[name]
        ptx = _ptx(build, s.name)
        assert "wgmma" not in ptx and "mma.sync" not in ptx


def _nvdisasm():
    import glob
    hits = glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/bin/nvdisasm") \
        + glob.glob(os.path.expanduser("~/.local/lib/python3*/site-packages/nvidia/*/bin/nvdisasm"))
    return hits[0] if hits else None


def test_weight_streams_vectorized_in_sass(compiled):
    """SASS audit found Triton 3.7 lowers mma B-operand global loads in the
    dot layout -> 16-bit scalar LDG.E.U16 (8x instruction bloat on the
    dominant 23.8GB/step weight stream). The u32-packed load pattern forces
    >=32-bit lanes; lock that in at the SASS level."""
    import subprocess
    nvd = _nvdisasm()
    if nvd is None:
        pytest.skip("nvdisasm not found (pip install nvidia-cuda-nvdisasm)")
    ptxas = os.environ["TRITON_PTXAS_PATH"]
    for name in ("op1 GEMM1 (w1/w3 + SiLU)",
                 "op1 GEMM2 det (split-K partials)",
                 "op1 GEMM2 atomic (fast path)"):
        s, build = compiled[name]
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        ptx_path = os.path.join(build, safe + ".ptx")
        cubin = os.path.join(build, safe + ".cubin")
        subprocess.run([ptxas, "-arch=sm_90a", "-O3", "-o", cubin, ptx_path],
                       check=True, capture_output=True)
        sass = subprocess.run([nvd, "-c", cubin], check=True,
                              capture_output=True, text=True).stdout
        loads = re.findall(r"LDG\.E[A-Z0-9.]*", sass)
        wide = [l for l in loads if ".64" in l or ".128" in l]
        # the u16 stragglers are the handful of x/h edge loads, not the
        # weight stream; the weight stream must appear as wide loads
        assert wide, f"{name}: no vectorized global loads in SASS ({loads})"
        narrow_w = [l for l in loads if "U16" in l and ".EF" in l]
        assert not narrow_w, (
            f"{name}: weight stream regressed to scalar 16-bit loads: {narrow_w}")

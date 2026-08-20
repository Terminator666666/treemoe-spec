"""PTX-level regression guards (no GPU needed).

Locks in facts established by the static-analysis rounds so future edits
can't silently regress them:
  * all kernels AOT-compile for sm_90a (caught a real int32/i64 bug once)
  * zero register spill at production launch configs
  * L2 eviction hints survive into PTX (Triton may drop them silently)
  * Kernel A's router GEMM uses Hopper wgmma (N=64 unlocks m64 tiles)

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
    assert len(compiled) == 9


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
    for name in ("op1 GEMM1 (w1/w3 + SiLU)",
                 "op1 GEMM2 det (split-K partials)"):
        s, build = compiled[name]
        ptx = _ptx(build, name)
        assert "evict_first" in ptx, f"{name}: weight-stream hint dropped"
        assert "evict_last" in ptx, f"{name}: reuse hint dropped"


def test_kernel_a_uses_wgmma(compiled):
    s, build = compiled["op1 Kernel A (fused route+bucket)"]
    ptx = _ptx(build, s.name)
    assert "wgmma" in ptx, "router GEMM lost Hopper wgmma (N=64 m-tile)"

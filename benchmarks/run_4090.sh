#!/usr/bin/env bash
# TreeMoE-Spec test plan for a single RTX 4090 24GB (sm_89, 1008 GB/s).
# Tiers 1-3; tier 4 (paper speedup numbers) still needs an H200 rental.
#
# Environment (once):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
#   pip install transformers safetensors pytest numpy
#   pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-nvdisasm   # ptxas + SASS guards
#   # tier 3 additionally: host RAM >= ~110GB and weights under
#   # checkpoints/mixtral-8x7b-instruct (BF16 safetensors, ~93GB download)
set -euo pipefail
cd "$(dirname "$0")/.."

# AutoDL images ship OMP_NUM_THREADS=0 (invalid -> libgomp warning + default)
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS="$(nproc)"
fi

echo "== tier 1a: CPU logic suite (fast sanity, no GPU used) =="
python -m pytest -q -m "not gpu and not model and not interpret"
TRITON_INTERPRET=1 python -m pytest -q -m interpret

echo "== tier 1b: GPU kernel correctness (first Triton runs on real silicon) =="
# includes test_spec_lossless_vs_ar_gpu_kernel_commit: the [root]+accepted
# _kv_commit_kernel path (grid max_depth+1) only reachable on CUDA -- this is
# the top-priority check after the 2026-08-31 commit-semantics fix
python -m pytest -q -m "gpu and not model" -x

echo "== tier 2a: op1 microbenchmark, real Mixtral shapes =="
python benchmarks/bench_op1.py --tree-sizes 32 64 128
# vLLM comparison column appears automatically if vllm is installed
# expect: N=32/64 ~3.13ms = 89% peak; N=128 should now hit fused Kernel A

echo "== tier 2b: verify static-analysis predictions with ncu (optional) =="
echo "   ncu --set full --kernel-name-base demangled \\"
echo "       -k 'regex:_moe_gemm' python benchmarks/bench_op1.py --tree-sizes 64"
echo "   check: occupancy ~38%/56%, LDG.E.64 weight loads, zero local-memory traffic"

echo "== tier 2c: offload prefetch hit-rate + host-DRAM break-even probes =="
python benchmarks/bench_cpu_expert.py
# AutoDL host DRAM BW vs PCIe 23.8GB/s decides the Fiddler-style CPU-expert path
if [ -d checkpoints/mixtral-8x7b-instruct ] || [ -d /root/autodl-tmp/Mixtral-8x7B-Instruct-v0.1 ]; then
  E2E_W=""
else
  # random weights: streaming/hit-rate numbers valid, accept_len is not.
  # Small workload: offload passes are PCIe-bound (~1s each) and a random
  # draft accepts ~nothing, so the full sweep would take hours.
  # --no-router-hint: measured best arm on random weights (degenerate
  # routing maximally favours the temporal predictor; the hint's verdict
  # belongs to tier 3 real weights).
  E2E_W="--random-weights --num-prompts 3 --max-new-tokens 32 --budgets 4 8 --no-router-hint"
fi
python benchmarks/bench_e2e.py --layout offload --prefetch-depth 2 $E2E_W
python benchmarks/bench_e2e.py --layout offload --prefetch-depth 2 --no-auto-bitmap $E2E_W
# hit_rate column: auto_bitmap temporal predictor vs full-copy baseline

if [ -d checkpoints/mixtral-8x7b-instruct ] || [ -d /root/autodl-tmp/Mixtral-8x7B-Instruct-v0.1 ]; then
  echo "== tier 3: red-line tests (offloaded, slow: expect ~30-60 min) =="
  # NOTE: rerun test_ar_logits_match_hf even if it passed before -- the
  # attention path moved to SDPA enable_gqa + gather_with_tail (2026-08-31),
  # backend kernel selection may differ from the repeat_interleave era
  python -m pytest -q -m "model and gpu" -x
else
  echo "== tier 3 skipped: checkpoints/mixtral-8x7b-instruct not found =="
fi

# tier 3+ (needs the EAGLE draft checkpoint, yuhuili/EAGLE-mixtral-instruct-8x7B):
#   First run the lossless red line on official MT-bench prompts. This saves
#   both token sequences and the first mismatch (if any) before any B<8 sweep:
#   python benchmarks/bench_e2e.py --layout offload --check-lossless \
#       --tree-sizes 64 --num-prompts 2 --max-new-tokens 32 \
#       --output-json artifacts/e2e_lossless.json
#   Then measure MAT -- the draft-side fixes (tree topology mask, committed-KV
#   pruning, prompt conditioning) should raise mean accept length; compare
#   mean_accept_len against the pre-fix baseline.

echo "all tiers done"

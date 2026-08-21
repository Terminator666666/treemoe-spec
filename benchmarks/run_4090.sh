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

echo "== tier 1a: CPU logic suite (fast sanity, no GPU used) =="
python -m pytest -q -m "not gpu and not model and not interpret"
TRITON_INTERPRET=1 python -m pytest -q -m interpret

echo "== tier 1b: GPU kernel correctness (first Triton runs on real silicon) =="
python -m pytest -q -m "gpu and not model" -x

echo "== tier 2a: op1 microbenchmark, real Mixtral shapes =="
python benchmarks/bench_op1.py --tree-sizes 32 64 128
# vLLM comparison column appears automatically if vllm is installed

echo "== tier 2b: verify static-analysis predictions with ncu (optional) =="
echo "   ncu --set full --kernel-name-base demangled \\"
echo "       -k 'regex:_moe_gemm' python benchmarks/bench_op1.py --tree-sizes 64"
echo "   check: occupancy ~38%/56%, LDG.E.64 weight loads, zero local-memory traffic"

if [ -d checkpoints/mixtral-8x7b-instruct ]; then
  echo "== tier 3: red-line tests (offloaded, slow: expect ~30-60 min) =="
  python -m pytest -q -m "model and gpu" -x
else
  echo "== tier 3 skipped: checkpoints/mixtral-8x7b-instruct not found =="
fi

echo "all tiers done"

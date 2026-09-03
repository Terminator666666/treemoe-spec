#!/usr/bin/env bash
# Reproduce the implemented RTX 4090 paper experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
	export OMP_NUM_THREADS="$(nproc)"
fi

echo "== Phase 0: observations (figs 1-3) =="
"$PYTHON" measurements/collect_routing.py --prompts benchmarks/data/mt_bench.jsonl \
	--out measurements/data/routing_traces.pt
"$PYTHON" measurements/analyze.py

echo "== correctness gates =="
"$PYTHON" -m pytest -m "not model" -x
"$PYTHON" -m pytest -m "model and gpu" -x   # requires Mixtral + EAGLE weights

echo "== op1 kernel benchmark (Task 2.4 gate) =="
"$PYTHON" benchmarks/bench_op1.py --tree-sizes 32 64 128

echo "== op2 transfer/overlap benchmark =="
"$PYTHON" benchmarks/bench_op2.py

echo "== AR baseline =="
"$PYTHON" benchmarks/bench_e2e.py --layout offload --ar-baseline \
	--num-prompts 20 --max-new-tokens 128

echo "== budget main experiment (N=64) =="
"$PYTHON" benchmarks/bench_e2e.py --layout offload \
	--budgets 2 3 4 5 6 8 --tree-sizes 64 \
	--num-prompts 20 --max-new-tokens 128 --top1-threshold 0

echo "== tree-size ablation (N=64 reuses the main experiment) =="
"$PYTHON" benchmarks/bench_e2e.py --layout offload \
	--budgets 2 4 8 --tree-sizes 16 32 \
	--num-prompts 20 --max-new-tokens 128 --top1-threshold 0

echo "== staging ablation: predictive temporal only =="
"$PYTHON" benchmarks/bench_e2e.py --layout offload --predictive-prefetch \
	--no-router-hint \
	--budgets 4 --tree-sizes 64 --num-prompts 20 --max-new-tokens 128 \
	--top1-threshold 0

echo "== staging ablation: predictive full copy =="
"$PYTHON" benchmarks/bench_e2e.py --layout offload --predictive-prefetch \
	--no-auto-bitmap --no-router-hint \
	--budgets 4 --tree-sizes 64 --num-prompts 20 --max-new-tokens 128 \
	--top1-threshold 0

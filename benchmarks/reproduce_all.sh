#!/usr/bin/env bash
# Reproduce all paper experiments (GPU machine, weights under checkpoints/).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Phase 0: observations (figs 1-3) =="
python measurements/collect_routing.py --out measurements/data/routing_traces.pt
python measurements/analyze.py

echo "== correctness gates =="
pytest -m "not model" -x
pytest -m "model and gpu" -x   # requires Mixtral + EAGLE weights

echo "== op1 kernel benchmark (Task 2.4 gate) =="
python benchmarks/bench_op1.py --tree-sizes 32 64 128

echo "== predictor training (Task 4.1 gate) =="
python measurements/train_predictor.py

echo "== end-to-end TPOT / accept-length / B-sweep =="
python benchmarks/bench_e2e.py --budgets 3 4 5 6 8 --tree-sizes 32 64 128

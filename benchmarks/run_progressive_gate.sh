#!/usr/bin/env bash
# Collect a lossless B=8 exact-JIT trace and evaluate progressive verification.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PCIE_GBPS="${PCIE_GBPS:-25.03}"
EXTRA_STAGE_OVERHEAD_MS="${EXTRA_STAGE_OVERHEAD_MS:-5.0}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/progressive_gate}"
TRACE_JSON="$ARTIFACT_DIR/execution_trace_b8_n64.json"
REPORT_JSON="$ARTIFACT_DIR/progressive_gate.json"

mkdir -p "$ARTIFACT_DIR"

echo "== collect B=8 N=64 exact-JIT execution trace =="
"$PYTHON" -u benchmarks/bench_e2e.py \
	--layout offload --budgets 8 --tree-sizes 64 \
	--num-prompts "$NUM_PROMPTS" --max-new-tokens "$MAX_NEW_TOKENS" \
	--warmup-new-tokens 8 \
	--top1-threshold 0 --execution-trace-baseline-first \
	--execution-trace-detail progressive \
	--execution-trace-json "$TRACE_JSON"

reference_args=()
if [[ -n "${REFERENCE_TPOT_MS:-}" ]]; then
	reference_args=(--reference-tpot-ms "$REFERENCE_TPOT_MS")
fi

echo "== evaluate progressive exact-verification admission gates =="
set +e
PYTHONPATH=. "$PYTHON" benchmarks/simulate_progressive_verification.py \
	"$TRACE_JSON" --pcie-gbps "$PCIE_GBPS" \
	--extra-stage-overhead-ms "$EXTRA_STAGE_OVERHEAD_MS" \
	--stage-grid 4,8,16,32 \
	--stage-grid 8,16,32 \
	--stage-grid 8,24,40 \
	--stage-grid 16,32,48 \
	--min-byte-reduction 0.20 --max-tpot-regression 0.10 \
	--output-json "$REPORT_JSON" --require-pass \
	"${reference_args[@]}"
status=$?
set -e

echo "trace: $TRACE_JSON"
echo "gate report: $REPORT_JSON"
if [[ $status -eq 2 ]]; then
	echo "progressive runtime gate: FAIL"
	exit 2
fi
if [[ $status -ne 0 ]]; then
	exit "$status"
fi
echo "progressive runtime gate: PASS"
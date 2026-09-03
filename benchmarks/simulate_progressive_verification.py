"""Replay progressive exact target verification from full execution traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from treemoe.engine.progressive_verify import (
    build_depth_stages,
    build_probability_stages,
    extract_layer_node_experts,
    replay_progressive_verification,
)


@dataclass(frozen=True)
class TraceSimulation:
    steps: int
    generated_tokens: int
    trace_emitted_tokens: int
    one_shot_rows: int
    progressive_rows: int
    persistent_lower_bound_rows: int
    one_shot_bytes: int
    progressive_bytes: int
    persistent_lower_bound_bytes: int
    executed_stages: int
    extra_stages: int
    final_stage_steps: int

    @property
    def row_reduction(self) -> float:
        return _reduction(self.progressive_rows, self.one_shot_rows)

    @property
    def byte_reduction(self) -> float:
        return _reduction(self.progressive_bytes, self.one_shot_bytes)


@dataclass(frozen=True)
class GateEvaluation:
    baseline_tpot_ms: float
    reference_tpot_ms: float
    estimated_tpot_ms: float
    tpot_regression: float
    transfer_saved_ms: float
    extra_stage_overhead_ms: float
    break_even_overhead_ms: float
    ten_percent_overhead_ms: float
    byte_gate_passed: bool
    tpot_gate_passed: bool

    @property
    def passed(self) -> bool:
        return self.byte_gate_passed and self.tpot_gate_passed


def simulate_config(
    config: dict,
    stage_budgets: tuple[int, ...],
    require_lossless: bool = True,
    strategy: str = "probability",
) -> TraceSimulation:
    steps = config["trace"]["steps"]
    if not steps:
        raise ValueError("trace contains no verification steps")
    if require_lossless:
        _validate_lossless_trace(config, steps)

    replays = []
    for step in steps:
        tree = step["tree"]
        if strategy == "probability":
            stages = build_probability_stages(
                tree["parent"], tree["accept_probability"], stage_budgets,
            )
        elif strategy == "depth":
            stages = build_depth_stages(tree["parent"], stage_budgets)
        else:
            raise ValueError(f"unknown stage strategy: {strategy}")
        layer_experts = extract_layer_node_experts(step)
        row_bytes = tuple(
            int(layer.get("expert_row_bytes", 0)) for layer in step["layers"]
        )
        replay = replay_progressive_verification(
            tree["parent"], step["acceptance"]["accepted_slots"], stages,
            layer_experts, layer_row_bytes=row_bytes,
        )
        replays.append(replay)

    return TraceSimulation(
        steps=len(replays),
        generated_tokens=int(config.get("generated_tokens") or sum(
            int(step["acceptance"]["emitted_tokens"]) for step in steps
        )),
        trace_emitted_tokens=sum(
            int(step["acceptance"]["emitted_tokens"]) for step in steps
        ),
        one_shot_rows=sum(row.one_shot_rows for row in replays),
        progressive_rows=sum(row.progressive_rows for row in replays),
        persistent_lower_bound_rows=sum(
            row.persistent_lower_bound_rows for row in replays
        ),
        one_shot_bytes=sum(row.one_shot_bytes for row in replays),
        progressive_bytes=sum(row.progressive_bytes for row in replays),
        persistent_lower_bound_bytes=sum(
            row.persistent_lower_bound_bytes for row in replays
        ),
        executed_stages=sum(row.executed_stages for row in replays),
        extra_stages=sum(row.executed_stages - 1 for row in replays),
        final_stage_steps=sum(
            row.stop_stage == len(
                build_probability_stages(
                    step["tree"]["parent"],
                    step["tree"]["accept_probability"], stage_budgets,
                ) if strategy == "probability" else build_depth_stages(
                    step["tree"]["parent"], stage_budgets,
                )
            ) - 1 for row, step in zip(replays, steps, strict=True)
        ),
    )


def evaluate_gate(
    config: dict,
    result: TraceSimulation,
    pcie_gbps: float,
    extra_stage_overhead_ms: float,
    min_byte_reduction: float = 0.20,
    max_tpot_regression: float = 0.10,
    reference_tpot_ms: float | None = None,
) -> GateEvaluation:
    if pcie_gbps <= 0:
        raise ValueError("pcie_gbps must be positive")
    if extra_stage_overhead_ms < 0:
        raise ValueError("extra_stage_overhead_ms must be non-negative")
    if result.generated_tokens <= 0:
        raise ValueError("trace must contain generated tokens")
    baseline_tpot = float(
        config.get("baseline_tpot_ms") or config.get("tpot_ms", 0.0)
    )
    if baseline_tpot <= 0:
        total_host_ms = sum(
            float(step.get("timing_ms", {}).get("total", {}).get("host", 0.0))
            for step in config["trace"]["steps"]
        )
        baseline_tpot = total_host_ms / result.generated_tokens
    if baseline_tpot <= 0:
        raise ValueError("trace must provide tpot_ms or total host timings")
    reference_tpot = (
        baseline_tpot if reference_tpot_ms is None else float(reference_tpot_ms)
    )
    if reference_tpot <= 0:
        raise ValueError("reference_tpot_ms must be positive")

    saved_bytes = result.one_shot_bytes - result.progressive_bytes
    transfer_saved_ms = saved_bytes / (pcie_gbps * 1e9) * 1e3
    overhead_total_ms = result.extra_stages * extra_stage_overhead_ms
    estimated_tpot = baseline_tpot + (
        overhead_total_ms - transfer_saved_ms
    ) / result.generated_tokens
    regression = estimated_tpot / reference_tpot - 1.0
    break_even = (
        transfer_saved_ms / result.extra_stages if result.extra_stages else 0.0
    )
    ten_percent_budget_ms = (
        (1.0 + max_tpot_regression) * reference_tpot - baseline_tpot
    ) * result.generated_tokens + transfer_saved_ms
    ten_percent_overhead = (
        ten_percent_budget_ms / result.extra_stages
        if result.extra_stages else float("inf")
    )
    return GateEvaluation(
        baseline_tpot_ms=baseline_tpot,
        reference_tpot_ms=reference_tpot,
        estimated_tpot_ms=estimated_tpot,
        tpot_regression=regression,
        transfer_saved_ms=transfer_saved_ms,
        extra_stage_overhead_ms=extra_stage_overhead_ms,
        break_even_overhead_ms=break_even,
        ten_percent_overhead_ms=ten_percent_overhead,
        byte_gate_passed=result.byte_reduction >= min_byte_reduction,
        tpot_gate_passed=regression <= max_tpot_regression,
    )


def print_report(
    config: dict,
    result: TraceSimulation,
    stage_budgets: tuple[int, ...],
    pcie_gbps: float,
    gate: GateEvaluation,
    strategy: str,
) -> None:
    repeated_rows = result.progressive_rows - result.persistent_lower_bound_rows
    label = (
        f"objective={config.get('routing_objective')} B={config.get('budget')} "
        f"N={config.get('tree_size')}"
    )
    print(f"\nCONFIG {label}")
    print(f"strategy={strategy} stages={stage_budgets}+full steps={result.steps}")
    print(
        f"generated_tokens={result.generated_tokens} "
        f"trace_emitted_tokens={result.trace_emitted_tokens}"
    )
    print(
        f"one_shot rows/step={result.one_shot_rows / result.steps:.2f} "
        f"GiB/step={_gib(result.one_shot_bytes / result.steps):.3f}"
    )
    print(
        f"progressive rows/step={result.progressive_rows / result.steps:.2f} "
        f"GiB/step={_gib(result.progressive_bytes / result.steps):.3f} "
        f"byte_reduction={result.byte_reduction:.2%}"
    )
    print(
        f"persistent_lower_bound rows/step="
        f"{result.persistent_lower_bound_rows / result.steps:.2f} "
        f"GiB/step={_gib(result.persistent_lower_bound_bytes / result.steps):.3f}"
    )
    print(
        f"executed_stages/step={result.executed_stages / result.steps:.3f} "
        f"full_tree_rate={result.final_stage_steps / result.steps:.2%} "
        f"repeated_rows/step={repeated_rows / result.steps:.2f}"
    )
    print(
        f"pcie_saved_ms/step@{pcie_gbps:g}GB/s="
        f"{gate.transfer_saved_ms / result.steps:.3f} "
        f"assumed_extra_stage_ms={gate.extra_stage_overhead_ms:.3f}"
    )
    print(
        f"baseline_tpot_ms={gate.baseline_tpot_ms:.3f} "
        f"reference_tpot_ms={gate.reference_tpot_ms:.3f} "
        f"estimated_tpot_ms={gate.estimated_tpot_ms:.3f} "
        f"regression={gate.tpot_regression:.2%}"
    )
    print(
        f"max_extra_stage_ms@0%={gate.break_even_overhead_ms:.3f} "
        f"max_extra_stage_ms@gate={gate.ten_percent_overhead_ms:.3f} "
        f"GATE={'PASS' if gate.passed else 'FAIL'} "
        f"bytes={'PASS' if gate.byte_gate_passed else 'FAIL'} "
        f"tpot={'PASS' if gate.tpot_gate_passed else 'FAIL'}"
    )


def _validate_lossless_trace(config: dict, steps: list[dict]) -> None:
    first_layer = next(
        (layer for step in steps for layer in step.get("layers", [])), None,
    )
    if first_layer is None or not first_layer.get("nodes"):
        raise ValueError("trace has no per-node router assignments")
    num_experts = config.get("num_experts")
    if num_experts is None:
        probabilities = first_layer["nodes"][0].get("router_probability")
        if probabilities is None:
            raise ValueError("trace must provide num_experts")
        num_experts = len(probabilities)
    num_experts = int(num_experts)
    budget = int(config.get("budget", first_layer.get("budget", 0)))
    if budget != num_experts:
        raise ValueError(
            f"B={budget} trace is approximate; progressive exact verification "
            f"requires a lossless B={num_experts} trace"
        )
    if config.get("staging_mode") != "jit_exact":
        raise ValueError(
            "H2D simulation requires an offload trace with staging_mode=jit_exact"
        )
    if any(
        int(layer.get("expert_row_bytes", 0)) <= 0
        for step in steps for layer in step.get("layers", [])
    ):
        raise ValueError("every traced layer must provide positive expert_row_bytes")
    expected_layers = config.get("num_layers")
    if expected_layers is not None and any(
        len(step.get("layers", [])) != int(expected_layers) for step in steps
    ):
        raise ValueError("one or more trace steps have incomplete layer records")


def _reduction(value: int, baseline: int) -> float:
    return 0.0 if baseline == 0 else 1.0 - value / baseline


def _gib(value: float) -> float:
    return value / 1024**3


def _stage_grid(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "stage grid must be comma-separated integers"
        ) from error
    if not budgets or any(item <= 0 for item in budgets):
        raise argparse.ArgumentTypeError("stage budgets must be positive")
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--stage-budgets", type=int, nargs="+", default=(8, 16, 32),
        help="prefix-closed stage node budgets; the full tree is appended",
    )
    parser.add_argument(
        "--stage-grid", type=_stage_grid, action="append", default=[],
        help="repeat for multiple candidates, e.g. 8,16,32 and 8,24,40",
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=("probability", "depth"),
        default=("probability", "depth"),
    )
    parser.add_argument("--pcie-gbps", type=float, default=25.03)
    parser.add_argument(
        "--extra-stage-overhead-ms", type=float, default=5.0,
        help="assumed fixed latency for each stage after the first",
    )
    parser.add_argument("--min-byte-reduction", type=float, default=0.20)
    parser.add_argument("--max-tpot-regression", type=float, default=0.10)
    parser.add_argument(
        "--reference-tpot-ms", type=float, default=None,
        help="optional external baseline such as the current B4 TPOT",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--require-pass", action="store_true",
        help="exit nonzero unless every trace configuration passes both gates",
    )
    parser.add_argument(
        "--allow-approximate-routing-trace", action="store_true",
        help="allow B<num_experts traces for debugging only",
    )
    args = parser.parse_args()
    artifact = json.loads(args.trace.read_text())
    summaries = []
    config_passed = []
    stage_candidates = args.stage_grid or [tuple(args.stage_budgets)]
    for config in artifact:
        candidate_passed = False
        for strategy in args.strategies:
            for stage_budgets in stage_candidates:
                result = simulate_config(
                    config, stage_budgets,
                    require_lossless=not args.allow_approximate_routing_trace,
                    strategy=strategy,
                )
                gate = evaluate_gate(
                    config, result, args.pcie_gbps, args.extra_stage_overhead_ms,
                    min_byte_reduction=args.min_byte_reduction,
                    max_tpot_regression=args.max_tpot_regression,
                    reference_tpot_ms=args.reference_tpot_ms,
                )
                print_report(
                    config, result, stage_budgets, args.pcie_gbps, gate, strategy,
                )
                summaries.append({
                    "routing_objective": config.get("routing_objective"),
                    "budget": config.get("budget"),
                    "tree_size": config.get("tree_size"),
                    "strategy": strategy,
                    "stage_budgets": list(stage_budgets),
                    "simulation": result.__dict__,
                    "gate": {**gate.__dict__, "passed": gate.passed},
                })
                candidate_passed = candidate_passed or gate.passed
        config_passed.append(candidate_passed)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summaries, indent=2) + "\n")
    if args.require_pass and not all(config_passed):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
import json
import sys

import pytest

from treemoe.engine.progressive_verify import (
    build_depth_stages,
    build_probability_stages,
    extract_layer_node_experts,
    replay_progressive_verification,
)


def test_probability_stages_are_nested_prefix_closed_and_complete():
    parent = [-1, 0, 0, 1, 1, 2, 3]
    probability = [1.0, 0.8, 0.7, 0.6, 0.2, 0.5, 0.4]

    stages = build_probability_stages(parent, probability, [3, 5])

    assert stages[-1] == tuple(range(len(parent)))
    assert all(set(left).issubset(right)
               for left, right in zip(stages, stages[1:]))
    for stage in stages:
        assert all(node == 0 or parent[node] in stage for node in stage)
    assert 6 not in stages[0]


def test_depth_baseline_prefers_shallow_nodes_over_high_probability_deep_path():
    parent = [-1, 0, 0, 1]
    probability = [1.0, 0.9, 0.1, 0.8]

    probability_stages = build_probability_stages(parent, probability, [3])
    depth_stages = build_depth_stages(parent, [3])

    assert probability_stages[0] == (0, 1, 3)
    assert depth_stages[0] == (0, 1, 2)


def test_replay_stops_when_exact_accepted_path_is_covered():
    parent = [-1, 0, 0, 1, 1, 2, 3]
    stages = ((0, 1, 2), (0, 1, 2, 3, 5), tuple(range(7)))
    layer_node_experts = (
        ((0, 1), (0, 1), (2, 3), (1, 2), (4, 5), (2, 3), (6, 7)),
        ((0, 2), (0, 2), (1, 3), (2, 4), (5, 6), (1, 3), (4, 7)),
    )

    replay = replay_progressive_verification(
        parent, accepted_slots=[1, 3], stages=stages,
        layer_node_experts=layer_node_experts,
    )

    assert replay.executed_stages == 2
    assert replay.executed_nodes == ((0, 1, 2), (3, 5))
    assert replay.one_shot_rows == 16
    assert replay.progressive_rows == 15
    assert replay.persistent_lower_bound_rows == 9
    assert replay.row_reduction == pytest.approx(0.0625)


def test_replay_rejects_non_prefix_closed_stage():
    with pytest.raises(ValueError, match="prefix-closed"):
        replay_progressive_verification(
            [-1, 0, 1], [1], ((0, 2), (0, 1, 2)),
            (((0,), (1,), (2,)),),
        )


def test_replay_rejects_slots_that_are_not_one_accepted_path():
    with pytest.raises(ValueError, match="root-descending path"):
        replay_progressive_verification(
            [-1, 0, 0], [1, 2], ((0, 1), (0, 1, 2)),
            (((0,), (1,), (2,)),),
        )


def test_extracts_natural_routing_instead_of_budgeted_selection():
    step = {
        "tree": {"num_valid": 2},
        "layers": [{
            "layer": 0,
            "nodes": [
                {"original_top2_experts": [1, 3], "selected_experts": [0, 1]},
                {"original_top2_experts": [2, 4], "selected_experts": [0, 2]},
            ],
        }],
    }

    assert extract_layer_node_experts(step) == (((1, 3), (2, 4)),)


def test_trace_simulator_rejects_budgeted_target_trace():
    from benchmarks.simulate_progressive_verification import (
        evaluate_gate,
        simulate_config,
    )

    config = {
        "budget": 1,
        "staging_mode": "jit_exact",
        "tpot_ms": 100.0,
        "trace": {"steps": [{
            "tree": {
                "num_valid": 2, "parent": [-1, 0],
                "accept_probability": [1.0, 0.5],
            },
            "acceptance": {"accepted_slots": [], "emitted_tokens": 1},
            "layers": [{
                "layer": 0, "expert_row_bytes": 10,
                "nodes": [
                    {"router_probability": [0.8, 0.2],
                     "original_top2_experts": [0, 1]},
                    {"router_probability": [0.6, 0.4],
                     "original_top2_experts": [0, 1]},
                ],
            }],
        }]},
    }

    with pytest.raises(ValueError, match="requires a lossless B=2 trace"):
        simulate_config(config, (1,))

    result = simulate_config(config, (1,), require_lossless=False)
    assert result.one_shot_bytes == 20
    assert result.progressive_bytes == 20
    assert result.executed_stages == 1
    gate = evaluate_gate(
        config, result, pcie_gbps=1.0, extra_stage_overhead_ms=5.0,
    )
    assert gate.estimated_tpot_ms == pytest.approx(100.0)
    assert gate.byte_gate_passed is False
    assert gate.tpot_gate_passed is True
    assert gate.passed is False


def test_gate_converts_aggregate_transfer_and_stage_cost_to_tpot():
    from benchmarks.simulate_progressive_verification import (
        TraceSimulation,
        evaluate_gate,
    )

    result = TraceSimulation(
        steps=1, generated_tokens=4, trace_emitted_tokens=5,
        one_shot_rows=10, progressive_rows=5,
        persistent_lower_bound_rows=5,
        one_shot_bytes=1_000_000, progressive_bytes=500_000,
        persistent_lower_bound_bytes=500_000,
        executed_stages=3, extra_stages=2, final_stage_steps=0,
    )
    config = {"tpot_ms": 100.0, "trace": {"steps": []}}

    gate = evaluate_gate(
        config, result, pcie_gbps=1.0, extra_stage_overhead_ms=0.1,
    )

    assert gate.transfer_saved_ms == pytest.approx(0.5)
    assert gate.estimated_tpot_ms == pytest.approx(99.925)
    assert gate.break_even_overhead_ms == pytest.approx(0.25)
    assert gate.ten_percent_overhead_ms == pytest.approx(20.25)
    assert gate.passed is True

    external_gate = evaluate_gate(
        config, result, pcie_gbps=1.0, extra_stage_overhead_ms=0.1,
        reference_tpot_ms=90.0,
    )
    assert external_gate.tpot_regression > 0.10
    assert external_gate.passed is False


def test_simulator_cli_writes_grid_report_and_enforces_gate(
    tmp_path, monkeypatch,
):
    from benchmarks.simulate_progressive_verification import main

    nodes = [
        {"original_top2_experts": [0]},
        {"original_top2_experts": [1]},
        {"original_top2_experts": [1]},
        {"original_top2_experts": [1]},
    ]
    artifact = [{
        "routing_objective": "mass", "budget": 2, "tree_size": 4,
        "staging_mode": "jit_exact", "tpot_ms": 140.0,
        "baseline_tpot_ms": 100.0, "generated_tokens": 1,
        "num_layers": 1, "num_experts": 2,
        "trace": {"steps": [{
            "tree": {
                "num_valid": 4, "parent": [-1, 0, 0, 1],
                "accept_probability": [1.0, 0.7, 0.4, 0.2],
            },
            "acceptance": {"accepted_slots": [], "emitted_tokens": 1},
            "layers": [{
                "layer": 0, "expert_row_bytes": 1_000_000,
                "nodes": nodes,
            }],
        }]},
    }]
    trace_path = tmp_path / "trace.json"
    report_path = tmp_path / "report.json"
    trace_path.write_text(json.dumps(artifact))
    monkeypatch.setattr(sys, "argv", [
        "simulate_progressive_verification.py", str(trace_path),
        "--stage-grid", "1", "--stage-grid", "2",
        "--strategies", "probability",
        "--pcie-gbps", "1", "--extra-stage-overhead-ms", "0",
        "--output-json", str(report_path), "--require-pass",
    ])

    main()

    report = json.loads(report_path.read_text())
    assert [row["stage_budgets"] for row in report] == [[1], [2]]
    assert [row["gate"]["passed"] for row in report] == [True, False]

    monkeypatch.setattr(sys, "argv", [
        "simulate_progressive_verification.py", str(trace_path),
        "--stage-grid", "1", "--min-byte-reduction", "0.75",
        "--strategies", "probability",
        "--require-pass",
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
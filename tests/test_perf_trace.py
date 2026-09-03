import time

import pytest
import torch

from treemoe.engine.perf_trace import ExecutionTracer


def test_execution_tracer_records_host_phases_and_tree_paths():
    tracer = ExecutionTracer()
    record = tracer.begin_step(0, 7, torch.device("cpu"))
    with tracer.phase(record, "draft_tree"):
        time.sleep(0.001)

    class Tree:
        num_valid = 4
        parent = torch.tensor([-1, 0, 1, 0])
        tokens = torch.tensor([10, 11, 12, 13])
        accept_prob = torch.tensor([1.0, 0.8, 0.5, 0.2])
        children = [[1, 3], [2], [], []]

    tracer.record_tree(record, Tree())
    layer = tracer.begin_layer(0)
    with tracer.phase(layer, "attention"):
        pass
    tracer.record_acceptance(record, [1, 2], [11, 12], 14)
    logits = torch.full((4, 20), -4.0)
    logits[0, 11] = 5.0
    logits[1, 12] = 4.0
    tracer.record_target_decisions(record, Tree(), logits, [1, 2])
    tracer.end_record()
    result = tracer.to_dict()["steps"][0]

    assert result["timing_ms"]["draft_tree"]["host"] >= 1.0
    assert result["timing_ms"]["total"]["host"] >= 1.0
    assert result["tree"]["paths"] == [[0], [0, 1], [0, 1, 2], [0, 3]]
    assert result["tree"]["depth"] == [0, 1, 2, 1]
    assert result["acceptance"]["accepted_path_probability"] == pytest.approx(
        [0.8, 0.5]
    )
    assert result["acceptance"]["emitted_tokens"] == 3
    assert result["target_nodes"][1]["proposed_target_rank"] == 1
    assert result["target_nodes"][1]["accepted"] is True
    assert result["target_nodes"][3]["accepted"] is False
    assert result["layers"][0]["layer"] == 0


def test_execution_trace_analyzer_prints_stage_and_path_report(capsys):
    from benchmarks.analyze_execution_trace import summarize

    tracer = ExecutionTracer()
    record = tracer.begin_step(0, 0, torch.device("cpu"))

    class Tree:
        num_valid = 2
        parent = torch.tensor([-1, 0])
        tokens = torch.tensor([10, 11])
        accept_prob = torch.tensor([1.0, 0.5])
        children = [[1], []]

    tracer.record_tree(record, Tree())
    level = tracer.begin_draft_level(1, 1, 0)
    level.update({
        "generated_candidates": 2,
        "next_frontier_nodes": 1,
        "frontier_pool_nodes": [0],
        "selected_frontier_pool_nodes": [1],
        "input_tokens": [10],
        "positions": [0],
        "candidate_token_ids": [[11, 12]],
        "candidate_logprob": [[-0.1, -1.0]],
    })
    layer = tracer.begin_layer(0)
    layer.update({
        "repair_rows": 1,
        "budget": 4,
        "routed_experts": [0, 1],
        "nodes": [{
            "node": 0,
            "accept_probability": 1.0,
            "router_probability": [0.7, 0.3],
            "original_top2_experts": [0, 1],
            "original_top2_probability": [0.7, 0.3],
            "selected_experts": [0, 1],
            "selected_gates": [0.7, 0.3],
        }],
    })
    tracer.record_acceptance(record, [1], [11], 12)
    logits = torch.full((2, 20), -4.0)
    logits[0, 11] = 5.0
    tracer.record_target_decisions(record, Tree(), logits, [1])
    tracer.end_record()
    summarize({
        "routing_objective": "mass",
        "budget": 4,
        "tree_size": 2,
        "num_prompts": 1,
        "trace": tracer.to_dict(),
    }, show_tree=True, show_draft=True, show_target=True, show_routing=True)
    output = capsys.readouterr().out

    assert "PER-STEP TREE / ACCEPTANCE / ENGINE" in output
    assert "LAYER HOTSPOTS" in output
    assert "TREE NODES" in output
    assert "DRAFT CANDIDATES" in output
    assert "TARGET DECISIONS" in output
    assert "ROUTING BY LAYER / NODE" in output
    assert "router=[e0=0.7000000" in output
    assert "0>1" in output


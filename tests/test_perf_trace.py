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
    tracer.begin_layer(0)["repair_rows"] = 1
    tracer.record_acceptance(record, [1], [11], 12)
    tracer.end_record()
    summarize({
        "routing_objective": "mass",
        "budget": 4,
        "tree_size": 2,
        "num_prompts": 1,
        "trace": tracer.to_dict(),
    }, show_tree=True)
    output = capsys.readouterr().out

    assert "PER-STEP TREE / ACCEPTANCE / ENGINE" in output
    assert "LAYER HOTSPOTS" in output
    assert "TREE NODES" in output
    assert "0>1" in output


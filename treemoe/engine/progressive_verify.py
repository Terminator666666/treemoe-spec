"""Offline planning and cost replay for progressive exact tree verification.

The planner builds nested, prefix-closed node sets.  Replay uses an observed
accepted path to stop at the first stage that contains every accepted node.
Because tree-attention nodes only attend to ancestors, incrementally executing
new nodes preserves the full-tree hidden states without approximating experts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProgressiveReplay:
    executed_stages: int
    stop_stage: int
    executed_nodes: tuple[tuple[int, ...], ...]
    one_shot_rows: int
    progressive_rows: int
    persistent_lower_bound_rows: int
    one_shot_bytes: int
    progressive_bytes: int
    persistent_lower_bound_bytes: int

    @property
    def row_reduction(self) -> float:
        if self.one_shot_rows == 0:
            return 0.0
        return 1.0 - self.progressive_rows / self.one_shot_rows


def build_probability_stages(
    parent: Sequence[int],
    accept_probability: Sequence[float],
    stage_budgets: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Build nested prefix-closed stages, ending with the complete tree."""
    if len(parent) != len(accept_probability):
        raise ValueError("parent and accept_probability must have equal length")
    if not parent:
        raise ValueError("tree must contain a root")
    if parent[0] != -1:
        raise ValueError("node 0 must be the root")
    if any(parent[node] < 0 or parent[node] >= node for node in range(1, len(parent))):
        raise ValueError("nodes must be in parent-before-child order")

    paths = [_path_to_root(node, parent) for node in range(len(parent))]
    order = sorted(
        range(1, len(parent)),
        key=lambda node: (-float(accept_probability[node]), len(paths[node]), node),
    )
    return _build_stages(parent, paths, order, stage_budgets)


def build_depth_stages(
    parent: Sequence[int],
    stage_budgets: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Build a breadth-first baseline with the same prefix-closure rule."""
    if not parent:
        raise ValueError("tree must contain a root")
    if parent[0] != -1:
        raise ValueError("node 0 must be the root")
    if any(parent[node] < 0 or parent[node] >= node for node in range(1, len(parent))):
        raise ValueError("nodes must be in parent-before-child order")
    paths = [_path_to_root(node, parent) for node in range(len(parent))]
    order = sorted(range(1, len(parent)), key=lambda node: (len(paths[node]), node))
    return _build_stages(parent, paths, order, stage_budgets)


def _build_stages(
    parent: Sequence[int],
    paths: Sequence[Sequence[int]],
    order: Sequence[int],
    stage_budgets: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    budgets = sorted({max(1, min(int(value), len(parent)))
                      for value in stage_budgets})
    if not budgets or budgets[-1] != len(parent):
        budgets.append(len(parent))
    selected = {0}
    stages: list[tuple[int, ...]] = []
    for budget in budgets:
        for node in order:
            missing = [value for value in paths[node] if value not in selected]
            if len(selected) + len(missing) <= budget:
                selected.update(missing)
            if len(selected) == budget:
                break
        if budget == len(parent):
            selected.update(range(len(parent)))
        stages.append(tuple(sorted(selected)))
    return tuple(stages)


def replay_progressive_verification(
    parent: Sequence[int],
    accepted_slots: Sequence[int],
    stages: Sequence[Sequence[int]],
    layer_node_experts: Sequence[Sequence[Sequence[int]]],
    layer_row_bytes: Sequence[int] | None = None,
) -> ProgressiveReplay:
    """Replay exact staged verification and count expert rows for new nodes."""
    if not stages:
        raise ValueError("at least one stage is required")
    _validate_stages(parent, stages)
    previous_accepted = 0
    for node in accepted_slots:
        node = int(node)
        if node <= 0 or node >= len(parent) or int(parent[node]) != previous_accepted:
            raise ValueError("accepted_slots must be a root-descending path")
        previous_accepted = node
    required = {0, *(int(node) for node in accepted_slots)}
    stop_stage = next(
        (index for index, stage in enumerate(stages)
         if required.issubset(set(stage))),
        None,
    )
    if stop_stage is None:
        raise ValueError("final stage does not cover the accepted path")

    executed_nodes: list[tuple[int, ...]] = []
    previous: set[int] = set()
    progressive_rows = 0
    progressive_bytes = 0
    persistent_rows = 0
    persistent_bytes = 0
    row_bytes = tuple(layer_row_bytes or [1] * len(layer_node_experts))
    if len(row_bytes) != len(layer_node_experts):
        raise ValueError("layer_row_bytes must match the number of layers")
    persistent_by_layer = [set() for _ in layer_node_experts]
    for stage in stages[:stop_stage + 1]:
        current = set(int(node) for node in stage)
        delta = tuple(sorted(current - previous))
        executed_nodes.append(delta)
        for layer_index, node_experts in enumerate(layer_node_experts):
            stage_experts = _expert_union(delta, node_experts)
            progressive_rows += len(stage_experts)
            progressive_bytes += len(stage_experts) * int(row_bytes[layer_index])
            new_persistent = stage_experts - persistent_by_layer[layer_index]
            persistent_rows += len(new_persistent)
            persistent_bytes += len(new_persistent) * int(row_bytes[layer_index])
            persistent_by_layer[layer_index].update(stage_experts)
        previous = current

    all_nodes = tuple(range(len(parent)))
    one_shot_layer_rows = [
        len(_expert_union(all_nodes, node_experts))
        for node_experts in layer_node_experts
    ]
    one_shot_rows = sum(one_shot_layer_rows)
    one_shot_bytes = sum(
        rows * int(row_bytes[layer_index])
        for layer_index, rows in enumerate(one_shot_layer_rows)
    )
    return ProgressiveReplay(
        executed_stages=stop_stage + 1,
        stop_stage=stop_stage,
        executed_nodes=tuple(executed_nodes),
        one_shot_rows=one_shot_rows,
        progressive_rows=progressive_rows,
        persistent_lower_bound_rows=persistent_rows,
        one_shot_bytes=one_shot_bytes,
        progressive_bytes=progressive_bytes,
        persistent_lower_bound_bytes=persistent_bytes,
    )


def extract_layer_node_experts(step: dict) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Read natural target-router top-k assignments from an execution trace."""
    num_nodes = int(step["tree"]["num_valid"])
    layers = []
    for layer in step.get("layers", []):
        nodes = layer.get("nodes", [])
        if len(nodes) < num_nodes:
            raise ValueError(
                f"layer {layer.get('layer')} has {len(nodes)} routing rows; "
                f"expected {num_nodes}"
            )
        layers.append(tuple(
            tuple(int(expert) for expert in nodes[node]["original_top2_experts"])
            for node in range(num_nodes)
        ))
    if not layers:
        raise ValueError("trace step has no per-node routing snapshots")
    return tuple(layers)


def _path_to_root(node: int, parent: Sequence[int]) -> tuple[int, ...]:
    path = []
    while node >= 0:
        path.append(node)
        node = int(parent[node])
    return tuple(reversed(path))


def _expert_union(
    nodes: Iterable[int], node_experts: Sequence[Sequence[int]],
) -> set[int]:
    return {
        int(expert)
        for node in nodes
        for expert in node_experts[int(node)]
    }


def _validate_stages(parent: Sequence[int], stages: Sequence[Sequence[int]]) -> None:
    previous: set[int] = set()
    for index, stage in enumerate(stages):
        current = {int(node) for node in stage}
        if 0 not in current:
            raise ValueError(f"stage {index} does not contain the root")
        if not previous.issubset(current):
            raise ValueError("stages must be nested")
        for node in current:
            if node < 0 or node >= len(parent):
                raise ValueError(f"invalid node {node} in stage {index}")
            if node != 0 and int(parent[node]) not in current:
                raise ValueError(f"stage {index} is not prefix-closed")
        previous = current
    if previous != set(range(len(parent))):
        raise ValueError("final stage must contain the complete tree")
"""Replay fixed-compute prefetch policies from an end-to-end routing trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def allocate_prefetch_rows(
    scores: list[list[float]], total: int, minimum: int, maximum: int,
) -> list[int]:
    layers = len(scores)
    if not layers or not minimum * layers <= total <= maximum * layers:
        raise ValueError("infeasible global prefetch budget")
    budgets = [minimum] * layers
    candidates = []
    for layer, row in enumerate(scores):
        ranked = sorted(row, reverse=True)
        if maximum > len(ranked):
            raise ValueError("maximum exceeds the number of experts")
        for rank in range(minimum, maximum):
            candidates.append((-ranked[rank], layer, rank))
    candidates.sort()
    for _, layer, _ in candidates[:total - minimum * layers]:
        budgets[layer] += 1
    return budgets


def predicted_sets(
    scores: list[list[float]], budgets: list[int],
) -> list[set[int]]:
    result = []
    for row, budget in zip(scores, budgets, strict=True):
        order = sorted(range(len(row)), key=lambda expert: (-row[expert], expert))
        result.append(set(order[:budget]))
    return result


def replay(
    demand_trace: list[list[list[float]]],
    expert_trace: list[list[list[int]]],
    budget: int,
    minimum: int,
    maximum: int,
    ema_decay: float,
) -> dict[str, float]:
    if len(demand_trace) != len(expert_trace) or not demand_trace:
        raise ValueError("demand and expert traces must have equal non-zero length")
    layers = len(demand_trace[0])
    ema = [row.copy() for row in demand_trace[0]]
    uniform_misses = 0
    global_misses = 0
    global_budget_changes = 0

    # Step zero is full-copy startup in the runtime and therefore has no repair.
    for step in range(1, len(expert_trace)):
        actual = [set(row) for row in expert_trace[step]]
        uniform_budgets = [budget] * layers
        global_budgets = allocate_prefetch_rows(
            ema, total=layers * budget, minimum=minimum, maximum=maximum,
        )
        uniform = predicted_sets(ema, uniform_budgets)
        adaptive = predicted_sets(ema, global_budgets)
        uniform_misses += sum(len(want - have) for want, have in zip(actual, uniform))
        global_misses += sum(len(want - have) for want, have in zip(actual, adaptive))
        global_budget_changes += sum(value != budget for value in global_budgets)
        current = demand_trace[step]
        ema = [
            [ema_decay * old + (1.0 - ema_decay) * new
             for old, new in zip(old_row, new_row, strict=True)]
            for old_row, new_row in zip(ema, current, strict=True)
        ]

    steps = len(expert_trace)
    reduction = (
        1.0 - global_misses / uniform_misses if uniform_misses else 0.0
    )
    return {
        "steps": float(steps),
        "uniform_misses": float(uniform_misses),
        "global_misses": float(global_misses),
        "uniform_misses_per_step": uniform_misses / steps,
        "global_misses_per_step": global_misses / steps,
        "repair_reduction": reduction,
        "changed_layer_budgets_per_predicted_step": (
            global_budget_changes / max(steps - 1, 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--maximum", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.8)
    args = parser.parse_args()

    rows = json.loads(args.trace.read_text())
    for row in rows:
        result = replay(
            row["demand_trace"], row["expert_trace"], args.budget,
            args.minimum, args.maximum, args.ema_decay,
        )
        print(f"B={row['budget']} N={row['tree_size']} steps={int(result['steps'])}")
        print(
            f"uniform repair rows/step: {result['uniform_misses_per_step']:.2f}"
        )
        print(
            f"global  repair rows/step: {result['global_misses_per_step']:.2f}"
        )
        print(f"repair reduction: {result['repair_reduction']:.1%}")
        print(
            "changed layer budgets/predicted step: "
            f"{result['changed_layer_budgets_per_predicted_step']:.1f}"
        )


if __name__ == "__main__":
    main()
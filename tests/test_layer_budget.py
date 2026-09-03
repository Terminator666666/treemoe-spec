from itertools import product

import pytest
import torch

from treemoe.engine.layer_budget import LayerBudgetAllocator, allocate_layer_budgets


def test_allocate_layer_budgets_respects_exact_global_constraint():
    demand = torch.tensor([
        [0.70, 0.20, 0.09, 0.01],
        [0.30, 0.25, 0.24, 0.21],
    ])
    budgets = allocate_layer_budgets(demand, total_budget=6, min_budget=2)
    assert torch.equal(budgets, torch.tensor([2, 4]))
    assert int(budgets.sum()) == 6


def test_allocate_layer_budgets_is_scale_invariant_and_deterministic():
    demand = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    expected = allocate_layer_budgets(demand, total_budget=6, min_budget=2)
    assert torch.equal(expected, torch.tensor([3, 3]))
    assert torch.equal(
        allocate_layer_budgets(demand * torch.tensor([[2.0], [7.0]]), 6, 2),
        expected,
    )


@pytest.mark.parametrize("total", [3, 9])
def test_allocate_layer_budgets_rejects_infeasible_total(total):
    with pytest.raises(ValueError, match="total_budget"):
        allocate_layer_budgets(torch.ones(2, 4), total_budget=total, min_budget=2)


def test_allocator_builds_top_expert_bitmap_with_matching_row_counts():
    allocator = LayerBudgetAllocator(
        num_layers=2, num_experts=4, average_budget=3,
        min_budget=2, ema_decay=0.0,
    )
    demand = torch.tensor([
        [0.70, 0.20, 0.09, 0.01],
        [0.30, 0.25, 0.24, 0.21],
    ])
    allocator.start_observation()
    allocator.observe(0, demand[0])
    allocator.observe(1, demand[1])
    plan = allocator.finish_observation()

    assert torch.equal(plan.budgets, torch.tensor([2, 4]))
    assert torch.equal(plan.prefetch_bitmap.sum(1), plan.budgets)
    assert torch.equal(plan.prefetch_bitmap[0], torch.tensor([True, True, False, False]))
    assert plan.prefetch_bitmap[1].all()


def test_allocator_requires_every_layer_observation():
    allocator = LayerBudgetAllocator(2, 4, average_budget=3)
    allocator.start_observation()
    allocator.observe(0, torch.ones(4))
    with pytest.raises(RuntimeError, match=r"layers \[1\]"):
        allocator.finish_observation()


def test_allocator_reset_drops_cross_prompt_history():
    allocator = LayerBudgetAllocator(2, 4, average_budget=3, ema_decay=0.5)
    allocator.start_observation()
    allocator.observe(0, torch.tensor([9.0, 1.0, 0.0, 0.0]))
    allocator.observe(1, torch.ones(4))
    allocator.finish_observation()

    allocator.reset()

    assert allocator._ema_demand is None
    assert allocator.plan.prefetch_bitmap is None
    assert torch.equal(allocator.plan.budgets, torch.tensor([3, 3]))


def test_uniform_control_uses_same_exact_plan_without_cross_layer_reallocation():
    allocator = LayerBudgetAllocator(
        2, 4, average_budget=3, min_budget=2, ema_decay=0.0, adaptive=False,
    )
    allocator.start_observation()
    allocator.observe(0, torch.tensor([0.70, 0.20, 0.09, 0.01]))
    allocator.observe(1, torch.tensor([0.30, 0.25, 0.24, 0.21]))
    plan = allocator.finish_observation()

    assert torch.equal(plan.budgets, torch.tensor([3, 3]))
    assert torch.equal(plan.prefetch_bitmap.sum(1), plan.budgets)


def test_marginal_greedy_matches_exhaustive_prefix_optimum():
    demand = torch.tensor([
        [0.70, 0.20, 0.09, 0.01],
        [0.30, 0.25, 0.24, 0.21],
        [0.50, 0.30, 0.15, 0.05],
    ])
    budgets = allocate_layer_budgets(demand, total_budget=8, min_budget=2)
    normalized = demand / demand.sum(1, keepdim=True)
    sorted_demand = normalized.sort(dim=1, descending=True).values

    def retained(candidate):
        return sum(
            float(sorted_demand[layer, :budget].sum())
            for layer, budget in enumerate(candidate)
        )

    feasible = [candidate for candidate in product(range(2, 5), repeat=3)
                if sum(candidate) == 8]
    assert retained(budgets.tolist()) == pytest.approx(
        max(retained(candidate) for candidate in feasible)
    )
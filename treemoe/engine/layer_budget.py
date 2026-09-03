"""Global transfer-constrained layer-wise expert budget allocation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerBudgetPlan:
    budgets: torch.Tensor
    prefetch_bitmap: torch.Tensor | None


def allocate_layer_budgets(
    demand: torch.Tensor,
    total_budget: int,
    min_budget: int = 2,
    max_budget: int | None = None,
    objective: str = "log_mass",
) -> torch.Tensor:
    """Allocate an exact global expert-row budget by marginal utility.

    Each layer first receives ``min_budget`` experts. Remaining rows are
    assigned by descending marginal utility. ``log_mass`` maximizes the
    product of per-layer retained router mass, while ``mass`` retains the
    original additive objective for ablation. Inputs and outputs live on CPU
    because the result is consumed as Python launch-time integers.
    """
    if demand.ndim != 2:
        raise ValueError(f"demand must be [layers, experts], got {tuple(demand.shape)}")
    num_layers, num_experts = demand.shape
    upper = num_experts if max_budget is None else max_budget
    if not 1 <= min_budget <= upper <= num_experts:
        raise ValueError("require 1 <= min_budget <= max_budget <= num_experts")
    minimum_total = num_layers * min_budget
    maximum_total = num_layers * upper
    if not minimum_total <= total_budget <= maximum_total:
        raise ValueError(
            f"total_budget must be in [{minimum_total}, {maximum_total}], got {total_budget}"
        )
    if not torch.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("demand must be finite and non-negative")
    if objective not in {"mass", "log_mass"}:
        raise ValueError("objective must be 'mass' or 'log_mass'")

    scores = demand.detach().float().cpu()
    scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    sorted_scores = scores.sort(dim=1, descending=True, stable=True).values
    retained = sorted_scores.cumsum(dim=1)
    budgets = torch.full((num_layers,), min_budget, dtype=torch.long)
    remaining = total_budget - minimum_total
    candidates = []
    for layer in range(num_layers):
        for rank in range(min_budget, upper):
            if objective == "mass":
                utility = float(sorted_scores[layer, rank])
            else:
                before = retained[layer, rank - 1].clamp_min(1e-12)
                after = retained[layer, rank].clamp_min(1e-12)
                utility = float(torch.log(after) - torch.log(before))
            candidates.append((-utility, layer, rank))
    candidates.sort()
    for _, layer, _ in candidates[:remaining]:
        budgets[layer] += 1
    return budgets


class LayerBudgetAllocator:
    """Online allocator driven by the previous target forward's router demand."""

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        average_budget: int,
        min_budget: int = 2,
        max_budget: int | None = None,
        ema_decay: float = 0.8,
        adaptive: bool = True,
        objective: str = "log_mass",
    ) -> None:
        upper = num_experts if max_budget is None else max_budget
        if not min_budget <= average_budget <= upper <= num_experts:
            raise ValueError(
                "require min_budget <= average_budget <= max_budget <= num_experts"
            )
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.total_budget = num_layers * average_budget
        self.average_budget = average_budget
        self.min_budget = min_budget
        self.max_budget = upper
        self.ema_decay = ema_decay
        self.adaptive = adaptive
        self.objective = objective
        initial = torch.full((num_layers,), average_budget, dtype=torch.long)
        self.plan = LayerBudgetPlan(initial, None)
        self._ema_demand: torch.Tensor | None = None
        self._observed: torch.Tensor | None = None
        self._seen = torch.zeros(num_layers, dtype=torch.bool)
        self.budget_histogram = torch.zeros(num_experts + 1, dtype=torch.long)

    def reset(self) -> None:
        self.plan = LayerBudgetPlan(
            torch.full((self.num_layers,), self.average_budget, dtype=torch.long),
            None,
        )
        self._ema_demand = None
        self._observed = None
        self._seen.zero_()

    def start_observation(self) -> None:
        self._seen.zero_()

    def record_plan_use(self) -> None:
        self.budget_histogram.add_(
            torch.bincount(self.plan.budgets, minlength=self.num_experts + 1)
        )

    def observe(self, layer_idx: int, demand: torch.Tensor) -> None:
        if demand.shape != (self.num_experts,):
            raise ValueError(
                f"layer demand must be [{self.num_experts}], got {tuple(demand.shape)}"
            )
        if self._observed is None or self._observed.device != demand.device:
            self._observed = torch.empty(
                self.num_layers, self.num_experts,
                dtype=torch.float32, device=demand.device,
            )
        self._observed[layer_idx].copy_(demand.detach().float())
        self._seen[layer_idx] = True

    def finish_observation(self) -> LayerBudgetPlan:
        if self._observed is None or not self._seen.all():
            missing = (~self._seen).nonzero().flatten().tolist()
            raise RuntimeError(f"missing router demand for layers {missing}")
        current = self._observed.cpu()
        current = current / current.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if self._ema_demand is None:
            self._ema_demand = current
        else:
            self._ema_demand.mul_(self.ema_decay).add_(
                current, alpha=1.0 - self.ema_decay
            )
        if self.adaptive:
            budgets = allocate_layer_budgets(
                self._ema_demand,
                self.total_budget,
                min_budget=self.min_budget,
                max_budget=self.max_budget,
                objective=self.objective,
            )
        else:
            budgets = torch.full(
                (self.num_layers,), self.average_budget, dtype=torch.long,
            )
        order = torch.argsort(self._ema_demand, dim=1, descending=True, stable=True)
        bitmap = torch.zeros_like(self._ema_demand, dtype=torch.bool)
        for layer, budget in enumerate(budgets.tolist()):
            bitmap[layer, order[layer, :budget]] = True
        self.plan = LayerBudgetPlan(budgets, bitmap)
        return self.plan

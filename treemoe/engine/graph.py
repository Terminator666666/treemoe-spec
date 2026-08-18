"""Task 3.2: whole-step CUDA Graph capture (spec §2).

Preconditions enforced here:
  * static tree shape (N, max_depth fixed at capture time)
  * all step inputs/outputs live in a persistent tensor pool
  * expert_budget is a graph-external writable scalar (CPU PI controller may
    update it between replays without recapture, spec §3.3)

Capture unit: one full draft -> verify -> commit iteration. Anything that
still syncs to CPU (reference paths, python tree building) must be replaced by
kernel/GPU equivalents before capture; `assert_capturable` walks known
offenders and raises with a task pointer instead of failing mid-capture.
"""

from __future__ import annotations

import torch


class StepGraph:
    def __init__(self, engine, device: str = "cuda"):
        self.engine = engine
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        # persistent I/O pool
        h = engine.target.cfg.hidden_dim
        self.in_last_token = torch.zeros((), dtype=torch.long, device=device)
        self.in_root_feature = torch.zeros(h, dtype=torch.bfloat16, device=device)
        self.out_tokens = torch.full((engine.max_depth + 1,), -1, dtype=torch.long, device=device)
        self.out_num = torch.zeros((), dtype=torch.long, device=device)
        self.out_feature = torch.zeros(h, dtype=torch.bfloat16, device=device)
        self.expert_budget = torch.tensor(engine.expert_budget, device=device)

    def assert_capturable(self) -> None:
        from treemoe.kernels.op1_tree_moe import HAS_TRITON

        problems = []
        if not HAS_TRITON:
            problems.append("Triton missing: op1/op4 fall back to reference (CPU sync)")
        if self.engine.target.moe_fn.__name__ == "naive_moe":
            problems.append("naive_moe injected: switch to op1 (plan Task 2.5) first")
        if problems:
            raise RuntimeError("not graph-capturable:\n  - " + "\n  - ".join(problems))

    def capture(self, warmup: int = 3) -> None:
        self.assert_capturable()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._step_inplace()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step_inplace()

    def _step_inplace(self) -> None:
        new_tokens, feature = self.engine.step(self.in_last_token, self.in_root_feature)
        n = min(len(new_tokens), self.out_tokens.shape[0])
        self.out_tokens.fill_(-1)
        self.out_tokens[:n] = torch.tensor(new_tokens[:n], device=self.device)
        self.out_num.fill_(n)
        self.out_feature.copy_(feature)

    def replay(self, last_token: torch.Tensor, root_feature: torch.Tensor):
        assert self.graph is not None, "call capture() first"
        self.in_last_token.copy_(last_token)
        self.in_root_feature.copy_(root_feature)
        self.graph.replay()
        return self.out_tokens, self.out_num, self.out_feature

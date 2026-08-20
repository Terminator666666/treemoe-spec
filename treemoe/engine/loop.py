"""Task 1.3 / 2.5: the draft -> verify -> commit main loop.

M1 milestone: correct but slow, all reference paths.
Op injection points:
  * moe_fn        -> treemoe.kernels.op1.tree_moe_forward (Task 2.5)
  * verify/commit -> treemoe.kernels.op4.fused_verify_commit (Task 3.1)
  * prefetcher    -> treemoe.kernels.op2 (Task 4.2)
Graph capture wraps step() (Task 3.2, engine/graph.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from treemoe.engine.tree import build_eagle2_tree
from treemoe.kernels.op4_commit import fused_verify_commit
from treemoe.model.mixtral import MixtralForward


@dataclass
class GenerationStats:
    steps: int = 0
    tokens: int = 0

    @property
    def mean_accept_len(self) -> float:
        return self.tokens / max(self.steps, 1)


class SpecDecodeEngine:
    def __init__(
        self,
        target: MixtralForward,
        draft,                        # EagleDraftModel
        tree_size: int = 64,
        max_depth: int = 6,
        expert_budget: int = 8,       # B=8 == lossless mode (spec §3.3)
        temperature: float = 0.0,
    ):
        self.target = target
        self.draft = draft
        self.tree_size = tree_size
        self.max_depth = max_depth
        self.expert_budget = expert_budget
        self.temperature = temperature
        self.stats = GenerationStats()

    @torch.inference_mode()
    def prefill(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run target prefill; returns (last_logits [V], last_penultimate [H])."""
        positions = torch.arange(token_ids.shape[0], device=token_ids.device)
        logits, hidden = self.target.forward(token_ids, positions, return_hidden=True)
        return logits[-1], hidden[-1]

    @torch.inference_mode()
    def step(self, last_token: torch.Tensor, root_feature: torch.Tensor):
        """One draft-verify-commit iteration. Returns (new_tokens list, next_feature)."""
        kv = self.target.kv
        root_pos = kv.seq_len

        tree = build_eagle2_tree(
            self.draft.step, last_token, root_feature, root_pos,
            tree_size=self.tree_size, max_depth=self.max_depth,
            device=last_token.device.type,
        )

        # verification forward over the whole tree (root token occupies slot 0)
        positions = root_pos + _depths(tree.parent, self.max_depth)
        logits, hidden = self.target.forward(
            tree.tokens.clamp(min=0), positions,
            tree_mask=tree.attn_mask, return_hidden=True,
        )

        draft_probs = torch.zeros_like(logits)  # greedy mode: unused by verifier
        res = fused_verify_commit(
            logits.float(), draft_probs, tree.tokens, tree.parent, tree.children,
            kv=kv, temperature=self.temperature, max_depth=self.max_depth,
        )

        # next root feature = penultimate hidden at the last accepted node (spec
        # §3.4 step 6) — pure tensor indexing, stays on GPU. num==0 falls back
        # to slot 0 (root): accepted_slots[0] is -1 then, clamp restores 0.
        last_idx = (res.num_accepted - 1).clamp(min=0)
        next_feature = hidden[res.accepted_slots[last_idx].clamp(min=0)]
        self._last_token_gpu = res.bonus_token  # generate() reuses, no re-upload

        # ONE device->host copy for everything the host actually needs
        # (token ids for output/EOS). The old code did 3+num tiny syncs.
        vals = torch.cat([
            res.num_accepted.view(1), res.bonus_token.view(1), res.accepted_tokens,
        ]).tolist()
        num = vals[0]
        new_tokens = vals[2:2 + num] + [vals[1]]

        self.stats.steps += 1
        self.stats.tokens += len(new_tokens)
        return new_tokens, next_feature

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 128,
                 eos_token_id: int = 2) -> list[int]:
        logits, feature = self.prefill(prompt_ids)
        last = logits.argmax()
        out = [int(last)]
        self.draft.reset()
        while len(out) < max_new_tokens:
            new_tokens, feature = self.step(last, feature)
            out.extend(new_tokens)
            # new_tokens[-1] is always the bonus token, which step() kept on
            # GPU — no host->device re-upload per step
            last = self._last_token_gpu
            if eos_token_id in new_tokens:
                break
        return out[:max_new_tokens]


def _depths(parent: torch.Tensor, max_depth: int) -> torch.Tensor:
    """Node depths by level propagation: d[i] = d[parent[i]] + 1 repeated
    max_depth times (BFS order converges level by level). Pure tensor ops —
    the old per-element loop cost 2N tiny GPU->CPU syncs per step."""
    root = parent < 0
    safe_parent = parent.clamp(min=0)
    d = torch.zeros_like(parent)
    for _ in range(max_depth):
        d = torch.where(root, 0, d[safe_parent] + 1)
    return d

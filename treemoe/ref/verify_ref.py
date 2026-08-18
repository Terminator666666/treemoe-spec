"""Reference for op4: tree rejection sampling + accept path + commit (spec §3.4).

Implements standard multi-round speculative sampling generalized to trees
(SpecInfer/EAGLE style): walk from the root; at each node compare target dist p
against draft dist q for each child; accepted child -> descend; all children
rejected -> sample bonus token from normalized max(0, p - q_mix) and stop.

Deterministic under a fixed torch.Generator — the Triton Philox kernel must
reproduce identical accept paths given the same uniforms (tested by injecting
the uniform stream, tests/test_op4.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VerifyResult:
    accepted_slots: torch.Tensor   # [max_depth+1] tree-node indices, -1 padded
    accepted_tokens: torch.Tensor  # [max_depth+1] token ids, -1 padded
    bonus_token: torch.Tensor      # [] scalar token id sampled after last accept
    num_accepted: torch.Tensor     # [] scalar (stays on GPU in graph mode)


def _postprocess(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        # greedy mode is handled by caller comparing argmax
        return logits
    return logits / temperature


def tree_verify_ref(
    target_logits: torch.Tensor,   # [N, V] fp32, node i's next-token dist
    draft_probs: torch.Tensor,     # [N, V] fp32, q used when proposing node i's children
    tree_tokens: torch.Tensor,     # [N] token id at each node (root = last committed token's next)
    tree_parent: torch.Tensor,     # [N] parent index, -1 for root
    children: list[list[int]],     # adjacency (CPU metadata; static per tree shape)
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    uniforms: torch.Tensor | None = None,  # [N] injected randomness for kernel parity tests
) -> VerifyResult:
    n, v = target_logits.shape
    device = target_logits.device
    accepted_slots: list[int] = []
    accepted_tokens: list[int] = []

    if uniforms is None:
        uniforms = torch.rand(n, generator=generator, device=device)

    node = 0  # root slot: its target_logits give the dist for its children
    while True:
        p = torch.softmax(_postprocess(target_logits[node], temperature), dim=-1)
        kids = children[node]
        accepted_child = -1
        residual = p.clone()
        for c in kids:
            tok = int(tree_tokens[c])
            if temperature == 0.0:
                ok = tok == int(p.argmax())
            else:
                q = draft_probs[c]  # already a probability distribution
                ratio = (residual[tok] / q[tok].clamp_min(1e-20)).clamp(max=1.0)
                ok = bool(uniforms[c] < ratio)
                if not ok:
                    residual = (residual - q).clamp_min(0)
                    residual = residual / residual.sum().clamp_min(1e-20)
            if ok:
                accepted_child = c
                break
        if accepted_child >= 0:
            accepted_slots.append(accepted_child)
            accepted_tokens.append(int(tree_tokens[accepted_child]))
            node = accepted_child
            continue
        # all children rejected (or leaf): sample bonus from residual / argmax
        if temperature == 0.0:
            bonus = p.argmax()
        else:
            bonus = torch.multinomial(residual, 1, generator=generator).squeeze(0)
        break

    pad = torch.full((n,), -1, dtype=torch.long, device=device)
    slots = pad.clone()
    toks = pad.clone()
    if accepted_slots:
        slots[: len(accepted_slots)] = torch.tensor(accepted_slots, device=device)
        toks[: len(accepted_tokens)] = torch.tensor(accepted_tokens, device=device)
    return VerifyResult(
        accepted_slots=slots,
        accepted_tokens=toks,
        bonus_token=bonus,
        num_accepted=torch.tensor(len(accepted_slots), device=device),
    )

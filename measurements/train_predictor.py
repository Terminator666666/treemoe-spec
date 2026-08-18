"""Task 4.1: train the 1M-param cross-layer router predictor (spec §3.2).

Data: run Mixtral over ShareGPT samples, collect
  (penultimate hidden state f_t, per-layer top-2 expert labels y_{t,l}).
Model: single Linear H -> L*E, 32 independent 8-way heads, cross-entropy on
both top-2 slots. Gate: recall@4 >= 0.70, else fall back to the
last-step-activation heuristic (spec §4 risk table).

Collection reuses measurements/collect_routing.py hooks plus a hidden-state
hook on model.model.layers[-2].
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from treemoe.kernels.op2_prefetch import RouterPredictor


def train(features: torch.Tensor, labels: torch.Tensor, epochs: int = 3,
          lr: float = 1e-3, device: str = "cuda") -> RouterPredictor:
    """features: [T, H] fp32; labels: [T, L, 2] int64 top-2 expert ids."""
    t, h = features.shape
    _, l, _ = labels.shape
    model = RouterPredictor(hidden=h, num_layers=l).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(features, labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    for ep in range(epochs):
        total = 0.0
        for f, y in dl:
            f, y = f.to(device), y.to(device)
            logits = model(f)                                  # [B, L, E]
            flat = logits.reshape(-1, logits.shape[-1])        # [B*L, E]
            loss = (
                F.cross_entropy(flat, y[..., 0].reshape(-1))
                + F.cross_entropy(flat, y[..., 1].reshape(-1))
            ) * 0.5
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * f.shape[0]
        print(f"epoch {ep}: loss {total / t:.4f}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="measurements/data/predictor_dataset.pt",
                    help="dict(features=[T,H], labels=[T,L,2]) from collection run")
    ap.add_argument("--out", default="checkpoints/router_predictor.pt")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    d = torch.load(args.data)
    features, labels = d["features"].float(), d["labels"].long()
    split = int(0.95 * features.shape[0])
    model = train(features[:split], labels[:split], epochs=args.epochs)

    recall4 = model.recall_at(features[split:].cuda(), labels[split:].cuda(), k=4)
    recall2 = model.recall_at(features[split:].cuda(), labels[split:].cuda(), k=2)
    print(f"recall@2={recall2:.3f} recall@4={recall4:.3f}")
    print("GATE:", "PASS" if recall4 >= 0.70 else
          "FAIL -> use last-step-activation heuristic (spec §4 risk table)")
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()

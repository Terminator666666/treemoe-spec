"""Task 0.3: produce the three observation figures from routing traces (spec §1.2).

Fig 1: tree size N vs expected #activated experts per layer (activation inflation)
Fig 2: parent-child / sibling top-2 expert Jaccard similarity (routing locality)
Fig 3: aggregated per-expert gating mass long-tail (budget-routing feasibility)

Decision gate (plan Task 0.3): proceed only if median parent-child Jaccard >= 0.5
and gating mass is clearly long-tailed.

Usage: python measurements/analyze.py --traces measurements/data/routing_traces.pt
Runs on CPU; no model needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

TREE_SIZES = (8, 16, 32, 64, 128)


def simulate_tree_node_groups(
    seq_len: int, n_nodes: int, max_depth: int = 6, branching: int = 4, seed: int = 0
) -> list[list[int]]:
    """Sample groups of trace positions that mimic EAGLE-2 tree token sets.

    A verification tree's nodes are consecutive speculative continuations of one
    prefix position, so we approximate a tree rooted at position p by the window
    [p, p+depth) with `branching` samples per depth level (nearby positions of
    the real trajectory share the tree's local-context statistics).
    Returns list of (node position list, parent position list aligned by index).
    """
    g = torch.Generator().manual_seed(seed)
    groups = []
    max_root = seq_len - max_depth - 1
    if max_root <= 1:
        return groups
    n_trees = min(200, max_root)
    roots = torch.randperm(max_root, generator=g)[:n_trees]
    for r in roots.tolist():
        nodes, parents = [r], [None]
        depth_nodes = [r]
        for d in range(1, max_depth):
            nxt = []
            for parent in depth_nodes:
                for _ in range(branching):
                    if len(nodes) >= n_nodes:
                        break
                    child = min(parent + 1, seq_len - 1)
                    nodes.append(child)
                    parents.append(parent)
                    nxt.append(child)
            depth_nodes = nxt or depth_nodes
            if len(nodes) >= n_nodes:
                break
        groups.append((nodes[:n_nodes], parents[:n_nodes]))
    return groups


def fig1_activation_inflation(traces: list[dict]) -> dict[int, float]:
    """E[#distinct experts per layer] as a function of tree size."""
    result = {}
    for n in TREE_SIZES:
        counts = []
        for tr in traces:
            layers = tr["layers"]
            seq_len = next(iter(layers.values()))[0].shape[0]
            for nodes, _ in simulate_tree_node_groups(seq_len, n):
                idx = torch.tensor(nodes)
                for ids, _gates in layers.values():
                    counts.append(ids[idx].unique().numel())
        result[n] = float(torch.tensor(counts, dtype=torch.float32).mean())
    return result


def fig2_routing_locality(traces: list[dict]) -> torch.Tensor:
    """Jaccard similarity of top-2 expert sets between adjacent positions."""
    sims = []
    for tr in traces:
        for ids, _ in tr["layers"].values():
            a, b = ids[:-1], ids[1:]  # parent/child proxy: consecutive positions
            for i in range(a.shape[0]):
                sa, sb = set(a[i].tolist()), set(b[i].tolist())
                sims.append(len(sa & sb) / len(sa | sb))
    return torch.tensor(sims)


def fig3_gating_longtail(traces: list[dict]) -> torch.Tensor:
    """Sorted per-expert aggregated gate mass within simulated trees, normalized."""
    masses = []
    for tr in traces:
        layers = tr["layers"]
        seq_len = next(iter(layers.values()))[0].shape[0]
        for nodes, _ in simulate_tree_node_groups(seq_len, 64):
            idx = torch.tensor(nodes)
            for ids, gates in layers.values():
                m = torch.zeros(8)
                m.scatter_add_(0, ids[idx].long().flatten(), gates[idx].flatten())
                m = m / m.sum().clamp_min(1e-9)
                masses.append(m.sort(descending=True).values)
    return torch.stack(masses).mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="measurements/data/routing_traces.pt")
    ap.add_argument("--outdir", default="measurements/data")
    args = ap.parse_args()
    traces = torch.load(args.traces)

    inflation = fig1_activation_inflation(traces)
    locality = fig2_routing_locality(traces)
    longtail = fig3_gating_longtail(traces)

    med_jaccard = locality.median().item()
    tail_mass_top5 = longtail[:5].sum().item()
    print("Fig1 activation inflation:", {k: round(v, 2) for k, v in inflation.items()})
    print(f"Fig2 median parent-child Jaccard: {med_jaccard:.3f}")
    print(f"Fig3 top-5 experts hold {tail_mass_top5 * 100:.1f}% of gate mass")
    print("--- decision gate ---")
    print("locality OK" if med_jaccard >= 0.5 else "locality WEAK: revisit op1/op3 design")
    print("longtail OK" if tail_mass_top5 >= 0.85 else "longtail WEAK: budget routing risky")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"fig1": inflation, "fig2": locality, "fig3": longtail},
        outdir / "observations.pt",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(list(inflation.keys()), list(inflation.values()), "o-")
        axes[0].set(xlabel="tree size N", ylabel="E[#experts/layer]", title="Activation inflation")
        axes[0].axhline(8, ls="--", c="gray")
        axes[1].hist(locality.numpy(), bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[1].set(xlabel="top-2 Jaccard", title="Routing locality")
        axes[2].bar(range(1, 9), longtail.numpy())
        axes[2].set(xlabel="expert rank", ylabel="gate mass", title="Gating long tail")
        fig.tight_layout()
        fig.savefig(outdir / "observations.png", dpi=150)
        print(f"figures -> {outdir / 'observations.png'}")
    except ImportError:
        print("matplotlib unavailable; skipped plotting")


if __name__ == "__main__":
    main()

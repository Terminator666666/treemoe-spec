"""Summarize TreeMoE-Spec full-stage execution traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _timing(record: dict, phase: str, clock: str) -> float:
    return record.get("timing_ms", {}).get(phase, {}).get(clock, 0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize(config: dict, show_tree: bool = False) -> None:
    trace = config["trace"]
    steps = trace["steps"]
    print(
        f"\nCONFIG objective={config['routing_objective']} B={config['budget']} "
        f"N={config['tree_size']} prompts={config['num_prompts']} "
        f"steps={len(steps)}"
    )
    if not steps:
        print("no verification steps")
        return

    prefills = trace.get("prefills", [])
    if prefills:
        print("\nPREFILL (mean ms / prompt)")
        print("phase                  host       gpu")
        for phase in ("total", "target", "draft_seed"):
            host = _mean([_timing(row, phase, "host") for row in prefills])
            gpu = _mean([_timing(row, phase, "gpu") for row in prefills])
            print(f"{phase:20s} {host:10.2f} {gpu:9.2f}")
        planned = _mean([
            float(sum(copy.get("rows") or 0
                      for copy in row.get("prefetch_copies", [])))
            for row in prefills
        ])
        print(f"{'planned_rows':20s} {planned:10.1f}")

    print("\nPER-STEP TREE / ACCEPTANCE / ENGINE")
    print(
        "step valid depth accepted emitted draftGPU targetGPU verifyGPU "
        "repair plan accepted_path path_tokens path_prob"
    )
    for step in steps:
        tree = step["tree"]
        acceptance = step["acceptance"]
        layers = step["layers"]
        repair_rows = sum(layer.get("repair_rows", 0) for layer in layers)
        planned_rows = sum(
            copy.get("rows") or 0 for copy in step.get("prefetch_copies", [])
        )
        accepted_slots = acceptance["accepted_slots"]
        full_path = (
            tree["paths"][accepted_slots[-1]] if accepted_slots else [0]
        )
        slots = ">".join(map(str, full_path))
        path_tokens = ">".join(str(tree["tokens"][node]) for node in full_path)
        probabilities = ">".join(
            f"{value:.3f}"
            for value in acceptance["accepted_path_probability"]
        ) or "-"
        print(
            f"{step['index']:4d} {tree['num_valid']:5d} {tree['max_depth']:5d} "
            f"{len(acceptance['accepted_slots']):8d} "
            f"{acceptance['emitted_tokens']:7d} "
            f"{_timing(step, 'draft_tree', 'gpu'):8.2f} "
            f"{_timing(step, 'target_verify', 'gpu'):9.2f} "
            f"{_timing(step, 'verify_commit', 'gpu'):9.2f} "
            f"{repair_rows:6d} {planned_rows:4d} {slots:>14s} "
            f"{path_tokens:>18s} {probabilities}"
        )
        if show_tree:
            print("  TREE NODES: node parent depth token accept_prob root_path")
            for node in range(tree["num_valid"]):
                path = ">".join(map(str, tree["paths"][node]))
                print(
                    f"  {node:4d} {tree['parent'][node]:6d} "
                    f"{tree['depth'][node]:5d} {tree['tokens'][node]:5d} "
                    f"{tree['accept_probability'][node]:11.6f} {path}"
                )

    print("\nDRAFT TREE LEVELS (mean ms / occurrence)")
    print("depth frontier candidates next  forwardGPU selectGPU selectHost")
    depths = sorted({
        level["depth"] for step in steps for level in step.get("draft_levels", [])
    })
    for depth in depths:
        levels = [
            level for step in steps for level in step.get("draft_levels", [])
            if level["depth"] == depth
        ]
        print(
            f"{depth:5d} {_mean([float(v['frontier_nodes']) for v in levels]):8.1f} "
            f"{_mean([float(v['generated_candidates']) for v in levels]):10.1f} "
            f"{_mean([float(v['next_frontier_nodes']) for v in levels]):4.1f} "
            f"{_mean([_timing(v, 'draft_forward', 'gpu') for v in levels]):11.2f} "
            f"{_mean([_timing(v, 'candidate_select', 'gpu') for v in levels]):9.2f} "
            f"{_mean([_timing(v, 'candidate_select', 'host') for v in levels]):10.2f}"
        )

    print("\nENGINE PHASES (mean ms / verification step)")
    print("phase                  host       gpu")
    for phase in (
        "total", "draft_tree", "tree_snapshot", "target_verify", "verify_commit",
        "result_d2h", "draft_commit",
    ):
        host = _mean([_timing(step, phase, "host") for step in steps])
        gpu = _mean([_timing(step, phase, "gpu") for step in steps])
        print(f"{phase:20s} {host:10.2f} {gpu:9.2f}")

    layer_phases = (
        "attention", "moe_prepare", "moe_total", "route", "expert_ids_d2h",
        "repair", "routing_snapshot", "expert_gemm", "cold_cpu",
        "prefetch_release",
    )
    print("\nLAYER HOTSPOTS (mean ms and rows / verification step)")
    print(
        "layer  attnGPU prepGPU moeGPU routeGPU idsHost repairGPU gemmGPU "
        "miss plan"
    )
    layer_count = max(len(step["layers"]) for step in steps)
    for layer_index in range(layer_count):
        rows = [
            step["layers"][layer_index] for step in steps
            if len(step["layers"]) > layer_index
        ]
        planned = []
        for step in steps:
            match = next(
                (copy for copy in step.get("prefetch_copies", [])
                 if copy["layer"] == layer_index),
                None,
            )
            planned.append((match or {}).get("rows") or 0)
        values = {
            phase: _mean([_timing(row, phase, "gpu") for row in rows])
            for phase in layer_phases
        }
        ids_host = _mean([
            _timing(row, "expert_ids_d2h", "host") for row in rows
        ])
        misses = _mean([float(row.get("repair_rows", 0)) for row in rows])
        print(
            f"{layer_index:5d} {values['attention']:8.2f} "
            f"{values['moe_prepare']:7.2f} {values['moe_total']:6.2f} "
            f"{values['route']:8.2f} {ids_host:7.2f} "
            f"{values['repair']:9.2f} {values['expert_gemm']:7.2f} "
            f"{misses:4.1f} {_mean([float(v) for v in planned]):4.1f}"
        )

    print("\nTARGET BREAKDOWN (sum of layer means, ms / step)")
    for phase in ("attention", "moe_prepare", "moe_total", "route", "repair",
                  "expert_gemm"):
        total = 0.0
        for layer_index in range(layer_count):
            rows = [
                step["layers"][layer_index] for step in steps
                if len(step["layers"]) > layer_index
            ]
            total += _mean([_timing(row, phase, "gpu") for row in rows])
        print(f"{phase:20s} {total:10.2f}")
    prefetch_gpu = _mean([
        sum(_timing(copy, "h2d", "gpu")
            for copy in step.get("prefetch_copies", []))
        for step in steps
    ])
    prefetch_wait_gpu = _mean([
        sum(_timing(copy, "slot_wait", "gpu")
            for copy in step.get("prefetch_copies", []))
        for step in steps
    ])
    print(f"{'prefetch_slot_wait':20s} {prefetch_wait_gpu:10.2f}")
    print(f"{'prefetch_h2d(side)':20s} {prefetch_gpu:10.2f}")
    print("Note: route/repair/expert_gemm are nested inside moe_total; side H2D overlaps.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--show-tree", action="store_true",
                        help="print every valid draft node and its root path")
    args = parser.parse_args()
    configs = json.loads(args.trace.read_text())
    for config in configs:
        summarize(config, show_tree=args.show_tree)


if __name__ == "__main__":
    main()
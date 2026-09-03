"""Opt-in structured execution tracing for end-to-end bottleneck diagnosis."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import platform
import time

import torch


class ExecutionTracer:
    """Collect host timings, CUDA-event timings, tree topology, and layer data.

    CUDA events are resolved only when ``to_dict`` is called. The normal
    inference path does not instantiate this class and pays no tracing cost.
    """

    def __init__(self, detail: str = "full") -> None:
        if detail not in {"full", "progressive"}:
            raise ValueError(f"unknown trace detail: {detail}")
        self.detail = detail
        self.prefills: list[dict] = []
        self.steps: list[dict] = []
        self.current_record: dict | None = None
        self.current_layer: dict | None = None
        self._pending: list[tuple[dict, torch.cuda.Event, torch.cuda.Event]] = []
        self._record_started = 0.0
        self._record_start_event: torch.cuda.Event | None = None
        self.metadata = {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.metadata["gpu"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "multiprocessors": props.multi_processor_count,
            }

    def begin_prefill(self, num_tokens: int, device: torch.device) -> dict:
        record = {
            "kind": "prefill",
            "num_tokens": num_tokens,
            "device": device.type,
            "layers": [],
            "prefetch_copies": [],
        }
        self.prefills.append(record)
        self.current_record = record
        self._start_record(record)
        return record

    def begin_step(self, index: int, root_position: int,
                   device: torch.device) -> dict:
        record = {
            "kind": "verification",
            "index": index,
            "root_position": root_position,
            "device": device.type,
            "layers": [],
            "prefetch_copies": [],
        }
        self.steps.append(record)
        self.current_record = record
        self._start_record(record)
        return record

    def end_record(self) -> None:
        if self.current_record is not None:
            timing = self.current_record.setdefault("timing_ms", {}).setdefault(
                "total", {"host": 0.0, "gpu": 0.0},
            )
            timing["host"] += (time.perf_counter() - self._record_started) * 1e3
            if self._record_start_event is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._pending.append((timing, self._record_start_event, end))
            if self.current_record.get("device") == "cuda" \
                    and torch.cuda.is_available():
                self.current_record["memory_end"] = self._memory_snapshot()
        self.current_record = None
        self.current_layer = None
        self._record_start_event = None

    def _start_record(self, record: dict) -> None:
        self._record_started = time.perf_counter()
        self._record_start_event = None
        if record["device"] == "cuda" and torch.cuda.is_available():
            record["memory_start"] = self._memory_snapshot()
            self._record_start_event = torch.cuda.Event(enable_timing=True)
            self._record_start_event.record()

    @staticmethod
    def _memory_snapshot() -> dict[str, int]:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "free_bytes": free,
            "total_bytes": total,
        }

    def begin_layer(self, layer_index: int) -> dict | None:
        if self.current_record is None:
            return None
        layer = {"layer": layer_index}
        self.current_record["layers"].append(layer)
        self.current_layer = layer
        return layer

    def begin_draft_level(self, depth: int, frontier_nodes: int,
                          cached_nodes: int) -> dict:
        if self.current_record is None or self.detail != "full":
            return {}
        level = {
            "depth": depth,
            "frontier_nodes": frontier_nodes,
            "cached_nodes": cached_nodes,
        }
        self.current_record.setdefault("draft_levels", []).append(level)
        return level

    def add_prefetch_copy(self, layer_index: int, experts: list[int] | None) -> dict:
        if self.current_record is None:
            return {}
        record = {
            "layer": layer_index,
            "experts": experts,
            "rows": len(experts) if experts is not None else None,
        }
        self.current_record["prefetch_copies"].append(record)
        return record

    @contextmanager
    def phase(self, record: dict | None, name: str,
              cuda: bool | None = None):
        if record is None:
            yield
            return
        if self.detail == "progressive":
            yield
            return
        timing = record.setdefault("timing_ms", {}).setdefault(
            name, {"host": 0.0, "gpu": 0.0},
        )
        use_cuda = (
            self.current_record is not None
            and self.current_record.get("device") == "cuda"
            if cuda is None else cuda
        )
        start_event = end_event = None
        if use_cuda and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        try:
            if "layer" in record:
                label = f"layer_{record['layer']:02d}/{name}"
            elif "depth" in record:
                label = f"draft_depth_{record['depth']}/{name}"
            elif record.get("kind") == "verification":
                label = f"step_{record['index']:03d}/{name}"
            else:
                label = f"prefill/{name}"
            with torch.profiler.record_function(label):
                yield
        finally:
            timing["host"] += (time.perf_counter() - started) * 1e3
            if start_event is not None and end_event is not None:
                end_event.record()
                self._pending.append((timing, start_event, end_event))

    def record_tree(self, record: dict, tree) -> None:
        valid = tree.num_valid
        parent = tree.parent[:valid].detach().cpu().tolist()
        tokens = tree.tokens[:valid].detach().cpu().tolist()
        probabilities = tree.accept_prob[:valid].detach().float().cpu().tolist()
        depths = []
        paths = []
        for node in range(valid):
            path = []
            cursor = node
            while cursor >= 0:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            paths.append(path)
            depths.append(len(path) - 1)
        record["tree"] = {
            "num_valid": valid,
            "tokens": tokens,
            "parent": parent,
            "children": tree.children[:valid],
            "accept_probability": probabilities,
            "depth": depths,
            "paths": paths,
            "max_depth": max(depths, default=0),
        }

    def record_acceptance(
        self,
        record: dict,
        accepted_slots: list[int],
        accepted_tokens: list[int],
        bonus_token: int,
    ) -> None:
        probabilities = record["tree"]["accept_probability"]
        record["acceptance"] = {
            "accepted_slots": accepted_slots,
            "accepted_tokens": accepted_tokens,
            "accepted_path_probability": [probabilities[i] for i in accepted_slots],
            "bonus_token": bonus_token,
            "emitted_tokens": len(accepted_tokens) + 1,
        }

    def record_target_decisions(
        self,
        record: dict,
        tree,
        logits: torch.Tensor,
        accepted_slots: list[int],
        top_k: int = 8,
    ) -> None:
        if self.detail != "full":
            return
        valid = tree.num_valid
        scores = logits[:valid].float()
        log_probs = torch.log_softmax(scores, dim=-1)
        top_values, top_tokens = log_probs.topk(top_k, dim=-1)
        parent = tree.parent[:valid]
        tokens = tree.tokens[:valid]
        child_rows = torch.arange(1, valid, device=logits.device)
        predictor_rows = parent[1:valid]
        proposed_tokens = tokens[1:valid]
        proposed_logits = scores[predictor_rows, proposed_tokens]
        proposed_log_probs = log_probs[predictor_rows, proposed_tokens]
        proposed_ranks = (
            scores[predictor_rows] > proposed_logits.unsqueeze(1)
        ).sum(dim=1) + 1
        packed = torch.cat([
            top_tokens.float(), top_values,
            proposed_logits.new_full((valid, 3), float("nan")),
        ], dim=1)
        packed[child_rows, 2 * top_k] = proposed_logits
        packed[child_rows, 2 * top_k + 1] = proposed_log_probs
        packed[child_rows, 2 * top_k + 2] = proposed_ranks.float()
        rows = packed.detach().cpu().tolist()
        tree_tokens = record["tree"]["tokens"]
        tree_parent = record["tree"]["parent"]
        accepted = set(accepted_slots)
        visited_parents = {0, *accepted_slots}

        def decision(node: int) -> str:
            if node == 0:
                return "root"
            parent_node = tree_parent[node]
            if parent_node not in visited_parents:
                return "not_visited"
            if node in accepted:
                return "accepted"
            accepted_child = next(
                (child for child in tree.children[parent_node]
                 if child in accepted),
                None,
            )
            if accepted_child is None:
                return "rejected"
            siblings = tree.children[parent_node]
            if siblings.index(node) < siblings.index(accepted_child):
                return "rejected_before_accept"
            return "not_evaluated_after_accept"

        record["target_nodes"] = [
            {
                "node": node,
                "predicts_children": tree.children[node],
                "target_top_tokens": [int(value) for value in row[:top_k]],
                "target_top_logprob": row[top_k:2 * top_k],
                "target_top_probability": [
                    math.exp(value)
                    for value in row[top_k:2 * top_k]
                ],
                "proposed_token": tree_tokens[node] if node > 0 else None,
                "predicting_parent": tree_parent[node] if node > 0 else None,
                "proposed_target_logit": row[2 * top_k] if node > 0 else None,
                "proposed_target_logprob": row[2 * top_k + 1] if node > 0 else None,
                "proposed_target_probability": (
                    math.exp(row[2 * top_k + 1])
                    if node > 0 else None
                ),
                "proposed_target_rank": (
                    int(row[2 * top_k + 2]) if node > 0 else None
                ),
                "accepted": node in accepted,
                "decision": decision(node),
            }
            for node, row in enumerate(rows)
        ]

    def to_dict(self) -> dict:
        for timing, start, end in self._pending:
            end.synchronize()
            timing["gpu"] += start.elapsed_time(end)
        self._pending.clear()
        return deepcopy({
            "metadata": self.metadata,
            "prefills": self.prefills,
            "steps": self.steps,
        })
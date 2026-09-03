"""Opt-in structured execution tracing for end-to-end bottleneck diagnosis."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import time

import torch


class ExecutionTracer:
    """Collect host timings, CUDA-event timings, tree topology, and layer data.

    CUDA events are resolved only when ``to_dict`` is called. The normal
    inference path does not instantiate this class and pays no tracing cost.
    """

    def __init__(self) -> None:
        self.prefills: list[dict] = []
        self.steps: list[dict] = []
        self.current_record: dict | None = None
        self.current_layer: dict | None = None
        self._pending: list[tuple[dict, torch.cuda.Event, torch.cuda.Event]] = []
        self._record_started = 0.0
        self._record_start_event: torch.cuda.Event | None = None

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
        self.current_record = None
        self.current_layer = None
        self._record_start_event = None

    def _start_record(self, record: dict) -> None:
        self._record_started = time.perf_counter()
        self._record_start_event = None
        if record["device"] == "cuda" and torch.cuda.is_available():
            self._record_start_event = torch.cuda.Event(enable_timing=True)
            self._record_start_event.record()

    def begin_layer(self, layer_index: int) -> dict | None:
        if self.current_record is None:
            return None
        layer = {"layer": layer_index}
        self.current_record["layers"].append(layer)
        self.current_layer = layer
        return layer

    def begin_draft_level(self, depth: int, frontier_nodes: int,
                          cached_nodes: int) -> dict:
        if self.current_record is None:
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

    def to_dict(self) -> dict:
        for timing, start, end in self._pending:
            end.synchronize()
            timing["gpu"] += start.elapsed_time(end)
        self._pending.clear()
        return deepcopy({"prefills": self.prefills, "steps": self.steps})
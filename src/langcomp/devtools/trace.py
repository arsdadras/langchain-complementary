"""
Zero-config local observability & cost tracing for langcomp / LangGraph runs.
Thread-safe capture of latency, token counts, cost estimates, errors, node names, and namespaces.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# Mapping of model name (or "default") -> (input_cost_per_1k_tokens, output_cost_per_1k_tokens)
DEFAULT_PRICE_TABLE: dict[str, tuple[float, float]] = {"default": (0.0, 0.0)}

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _validate_run_id(run_id: str) -> str:
    if not _SAFE_RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError(
            f"Invalid run_id {run_id!r}: must match ^[A-Za-z0-9._-]{{1,128}}$ and not contain '..'."
        )
    return run_id


class TraceCollector(BaseCallbackHandler):
    """Thread-safe local token, cost, latency, error, and subagent namespace tracer.

    Attach to any LangGraph / LangChain / langcomp invocation via::

        config = {"callbacks": [collector]}
    """

    def __init__(
        self,
        run_id: str | None = None,
        price_table: dict[str, tuple[float, float]] | None = None,
        out_dir: str | Path | None = ".langcomp/traces",
        *,
        max_events: int | None = None,
    ):
        self.run_id = _validate_run_id(run_id) if run_id is not None else uuid.uuid4().hex[:12]
        self.price_table = (
            dict(price_table) if price_table is not None else dict(DEFAULT_PRICE_TABLE)
        )
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self._max_events = max_events
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._inflight: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def events(self) -> list[dict[str, Any]]:
        """Snapshot list of collected trace events."""
        with self._lock:
            return list(self._events)

    def _record_start(
        self,
        serialized: dict[str, Any] | None,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        meta = metadata or {}
        inv = invocation_params or {}
        explicit_model = inv.get("model_name") or inv.get("model") or meta.get("ls_model_name")
        fallback_model = (serialized or {}).get("name")
        with self._lock:
            self._inflight[str(run_id)] = {
                "started": time.monotonic(),
                "node": meta.get("langgraph_node"),
                "checkpoint_ns": meta.get("langgraph_checkpoint_ns"),
                "model": explicit_model,
                "fallback_model": fallback_model,
                "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
            }

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del prompts
        self._record_start(
            serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            invocation_params=invocation_params,
            **kwargs,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del messages
        self._record_start(
            serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            invocation_params=invocation_params,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        with self._lock:
            call_info = self._inflight.pop(str(run_id), None) or {}
            started = call_info.get("started", time.monotonic())
            latency = max(0.0, time.monotonic() - started)
            model = call_info.get("model") or call_info.get("fallback_model")
            event: dict[str, Any] = {
                "run_id": self.run_id,
                "call_run_id": str(run_id),
                "parent_run_id": call_info.get("parent_run_id")
                or (str(parent_run_id) if parent_run_id is not None else None),
                "timestamp": time.time(),
                "node": call_info.get("node"),
                "checkpoint_ns": call_info.get("checkpoint_ns"),
                "model": model,
                "latency_s": round(latency, 4),
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            self._events.append(event)
            self._write_locked(event)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        with self._lock:
            call_info = self._inflight.pop(str(run_id), None) or {}

        started = call_info.get("started", time.monotonic())
        node = call_info.get("node")
        checkpoint_ns = call_info.get("checkpoint_ns")
        explicit_model = call_info.get("model")
        fallback_model = call_info.get("fallback_model")
        latency = max(0.0, time.monotonic() - started)

        input_tokens = 0
        output_tokens = 0
        found_usage = False
        response_model: str | None = None

        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                message = getattr(gen, "message", None)
                if message is None:
                    continue
                resp_meta = getattr(message, "response_metadata", None) or {}
                if not response_model and isinstance(resp_meta, dict):
                    response_model = resp_meta.get("model_name") or resp_meta.get("model")
                meta = getattr(message, "usage_metadata", None)
                if isinstance(meta, dict):
                    in_val = _safe_int(meta.get("input_tokens"))
                    out_val = _safe_int(meta.get("output_tokens"))
                    if in_val or out_val or "input_tokens" in meta or "output_tokens" in meta:
                        input_tokens += in_val
                        output_tokens += out_val
                        if in_val or out_val:
                            found_usage = True

        if not found_usage:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
            if isinstance(token_usage, dict):
                input_tokens = _safe_int(
                    token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
                )
                output_tokens = _safe_int(
                    token_usage.get("completion_tokens", token_usage.get("output_tokens"))
                )
            if not response_model and isinstance(llm_output, dict):
                response_model = llm_output.get("model_name") or llm_output.get("model")

        model = response_model or explicit_model or fallback_model
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        event: dict[str, Any] = {
            "run_id": self.run_id,
            "call_run_id": str(run_id),
            "parent_run_id": call_info.get("parent_run_id")
            or (str(parent_run_id) if parent_run_id is not None else None),
            "timestamp": time.time(),
            "node": node,
            "checkpoint_ns": checkpoint_ns,
            "model": model,
            "latency_s": round(latency, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": self._estimate_cost(usage, model=model),
            "status": "ok",
        }
        with self._lock:
            self._events.append(event)
            self._write_locked(event)

    def _lookup_rates(self, model: str | None) -> tuple[float, float]:
        default_rates = self.price_table.get("default", (0.0, 0.0))
        if not model:
            return default_rates
        if model in self.price_table:
            return self.price_table[model]
        # Longest-prefix fallback for dated model IDs (e.g. "gpt-4o-2024-08-06" -> "gpt-4o")
        best_match = ""
        for key in self.price_table:
            if key != "default" and model.startswith(key) and len(key) > len(best_match):
                best_match = key
        if best_match:
            return self.price_table[best_match]
        return default_rates

    def _estimate_cost(self, usage: dict[str, Any], model: str | None = None) -> float:
        in_price, out_price = self._lookup_rates(model)
        cost = (
            _safe_int(usage.get("input_tokens")) / 1000.0 * in_price
            + _safe_int(usage.get("output_tokens")) / 1000.0 * out_price
        )
        return round(cost, 6)

    def _write_locked(self, event: dict[str, Any]) -> None:
        if self.out_dir is None:
            return
        base = self.out_dir.resolve()
        base.mkdir(parents=True, exist_ok=True)
        target = (base / f"{self.run_id}.jsonl").resolve()
        if not target.is_relative_to(base) or target.is_symlink():
            raise ValueError(f"Refusing to write trace file outside {base}: {target}")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def summarize(self) -> dict[str, Any]:
        """Aggregate collected trace events into totals and per-node/per-model breakdowns."""
        return summarize(self.events)


def summarize(
    collector_or_events: TraceCollector | Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize trace events into totals and per-node / per-model breakdowns."""
    events = (
        collector_or_events.events
        if isinstance(collector_or_events, TraceCollector)
        else list(collector_or_events)
    )
    by_node: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    total_latency = 0.0
    total_input = 0
    total_output = 0
    total_cost = 0.0
    error_count = 0

    for event in events:
        latency = float(event.get("latency_s", 0.0) or 0.0)
        in_tok = _safe_int(event.get("input_tokens"))
        out_tok = _safe_int(event.get("output_tokens"))
        cost = float(event.get("cost_usd", 0.0) or 0.0)
        if event.get("status") == "error":
            error_count += 1

        total_latency += latency
        total_input += in_tok
        total_output += out_tok
        total_cost += cost

        node_key = event.get("node") or "<root>"
        node_bucket = by_node.setdefault(
            node_key,
            {"calls": 0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        node_bucket["calls"] += 1
        node_bucket["latency_s"] += latency
        node_bucket["input_tokens"] += in_tok
        node_bucket["output_tokens"] += out_tok
        node_bucket["cost_usd"] += cost

        model_key = event.get("model") or "<unknown>"
        model_bucket = by_model.setdefault(
            model_key,
            {"calls": 0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        model_bucket["calls"] += 1
        model_bucket["latency_s"] += latency
        model_bucket["input_tokens"] += in_tok
        model_bucket["output_tokens"] += out_tok
        model_bucket["cost_usd"] += cost

    for bucket in by_node.values():
        bucket["latency_s"] = round(bucket["latency_s"], 4)
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    for bucket in by_model.values():
        bucket["latency_s"] = round(bucket["latency_s"], 4)
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)

    return {
        "total_calls": len(events),
        "error_calls": error_count,
        "total_latency_s": round(total_latency, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 6),
        "by_node": by_node,
        "by_model": by_model,
    }


__all__ = ["DEFAULT_PRICE_TABLE", "TraceCollector", "summarize"]

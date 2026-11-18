"""
Zero-config local observability & cost tracing for langcomp / LangGraph runs.
Captures latency, token counts, cost estimates, node names, and checkpoint namespaces to JSONL.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain_core.callbacks import BaseCallbackHandler

# Mapping of model name (or "default") -> (input_cost_per_1k_tokens, output_cost_per_1k_tokens)
DEFAULT_PRICE_TABLE: Dict[str, Tuple[float, float]] = {"default": (0.0, 0.0)}


class TraceCollector(BaseCallbackHandler):
    """Local token, cost, latency, and subagent namespace tracer.

    Attach to any LangGraph / LangChain / langcomp invocation via::

        config = {"callbacks": [collector]}
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        price_table: Optional[Dict[str, Tuple[float, float]]] = None,
        out_dir: Optional[Union[str, Path]] = ".langcomp/traces",
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.price_table = dict(price_table) if price_table is not None else dict(DEFAULT_PRICE_TABLE)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self._inflight: Dict[str, Dict[str, Any]] = {}

    def _record_start(
        self,
        serialized: Optional[Dict[str, Any]],
        *,
        run_id: Any,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        inv = invocation_params or {}
        model_name = (
            inv.get("model_name")
            or inv.get("model")
            or meta.get("ls_model_name")
            or (serialized or {}).get("name")
        )
        self._inflight[str(run_id)] = {
            "started": time.monotonic(),
            "node": meta.get("langgraph_node"),
            "checkpoint_ns": meta.get("langgraph_checkpoint_ns"),
            "model": model_name,
        }

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(
            serialized,
            run_id=run_id,
            metadata=metadata,
            invocation_params=invocation_params,
            **kwargs,
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(
            serialized,
            run_id=run_id,
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
        self._inflight.pop(str(run_id), None)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        call_info = self._inflight.pop(str(run_id), None) or {}
        started = call_info.get("started", time.monotonic())
        node = call_info.get("node")
        checkpoint_ns = call_info.get("checkpoint_ns")
        model = call_info.get("model")
        latency = max(0.0, time.monotonic() - started)

        input_tokens = 0
        output_tokens = 0
        found_usage = False

        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                message = getattr(gen, "message", None)
                if message is None:
                    continue
                resp_meta = getattr(message, "response_metadata", None) or {}
                if not model and isinstance(resp_meta, dict):
                    model = resp_meta.get("model_name") or resp_meta.get("model")
                meta = getattr(message, "usage_metadata", None)
                if meta:
                    found_usage = True
                    input_tokens += int(meta.get("input_tokens", 0))
                    output_tokens += int(meta.get("output_tokens", 0))

        if not found_usage:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
            if isinstance(token_usage, dict):
                input_tokens = int(
                    token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
                )
                output_tokens = int(
                    token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
                )

        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        event = {
            "run_id": self.run_id,
            "node": node,
            "checkpoint_ns": checkpoint_ns,
            "model": model,
            "latency_s": round(latency, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": self._estimate_cost(usage, model=model),
        }
        self.events.append(event)
        self._write(event)

    def _estimate_cost(self, usage: Dict[str, Any], model: Optional[str] = None) -> float:
        default_rates = self.price_table.get("default", (0.0, 0.0))
        in_price, out_price = self.price_table.get(model or "", default_rates)
        cost = (
            usage.get("input_tokens", 0) / 1000.0 * in_price
            + usage.get("output_tokens", 0) / 1000.0 * out_price
        )
        return round(cost, 6)

    def _write(self, event: Dict[str, Any]) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / f"{self.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def summarize(self) -> Dict[str, Any]:
        """Aggregate collected trace events into totals and per-node breakdowns."""
        return summarize(self.events)


def summarize(
    collector_or_events: Union[TraceCollector, Iterable[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Summarize trace events into totals and per-node / per-model breakdowns."""
    events = (
        collector_or_events.events
        if isinstance(collector_or_events, TraceCollector)
        else list(collector_or_events)
    )
    by_node: Dict[str, Dict[str, Any]] = {}
    by_model: Dict[str, Dict[str, Any]] = {}
    total_latency = 0.0
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for event in events:
        latency = float(event.get("latency_s", 0.0))
        in_tok = int(event.get("input_tokens", 0))
        out_tok = int(event.get("output_tokens", 0))
        cost = float(event.get("cost_usd", 0.0))

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
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        model_bucket["calls"] += 1
        model_bucket["input_tokens"] += in_tok
        model_bucket["output_tokens"] += out_tok
        model_bucket["cost_usd"] += cost

    for bucket in by_node.values():
        bucket["latency_s"] = round(bucket["latency_s"], 4)
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    for bucket in by_model.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)

    return {
        "total_calls": len(events),
        "total_latency_s": round(total_latency, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 6),
        "by_node": by_node,
        "by_model": by_model,
    }

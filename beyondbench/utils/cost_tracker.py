"""
Cost tracking for LLM evaluations.

Provides per-provider 2025 pricing, cost-per-task breakdowns, and
pre-evaluation cost estimates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# 2025 pricing table  ($ per 1K tokens)
# ---------------------------------------------------------------------------
# Source: public pricing pages, April 2025.  Update as needed.
PROVIDER_PRICING: Dict[str, Dict[str, Dict[str, float]]] = {
    # ---- OpenAI ----
    "openai": {
        "gpt-4o":                {"input": 0.0025,  "output": 0.010},
        "gpt-4o-mini":           {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo":           {"input": 0.010,   "output": 0.030},
        "gpt-4":                 {"input": 0.030,   "output": 0.060},
        "gpt-3.5-turbo":        {"input": 0.0005,  "output": 0.0015},
        "gpt-5":                 {"input": 0.002,   "output": 0.008},
        "gpt-5-mini":            {"input": 0.0004,  "output": 0.0016},
        "o1":                    {"input": 0.015,   "output": 0.060},
        "o1-mini":               {"input": 0.003,   "output": 0.012},
        "o3":                    {"input": 0.010,   "output": 0.040},
        "o3-mini":               {"input": 0.0011,  "output": 0.0044},
        "o4-mini":               {"input": 0.0011,  "output": 0.0044},
        "_default":              {"input": 0.002,   "output": 0.008},
    },
    # ---- Anthropic ----
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 0.003,  "output": 0.015},
        "claude-3-5-haiku-20241022":  {"input": 0.0008, "output": 0.004},
        "claude-sonnet-4-20250514":   {"input": 0.003,  "output": 0.015},
        "claude-opus-4-20250514":     {"input": 0.015,  "output": 0.075},
        "claude-3-opus-20240229":     {"input": 0.015,  "output": 0.075},
        "_default":                   {"input": 0.003,  "output": 0.015},
    },
    # ---- Google Gemini ----
    "gemini": {
        "gemini-2.5-pro":          {"input": 0.00125, "output": 0.010},
        "gemini-2.5-flash":        {"input": 0.000075,"output": 0.0003},
        "gemini-2.5-flash-lite":   {"input": 0.000038,"output": 0.00015},
        "gemini-1.5-pro":          {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash":        {"input": 0.000075,"output": 0.0003},
        "_default":                {"input": 0.00125, "output": 0.005},
    },
    # ---- Local (free, $0) ----
    "local": {
        "_default": {"input": 0.0, "output": 0.0},
    },
}


@dataclass
class TaskCost:
    """Cost breakdown for a single task."""
    task_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    request_count: int = 0
    cost_per_correct: float = 0.0  # filled in externally when accuracy is known


@dataclass
class CostReport:
    """Full cost report for an evaluation run."""
    provider: str = ""
    model_id: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cost: float = 0.0
    per_task: Dict[str, TaskCost] = field(default_factory=dict)
    pricing_per_1k: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "input_cost_usd": round(self.total_input_cost, 6),
                "output_cost_usd": round(self.total_output_cost, 6),
                "total_cost_usd": round(self.total_cost, 6),
            },
            "pricing_per_1k_tokens": self.pricing_per_1k,
            "per_task": {
                name: {
                    "input_tokens": tc.input_tokens,
                    "output_tokens": tc.output_tokens,
                    "input_cost_usd": round(tc.input_cost, 6),
                    "output_cost_usd": round(tc.output_cost, 6),
                    "total_cost_usd": round(tc.total_cost, 6),
                    "request_count": tc.request_count,
                    "cost_per_correct_usd": round(tc.cost_per_correct, 6) if tc.cost_per_correct else None,
                }
                for name, tc in self.per_task.items()
            },
        }


class CostTracker:
    """
    Track inference costs per task and produce cost reports.

    Pricing is looked up from PROVIDER_PRICING (2025 rates).
    Custom pricing can be supplied via the constructor.

    Example::

        tracker = CostTracker(provider="openai", model_id="gpt-4o")
        print(tracker.estimate("Will use ~10k input, ~5k output tokens"))
        tracker.record(task_name="sum", input_tokens=120, output_tokens=30)
        report = tracker.get_report()
    """

    def __init__(
        self,
        provider: str = "local",
        model_id: str = "",
        custom_input_per_1k: Optional[float] = None,
        custom_output_per_1k: Optional[float] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.provider = provider.lower()
        self.model_id = model_id

        # Resolve pricing
        provider_prices = PROVIDER_PRICING.get(self.provider, PROVIDER_PRICING["local"])
        # Try exact model match, then prefix match, then _default
        prices = self._resolve_model_price(provider_prices, model_id)

        self._input_per_1k: float = custom_input_per_1k if custom_input_per_1k is not None else prices["input"]
        self._output_per_1k: float = custom_output_per_1k if custom_output_per_1k is not None else prices["output"]

        if self._input_per_1k > 0 or self._output_per_1k > 0:
            self.logger.info(
                f"Cost tracker: provider={self.provider}, model={model_id}, "
                f"input=${self._input_per_1k}/1K, output=${self._output_per_1k}/1K"
            )
        else:
            self.logger.debug("Cost tracker: local model — no cost tracking.")

        # Accumulators
        self._per_task: Dict[str, TaskCost] = {}

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_price(
        provider_prices: Dict[str, Dict[str, float]], model_id: str
    ) -> Dict[str, float]:
        """Match model_id to pricing entry: exact → prefix → _default."""
        if model_id in provider_prices:
            return provider_prices[model_id]
        # Prefix match (e.g. "claude-3-5" matches "claude-3-5-sonnet-...")
        for key, prices in provider_prices.items():
            if key != "_default" and model_id.startswith(key):
                return prices
        return provider_prices.get("_default", {"input": 0.0, "output": 0.0})

    def get_pricing(self) -> Dict[str, float]:
        return {"input_per_1k": self._input_per_1k, "output_per_1k": self._output_per_1k}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        task_name: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record token usage for a single request."""
        input_cost = (input_tokens / 1000.0) * self._input_per_1k
        output_cost = (output_tokens / 1000.0) * self._output_per_1k

        if task_name not in self._per_task:
            self._per_task[task_name] = TaskCost(task_name=task_name)

        tc = self._per_task[task_name]
        tc.input_tokens += input_tokens
        tc.output_tokens += output_tokens
        tc.input_cost += input_cost
        tc.output_cost += output_cost
        tc.total_cost += input_cost + output_cost
        tc.request_count += 1

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Quick one-shot cost estimate for a (model, tokens) combination.

        Looks up pricing across all provider pricing tables (exact then prefix
        match) and returns the USD cost as a float. Returns 0.0 for local / unknown
        models with no pricing data.

        Example::

            ct = CostTracker()
            cost = ct.estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        """
        if not model:
            return 0.0

        # Search every provider for a matching model
        for provider, prices in PROVIDER_PRICING.items():
            if provider == "local":
                continue
            if model in prices:
                p = prices[model]
                return (input_tokens / 1000.0) * p["input"] + (output_tokens / 1000.0) * p["output"]
        # Prefix match second pass
        for provider, prices in PROVIDER_PRICING.items():
            if provider == "local":
                continue
            for key, p in prices.items():
                if key == "_default":
                    continue
                if model.startswith(key):
                    return (input_tokens / 1000.0) * p["input"] + (output_tokens / 1000.0) * p["output"]
        # Fall back to instance pricing if nothing else matched
        return (input_tokens / 1000.0) * self._input_per_1k + (output_tokens / 1000.0) * self._output_per_1k

    def estimate(
        self,
        avg_input_tokens: int,
        avg_output_tokens: int,
        n_tasks: int,
        datapoints_per_task: int,
    ) -> Dict[str, Any]:
        """
        Produce a pre-evaluation cost estimate.

        Returns a dict with total_tokens, estimated_cost, and a human-readable message.
        """
        total_requests = n_tasks * datapoints_per_task
        total_input = avg_input_tokens * total_requests
        total_output = avg_output_tokens * total_requests
        total_cost = (
            (total_input / 1000.0) * self._input_per_1k
            + (total_output / 1000.0) * self._output_per_1k
        )

        msg = (
            f"Estimated cost: ${total_cost:.4f} USD "
            f"({total_requests} requests × "
            f"~{avg_input_tokens} input + ~{avg_output_tokens} output tokens)"
        )
        if self._input_per_1k == 0 and self._output_per_1k == 0:
            msg = "Local model — no API cost."

        return {
            "total_requests": total_requests,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(total_cost, 6),
            "message": msg,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def attach_accuracy(self, task_accuracy: Dict[str, float]):
        """
        Attach accuracy info so cost_per_correct can be computed.

        Args:
            task_accuracy: {task_name: accuracy_float_0_to_1}
        """
        for task_name, acc in task_accuracy.items():
            if task_name in self._per_task:
                tc = self._per_task[task_name]
                correct_fraction = max(acc, 1e-9)
                # cost per correct answer = total_cost / (request_count * accuracy)
                n_correct = tc.request_count * correct_fraction
                tc.cost_per_correct = tc.total_cost / max(n_correct, 1e-9)

    def get_report(self) -> CostReport:
        """Return a CostReport with all accumulated data."""
        report = CostReport(
            provider=self.provider,
            model_id=self.model_id,
            per_task=self._per_task,
            pricing_per_1k={"input": self._input_per_1k, "output": self._output_per_1k},
        )
        for tc in self._per_task.values():
            report.total_input_tokens += tc.input_tokens
            report.total_output_tokens += tc.output_tokens
            report.total_input_cost += tc.input_cost
            report.total_output_cost += tc.output_cost
            report.total_cost += tc.total_cost
        return report

    @staticmethod
    def list_providers() -> List[str]:
        """Return all providers with pricing data."""
        return [p for p in PROVIDER_PRICING if p != "local"]

    @staticmethod
    def list_models(provider: str) -> List[str]:
        """Return all models with pricing for a given provider."""
        prices = PROVIDER_PRICING.get(provider.lower(), {})
        return [k for k in prices if k != "_default"]

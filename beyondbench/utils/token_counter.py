"""Token counting utilities with model-specific tokenizer support and per-request tracking."""

import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """Per-request token usage record."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    task_name: str = ""
    fold: int = 0
    timestamp: float = field(default_factory=time.time)
    counting_method: str = "unknown"  # tiktoken | model_tokenizer | estimate


@dataclass
class TokenAnalytics:
    """Aggregated token analytics for a set of requests."""
    count: int = 0
    # Input token stats
    input_avg: float = 0.0
    input_min: int = 0
    input_max: int = 0
    input_p50: float = 0.0
    input_p95: float = 0.0
    # Output token stats
    output_avg: float = 0.0
    output_min: int = 0
    output_max: int = 0
    output_p50: float = 0.0
    output_p95: float = 0.0
    # Total
    total_input: int = 0
    total_output: int = 0
    # Throughput
    avg_tokens_per_second: float = 0.0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "input": {
                "avg": round(self.input_avg, 1),
                "min": self.input_min,
                "max": self.input_max,
                "p50": round(self.input_p50, 1),
                "p95": round(self.input_p95, 1),
                "total": self.total_input,
            },
            "output": {
                "avg": round(self.output_avg, 1),
                "min": self.output_min,
                "max": self.output_max,
                "p50": round(self.output_p50, 1),
                "p95": round(self.output_p95, 1),
                "total": self.total_output,
            },
            "throughput": {
                "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
            },
        }


class TokenCounter:
    """
    Unified token counting with multiple backend support and per-request tracking.

    Counting priority:
      1. Model-specific tokenizer (HuggingFace AutoTokenizer for local models)
      2. tiktoken cl100k_base (good proxy for most GPT-style models)
      3. Word-based estimation (fallback)

    Usage warning is emitted at most once per counter instance per method
    to avoid log spam.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model_id = model_id

        # Tokenizer state
        self.tiktoken_encoder = None
        self.model_tokenizer = None
        self._warned_tiktoken = False
        self._warned_model_tok = False
        self._warned_fallback = False

        # Per-request log
        self._request_log: List[TokenUsage] = []

        self._init_tiktoken()
        if model_id:
            self._init_model_tokenizer(model_id)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_tiktoken(self):
        try:
            import tiktoken
            self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            self.logger.debug("Initialized tiktoken cl100k_base encoder")
        except ImportError:
            self.logger.debug("tiktoken not available")
        except Exception as e:
            self.logger.debug(f"tiktoken init failed: {e}")

    def _init_model_tokenizer(self, model_id: str):
        try:
            from transformers import AutoTokenizer
            self.model_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )
            self.logger.info(f"Loaded model tokenizer for {model_id}")
        except Exception as e:
            self.logger.debug(f"Could not load model tokenizer for {model_id}: {e}")

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text using the best available method."""
        if not text:
            return 0

        # 1. Model-specific tokenizer
        if self.model_tokenizer is not None:
            try:
                return len(self.model_tokenizer.encode(text, add_special_tokens=False))
            except Exception as e:
                if not self._warned_model_tok:
                    self.logger.warning(f"Model tokenizer encoding failed (will use tiktoken): {e}")
                    self._warned_model_tok = True

        # 2. tiktoken
        if self.tiktoken_encoder is not None:
            try:
                return len(self.tiktoken_encoder.encode(text))
            except Exception as e:
                if not self._warned_tiktoken:
                    self.logger.warning(f"tiktoken encoding failed (will use estimation): {e}")
                    self._warned_tiktoken = True

        # 3. Estimation fallback
        if not self._warned_fallback:
            self.logger.warning(
                "No tokenizer available — using word-based token estimation. "
                "Install tiktoken for better accuracy: pip install tiktoken"
            )
            self._warned_fallback = True
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Rough word-based token estimation (≈ 0.75 words per token)."""
        return max(1, int(len(text.split()) * 1.3))

    def _counting_method(self) -> str:
        if self.model_tokenizer is not None:
            return "model_tokenizer"
        if self.tiktoken_encoder is not None:
            return "tiktoken"
        return "estimate"

    # ------------------------------------------------------------------
    # Per-request tracking
    # ------------------------------------------------------------------

    def record_request(
        self,
        prompt: str,
        response: str,
        latency_ms: float = 0.0,
        task_name: str = "",
        fold: int = 0,
    ) -> TokenUsage:
        """
        Record a single model request, return TokenUsage.

        Args:
            prompt: The full prompt sent to the model.
            response: The model's response text.
            latency_ms: Observed latency in milliseconds.
            task_name: Name of the task being evaluated.
            fold: Current fold index.

        Returns:
            TokenUsage with filled-in token counts and throughput.
        """
        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(response)
        total_tokens = input_tokens + output_tokens

        tps = 0.0
        if latency_ms > 0:
            tps = output_tokens / (latency_ms / 1000.0)

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tps,
            latency_ms=latency_ms,
            task_name=task_name,
            fold=fold,
            timestamp=time.time(),
            counting_method=self._counting_method(),
        )
        self._request_log.append(usage)
        return usage

    def get_request_log(self) -> List[TokenUsage]:
        """Return all recorded token usage records."""
        return list(self._request_log)

    def clear_log(self):
        """Clear the per-request log."""
        self._request_log.clear()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def compute_analytics(
        self, usages: Optional[List[TokenUsage]] = None
    ) -> TokenAnalytics:
        """
        Compute aggregate analytics from a list of TokenUsage records.

        If usages is None, uses the internal request log.
        """
        records = usages if usages is not None else self._request_log
        if not records:
            return TokenAnalytics()

        inputs = np.array([r.input_tokens for r in records], dtype=float)
        outputs = np.array([r.output_tokens for r in records], dtype=float)
        tps = np.array([r.tokens_per_second for r in records], dtype=float)
        latencies = np.array([r.latency_ms for r in records], dtype=float)

        return TokenAnalytics(
            count=len(records),
            input_avg=float(np.mean(inputs)),
            input_min=int(np.min(inputs)),
            input_max=int(np.max(inputs)),
            input_p50=float(np.percentile(inputs, 50)),
            input_p95=float(np.percentile(inputs, 95)),
            output_avg=float(np.mean(outputs)),
            output_min=int(np.min(outputs)),
            output_max=int(np.max(outputs)),
            output_p50=float(np.percentile(outputs, 50)),
            output_p95=float(np.percentile(outputs, 95)),
            total_input=int(np.sum(inputs)),
            total_output=int(np.sum(outputs)),
            avg_tokens_per_second=float(np.mean(tps[tps > 0])) if np.any(tps > 0) else 0.0,
            avg_latency_ms=float(np.mean(latencies[latencies > 0])) if np.any(latencies > 0) else 0.0,
        )

    def compute_task_analytics(self) -> Dict[str, TokenAnalytics]:
        """Return per-task analytics from the internal request log."""
        from collections import defaultdict
        by_task: Dict[str, List[TokenUsage]] = defaultdict(list)
        for r in self._request_log:
            by_task[r.task_name].append(r)
        return {task: self.compute_analytics(usages) for task, usages in by_task.items()}

    def compute_difficulty_analytics(
        self, task_difficulty_map: Dict[str, str]
    ) -> Dict[str, TokenAnalytics]:
        """
        Return per-difficulty analytics.

        Args:
            task_difficulty_map: {task_name: difficulty_label} e.g. {'sum': 'easy'}
        """
        from collections import defaultdict
        by_diff: Dict[str, List[TokenUsage]] = defaultdict(list)
        for r in self._request_log:
            diff = task_difficulty_map.get(r.task_name, "unknown")
            by_diff[diff].append(r)
        return {diff: self.compute_analytics(usages) for diff, usages in by_diff.items()}

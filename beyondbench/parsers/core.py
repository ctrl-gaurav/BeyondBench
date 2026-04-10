"""
UnifiedParser — central parsing engine for BeyondBench.

Replaces the fragmented per-task parsers with a single, configurable pipeline.
Target: <1ms per parse for all strategies combined.

Usage:
    from beyondbench.parsers.core import UnifiedParser
    from beyondbench.parsers.task_configs import get_task_config

    config = get_task_config("sum")
    parser = UnifiedParser(config)
    result = parser.parse(response, model_id="Qwen/Qwen2.5-1.5B-Instruct")
    # result.value, result.confidence, result.strategy_used
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Any, Dict, List

from .strategies.boxed_strategy import ParseResult, BoxedStrategy
from .strategies.explicit_statement_strategy import ExplicitStatementStrategy
from .strategies.code_block_strategy import CodeBlockStrategy
from .strategies.latex_math_strategy import LatexMathStrategy
from .strategies.json_strategy import JsonStrategy
from .strategies.list_strategy import ListStrategy
from .strategies.grid_strategy import GridStrategy
from .strategies.comparison_strategy import ComparisonStrategy
from .strategies.sequence_strategy import SequenceStrategy
from .strategies.fallback_strategy import FallbackStrategy
from .model_adapters import get_adapter, BaseAdapter
from .task_configs import ParserConfig, get_task_config

logger = logging.getLogger(__name__)

# ============================================================================
# Strategy registry
# ============================================================================

_STRATEGY_INSTANCES: Dict[str, Any] = {
    "boxed":              BoxedStrategy(),
    "explicit_statement": ExplicitStatementStrategy(),
    "code_block":         CodeBlockStrategy(),
    "latex_math":         LatexMathStrategy(),
    "json":               JsonStrategy(),
    "list":               ListStrategy(),
    "grid":               GridStrategy(),
    "comparison":         ComparisonStrategy(),
    "sequence":           SequenceStrategy(),
    "fallback":           FallbackStrategy(),
}

# ============================================================================
# Type conversion helpers
# ============================================================================

def _to_int(value: str) -> Optional[int]:
    try:
        return int(float(value.replace(',', '')))
    except (ValueError, TypeError, AttributeError):
        return None


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(',', ''))
    except (ValueError, TypeError, AttributeError):
        return None


def _to_list(value: str) -> Optional[List]:
    import ast
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    # Try comma-separated
    parts = [p.strip() for p in value.split(',') if p.strip()]
    result = []
    for p in parts:
        try:
            f = float(p)
            result.append(int(f) if f == int(f) else f)
        except ValueError:
            result.append(p)
    return result if result else None


def _convert(value: str, expected_type: str) -> Any:
    """Convert extracted string value to the expected type."""
    if value is None:
        return None

    if expected_type == "int":
        return _to_int(value)
    elif expected_type == "float":
        result = _to_float(value)
        if result is None:
            # Try int conversion as fallback
            return _to_int(value)
        return result
    elif expected_type == "list":
        return _to_list(value)
    elif expected_type == "grid":
        return _to_list(value)  # Grid is list-of-lists; caller validates structure
    elif expected_type == "comparison":
        # Already normalized by ComparisonStrategy
        return value
    elif expected_type == "boolean":
        low = value.lower().strip()
        if low in ("true", "yes", "1", "satisfiable", "sat"):
            return True
        if low in ("false", "no", "0", "unsatisfiable", "unsat"):
            return False
        return None
    elif expected_type == "auto":
        # Try int → float → string
        vi = _to_int(value)
        if vi is not None and str(vi) == value.replace(',', '').strip():
            return vi
        vf = _to_float(value)
        if vf is not None:
            return vf
        return value
    else:
        return value


# ============================================================================
# UnifiedParser
# ============================================================================

class UnifiedParser:
    """
    Unified, configurable parser for all BeyondBench tasks.

    Runs a configurable strategy pipeline in priority order.
    Returns the highest-confidence ParseResult among those that succeeded.

    Design goals:
    - <1ms per parse for typical responses
    - Independently testable strategies
    - Backwards compatible: can fall back to legacy parsers
    - Model-adaptive: per-model artifact stripping
    """

    def __init__(self, config: Optional[ParserConfig] = None, task_name: Optional[str] = None):
        """
        Initialize the parser.

        Args:
            config: ParserConfig specifying strategies and expected_type.
                    If None, task_name is used to look up config.
            task_name: Task name for auto-config lookup (used if config is None).
        """
        if config is None:
            if task_name:
                config = get_task_config(task_name)
            else:
                config = get_task_config("unknown")

        self.config = config

    def parse(
        self,
        response: str,
        model_id: Optional[str] = None,
        adapter: Optional[BaseAdapter] = None,
    ) -> ParseResult:
        """
        Parse a model response using the configured strategy pipeline.

        Args:
            response: Raw model response text.
            model_id: Model identifier for auto-selecting the right adapter.
            adapter: Pre-constructed adapter (overrides model_id).

        Returns:
            ParseResult with the best extracted value, or a failed result.
        """
        if not response:
            return ParseResult(value=None, confidence=0.0, strategy_used="none")

        # 1. Apply model-specific normalization
        if adapter is None:
            adapter = get_adapter(model_id)
        text = adapter.normalize(response)

        # 2. Basic pre-processing — normalize newline escapes, collapse whitespace
        text = text.replace('\\n', '\n').replace('\\t', ' ')
        text = re.sub(r'\r\n', '\n', text)

        # 3. Run strategies in priority order
        expected_type = self.config.expected_type
        results: List[ParseResult] = []

        for strategy_name in self.config.strategies:
            strategy = _STRATEGY_INSTANCES.get(strategy_name)
            if strategy is None:
                logger.debug("Unknown strategy: %s", strategy_name)
                continue

            try:
                # Strategies accept optional expected_type hint
                extract_fn = strategy.extract
                import inspect
                sig = inspect.signature(extract_fn)
                if 'expected_type' in sig.parameters:
                    result = extract_fn(text, expected_type=expected_type)
                else:
                    result = extract_fn(text)
            except Exception as exc:
                logger.debug("Strategy %s raised %s: %s", strategy_name, type(exc).__name__, exc)
                continue

            if result.success:
                results.append(result)
                # Early exit if high-confidence match found
                if result.confidence >= 0.90:
                    break

        if not results:
            return ParseResult(value=None, confidence=0.0, strategy_used="none")

        # 4. Try candidates in confidence order; use first that converts successfully
        results_sorted = sorted(results, key=lambda r: r.confidence, reverse=True)
        best = None
        converted_value = None

        for candidate in results_sorted:
            cv = _convert(candidate.value, expected_type)
            if cv is not None:
                best = candidate
                converted_value = cv
                break

        if best is None:
            # All conversions failed — return highest-confidence raw result
            best = results_sorted[0]
            converted_value = best.value  # keep as string

        # 5. Apply post-processors
        for proc in self.config.post_processors:
            try:
                converted_value = proc(converted_value)
            except Exception as exc:
                logger.debug("Post-processor failed: %s", exc)

        return ParseResult(
            value=converted_value,
            confidence=best.confidence,
            strategy_used=best.strategy_used,
            raw_match=best.raw_match,
        )

    def parse_with_legacy_fallback(
        self,
        response: str,
        legacy_fn,
        model_id: Optional[str] = None,
        log_disagreement: bool = True,
    ):
        """
        Parse using unified parser; fall back to legacy parser on failure.
        Logs when the two parsers disagree (useful for validation).

        Args:
            response: Raw model response text.
            legacy_fn: Callable that takes response and returns (instruction_followed, answer)
                       or just answer.
            model_id: Model identifier for adapter selection.
            log_disagreement: Whether to log when unified and legacy disagree.

        Returns:
            Tuple of (instruction_followed: bool, answer: Any).
        """
        unified_result = self.parse(response, model_id=model_id)

        # Run legacy parser
        try:
            legacy_output = legacy_fn(response)
            if isinstance(legacy_output, tuple):
                legacy_followed, legacy_answer = legacy_output
            else:
                legacy_followed = legacy_output is not None
                legacy_answer = legacy_output
        except Exception as exc:
            logger.debug("Legacy parser failed: %s", exc)
            legacy_answer = None
            legacy_followed = False

        # Use unified result if available
        if unified_result.success:
            instruction_followed = unified_result.confidence >= 0.90
            unified_answer = unified_result.value

            if log_disagreement and legacy_answer is not None and unified_answer != legacy_answer:
                logger.debug(
                    "Parser disagreement | unified=%r (%.2f via %s) | legacy=%r",
                    unified_answer,
                    unified_result.confidence,
                    unified_result.strategy_used,
                    legacy_answer,
                )

            return instruction_followed, unified_answer

        # Fall back to legacy
        if legacy_answer is not None:
            return legacy_followed, legacy_answer

        return False, None


# ============================================================================
# Convenience functions
# ============================================================================

def parse_response(
    response: str,
    task_name: str,
    model_id: Optional[str] = None,
) -> ParseResult:
    """
    One-shot convenience function: parse a response for a given task.

    Args:
        response: Raw model response text.
        task_name: Task name for config lookup.
        model_id: Model identifier for adapter selection.

    Returns:
        ParseResult.
    """
    parser = UnifiedParser(task_name=task_name)
    return parser.parse(response, model_id=model_id)

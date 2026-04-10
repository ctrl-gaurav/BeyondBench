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

# Model profile type (imported lazily to avoid circular imports at module load)
_ModelProfile = None  # set on first use via _get_profile_type()

logger = logging.getLogger(__name__)

# ============================================================================
# Parse confidence threshold (Phase 4.2.2)
# ============================================================================

#: Minimum confidence for a parse to be considered successful.
#: Results below this threshold are logged as "low confidence".
CONFIDENCE_THRESHOLD_LOW = 0.3

#: Confidence below which results are logged as "unparseable" (no strategy
#: returned a result above this floor).
CONFIDENCE_THRESHOLD_UNPARSEABLE = 0.0

# Per-model unparseable tracking {model_id: [total, unparseable_count]}
_unparseable_stats: Dict[str, List[int]] = {}


def get_unparseable_rate(model_id: str) -> float:
    """Return the fraction of parse calls that produced no result for *model_id*."""
    stats = _unparseable_stats.get(model_id)
    if not stats or stats[0] == 0:
        return 0.0
    return stats[1] / stats[0]


def reset_unparseable_stats() -> None:
    """Clear accumulated unparseable statistics (useful between evaluations)."""
    _unparseable_stats.clear()


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

# Pre-compute which strategies accept expected_type to avoid inspect overhead
import inspect as _inspect
_STRATEGY_ACCEPTS_EXPECTED_TYPE: Dict[str, bool] = {}
for _name, _strat in _STRATEGY_INSTANCES.items():
    _sig = _inspect.signature(_strat.extract)
    _STRATEGY_ACCEPTS_EXPECTED_TYPE[_name] = 'expected_type' in _sig.parameters

# ============================================================================
# Type conversion helpers
# ============================================================================

def _to_int(value: str) -> Optional[int]:
    try:
        cleaned = str(value).strip().replace(',', '')
        # Handle "42.0" → 42
        f = float(cleaned)
        if f != f:  # NaN check
            return None
        return int(f)
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def _to_float(value: str) -> Optional[float]:
    try:
        cleaned = str(value).strip().replace(',', '')
        f = float(cleaned)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def _to_list(value: str) -> Optional[List]:
    import ast
    if value is None:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None

    # Already a list (e.g. from grid strategy returning str(list))
    try:
        parsed = ast.literal_eval(value_str)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    except (ValueError, SyntaxError):
        pass

    # Try JSON parsing for "[1,2,3]" style
    import json
    try:
        parsed = json.loads(value_str)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try comma-separated
    parts = [p.strip() for p in value_str.split(',') if p.strip()]
    if len(parts) <= 1 and ',' not in value_str:
        # Single value is not a list
        return None
    result = []
    for p in parts:
        try:
            f = float(p)
            result.append(int(f) if f == int(f) else f)
        except ValueError:
            result.append(p)
    return result if result else None


def _convert(value: Any, expected_type: str) -> Any:
    """Convert extracted string value to the expected type."""
    if value is None:
        return None

    # If already the right type, return directly
    if expected_type == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected_type == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if expected_type == "list" and isinstance(value, list):
        return value

    value_str = str(value)

    if expected_type == "int":
        return _to_int(value_str)
    elif expected_type == "float":
        result = _to_float(value_str)
        if result is None:
            return _to_int(value_str)
        return result
    elif expected_type == "list":
        return _to_list(value_str)
    elif expected_type == "grid":
        return _to_list(value_str)  # Grid is list-of-lists; caller validates structure
    elif expected_type == "comparison":
        # Normalize comparison values (may come from boxed/explicit strategies as raw symbols)
        from .strategies.comparison_strategy import _NORMALIZE_MAP
        cleaned = value_str.strip().lower().rstrip('.')
        if cleaned in _NORMALIZE_MAP:
            return _NORMALIZE_MAP[cleaned]
        # Check for multi-word matches
        for phrase in ("greater than", "less than", "equal to"):
            if phrase in cleaned:
                return phrase
        return value_str.strip() if isinstance(value, str) else value
    elif expected_type == "boolean":
        low = value_str.lower().strip()
        if low in ("true", "yes", "1", "satisfiable", "sat"):
            return True
        if low in ("false", "no", "0", "unsatisfiable", "unsat"):
            return False
        return None
    elif expected_type == "str":
        return value_str.strip()
    elif expected_type == "auto":
        # Try int → float → string
        vi = _to_int(value_str)
        if vi is not None and str(vi) == value_str.replace(',', '').strip():
            return vi
        vf = _to_float(value_str)
        if vf is not None:
            return vf
        return value_str.strip() if isinstance(value, str) else value
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

    # ------------------------------------------------------------------ #
    # Phase 4 helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_profile(model_id: Optional[str]) -> Optional[Any]:
        """
        Load a ModelProfile for *model_id*, checking built-in profiles first,
        then the user's ~/.beyondbench/profiles/ cache.

        Returns None if no profile is available.
        """
        if not model_id:
            return None

        # 1. Try pre-built profiles (no I/O overhead for known models)
        try:
            from ..utils.model_profiles import load_builtin_profile
            data = load_builtin_profile(model_id)
            if data is not None:
                from ..utils.model_profiler import ModelProfile
                return ModelProfile(**data)
        except Exception as exc:
            logger.debug("Built-in profile lookup failed: %s", exc)

        # 2. Try user cache
        try:
            from ..utils.model_profiler import ModelProfiler
            return ModelProfiler.load(model_id)
        except Exception as exc:
            logger.debug("Cached profile lookup failed: %s", exc)

        return None

    @staticmethod
    def _apply_profile_strategies(
        base_strategies: List[str],
        profile: Any,
    ) -> List[str]:
        """
        Reorder *base_strategies* according to a ModelProfile's recommendations.

        The profile's recommended_strategies list only influences relative order
        among strategies that are already present in *base_strategies*.  Any
        strategy not in the profile's list keeps its original relative position.
        """
        if profile is None or not profile.recommended_strategies:
            return base_strategies

        # Build a priority map from the profile (lower index = higher priority)
        priority: Dict[str, int] = {s: i for i, s in enumerate(profile.recommended_strategies)}

        # Assign priorities: profile-known strategies get profile priority,
        # unknown strategies go after them (preserving their original order).
        base_len = len(base_strategies)
        profile_len = len(profile.recommended_strategies)

        def _sort_key(strategy: str) -> int:
            return priority.get(strategy, profile_len + base_strategies.index(strategy))

        return sorted(base_strategies, key=_sort_key)

    def parse(
        self,
        response: str,
        model_id: Optional[str] = None,
        adapter: Optional[BaseAdapter] = None,
        model_profile: Optional[Any] = None,
    ) -> ParseResult:
        """
        Parse a model response using the configured strategy pipeline.

        Args:
            response: Raw model response text.
            model_id: Model identifier for auto-selecting the right adapter.
            adapter: Pre-constructed adapter (overrides model_id).
            model_profile: Optional ModelProfile for strategy reordering.
                           If None and model_id is provided, a profile is
                           automatically loaded from built-in data or cache.

        Returns:
            ParseResult with the best extracted value, or a failed result.
        """
        if not response or not response.strip():
            # Track unparseable for confidence stats
            if model_id:
                stats = _unparseable_stats.setdefault(model_id, [0, 0])
                stats[0] += 1
                stats[1] += 1
            return ParseResult(value=None, confidence=0.0, strategy_used="none")

        # 1. Apply model-specific normalization
        if adapter is None:
            adapter = get_adapter(model_id)
        text = adapter.normalize(response)

        # 2. Basic pre-processing — normalize newline escapes, collapse whitespace
        text = text.replace('\\n', '\n').replace('\\t', ' ')
        text = re.sub(r'\r\n', '\n', text)

        # 3. (Phase 4) Resolve model profile and reorder strategies
        if model_profile is None and model_id:
            model_profile = self._load_profile(model_id)

        strategies = self._apply_profile_strategies(self.config.strategies, model_profile)

        # 4. Run strategies in priority order (profile-adjusted)
        expected_type = self.config.expected_type
        results: List[ParseResult] = []

        for strategy_name in strategies:
            strategy = _STRATEGY_INSTANCES.get(strategy_name)
            if strategy is None:
                logger.debug("Unknown strategy: %s", strategy_name)
                continue

            try:
                if _STRATEGY_ACCEPTS_EXPECTED_TYPE.get(strategy_name, False):
                    result = strategy.extract(text, expected_type=expected_type)
                else:
                    result = strategy.extract(text)
            except Exception as exc:
                logger.debug("Strategy %s raised %s: %s", strategy_name, type(exc).__name__, exc)
                continue

            if result.success:
                results.append(result)
                # Early exit if high-confidence match found
                if result.confidence >= 0.90:
                    break

        # Track per-model call count
        if model_id:
            stats = _unparseable_stats.setdefault(model_id, [0, 0])
            stats[0] += 1

        if not results:
            if model_id:
                _unparseable_stats[model_id][1] += 1
            logger.debug(
                "Unparseable response (no strategy matched) for model=%s task_type=%s",
                model_id, expected_type,
            )
            return ParseResult(value=None, confidence=0.0, strategy_used="none")

        # 5. Filter out low-confidence results (Phase 4.2.2)
        viable = [r for r in results if r.confidence > CONFIDENCE_THRESHOLD_LOW]
        if not viable:
            viable = results  # keep all if nothing exceeds the threshold

        # 6. Try candidates in confidence order; use first that converts successfully
        results_sorted = sorted(viable, key=lambda r: r.confidence, reverse=True)
        best = None
        converted_value = None

        for candidate in results_sorted:
            cv = _convert(candidate.value, expected_type)
            if cv is not None:
                best = candidate
                converted_value = cv
                break

        if best is None:
            # All conversions failed
            best = results_sorted[0]
            # For numeric/boolean types, don't return unconvertible strings
            if expected_type in ("int", "float", "boolean"):
                if model_id:
                    _unparseable_stats[model_id][1] += 1
                return ParseResult(value=None, confidence=0.0, strategy_used="none")
            converted_value = best.value  # keep as string for str/auto/comparison types

        # 7. Apply post-processors
        for proc in self.config.post_processors:
            try:
                converted_value = proc(converted_value)
            except Exception as exc:
                logger.debug("Post-processor failed: %s", exc)

        # 8. Warn if best confidence is low
        if best.confidence < CONFIDENCE_THRESHOLD_LOW:
            logger.debug(
                "Low-confidence parse (%.2f via %s) for model=%s — full response logged at DEBUG",
                best.confidence, best.strategy_used, model_id,
            )

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
    model_profile: Optional[Any] = None,
) -> ParseResult:
    """
    One-shot convenience function: parse a response for a given task.

    Args:
        response: Raw model response text.
        task_name: Task name for config lookup.
        model_id: Model identifier for adapter selection and profile lookup.
        model_profile: Optional pre-loaded ModelProfile (skips auto-lookup).

    Returns:
        ParseResult.
    """
    parser = UnifiedParser(task_name=task_name)
    return parser.parse(response, model_id=model_id, model_profile=model_profile)

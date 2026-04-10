"""
Phase 4 tests: Model Behavior Profiling & Per-Model Parsing.

These are unit tests (no GPU required). They use synthetic responses and
mock handlers to verify profiling logic and profile-driven parsing.
"""

import json
import time
from pathlib import Path
from typing import List
import pytest


# ============================================================================
# ModelProfile / ModelProfiler unit tests
# ============================================================================

class TestModelProfile:
    def test_profile_dataclass_defaults(self):
        from beyondbench.utils.model_profiler import ModelProfile
        p = ModelProfile(model_id="test-model")
        assert p.boxed_rate == 0.0
        assert p.explicit_rate == 0.0
        assert p.recommended_strategies == []
        assert p.quirks == []
        assert p.profile_version == "1.0"

    def test_profile_serialization_roundtrip(self, tmp_path):
        from beyondbench.utils.model_profiler import ModelProfile
        from dataclasses import asdict

        p = ModelProfile(
            model_id="test/model",
            boxed_rate=0.75,
            explicit_rate=0.60,
            cot_rate=0.40,
            recommended_strategies=["boxed", "explicit_statement", "fallback"],
            quirks=["test quirk"],
            num_calibration_samples=20,
            calibration_timestamp=1744300000.0,
        )
        path = tmp_path / "test_model.json"
        with open(path, "w") as f:
            json.dump(asdict(p), f)

        with open(path) as f:
            data = json.load(f)
        p2 = ModelProfile(**data)

        assert p2.model_id == p.model_id
        assert p2.boxed_rate == p.boxed_rate
        assert p2.recommended_strategies == p.recommended_strategies


class TestModelProfiler:
    """Test profiling logic using a mock handler."""

    def _make_profiler(self, responses: List[str]):
        """Build a ModelProfiler backed by a canned response list."""
        from beyondbench.utils.model_profiler import ModelProfiler

        class _FakeHandler:
            def __init__(self, resps):
                self._resps = list(resps)
                self._idx = 0
                self.model_id = "fake/model"

            def generate_response(self, prompt, **kwargs):
                if self._idx < len(self._resps):
                    r = self._resps[self._idx]
                    self._idx += 1
                    return r
                return "42"

        return ModelProfiler(_FakeHandler(responses), model_id="fake/model")

    def test_boxed_rate_detection(self):
        responses = [r"The answer is $\boxed{42}$." for _ in range(20)]
        profiler = self._make_profiler(responses)
        profile = profiler.profile(max_tokens=64)
        assert profile.boxed_rate == pytest.approx(1.0)
        assert profile.num_calibration_samples == 20

    def test_explicit_rate_detection(self):
        responses = ["The answer is 42." for _ in range(20)]
        profiler = self._make_profiler(responses)
        profile = profiler.profile(max_tokens=64)
        assert profile.explicit_rate == pytest.approx(1.0)
        assert profile.boxed_rate == pytest.approx(0.0)

    def test_cot_rate_detection(self):
        responses = ["Let me think step by step. 1+1=2. The answer is 2." for _ in range(20)]
        profiler = self._make_profiler(responses)
        profile = profiler.profile(max_tokens=64)
        assert profile.cot_rate > 0.5

    def test_avg_length(self):
        responses = ["x" * 100 for _ in range(20)]
        profiler = self._make_profiler(responses)
        profile = profiler.profile(max_tokens=64)
        assert profile.avg_response_length == pytest.approx(100.0)
        assert profile.avg_token_estimate == pytest.approx(25.0)

    def test_non_latin_detection(self):
        # Mix Latin and non-Latin responses
        responses = ["答案是42。" for _ in range(10)] + ["The answer is 42." for _ in range(10)]
        profiler = self._make_profiler(responses)
        profile = profiler.profile(max_tokens=64)
        assert profile.non_latin_rate == pytest.approx(0.5)

    def test_recommended_strategies_order(self):
        # High boxed rate → boxed should be first
        responses = [r"\boxed{42}" for _ in range(20)]
        profiler = self._make_profiler(responses)
        profile = profiler.profile(max_tokens=64)
        assert profile.recommended_strategies[0] == "boxed"
        assert "fallback" in profile.recommended_strategies

    def test_qwen_quirk_detected(self):
        responses = ["答案是42。" for _ in range(5)] + ["42" for _ in range(15)]
        profiler = self._make_profiler(responses)
        profiler.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        profile = profiler.profile(max_tokens=64)
        assert any("Qwen" in q or "Chinese" in q for q in profile.quirks)

    def test_save_and_load(self, tmp_path):
        from beyondbench.utils.model_profiler import ModelProfiler
        responses = [r"\boxed{42}" for _ in range(20)]
        profiler = self._make_profiler(responses)
        profiler.model_id = "test/save-model"
        profile = profiler.profile(max_tokens=64)
        profiler.save(profile, str(tmp_path))

        loaded = ModelProfiler.load("test/save-model", str(tmp_path))
        assert loaded is not None
        assert loaded.model_id == "test/save-model"
        assert loaded.boxed_rate == profile.boxed_rate

    def test_load_missing_returns_none(self, tmp_path):
        from beyondbench.utils.model_profiler import ModelProfiler
        result = ModelProfiler.load("does/not/exist", str(tmp_path))
        assert result is None


# ============================================================================
# Built-in profiles
# ============================================================================

class TestBuiltinProfiles:
    @pytest.mark.parametrize("model_id,expected_key", [
        ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b"),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama-3.2-3b"),
        ("microsoft/Phi-3.5-mini-instruct", "phi-3.5-mini"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral-7b"),
        ("google/gemma-2-2b-it", "gemma-2-2b"),
        ("google/gemma-2-9b-it", "gemma-2-9b"),
        ("gpt-4o", "gpt-4o"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("claude-sonnet-4-20250514", "claude-sonnet-4"),
    ])
    def test_builtin_profile_found(self, model_id, expected_key):
        from beyondbench.utils.model_profiles import load_builtin_profile
        data = load_builtin_profile(model_id)
        assert data is not None, f"No built-in profile found for {model_id}"
        assert "model_id" in data
        assert "recommended_strategies" in data
        assert "boxed_rate" in data

    def test_unknown_model_returns_none(self):
        from beyondbench.utils.model_profiles import load_builtin_profile
        assert load_builtin_profile("unknown/obscure-model-xyz") is None

    def test_list_builtin_models(self):
        from beyondbench.utils.model_profiles import list_builtin_models
        models = list_builtin_models()
        assert len(models) >= 14
        assert "qwen2.5-7b" in models

    def test_all_profiles_valid_json(self):
        from pathlib import Path
        profiles_dir = Path(__file__).parent.parent / "beyondbench" / "utils" / "model_profiles"
        for json_file in profiles_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            assert "model_id" in data, f"Missing model_id in {json_file}"
            assert "boxed_rate" in data, f"Missing boxed_rate in {json_file}"
            assert isinstance(data["recommended_strategies"], list)

    @pytest.mark.parametrize("model_id", [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "gpt-4o",
    ])
    def test_profile_rates_in_range(self, model_id):
        from beyondbench.utils.model_profiles import load_builtin_profile
        data = load_builtin_profile(model_id)
        assert data is not None
        for key in ("boxed_rate", "explicit_rate", "code_block_rate", "cot_rate", "non_latin_rate"):
            val = data[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range for {model_id}"


# ============================================================================
# Profile-driven UnifiedParser
# ============================================================================

class TestProfileDrivenParsing:
    def test_strategy_reorder_high_boxed(self):
        """Profile with high boxed rate should reorder boxed to front."""
        from beyondbench.parsers.core import UnifiedParser
        from beyondbench.utils.model_profiler import ModelProfile

        profile = ModelProfile(
            model_id="test",
            boxed_rate=0.95,
            explicit_rate=0.20,
            recommended_strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        )

        parser = UnifiedParser(task_name="sum")
        reordered = parser._apply_profile_strategies(
            ["explicit_statement", "boxed", "code_block", "latex_math", "fallback"],
            profile,
        )
        assert reordered[0] == "boxed"

    def test_strategy_reorder_high_explicit(self):
        """Profile with high explicit rate should reorder explicit_statement to front."""
        from beyondbench.parsers.core import UnifiedParser
        from beyondbench.utils.model_profiler import ModelProfile

        profile = ModelProfile(
            model_id="test",
            boxed_rate=0.10,
            explicit_rate=0.90,
            recommended_strategies=["explicit_statement", "boxed", "latex_math", "code_block", "fallback"],
        )

        parser = UnifiedParser(task_name="sum")
        reordered = parser._apply_profile_strategies(
            ["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
            profile,
        )
        assert reordered[0] == "explicit_statement"

    def test_no_profile_keeps_original_order(self):
        from beyondbench.parsers.core import UnifiedParser
        parser = UnifiedParser(task_name="sum")
        original = ["boxed", "explicit_statement", "fallback"]
        result = parser._apply_profile_strategies(original, None)
        assert result == original

    def test_parse_with_builtin_profile(self):
        """parse() with model_id for a known model should not raise."""
        from beyondbench.parsers.core import UnifiedParser
        parser = UnifiedParser(task_name="sum")
        result = parser.parse(
            r"The answer is \boxed{42}.",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
        )
        assert result.success
        assert result.value == 42

    def test_parse_with_explicit_profile_object(self):
        from beyondbench.parsers.core import UnifiedParser
        from beyondbench.utils.model_profiler import ModelProfile

        profile = ModelProfile(
            model_id="llama-test",
            explicit_rate=0.90,
            boxed_rate=0.05,
            recommended_strategies=["explicit_statement", "boxed", "latex_math", "code_block", "fallback"],
        )
        parser = UnifiedParser(task_name="sum")
        result = parser.parse(
            "The answer is 99.",
            model_profile=profile,
        )
        assert result.success
        assert result.value == 99

    def test_load_profile_builtin(self):
        from beyondbench.parsers.core import UnifiedParser
        profile = UnifiedParser._load_profile("Qwen/Qwen2.5-7B-Instruct")
        assert profile is not None
        assert profile.boxed_rate > 0

    def test_load_profile_unknown_returns_none(self):
        from beyondbench.parsers.core import UnifiedParser
        profile = UnifiedParser._load_profile("totally/unknown-model-xyz-abc")
        assert profile is None

    def test_load_profile_none_model_id(self):
        from beyondbench.parsers.core import UnifiedParser
        profile = UnifiedParser._load_profile(None)
        assert profile is None


# ============================================================================
# Confidence thresholds and unparseable tracking
# ============================================================================

class TestConfidenceTracking:
    def setup_method(self):
        from beyondbench.parsers.core import reset_unparseable_stats
        reset_unparseable_stats()

    def test_unparseable_empty_response(self):
        from beyondbench.parsers.core import UnifiedParser, get_unparseable_rate
        parser = UnifiedParser(task_name="sum")
        result = parser.parse("", model_id="test-model")
        assert not result.success
        assert get_unparseable_rate("test-model") == pytest.approx(1.0)

    def test_parseable_does_not_count_as_unparseable(self):
        from beyondbench.parsers.core import UnifiedParser, get_unparseable_rate
        parser = UnifiedParser(task_name="sum")
        # Parse a clearly answerable response
        result = parser.parse(r"\boxed{7}", model_id="test-model-ok")
        assert result.success
        assert get_unparseable_rate("test-model-ok") == pytest.approx(0.0)

    def test_unparseable_rate_accumulates(self):
        from beyondbench.parsers.core import UnifiedParser, get_unparseable_rate
        parser = UnifiedParser(task_name="sum")
        model = "accum-model"
        # 2 successful, 1 failed
        parser.parse(r"\boxed{1}", model_id=model)
        parser.parse(r"\boxed{2}", model_id=model)
        parser.parse("", model_id=model)  # empty → unparseable
        rate = get_unparseable_rate(model)
        assert rate == pytest.approx(1 / 3, abs=0.01)

    def test_reset_clears_stats(self):
        from beyondbench.parsers.core import UnifiedParser, get_unparseable_rate, reset_unparseable_stats
        parser = UnifiedParser(task_name="sum")
        parser.parse("", model_id="reset-model")
        assert get_unparseable_rate("reset-model") > 0
        reset_unparseable_stats()
        assert get_unparseable_rate("reset-model") == 0.0

    def test_confidence_threshold_exported(self):
        from beyondbench.parsers.core import CONFIDENCE_THRESHOLD_LOW
        assert 0.0 < CONFIDENCE_THRESHOLD_LOW < 1.0

    def test_parse_response_convenience(self):
        from beyondbench.parsers.core import parse_response
        result = parse_response(r"\boxed{55}", task_name="sum", model_id="gpt-4o")
        assert result.success
        assert result.value == 55


# ============================================================================
# parsers/__init__.py exports
# ============================================================================

class TestParserExports:
    def test_phase4_exports_available(self):
        from beyondbench.parsers import (
            get_unparseable_rate,
            reset_unparseable_stats,
            CONFIDENCE_THRESHOLD_LOW,
        )
        assert callable(get_unparseable_rate)
        assert callable(reset_unparseable_stats)
        assert isinstance(CONFIDENCE_THRESHOLD_LOW, float)

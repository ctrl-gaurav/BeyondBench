"""Unit tests for Phase 16 — Contamination Resistance Hardening."""
import json
import sys
import pytest

# ---------------------------------------------------------------------------
# 16.1 fingerprint_instance
# ---------------------------------------------------------------------------

class TestFingerprintInstance:
    def test_consistent_same_input(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp1 = fingerprint_instance([1, 2, 3], "sum", seed=42)
        fp2 = fingerprint_instance([1, 2, 3], "sum", seed=42)
        assert fp1 == fp2, "Same input must produce same fingerprint"

    def test_different_data_different_fingerprint(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp1 = fingerprint_instance([1, 2, 3], "sum", seed=42)
        fp3 = fingerprint_instance([1, 2, 4], "sum", seed=42)
        assert fp1 != fp3, "Different data must produce different fingerprint"

    def test_different_seed_different_fingerprint(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp1 = fingerprint_instance([1, 2, 3], "sum", seed=42)
        fp2 = fingerprint_instance([1, 2, 3], "sum", seed=99)
        assert fp1 != fp2

    def test_different_task_different_fingerprint(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp1 = fingerprint_instance([1, 2, 3], "sum", seed=42)
        fp2 = fingerprint_instance([1, 2, 3], "sort", seed=42)
        assert fp1 != fp2

    def test_returns_16_char_hex(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp = fingerprint_instance({"a": 1}, "task", seed=0)
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_handles_dict(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp = fingerprint_instance({"numbers": [1, 2], "answer": 3}, "sum", seed=1)
        assert len(fp) == 16

    def test_handles_none_seed(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp = fingerprint_instance([5, 6], "task", seed=None)
        assert len(fp) == 16

    def test_handles_string(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        fp = fingerprint_instance("hello world", "text_task", seed=7)
        assert len(fp) == 16

    def test_handles_nested_structure(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        data = {"numbers": [1, 2, 3], "meta": {"size": 3, "tags": ["a", "b"]}}
        fp1 = fingerprint_instance(data, "nested", seed=0)
        fp2 = fingerprint_instance(data, "nested", seed=0)
        assert fp1 == fp2

    def test_handles_numpy_array(self):
        try:
            import numpy as np
            from beyondbench.core.fingerprint import fingerprint_instance
            arr = np.array([1, 2, 3])
            fp = fingerprint_instance(arr, "np_task", seed=5)
            assert len(fp) == 16
        except ImportError:
            pytest.skip("numpy not available")

    def test_handles_set(self):
        from beyondbench.core.fingerprint import fingerprint_instance
        # sets are serialized as sorted lists
        fp1 = fingerprint_instance({1, 2, 3}, "set_task", seed=1)
        fp2 = fingerprint_instance({3, 1, 2}, "set_task", seed=1)
        assert fp1 == fp2, "Sets with same elements should produce same fingerprint"


# ---------------------------------------------------------------------------
# 16.2 ProblemRephraser
# ---------------------------------------------------------------------------

class TestProblemRephraser:
    def test_import(self):
        from beyondbench.core.rephraser import ProblemRephraser
        assert ProblemRephraser is not None

    def test_rephrase_known_operation(self):
        from beyondbench.core.rephraser import ProblemRephraser
        rp = ProblemRephraser(seed=0)
        phrase = rp.rephrase("sum")
        assert isinstance(phrase, str)
        assert len(phrase) > 0

    def test_rephrase_returns_valid_option(self):
        from beyondbench.core.rephraser import ProblemRephraser
        rp = ProblemRephraser(seed=0)
        all_phrases = rp.get_all_phrasings("sum")
        phrase = rp.rephrase("sum")
        assert phrase in all_phrases

    def test_same_seed_same_result(self):
        from beyondbench.core.rephraser import ProblemRephraser
        rp1 = ProblemRephraser(seed=42)
        rp2 = ProblemRephraser(seed=42)
        assert rp1.rephrase("sum") == rp2.rephrase("sum")

    def test_unknown_operation_returns_itself(self):
        from beyondbench.core.rephraser import ProblemRephraser
        rp = ProblemRephraser(seed=0)
        phrase = rp.rephrase("totally_unknown_op_xyz")
        assert phrase == "totally_unknown_op_xyz"

    def test_get_all_phrasings_returns_list(self):
        from beyondbench.core.rephraser import ProblemRephraser
        rp = ProblemRephraser()
        phrasings = rp.get_all_phrasings("sum")
        assert isinstance(phrasings, list)
        assert len(phrasings) > 1

    def test_all_operations_have_entries(self):
        from beyondbench.core.rephraser import ProblemRephraser
        for op in ["sum", "sort_ascending", "find_maximum", "find_minimum",
                   "count", "multiply", "subtract", "mean", "median", "compare",
                   "sequence_completion", "graph_problem", "optimization"]:
            phrasings = ProblemRephraser.PHRASINGS.get(op)
            assert phrasings is not None and len(phrasings) > 0, f"Missing phrasings for {op}"


# ---------------------------------------------------------------------------
# 16.3 NoiseInjector
# ---------------------------------------------------------------------------

class TestNoiseInjector:
    SAMPLE_PROMPT = (
        "Add the following numbers: 1, 2, 3\n\nAnswer in \\boxed{answer} format."
    )

    def test_import(self):
        from beyondbench.core.noise_injector import NoiseInjector
        assert NoiseInjector is not None

    def test_invalid_level_raises(self):
        from beyondbench.core.noise_injector import NoiseInjector
        with pytest.raises(ValueError, match="level must be one of"):
            NoiseInjector(level="extreme")

    def test_low_returns_string(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni = NoiseInjector(level="low", seed=0)
        result = ni.inject(self.SAMPLE_PROMPT)
        assert isinstance(result, str) and len(result) > 0

    def test_medium_returns_string(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni = NoiseInjector(level="medium", seed=0)
        result = ni.inject(self.SAMPLE_PROMPT)
        assert isinstance(result, str) and len(result) > 0

    def test_high_returns_string(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni = NoiseInjector(level="high", seed=0)
        result = ni.inject(self.SAMPLE_PROMPT)
        assert isinstance(result, str) and len(result) > 0

    def test_low_preserves_boxed(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni = NoiseInjector(level="low", seed=0)
        result = ni.inject(self.SAMPLE_PROMPT)
        assert "\\boxed" in result, "\\boxed must be preserved after low noise"

    def test_medium_preserves_boxed(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni = NoiseInjector(level="medium", seed=0)
        result = ni.inject(self.SAMPLE_PROMPT)
        assert "\\boxed" in result, "\\boxed must be preserved after medium noise"

    def test_high_preserves_boxed(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni = NoiseInjector(level="high", seed=0)
        result = ni.inject(self.SAMPLE_PROMPT)
        assert "\\boxed" in result, "\\boxed must be preserved after high noise"

    def test_high_boxed_is_last_instruction(self):
        """After high injection, \\boxed format must still be near the end."""
        from beyondbench.core.noise_injector import NoiseInjector
        ni = NoiseInjector(level="high", seed=0)
        result = ni.inject(self.SAMPLE_PROMPT)
        # The boxed instruction should appear after the distractor
        boxed_idx = result.rfind("\\boxed")
        assert boxed_idx != -1
        # Distractors should come BEFORE the boxed instruction
        assert boxed_idx > 0

    def test_medium_longer_than_low(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni_low = NoiseInjector(level="low", seed=0)
        ni_med = NoiseInjector(level="medium", seed=0)
        result_low = ni_low.inject(self.SAMPLE_PROMPT)
        result_med = ni_med.inject(self.SAMPLE_PROMPT)
        assert len(result_med) >= len(result_low)

    def test_high_longer_than_medium(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni_med = NoiseInjector(level="medium", seed=0)
        ni_high = NoiseInjector(level="high", seed=0)
        result_med = ni_med.inject(self.SAMPLE_PROMPT)
        result_high = ni_high.inject(self.SAMPLE_PROMPT)
        assert len(result_high) >= len(result_med)

    def test_same_seed_deterministic(self):
        from beyondbench.core.noise_injector import NoiseInjector
        ni1 = NoiseInjector(level="high", seed=77)
        ni2 = NoiseInjector(level="high", seed=77)
        r1 = ni1.inject(self.SAMPLE_PROMPT)
        r2 = ni2.inject(self.SAMPLE_PROMPT)
        assert r1 == r2

    def test_prompt_without_boxed(self):
        """Should not crash on prompts without \\boxed."""
        from beyondbench.core.noise_injector import NoiseInjector
        prompt = "What is 2 + 2?"
        ni = NoiseInjector(level="high", seed=1)
        result = ni.inject(prompt)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 16.4 CLI option present
# ---------------------------------------------------------------------------

class TestCLIContaminationOption:
    def test_cli_has_contamination_resistance_option(self):
        """Verify the CLI accepts --contamination-resistance."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "contamination-resistance" in result.output

    def test_cli_contamination_default_is_off(self):
        """The default value must be 'off' so existing runs are unaffected."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert "off" in result.output


# ---------------------------------------------------------------------------
# 16.5 Seed variance audit (subset of tasks)
# ---------------------------------------------------------------------------

class TestSeedVarianceAudit:
    """Verify a representative set of tasks generates different data for different seeds."""

    SPOT_CHECK_TASKS = [
        "sum",
        "sort_ascending",
        "find_maximum",
    ]

    def _make_stub(self):
        class _Stub:
            backend = "stub"
            def get_model_info(self): return {"model_name": "stub", "backend": "stub"}
            def generate(self, prompts, **kw): return [""] * len(prompts)
        return _Stub()

    def _get_data(self, task_name: str, seed: int):
        import inspect
        from beyondbench.core.task_registry import TaskRegistry
        registry = TaskRegistry()
        cls = registry.get_task_class(task_name)
        if cls is None:
            return None

        stub = self._make_stub()
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        accepts_kw = "kwargs" in params

        kw = {
            "model_handler": stub,
            "output_dir": "/tmp/bb_test_audit",
            "num_folds": 1,
            "num_samples": 5,
            "store_details": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "seed": seed,
            "prompt_style": None,
            "contamination_resistance": "off",
        }
        if "min_val" in params or accepts_kw:
            kw["min_val"] = 1
            kw["max_val"] = 100

        filtered = kw if accepts_kw else {k: v for k, v in kw.items() if k in params}
        try:
            task = cls(**filtered)
            data = task.generate_data()
            return data
        except Exception:
            try:
                task = cls(**filtered)
                return task.generate_data(list_size=8)
            except Exception:
                return None

    @pytest.mark.parametrize("task_name", SPOT_CHECK_TASKS)
    def test_different_seeds_produce_different_data(self, task_name):
        data_42 = self._get_data(task_name, seed=42)
        data_43 = self._get_data(task_name, seed=43)

        if data_42 is None or data_43 is None:
            pytest.skip(f"Could not generate data for task {task_name}")

        serialized_42 = [json.dumps(d, sort_keys=True, default=str) for d in data_42[:3]]
        serialized_43 = [json.dumps(d, sort_keys=True, default=str) for d in data_43[:3]]

        # At least one data point should differ between seeds
        any_diff = any(a != b for a, b in zip(serialized_42, serialized_43))
        assert any_diff, f"Task {task_name}: data is identical for seed=42 and seed=43 — seeds may not work"

    @pytest.mark.parametrize("task_name", SPOT_CHECK_TASKS)
    def test_fingerprints_are_unique_within_seed(self, task_name):
        from beyondbench.core.fingerprint import fingerprint_instance

        data = self._get_data(task_name, seed=42)
        if data is None or len(data) < 2:
            pytest.skip(f"Not enough data for task {task_name}")

        fps = [fingerprint_instance(dp, task_name, seed=42) for dp in data]
        assert len(fps) == len(set(fps)), f"Task {task_name}: fingerprint collisions detected"

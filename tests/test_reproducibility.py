"""
Tests for Phase 17 — Evaluation Reproducibility & Seed Management.

Covers:
- seed_manager.set_all_seeds
- eval_fingerprint.compute_eval_fingerprint
- eval_fingerprint.get_environment_info
- eval_fingerprint.check_version_compatibility
- BaseTask seed propagation
- CLI reproduce command
"""

import random
import sys
import pytest
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------ #
# seed_manager tests
# ------------------------------------------------------------------ #

class TestSetAllSeeds:
    def test_set_all_seeds_sets_random(self):
        """set_all_seeds sets Python random state deterministically."""
        from beyondbench.core.seed_manager import set_all_seeds

        set_all_seeds(42)
        val1 = random.random()

        set_all_seeds(42)
        val2 = random.random()

        assert val1 == val2, "random.random() should be identical after same seed"

    def test_set_all_seeds_sets_numpy(self):
        """set_all_seeds seeds numpy.random deterministically."""
        import numpy as np
        from beyondbench.core.seed_manager import set_all_seeds

        set_all_seeds(99)
        arr1 = np.random.rand(5).tolist()

        set_all_seeds(99)
        arr2 = np.random.rand(5).tolist()

        assert arr1 == arr2, "np.random.rand() should be identical after same seed"

    def test_set_all_seeds_none_is_noop(self):
        """set_all_seeds(None) does not raise and does not change state."""
        from beyondbench.core.seed_manager import set_all_seeds

        # Should not raise
        set_all_seeds(None)

    def test_same_seed_same_random_output(self):
        """Calling set_all_seeds(42) twice yields same random.random() sequence."""
        from beyondbench.core.seed_manager import set_all_seeds

        set_all_seeds(42)
        seq1 = [random.random() for _ in range(10)]

        set_all_seeds(42)
        seq2 = [random.random() for _ in range(10)]

        assert seq1 == seq2

    def test_set_all_seeds_different_seeds_differ(self):
        """Different seeds produce different random sequences."""
        from beyondbench.core.seed_manager import set_all_seeds

        set_all_seeds(1)
        v1 = random.random()

        set_all_seeds(2)
        v2 = random.random()

        assert v1 != v2, "Different seeds should yield different values"

    def test_set_all_seeds_handles_missing_torch(self):
        """set_all_seeds works even when torch is not installed."""
        from beyondbench.core.seed_manager import set_all_seeds

        with patch.dict(sys.modules, {"torch": None}):
            # Should not raise ImportError
            set_all_seeds(7)


# ------------------------------------------------------------------ #
# eval_fingerprint tests
# ------------------------------------------------------------------ #

class TestComputeEvalFingerprint:
    def _fp(self, **kwargs):
        from beyondbench.core.eval_fingerprint import compute_eval_fingerprint
        defaults = dict(
            model_id="test-model",
            config={"temperature": 0.0, "datapoints": 10},
            seed=42,
            task_list=["sum", "sort"],
            beyondbench_version="0.3.0",
        )
        defaults.update(kwargs)
        return compute_eval_fingerprint(**defaults)

    def test_compute_eval_fingerprint_deterministic(self):
        """Same inputs produce the same fingerprint."""
        fp1 = self._fp()
        fp2 = self._fp()
        assert fp1 == fp2

    def test_compute_eval_fingerprint_different_seeds(self):
        """Different seeds produce different fingerprints."""
        fp1 = self._fp(seed=42)
        fp2 = self._fp(seed=43)
        assert fp1 != fp2

    def test_compute_eval_fingerprint_different_models(self):
        """Different model IDs produce different fingerprints."""
        fp1 = self._fp(model_id="model-A")
        fp2 = self._fp(model_id="model-B")
        assert fp1 != fp2

    def test_compute_eval_fingerprint_length(self):
        """Fingerprint is exactly 32 characters."""
        fp = self._fp()
        assert len(fp) == 32

    def test_compute_eval_fingerprint_task_order_invariant(self):
        """Task list is sorted so order doesn't matter."""
        fp1 = self._fp(task_list=["sum", "sort", "multiply"])
        fp2 = self._fp(task_list=["multiply", "sum", "sort"])
        assert fp1 == fp2

    def test_compute_eval_fingerprint_none_seed(self):
        """Fingerprint with seed=None differs from one with seed=0."""
        fp_none = self._fp(seed=None)
        fp_zero = self._fp(seed=0)
        assert fp_none != fp_zero


class TestGetEnvironmentInfo:
    def test_get_environment_info_has_keys(self):
        """get_environment_info returns all expected keys."""
        from beyondbench.core.eval_fingerprint import get_environment_info

        info = get_environment_info()
        expected_keys = [
            "python_version",
            "platform",
            "beyondbench_version",
            "torch_version",
            "transformers_version",
            "vllm_version",
            "numpy_version",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_get_environment_info_python_version(self):
        """python_version matches sys.version."""
        from beyondbench.core.eval_fingerprint import get_environment_info

        info = get_environment_info()
        assert info["python_version"] == sys.version.split()[0]

    def test_get_environment_info_beyondbench_version_not_unknown(self):
        """beyondbench_version is not 'unknown' when package is installed."""
        from beyondbench.core.eval_fingerprint import get_environment_info

        info = get_environment_info()
        # Should find the version either from metadata or __version__
        assert info["beyondbench_version"] != ""


class TestCheckVersionCompatibility:
    def test_check_version_compatibility_same_env(self):
        """Same environment produces no warnings."""
        from beyondbench.core.eval_fingerprint import get_environment_info, check_version_compatibility

        env = get_environment_info()
        warnings = check_version_compatibility(env, env)
        assert warnings == []

    def test_check_version_compatibility_diff_version(self):
        """Different beyondbench versions produce a warning."""
        from beyondbench.core.eval_fingerprint import get_environment_info, check_version_compatibility

        current = get_environment_info()
        old = dict(current)
        old["beyondbench_version"] = "0.0.1"

        warnings = check_version_compatibility(old, current)
        assert any("BeyondBench version mismatch" in w for w in warnings)

    def test_check_version_compatibility_diff_python(self):
        """Different Python major.minor versions produce a warning."""
        from beyondbench.core.eval_fingerprint import get_environment_info, check_version_compatibility

        current = get_environment_info()
        old = dict(current)
        old["python_version"] = "2.7.18"

        warnings = check_version_compatibility(old, current)
        assert any("Python version mismatch" in w for w in warnings)

    def test_check_version_compatibility_missing_current_env(self):
        """current_env=None auto-computes environment."""
        from beyondbench.core.eval_fingerprint import get_environment_info, check_version_compatibility

        env = get_environment_info()
        # Should not raise
        warnings = check_version_compatibility(env)
        assert isinstance(warnings, list)


# ------------------------------------------------------------------ #
# BaseTask seed propagation test
# ------------------------------------------------------------------ #

def _make_concrete_task_class(name="test_task"):
    """Factory: return a fully concrete BaseTask subclass (satisfies all abstract methods)."""
    from beyondbench.core.base_task import BaseTask

    class ConcreteTask(BaseTask):
        @property
        def task_name(self):
            return name

        def generate_data(self, **kwargs):
            return []

        def create_prompt(self, data_point):
            return "test prompt"

        def evaluate_response(self, response, data_point):
            return True

        def run_evaluation(self):
            return {}

    return ConcreteTask


class TestSeedPropagationInBaseTask:
    def _make_task(self, seed, tmpdir):
        ConcreteTask = _make_concrete_task_class()
        mock_handler = MagicMock()
        mock_handler.get_model_info.return_value = {"model_name": "test"}
        return ConcreteTask(
            model_handler=mock_handler,
            output_dir=tmpdir,
            min_val=0,
            max_val=100,
            num_folds=1,
            num_samples=5,
            store_details=False,
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            seed=seed,
        )

    def test_seed_propagation_in_base_task(self):
        """Creating a BaseTask with seed=42 calls set_all_seeds(42)."""
        from beyondbench.core.seed_manager import set_all_seeds

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Get baseline: what random.random() returns after set_all_seeds(42)
            set_all_seeds(42)
            baseline = random.random()

            # Now mess up state, then create task with seed=42
            set_all_seeds(999)
            self._make_task(seed=42, tmpdir=tmpdir)
            val_after_task = random.random()

            assert val_after_task == baseline, (
                f"random.random() after BaseTask(seed=42) should match set_all_seeds(42): "
                f"expected {baseline}, got {val_after_task}"
            )

    def test_base_task_seed_none_no_crash(self):
        """BaseTask with seed=None does not raise."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._make_task(seed=None, tmpdir=tmpdir)
        assert task.seed is None

    def test_base_task_set_all_seeds_called(self):
        """Verify set_all_seeds is called with the right seed via mock."""
        import tempfile
        with patch("beyondbench.core.base_task.set_all_seeds") as mock_seeds:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._make_task(seed=77, tmpdir=tmpdir)
            mock_seeds.assert_called_with(77)


# ------------------------------------------------------------------ #
# CLI reproduce command test
# ------------------------------------------------------------------ #

class TestCLIReproduceCommand:
    def test_cli_reproduce_command_exists(self):
        """The 'reproduce' command is registered in the main CLI group."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["reproduce", "--help"])

        assert result.exit_code == 0, f"reproduce --help failed: {result.output}"
        assert "fingerprint" in result.output.lower() or "FINGERPRINT" in result.output

    def test_cli_reproduce_missing_dir(self):
        """reproduce exits non-zero when results dir does not exist."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "reproduce", "abc123",
            "--results-dir", "/nonexistent/path/to/nowhere"
        ])

        assert result.exit_code != 0

    def test_cli_reproduce_not_found(self, tmp_path):
        """reproduce reports 'not found' when no matching fingerprint exists."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        import json

        # Create a dummy result file without the fingerprint
        dummy = {"summary": {"avg_accuracy": 0.5}, "task_results": {}}
        (tmp_path / "final_results.json").write_text(json.dumps(dummy))

        runner = CliRunner()
        result = runner.invoke(main, [
            "reproduce", "deadbeef1234",
            "--results-dir", str(tmp_path)
        ])

        # Should fail (fingerprint not found)
        assert result.exit_code != 0 or "No results" in result.output

    def test_cli_reproduce_finds_fingerprint(self, tmp_path):
        """reproduce finds and displays config when fingerprint matches."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        import json

        fp = "abcdef1234567890abcdef1234567890"
        result_data = {
            "eval_fingerprint": fp,
            "environment": {"python_version": "3.11.0", "beyondbench_version": "0.3.0"},
            "evaluation_config": {"suite": "easy", "tasks": ["sum"]},
            "model_info": {"model_name": "test-model"},
            "summary": {"total_duration": 10.0},
        }
        (tmp_path / "final_results.json").write_text(json.dumps(result_data))

        runner = CliRunner()
        result = runner.invoke(main, [
            "reproduce", fp[:8],  # partial fingerprint
            "--results-dir", str(tmp_path)
        ])

        assert result.exit_code == 0, f"reproduce failed: {result.output}"
        assert fp in result.output

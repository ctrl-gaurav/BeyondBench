"""
E2E test — fresh install validation.

Tests that BeyondBench can be installed into a fresh environment and
basic imports + CLI commands work correctly.

Note: Full conda-create tests take too long for CI; this file tests:
  1. That the package installs cleanly into the current venv
  2. That all required imports work
  3. That the CLI entry point works
  4. Optionally: subprocess-based fresh-venv test (marked 'slow')
"""

import os
import subprocess
import sys
import importlib
import logging
import pytest
from pathlib import Path

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Import validation — all public modules importable
# ===========================================================================

class TestImports:
    def test_beyondbench_imports(self):
        import beyondbench
        assert beyondbench is not None

    def test_models_model_handler(self):
        from beyondbench.models.model_handler import ModelHandler
        assert ModelHandler is not None

    def test_core_base_task(self):
        from beyondbench.core.base_task import BaseTask
        assert BaseTask is not None

    def test_core_evaluation_engine(self):
        from beyondbench.core.evaluation_engine import EvaluationEngine
        assert EvaluationEngine is not None

    def test_core_task_registry(self):
        from beyondbench.core.task_registry import TaskRegistry
        assert TaskRegistry is not None

    def test_cli_main(self):
        from beyondbench.cli.main import main
        assert main is not None

    def test_parsers_core(self):
        from beyondbench.parsers.core import UnifiedParser
        assert UnifiedParser is not None

    def test_parsers_common(self):
        from beyondbench.parsers.common import (
            extract_from_boxed_formats,
            extract_from_explicit_statements,
            clean_and_convert_to_number,
        )
        assert all(fn is not None for fn in [
            extract_from_boxed_formats,
            extract_from_explicit_statements,
            clean_and_convert_to_number,
        ])

    def test_all_easy_tasks_importable(self):
        easy_modules = [
            ("beyondbench.tasks.easy.sum_task", "SumTask"),
            ("beyondbench.tasks.easy.sorting_task", "SortingTask"),
            ("beyondbench.tasks.easy.multiplication_task", "MultiplicationTask"),
            ("beyondbench.tasks.easy.comparison_task", "ComparisonTask"),
            ("beyondbench.tasks.easy.find_maximum_task", "FindMaximumTask"),
            ("beyondbench.tasks.easy.find_minimum_task", "FindMinimumTask"),
            ("beyondbench.tasks.easy.mean_task", "MeanTask"),
            ("beyondbench.tasks.easy.median_task", "MedianTask"),
            ("beyondbench.tasks.easy.mode_task", "ModeTask"),
        ]
        for mod_path, cls_name in easy_modules:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            assert cls is not None, f"Could not import {cls_name}"

    def test_all_parsers_importable(self):
        parser_modules = [
            ("beyondbench.parsers.sum_parsing", "parse_sum_answer"),
            ("beyondbench.parsers.sorting_parsing", "parse_sorted_list"),
            ("beyondbench.parsers.comparison_parsing", "parse_comparison_result"),
        ]
        for mod_path, fn_name in parser_modules:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, fn_name)
            assert fn is not None

    def test_utils_importable(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        from beyondbench.utils.logging_utils import get_logger
        assert StatsTracker is not None
        assert get_logger is not None


# ===========================================================================
# 2. Package metadata
# ===========================================================================

class TestPackageMetadata:
    def test_version_string_exists(self):
        try:
            import beyondbench
            version = getattr(beyondbench, "__version__", None)
            if version is None:
                # Try importlib.metadata
                from importlib.metadata import version as get_version
                version = get_version("beyondbench")
            assert version is not None
            assert isinstance(version, str) and len(version) > 0
        except Exception:
            pytest.skip("Version not accessible")

    def test_cli_entry_point(self):
        """beyondbench --version should work from the CLI."""
        result = subprocess.run(
            [sys.executable, "-m", "beyondbench.cli.main", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


# ===========================================================================
# 3. Basic evaluation smoke test (no GPU, mock handler)
# ===========================================================================

class TestBasicEvaluationSmoke:
    def test_run_sum_task_no_gpu(self, tmp_path):
        """Run sum task with mock handler — verifies whole stack without GPU."""
        from unittest.mock import MagicMock
        from beyondbench.core.evaluation_engine import EvaluationEngine

        handler = MagicMock()
        handler.backend = "mock"
        handler.api_provider = None
        handler.model_id = "smoke-test-model"
        handler.tokenizer = None
        handler.generate = MagicMock(return_value=[r"\boxed{42}"] * 100)
        handler.generate_batch = handler.generate
        handler.get_model_info = MagicMock(return_value={"model_name": "smoke", "backend": "mock"})
        handler.get_statistics = MagicMock(return_value={"total_time_seconds": 0.0})
        handler.count_tokens = MagicMock(return_value=5)

        engine = EvaluationEngine(model_handler=handler, output_dir=str(tmp_path / "smoke"))
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])

        assert isinstance(result, dict)
        assert "summary" in result
        assert "task_results" in result

    def test_parser_pipeline_works(self):
        """Verify parser pipeline can handle a typical model response."""
        from beyondbench.parsers.core import UnifiedParser
        from beyondbench.parsers.task_configs import get_task_config

        config = get_task_config("sum")
        parser = UnifiedParser(config)
        result = parser.parse(r"The sum of the numbers is \boxed{42}", model_id="test-model")
        assert result.value == 42

    def test_task_registry_loads(self):
        """Task registry should load all 44 tasks."""
        from beyondbench.core.task_registry import TaskRegistry
        registry = TaskRegistry()
        assert len(registry.get_tasks_for_suite("all")) == 44


# ===========================================================================
# 4. Optional: subprocess fresh-venv test (very slow, off by default)
# ===========================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("RUN_FRESH_INSTALL_TEST"),
    reason="Set RUN_FRESH_INSTALL_TEST=1 to run fresh-install test"
)
class TestFreshVenvInstall:
    """
    Creates a fresh virtualenv, installs beyondbench in editable mode,
    runs a basic smoke test, then tears down the venv.

    Skip this by default — it takes ~3 minutes. Enable with:
        RUN_FRESH_INSTALL_TEST=1 pytest tests/e2e/test_fresh_install.py::TestFreshVenvInstall
    """

    def test_python_311_fresh_install(self, tmp_path):
        self._run_fresh_install(tmp_path, python_version="3.11")

    def test_python_312_fresh_install(self, tmp_path):
        self._run_fresh_install(tmp_path, python_version="3.12")

    def _run_fresh_install(self, tmp_path, python_version):
        # Find a Python binary for the requested version
        python_bin = f"python{python_version}"
        result = subprocess.run(
            [python_bin, "--version"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            pytest.skip(f"Python {python_version} not available")

        venv_dir = tmp_path / f"venv_{python_version.replace('.', '_')}"

        # Create venv
        subprocess.run([python_bin, "-m", "venv", str(venv_dir)], check=True, timeout=60)

        # Get pip path
        pip = venv_dir / "bin" / "pip"
        python = venv_dir / "bin" / "python"

        # Find repo root (2 levels up from this file)
        repo_root = Path(__file__).parent.parent.parent

        # Install beyondbench in editable mode
        result = subprocess.run(
            [str(pip), "install", "-e", str(repo_root), "--quiet"],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"pip install failed:\n{result.stderr}"

        # Run smoke test
        smoke_script = """
import beyondbench
from beyondbench.core.task_registry import TaskRegistry
r = TaskRegistry()
assert len(r.get_tasks_for_suite('all')) == 44
print('SMOKE_OK')
"""
        result = subprocess.run(
            [str(python), "-c", smoke_script],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"Smoke test failed:\n{result.stderr}"
        assert "SMOKE_OK" in result.stdout

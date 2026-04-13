"""Tests that validate the CI/CD configuration itself (Phase 20)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Root of the repository (parent of the tests/ directory)
REPO_ROOT = Path(__file__).parent.parent


class TestWorkflowYAML:
    """Verify all GitHub Actions workflow YAML files parse correctly."""

    def _load_yaml(self, path: Path) -> dict:
        """Load and return a YAML file as a dict."""
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)

    def test_test_yml_parses(self):
        """test.yml must be valid YAML."""
        path = REPO_ROOT / ".github" / "workflows" / "test.yml"
        assert path.exists(), f"Missing workflow file: {path}"
        data = self._load_yaml(path)
        assert data is not None

    def test_gpu_test_yml_parses(self):
        """gpu_test.yml must be valid YAML."""
        path = REPO_ROOT / ".github" / "workflows" / "gpu_test.yml"
        assert path.exists(), f"Missing workflow file: {path}"
        data = self._load_yaml(path)
        assert data is not None

    def test_publish_yml_parses(self):
        """publish.yml must be valid YAML."""
        path = REPO_ROOT / ".github" / "workflows" / "publish.yml"
        assert path.exists(), f"Missing workflow file: {path}"
        data = self._load_yaml(path)
        assert data is not None

    def test_deploy_react_yml_unchanged(self):
        """deploy-react.yml must still exist and be valid YAML (we must not touch it)."""
        path = REPO_ROOT / ".github" / "workflows" / "deploy-react.yml"
        assert path.exists(), f"Missing workflow file: {path}"
        data = self._load_yaml(path)
        assert data is not None

    def test_test_yml_has_test_job(self):
        """test.yml must contain a test job."""
        path = REPO_ROOT / ".github" / "workflows" / "test.yml"
        data = self._load_yaml(path)
        assert "jobs" in data
        assert "test" in data["jobs"], "test.yml is missing the 'test' job"

    def test_test_yml_matrix_python_versions(self):
        """test.yml matrix must include 3.10, 3.11, 3.12, 3.13, 3.14."""
        path = REPO_ROOT / ".github" / "workflows" / "test.yml"
        data = self._load_yaml(path)
        matrix = data["jobs"]["test"]["strategy"]["matrix"]
        versions = [str(v) for v in matrix["python-version"]]
        for ver in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            assert ver in versions, f"Python {ver} missing from test matrix"

    def test_gpu_test_yml_has_integration_job(self):
        """gpu_test.yml must contain gpu-integration job."""
        path = REPO_ROOT / ".github" / "workflows" / "gpu_test.yml"
        data = self._load_yaml(path)
        assert "jobs" in data
        assert "gpu-integration" in data["jobs"]

    def test_gpu_test_yml_has_weekly_eval_job(self):
        """gpu_test.yml must contain weekly-full-eval job."""
        path = REPO_ROOT / ".github" / "workflows" / "gpu_test.yml"
        data = self._load_yaml(path)
        assert "weekly-full-eval" in data["jobs"]

    def test_gpu_test_yml_has_schedule_trigger(self):
        """gpu_test.yml must be triggered on schedule.

        Note: PyYAML parses the bare YAML key ``on`` as the Python boolean
        ``True`` (YAML 1.1 compatibility), so we must check both ``"on"`` and
        the boolean ``True`` as possible keys.
        """
        path = REPO_ROOT / ".github" / "workflows" / "gpu_test.yml"
        data = self._load_yaml(path)
        # PyYAML 5.x parses bare `on:` as True; try both
        triggers = data.get("on", None) or data.get(True, {}) or {}
        assert "schedule" in triggers, "gpu_test.yml missing schedule trigger"

    def test_publish_yml_has_testpypi_job(self):
        """publish.yml must have publish-testpypi job before publish-pypi."""
        path = REPO_ROOT / ".github" / "workflows" / "publish.yml"
        data = self._load_yaml(path)
        assert "publish-testpypi" in data["jobs"]

    def test_publish_yml_has_test_before_publish_job(self):
        """publish.yml must run tests before publishing."""
        path = REPO_ROOT / ".github" / "workflows" / "publish.yml"
        data = self._load_yaml(path)
        assert "test-before-publish" in data["jobs"]

    def test_publish_yml_pypi_needs_testpypi(self):
        """publish-pypi must depend on publish-testpypi."""
        path = REPO_ROOT / ".github" / "workflows" / "publish.yml"
        data = self._load_yaml(path)
        needs = data["jobs"]["publish-pypi"].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "publish-testpypi" in needs, "publish-pypi should need publish-testpypi"


class TestPreCommitConfig:
    """Verify .pre-commit-config.yaml parses and contains required hooks."""

    def _load_yaml(self, path: Path) -> dict:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)

    def test_pre_commit_config_parses(self):
        """pre-commit config must be valid YAML."""
        path = REPO_ROOT / ".pre-commit-config.yaml"
        assert path.exists(), f"Missing: {path}"
        data = self._load_yaml(path)
        assert data is not None
        assert "repos" in data

    def test_pre_commit_has_ruff(self):
        """pre-commit config must include ruff hooks."""
        path = REPO_ROOT / ".pre-commit-config.yaml"
        data = self._load_yaml(path)
        hook_ids = []
        for repo in data["repos"]:
            for hook in repo.get("hooks", []):
                hook_ids.append(hook["id"])
        assert "ruff" in hook_ids, "ruff hook missing from pre-commit config"
        assert "ruff-format" in hook_ids, "ruff-format hook missing from pre-commit config"

    def test_pre_commit_has_standard_hooks(self):
        """pre-commit config must include standard pre-commit-hooks."""
        path = REPO_ROOT / ".pre-commit-config.yaml"
        data = self._load_yaml(path)
        hook_ids = []
        for repo in data["repos"]:
            for hook in repo.get("hooks", []):
                hook_ids.append(hook["id"])
        for expected in ("trailing-whitespace", "end-of-file-fixer", "check-yaml"):
            assert expected in hook_ids, f"Hook '{expected}' missing from pre-commit config"

    def test_pre_commit_has_mypy(self):
        """pre-commit config must include mypy hook."""
        path = REPO_ROOT / ".pre-commit-config.yaml"
        data = self._load_yaml(path)
        hook_ids = []
        for repo in data["repos"]:
            for hook in repo.get("hooks", []):
                hook_ids.append(hook["id"])
        assert "mypy" in hook_ids, "mypy hook missing from pre-commit config"

    def test_pre_commit_has_pytest_unit(self):
        """pre-commit config must include pytest-unit local hook."""
        path = REPO_ROOT / ".pre-commit-config.yaml"
        data = self._load_yaml(path)
        hook_ids = []
        for repo in data["repos"]:
            for hook in repo.get("hooks", []):
                hook_ids.append(hook["id"])
        assert "pytest-unit" in hook_ids, "pytest-unit hook missing from pre-commit config"

    def test_pre_commit_has_isort(self):
        """pre-commit config must include isort hook."""
        path = REPO_ROOT / ".pre-commit-config.yaml"
        data = self._load_yaml(path)
        hook_ids = []
        for repo in data["repos"]:
            for hook in repo.get("hooks", []):
                hook_ids.append(hook["id"])
        assert "isort" in hook_ids, "isort hook missing from pre-commit config"


class TestRuffClean:
    """Verify ruff is installed and runs without crashing."""

    def test_ruff_is_installed(self):
        """ruff must be importable / runnable in this environment."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ruff not available: {result.stderr}"
        assert "ruff" in result.stdout.lower()

    def test_ruff_lint_on_ci_quality_file(self):
        """ruff check must exit 0 on the test_ci_quality.py file itself."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", __file__],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"ruff check found issues in test_ci_quality.py:\n{result.stdout}\n{result.stderr}"
        )

    def test_ruff_config_present(self):
        """pyproject.toml must have a [tool.ruff] section."""
        path = REPO_ROOT / "pyproject.toml"
        content = path.read_text()
        assert "[tool.ruff]" in content, "[tool.ruff] section missing from pyproject.toml"

    def test_ruff_lint_check_exits_cleanly(self):
        """ruff check on the beyondbench package must exit with code 0 or 1 (not a crash)."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", str(REPO_ROOT / "beyondbench")],
            capture_output=True,
            text=True,
        )
        # Exit code 0 = no issues, 1 = issues found, 2 = internal error/crash
        # We accept 0 or 1; only a crash (code 2+) is a failure here.
        assert result.returncode in (0, 1), (
            f"ruff exited with unexpected code {result.returncode}:\n{result.stderr}"
        )


class TestImports:
    """Verify top-level imports work cleanly."""

    def test_import_beyondbench(self):
        """beyondbench must import without errors."""
        import beyondbench

        assert hasattr(beyondbench, "__version__")

    def test_version_string(self):
        """beyondbench.__version__ must be a non-empty string."""
        import beyondbench

        assert isinstance(beyondbench.__version__, str)
        assert len(beyondbench.__version__) > 0

    def test_import_core_modules(self):
        """Core modules must be importable."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        from beyondbench.core.task_registry import TaskRegistry

        assert EvaluationEngine is not None
        assert TaskRegistry is not None

    def test_no_import_side_effects(self):
        """Importing beyondbench should not raise any exceptions."""
        try:
            import beyondbench  # noqa: F401
        except Exception as exc:
            pytest.fail(f"Import raised an exception: {exc}")


class TestCoverageConfig:
    """Verify coverage configuration exists in pyproject.toml."""

    def _load_toml(self) -> dict:
        path = REPO_ROOT / "pyproject.toml"
        assert path.exists(), f"Missing: {path}"
        try:
            import tomllib
        except ImportError:
            # Python < 3.11
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                pytest.skip("tomllib/tomli not available — skipping TOML parse test")
        with open(path, "rb") as f:
            return tomllib.load(f)

    def test_coverage_run_section_exists(self):
        """pyproject.toml must have [tool.coverage.run] section."""
        data = self._load_toml()
        assert "coverage" in data.get("tool", {}), "No [tool.coverage] section found"
        assert "run" in data["tool"]["coverage"], "No [tool.coverage.run] section found"

    def test_coverage_report_section_exists(self):
        """pyproject.toml must have [tool.coverage.report] section."""
        data = self._load_toml()
        assert "report" in data["tool"]["coverage"], "No [tool.coverage.report] section found"

    def test_coverage_fail_under_set(self):
        """[tool.coverage.report] must have fail_under configured."""
        data = self._load_toml()
        report = data["tool"]["coverage"]["report"]
        assert "fail_under" in report, "fail_under not set in [tool.coverage.report]"
        assert report["fail_under"] >= 0, "fail_under must be a non-negative number"

    def test_coverage_fail_under_value(self):
        """fail_under should be at least 50 (the project threshold for unit-only CI)."""
        data = self._load_toml()
        report = data["tool"]["coverage"]["report"]
        assert report["fail_under"] >= 50, (
            f"Expected fail_under>=50, got {report['fail_under']}"
        )

    def test_coverage_source_configured(self):
        """[tool.coverage.run] must specify the source package."""
        data = self._load_toml()
        run = data["tool"]["coverage"]["run"]
        assert "source" in run, "source not set in [tool.coverage.run]"
        assert "beyondbench" in run["source"]

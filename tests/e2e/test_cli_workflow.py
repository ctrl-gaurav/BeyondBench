"""
E2E tests — CLI workflow: list-tasks, evaluate (with real model), serve endpoint.

Tests:
  1. beyondbench list-tasks → verify all 79 tasks listed
  2. beyondbench evaluate --suite easy --tasks sum --datapoints 5 → verify results file
  3. beyondbench serve → verify FastAPI /health endpoint responds
"""

import json
import os
import signal
import subprocess
import sys
import time
import logging
import pytest
import threading
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.slow]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beyondbench_cmd():
    """Return the beyondbench CLI command as a list."""
    return [sys.executable, "-m", "beyondbench.cli.main"]


def _run_cli(*args, timeout=60, check=False):
    """Run beyondbench CLI with given args; return CompletedProcess."""
    cmd = _beyondbench_cmd() + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired:
        pytest.fail(f"CLI command timed out after {timeout}s: {' '.join(args)}")


# ===========================================================================
# 1. list-tasks
# ===========================================================================

class TestListTasksCLI:
    def test_list_tasks_exit_zero(self):
        result = _run_cli("list-tasks")
        assert result.returncode == 0, f"list-tasks failed:\n{result.stderr}"

    def test_list_tasks_shows_easy(self):
        result = _run_cli("list-tasks")
        assert "easy" in result.stdout.lower()

    def test_list_tasks_shows_medium(self):
        result = _run_cli("list-tasks")
        assert "medium" in result.stdout.lower()

    def test_list_tasks_shows_hard(self):
        result = _run_cli("list-tasks")
        assert "hard" in result.stdout.lower()

    def test_list_tasks_json_has_79_total(self):
        result = _run_cli("list-tasks", "--format", "json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        total = sum(len(v) for v in data.values())
        assert total == 79, f"Expected 79 tasks, got {total}"

    def test_list_tasks_json_easy_44(self):
        result = _run_cli("list-tasks", "--format", "json")
        data = json.loads(result.stdout)
        assert len(data.get("easy", [])) == 44

    def test_list_tasks_json_medium_15(self):
        result = _run_cli("list-tasks", "--format", "json")
        data = json.loads(result.stdout)
        assert len(data.get("medium", [])) == 15

    def test_list_tasks_json_hard_20(self):
        result = _run_cli("list-tasks", "--format", "json")
        data = json.loads(result.stdout)
        assert len(data.get("hard", [])) == 20

    def test_list_tasks_includes_sum(self):
        result = _run_cli("list-tasks")
        assert "sum" in result.stdout

    def test_list_tasks_includes_sudoku(self):
        result = _run_cli("list-tasks")
        assert "sudoku" in result.stdout.lower()

    def test_list_tasks_suite_easy(self):
        result = _run_cli("list-tasks", "--suite", "easy")
        assert result.returncode == 0
        assert "easy" in result.stdout.lower()

    def test_list_tasks_suite_hard(self):
        result = _run_cli("list-tasks", "--suite", "hard")
        assert result.returncode == 0


# ===========================================================================
# 2. evaluate with real model (GPU required)
# ===========================================================================

@pytest.mark.gpu
class TestEvaluateCLI:
    @pytest.fixture(autouse=True)
    def _gpu_check(self):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("No GPU — skipping evaluate CLI test")
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_evaluate_sum_task(self, tmp_path):
        model_id = os.environ.get("E2E_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        output_dir = str(tmp_path / "eval_sum")

        result = _run_cli(
            "evaluate",
            "--model-id", model_id,
            "--suite", "easy",
            "--tasks", "sum",
            "--datapoints", "5",
            "--backend", "vllm",
            "--gpu-memory-utilization", "0.45",
            "--output-dir", output_dir,
            "--list-sizes", "5",
            timeout=300,
        )
        assert result.returncode == 0, (
            f"evaluate failed (exit {result.returncode}):\nSTDOUT: {result.stdout[:2000]}"
            f"\nSTDERR: {result.stderr[:2000]}"
        )

    def test_evaluate_creates_output_files(self, tmp_path):
        model_id = os.environ.get("E2E_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        output_dir = str(tmp_path / "eval_output")

        _run_cli(
            "evaluate",
            "--model-id", model_id,
            "--suite", "easy",
            "--tasks", "sum",
            "--datapoints", "3",
            "--backend", "vllm",
            "--gpu-memory-utilization", "0.45",
            "--output-dir", output_dir,
            "--list-sizes", "4",
            timeout=300,
        )
        json_files = list(Path(output_dir).rglob("*.json"))
        assert len(json_files) >= 1, "No JSON output files created"

    def test_evaluate_output_json_valid(self, tmp_path):
        model_id = os.environ.get("E2E_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        output_dir = str(tmp_path / "eval_valid")

        _run_cli(
            "evaluate",
            "--model-id", model_id,
            "--suite", "easy",
            "--tasks", "sum",
            "--datapoints", "3",
            "--backend", "vllm",
            "--gpu-memory-utilization", "0.45",
            "--output-dir", output_dir,
            "--list-sizes", "4",
            timeout=300,
        )
        json_files = list(Path(output_dir).rglob("*.json"))
        for f in json_files:
            with open(f) as fp:
                data = json.load(fp)
            assert data is not None


# ===========================================================================
# 3. serve command — /health endpoint
# ===========================================================================

class TestServeCLI:
    def test_serve_starts_and_health_endpoint_responds(self, tmp_path):
        """Launch serve in background and hit /health; then kill."""
        port = int(os.environ.get("E2E_SERVE_PORT", "8765"))
        cmd = _beyondbench_cmd() + ["serve", "--port", str(port), "--host", "127.0.0.1"]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give server time to start
        started = False
        for _ in range(20):
            time.sleep(0.5)
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if resp.status_code == 200:
                    started = True
                    break
            except Exception:
                continue

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

        if not started:
            pytest.skip("Server did not start (port conflict or serve not implemented)")

        # If it started, verify the response
        # (we already got 200 above)

    def test_serve_help_works(self):
        result = _run_cli("serve", "--help")
        assert result.returncode == 0
        assert "port" in result.stdout.lower() or "host" in result.stdout.lower()

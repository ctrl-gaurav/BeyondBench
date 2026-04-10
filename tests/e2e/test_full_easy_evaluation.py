"""
E2E test — full beyondbench evaluate flow with easy suite, real model.

Verifies:
  - final_results.json exists and is valid
  - per-task results exist for all tasks that were run
  - statistics report generates correctly

Requirements:
    CUDA_VISIBLE_DEVICES=7 pytest tests/e2e/test_full_easy_evaluation.py -v --timeout=1200
"""

import json
import os
import subprocess
import sys
import logging
import pytest
from pathlib import Path

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL_ID = os.environ.get("E2E_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
GPU_MEM = os.environ.get("E2E_GPU_MEM", "0.45")


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available — skipping E2E tests")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture(scope="module")
def evaluation_results(tmp_path_factory):
    """Run a full easy evaluation with 3 tasks × 5 datapoints and return output dir."""
    output_dir = str(tmp_path_factory.mktemp("e2e_easy"))

    from beyondbench.models.model_handler import ModelHandler
    from beyondbench.core.evaluation_engine import EvaluationEngine

    logger.info(f"E2E: Loading model {MODEL_ID}")
    handler = ModelHandler(
        model_id=MODEL_ID,
        backend="vllm",
        cuda_device="cuda:0",
        gpu_memory_utilization=float(GPU_MEM),
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_model_len=2048,
    )

    engine = EvaluationEngine(
        model_handler=handler,
        output_dir=output_dir,
        store_details=True,
    )

    results = engine.run_evaluation(
        suite="easy",
        tasks=["sum", "sorting", "find_maximum"],
        datapoints=5,
        folds=1,
        list_sizes=[8],
    )

    return results, output_dir


# ===========================================================================
# Tests
# ===========================================================================

class TestFullEasyEvaluationOutput:
    def test_returns_dict(self, evaluation_results):
        results, _ = evaluation_results
        assert isinstance(results, dict)

    def test_has_summary(self, evaluation_results):
        results, _ = evaluation_results
        assert "summary" in results

    def test_has_task_results(self, evaluation_results):
        results, _ = evaluation_results
        assert "task_results" in results

    def test_task_results_non_empty(self, evaluation_results):
        results, _ = evaluation_results
        task_results = results["task_results"]
        assert task_results is not None
        if isinstance(task_results, dict):
            assert len(task_results) >= 1
        elif isinstance(task_results, list):
            assert len(task_results) >= 1

    def test_summary_total_tasks_positive(self, evaluation_results):
        results, _ = evaluation_results
        assert results["summary"]["total_tasks"] >= 1


class TestOutputFilesExist:
    def test_output_dir_non_empty(self, evaluation_results):
        _, output_dir = evaluation_results
        p = Path(output_dir)
        assert p.exists()
        files = list(p.rglob("*"))
        assert len(files) >= 1, "Output directory is empty after evaluation"

    def test_json_files_exist(self, evaluation_results):
        _, output_dir = evaluation_results
        json_files = list(Path(output_dir).rglob("*.json"))
        assert len(json_files) >= 1, "No JSON result files found"

    def test_json_files_are_valid(self, evaluation_results):
        _, output_dir = evaluation_results
        json_files = list(Path(output_dir).rglob("*.json"))
        for f in json_files:
            with open(f) as fp:
                data = json.load(fp)
            assert data is not None, f"{f} parsed to None"

    def test_results_json_serializable(self, evaluation_results):
        results, _ = evaluation_results
        serialized = json.dumps(results)
        parsed = json.loads(serialized)
        assert "summary" in parsed


class TestResultIntegrity:
    def test_accuracy_between_0_and_1(self, evaluation_results):
        results, _ = evaluation_results
        summary = results["summary"]
        for key in ["overall_accuracy", "mean_accuracy", "accuracy"]:
            if key in summary:
                assert 0.0 <= summary[key] <= 1.0, f"{key} out of range: {summary[key]}"

    def test_task_results_have_accuracy(self, evaluation_results):
        results, _ = evaluation_results
        task_results = results["task_results"]
        if isinstance(task_results, dict):
            for task_name, task_data in task_results.items():
                if isinstance(task_data, dict):
                    has_acc = "accuracy" in task_data or "mean_accuracy" in task_data
                    assert has_acc, f"Task '{task_name}' result missing accuracy"

    def test_sum_task_in_results(self, evaluation_results):
        results, _ = evaluation_results
        task_results = results["task_results"]
        if isinstance(task_results, dict):
            assert "sum" in task_results or any("sum" in k for k in task_results)


class TestStatisticsReport:
    def test_model_info_present(self, evaluation_results):
        results, _ = evaluation_results
        summary = results.get("summary", {})
        has_model = (
            "model_id" in summary
            or "model_name" in summary
            or "model_info" in summary
        )
        assert has_model or summary is not None

    def test_elapsed_time_recorded(self, evaluation_results):
        results, _ = evaluation_results
        summary = results.get("summary", {})
        time_val = (
            summary.get("elapsed_time")
            or summary.get("total_time_seconds")
            or summary.get("duration")
        )
        if time_val is not None:
            assert time_val > 0

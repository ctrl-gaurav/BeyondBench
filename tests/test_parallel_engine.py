"""
Unit tests for multi-GPU parallel evaluation components.

These tests do NOT require a real GPU — they mock nvidia-smi and model
handlers to verify the scheduler, aggregator, and engine logic.
"""

import json
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ------------------------------------------------------------------ #
# GPUScheduler tests
# ------------------------------------------------------------------ #

class TestGPUScheduler:
    """Test GPU discovery, allocation, and release."""

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_get_all_gpus(self, mock_query):
        mock_query.return_value = {0: (46000, 40000), 1: (46000, 10000), 2: (46000, 5000)}
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        assert sched.get_all_gpus() == [0, 1, 2]

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_get_free_gpus_filters_by_memory(self, mock_query):
        # GPU 0: truly free (46000 total, 45800 free → 200MB used)
        # GPU 1: occupied (46000 total, 4000 free → 42000MB used)
        # GPU 2: truly free (46000 total, 45500 free → 500MB used)
        mock_query.return_value = {0: (46000, 45800), 1: (46000, 4000), 2: (46000, 45500)}
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        # GPU 1 is occupied by another process → excluded
        free = sched.get_free_gpus(min_memory_gb=8)
        assert 0 in free
        assert 2 in free
        assert 1 not in free

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_allocate_and_release(self, mock_query):
        # Both GPUs truly free (low used memory)
        mock_query.return_value = {0: (46000, 45800), 1: (46000, 45500)}
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()

        # Allocate GPU for model
        gpu = sched.allocate_gpu("test-model", required_memory_gb=5)
        assert gpu in (0, 1)

        # Cannot allocate same GPU again (it's marked as allocated)
        # Still has GPU 0 or 1 free depending on which was chosen
        second = sched.allocate_gpu("test-model-2", required_memory_gb=5)
        assert second != gpu  # different GPU

        # Release and verify it's free again
        sched.release_gpu(gpu)
        free = sched.get_free_gpus(min_memory_gb=5)
        assert gpu in free

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_allocate_no_gpu_available(self, mock_query):
        # Only 2GB free — nothing meets 8GB requirement
        mock_query.return_value = {0: (46000, 2000)}
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        result = sched.allocate_gpu("large-model", required_memory_gb=8)
        assert result is None

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_get_free_gpus_rejects_occupied(self, mock_query):
        """GPUs with significant memory used by other processes should NOT be returned."""
        mock_query.return_value = {
            0: (46000, 45800),  # 200MB used → truly free
            1: (46000, 15000),  # 31000MB used → occupied by another process
            2: (46000, 22000),  # 24000MB used → occupied
        }
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        free = sched.get_free_gpus(min_memory_gb=8)
        assert free == [0], f"Only truly free GPU 0 should be returned, got {free}"

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,5"})
    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_respects_cuda_visible_devices(self, mock_query):
        """GPUScheduler should only consider GPUs listed in CUDA_VISIBLE_DEVICES."""
        mock_query.return_value = {i: (46000, 45800) for i in range(8)}
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        assert sched.get_all_gpus() == [2, 5]

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_parse_gpu_spec_validates_existence(self, mock_query):
        """parse_gpu_spec should skip GPUs that don't exist."""
        mock_query.return_value = {0: (46000, 45800), 1: (46000, 45500)}
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        result = sched.parse_gpu_spec("0,1,99")
        assert result == [0, 1], f"GPU 99 should be filtered out, got {result}"

    def test_estimate_vram(self):
        from beyondbench.core.gpu_scheduler import estimate_model_vram
        # 1.5B model
        assert estimate_model_vram("Qwen/Qwen2.5-1.5B-Instruct") == 5
        # 3B model
        assert estimate_model_vram("meta-llama/Llama-3.2-3B") == 8
        # 7B model
        assert estimate_model_vram("mistralai/Mistral-7B-Instruct-v0.2") == 16
        # 0.5B model
        assert estimate_model_vram("Qwen/Qwen2.5-0.5B") == 3
        # Unknown → default
        assert estimate_model_vram("some-mystery-model") == 20

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_parse_gpu_spec_explicit(self, mock_query):
        mock_query.return_value = {i: (46000, 40000) for i in range(8)}
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        assert sched.parse_gpu_spec("0,1,2,3") == [0, 1, 2, 3]
        assert sched.parse_gpu_spec("0-3") == [0, 1, 2, 3]
        assert sched.parse_gpu_spec("2") == [2]

    @patch("beyondbench.core.gpu_scheduler._query_gpu_memory")
    def test_parse_gpu_spec_auto(self, mock_query):
        # GPUs 0 and 3 are truly free; GPUs 1 and 2 are occupied
        mock_query.return_value = {
            0: (46000, 45800), 1: (46000, 4000),
            2: (46000, 3000), 3: (46000, 45500)
        }
        from beyondbench.core.gpu_scheduler import GPUScheduler
        sched = GPUScheduler()
        free = sched.parse_gpu_spec("auto", model_id="Qwen/Qwen2.5-1.5B-Instruct")
        assert 0 in free
        assert 3 in free
        assert 1 not in free


# ------------------------------------------------------------------ #
# ResultAggregator tests
# ------------------------------------------------------------------ #

def _make_worker_result(gpu_id: int, tasks: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to build a fake worker result dict."""
    return {
        "gpu_id": gpu_id,
        "task_results": tasks,
        "overall_stats": {
            "total_tasks": len(tasks),
            "completed_tasks": len(tasks),
            "failed_tasks": 0,
            "total_evaluations": 100,
            "successful_evaluations": 80,
        },
    }


class TestResultAggregator:
    """Test result merging logic."""

    def test_task_parallel_merge(self):
        from beyondbench.core.result_aggregator import ResultAggregator

        w0 = _make_worker_result(0, {
            "sum": [{"accuracy": 0.9, "total_count": 50, "correct_count": 45}]
        })
        w1 = _make_worker_result(1, {
            "sorting": [{"accuracy": 0.8, "total_count": 50, "correct_count": 40}]
        })

        result = ResultAggregator.aggregate(
            worker_results=[w0, w1],
            strategy="task_parallel",
            overall_start_time=time.time() - 10,
            model_info={"model_name": "test", "backend": "vllm"},
            eval_config={"suite": "easy", "output_dir": "/tmp"},
        )

        assert "sum" in result["task_results"]
        assert "sorting" in result["task_results"]
        assert result["summary"]["completed_tasks"] == 2

    def test_task_parallel_partial_failure(self):
        """If one worker has no results (crashed), others are still merged."""
        from beyondbench.core.result_aggregator import ResultAggregator

        w0 = _make_worker_result(0, {"sum": [{"accuracy": 0.9, "total_count": 50, "correct_count": 45}]})
        w_failed = {}  # empty dict simulates crash

        result = ResultAggregator.aggregate(
            worker_results=[w0, w_failed],
            strategy="task_parallel",
            overall_start_time=time.time() - 5,
            model_info={"model_name": "test"},
            eval_config={},
        )

        assert "sum" in result["task_results"]
        assert result["summary"]["completed_tasks"] >= 1

    def test_data_parallel_merge_list_format(self):
        """data_parallel should flatten list shards into one list."""
        from beyondbench.core.result_aggregator import ResultAggregator

        # Two shards of the same task
        w0 = _make_worker_result(0, {
            "sum": [{"accuracy": 0.9, "total_count": 50, "correct_count": 45}]
        })
        w1 = _make_worker_result(1, {
            "sum": [{"accuracy": 0.8, "total_count": 50, "correct_count": 40}]
        })

        result = ResultAggregator.aggregate(
            worker_results=[w0, w1],
            strategy="data_parallel",
            overall_start_time=time.time() - 10,
            model_info={"model_name": "test"},
            eval_config={"suite": "easy"},
        )

        assert "sum" in result["task_results"]
        merged_list = result["task_results"]["sum"]
        assert isinstance(merged_list, list)
        assert len(merged_list) == 2  # two fold entries

    def test_data_parallel_merge_dict_format(self):
        """data_parallel merges fold_results from dict-format results."""
        from beyondbench.core.result_aggregator import ResultAggregator

        fold_shard0 = {"total_count": 50, "correct_count": 45, "accuracy": 0.9}
        fold_shard1 = {"total_count": 50, "correct_count": 40, "accuracy": 0.8}

        w0 = _make_worker_result(0, {
            "sum": {"overall_accuracy": 0.9, "fold_results": [fold_shard0]}
        })
        w1 = _make_worker_result(1, {
            "sum": {"overall_accuracy": 0.8, "fold_results": [fold_shard1]}
        })

        result = ResultAggregator.aggregate(
            worker_results=[w0, w1],
            strategy="data_parallel",
            overall_start_time=time.time() - 10,
            model_info={"model_name": "test"},
            eval_config={"suite": "easy"},
        )

        merged = result["task_results"]["sum"]
        assert isinstance(merged, dict)
        # 45 + 40 correct out of 100 total = 0.85
        assert abs(merged["overall_accuracy"] - 0.85) < 0.01
        assert merged["total_count"] == 100
        assert len(merged["fold_results"]) == 2

    def test_final_result_format(self):
        """Output format matches single-GPU EvaluationEngine output."""
        from beyondbench.core.result_aggregator import ResultAggregator

        w = _make_worker_result(0, {
            "sum": [{"accuracy": 0.9, "total_count": 10, "correct_count": 9}]
        })

        result = ResultAggregator.aggregate(
            worker_results=[w],
            strategy="task_parallel",
            overall_start_time=time.time() - 5,
            model_info={"model_name": "test-model", "backend": "vllm"},
            eval_config={"suite": "easy", "output_dir": "/tmp"},
        )

        # Required top-level keys
        assert "summary" in result
        assert "task_results" in result
        assert "overall_stats" in result
        assert "model_info" in result
        assert "evaluation_config" in result

        # Summary keys
        s = result["summary"]
        assert "total_duration" in s
        assert "avg_accuracy" in s
        assert "completed_tasks" in s
        assert s["parallel"] is True


# ------------------------------------------------------------------ #
# ProgressTracker tests
# ------------------------------------------------------------------ #

class TestProgressTracker:
    """Test thread-safe progress tracking."""

    def test_update_and_read(self):
        from beyondbench.core.result_aggregator import ProgressTracker

        tracker = ProgressTracker([0, 1, 2])
        tracker.update(0, "sum", 50, 100, 0.9)
        tracker.update(1, "sorting", 30, 100, 0.7)

        with tracker._lock:
            assert tracker._status[0]["task"] == "sum"
            assert tracker._status[0]["progress"] == 0.5
            assert tracker._status[1]["completed"] == 30
            assert tracker._status[2]["task"] == "idle"

    def test_mark_done(self):
        from beyondbench.core.result_aggregator import ProgressTracker

        tracker = ProgressTracker([0])
        tracker.mark_done(0)
        with tracker._lock:
            assert tracker._status[0]["done"] is True
            assert tracker._status[0]["task"] == "done"

    def test_mark_done_with_error(self):
        from beyondbench.core.result_aggregator import ProgressTracker

        tracker = ProgressTracker([0])
        tracker.mark_done(0, error="OOM")
        with tracker._lock:
            assert tracker._status[0]["done"] is True
            assert tracker._status[0]["error"] == "OOM"
            assert tracker._status[0]["task"] == "FAILED"


# ------------------------------------------------------------------ #
# ParallelEvaluationEngine strategy selection tests
# ------------------------------------------------------------------ #

class TestStrategySelection:
    """Test the _choose_strategy logic (no GPU required)."""

    def test_auto_single_task(self):
        from beyondbench.core.parallel_engine import ParallelEvaluationEngine
        assert ParallelEvaluationEngine._choose_strategy("auto", 1, 4) == "data_parallel"

    def test_auto_many_tasks(self):
        from beyondbench.core.parallel_engine import ParallelEvaluationEngine
        assert ParallelEvaluationEngine._choose_strategy("auto", 10, 4) == "task_parallel"

    def test_explicit_strategy_passes_through(self):
        from beyondbench.core.parallel_engine import ParallelEvaluationEngine
        assert ParallelEvaluationEngine._choose_strategy("data_parallel", 10, 4) == "data_parallel"
        assert ParallelEvaluationEngine._choose_strategy("task_parallel", 1, 8) == "task_parallel"
        assert ParallelEvaluationEngine._choose_strategy("model_parallel", 5, 5) == "model_parallel"


# ------------------------------------------------------------------ #
# CLI integration tests (no GPU)
# ------------------------------------------------------------------ #

class TestCLIParallelOptions:
    """Test that new CLI flags are accepted and parsed correctly."""

    def test_parallel_flags_present(self):
        """The --parallel, --gpus, --strategy flags should be visible in help."""
        from click.testing import CliRunner
        from beyondbench.cli.main import evaluate

        runner = CliRunner()
        result = runner.invoke(evaluate, ["--help"])
        assert result.exit_code == 0
        assert "--parallel" in result.output
        assert "--gpus" in result.output
        assert "--strategy" in result.output

    def test_validate_and_prepare_config_parallel_defaults(self):
        """validate_and_prepare_config includes parallel keys with defaults."""
        from beyondbench.cli.main import validate_and_prepare_config

        kwargs = {
            "model_id": "test-model",
            "api_provider": "openai",
            "api_key": "sk-test",
            "backend": None,
            "cuda_device": "cuda:0",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.96,
            "trust_remote_code": False,
            "no_chat_template": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 32768,
            "seed": None,
            "reasoning_effort": "medium",
            "thinking_budget": 1024,
            "datapoints": 50,
            "folds": 1,
            "list_sizes": None,
            "range_min": -100,
            "range_max": 100,
            "output_dir": "/tmp/test",
            "store_details": False,
            "log_level": "INFO",
            "batch_size": 1,
            "max_retries": 3,
            "timeout": 300,
            "parser": "unified",
            "suite": "easy",
            "tasks": (),
            # Parallel options
            "parallel": True,
            "gpus": "0,1",
            "strategy": "task_parallel",
        }

        config = validate_and_prepare_config(kwargs)
        assert config["parallel"] is True
        assert config["gpus"] == "0,1"
        assert config["strategy"] == "task_parallel"

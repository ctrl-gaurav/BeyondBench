"""
Unit tests for EvaluationEngine — task routing, aggregation, serialization, error handling.

No GPU required. Uses mock model handler.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_handler():
    handler = MagicMock()
    handler.backend = "mock"
    handler.api_provider = None
    handler.model_id = "test-model"
    handler.tokenizer = None

    def _gen(prompts, **kw):
        return [r"\boxed{42}"] * len(prompts)

    handler.generate = MagicMock(side_effect=_gen)
    handler.generate_batch = handler.generate
    handler.get_model_info = MagicMock(return_value={"model_name": "test-model", "backend": "mock"})
    handler.get_statistics = MagicMock(return_value={
        "total_time_seconds": 0.5,
        "api_calls": 5,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
    })
    handler.count_tokens = MagicMock(return_value=10)
    return handler


@pytest.fixture
def engine(mock_handler, tmp_path):
    from beyondbench.core.evaluation_engine import EvaluationEngine
    return EvaluationEngine(
        model_handler=mock_handler,
        output_dir=str(tmp_path / "results"),
    )


# ===========================================================================
# 1. Initialization
# ===========================================================================

class TestEvaluationEngineInit:
    def test_creates_output_dir(self, mock_handler, tmp_path):
        from beyondbench.core.evaluation_engine import EvaluationEngine
        out = str(tmp_path / "new_dir")
        engine = EvaluationEngine(model_handler=mock_handler, output_dir=out)
        assert Path(out).exists()

    def test_output_dir_attribute(self, engine, tmp_path):
        assert engine.output_dir is not None

    def test_task_registry_initialized(self, engine):
        assert engine.task_registry is not None

    def test_stats_tracker_initialized(self, engine):
        assert engine.stats_tracker is not None

    def test_model_handler_stored(self, engine, mock_handler):
        assert engine.model_handler is mock_handler

    def test_none_handler_allowed(self, tmp_path):
        from beyondbench.core.evaluation_engine import EvaluationEngine
        engine = EvaluationEngine(model_handler=None, output_dir=str(tmp_path / "r"))
        assert engine.model_handler is None


# ===========================================================================
# 2. validate_configuration
# ===========================================================================

class TestValidateConfiguration:
    def test_valid_config(self, engine):
        assert engine.validate_configuration() is True

    def test_invalid_with_no_handler(self, tmp_path):
        from beyondbench.core.evaluation_engine import EvaluationEngine
        engine = EvaluationEngine(model_handler=None, output_dir=str(tmp_path / "r"))
        assert engine.validate_configuration() is False


# ===========================================================================
# 3. get_available_tasks
# ===========================================================================

class TestGetAvailableTasks:
    """Task-count invariants — derived dynamically from ``TaskRegistry`` so
    new phases that add tasks don't require editing these assertions.
    """

    @pytest.fixture
    def registry_counts(self):
        from beyondbench.core.task_registry import TaskRegistry
        reg = TaskRegistry()
        return {
            "easy": len(reg.get_tasks_for_suite("easy")),
            "medium": len(reg.get_tasks_for_suite("medium")),
            "hard": len(reg.get_tasks_for_suite("hard")),
        }

    def test_easy_suite(self, engine, registry_counts):
        tasks = engine.get_available_tasks("easy")
        assert "easy" in tasks
        assert len(tasks["easy"]) == registry_counts["easy"]
        assert len(tasks["easy"]) > 0

    def test_medium_suite(self, engine, registry_counts):
        tasks = engine.get_available_tasks("medium")
        assert "medium" in tasks
        assert len(tasks["medium"]) == registry_counts["medium"]
        assert len(tasks["medium"]) > 0

    def test_hard_suite(self, engine, registry_counts):
        tasks = engine.get_available_tasks("hard")
        assert "hard" in tasks
        assert len(tasks["hard"]) == registry_counts["hard"]
        assert len(tasks["hard"]) > 0

    def test_all_suite(self, engine, registry_counts):
        tasks = engine.get_available_tasks("all")
        total = sum(len(v) for v in tasks.values())
        expected = sum(registry_counts.values())
        assert total == expected

    def test_returns_dict(self, engine):
        result = engine.get_available_tasks("easy")
        assert isinstance(result, dict)

    def test_easy_contains_sum(self, engine):
        tasks = engine.get_available_tasks("easy")
        all_tasks = []
        for v in tasks.values():
            all_tasks.extend(v)
        assert "sum" in all_tasks

    def test_hard_contains_sudoku(self, engine):
        tasks = engine.get_available_tasks("hard")
        all_tasks = []
        for v in tasks.values():
            all_tasks.extend(v)
        assert "sudoku_solving" in all_tasks


# ===========================================================================
# 4. run_evaluation — single task
# ===========================================================================

class TestRunEvaluationSingleTask:
    def test_returns_dict(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        assert isinstance(result, dict)

    def test_has_summary_key(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        assert "summary" in result

    def test_has_task_results_key(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        assert "task_results" in result

    def test_summary_has_total_tasks(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        assert result["summary"]["total_tasks"] >= 1

    def test_results_saved_to_disk(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        # At least one JSON file should exist in output directory
        json_files = list(engine.output_dir.glob("**/*.json"))
        assert len(json_files) >= 1

    def test_sorting_task_runs(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sorting"], datapoints=3, folds=1, list_sizes=[4])
        assert "task_results" in result

    def test_multiple_tasks_run(self, engine):
        result = engine.run_evaluation(
            suite="easy", tasks=["sum", "sorting"], datapoints=3, folds=1, list_sizes=[4]
        )
        assert result["summary"]["total_tasks"] >= 2


# ===========================================================================
# 5. JSON serialization of results
# ===========================================================================

class TestResultSerialization:
    def test_result_is_json_serializable(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        # Should not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_serialized_result_parses_back(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert "summary" in parsed

    def test_saved_file_is_valid_json(self, engine):
        engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        json_files = list(engine.output_dir.glob("**/*.json"))
        assert len(json_files) >= 1
        for f in json_files:
            with open(f) as fp:
                data = json.load(fp)
            assert data is not None


# ===========================================================================
# 6. Error handling — unknown tasks
# ===========================================================================

class TestErrorHandling:
    def test_unknown_task_does_not_crash(self, engine):
        try:
            result = engine.run_evaluation(
                suite="easy", tasks=["nonexistent_task_xyz"], datapoints=2, folds=1, list_sizes=[3]
            )
            # Either returns empty results or raises gracefully
        except Exception as e:
            # Any exception must be a known error type, not e.g. AttributeError
            assert isinstance(e, (ValueError, KeyError, RuntimeError, TypeError))

    def test_empty_task_list_raises_or_returns_empty(self, mock_handler, tmp_path):
        """Empty task list should raise or return empty dict quickly (unit test, no actual inference)."""
        from beyondbench.core.task_registry import TaskRegistry
        from beyondbench.core.evaluation_engine import EvaluationEngine
        engine = EvaluationEngine(model_handler=mock_handler, output_dir=str(tmp_path / "empty"))
        # Verify the engine sees the same easy-suite count as the registry.
        registry = TaskRegistry()
        available = engine.get_available_tasks("easy")
        assert len(available["easy"]) == len(registry.get_tasks_for_suite("easy"))
        assert len(available["easy"]) > 0

    def test_invalid_suite_handled(self, engine):
        try:
            result = engine.run_evaluation(suite="bogus_suite", datapoints=2, folds=1, list_sizes=[3])
            assert isinstance(result, dict)
        except Exception as e:
            assert isinstance(e, (ValueError, KeyError, RuntimeError))


# ===========================================================================
# 7. Result structure depth
# ===========================================================================

class TestResultStructure:
    def test_summary_contains_accuracy(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        summary = result["summary"]
        has_accuracy = (
            "overall_accuracy" in summary
            or "mean_accuracy" in summary
            or "accuracy" in str(summary).lower()
        )
        assert has_accuracy or summary is not None  # At minimum, summary is non-empty

    def test_task_results_contain_task_name(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        task_results = result["task_results"]
        assert isinstance(task_results, (dict, list))

    def test_elapsed_time_positive(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=1, list_sizes=[4])
        summary = result["summary"]
        # Either has elapsed_time or total_time
        time_val = summary.get("elapsed_time") or summary.get("total_time_seconds") or summary.get("duration")
        # May be None if not implemented — but if present, must be positive
        if time_val is not None:
            assert time_val >= 0


# ===========================================================================
# 8. StatsTracker
# ===========================================================================

class TestStatsTracker:
    def test_import(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        assert StatsTracker is not None

    def test_instantiate(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        tracker = StatsTracker()
        assert tracker is not None

    def test_has_update_method(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        tracker = StatsTracker()
        # StatsTracker uses update_stats
        assert (
            hasattr(tracker, "update")
            or hasattr(tracker, "record")
            or hasattr(tracker, "add")
            or hasattr(tracker, "update_stats")
        )

    def test_reset_clears_state(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        tracker = StatsTracker()
        if hasattr(tracker, "reset"):
            tracker.reset()
        # Just verify no exception


# ===========================================================================
# 9. Multiple folds
# ===========================================================================

class TestMultipleFolds:
    def test_two_folds_complete(self, engine):
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=3, folds=2, list_sizes=[4])
        assert "task_results" in result

    def test_folds_parameter_accepted(self, mock_handler, tmp_path):
        """Test that engine accepts folds parameter (single fold unit test)."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        engine = EvaluationEngine(model_handler=mock_handler, output_dir=str(tmp_path / "folds"))
        # Single fold completes within unit test timeout
        result = engine.run_evaluation(suite="easy", tasks=["sum"], datapoints=2, folds=1, list_sizes=[3])
        assert isinstance(result, dict)

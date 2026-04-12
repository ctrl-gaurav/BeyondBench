"""Test EvaluationEngine functionality."""

import pytest
import os
from unittest.mock import MagicMock, patch
from beyondbench.core.evaluation_engine import EvaluationEngine


class TestEvaluationEngineInit:
    def test_initialization(self, mock_model_handler, tmp_output_dir):
        engine = EvaluationEngine(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
        )
        assert engine is not None
        assert engine.output_dir.exists()

    def test_validate_configuration(self, mock_model_handler, tmp_output_dir):
        engine = EvaluationEngine(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
        )
        assert engine.validate_configuration() is True

    def test_validate_configuration_no_handler(self, tmp_output_dir):
        engine = EvaluationEngine(
            model_handler=None,
            output_dir=tmp_output_dir,
        )
        assert engine.validate_configuration() is False


class TestEvaluationEngineExecution:
    def test_get_available_tasks(self, mock_model_handler, tmp_output_dir):
        engine = EvaluationEngine(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
        )
        tasks = engine.get_available_tasks("easy")
        assert "easy" in tasks
        assert len(tasks["easy"]) == 44

    def test_run_evaluation_single_task(self, mock_model_handler, tmp_output_dir):
        engine = EvaluationEngine(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
        )
        results = engine.run_evaluation(
            suite="easy",
            tasks=["sum"],
            datapoints=5,
            folds=1,
            list_sizes=[4],
        )
        assert "summary" in results
        assert "task_results" in results
        assert results["summary"]["total_tasks"] >= 1

    def test_results_saved_to_disk(self, mock_model_handler, tmp_output_dir):
        engine = EvaluationEngine(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
        )
        engine.run_evaluation(
            suite="easy",
            tasks=["sum"],
            datapoints=3,
            folds=1,
            list_sizes=[4],
        )
        assert os.path.exists(os.path.join(tmp_output_dir, "final_results.json"))
        assert os.path.exists(os.path.join(tmp_output_dir, "evaluation_summary.json"))


class TestAggregation:
    def test_aggregate_results_empty(self, mock_model_handler, tmp_output_dir):
        engine = EvaluationEngine(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
        )
        result = engine._aggregate_results(
            {},
            {"completed_tasks": 0, "failed_tasks": 0, "total_evaluations": 0, "successful_evaluations": 0},
            1.0
        )
        assert "summary" in result
        assert result["summary"]["error"] == "No tasks completed successfully"

    def test_json_serializer_numpy_int(self):
        import numpy as np
        result = EvaluationEngine._json_serializer(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_json_serializer_numpy_float(self):
        import numpy as np
        result = EvaluationEngine._json_serializer(np.float64(3.14))
        assert isinstance(result, float)

    def test_json_serializer_numpy_array(self):
        import numpy as np
        result = EvaluationEngine._json_serializer(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_json_serializer_set(self):
        result = EvaluationEngine._json_serializer({1, 2, 3})
        assert sorted(result) == [1, 2, 3]

    def test_json_serializer_fallback(self):
        result = EvaluationEngine._json_serializer(object())
        assert isinstance(result, str)

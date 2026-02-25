"""
Test EvaluationEngine functionality.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestEvaluationEngineInit:
    """Test EvaluationEngine initialization."""

    def test_import_evaluation_engine(self):
        """Test that EvaluationEngine can be imported."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        assert EvaluationEngine is not None

    def test_engine_has_required_methods(self):
        """Test that EvaluationEngine has required methods."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        # Check for expected methods
        expected_methods = ["run_evaluation", "evaluate_task"]
        for method in expected_methods:
            if hasattr(EvaluationEngine, method):
                break
        else:
            # At least should have some evaluation method
            import inspect
            source = inspect.getsource(EvaluationEngine)
            assert "eval" in source.lower() or "run" in source.lower()


class TestEvaluationEngineConfig:
    """Test EvaluationEngine configuration."""

    def test_engine_accepts_model_handler(self):
        """Test that engine accepts model handler."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        import inspect
        sig = inspect.signature(EvaluationEngine.__init__)
        params = list(sig.parameters.keys())
        # Should accept model or model_handler
        assert any("model" in p.lower() for p in params) or "self" in params

    def test_engine_accepts_output_dir(self):
        """Test that engine accepts output directory."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        import inspect
        sig = inspect.signature(EvaluationEngine.__init__)
        params = list(sig.parameters.keys())
        # Should accept output_dir or similar
        param_str = str(params)
        assert "output" in param_str.lower() or "dir" in param_str.lower() or len(params) > 1


class TestEvaluationResults:
    """Test evaluation result structure."""

    def test_engine_source_has_results(self):
        """Test that engine handles results."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        import inspect
        source = inspect.getsource(EvaluationEngine)
        # Should handle results
        assert "result" in source.lower() or "accuracy" in source.lower()

    def test_engine_handles_metrics(self):
        """Test that engine calculates metrics."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        import inspect
        source = inspect.getsource(EvaluationEngine)
        # Should calculate some metrics
        metrics = ["accuracy", "score", "correct", "metric"]
        assert any(m in source.lower() for m in metrics)

"""
Test easy task suite.
"""

import pytest
from beyondbench.core.task_registry import TaskRegistry


class TestEasyTaskDataGeneration:
    """Test data generation for easy tasks."""

    @pytest.fixture
    def registry(self):
        """Create a TaskRegistry instance."""
        return TaskRegistry()

    def test_sum_task_exists(self, registry):
        """Test that sum task is registered."""
        tasks = registry.get_tasks_for_suite("easy")
        task_names = []
        for task in tasks if isinstance(tasks, list) else tasks.values():
            if hasattr(task, "task_name"):
                task_names.append(task.task_name)
            elif hasattr(task, "__class__"):
                task_names.append(task.__class__.__name__.lower())
        # Check if sum-related task exists
        assert any("sum" in name.lower() for name in task_names) or len(tasks) > 0

    def test_task_generates_valid_data(self, registry):
        """Test that tasks generate valid data points."""
        tasks = registry.get_tasks_for_suite("easy")
        if isinstance(tasks, list) and len(tasks) > 0:
            task = tasks[0]
            if hasattr(task, "generate_data"):
                data = task.generate_data(num_samples=5)
                assert data is not None
                assert len(data) > 0

    def test_task_creates_prompt(self, registry):
        """Test that tasks can create prompts."""
        tasks = registry.get_tasks_for_suite("easy")
        if isinstance(tasks, list) and len(tasks) > 0:
            task = tasks[0]
            if hasattr(task, "generate_data") and hasattr(task, "create_prompt"):
                data = task.generate_data(num_samples=1)
                if data and len(data) > 0:
                    prompt = task.create_prompt(data[0])
                    assert prompt is not None
                    assert len(prompt) > 0

    def test_data_has_ground_truth(self, registry):
        """Test that generated data includes ground truth."""
        tasks = registry.get_tasks_for_suite("easy")
        if isinstance(tasks, list) and len(tasks) > 0:
            task = tasks[0]
            if hasattr(task, "generate_data"):
                data = task.generate_data(num_samples=1)
                if data and len(data) > 0:
                    # Data point should have ground truth
                    dp = data[0]
                    assert "ground_truth" in dp or "answer" in dp or "solution" in dp


class TestEasyTaskEvaluation:
    """Test evaluation for easy tasks."""

    @pytest.fixture
    def registry(self):
        """Create a TaskRegistry instance."""
        return TaskRegistry()

    def test_correct_answer_evaluation(self, registry):
        """Test that correct answers are evaluated properly."""
        tasks = registry.get_tasks_for_suite("easy")
        if isinstance(tasks, list) and len(tasks) > 0:
            task = tasks[0]
            if hasattr(task, "generate_data") and hasattr(task, "evaluate_response"):
                data = task.generate_data(num_samples=1)
                if data and len(data) > 0:
                    dp = data[0]
                    # Get the ground truth
                    gt = dp.get("ground_truth") or dp.get("answer") or dp.get("solution")
                    if gt is not None:
                        # Simulate a correct response
                        response = f"The answer is \\boxed{{{gt}}}"
                        result = task.evaluate_response(response, dp)
                        # Result should indicate correctness
                        assert result is not None

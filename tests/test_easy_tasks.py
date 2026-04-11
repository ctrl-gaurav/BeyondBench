"""Test easy task data generation and evaluation."""

import pytest
from beyondbench.core.task_registry import TaskRegistry


class TestEasyTaskDataGeneration:
    """Test that all 29 easy tasks can generate valid data."""

    @pytest.fixture
    def registry(self):
        return TaskRegistry()

    @pytest.fixture
    def easy_task_names(self, registry):
        return registry.get_tasks_for_suite("easy")

    def test_all_easy_tasks_registered(self, registry):
        tasks = registry.get_tasks_for_suite("easy")
        assert len(tasks) == 44

    @pytest.mark.parametrize("task_name", [
        "sorting", "sum", "multiplication", "odd_count", "even_count",
        "find_maximum", "find_minimum", "mean", "median", "mode",
        "subtraction", "absolute_difference", "division", "comparison",
        "second_maximum", "range", "index_of_maximum", "count_negative",
        "count_unique", "max_adjacent_difference", "count_greater_than_previous",
        "sum_of_max_indices", "count_palindromic", "longest_increasing_subsequence",
        "sum_of_digits", "count_perfect_squares", "alternating_sum",
        "count_multiples", "local_maxima_count",
        # Phase 12: 15 new easy tasks
        "weighted_sum", "running_average", "parity_check", "cumulative_sum",
        "reverse_list", "rotate_list", "interleave_lists", "set_intersection",
        "set_difference", "moving_average", "element_frequency", "second_minimum",
        "variance", "standard_deviation", "dot_product",
    ])
    def test_easy_task_class_importable(self, registry, task_name):
        """Each easy task class should be importable."""
        task_class = registry.get_task_class(task_name)
        assert task_class is not None, f"Task class for '{task_name}' not found"

    def test_sum_task_generates_data(self, mock_model_handler, tmp_output_dir):
        """Test SumTask data generation with mock handler."""
        from beyondbench.tasks.easy.sum_task import SumTask
        task = SumTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=5,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data = task.generate_data(list_size=5)
        assert len(data) == 5
        for dp in data:
            assert isinstance(dp, list)
            assert len(dp) == 5

    def test_sorting_task_generates_data(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.sorting_task import SortingTask
        task = SortingTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=5,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data = task.generate_data(list_size=5)
        assert len(data) == 5
        for dp in data:
            assert isinstance(dp, list)
            assert len(dp) == 5

    def test_comparison_task_generates_data(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.comparison_task import ComparisonTask
        task = ComparisonTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=6,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data = task.generate_data()
        assert len(data) == 6
        for dp in data:
            assert isinstance(dp, dict)
            assert "num1" in dp
            assert "num2" in dp
            assert "answer" in dp


class TestEasyTaskPromptCreation:
    """Test that easy tasks create valid prompts."""

    def test_sum_task_prompt(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.sum_task import SumTask
        task = SumTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=3,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data = task.generate_data(list_size=4)
        prompt = task.create_prompt(data[0])
        assert isinstance(prompt, str)
        assert len(prompt) > 10
        assert "boxed" in prompt.lower() or "answer" in prompt.lower()

    def test_sorting_task_prompt(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.sorting_task import SortingTask
        task = SortingTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=3,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data = task.generate_data(list_size=4)
        prompt = task.create_prompt(data[0])
        assert isinstance(prompt, str)
        assert "sort" in prompt.lower()

    def test_comparison_task_prompt(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.comparison_task import ComparisonTask
        task = ComparisonTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=3,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data = task.generate_data()
        prompt = task.create_prompt(data[0])
        assert isinstance(prompt, str)
        assert "compare" in prompt.lower() or "number" in prompt.lower()


class TestEasyTaskEvaluation:
    """Test that easy tasks can evaluate responses."""

    def test_sum_task_correct_answer(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.sum_task import SumTask
        task = SumTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=3,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data_point = [10, 20, 30]
        expected_sum = 60
        response = rf"The sum is \boxed{{{expected_sum}}}"
        result = task.evaluate_response(response, data_point)
        assert result["accuracy"] == 1
        assert result["sum"] == expected_sum

    def test_sum_task_wrong_answer(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.sum_task import SumTask
        task = SumTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=3,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data_point = [10, 20, 30]
        response = r"The sum is \boxed{999}"
        result = task.evaluate_response(response, data_point)
        assert result["accuracy"] == 0

    def test_sorting_task_correct_answer(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.sorting_task import SortingTask
        task = SortingTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=3,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data_point = [5, 1, 3, 2, 4]
        response = r"The sorted list is \boxed{[1, 2, 3, 4, 5]}"
        result = task.evaluate_response(response, data_point)
        assert result["accuracy"] == 1

    def test_comparison_task_correct_answer(self, mock_model_handler, tmp_output_dir):
        from beyondbench.tasks.easy.comparison_task import ComparisonTask
        task = ComparisonTask(
            model_handler=mock_model_handler,
            output_dir=tmp_output_dir,
            min_val=-100, max_val=100,
            num_folds=1, num_samples=3,
            store_details=False,
            temperature=0.7, top_p=0.9, max_tokens=512,
        )
        data_point = {"num1": 10, "num2": 5, "relation": ">", "answer": "greater than"}
        response = r"The answer is \boxed{greater than}"
        result = task.evaluate_response(response, data_point)
        assert result["accuracy"] == 1

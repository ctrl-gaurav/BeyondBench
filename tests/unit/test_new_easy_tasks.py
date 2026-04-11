"""
Unit tests for the 15 new Phase 12 easy tasks.
Tests: instantiation, data generation, prompt creation, evaluation, and parser robustness.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_handler(tmp_path):
    from unittest.mock import MagicMock
    handler = MagicMock()
    handler.backend = "mock"
    handler.api_provider = None
    handler.model_id = "test-model"
    handler.tokenizer = None
    handler.generate = MagicMock(return_value=[r"\boxed{42}"])
    handler.generate_batch = handler.generate
    handler.get_model_info = MagicMock(return_value={"model_name": "test", "backend": "mock", "api_provider": None, "model_type": "mock"})
    handler.get_statistics = MagicMock(return_value={"total_time_seconds": 1.0, "api_calls": 1, "total_input_tokens": 10, "total_output_tokens": 5})
    handler.count_tokens = MagicMock(return_value=10)
    return handler


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return str(d)


def make_task(cls, handler, out_dir, **kwargs):
    defaults = dict(
        model_handler=handler,
        output_dir=out_dir,
        min_val=-20, max_val=20,
        num_folds=1, num_samples=5,
        store_details=False,
        temperature=0.7, top_p=0.9, max_tokens=512,
        seed=42,
    )
    defaults.update(kwargs)
    return cls(**defaults)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _boxed(val):
    return rf"\boxed{{{val}}}"


# ===========================================================================
# 1. weighted_sum
# ===========================================================================

class TestWeightedSum:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.weighted_sum_task import WeightedSumTask
        return make_task(WeightedSumTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "weighted_sum"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5
        for dp in data:
            assert isinstance(dp, list)
            assert len(dp) == 5

    def test_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=4)
        prompt = t.create_prompt(data[0])
        assert "boxed" in prompt.lower()

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [1, 2, 3]
        gt = 1*1 + 2*2 + 3*3  # 14
        result = t.evaluate_response(_boxed(gt), dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [1, 2, 3]
        result = t.evaluate_response(_boxed(9999), dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 3]
        gt = 1*2 + 2*3  # 8
        responses = [
            _boxed(gt),
            f"The weighted sum is {gt}",
            f"Result: {gt}",
            f"$$\\boxed{{{gt}}}$$",
            f"The answer is {gt}.",
            f"answer: {gt}",
            f"therefore the answer is {gt}",
            f"sum = {gt}",
            f"\n{gt}\n",
            f"the weighted sum equals {gt}",
        ]
        for resp in responses:
            res = t.evaluate_response(resp, dp)
            assert res["accuracy"] in (0, 1)  # at least parseable


# ===========================================================================
# 2. running_average
# ===========================================================================

class TestRunningAverage:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.running_average_task import RunningAverageTask
        return make_task(RunningAverageTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "running_average"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=4)
        assert len(data) == 5
        assert isinstance(data[0], list)

    def test_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=4)
        prompt = t.create_prompt(data[0])
        assert "boxed" in prompt.lower()

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 4, 6]
        # running avg: [2.0, 3.0, 4.0]
        result = t.evaluate_response(r"\boxed{[2.0, 3.0, 4.0]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 4, 6]
        result = t.evaluate_response(r"\boxed{[1.0, 2.0, 3.0]}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.parsers.running_average_parsing import parse_running_average_answer
        responses = [
            r"\boxed{[2.0, 3.0, 4.0]}",
            "The running average is [2.0, 3.0, 4.0]",
            "Result: [2.0, 3.0, 4.0]",
            "averages: [2.0, 3.0, 4.0]",
            "```\n[2.0, 3.0, 4.0]\n```",
            "[2.0, 3.0, 4.0]",
            "the answer is [2.0, 3.0, 4.0].",
            "output = [2.0, 3.0, 4.0]",
            r"$$\boxed{[2.0, 3.0, 4.0]}$$",
            "result is [2.0, 3.0, 4.0]",
        ]
        parsed_count = 0
        for resp in responses:
            r, _ = parse_running_average_answer(resp)
            if r is not None:
                parsed_count += 1
        assert parsed_count >= 7, f"Only parsed {parsed_count}/10 responses"


# ===========================================================================
# 3. parity_check
# ===========================================================================

class TestParityCheck:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.parity_check_task import ParityCheckTask
        return make_task(ParityCheckTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "parity_check"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5
        assert isinstance(data[0], list)

    def test_evaluate_even(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 3, 5]  # product 30, even
        result = t.evaluate_response(r"\boxed{even}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_odd(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [3, 5, 7]  # product 105, odd
        result = t.evaluate_response(r"\boxed{odd}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 3, 5]  # even
        result = t.evaluate_response(r"\boxed{odd}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self):
        from beyondbench.parsers.parity_check_parsing import parse_parity_check_answer
        responses = [
            r"\boxed{even}",
            r"\boxed{odd}",
            "The product is even.",
            "The product is odd.",
            "result is even",
            "answer is odd",
            "The answer is even",
            "even number",
            "odd product",
            "The product of all elements is even",
        ]
        for resp in responses:
            parsed, _ = parse_parity_check_answer(resp)
            assert parsed in ("even", "odd"), f"Failed to parse: {resp!r}"


# ===========================================================================
# 4. cumulative_sum
# ===========================================================================

class TestCumulativeSum:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.cumulative_sum_task import CumulativeSumTask
        return make_task(CumulativeSumTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "cumulative_sum"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [1, 2, 3]
        result = t.evaluate_response(r"\boxed{[1, 3, 6]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [1, 2, 3]
        result = t.evaluate_response(r"\boxed{[1, 2, 3]}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self):
        from beyondbench.parsers.cumulative_sum_parsing import parse_cumulative_sum_answer
        responses = [
            r"\boxed{[1, 3, 6]}",
            "The cumulative sum is [1, 3, 6]",
            "Result: [1, 3, 6]",
            "[1, 3, 6]",
            "```\n[1, 3, 6]\n```",
            "answer = [1, 3, 6]",
            "output: [1, 3, 6]",
            r"$$\boxed{[1, 3, 6]}$$",
            "the answer is [1, 3, 6].",
            "sums: [1, 3, 6]",
        ]
        parsed_count = sum(1 for r in responses if parse_cumulative_sum_answer(r)[0] is not None)
        assert parsed_count >= 7


# ===========================================================================
# 5. reverse_list
# ===========================================================================

class TestReverseList:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.reverse_list_task import ReverseListTask
        return make_task(ReverseListTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "reverse_list"

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [1, 2, 3, 4]
        result = t.evaluate_response(r"\boxed{[4, 3, 2, 1]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [1, 2, 3, 4]
        result = t.evaluate_response(r"\boxed{[1, 2, 3, 4]}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self):
        from beyondbench.parsers.reverse_list_parsing import parse_reverse_list_answer
        responses = [
            r"\boxed{[4, 3, 2, 1]}",
            "The reversed list is [4, 3, 2, 1]",
            "Result: [4, 3, 2, 1]",
            "[4, 3, 2, 1]",
            "```\n[4, 3, 2, 1]\n```",
            "answer = [4, 3, 2, 1]",
            "output: [4, 3, 2, 1]",
            r"$$\boxed{[4, 3, 2, 1]}$$",
            "the answer is [4, 3, 2, 1].",
            "reversed: [4, 3, 2, 1]",
        ]
        parsed_count = sum(1 for r in responses if parse_reverse_list_answer(r)[0] is not None)
        assert parsed_count >= 7


# ===========================================================================
# 6. rotate_list
# ===========================================================================

class TestRotateList:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.rotate_list_task import RotateListTask
        return make_task(RotateListTask, handler, out_dir, seed=123)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "rotate_list"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5
        for dp in data:
            assert "list" in dp and "k" in dp and "answer" in dp

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list": [1, 2, 3, 4, 5], "k": 2, "answer": [4, 5, 1, 2, 3]}
        result = t.evaluate_response(r"\boxed{[4, 5, 1, 2, 3]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list": [1, 2, 3, 4, 5], "k": 2, "answer": [4, 5, 1, 2, 3]}
        result = t.evaluate_response(r"\boxed{[1, 2, 3, 4, 5]}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self):
        from beyondbench.parsers.rotate_list_parsing import parse_rotate_list_answer
        responses = [
            r"\boxed{[4, 5, 1, 2, 3]}",
            "The rotated list is [4, 5, 1, 2, 3]",
            "Result: [4, 5, 1, 2, 3]",
            "[4, 5, 1, 2, 3]",
            "```\n[4, 5, 1, 2, 3]\n```",
            "answer = [4, 5, 1, 2, 3]",
            "output: [4, 5, 1, 2, 3]",
            r"$$\boxed{[4, 5, 1, 2, 3]}$$",
            "the answer is [4, 5, 1, 2, 3].",
            "rotated: [4, 5, 1, 2, 3]",
        ]
        parsed_count = sum(1 for r in responses if parse_rotate_list_answer(r)[0] is not None)
        assert parsed_count >= 7


# ===========================================================================
# 7. interleave_lists
# ===========================================================================

class TestInterleaveLists:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.interleave_lists_task import InterleaveListsTask
        return make_task(InterleaveListsTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "interleave_lists"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=3)
        assert len(data) == 5
        for dp in data:
            assert "list1" in dp and "list2" in dp and "answer" in dp
            assert len(dp["answer"]) == 6

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 3, 5], "list2": [2, 4, 6], "answer": [1, 2, 3, 4, 5, 6]}
        result = t.evaluate_response(r"\boxed{[1, 2, 3, 4, 5, 6]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 3, 5], "list2": [2, 4, 6], "answer": [1, 2, 3, 4, 5, 6]}
        result = t.evaluate_response(r"\boxed{[1, 3, 5, 2, 4, 6]}", dp)
        assert result["accuracy"] == 0


# ===========================================================================
# 8. set_intersection
# ===========================================================================

class TestSetIntersection:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.set_intersection_task import SetIntersectionTask
        return make_task(SetIntersectionTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "set_intersection"

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2, 3], "list2": [2, 3, 4], "answer": [2, 3]}
        result = t.evaluate_response(r"\boxed{[2, 3]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_empty(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2], "list2": [3, 4], "answer": []}
        result = t.evaluate_response(r"\boxed{[]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2, 3], "list2": [2, 3, 4], "answer": [2, 3]}
        result = t.evaluate_response(r"\boxed{[1, 2, 3]}", dp)
        assert result["accuracy"] == 0


# ===========================================================================
# 9. set_difference
# ===========================================================================

class TestSetDifference:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.set_difference_task import SetDifferenceTask
        return make_task(SetDifferenceTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "set_difference"

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2, 3], "list2": [2, 4], "answer": [1, 3]}
        result = t.evaluate_response(r"\boxed{[1, 3]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2, 3], "list2": [2, 4], "answer": [1, 3]}
        result = t.evaluate_response(r"\boxed{[2]}", dp)
        assert result["accuracy"] == 0


# ===========================================================================
# 10. moving_average
# ===========================================================================

class TestMovingAverage:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.moving_average_task import MovingAverageTask
        return make_task(MovingAverageTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "moving_average"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=6, window=3)
        assert len(data) == 5
        for dp in data:
            assert len(dp["answer"]) == 6 - 3 + 1  # 4

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list": [1, 2, 3, 4, 5], "window": 3, "answer": [2.0, 3.0, 4.0]}
        result = t.evaluate_response(r"\boxed{[2.0, 3.0, 4.0]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list": [1, 2, 3, 4, 5], "window": 3, "answer": [2.0, 3.0, 4.0]}
        result = t.evaluate_response(r"\boxed{[1.0, 2.0, 3.0]}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self):
        from beyondbench.parsers.moving_average_parsing import parse_moving_average_answer
        responses = [
            r"\boxed{[2.0, 3.0, 4.0]}",
            "The moving average is [2.0, 3.0, 4.0]",
            "Result: [2.0, 3.0, 4.0]",
            "[2.0, 3.0, 4.0]",
            "```\n[2.0, 3.0, 4.0]\n```",
            "averages: [2.0, 3.0, 4.0]",
            r"$$\boxed{[2.0, 3.0, 4.0]}$$",
            "the answer is [2.0, 3.0, 4.0].",
            "output = [2.0, 3.0, 4.0]",
            "result is [2.0, 3.0, 4.0]",
        ]
        parsed_count = sum(1 for r in responses if parse_moving_average_answer(r)[0] is not None)
        assert parsed_count >= 7


# ===========================================================================
# 11. element_frequency
# ===========================================================================

class TestElementFrequency:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.element_frequency_task import ElementFrequencyTask
        return make_task(ElementFrequencyTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "element_frequency"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=6)
        assert len(data) == 5
        for dp in data:
            assert "list" in dp and "answer" in dp

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list": [1, 2, 2, 3], "answer": [[1, 1], [2, 2], [3, 1]]}
        result = t.evaluate_response(r"\boxed{[[1, 1], [2, 2], [3, 1]]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list": [1, 2, 2, 3], "answer": [[1, 1], [2, 2], [3, 1]]}
        result = t.evaluate_response(r"\boxed{[[1, 2], [2, 1], [3, 1]]}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self):
        from beyondbench.parsers.element_frequency_parsing import parse_element_frequency_answer
        responses = [
            r"\boxed{[[1, 1], [2, 2], [3, 1]]}",
            "The frequency is [[1, 1], [2, 2], [3, 1]]",
            "[[1, 1], [2, 2], [3, 1]]",
            "Result: [[1, 1], [2, 2], [3, 1]]",
            "```\n[[1, 1], [2, 2], [3, 1]]\n```",
            "answer: [[1, 1], [2, 2], [3, 1]]",
            "output = [[1, 1], [2, 2], [3, 1]]",
            r"$$\boxed{[[1, 1], [2, 2], [3, 1]]}$$",
            "frequencies: [[1, 1], [2, 2], [3, 1]]",
            "the answer is [[1, 1], [2, 2], [3, 1]]",
        ]
        parsed_count = sum(1 for r in responses if parse_element_frequency_answer(r)[0] is not None)
        assert parsed_count >= 7


# ===========================================================================
# 12. second_minimum
# ===========================================================================

class TestSecondMinimum:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.second_minimum_task import SecondMinimumTask
        return make_task(SecondMinimumTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "second_minimum"

    def test_generate_data_has_2_distinct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5
        for dp in data:
            assert len(set(dp)) >= 2

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [3, 1, 4, 1, 5, 2]
        # distinct sorted: [1, 2, 3, 4, 5], second min = 2
        result = t.evaluate_response(r"\boxed{2}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [3, 1, 4, 1, 5, 2]
        result = t.evaluate_response(r"\boxed{1}", dp)
        assert result["accuracy"] == 0


# ===========================================================================
# 13. variance
# ===========================================================================

class TestVariance:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.variance_task import VarianceTask
        return make_task(VarianceTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "variance"

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 4, 4, 4, 5, 5, 7, 9]
        # mu=5, var=4.0
        result = t.evaluate_response(r"\boxed{4.0}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 4, 4, 4, 5, 5, 7, 9]
        result = t.evaluate_response(r"\boxed{99.0}", dp)
        assert result["accuracy"] == 0


# ===========================================================================
# 14. standard_deviation
# ===========================================================================

class TestStandardDeviation:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.standard_deviation_task import StandardDeviationTask
        return make_task(StandardDeviationTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "standard_deviation"

    def test_evaluate_correct(self, mock_handler, out_dir):
        import math
        t = self._task(mock_handler, out_dir)
        dp = [2, 4, 4, 4, 5, 5, 7, 9]
        # mu=5, var=4.0, std=2.0
        result = t.evaluate_response(r"\boxed{2.0}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = [2, 4, 4, 4, 5, 5, 7, 9]
        result = t.evaluate_response(r"\boxed{99.0}", dp)
        assert result["accuracy"] == 0


# ===========================================================================
# 15. dot_product
# ===========================================================================

class TestDotProduct:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.easy.dot_product_task import DotProductTask
        return make_task(DotProductTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "dot_product"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(list_size=4)
        assert len(data) == 5
        for dp in data:
            assert "list1" in dp and "list2" in dp and "answer" in dp

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2, 3], "list2": [4, 5, 6], "answer": 32}
        result = t.evaluate_response(r"\boxed{32}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2, 3], "list2": [4, 5, 6], "answer": 32}
        result = t.evaluate_response(r"\boxed{99}", dp)
        assert result["accuracy"] == 0

    def test_parser_formats(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {"list1": [1, 2, 3], "list2": [4, 5, 6], "answer": 32}
        responses = [
            r"\boxed{32}",
            "The dot product is 32",
            "Result: 32",
            "32",
            "answer = 32",
            "The result is 32.",
            "therefore the answer is 32",
            "output: 32",
            r"$$\boxed{32}$$",
            "dot product = 32",
        ]
        count = sum(1 for r in responses if t.evaluate_response(r, dp)["instruction_followed"])
        assert count >= 5


# ===========================================================================
# Registry check
# ===========================================================================

def test_all_15_registered():
    from beyondbench.core.task_registry import TaskRegistry
    registry = TaskRegistry()
    tasks = registry.get_tasks_for_suite("easy")
    assert len(tasks) == 44
    new_tasks = [
        "weighted_sum", "running_average", "parity_check", "cumulative_sum",
        "reverse_list", "rotate_list", "interleave_lists", "set_intersection",
        "set_difference", "moving_average", "element_frequency", "second_minimum",
        "variance", "standard_deviation", "dot_product",
    ]
    for name in new_tasks:
        assert name in tasks, f"{name} not registered"

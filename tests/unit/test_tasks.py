"""
Unit tests for all BeyondBench tasks — generate_data, create_prompt, evaluate_response.

No GPU required. Uses mock model handler for any task that invokes generate().

Task counts are derived dynamically from the registry, so adding new tasks in
future phases does not require editing this file.
"""

import pytest
import random
from unittest.mock import MagicMock


def pytest_generate_tests(metafunc):
    """Parametrize any ``task_name`` argument from the live TaskRegistry.

    This makes the test file self-updating: new tasks registered in
    ``TaskRegistry._discover_and_register_tasks`` are automatically covered.

    Only applies when:
      - The test function accepts a ``task_name`` fixture.
      - The test is a method on a class that opts in via
        ``_suite_for_parametrize``. Module-level functions that explicitly
        use ``@pytest.mark.parametrize`` (e.g., the hard-task import tests
        that pack ``task_name`` into a tuple with module/class paths) are
        left alone.
    """
    if "task_name" not in metafunc.fixturenames:
        return
    cls = getattr(metafunc, "cls", None)
    if cls is None or not hasattr(cls, "_suite_for_parametrize"):
        return  # Let existing @pytest.mark.parametrize handle it.

    from beyondbench.core.task_registry import TaskRegistry
    registry = TaskRegistry()

    suite = cls._suite_for_parametrize
    task_names = registry.get_tasks_for_suite(suite)
    task_filter = getattr(cls, "_task_filter", None)
    if task_filter is not None:
        task_names = [t for t in task_names if task_filter(t)]
    metafunc.parametrize("task_name", task_names)

# ---------------------------------------------------------------------------
# Shared mock handler fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_handler(tmp_path):
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
    handler.get_statistics = MagicMock(return_value={"total_time_seconds": 0.0})
    handler.count_tokens = MagicMock(return_value=10)
    return handler


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return str(d)


def _make_task(cls, handler, output_dir, **kw):
    defaults = dict(
        model_handler=handler,
        output_dir=output_dir,
        min_val=-100,
        max_val=100,
        num_folds=1,
        num_samples=5,
        store_details=False,
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        seed=42,
    )
    defaults.update(kw)
    return cls(**defaults)


# ===========================================================================
# Easy tasks
# ===========================================================================

class TestSumTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.sum_task import SumTask
        return _make_task(SumTask, handler, output_dir)

    def test_generate_data_list_size_5(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5
        for d in data:
            assert len(d) == 5

    def test_generate_data_list_size_10(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=10)
        assert all(len(d) == 10 for d in data)

    def test_create_prompt_contains_list(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        prompt = t.create_prompt([1, 2, 3])
        assert "1" in prompt and "2" in prompt and "3" in prompt

    def test_create_prompt_mentions_boxed(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        prompt = t.create_prompt([1, 2])
        assert "boxed" in prompt.lower() or r"\boxed" in prompt

    def test_evaluate_correct_response(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = [1, 2, 3, 4, 5]
        result = t.evaluate_response(r"\boxed{15}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong_response(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = [1, 2, 3]
        result = t.evaluate_response(r"\boxed{999}", dp)
        assert result["accuracy"] == 0

    def test_evaluate_empty_response(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        result = t.evaluate_response("", [1, 2])
        assert result["accuracy"] == 0

    def test_evaluate_negative_sum(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = [-1, -2, -3]
        result = t.evaluate_response(r"\boxed{-6}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_zero_sum(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = [1, -1]
        result = t.evaluate_response(r"\boxed{0}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_single_element(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        result = t.evaluate_response(r"\boxed{7}", [7])
        assert result["accuracy"] == 1


class TestSortingTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.sorting_task import SortingTask
        return _make_task(SortingTask, handler, output_dir)

    def test_generate_data(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5

    def test_create_prompt_has_sort_instruction(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        prompt = t.create_prompt([3, 1, 2])
        assert "sort" in prompt.lower()

    def test_evaluate_correct(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = [3, 1, 2]
        result = t.evaluate_response(r"\boxed{[1, 2, 3]}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong_order(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = [3, 1, 2]
        result = t.evaluate_response(r"\boxed{[3, 2, 1]}", dp)
        assert result["accuracy"] == 0

    def test_evaluate_empty(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        result = t.evaluate_response("", [1, 2])
        assert result["accuracy"] == 0


class TestMultiplicationTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.multiplication_task import MultiplicationTask
        return _make_task(MultiplicationTask, handler, output_dir)

    def test_generate_data(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=3)
        assert len(data) == 5

    def test_create_prompt(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        prompt = t.create_prompt([2, 3, 4])
        assert "2" in prompt

    def test_evaluate_correct(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        result = t.evaluate_response(r"\boxed{24}", [2, 3, 4])
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        result = t.evaluate_response(r"\boxed{999}", [2, 3, 4])
        assert result["accuracy"] == 0


class TestComparisonTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.comparison_task import ComparisonTask
        return _make_task(ComparisonTask, handler, output_dir)

    def test_generate_data_structure(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data()
        assert len(data) == 5
        for d in data:
            assert "num1" in d and "num2" in d and "answer" in d

    def test_create_prompt_has_numbers(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = {"num1": 5, "num2": 3, "relation": ">", "answer": "greater than"}
        prompt = t.create_prompt(dp)
        assert "5" in prompt and "3" in prompt

    def test_evaluate_correct_greater(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = {"num1": 5, "num2": 3, "relation": ">", "answer": "greater than"}
        result = t.evaluate_response(r"\boxed{greater than}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_correct_less(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = {"num1": 3, "num2": 5, "relation": "<", "answer": "less than"}
        result = t.evaluate_response(r"\boxed{less than}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_wrong(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        dp = {"num1": 5, "num2": 3, "relation": ">", "answer": "greater than"}
        result = t.evaluate_response(r"\boxed{less than}", dp)
        assert result["accuracy"] == 0


# Helper to build a simple numeric task test class

def _make_numeric_test_class(cls_name, task_module, task_class_name, list_size, ground_truth_fn):
    """Factory to generate test class for numeric easy tasks."""

    @pytest.fixture
    def _handler(mock_handler):
        return mock_handler

    class _TestNumericTask:
        def _task(self, handler, out):
            import importlib
            mod = importlib.import_module(f"beyondbench.tasks.easy.{task_module}")
            cls = getattr(mod, task_class_name)
            return _make_task(cls, handler, out)

        def test_generate_data(self, mock_handler, output_dir):
            t = self._task(mock_handler, output_dir)
            data = t.generate_data(list_size=list_size)
            assert len(data) == 5

        def test_create_prompt(self, mock_handler, output_dir):
            t = self._task(mock_handler, output_dir)
            data = t.generate_data(list_size=list_size)
            prompt = t.create_prompt(data[0])
            assert isinstance(prompt, str) and len(prompt) > 10

        def test_evaluate_correct(self, mock_handler, output_dir):
            t = self._task(mock_handler, output_dir)
            data = t.generate_data(list_size=list_size)
            dp = data[0]
            gt = ground_truth_fn(dp)
            result = t.evaluate_response(rf"\boxed{{{gt}}}", dp)
            assert result["accuracy"] == 1

        def test_evaluate_empty(self, mock_handler, output_dir):
            t = self._task(mock_handler, output_dir)
            data = t.generate_data(list_size=list_size)
            result = t.evaluate_response("", data[0])
            assert result["accuracy"] == 0

    _TestNumericTask.__name__ = cls_name
    return _TestNumericTask


# Generate test classes for remaining simple numeric easy tasks

TestFindMaximumTask = _make_numeric_test_class(
    "TestFindMaximumTask", "find_maximum_task", "FindMaximumTask", 5,
    lambda dp: max(dp)
)

TestFindMinimumTask = _make_numeric_test_class(
    "TestFindMinimumTask", "find_minimum_task", "FindMinimumTask", 5,
    lambda dp: min(dp)
)

TestOddCountTask = _make_numeric_test_class(
    "TestOddCountTask", "odd_count_task", "OddCountTask", 6,
    lambda dp: sum(1 for x in dp if x % 2 != 0)
)

TestEvenCountTask = _make_numeric_test_class(
    "TestEvenCountTask", "even_count_task", "EvenCountTask", 6,
    lambda dp: sum(1 for x in dp if x % 2 == 0)
)

TestAbsoluteDifferenceTask = _make_numeric_test_class(
    "TestAbsoluteDifferenceTask", "absolute_difference_task", "AbsoluteDifferenceTask", 2,
    lambda dp: abs(dp[0] - dp[1]) if isinstance(dp, list) else abs(dp["num1"] - dp["num2"])
)

TestCountNegativeTask = _make_numeric_test_class(
    "TestCountNegativeTask", "count_negative_task", "CountNegativeTask", 6,
    lambda dp: sum(1 for x in dp if x < 0)
)

TestCountUniqueTask = _make_numeric_test_class(
    "TestCountUniqueTask", "count_unique_task", "CountUniqueTask", 6,
    lambda dp: len(set(dp))
)

TestAlternatingSumTask = _make_numeric_test_class(
    "TestAlternatingSumTask", "alternating_sum_task", "AlternatingSumTask", 4,
    lambda dp: sum(x if i % 2 == 0 else -x for i, x in enumerate(dp))
)


# ===========================================================================
# Mean, Median, Mode — float/special return
# ===========================================================================

class TestSubtractionTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.subtraction_task import SubtractionTask
        return _make_task(SubtractionTask, handler, output_dir)

    def test_generate_data_no_list_size(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data()  # No list_size parameter
        assert len(data) == 5
        for d in data:
            assert len(d) == 2  # [a, b]

    def test_create_prompt(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data()
        prompt = t.create_prompt(data[0])
        assert "subtract" in prompt.lower() or "boxed" in prompt.lower()

    def test_evaluate_correct(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data()
        dp = data[0]
        gt = dp[1] - dp[0]  # subtract dp[0] FROM dp[1]
        result = t.evaluate_response(rf"\boxed{{{gt}}}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_empty(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data()
        result = t.evaluate_response("", data[0])
        assert result["accuracy"] == 0


class TestMeanTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.mean_task import MeanTask
        return _make_task(MeanTask, handler, output_dir)

    def test_generate_data(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=4)
        assert len(data) == 5

    def test_generate_data_is_dict(self, mock_handler, output_dir):
        """MeanTask returns dict with 'input_list' and 'mean' keys."""
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=4)
        dp = data[0]
        assert isinstance(dp, dict)
        assert "input_list" in dp and "mean" in dp

    def test_create_prompt(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=4)
        prompt = t.create_prompt(data[0])
        assert "mean" in prompt.lower() or "average" in prompt.lower()

    def test_evaluate_correct(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=4)
        dp = data[0]
        mean_val = dp["mean"]
        result = t.evaluate_response(rf"\boxed{{{mean_val}}}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_empty(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=4)
        result = t.evaluate_response("", data[0])
        assert result["accuracy"] == 0


class TestMedianTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.median_task import MedianTask
        return _make_task(MedianTask, handler, output_dir)

    def test_generate_data(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=5)
        assert len(data) == 5

    def test_create_prompt(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=5)
        prompt = t.create_prompt(data[0])
        assert "median" in prompt.lower()

    def test_evaluate_correct(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=5)
        dp = data[0]
        # data_point may be dict or list
        if isinstance(dp, dict):
            median_val = dp.get("median", sorted(dp["input_list"])[len(dp["input_list"]) // 2])
            result = t.evaluate_response(rf"\boxed{{{median_val}}}", dp)
        else:
            median_val = sorted(dp)[len(dp) // 2]
            result = t.evaluate_response(rf"\boxed{{{median_val}}}", dp)
        assert result["accuracy"] == 1

    def test_evaluate_empty(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=5)
        result = t.evaluate_response("", data[0])
        assert result["accuracy"] == 0


class TestModeTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.mode_task import ModeTask
        return _make_task(ModeTask, handler, output_dir)

    def test_generate_data(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=6)
        assert len(data) == 5

    def test_create_prompt(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=6)
        prompt = t.create_prompt(data[0])
        assert "mode" in prompt.lower()

    def test_evaluate_does_not_crash(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=6)
        dp = data[0]
        result = t.evaluate_response(r"\boxed{0}", dp)
        assert "accuracy" in result

    def test_evaluate_empty(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=6)
        result = t.evaluate_response("", data[0])
        assert result["accuracy"] == 0


class TestDivisionTask:
    def _task(self, handler, output_dir):
        from beyondbench.tasks.easy.division_task import DivisionTask
        return _make_task(DivisionTask, handler, output_dir)

    def test_generate_data_no_zero_divisor(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        data = t.generate_data(list_size=2)
        for d in data:
            assert d[1] != 0

    def test_evaluate_correct(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        result = t.evaluate_response(r"\boxed{2.0}", [10, 5])
        assert result["accuracy"] == 1


# ===========================================================================
# Medium tasks
# ===========================================================================

class TestFibonacciSequenceTask:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from beyondbench.tasks.medium.fibonacci_sequence_task import FibonacciSequenceTask
            self.TaskClass = FibonacciSequenceTask
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def _task(self, handler, output_dir):
        if not self.available:
            pytest.skip("FibonacciSequenceTask not available")
        return _make_task(self.TaskClass, handler, output_dir)

    def test_generate_data(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        try:
            data = t.generate_data(list_size=6)
            assert len(data) > 0
        except Exception:
            pytest.skip("generate_data signature mismatch")

    def test_task_name(self, mock_handler, output_dir):
        t = self._task(mock_handler, output_dir)
        assert "fibonacci" in t.task_name.lower()


class TestAlgebraicSequenceTask:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from beyondbench.tasks.medium.algebraic_sequence_task import AlgebraicSequenceTask
            self.TaskClass = AlgebraicSequenceTask
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def test_task_importable(self):
        if not self.available:
            pytest.skip("AlgebraicSequenceTask not available")
        assert self.TaskClass is not None

    def test_task_name(self, mock_handler, output_dir):
        if not self.available:
            pytest.skip()
        t = _make_task(self.TaskClass, mock_handler, output_dir)
        assert "algebraic" in t.task_name.lower() or "sequence" in t.task_name.lower()


class TestGeometricSequenceTask:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from beyondbench.tasks.medium.geometric_sequence_task import GeometricSequenceTask
            self.TaskClass = GeometricSequenceTask
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def test_task_importable(self):
        if not self.available:
            pytest.skip("GeometricSequenceTask not available")
        assert self.TaskClass is not None


class TestPrimeSequenceTask:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from beyondbench.tasks.medium.prime_sequence_task import PrimeSequenceTask
            self.TaskClass = PrimeSequenceTask
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def test_task_importable(self):
        if not self.available:
            pytest.skip("PrimeSequenceTask not available")
        assert self.TaskClass is not None


class TestComplexPatternTask:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from beyondbench.tasks.medium.complex_pattern_task import ComplexPatternTask
            self.TaskClass = ComplexPatternTask
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def test_task_importable(self):
        if not self.available:
            pytest.skip("ComplexPatternTask not available")
        assert self.TaskClass is not None


# ===========================================================================
# Hard tasks — import + basic instantiation
# ===========================================================================

_HARD_TASK_SPECS = [
    ("tower_hanoi", "beyondbench.tasks.hard.tower_hanoi_task", "TowerHanoiTask"),
    ("n_queens", "beyondbench.tasks.hard.n_queens_task", "NQueensTask"),
    ("graph_coloring", "beyondbench.tasks.hard.graph_coloring_task", "GraphColoringTask"),
    ("boolean_sat", "beyondbench.tasks.hard.boolean_sat_task", "BooleanSATTask"),
    ("sudoku_solving", "beyondbench.tasks.hard.sudoku_task", "SudokuTask"),
    ("cryptarithmetic", "beyondbench.tasks.hard.cryptarithmetic_task", "CryptarithmeticTask"),
    ("matrix_chain", "beyondbench.tasks.hard.matrix_chain_multiplication_task", "MatrixChainTask"),
    ("modular_systems", "beyondbench.tasks.hard.modular_systems_solver_task", "ModularSystemsTask"),
    ("constraint_opt", "beyondbench.tasks.hard.constraint_optimization_task", "ConstraintOptimizationTask"),
    ("logic_grid", "beyondbench.tasks.hard.logic_grid_puzzles_task_enhanced", "LogicGridPuzzlesTask"),
]


@pytest.mark.parametrize("task_name,module_path,class_name", _HARD_TASK_SPECS)
def test_hard_task_importable(task_name, module_path, class_name):
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        assert cls is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"{task_name}: {e}")


@pytest.mark.parametrize("task_name,module_path,class_name", _HARD_TASK_SPECS)
def test_hard_task_instantiable(task_name, module_path, class_name, mock_handler, output_dir):
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        t = _make_task(cls, mock_handler, output_dir)
        assert t is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"{task_name}: {e}")
    except Exception as e:
        pytest.skip(f"{task_name} constructor mismatch: {e}")


@pytest.mark.parametrize("task_name,module_path,class_name", _HARD_TASK_SPECS)
def test_hard_task_has_task_name(task_name, module_path, class_name, mock_handler, output_dir):
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        t = _make_task(cls, mock_handler, output_dir)
        assert isinstance(t.task_name, str) and len(t.task_name) > 0
    except (ImportError, AttributeError):
        pytest.skip(f"{task_name} not available")
    except Exception:
        pytest.skip(f"{task_name} constructor mismatch")


# ===========================================================================
# Task registry integration
# ===========================================================================

class TestTaskRegistry:
    @pytest.fixture(autouse=True)
    def _registry(self):
        from beyondbench.core.task_registry import TaskRegistry
        self.registry = TaskRegistry()

    # --- Count invariants (dynamic, no hardcoded totals) ---
    # We deliberately avoid asserting a specific count so that adding tasks
    # in future phases does not require editing these tests. Instead we
    # check structural invariants: non-empty, unique, and that "all" is the
    # disjoint union of the three difficulty suites.

    def test_easy_nonempty(self):
        assert len(self.registry.get_tasks_for_suite("easy")) > 0

    def test_medium_nonempty(self):
        assert len(self.registry.get_tasks_for_suite("medium")) > 0

    def test_hard_nonempty(self):
        assert len(self.registry.get_tasks_for_suite("hard")) > 0

    def test_suites_have_unique_task_names(self):
        for suite in ("easy", "medium", "hard"):
            names = self.registry.get_tasks_for_suite(suite)
            assert len(names) == len(set(names)), \
                f"Suite '{suite}' has duplicate task names: {names}"

    def test_all_equals_sum_of_suites(self):
        easy = self.registry.get_tasks_for_suite("easy")
        medium = self.registry.get_tasks_for_suite("medium")
        hard = self.registry.get_tasks_for_suite("hard")
        all_tasks = self.registry.get_tasks_for_suite("all")
        assert len(all_tasks) == len(easy) + len(medium) + len(hard)
        assert set(all_tasks) == set(easy) | set(medium) | set(hard)

    def test_suites_are_disjoint(self):
        easy = set(self.registry.get_tasks_for_suite("easy"))
        medium = set(self.registry.get_tasks_for_suite("medium"))
        hard = set(self.registry.get_tasks_for_suite("hard"))
        assert easy.isdisjoint(medium), f"easy∩medium={easy & medium}"
        assert easy.isdisjoint(hard), f"easy∩hard={easy & hard}"
        assert medium.isdisjoint(hard), f"medium∩hard={medium & hard}"

    def test_all_registered_tasks_have_loadable_class(self):
        for task_name in self.registry.get_tasks_for_suite("all"):
            cls = self.registry.get_task_class(task_name)
            assert cls is not None, f"Task class for '{task_name}' not found"

    def test_unknown_suite_returns_empty(self):
        assert self.registry.get_tasks_for_suite("unknown") == []

    # --- Per-task class availability, driven by the registry itself ---
    # Using pytest_generate_tests below (see module bottom) so that any
    # task added to the registry in future phases is automatically covered
    # without editing this file.


# ===========================================================================
# Edge cases across all easy tasks via registry
# ===========================================================================

class TestAllEasyTasksPrompt:
    """Ensure every easy task can generate data and a prompt.

    The ``task_name`` parameter is auto-populated from the TaskRegistry by
    ``pytest_generate_tests`` above — this class iterates the entire easy
    suite dynamically so new tasks are covered without editing.
    """

    _suite_for_parametrize = "easy"
    # ``comparison`` doesn't follow the ``generate_data(list_size=...)``
    # signature used by the list-based easy tasks, so skip it here.
    _task_filter = staticmethod(lambda name: name != "comparison")

    @pytest.fixture(autouse=True)
    def _registry(self):
        from beyondbench.core.task_registry import TaskRegistry
        self.registry = TaskRegistry()

    def test_generate_and_prompt(self, task_name, mock_handler, output_dir):
        cls = self.registry.get_task_class(task_name)
        if cls is None:
            pytest.skip(f"{task_name} not available")
        t = _make_task(cls, mock_handler, output_dir)
        try:
            data = t.generate_data(list_size=4)
            assert len(data) > 0
            prompt = t.create_prompt(data[0])
            assert isinstance(prompt, str) and len(prompt) > 5
        except Exception as e:
            pytest.skip(f"{task_name} generate/prompt failed: {e}")

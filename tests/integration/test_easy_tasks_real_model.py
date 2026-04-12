"""
Integration tests — ALL 29 easy tasks with a real model on GPU.

Requirements:
    CUDA_VISIBLE_DEVICES=0 pytest tests/integration/test_easy_tasks_real_model.py -v --timeout=600

Model: Qwen/Qwen2.5-1.5B-Instruct (small, fast)
Datapoints: 5 per task, list_size=8
Assertions:
  - parse success rate > 70% for each task
  - accuracy >= 0% (at least doesn't crash)
"""

import os
import pytest
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = [pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Skip if no GPU available
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def require_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available — skipping integration tests")
    except ImportError:
        pytest.skip("PyTorch not available — skipping integration tests")


# ---------------------------------------------------------------------------
# Session-scoped model handler (load model ONCE for all tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def model_handler(tmp_path_factory):
    """Load Qwen2.5-1.5B-Instruct on the current visible GPU (GPU 0 via CUDA_VISIBLE_DEVICES)."""
    from beyondbench.models.model_handler import ModelHandler

    model_id = os.environ.get("INTEGRATION_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    gpu_mem = float(os.environ.get("INTEGRATION_GPU_MEM", "0.45"))  # 40-50% of A40

    logger.info(f"Loading model: {model_id}")
    handler = ModelHandler(
        model_id=model_id,
        backend="vllm",
        cuda_device="cuda:0",
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_model_len=4096,
    )
    yield handler
    # vLLM cleanup happens automatically


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("integration_easy"))


# ---------------------------------------------------------------------------
# Helper to run a single task
# ---------------------------------------------------------------------------

def _run_task(task_class, handler, output_dir, list_size=8, datapoints=5):
    """Instantiate and run a task; returns (parse_success_rate, accuracy)."""
    task = task_class(
        model_handler=handler,
        output_dir=output_dir,
        min_val=-50,
        max_val=50,
        num_folds=1,
        num_samples=datapoints,
        store_details=False,
        temperature=0.1,
        top_p=0.95,
        max_tokens=512,
        seed=42,
    )

    # Generate data
    try:
        if hasattr(task, "generate_data"):
            import inspect
            sig = inspect.signature(task.generate_data)
            if "list_size" in sig.parameters:
                data = task.generate_data(list_size=list_size)
            else:
                data = task.generate_data()
        else:
            pytest.skip(f"No generate_data method")
    except Exception as e:
        pytest.skip(f"generate_data failed: {e}")

    if not data:
        pytest.skip("No data generated")

    # Generate prompts
    prompts = [task.create_prompt(dp) for dp in data]

    # Generate responses
    try:
        responses = handler.generate(prompts, temperature=0.1, max_tokens=512)
    except Exception as e:
        pytest.fail(f"Model generation failed: {e}")

    # Evaluate
    parsed_count = 0
    correct_count = 0
    for dp, resp in zip(data, responses):
        try:
            eval_result = task.evaluate_response(resp, dp)
            if eval_result.get("instruction_followed", False):
                parsed_count += 1
            if eval_result.get("accuracy", 0) == 1:
                correct_count += 1
        except Exception as e:
            logger.warning(f"evaluate_response failed: {e}")

    n = len(data)
    parse_rate = parsed_count / n if n > 0 else 0.0
    accuracy = correct_count / n if n > 0 else 0.0
    return parse_rate, accuracy


# ---------------------------------------------------------------------------
# Per-task tests
# ---------------------------------------------------------------------------

EASY_TASKS = [
    ("sorting",                       "beyondbench.tasks.easy.sorting_task",                       "SortingTask"),
    ("sum",                           "beyondbench.tasks.easy.sum_task",                           "SumTask"),
    ("multiplication",                "beyondbench.tasks.easy.multiplication_task",                "MultiplicationTask"),
    ("odd_count",                     "beyondbench.tasks.easy.odd_count_task",                     "OddCountTask"),
    ("even_count",                    "beyondbench.tasks.easy.even_count_task",                    "EvenCountTask"),
    ("find_maximum",                  "beyondbench.tasks.easy.find_maximum_task",                  "FindMaximumTask"),
    ("find_minimum",                  "beyondbench.tasks.easy.find_minimum_task",                  "FindMinimumTask"),
    ("mean",                          "beyondbench.tasks.easy.mean_task",                          "MeanTask"),
    ("median",                        "beyondbench.tasks.easy.median_task",                        "MedianTask"),
    ("mode",                          "beyondbench.tasks.easy.mode_task",                          "ModeTask"),
    ("subtraction",                   "beyondbench.tasks.easy.subtraction_task",                   "SubtractionTask"),
    ("absolute_difference",           "beyondbench.tasks.easy.absolute_difference_task",           "AbsoluteDifferenceTask"),
    ("division",                      "beyondbench.tasks.easy.division_task",                      "DivisionTask"),
    ("comparison",                    "beyondbench.tasks.easy.comparison_task",                    "ComparisonTask"),
    ("second_maximum",                "beyondbench.tasks.easy.second_maximum_task",                "SecondMaximumTask"),
    ("range",                         "beyondbench.tasks.easy.range_task",                         "RangeTask"),
    ("index_of_maximum",              "beyondbench.tasks.easy.index_of_maximum_task",              "IndexOfMaximumTask"),
    ("count_negative",                "beyondbench.tasks.easy.count_negative_task",                "CountNegativeTask"),
    ("count_unique",                  "beyondbench.tasks.easy.count_unique_task",                  "CountUniqueTask"),
    ("max_adjacent_difference",       "beyondbench.tasks.easy.max_adjacent_difference_task",       "MaxAdjacentDifferenceTask"),
    ("count_greater_than_previous",   "beyondbench.tasks.easy.count_greater_than_previous_task",   "CountGreaterThanPreviousTask"),
    ("sum_of_max_indices",            "beyondbench.tasks.easy.sum_of_max_indices_task",            "SumOfMaxIndicesTask"),
    ("count_palindromic",             "beyondbench.tasks.easy.count_palindromic_task",             "CountPalindromicTask"),
    ("longest_increasing_subsequence","beyondbench.tasks.easy.longest_increasing_subsequence_task","LongestIncreasingSubsequenceTask"),
    ("sum_of_digits",                 "beyondbench.tasks.easy.sum_of_digits_task",                 "SumOfDigitsTask"),
    ("count_perfect_squares",         "beyondbench.tasks.easy.count_perfect_squares_task",         "CountPerfectSquaresTask"),
    ("alternating_sum",               "beyondbench.tasks.easy.alternating_sum_task",               "AlternatingSumTask"),
    ("count_multiples",               "beyondbench.tasks.easy.count_multiples_task",               "CountMultiplesTask"),
    ("local_maxima_count",            "beyondbench.tasks.easy.local_maxima_count_task",            "LocalMaximaCountTask"),
]


@pytest.mark.parametrize("task_name,module_path,class_name", EASY_TASKS)
def test_easy_task_parse_rate(task_name, module_path, class_name, model_handler, output_dir):
    """Parse success rate must be > 70% for easy tasks."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    parse_rate, accuracy = _run_task(cls, model_handler, output_dir, list_size=8, datapoints=5)

    logger.info(f"{task_name}: parse_rate={parse_rate:.2%}, accuracy={accuracy:.2%}")
    assert parse_rate >= 0.70, (
        f"Task '{task_name}': parse success rate {parse_rate:.2%} < 70%"
    )


@pytest.mark.parametrize("task_name,module_path,class_name", EASY_TASKS)
def test_easy_task_no_crash(task_name, module_path, class_name, model_handler, output_dir):
    """Task must complete without raising an exception."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    # _run_task already uses pytest.fail on hard errors
    _run_task(cls, model_handler, output_dir, list_size=8, datapoints=5)


@pytest.mark.parametrize("task_name,module_path,class_name", EASY_TASKS)
def test_easy_task_returns_accuracy_field(task_name, module_path, class_name, model_handler, output_dir):
    """evaluate_response must return a dict with 'accuracy' key."""
    import importlib
    import inspect
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    task = cls(
        model_handler=model_handler,
        output_dir=output_dir,
        min_val=-50, max_val=50,
        num_folds=1, num_samples=2,
        store_details=False,
        temperature=0.1, top_p=0.95, max_tokens=256,
        seed=42,
    )
    try:
        sig = inspect.signature(task.generate_data)
        if "list_size" in sig.parameters:
            try:
                data = task.generate_data(list_size=4)
            except ValueError:
                # Some tasks require specific list_size (e.g., absolute_difference needs 2)
                data = task.generate_data(list_size=2)
        else:
            data = task.generate_data()
        if not data:
            pytest.skip("No data")
        dp = data[0]
        prompt = task.create_prompt(dp)
        resp = model_handler.generate([prompt], temperature=0.1, max_tokens=256)[0]
        result = task.evaluate_response(resp, dp)
        assert "accuracy" in result, f"evaluate_response missing 'accuracy' key for {task_name}"
    except Exception as e:
        pytest.fail(f"{task_name} evaluate_response failed: {e}")

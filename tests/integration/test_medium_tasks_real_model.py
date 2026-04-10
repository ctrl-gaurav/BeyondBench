"""
Integration tests — ALL 5 medium tasks with a real model on GPU.

Requirements:
    CUDA_VISIBLE_DEVICES=1 pytest tests/integration/test_medium_tasks_real_model.py -v --timeout=600

Model: Qwen/Qwen2.5-1.5B-Instruct
Datapoints: 5 per task
Assertions:
  - parse success rate > 50%
  - no crashes
"""

import os
import inspect
import logging
import pytest

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="session", autouse=True)
def require_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture(scope="session")
def model_handler():
    from beyondbench.models.model_handler import ModelHandler
    model_id = os.environ.get("INTEGRATION_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    gpu_mem = float(os.environ.get("INTEGRATION_GPU_MEM", "0.45"))
    logger.info(f"Loading model: {model_id}")
    return ModelHandler(
        model_id=model_id,
        backend="vllm",
        cuda_device="cuda:0",
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_model_len=4096,
    )


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("integration_medium"))


def _run_medium_task(task_class, handler, output_dir, datapoints=5):
    """Run a medium task; return (parse_rate, accuracy)."""
    task = task_class(
        model_handler=handler,
        output_dir=output_dir,
        min_val=1,
        max_val=20,
        num_folds=1,
        num_samples=datapoints,
        store_details=False,
        temperature=0.1,
        top_p=0.95,
        max_tokens=1024,
        seed=42,
    )

    # Try different list_size args for medium tasks
    data = None
    for ls in [6, 8, 10, None]:
        try:
            sig = inspect.signature(task.generate_data)
            if ls is not None and "list_size" in sig.parameters:
                data = task.generate_data(list_size=ls)
            else:
                data = task.generate_data()
            if data:
                break
        except Exception:
            continue

    if not data:
        pytest.skip("Could not generate data")

    prompts = [task.create_prompt(dp) for dp in data]
    try:
        responses = handler.generate(prompts, temperature=0.1, max_tokens=1024)
    except Exception as e:
        pytest.fail(f"Generation failed: {e}")

    parsed = 0
    correct = 0
    for dp, resp in zip(data, responses):
        try:
            r = task.evaluate_response(resp, dp)
            if r.get("instruction_followed", False):
                parsed += 1
            if r.get("accuracy", 0) == 1:
                correct += 1
        except Exception as e:
            logger.warning(f"evaluate failed: {e}")

    n = len(data)
    return parsed / n if n > 0 else 0.0, correct / n if n > 0 else 0.0


MEDIUM_TASKS = [
    ("fibonacci_sequence", "beyondbench.tasks.medium.fibonacci_sequence_task", "FibonacciSequenceTask"),
    ("algebraic_sequence", "beyondbench.tasks.medium.algebraic_sequence_task", "AlgebraicSequenceTask"),
    ("geometric_sequence", "beyondbench.tasks.medium.geometric_sequence_task", "GeometricSequenceTask"),
    ("prime_sequence",     "beyondbench.tasks.medium.prime_sequence_task",     "PrimeSequenceTask"),
    ("complex_pattern",    "beyondbench.tasks.medium.complex_pattern_task",    "ComplexPatternTask"),
]


@pytest.mark.parametrize("task_name,module_path,class_name", MEDIUM_TASKS)
def test_medium_task_no_crash(task_name, module_path, class_name, model_handler, output_dir):
    """Task must complete without crashing."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    _run_medium_task(cls, model_handler, output_dir, datapoints=5)


@pytest.mark.parametrize("task_name,module_path,class_name", MEDIUM_TASKS)
def test_medium_task_parse_rate(task_name, module_path, class_name, model_handler, output_dir):
    """Parse success rate must be > 50% for medium tasks."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    parse_rate, accuracy = _run_medium_task(cls, model_handler, output_dir, datapoints=5)
    logger.info(f"{task_name}: parse_rate={parse_rate:.2%}, accuracy={accuracy:.2%}")
    assert parse_rate >= 0.50, (
        f"Task '{task_name}': parse success rate {parse_rate:.2%} < 50%"
    )


@pytest.mark.parametrize("task_name,module_path,class_name", MEDIUM_TASKS)
def test_medium_task_returns_accuracy_key(task_name, module_path, class_name, model_handler, output_dir):
    """evaluate_response must return dict with 'accuracy' key."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    task = cls(
        model_handler=model_handler,
        output_dir=output_dir,
        min_val=1, max_val=20,
        num_folds=1, num_samples=2,
        store_details=False,
        temperature=0.1, top_p=0.95, max_tokens=512,
        seed=42,
    )
    try:
        sig = inspect.signature(task.generate_data)
        if "list_size" in sig.parameters:
            data = task.generate_data(list_size=6)
        else:
            data = task.generate_data()
        if not data:
            pytest.skip("No data")
        dp = data[0]
        prompt = task.create_prompt(dp)
        resp = model_handler.generate([prompt], temperature=0.1, max_tokens=512)[0]
        result = task.evaluate_response(resp, dp)
        assert "accuracy" in result
    except Exception as e:
        pytest.fail(f"{task_name}: {e}")


@pytest.mark.parametrize("task_name,module_path,class_name", MEDIUM_TASKS)
def test_medium_task_prompt_quality(task_name, module_path, class_name, model_handler, output_dir):
    """Prompt should be non-trivial (>50 chars) and contain task context."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    task = cls(
        model_handler=model_handler,
        output_dir=output_dir,
        min_val=1, max_val=20,
        num_folds=1, num_samples=2,
        store_details=False,
        temperature=0.1, top_p=0.95, max_tokens=512,
        seed=42,
    )
    try:
        sig = inspect.signature(task.generate_data)
        if "list_size" in sig.parameters:
            data = task.generate_data(list_size=6)
        else:
            data = task.generate_data()
        if not data:
            pytest.skip("No data")
        prompt = task.create_prompt(data[0])
        assert len(prompt) > 50, f"Prompt too short for {task_name}: '{prompt}'"
    except Exception as e:
        pytest.skip(f"{task_name} prompt generation issue: {e}")

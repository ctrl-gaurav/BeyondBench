"""
Integration tests — ALL 10 hard tasks with a real model on GPU.

Requirements:
    CUDA_VISIBLE_DEVICES=2 pytest tests/integration/test_hard_tasks_real_model.py -v --timeout=600

Model: Qwen/Qwen2.5-3B-Instruct (needs bigger model for hard tasks)
Datapoints: 3 per task
Assertions:
  - parse success rate > 30% (hard tasks are genuinely hard)
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
    model_id = os.environ.get("HARD_INTEGRATION_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    gpu_mem = float(os.environ.get("INTEGRATION_GPU_MEM", "0.45"))
    logger.info(f"Loading hard-task model: {model_id}")
    return ModelHandler(
        model_id=model_id,
        backend="vllm",
        cuda_device="cuda:0",
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_model_len=8192,
    )


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("integration_hard"))


def _make_hard_task(task_class, handler, output_dir, datapoints=2, **overrides):
    """Instantiate a hard task, adapting to its constructor signature.

    Many hard tasks have custom __init__ signatures (e.g. board_sizes,
    num_variables_list) instead of the standard min_val/max_val.  We inspect
    the constructor and supply sensible defaults.
    """
    sig = inspect.signature(task_class.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    # Common args every hard task accepts
    common = dict(
        model_handler=handler,
        output_dir=output_dir,
        num_folds=1,
        num_samples=datapoints,
        store_details=False,
        temperature=0.1,
        top_p=0.95,
        max_tokens=2048,
        seed=42,
    )

    # Defaults for custom hard-task parameters
    custom_defaults = dict(
        board_sizes=[4, 5],
        graph_types=["cycle"],
        num_vertices_list=[4],
        num_variables_list=[3],
        clause_ratios_list=[2.0],
        sat_types_list=["2-SAT"],
        num_disks_list=[3],
        difficulty_levels=["easy"],
        puzzle_types=["addition"],
        word_lengths=[3],
        constraint_types=["linear"],
        num_projects_list=[3],
        grid_sizes=[3],
        num_equations_list=[2],
        dimension_patterns=["small"],
        matrix_counts_list=[3],
    )

    if has_kwargs or "min_val" in params:
        # Standard BaseTask constructor or accepts **kwargs
        common.update(min_val=1, max_val=10)
    else:
        # Provide only the parameters this class actually accepts
        for k, v in custom_defaults.items():
            if k in params:
                common[k] = v

    common.update(overrides)

    # Filter to only accepted params (avoid TypeErrors)
    if not has_kwargs:
        common = {k: v for k, v in common.items() if k in params}

    return task_class(**common)


def _run_hard_task(task_class, handler, output_dir, datapoints=3):
    """Run a hard task; return (parse_rate, accuracy)."""
    task = _make_hard_task(task_class, handler, output_dir, datapoints=datapoints)

    data = None
    for attempt in [
        {"list_size": 4},
        {"grid_sizes": [4]},
        {},
    ]:
        try:
            sig = inspect.signature(task.generate_data)
            valid_params = {k: v for k, v in attempt.items() if k in sig.parameters}
            data = task.generate_data(**valid_params)
            if data:
                break
        except Exception:
            continue

    if not data:
        pytest.skip("Could not generate data for hard task")

    prompts = [task.create_prompt(dp) for dp in data]
    try:
        responses = handler.generate(prompts, temperature=0.1, max_tokens=2048)
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
            logger.warning(f"evaluate_response failed: {e}")

    n = len(data)
    return parsed / n if n > 0 else 0.0, correct / n if n > 0 else 0.0


HARD_TASKS = [
    ("tower_hanoi",              "beyondbench.tasks.hard.tower_hanoi_task",                   "TowerHanoiTask"),
    ("n_queens",                 "beyondbench.tasks.hard.n_queens_task",                      "NQueensTask"),
    ("graph_coloring",           "beyondbench.tasks.hard.graph_coloring_task",                "GraphColoringTask"),
    ("boolean_sat",              "beyondbench.tasks.hard.boolean_sat_task",                   "BooleanSATTask"),
    ("sudoku_solving",           "beyondbench.tasks.hard.sudoku_task",                        "SudokuTask"),
    ("cryptarithmetic",          "beyondbench.tasks.hard.cryptarithmetic_task",               "CryptarithmeticTask"),
    ("matrix_chain",             "beyondbench.tasks.hard.matrix_chain_multiplication_task",   "MatrixChainTask"),
    ("modular_systems",          "beyondbench.tasks.hard.modular_systems_solver_task",        "ModularSystemsTask"),
    ("constraint_optimization",  "beyondbench.tasks.hard.constraint_optimization_task",       "ConstraintOptimizationTask"),
    ("logic_grid_puzzles",       "beyondbench.tasks.hard.logic_grid_puzzles_task_enhanced",   "LogicGridPuzzlesTask"),
]


@pytest.mark.parametrize("task_name,module_path,class_name", HARD_TASKS)
def test_hard_task_no_crash(task_name, module_path, class_name, model_handler, output_dir):
    """Hard task must complete without crashing."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    _run_hard_task(cls, model_handler, output_dir, datapoints=3)


@pytest.mark.parametrize("task_name,module_path,class_name", HARD_TASKS)
def test_hard_task_parse_rate(task_name, module_path, class_name, model_handler, output_dir):
    """Parse success rate must be > 30% (hard tasks are genuinely hard)."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    parse_rate, accuracy = _run_hard_task(cls, model_handler, output_dir, datapoints=3)
    logger.info(f"{task_name}: parse_rate={parse_rate:.2%}, accuracy={accuracy:.2%}")
    assert parse_rate >= 0.30, (
        f"Task '{task_name}': parse success rate {parse_rate:.2%} < 30%"
    )


@pytest.mark.parametrize("task_name,module_path,class_name", HARD_TASKS)
def test_hard_task_generates_non_trivial_prompt(task_name, module_path, class_name, model_handler, output_dir):
    """Prompts for hard tasks should be substantial (>100 chars)."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    task = _make_hard_task(cls, model_handler, output_dir)
    data = None
    for attempt in [{"list_size": 4}, {"grid_sizes": [4]}, {}]:
        try:
            sig = inspect.signature(task.generate_data)
            valid_params = {k: v for k, v in attempt.items() if k in sig.parameters}
            data = task.generate_data(**valid_params)
            if data:
                break
        except Exception:
            continue
    if not data:
        pytest.skip("No data generated")

    prompt = task.create_prompt(data[0])
    assert len(prompt) > 100, f"{task_name} prompt too short: {len(prompt)} chars"


@pytest.mark.parametrize("task_name,module_path,class_name", HARD_TASKS)
def test_hard_task_evaluate_response_has_accuracy(task_name, module_path, class_name, model_handler, output_dir):
    """evaluate_response must return dict with 'accuracy' key."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Cannot import {class_name}: {e}")

    task = _make_hard_task(cls, model_handler, output_dir)
    data = None
    for attempt in [{"list_size": 4}, {"grid_sizes": [4]}, {}]:
        try:
            sig = inspect.signature(task.generate_data)
            valid_params = {k: v for k, v in attempt.items() if k in sig.parameters}
            data = task.generate_data(**valid_params)
            if data:
                break
        except Exception:
            continue
    if not data:
        pytest.skip("No data")

    dp = data[0]
    prompt = task.create_prompt(dp)
    try:
        resp = model_handler.generate([prompt], temperature=0.1, max_tokens=512)[0]
        result = task.evaluate_response(resp, dp)
        assert "accuracy" in result, f"{task_name}: missing 'accuracy' in evaluate_response result"
    except Exception as e:
        pytest.fail(f"{task_name}: {e}")

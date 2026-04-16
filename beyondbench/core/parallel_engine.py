"""
Parallel evaluation engine for BeyondBench.

Extends EvaluationEngine with multi-GPU support via multiprocessing.
Each GPU worker runs in a separate process with its own CUDA context.

Strategies
----------
task_parallel   — each GPU evaluates different tasks (best when #tasks >> #gpus)
data_parallel   — split datapoints across GPUs for the same task (few tasks, many GPUs)
model_parallel  — each GPU evaluates same tasks with a different model (comparison mode)
auto            — choose strategy based on #tasks vs #gpus
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluation_engine import EvaluationEngine
from .gpu_scheduler import GPUScheduler, estimate_model_vram
from .result_aggregator import ProgressTracker, ResultAggregator
from ..utils.logging_utils import get_logger

logger = get_logger("ParallelEngine")


# ------------------------------------------------------------------ #
# Worker entry-point (runs in a child process)
# ------------------------------------------------------------------ #

def _worker_task_parallel(
    gpu_id: int,
    model_id: str,
    model_kwargs: Dict[str, Any],
    tasks: List[str],
    output_dir: str,
    eval_params: Dict[str, Any],
    engine_params: Dict[str, Any],
    result_queue: "mp.Queue",
    progress_queue: "mp.Queue",
) -> None:
    """Worker for task_parallel strategy.

    Evaluates a subset of tasks on one GPU.
    Puts (gpu_id, result_dict | exception_str) into result_queue.
    """
    # Isolate this process to the assigned GPU BEFORE importing torch/vllm
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Re-seed logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU {gpu_id}] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    child_logger = logging.getLogger(f"Worker-GPU{gpu_id}")
    child_logger.info(f"Starting — tasks: {tasks}")

    try:
        from ..models.model_handler import ModelHandler
        from ..core.evaluation_engine import EvaluationEngine
        from ..core.task_registry import TaskRegistry

        # gpu_memory_utilization: respect the value passed in, which the parent
        # already capped to a safe level (default 0.45 for parallel mode)
        handler = ModelHandler(model_id=model_id, **model_kwargs)

        task_results: Dict[str, Any] = {}
        overall_stats = {
            "total_tasks": len(tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_evaluations": 0,
            "successful_evaluations": 0,
        }

        registry = TaskRegistry()

        for task_idx, task_name in enumerate(tasks):
            try:
                child_logger.info(f"Running task {task_name} ({task_idx+1}/{len(tasks)})")

                # Signal progress
                progress_queue.put({
                    "gpu_id": gpu_id,
                    "task": task_name,
                    "completed": task_idx,
                    "total": len(tasks),
                    "accuracy": 0.0,
                })

                # Build a minimal engine just for running a single task
                import inspect
                task_class = registry.get_task_class(task_name)
                if not task_class:
                    child_logger.error(f"Task class not found: {task_name}")
                    overall_stats["failed_tasks"] += 1
                    continue

                init_sig = inspect.signature(task_class.__init__)
                init_params = set(init_sig.parameters.keys()) - {"self"}
                accepts_kwargs = "kwargs" in init_params

                common_kwargs = {
                    "model_handler": handler,
                    "output_dir": str(Path(output_dir) / "task_results"),
                    "num_folds": eval_params.get("folds", 1),
                    "num_samples": eval_params.get("datapoints", 100),
                    "store_details": engine_params.get("store_details", False),
                    "temperature": eval_params.get("temperature", 0.7),
                    "top_p": eval_params.get("top_p", 0.9),
                    "max_tokens": eval_params.get("max_tokens", 32768),
                    "seed": eval_params.get("seed"),
                    "max_retries": engine_params.get("max_retries", 3),
                    "min_val": eval_params.get("range_min", 1),
                    "max_val": eval_params.get("range_max", 100),
                }

                hard_defaults = {
                    "board_sizes": [4, 5, 6, 8],
                    "num_disks_list": [3, 4, 5],
                    "num_variables_list": [4, 6, 8],
                    "sat_types_list": ["random_k"],
                    "clause_ratios_list": [3.0],
                    "graph_types": ["random", "planar"],
                    "num_vertices_list": [5, 8, 10],
                    "grid_sizes": [3, 4],
                    "difficulty_levels": ["easy", "medium", "hard"],
                    "word_lengths": [3, 4, 5],
                    "puzzle_types": ["addition"],
                    "num_equations_list": [2, 3],
                    "constraint_types": ["standard"],
                    "matrix_counts_list": [3, 4, 5],
                    "dimension_patterns": ["uniform"],
                    "num_projects_list": [3, 4, 5],
                    "technique_levels": ["basic"],
                }
                for pname, pdefault in hard_defaults.items():
                    if pname in init_params:
                        common_kwargs[pname] = eval_params.get(pname, pdefault)

                if accepts_kwargs:
                    filtered = common_kwargs
                else:
                    filtered = {k: v for k, v in common_kwargs.items() if k in init_params}

                task_instance = task_class(**filtered)

                list_sizes = eval_params.get("list_sizes") or [8, 16, 32]
                sig = inspect.signature(task_instance.run_evaluation)
                params = sig.parameters

                if "list_sizes" in params:
                    task_result = task_instance.run_evaluation(list_sizes=list_sizes)
                elif len(params) > 0 and list(params.keys())[0] != "self":
                    try:
                        task_result = task_instance.run_evaluation(list_sizes)
                    except TypeError:
                        task_result = task_instance.run_evaluation()
                else:
                    task_result = task_instance.run_evaluation()

                task_results[task_name] = task_result
                overall_stats["completed_tasks"] += 1

                # Extract accuracy for progress reporting
                acc = _extract_task_accuracy(task_result)
                child_logger.info(f"Task {task_name} done: accuracy={acc:.3f}")

                progress_queue.put({
                    "gpu_id": gpu_id,
                    "task": task_name,
                    "completed": task_idx + 1,
                    "total": len(tasks),
                    "accuracy": acc,
                })

                # Update eval stats
                if isinstance(task_result, list) and task_result:
                    for m in task_result:
                        total = m.get("total_count", m.get("num_samples", 0))
                        correct = m.get("correct_count", int(m.get("accuracy", 0) * total))
                        overall_stats["total_evaluations"] += total
                        overall_stats["successful_evaluations"] += correct
                elif isinstance(task_result, dict) and "summary" in task_result:
                    s = task_result["summary"]
                    overall_stats["total_evaluations"] += s.get("total_datapoints", 0)
                    overall_stats["successful_evaluations"] += s.get("successful_datapoints", 0)

            except Exception:
                err = traceback.format_exc()
                child_logger.error(f"Task {task_name} failed:\n{err}")
                overall_stats["failed_tasks"] += 1

        # Slim down results to avoid large queue payloads, then serialize
        try:
            slim_results = _slim_result(task_results)
            payload = json.dumps({
                "gpu_id": gpu_id,
                "task_results": slim_results,
                "overall_stats": overall_stats,
            }, default=_json_default)
        except Exception as serial_err:
            child_logger.error(f"Serialization error: {serial_err}")
            payload = json.dumps({
                "gpu_id": gpu_id,
                "task_results": {t: {"overall_accuracy": _extract_task_accuracy(r)} for t, r in task_results.items()},
                "overall_stats": overall_stats,
            }, default=_json_default)
        result_queue.put(payload)

    except Exception:
        err = traceback.format_exc()
        child_logger.error(f"Worker crashed:\n{err}")
        result_queue.put(json.dumps({"gpu_id": gpu_id, "error": err, "task_results": {}, "overall_stats": {}}))

    finally:
        # Free GPU memory before the process exits
        if 'handler' in locals() and handler is not None:
            try:
                handler.cleanup()
            except Exception:
                pass
        progress_queue.put({"gpu_id": gpu_id, "done": True})


def _worker_data_parallel(
    gpu_id: int,
    model_id: str,
    model_kwargs: Dict[str, Any],
    task_name: str,
    data_shard: List[Any],   # pre-generated data points for this shard
    output_dir: str,
    eval_params: Dict[str, Any],
    engine_params: Dict[str, Any],
    result_queue: "mp.Queue",
    progress_queue: "mp.Queue",
) -> None:
    """Worker for data_parallel strategy.

    Receives a pre-generated data shard for a single task and evaluates it.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU {gpu_id}] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    child_logger = logging.getLogger(f"Worker-GPU{gpu_id}")
    child_logger.info(f"Starting data_parallel — task: {task_name}, shard size: {len(data_shard)}")

    try:
        from ..models.model_handler import ModelHandler
        from ..core.task_registry import TaskRegistry
        import inspect

        handler = ModelHandler(model_id=model_id, **model_kwargs)
        registry = TaskRegistry()

        task_class = registry.get_task_class(task_name)
        if not task_class:
            raise ValueError(f"Task class not found: {task_name}")

        init_sig = inspect.signature(task_class.__init__)
        init_params = set(init_sig.parameters.keys()) - {"self"}
        accepts_kwargs = "kwargs" in init_params

        common_kwargs = {
            "model_handler": handler,
            "output_dir": str(Path(output_dir) / "task_results"),
            "num_folds": eval_params.get("folds", 1),
            "num_samples": len(data_shard),  # shard size
            "store_details": engine_params.get("store_details", False),
            "temperature": eval_params.get("temperature", 0.7),
            "top_p": eval_params.get("top_p", 0.9),
            "max_tokens": eval_params.get("max_tokens", 32768),
            "seed": eval_params.get("seed"),
            "max_retries": engine_params.get("max_retries", 3),
            "min_val": eval_params.get("range_min", 1),
            "max_val": eval_params.get("range_max", 100),
        }

        if accepts_kwargs:
            filtered = common_kwargs
        else:
            filtered = {k: v for k, v in common_kwargs.items() if k in init_params}

        task_instance = task_class(**filtered)

        # Inject the pre-generated shard directly so the task uses it
        # instead of re-generating data.  Tasks store data in various
        # attributes — inject into the most common one.
        if hasattr(task_instance, "_data"):
            task_instance._data = data_shard
        elif hasattr(task_instance, "data"):
            task_instance.data = data_shard

        # Run evaluation — use run_fold for lower-level access if available
        progress_queue.put({
            "gpu_id": gpu_id,
            "task": task_name,
            "completed": 0,
            "total": len(data_shard),
            "accuracy": 0.0,
        })

        task_result = _run_task_on_data(task_instance, data_shard, eval_params, progress_queue, gpu_id)

        acc = _extract_task_accuracy(task_result)
        child_logger.info(f"Shard done: accuracy={acc:.3f}")

        progress_queue.put({
            "gpu_id": gpu_id,
            "task": task_name,
            "completed": len(data_shard),
            "total": len(data_shard),
            "accuracy": acc,
        })

        try:
            slim_tr = _slim_result({task_name: task_result})
            payload = json.dumps({
                "gpu_id": gpu_id,
                "task_results": slim_tr,
                "overall_stats": {
                    "total_tasks": 1,
                    "completed_tasks": 1,
                    "failed_tasks": 0,
                    "total_evaluations": len(data_shard),
                    "successful_evaluations": int(acc * len(data_shard)),
                },
            }, default=_json_default)
        except Exception as serial_err:
            child_logger.error(f"Serialization error: {serial_err}")
            payload = json.dumps({
                "gpu_id": gpu_id,
                "task_results": {task_name: {"overall_accuracy": acc}},
                "overall_stats": {"total_tasks": 1, "completed_tasks": 1, "failed_tasks": 0,
                                  "total_evaluations": len(data_shard),
                                  "successful_evaluations": int(acc * len(data_shard))},
            }, default=_json_default)
        result_queue.put(payload)

    except Exception:
        err = traceback.format_exc()
        child_logger.error(f"Data worker crashed:\n{err}")
        result_queue.put(json.dumps({"gpu_id": gpu_id, "error": err, "task_results": {}, "overall_stats": {}}))

    finally:
        # Free GPU memory before the process exits
        if 'handler' in locals() and handler is not None:
            try:
                handler.cleanup()
            except Exception:
                pass
        progress_queue.put({"gpu_id": gpu_id, "done": True})


def _run_task_on_data(task_instance, data_shard, eval_params, progress_queue, gpu_id):
    """Run a task's evaluation loop on a pre-supplied data shard.

    Falls back to the standard run_evaluation() if we can't inject data.
    The task may re-generate data but will at least use the same num_samples.
    """
    import inspect

    list_sizes = eval_params.get("list_sizes") or [8, 16, 32]
    sig = inspect.signature(task_instance.run_evaluation)
    params = sig.parameters

    if "list_sizes" in params:
        return task_instance.run_evaluation(list_sizes=list_sizes)
    elif len(params) > 0 and list(params.keys())[0] != "self":
        try:
            return task_instance.run_evaluation(list_sizes)
        except TypeError:
            return task_instance.run_evaluation()
    else:
        return task_instance.run_evaluation()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _slim_result(result: Any) -> Any:
    """Strip per-sample detail lists to keep result payloads small.

    Keeps summary/accuracy metrics but drops heavy list fields.
    Replaces them with pre-computed averages already in the dict.
    """
    # Keys whose VALUES are full response/prompt strings or large lists
    _DROP_KEYS = {
        "responses", "raw_responses", "prompts_processed", "details",
        "per_sample", "fold_results_raw", "response_texts",
        # BaseTask fold-level list accumulators
        "response_lengths", "word_counts", "output_tokens",
    }

    if isinstance(result, list):
        return [_slim_result(m) for m in result]

    if isinstance(result, dict):
        slimmed = {}
        for k, v in result.items():
            if k in _DROP_KEYS:
                continue
            # Drop any value that is a list of strings (response texts)
            if isinstance(v, list) and v and isinstance(v[0], str) and len(str(v)) > 1000:
                continue
            slimmed[k] = _slim_result(v)
        return slimmed

    return result


def _extract_task_accuracy(result) -> float:
    if isinstance(result, list) and result:
        return sum(m.get("accuracy", 0) for m in result) / len(result)
    if isinstance(result, dict):
        if "summary" in result:
            return result["summary"].get("avg_accuracy", 0.0)
        return result.get("overall_accuracy", result.get("accuracy", 0.0))
    return 0.0


def _json_default(obj):
    import numpy as np
    from enum import Enum
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


# ------------------------------------------------------------------ #
# ParallelEvaluationEngine
# ------------------------------------------------------------------ #

class ParallelEvaluationEngine(EvaluationEngine):
    """Multi-GPU evaluation engine.

    Extends EvaluationEngine with ``run_parallel_evaluation()``.
    The parent process does NOT load any model — workers do.
    """

    def __init__(
        self,
        model_handler,  # kept for API compatibility; not used for inference
        output_dir: str = "./beyondbench_results",
        store_details: bool = False,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(
            model_handler=model_handler,
            output_dir=output_dir,
            store_details=store_details,
            max_retries=max_retries,
            **kwargs,
        )
        self.scheduler = GPUScheduler()

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def run_parallel_evaluation(
        self,
        suite: str = "all",
        tasks: Optional[List[str]] = None,
        gpu_ids: Optional[List[int]] = None,
        strategy: str = "auto",
        model_id: str = "",
        model_kwargs: Optional[Dict[str, Any]] = None,
        datapoints: int = 100,
        folds: int = 1,
        **eval_params,
    ) -> Dict[str, Any]:
        """Run evaluation across multiple GPUs in parallel.

        Args:
            suite:        Task suite ("easy", "medium", "hard", "all")
            tasks:        Specific tasks to evaluate (None = all in suite)
            gpu_ids:      GPU IDs to use (None = auto-detect)
            strategy:     "task_parallel" | "data_parallel" | "model_parallel" | "auto"
            model_id:     HuggingFace model path
            model_kwargs: Additional kwargs passed to ModelHandler in each worker
            datapoints:   Datapoints per task/config
            folds:        Cross-validation folds
            **eval_params: Forwarded to task instances

        Returns:
            Dict identical in format to EvaluationEngine.run_evaluation()
        """
        start_time = time.time()

        # Resolve task list
        if tasks:
            task_list = tasks
        else:
            task_list = self.task_registry.get_tasks_for_suite(suite)
        if not task_list:
            raise ValueError(f"No tasks found for suite='{suite}' tasks='{tasks}'")

        # Resolve GPU list
        if gpu_ids is None:
            gpu_ids = self.scheduler.parse_gpu_spec("auto", model_id)
        if not gpu_ids:
            logger.warning("No free GPUs found — falling back to single-GPU sequential evaluation")
            return self.run_evaluation(
                suite=suite, tasks=task_list, datapoints=datapoints, folds=folds, **eval_params
            )

        # Choose strategy
        chosen_strategy = self._choose_strategy(strategy, len(task_list), len(gpu_ids))
        logger.info(
            f"Parallel evaluation: {len(task_list)} tasks on GPUs {gpu_ids} "
            f"strategy={chosen_strategy}"
        )

        # Build eval_params dict for workers
        worker_eval = {
            "datapoints": datapoints,
            "folds": folds,
            "suite": suite,
            "output_dir": str(self.output_dir),
            "strategy": chosen_strategy,
            "gpus": gpu_ids,
            **eval_params,
        }
        worker_engine = {
            "max_retries": self.max_retries,
            "store_details": self.store_details,
        }

        # Prepare model kwargs for workers.
        # Each worker gets an exclusive GPU (scheduler ensures GPUs are truly free),
        # so we use the user's gpu_memory_utilization directly. Default 0.45 is
        # conservative but safe for shared machines.
        mw = dict(model_kwargs or {})
        mw.pop("model_id", None)
        mw.setdefault("gpu_memory_utilization", 0.45)
        # Workers always use vllm
        mw.setdefault("backend", "vllm")

        # Dispatch based on strategy
        if chosen_strategy == "data_parallel":
            worker_results = self._run_data_parallel(
                task_list, gpu_ids, model_id, mw, worker_eval, worker_engine
            )
        else:  # task_parallel / model_parallel / auto→task_parallel
            worker_results = self._run_task_parallel(
                task_list, gpu_ids, model_id, mw, worker_eval, worker_engine
            )

        # Aggregate
        model_info = self.model_handler.get_model_info()
        model_info["model_name"] = model_id or model_info.get("model_name", "")
        final = ResultAggregator.aggregate(
            worker_results=worker_results,
            strategy=chosen_strategy,
            overall_start_time=start_time,
            model_info=model_info,
            eval_config=worker_eval,
        )

        self._save_results(final)
        duration = time.time() - start_time
        logger.info(
            f"Parallel evaluation done in {duration:.1f}s "
            f"({final['summary'].get('avg_accuracy', 0):.3f} avg accuracy)"
        )
        return final

    # ------------------------------------------------------------------ #
    # Strategy dispatch
    # ------------------------------------------------------------------ #

    @staticmethod
    def _choose_strategy(strategy: str, num_tasks: int, num_gpus: int) -> str:
        if strategy != "auto":
            return strategy
        # auto: if more tasks than GPUs, use task_parallel (best throughput);
        # if ≤1 tasks, use data_parallel to saturate all GPUs.
        if num_tasks <= 1:
            return "data_parallel"
        return "task_parallel"

    # ------------------------------------------------------------------ #
    # task_parallel dispatcher
    # ------------------------------------------------------------------ #

    def _run_task_parallel(
        self,
        task_list: List[str],
        gpu_ids: List[int],
        model_id: str,
        model_kwargs: Dict[str, Any],
        eval_params: Dict[str, Any],
        engine_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Distribute tasks round-robin across GPUs, run in parallel."""

        # Assign tasks to GPUs (round-robin)
        assignments: Dict[int, List[str]] = {g: [] for g in gpu_ids}
        for i, task in enumerate(task_list):
            assignments[gpu_ids[i % len(gpu_ids)]].append(task)

        logger.info(f"Task assignments: { {g: tl for g, tl in assignments.items() if tl} }")

        ctx = mp.get_context("spawn")
        result_queue: "mp.Queue" = ctx.Queue()
        progress_queue: "mp.Queue" = ctx.Queue()

        # Progress tracker
        tracker = ProgressTracker(gpu_ids)
        tracker.start()

        # Launch workers
        procs = []
        active_gpus = [g for g in gpu_ids if assignments[g]]
        for gpu_id in active_gpus:
            p = ctx.Process(
                target=_worker_task_parallel,
                args=(
                    gpu_id,
                    model_id,
                    model_kwargs,
                    assignments[gpu_id],
                    str(self.output_dir),
                    eval_params,
                    engine_params,
                    result_queue,
                    progress_queue,
                ),
                daemon=True,
            )
            p.start()
            procs.append((gpu_id, p))
            logger.info(f"Launched worker process {p.pid} on GPU {gpu_id}")

        # Drain progress updates and collect results
        worker_results = self._collect_results(
            procs, result_queue, progress_queue, tracker, active_gpus
        )
        tracker.stop()
        return worker_results

    # ------------------------------------------------------------------ #
    # data_parallel dispatcher
    # ------------------------------------------------------------------ #

    def _run_data_parallel(
        self,
        task_list: List[str],
        gpu_ids: List[int],
        model_id: str,
        model_kwargs: Dict[str, Any],
        eval_params: Dict[str, Any],
        engine_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Split datapoints for each task across GPUs."""

        datapoints = eval_params.get("datapoints", 100)
        num_gpus = len(gpu_ids)
        shard_size = max(1, datapoints // num_gpus)

        ctx = mp.get_context("spawn")
        result_queue: "mp.Queue" = ctx.Queue()
        progress_queue: "mp.Queue" = ctx.Queue()

        tracker = ProgressTracker(gpu_ids)
        tracker.start()

        procs = []
        active_gpus: List[int] = []

        for task_name in task_list:
            for i, gpu_id in enumerate(gpu_ids):
                # Simple shard: each worker gets a slice of the data range
                # We pass a synthetic "shard" list as index range so workers
                # know their portion.  The actual data generation happens
                # inside the worker using the task's own RNG seeded by shard.
                shard_start = i * shard_size
                shard_end = shard_start + shard_size if i < num_gpus - 1 else datapoints
                shard_indices = list(range(shard_start, shard_end))

                # Build a modified eval_params with the shard size
                shard_eval = dict(eval_params)
                shard_eval["datapoints"] = len(shard_indices)
                # Offset the seed per shard so each generates different data
                if shard_eval.get("seed") is not None:
                    shard_eval["seed"] = shard_eval["seed"] + i * 1000

                p = ctx.Process(
                    target=_worker_data_parallel,
                    args=(
                        gpu_id,
                        model_id,
                        model_kwargs,
                        task_name,
                        shard_indices,       # informational — workers use datapoints count
                        str(self.output_dir),
                        shard_eval,
                        engine_params,
                        result_queue,
                        progress_queue,
                    ),
                    daemon=True,
                )
                p.start()
                procs.append((gpu_id, p))
                active_gpus.append(gpu_id)
                logger.info(
                    f"Launched data-parallel worker on GPU {gpu_id} "
                    f"for task {task_name} shard {shard_start}-{shard_end}"
                )

        worker_results = self._collect_results(
            procs, result_queue, progress_queue, tracker, active_gpus
        )
        tracker.stop()
        return worker_results

    # ------------------------------------------------------------------ #
    # Result collection
    # ------------------------------------------------------------------ #

    def _collect_results(
        self,
        procs: List[tuple],
        result_queue: "mp.Queue",
        progress_queue: "mp.Queue",
        tracker: ProgressTracker,
        active_gpus: List[int],
    ) -> List[Dict[str, Any]]:
        """Collect results from worker processes, draining progress updates."""
        import queue as queue_module
        import threading

        worker_results: List[Dict[str, Any]] = []
        done_gpus: set = set()
        expected = len(procs)
        results_received = 0

        # Progress drain thread
        def drain_progress():
            while len(done_gpus) < expected:
                try:
                    msg = progress_queue.get(timeout=0.5)
                    if msg.get("done"):
                        done_gpus.add(msg["gpu_id"])
                    else:
                        tracker.update(
                            gpu_id=msg["gpu_id"],
                            task=msg.get("task", ""),
                            completed=msg.get("completed", 0),
                            total=msg.get("total", 1),
                            accuracy=msg.get("accuracy", 0.0),
                        )
                except Exception:
                    pass

        drain_thread = threading.Thread(target=drain_progress, daemon=True)
        drain_thread.start()

        # Wait for all result payloads
        while results_received < expected:
            try:
                raw = result_queue.get(timeout=600)
                payload = json.loads(raw)
                gpu_id = payload.get("gpu_id", -1)
                if "error" in payload and not payload.get("task_results"):
                    logger.error(f"GPU {gpu_id} worker failed: {payload['error']}")
                    tracker.mark_done(gpu_id, error=payload["error"])
                else:
                    tracker.mark_done(gpu_id)
                    worker_results.append(payload)
                results_received += 1
            except Exception as exc:
                logger.error(f"Result collection error: {exc}")
                break

        # Wait for all processes to exit
        for gpu_id, p in procs:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker on GPU {gpu_id} still alive after timeout — terminating")
                p.terminate()
                p.join(timeout=5)

        drain_thread.join(timeout=5)
        return worker_results

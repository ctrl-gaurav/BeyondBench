"""
Result aggregator for multi-GPU parallel evaluation.

Merges results from multiple worker processes, handles partial failures,
deduplicates data_parallel results, and produces a unified final_results.json
identical in format to single-GPU evaluation output.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ResultAggregator")


# ------------------------------------------------------------------ #
# ProgressTracker
# ------------------------------------------------------------------ #

class ProgressTracker:
    """Thread-safe live progress across all GPU workers.

    Workers call update() to report their current status.
    A background thread renders a rich multi-row table every 2 seconds.
    """

    def __init__(self, gpu_ids: List[int]):
        self._lock = threading.Lock()
        self._status: Dict[int, Dict[str, Any]] = {
            gpu_id: {
                "task": "idle",
                "progress": 0.0,
                "accuracy": 0.0,
                "completed": 0,
                "total": 0,
                "start_time": time.time(),
                "done": False,
                "error": None,
            }
            for gpu_id in gpu_ids
        }
        self._render_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._console = None  # lazy-init
        self._live = None     # rich Live context

    # ---- public update API called from worker threads ----------------

    def update(self, gpu_id: int, task: str, completed: int, total: int, accuracy: float = 0.0):
        """Update per-GPU progress."""
        with self._lock:
            st = self._status[gpu_id]
            st["task"] = task
            st["completed"] = completed
            st["total"] = total
            st["progress"] = completed / max(1, total)
            st["accuracy"] = accuracy

    def mark_done(self, gpu_id: int, error: Optional[str] = None):
        """Mark a GPU worker as finished."""
        with self._lock:
            self._status[gpu_id]["done"] = True
            self._status[gpu_id]["error"] = error
            if error is None:
                self._status[gpu_id]["task"] = "done"
            else:
                self._status[gpu_id]["task"] = "FAILED"

    # ---- lifecycle ---------------------------------------------------

    def start(self):
        """Start background rendering thread with rich Live display."""
        self._stop_event.clear()
        try:
            from rich.console import Console
            from rich.live import Live
            self._console = Console()
            self._live = Live(
                self._build_table(),
                console=self._console,
                refresh_per_second=0.5,
                transient=False,
            )
            self._live.start()
        except Exception:
            self._live = None
            self._console = None

        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def stop(self):
        """Stop background rendering and print final state."""
        self._stop_event.set()
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=5)
        if self._live:
            try:
                self._live.update(self._build_table())
                self._live.stop()
            except Exception:
                pass
        else:
            self._log_status()

    # ---- rendering ---------------------------------------------------

    def _render_loop(self):
        while not self._stop_event.is_set():
            if self._live:
                try:
                    self._live.update(self._build_table())
                except Exception:
                    pass
            else:
                self._log_status()
            self._stop_event.wait(timeout=2.0)

    def _build_table(self):
        """Build the rich table from current status snapshot."""
        try:
            from rich.table import Table
        except ImportError:
            return None

        with self._lock:
            snapshot = {gid: dict(st) for gid, st in self._status.items()}

        table = Table(title="GPU Evaluation Progress", show_lines=False)
        table.add_column("GPU", style="bold cyan", width=5)
        table.add_column("Task", style="white", min_width=28)
        table.add_column("Progress", justify="right", width=12)
        table.add_column("Accuracy", justify="right", width=10)
        table.add_column("Status", justify="center", width=8)

        for gpu_id in sorted(snapshot):
            st = snapshot[gpu_id]
            elapsed = time.time() - st["start_time"]
            progress_str = f"{st['completed']}/{st['total']} ({st['progress']:.0%})"
            acc_str = f"{st['accuracy']:.3f}" if st["completed"] > 0 else "-"

            if st["error"]:
                status_str = "[red]FAIL[/red]"
            elif st["done"]:
                status_str = "[green]DONE[/green]"
            else:
                status_str = f"[yellow]{elapsed:.0f}s[/yellow]"

            table.add_row(str(gpu_id), st["task"], progress_str, acc_str, status_str)

        return table

    def _log_status(self):
        """Fallback when rich is not available."""
        with self._lock:
            snapshot = {gid: dict(st) for gid, st in self._status.items()}
        for gpu_id in sorted(snapshot):
            st = snapshot[gpu_id]
            logger.info(
                f"GPU {gpu_id}: {st['task']} "
                f"{st['completed']}/{st['total']} acc={st['accuracy']:.3f}"
            )


# ------------------------------------------------------------------ #
# ResultAggregator
# ------------------------------------------------------------------ #

class ResultAggregator:
    """Merge results from multiple GPU workers.

    Supports:
    - task_parallel: each worker has distinct tasks → simple merge
    - data_parallel: workers share the same task with different data shards
                     → merge fold_results / re-compute averages
    - model_parallel: same tasks but different models (future use)
    """

    @staticmethod
    def aggregate(
        worker_results: List[Dict[str, Any]],
        strategy: str,
        overall_start_time: float,
        model_info: Dict[str, Any],
        eval_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge worker results into a single final_results.json payload.

        Args:
            worker_results:   List of dicts returned by each worker.
                              Each dict has keys: task_results, overall_stats, gpu_id.
            strategy:         "task_parallel" | "data_parallel" | "model_parallel" | "auto"
            overall_start_time: wall-clock start (time.time())
            model_info:       From model_handler.get_model_info()
            eval_config:      Evaluation config dict (for metadata)

        Returns:
            Dict in the same format as EvaluationEngine._aggregate_results()
        """
        if strategy == "data_parallel":
            return ResultAggregator._merge_data_parallel(
                worker_results, overall_start_time, model_info, eval_config
            )
        else:  # task_parallel, model_parallel, auto
            return ResultAggregator._merge_task_parallel(
                worker_results, overall_start_time, model_info, eval_config
            )

    # ------------------------------------------------------------------ #
    # task_parallel merge
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_task_parallel(
        worker_results: List[Dict[str, Any]],
        overall_start_time: float,
        model_info: Dict[str, Any],
        eval_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge disjoint task results from multiple workers."""
        merged_task_results: Dict[str, Any] = {}
        overall_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_evaluations": 0,
            "successful_evaluations": 0,
        }

        for wr in worker_results:
            if not wr or "task_results" not in wr:
                logger.warning(f"Worker result missing task_results: {wr}")
                continue
            tr = wr.get("task_results", {})
            ws = wr.get("overall_stats", {})

            merged_task_results.update(tr)
            overall_stats["total_tasks"]            += ws.get("total_tasks", len(tr))
            overall_stats["completed_tasks"]        += ws.get("completed_tasks", len(tr))
            overall_stats["failed_tasks"]           += ws.get("failed_tasks", 0)
            overall_stats["total_evaluations"]      += ws.get("total_evaluations", 0)
            overall_stats["successful_evaluations"] += ws.get("successful_evaluations", 0)

        return ResultAggregator._build_final(
            merged_task_results, overall_stats, overall_start_time, model_info, eval_config
        )

    # ------------------------------------------------------------------ #
    # data_parallel merge
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_data_parallel(
        worker_results: List[Dict[str, Any]],
        overall_start_time: float,
        model_info: Dict[str, Any],
        eval_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge sharded results for the same task(s) from multiple workers."""
        # Collect per-task shards
        task_shards: Dict[str, List[Any]] = {}

        for wr in worker_results:
            if not wr or "task_results" not in wr:
                continue
            for task_name, result in wr["task_results"].items():
                task_shards.setdefault(task_name, []).append(result)

        merged_task_results: Dict[str, Any] = {}
        overall_stats = {
            "total_tasks": len(task_shards),
            "completed_tasks": len(task_shards),
            "failed_tasks": 0,
            "total_evaluations": 0,
            "successful_evaluations": 0,
        }

        for task_name, shards in task_shards.items():
            merged = ResultAggregator._merge_task_shards(shards)
            merged_task_results[task_name] = merged

            # Update stats
            if isinstance(merged, list):
                for m in merged:
                    total = m.get("total_count", m.get("num_samples", 0))
                    correct = m.get("correct_count", int(m.get("accuracy", 0) * total))
                    overall_stats["total_evaluations"]      += total
                    overall_stats["successful_evaluations"] += correct
            elif isinstance(merged, dict) and "summary" in merged:
                s = merged["summary"]
                overall_stats["total_evaluations"]      += s.get("total_datapoints", 0)
                overall_stats["successful_evaluations"] += s.get("successful_datapoints", 0)

        return ResultAggregator._build_final(
            merged_task_results, overall_stats, overall_start_time, model_info, eval_config
        )

    @staticmethod
    def _merge_task_shards(shards: List[Any]) -> Any:
        """Merge multiple result shards for the same task.

        Handles both list-of-metrics and dict-with-summary formats.
        """
        if not shards:
            return {}

        # Determine format from first non-None shard
        sample = next((s for s in shards if s is not None), None)
        if sample is None:
            return {}

        if isinstance(sample, list):
            # Flatten list shards
            merged: List[Any] = []
            for shard in shards:
                if isinstance(shard, list):
                    merged.extend(shard)
            return merged

        if isinstance(sample, dict):
            # Merge dict format — rebuild summary from fold_results
            all_fold_results: List[Any] = []
            for shard in shards:
                if isinstance(shard, dict):
                    folds = shard.get("fold_results", [])
                    if folds:
                        all_fold_results.extend(folds)

            if all_fold_results:
                total = sum(f.get("total_count", 0) for f in all_fold_results)
                correct = sum(f.get("correct_count", 0) for f in all_fold_results)
                accuracy = correct / total if total > 0 else 0.0
                merged_dict = {
                    "overall_accuracy": accuracy,
                    "fold_results": all_fold_results,
                    "total_count": total,
                    "correct_count": correct,
                }
                # Carry over summary if present
                if "summary" in sample:
                    merged_summary = dict(sample["summary"])
                    merged_summary["avg_accuracy"] = accuracy
                    merged_summary["total_datapoints"] = total
                    merged_summary["successful_datapoints"] = correct
                    merged_dict["summary"] = merged_summary
                return merged_dict

            # Fall back: average top-level accuracies
            accs = [s.get("overall_accuracy", s.get("accuracy", 0)) for s in shards if isinstance(s, dict)]
            avg_acc = sum(accs) / len(accs) if accs else 0.0
            merged_dict = dict(sample)
            merged_dict["overall_accuracy"] = avg_acc
            return merged_dict

        return sample

    # ------------------------------------------------------------------ #
    # Final result builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_final(
        task_results: Dict[str, Any],
        overall_stats: Dict[str, Any],
        overall_start_time: float,
        model_info: Dict[str, Any],
        eval_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the canonical final_results dict."""
        total_duration = time.time() - overall_start_time

        # Compute accuracy across tasks
        accuracies = []
        for task_name, result in task_results.items():
            if isinstance(result, list) and result:
                accs = [m.get("accuracy", 0) for m in result]
                accuracies.append(sum(accs) / len(accs))
            elif isinstance(result, dict):
                if "summary" in result and "avg_accuracy" in result["summary"]:
                    accuracies.append(result["summary"]["avg_accuracy"])
                elif "overall_accuracy" in result:
                    accuracies.append(result["overall_accuracy"])

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        total_ev = overall_stats.get("total_evaluations", 0)
        succ_ev = overall_stats.get("successful_evaluations", 0)

        summary = {
            "total_duration": total_duration,
            "total_tasks": overall_stats.get("total_tasks", len(task_results)),
            "completed_tasks": overall_stats.get("completed_tasks", len(task_results)),
            "failed_tasks": overall_stats.get("failed_tasks", 0),
            "total_evaluations": total_ev,
            "successful_evaluations": succ_ev,
            "success_rate": succ_ev / max(1, total_ev),
            "avg_accuracy": avg_accuracy,
            "evaluations_per_second": total_ev / max(1.0, total_duration),
            "parallel": True,
        }

        return {
            "summary": summary,
            "task_results": task_results,
            "overall_stats": overall_stats,
            "model_info": model_info,
            "evaluation_config": {
                "suite": eval_config.get("suite", ""),
                "tasks": list(task_results.keys()),
                "output_dir": eval_config.get("output_dir", ""),
                "parallel": True,
                "strategy": eval_config.get("strategy", "task_parallel"),
                "gpus": eval_config.get("gpus", []),
            },
        }

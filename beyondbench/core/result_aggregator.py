"""
Result aggregator for multi-GPU parallel evaluation.

Merges results from multiple worker processes, handles partial failures,
deduplicates data_parallel results, and produces a unified final_results.json
identical in format to single-GPU evaluation output.
"""

import logging
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ResultAggregator")


# ------------------------------------------------------------------ #
# GPUUtilizationSampler
# ------------------------------------------------------------------ #

class GPUUtilizationSampler:
    """Best-effort background sampler for per-GPU utilization + memory.

    Uses ``nvidia-smi`` (fast, no GPU drivers/Python bindings needed). Sampling
    runs in a daemon thread; public API is ``sample()`` which returns the most
    recent snapshot dict:

        { gpu_id: {"util": 73, "mem_used_mib": 18542, "mem_total_mib": 24564} }

    If ``nvidia-smi`` is unavailable or sampling fails, ``sample()`` returns
    an empty dict — callers should treat that as "no GPU info" without
    erroring out. Designed to be safe for CPU-only boxes and CI.
    """

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self._interval = max(0.5, float(interval_seconds))
        self._latest: Dict[int, Dict[str, int]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available: bool = shutil.which("nvidia-smi") is not None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="gpu-util-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def sample(self) -> Dict[int, Dict[str, int]]:
        with self._lock:
            return {gid: dict(snap) for gid, snap in self._latest.items()}

    def sample_once(self) -> Dict[int, Dict[str, int]]:
        """Synchronous one-shot sample. Useful when you don't want a thread."""
        return self._query()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._query()
                with self._lock:
                    self._latest = snap
            except Exception as exc:
                # Never crash; simply degrade to no data.
                logger.debug("GPU sampler query failed: %s", exc)
            self._stop_event.wait(timeout=self._interval)

    @staticmethod
    def _query() -> Dict[int, Dict[str, int]]:
        """Run nvidia-smi once and parse into a dict."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except Exception:
            return {}
        if result.returncode != 0:
            return {}

        out: Dict[int, Dict[str, int]] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                gpu_id = int(parts[0])
                util = int(parts[1])
                mem_used = int(parts[2])
                mem_total = int(parts[3])
            except ValueError:
                continue
            out[gpu_id] = {
                "util": util,
                "mem_used_mib": mem_used,
                "mem_total_mib": mem_total,
            }
        return out


def summarize_gpu_utilization(snapshot: Dict[int, Dict[str, int]]) -> str:
    """Render a GPU snapshot as a compact string for tqdm postfix / logs.

    Example: ``"GPU0 73%/18.1GB GPU1 45%/9.2GB"``. Returns an empty string
    when the snapshot is empty so callers can safely concatenate.
    """
    if not snapshot:
        return ""
    parts = []
    for gid in sorted(snapshot):
        snap = snapshot[gid]
        mem_used_gib = snap.get("mem_used_mib", 0) / 1024.0
        parts.append(f"GPU{gid} {snap.get('util', 0)}%/{mem_used_gib:.1f}GB")
    return " ".join(parts)


# ------------------------------------------------------------------ #
# ProgressTracker
# ------------------------------------------------------------------ #

class ProgressTracker:
    """Thread-safe live progress across all GPU workers.

    Workers call update() to report their current status.
    A background thread renders a rich multi-row table every 2 seconds.
    """

    def __init__(self, gpu_ids: List[int], *, sample_gpu_util: bool = True):
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
        # Background GPU utilization sampler (opt-out via sample_gpu_util=False)
        self._gpu_sampler: Optional[GPUUtilizationSampler] = (
            GPUUtilizationSampler() if sample_gpu_util else None
        )

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
        # Start GPU util sampler in the background (noop if no nvidia-smi)
        if self._gpu_sampler is not None:
            try:
                self._gpu_sampler.start()
            except Exception:
                pass
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
        if self._gpu_sampler is not None:
            try:
                self._gpu_sampler.stop()
            except Exception:
                pass
        if self._live:
            try:
                self._live.update(self._build_table())
                self._live.stop()
            except Exception:
                pass
        else:
            self._log_status()

    def gpu_snapshot(self) -> Dict[int, Dict[str, int]]:
        """Public: latest GPU util/memory snapshot for the worker GPUs.

        Empty dict if sampling is disabled or ``nvidia-smi`` isn't available.
        """
        if self._gpu_sampler is None:
            return {}
        return self._gpu_sampler.sample()

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

        gpu_snap = self.gpu_snapshot()

        table = Table(title="GPU Evaluation Progress", show_lines=False)
        table.add_column("GPU", style="bold cyan", width=5)
        table.add_column("Task", style="white", min_width=28)
        table.add_column("Progress", justify="right", width=12)
        table.add_column("Accuracy", justify="right", width=10)
        table.add_column("Util", justify="right", width=7)
        table.add_column("Memory", justify="right", width=14)
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

            gpu_info = gpu_snap.get(gpu_id, {})
            util_val = gpu_info.get("util")
            mem_used = gpu_info.get("mem_used_mib")
            mem_total = gpu_info.get("mem_total_mib")
            if util_val is None:
                util_str = "-"
                mem_str = "-"
            else:
                util_color = "green" if util_val >= 50 else ("yellow" if util_val >= 10 else "dim")
                util_str = f"[{util_color}]{util_val}%[/{util_color}]"
                if mem_used is not None and mem_total:
                    mem_str = f"{mem_used / 1024.0:.1f}/{mem_total / 1024.0:.1f}GB"
                else:
                    mem_str = "-"

            table.add_row(
                str(gpu_id), st["task"], progress_str, acc_str,
                util_str, mem_str, status_str,
            )

        return table

    def _log_status(self):
        """Fallback when rich is not available."""
        with self._lock:
            snapshot = {gid: dict(st) for gid, st in self._status.items()}
        gpu_snap = self.gpu_snapshot()
        for gpu_id in sorted(snapshot):
            st = snapshot[gpu_id]
            gpu_info = gpu_snap.get(gpu_id, {})
            util_part = ""
            if gpu_info:
                util_part = (
                    f" util={gpu_info.get('util', 0)}%"
                    f" mem={gpu_info.get('mem_used_mib', 0) / 1024.0:.1f}GB"
                )
            logger.info(
                f"GPU {gpu_id}: {st['task']} "
                f"{st['completed']}/{st['total']} acc={st['accuracy']:.3f}"
                f"{util_part}"
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

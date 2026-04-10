"""
Thread-safe data bridge between EvaluationEngine and the Gradio dashboard.

The evaluation runs in a worker thread/process; the dashboard polls this
bridge from the main thread.  All writes are protected by a Lock so there
are no race conditions.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    EVALUATION_STARTED = "evaluation_started"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    FOLD_COMPLETED = "fold_completed"
    EVALUATION_COMPLETE = "evaluation_complete"
    ERROR = "error"
    LOG = "log"
    GPU_UPDATE = "gpu_update"
    TOKEN_UPDATE = "token_update"


@dataclass
class DashboardEvent:
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


class DataBridge:
    """
    Central, thread-safe bridge between an EvaluationEngine and the dashboard.

    Usage (evaluation side):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=10, model_info={...})
        bridge.notify_task_started("sum")
        bridge.notify_task_completed("sum", accuracy=0.92, tokens=1500)
        bridge.notify_evaluation_complete(results)

    Usage (dashboard side):
        state = bridge.get_state()   # snapshot, safe to read anytime
        events = bridge.drain_events()  # get & clear the event queue
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._event_queue: queue.Queue[DashboardEvent] = queue.Queue()

        # Evaluation state — mutated by evaluation thread, read by dashboard thread
        self._state: Dict[str, Any] = {
            "status": "idle",           # idle | running | completed | failed
            "model_info": {},
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_task": None,
            "start_time": None,
            "end_time": None,
            "task_results": {},         # task_name -> {accuracy, tokens, latency, ...}
            "suite_accuracies": {"easy": [], "medium": [], "hard": []},
            "log_lines": [],            # list of (timestamp, level, message)
            "gpu_stats": [],            # list of per-GPU dicts
            "token_stats": {            # aggregate token info
                "total_input": 0,
                "total_output": 0,
                "per_task": {},
            },
        }

    # ------------------------------------------------------------------ #
    # Write API (called from evaluation thread)
    # ------------------------------------------------------------------ #

    def notify_evaluation_started(self, total_tasks: int, model_info: Dict[str, Any]):
        with self._lock:
            self._state["status"] = "running"
            self._state["total_tasks"] = total_tasks
            self._state["completed_tasks"] = 0
            self._state["failed_tasks"] = 0
            self._state["model_info"] = model_info
            self._state["start_time"] = time.time()
            self._state["task_results"] = {}
            self._state["log_lines"] = []
        self._put_event(EventType.EVALUATION_STARTED, {
            "total_tasks": total_tasks,
            "model_info": model_info,
        })

    def notify_task_started(self, task_name: str):
        with self._lock:
            self._state["current_task"] = task_name
        self._put_event(EventType.TASK_STARTED, {"task_name": task_name})

    def notify_task_completed(
        self,
        task_name: str,
        accuracy: float,
        suite: str = "easy",
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_s: float = 0.0,
        parse_success_rate: float = 1.0,
        failed: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ):
        with self._lock:
            if failed:
                self._state["failed_tasks"] += 1
            else:
                self._state["completed_tasks"] += 1

            self._state["current_task"] = None
            self._state["task_results"][task_name] = {
                "accuracy": accuracy,
                "suite": suite,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_s": latency_s,
                "parse_success_rate": parse_success_rate,
                "failed": failed,
                **(extra or {}),
            }

            suite_key = suite if suite in ("easy", "medium", "hard") else "easy"
            self._state["suite_accuracies"][suite_key].append(accuracy)

            self._state["token_stats"]["total_input"] += tokens_in
            self._state["token_stats"]["total_output"] += tokens_out
            self._state["token_stats"]["per_task"][task_name] = {
                "input": tokens_in,
                "output": tokens_out,
            }

        self._put_event(EventType.TASK_COMPLETED, {
            "task_name": task_name,
            "accuracy": accuracy,
            "failed": failed,
        })

    def notify_fold_completed(self, task_name: str, fold: int, accuracy: float):
        self._put_event(EventType.FOLD_COMPLETED, {
            "task_name": task_name,
            "fold": fold,
            "accuracy": accuracy,
        })

    def notify_evaluation_complete(self, results: Dict[str, Any]):
        with self._lock:
            self._state["status"] = "completed"
            self._state["end_time"] = time.time()
            self._state["current_task"] = None
        self._put_event(EventType.EVALUATION_COMPLETE, {"results": results})

    def notify_error(self, message: str, task_name: Optional[str] = None):
        with self._lock:
            self._state["status"] = "failed"
        self._put_event(EventType.ERROR, {"message": message, "task_name": task_name})

    def push_log(self, level: str, message: str):
        """Push a log line from the evaluation (called by logging handler)."""
        ts = time.time()
        with self._lock:
            self._state["log_lines"].append((ts, level, message))
            # Keep only the last 2000 lines to avoid unbounded growth
            if len(self._state["log_lines"]) > 2000:
                self._state["log_lines"] = self._state["log_lines"][-2000:]
        self._put_event(EventType.LOG, {"level": level, "message": message, "timestamp": ts})

    def update_gpu_stats(self, gpu_stats: List[Dict[str, Any]]):
        """Push fresh GPU metrics (called from the GPU monitor thread)."""
        with self._lock:
            self._state["gpu_stats"] = gpu_stats
        self._put_event(EventType.GPU_UPDATE, {"gpu_stats": gpu_stats})

    # ------------------------------------------------------------------ #
    # Read API (called from dashboard thread)
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current state (safe to read from any thread)."""
        with self._lock:
            import copy
            return copy.deepcopy(self._state)

    def drain_events(self) -> List[DashboardEvent]:
        """Drain all pending events from the queue (non-blocking)."""
        events: List[DashboardEvent] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def elapsed_seconds(self) -> Optional[float]:
        """Return elapsed evaluation time in seconds, or None if not started."""
        with self._lock:
            start = self._state.get("start_time")
        if start is None:
            return None
        end = self._state.get("end_time") or time.time()
        return end - start

    def is_running(self) -> bool:
        with self._lock:
            return self._state["status"] == "running"

    def is_complete(self) -> bool:
        with self._lock:
            return self._state["status"] in ("completed", "failed")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _put_event(self, event_type: EventType, data: Dict[str, Any]):
        self._event_queue.put(DashboardEvent(event_type=event_type, data=data))


# ------------------------------------------------------------------ #
# Logging handler that feeds log lines into the bridge
# ------------------------------------------------------------------ #

import logging  # noqa: E402


class BridgeLogHandler(logging.Handler):
    """Logging handler that pushes records into a DataBridge."""

    def __init__(self, bridge: DataBridge, level: int = logging.DEBUG):
        super().__init__(level)
        self.bridge = bridge

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.bridge.push_log(record.levelname, msg)
        except Exception:
            self.handleError(record)

"""Baseline management for regression testing in BeyondBench evaluations.

Provides the BaselineManager class for saving, loading, and comparing
evaluation baselines so regressions can be detected across runs.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default directory (relative to the project root) where baselines are stored.
_DEFAULT_BASELINES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "baselines"


def _baselines_dir() -> Path:
    """Return the baselines directory, creating it if necessary."""
    d = _DEFAULT_BASELINES_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_task_metrics(task_entries: Any) -> Dict[str, float]:
    """Extract averaged metrics from a task's result entries.

    ``task_entries`` may be:
      - a list of dicts (normal case in final_results.json)
      - a single dict (legacy format)
      - something else (tolerate gracefully)

    Returns a dict with keys:
      accuracy, parsing_success_rate, generation_success_rate
    """
    if isinstance(task_entries, dict):
        # Single-dict legacy format
        task_entries = [task_entries]

    if not isinstance(task_entries, list) or not task_entries:
        return {"accuracy": 0.0, "parsing_success_rate": 0.0, "generation_success_rate": 0.0}

    def _avg(key: str, default: float = 1.0) -> float:
        values = [e[key] for e in task_entries if isinstance(e, dict) and key in e]
        if not values:
            return default
        return statistics.mean(values)

    return {
        "accuracy": _avg("accuracy", default=0.0),
        "parsing_success_rate": _avg("parsing_success_rate", default=1.0),
        "generation_success_rate": _avg("generation_success_rate", default=1.0),
    }


def _load_final_results(results_dir: str | Path) -> Dict[str, Any]:
    """Load and validate final_results.json from *results_dir*."""
    results_path = Path(results_dir) / "final_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No final_results.json found in '{results_dir}'. "
            "Make sure the evaluation has completed successfully."
        )
    with results_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"final_results.json in '{results_dir}' has unexpected format (expected a JSON object).")
    if "task_results" not in data:
        raise ValueError(
            f"final_results.json in '{results_dir}' is missing the 'task_results' key. "
            "The file may be from an older version of BeyondBench."
        )
    return data


class BaselineManager:
    """Save, load, and compare evaluation baselines.

    Baselines are stored as JSON files in ``tests/baselines/`` (relative to the
    project root) so they can be committed to source control alongside the
    codebase and tracked over time.

    Usage::

        mgr = BaselineManager()

        # After running an evaluation:
        mgr.save("./beyondbench_results", "v0.2.0-qwen-1.5b")

        # Later, to check for regressions:
        report = mgr.compare("./new_results", "v0.2.0-qwen-1.5b")
        regressed = mgr.check_regression("./new_results", "v0.2.0-qwen-1.5b")
    """

    def __init__(self, baselines_dir: Optional[str | Path] = None) -> None:
        if baselines_dir is None:
            self._dir = _baselines_dir()
        else:
            self._dir = Path(baselines_dir)
            self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(self, results_dir: str | Path, name: str) -> Path:
        """Save a baseline derived from *results_dir*'s ``final_results.json``.

        Parameters
        ----------
        results_dir:
            Directory containing ``final_results.json``.
        name:
            Human-readable baseline name, e.g. ``"v0.2.0-qwen-1.5b"``.
            The baseline will be stored as ``tests/baselines/{name}.json``.

        Returns
        -------
        Path
            Path to the written baseline file.
        """
        logger.info("Saving baseline '%s' from '%s'", name, results_dir)
        data = _load_final_results(results_dir)

        task_baselines: Dict[str, Dict[str, float]] = {}
        for task_name, task_entries in data["task_results"].items():
            task_baselines[task_name] = _extract_task_metrics(task_entries)

        baseline: Dict[str, Any] = {
            "name": name,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model_info": data.get("model_info", {}),
            "evaluation_config": data.get("evaluation_config", {}),
            "task_baselines": task_baselines,
        }

        out_path = self._dir / f"{name}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(baseline, fh, indent=2, default=str)
            fh.write("\n")

        logger.info(
            "Baseline '%s' saved to '%s' (%d tasks)",
            name,
            out_path,
            len(task_baselines),
        )
        return out_path

    def load(self, name: str) -> Dict[str, Any]:
        """Load a baseline by name.

        Parameters
        ----------
        name:
            Baseline name (without ``.json`` extension).

        Returns
        -------
        dict
            The parsed baseline JSON.

        Raises
        ------
        FileNotFoundError
            If the baseline file does not exist.
        """
        baseline_path = self._dir / f"{name}.json"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline '{name}' not found. Expected file: '{baseline_path}'. "
                "Run 'beyondbench baseline save' to create one."
            )
        with baseline_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug("Loaded baseline '%s' from '%s'", name, baseline_path)
        return data

    def list_baselines(self) -> List[str]:
        """Return names of all saved baselines (without ``.json`` suffix)."""
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def compare(
        self,
        current_results_dir: str | Path,
        baseline_name: str,
    ) -> Dict[str, Any]:
        """Compare current evaluation results against a stored baseline.

        Parameters
        ----------
        current_results_dir:
            Directory containing the current run's ``final_results.json``.
        baseline_name:
            Name of the previously saved baseline.

        Returns
        -------
        dict with keys:
            - ``baseline_name`` – name of the baseline used
            - ``improved`` – list of task names where accuracy improved
            - ``regressed`` – list of task names where accuracy dropped
            - ``unchanged`` – list of task names with no change
            - ``new_tasks`` – tasks in current results but not in baseline
            - ``missing_tasks`` – tasks in baseline but not in current results
            - ``task_deltas`` – per-task dict of metric deltas
            - ``summary`` – overall statistical summary
        """
        logger.info(
            "Comparing results in '%s' against baseline '%s'",
            current_results_dir,
            baseline_name,
        )
        baseline = self.load(baseline_name)
        current_data = _load_final_results(current_results_dir)

        baseline_tasks: Dict[str, Dict[str, float]] = baseline.get("task_baselines", {})
        current_tasks: Dict[str, Dict[str, float]] = {}
        for task_name, entries in current_data["task_results"].items():
            current_tasks[task_name] = _extract_task_metrics(entries)

        all_tasks = set(baseline_tasks) | set(current_tasks)
        task_deltas: Dict[str, Dict[str, Any]] = {}
        improved: List[str] = []
        regressed: List[str] = []
        unchanged: List[str] = []
        new_tasks: List[str] = []
        missing_tasks: List[str] = []

        for task in sorted(all_tasks):
            if task not in baseline_tasks:
                new_tasks.append(task)
                task_deltas[task] = {
                    "status": "new",
                    "current": current_tasks[task],
                    "baseline": None,
                    "accuracy_delta": None,
                    "parsing_success_rate_delta": None,
                    "generation_success_rate_delta": None,
                }
                continue

            if task not in current_tasks:
                missing_tasks.append(task)
                task_deltas[task] = {
                    "status": "missing",
                    "current": None,
                    "baseline": baseline_tasks[task],
                    "accuracy_delta": None,
                    "parsing_success_rate_delta": None,
                    "generation_success_rate_delta": None,
                }
                continue

            b = baseline_tasks[task]
            c = current_tasks[task]
            acc_delta = c["accuracy"] - b["accuracy"]
            parse_delta = c["parsing_success_rate"] - b["parsing_success_rate"]
            gen_delta = c["generation_success_rate"] - b["generation_success_rate"]

            if acc_delta > 1e-9:
                status = "improved"
                improved.append(task)
            elif acc_delta < -1e-9:
                status = "regressed"
                regressed.append(task)
            else:
                status = "unchanged"
                unchanged.append(task)

            task_deltas[task] = {
                "status": status,
                "current": c,
                "baseline": b,
                "accuracy_delta": round(acc_delta, 6),
                "parsing_success_rate_delta": round(parse_delta, 6),
                "generation_success_rate_delta": round(gen_delta, 6),
            }

        # Build summary statistics
        compared_tasks = [
            t for t in task_deltas
            if task_deltas[t]["accuracy_delta"] is not None
        ]
        accuracy_deltas = [task_deltas[t]["accuracy_delta"] for t in compared_tasks]

        summary: Dict[str, Any] = {
            "total_tasks_compared": len(compared_tasks),
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "unchanged_count": len(unchanged),
            "new_tasks_count": len(new_tasks),
            "missing_tasks_count": len(missing_tasks),
        }
        if accuracy_deltas:
            summary["mean_accuracy_delta"] = round(statistics.mean(accuracy_deltas), 6)
            summary["max_regression"] = round(min(accuracy_deltas), 6)
            summary["max_improvement"] = round(max(accuracy_deltas), 6)
        else:
            summary["mean_accuracy_delta"] = 0.0
            summary["max_regression"] = 0.0
            summary["max_improvement"] = 0.0

        return {
            "baseline_name": baseline_name,
            "baseline_timestamp": baseline.get("timestamp"),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "new_tasks": new_tasks,
            "missing_tasks": missing_tasks,
            "task_deltas": task_deltas,
            "summary": summary,
        }

    def check_regression(
        self,
        current_results_dir: str | Path,
        baseline_name: str,
        accuracy_threshold: float = 0.10,
        parser_threshold: float = 0.05,
    ) -> bool:
        """Return True if any task has regressed beyond the specified thresholds.

        Parameters
        ----------
        current_results_dir:
            Directory containing current ``final_results.json``.
        baseline_name:
            Name of the baseline to compare against.
        accuracy_threshold:
            Maximum allowed accuracy drop (default 10 %). Any task with an
            accuracy drop *greater than* this value triggers regression.
        parser_threshold:
            Maximum allowed parser-success-rate drop (default 5 %).

        Returns
        -------
        bool
            ``True`` if a regression is detected, ``False`` otherwise.
        """
        report = self.compare(current_results_dir, baseline_name)
        task_deltas = report["task_deltas"]

        for task, delta in task_deltas.items():
            acc_delta = delta.get("accuracy_delta")
            parse_delta = delta.get("parsing_success_rate_delta")

            if acc_delta is not None and acc_delta < -accuracy_threshold:
                logger.warning(
                    "Regression detected in task '%s': accuracy dropped by %.4f (threshold %.4f)",
                    task,
                    abs(acc_delta),
                    accuracy_threshold,
                )
                return True

            if parse_delta is not None and parse_delta < -parser_threshold:
                logger.warning(
                    "Regression detected in task '%s': parser success rate dropped by %.4f (threshold %.4f)",
                    task,
                    abs(parse_delta),
                    parser_threshold,
                )
                return True

        return False

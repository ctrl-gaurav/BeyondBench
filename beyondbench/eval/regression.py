"""Regression detection utilities for CI integration.

Provides RegressionChecker (a thin wrapper around BaselineManager) and the
standalone ``run_regression_check`` function that can be called from CI
scripts and returns a POSIX exit code (0 = pass, 1 = fail).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging_utils import get_logger
from .baseline import BaselineManager

logger = get_logger(__name__)


class RegressionChecker:
    """Convenience wrapper around :class:`BaselineManager` for CI usage.

    Example::

        checker = RegressionChecker()
        exit_code = checker.run("./new_results", "v0.2.0-qwen-1.5b")
        sys.exit(exit_code)
    """

    def __init__(self, baselines_dir: Optional[str | Path] = None) -> None:
        self._manager = BaselineManager(baselines_dir=baselines_dir)

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def check_regression(
        self,
        current_results_dir: str | Path,
        baseline_name: str,
        accuracy_threshold: float = 0.10,
        parser_threshold: float = 0.05,
    ) -> bool:
        """Delegate to :meth:`BaselineManager.check_regression`."""
        return self._manager.check_regression(
            current_results_dir=current_results_dir,
            baseline_name=baseline_name,
            accuracy_threshold=accuracy_threshold,
            parser_threshold=parser_threshold,
        )

    def compare(
        self,
        current_results_dir: str | Path,
        baseline_name: str,
    ) -> Dict[str, Any]:
        """Delegate to :meth:`BaselineManager.compare`."""
        return self._manager.compare(
            current_results_dir=current_results_dir,
            baseline_name=baseline_name,
        )

    def run(
        self,
        current_results_dir: str | Path,
        baseline_name: str,
        accuracy_threshold: float = 0.10,
        parser_threshold: float = 0.05,
        print_report: bool = True,
    ) -> int:
        """Run regression check and return POSIX exit code.

        Parameters
        ----------
        current_results_dir:
            Directory containing the current run's ``final_results.json``.
        baseline_name:
            Name of the baseline to compare against.
        accuracy_threshold:
            Max allowed accuracy drop before a regression is declared.
        parser_threshold:
            Max allowed parser-success-rate drop.
        print_report:
            If True, print a human-readable report to stdout.

        Returns
        -------
        int
            0 if no regression detected, 1 otherwise.
        """
        try:
            report = self.compare(current_results_dir, baseline_name)
        except FileNotFoundError as exc:
            logger.error("Regression check failed: %s", exc)
            if print_report:
                print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        regressed = self._manager.check_regression(
            current_results_dir=current_results_dir,
            baseline_name=baseline_name,
            accuracy_threshold=accuracy_threshold,
            parser_threshold=parser_threshold,
        )

        if print_report:
            _print_regression_report(report, accuracy_threshold, parser_threshold)

        return 1 if regressed else 0


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def run_regression_check(
    results_dir: str | Path,
    baseline_name: str,
    accuracy_threshold: float = 0.10,
    parser_threshold: float = 0.05,
    baselines_dir: Optional[str | Path] = None,
    print_report: bool = True,
) -> int:
    """Run a regression check and return a POSIX exit code.

    This function is designed to be called from CI pipelines or post-evaluation
    scripts.  It returns 0 when all tasks pass the thresholds and 1 when at
    least one regression is detected.

    Parameters
    ----------
    results_dir:
        Directory containing the current run's ``final_results.json``.
    baseline_name:
        Name of the baseline to compare against.
    accuracy_threshold:
        Max allowed accuracy drop (default 10 %).
    parser_threshold:
        Max allowed parser-success-rate drop (default 5 %).
    baselines_dir:
        Override the default baselines directory.
    print_report:
        If True, print a human-readable report to stdout.

    Returns
    -------
    int
        0 (no regression) or 1 (regression detected).
    """
    checker = RegressionChecker(baselines_dir=baselines_dir)
    return checker.run(
        current_results_dir=results_dir,
        baseline_name=baseline_name,
        accuracy_threshold=accuracy_threshold,
        parser_threshold=parser_threshold,
        print_report=print_report,
    )


# ---------------------------------------------------------------------------
# Internal report formatter
# ---------------------------------------------------------------------------


def _print_regression_report(
    report: Dict[str, Any],
    accuracy_threshold: float,
    parser_threshold: float,
) -> None:
    """Print a human-readable regression report to stdout."""
    sep = "=" * 70
    print(sep)
    print("  BEYONDBENCH REGRESSION REPORT")
    print(sep)
    print(f"  Baseline : {report['baseline_name']}")
    if report.get("baseline_timestamp"):
        print(f"  Saved at : {report['baseline_timestamp']}")
    print(f"  Thresholds: accuracy > {accuracy_threshold:.0%} drop, "
          f"parser > {parser_threshold:.0%} drop")
    print()

    summary = report["summary"]
    print(f"  Tasks compared : {summary['total_tasks_compared']}")
    print(f"  Improved       : {summary['improved_count']}")
    print(f"  Regressed      : {summary['regressed_count']}")
    print(f"  Unchanged      : {summary['unchanged_count']}")
    if summary["new_tasks_count"]:
        print(f"  New tasks      : {summary['new_tasks_count']}")
    if summary["missing_tasks_count"]:
        print(f"  Missing tasks  : {summary['missing_tasks_count']}")
    mean_delta = summary.get("mean_accuracy_delta", 0.0)
    print(f"  Mean accuracy delta : {mean_delta:+.4f}")
    print()

    task_deltas = report["task_deltas"]

    # --- Regressions ---
    regressed_tasks = [t for t, d in task_deltas.items() if d["status"] == "regressed"]
    if regressed_tasks:
        print("  REGRESSIONS:")
        print("  " + "-" * 58)
        for task in regressed_tasks:
            d = task_deltas[task]
            acc_delta = d["accuracy_delta"]
            parse_delta = d["parsing_success_rate_delta"]
            flags = []
            if acc_delta is not None and acc_delta < -accuracy_threshold:
                flags.append("EXCEEDS_THRESHOLD")
            marker = " [!]" if flags else ""
            print(
                f"    {task:<28s}  acc: {acc_delta:+.4f}"
                f"  parse: {parse_delta:+.4f}{marker}"
            )
        print()

    # --- Improvements ---
    improved_tasks = [t for t, d in task_deltas.items() if d["status"] == "improved"]
    if improved_tasks:
        print("  IMPROVEMENTS:")
        print("  " + "-" * 58)
        for task in improved_tasks:
            d = task_deltas[task]
            print(
                f"    {task:<28s}  acc: {d['accuracy_delta']:+.4f}"
                f"  parse: {d['parsing_success_rate_delta']:+.4f}"
            )
        print()

    # --- New / missing ---
    if report["new_tasks"]:
        print(f"  NEW TASKS  : {', '.join(report['new_tasks'])}")
        print()
    if report["missing_tasks"]:
        print(f"  MISSING TASKS: {', '.join(report['missing_tasks'])}")
        print()

    # Final verdict
    any_regression = any(
        d["status"] == "regressed"
        and d["accuracy_delta"] is not None
        and d["accuracy_delta"] < -accuracy_threshold
        for d in task_deltas.values()
    ) or any(
        d.get("parsing_success_rate_delta") is not None
        and d["parsing_success_rate_delta"] < -parser_threshold
        for d in task_deltas.values()
    )

    if any_regression:
        print("  RESULT: FAIL - regression(s) detected beyond threshold")
    else:
        print("  RESULT: PASS - no regressions beyond threshold")

    print(sep)

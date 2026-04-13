"""Comprehensive tests for Phase 19: Regression Testing & Baseline Management.

Tests cover:
  - BaselineManager.save / load round-trip
  - BaselineManager.compare (improvements, regressions, unchanged)
  - BaselineManager.check_regression (threshold detection)
  - Missing baseline / invalid results error handling
  - RegressionChecker and run_regression_check
  - CLI commands: baseline save, compare, list
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_final_results(
    tasks: Dict[str, list[dict]],
    model_name: str = "test-model",
) -> Dict[str, Any]:
    """Build a minimal final_results.json payload."""
    all_acc = []
    for entries in tasks.values():
        for e in entries:
            all_acc.append(e.get("accuracy", 0.0))
    avg = sum(all_acc) / len(all_acc) if all_acc else 0.0
    return {
        "summary": {
            "total_tasks": len(tasks),
            "avg_accuracy": avg,
            "total_evaluations": sum(len(v) for v in tasks.values()),
        },
        "task_results": tasks,
        "overall_stats": {},
        "model_info": {"model_name": model_name, "backend": "mock"},
        "evaluation_config": {"suite": "easy"},
    }


def _write_results(tmp_dir: Path, tasks: Dict[str, list[dict]], model_name: str = "test-model") -> Path:
    """Write a final_results.json into *tmp_dir* and return the dir."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    payload = _make_final_results(tasks, model_name=model_name)
    (tmp_dir / "final_results.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def baselines_tmp(tmp_path: Path) -> Path:
    """A fresh temporary directory for storing baselines."""
    d = tmp_path / "baselines"
    d.mkdir()
    return d


@pytest.fixture()
def results_v1(tmp_path: Path) -> Path:
    """First evaluation results (baseline snapshot)."""
    tasks = {
        "sum": [
            {"accuracy": 0.9, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
            {"accuracy": 0.8, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
        ],
        "sorting": [
            {"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0},
        ],
        "max_value": [
            {"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
        ],
    }
    return _write_results(tmp_path / "v1_results", tasks)


@pytest.fixture()
def results_v2_improved(tmp_path: Path) -> Path:
    """Second evaluation - sum improved, sorting improved, max_value same."""
    tasks = {
        "sum": [
            {"accuracy": 0.95, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
            {"accuracy": 0.90, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
        ],
        "sorting": [
            {"accuracy": 0.75, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
        ],
        "max_value": [
            {"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
        ],
    }
    return _write_results(tmp_path / "v2_improved", tasks)


@pytest.fixture()
def results_v2_regressed(tmp_path: Path) -> Path:
    """Second evaluation - sum regressed significantly, sorting unchanged."""
    tasks = {
        "sum": [
            {"accuracy": 0.5, "parsing_success_rate": 0.8, "generation_success_rate": 1.0},
            {"accuracy": 0.4, "parsing_success_rate": 0.8, "generation_success_rate": 1.0},
        ],
        "sorting": [
            {"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0},
        ],
        "max_value": [
            {"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0},
        ],
    }
    return _write_results(tmp_path / "v2_regressed", tasks)


# ---------------------------------------------------------------------------
# BaselineManager: save / load round-trip
# ---------------------------------------------------------------------------


class TestBaselineManagerSaveLoad:
    def test_save_creates_json_file(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        out_path = mgr.save(results_v1, "test-v1")

        assert out_path.exists(), "Baseline file should be created"
        assert out_path.suffix == ".json"
        assert out_path.stem == "test-v1"

    def test_save_returns_correct_path(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        out_path = mgr.save(results_v1, "my-baseline")
        assert out_path == baselines_tmp / "my-baseline.json"

    def test_save_baseline_has_required_fields(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "test-v1")
        data = mgr.load("test-v1")

        assert data["name"] == "test-v1"
        assert "timestamp" in data
        assert "model_info" in data
        assert "evaluation_config" in data
        assert "task_baselines" in data

    def test_save_captures_all_tasks(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "test-v1")
        data = mgr.load("test-v1")

        assert set(data["task_baselines"].keys()) == {"sum", "sorting", "max_value"}

    def test_save_averages_accuracy_across_folds(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "test-v1")
        data = mgr.load("test-v1")

        # sum task: (0.9 + 0.8) / 2 = 0.85
        assert abs(data["task_baselines"]["sum"]["accuracy"] - 0.85) < 1e-9

    def test_save_captures_parsing_success_rate(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "test-v1")
        data = mgr.load("test-v1")

        assert data["task_baselines"]["sum"]["parsing_success_rate"] == 1.0
        assert abs(data["task_baselines"]["sorting"]["parsing_success_rate"] - 0.95) < 1e-9

    def test_save_captures_model_info(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "test-v1")
        data = mgr.load("test-v1")

        assert data["model_info"]["model_name"] == "test-model"

    def test_load_missing_baseline_raises_file_not_found(self, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        with pytest.raises(FileNotFoundError, match="not found"):
            mgr.load("nonexistent-baseline")

    def test_save_overwrites_existing(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "test-v1")
        # Overwrite with a different results (single-task results)
        d = baselines_tmp.parent / "single"
        tasks = {"only_task": [{"accuracy": 0.5, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}]}
        _write_results(d, tasks)
        mgr.save(d, "test-v1")
        data = mgr.load("test-v1")
        assert set(data["task_baselines"].keys()) == {"only_task"}

    def test_list_baselines_empty(self, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        assert mgr.list_baselines() == []

    def test_list_baselines_returns_names(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "alpha")
        mgr.save(results_v1, "beta")
        names = mgr.list_baselines()
        assert "alpha" in names
        assert "beta" in names

    def test_single_fold_results(self, baselines_tmp, tmp_path):
        """Baseline should work when each task has exactly one fold entry."""
        from beyondbench.eval.baseline import BaselineManager

        tasks = {"sum": [{"accuracy": 0.75, "parsing_success_rate": 0.9, "generation_success_rate": 1.0}]}
        d = _write_results(tmp_path / "single_fold", tasks)
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(d, "single-fold")
        data = mgr.load("single-fold")
        assert abs(data["task_baselines"]["sum"]["accuracy"] - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# BaselineManager: compare
# ---------------------------------------------------------------------------


class TestBaselineManagerCompare:
    def test_compare_returns_improved_tasks(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(results_v2_improved, "v1")

        assert "sum" in report["improved"]
        assert "sorting" in report["improved"]

    def test_compare_returns_regressed_tasks(self, results_v1, results_v2_regressed, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(results_v2_regressed, "v1")

        assert "sum" in report["regressed"]

    def test_compare_unchanged_task(self, results_v1, results_v2_regressed, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(results_v2_regressed, "v1")

        # sorting is unchanged (0.6 in both)
        assert "sorting" in report["unchanged"]
        # max_value is unchanged (0.7 in both)
        assert "max_value" in report["unchanged"]

    def test_compare_task_deltas_accuracy(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(results_v2_improved, "v1")

        # sum: baseline = 0.85, current = (0.95+0.90)/2 = 0.925 => delta = +0.075
        acc_delta = report["task_deltas"]["sum"]["accuracy_delta"]
        assert abs(acc_delta - 0.075) < 1e-6

    def test_compare_task_deltas_negative(self, results_v1, results_v2_regressed, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(results_v2_regressed, "v1")

        # sum: baseline = 0.85, current = (0.5+0.4)/2 = 0.45 => delta = -0.40
        acc_delta = report["task_deltas"]["sum"]["accuracy_delta"]
        assert acc_delta < -0.35

    def test_compare_summary_counts(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(results_v2_improved, "v1")

        s = report["summary"]
        assert s["total_tasks_compared"] == 3
        assert s["improved_count"] == 2
        assert s["regressed_count"] == 0
        assert s["unchanged_count"] == 1

    def test_compare_new_tasks_detected(self, results_v1, baselines_tmp, tmp_path):
        """A task present in current results but not in baseline appears in new_tasks."""
        from beyondbench.eval.baseline import BaselineManager

        extra_tasks = {
            "sum": [{"accuracy": 0.9, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
            "sorting": [{"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0}],
            "max_value": [{"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
            "brand_new_task": [{"accuracy": 0.8, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
        }
        d = _write_results(tmp_path / "extra", extra_tasks)

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(d, "v1")

        assert "brand_new_task" in report["new_tasks"]

    def test_compare_missing_tasks_detected(self, results_v1, baselines_tmp, tmp_path):
        """A task in baseline but absent from current appears in missing_tasks."""
        from beyondbench.eval.baseline import BaselineManager

        fewer_tasks = {
            "sum": [{"accuracy": 0.9, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
        }
        d = _write_results(tmp_path / "fewer", fewer_tasks)

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(d, "v1")

        assert "sorting" in report["missing_tasks"]
        assert "max_value" in report["missing_tasks"]

    def test_compare_baseline_name_in_report(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "my-named-baseline")
        report = mgr.compare(results_v2_improved, "my-named-baseline")
        assert report["baseline_name"] == "my-named-baseline"

    def test_compare_missing_baseline_raises(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        with pytest.raises(FileNotFoundError):
            mgr.compare(results_v1, "ghost-baseline")

    def test_compare_parsing_success_rate_delta(self, results_v1, results_v2_regressed, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        report = mgr.compare(results_v2_regressed, "v1")

        # sum: baseline parsing = 1.0, current parsing = 0.8 => delta = -0.2
        parse_delta = report["task_deltas"]["sum"]["parsing_success_rate_delta"]
        assert abs(parse_delta - (-0.2)) < 1e-6


# ---------------------------------------------------------------------------
# BaselineManager: check_regression
# ---------------------------------------------------------------------------


class TestBaselineManagerCheckRegression:
    def test_check_regression_no_regression(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        assert mgr.check_regression(results_v2_improved, "v1") is False

    def test_check_regression_detects_regression(self, results_v1, results_v2_regressed, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        # sum drops by ~0.40, well above 0.10 threshold
        assert mgr.check_regression(results_v2_regressed, "v1") is True

    def test_check_regression_below_threshold_passes(self, results_v1, baselines_tmp, tmp_path):
        """A small drop (5%) should pass with default 10% threshold."""
        from beyondbench.eval.baseline import BaselineManager

        tasks = {
            "sum": [{"accuracy": 0.80, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
            "sorting": [{"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0}],
            "max_value": [{"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
        }
        # sum baseline is 0.85, current 0.80 => drop of 0.05 (< 0.10 threshold)
        d = _write_results(tmp_path / "slight_drop", tasks)
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        assert mgr.check_regression(d, "v1", accuracy_threshold=0.10) is False

    def test_check_regression_custom_threshold(self, results_v1, baselines_tmp, tmp_path):
        """A 5% drop should trigger regression when threshold is set to 3%."""
        from beyondbench.eval.baseline import BaselineManager

        tasks = {
            "sum": [{"accuracy": 0.80, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
            "sorting": [{"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0}],
            "max_value": [{"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
        }
        d = _write_results(tmp_path / "slight_drop2", tasks)
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        assert mgr.check_regression(d, "v1", accuracy_threshold=0.03) is True

    def test_check_regression_parser_threshold(self, results_v1, baselines_tmp, tmp_path):
        """A parser drop of 10% should trigger regression with parser_threshold=0.05."""
        from beyondbench.eval.baseline import BaselineManager

        tasks = {
            "sum": [{"accuracy": 0.85, "parsing_success_rate": 0.88, "generation_success_rate": 1.0}],
            "sorting": [{"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0}],
            "max_value": [{"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
        }
        # sum: baseline parsing = 1.0, current = 0.88 => drop = 0.12 > 0.05
        d = _write_results(tmp_path / "parse_drop", tasks)
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        assert mgr.check_regression(d, "v1", parser_threshold=0.05) is True

    def test_check_regression_parser_below_threshold(self, results_v1, baselines_tmp, tmp_path):
        """A small parser drop (2%) should pass with parser_threshold=0.05."""
        from beyondbench.eval.baseline import BaselineManager

        tasks = {
            "sum": [{"accuracy": 0.85, "parsing_success_rate": 0.98, "generation_success_rate": 1.0}],
            "sorting": [{"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0}],
            "max_value": [{"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
        }
        d = _write_results(tmp_path / "small_parse_drop", tasks)
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        assert mgr.check_regression(d, "v1", parser_threshold=0.05) is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_save_missing_final_results_raises(self, baselines_tmp, tmp_path):
        from beyondbench.eval.baseline import BaselineManager

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        with pytest.raises(FileNotFoundError, match="final_results.json"):
            mgr.save(empty_dir, "will-fail")

    def test_save_malformed_json_raises(self, baselines_tmp, tmp_path):
        from beyondbench.eval.baseline import BaselineManager

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "final_results.json").write_text("not-json-at-all", encoding="utf-8")
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        with pytest.raises(Exception):
            mgr.save(bad_dir, "will-fail")

    def test_save_missing_task_results_key_raises(self, baselines_tmp, tmp_path):
        from beyondbench.eval.baseline import BaselineManager

        bad_dir = tmp_path / "no_task_results"
        bad_dir.mkdir()
        (bad_dir / "final_results.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        with pytest.raises(ValueError, match="task_results"):
            mgr.save(bad_dir, "will-fail")

    def test_compare_missing_results_dir_raises(self, results_v1, baselines_tmp, tmp_path):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        mgr.save(results_v1, "v1")
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            mgr.compare(nonexistent, "v1")

    def test_check_regression_missing_baseline_raises(self, results_v1, baselines_tmp):
        from beyondbench.eval.baseline import BaselineManager

        mgr = BaselineManager(baselines_dir=baselines_tmp)
        with pytest.raises(FileNotFoundError):
            mgr.check_regression(results_v1, "missing-baseline")

    def test_empty_task_results(self, baselines_tmp, tmp_path):
        """Empty task_results dict should not crash save."""
        from beyondbench.eval.baseline import BaselineManager

        d = tmp_path / "empty_tasks"
        d.mkdir()
        payload = _make_final_results({})
        (d / "final_results.json").write_text(json.dumps(payload), encoding="utf-8")
        mgr = BaselineManager(baselines_dir=baselines_tmp)
        out = mgr.save(d, "empty")
        data = mgr.load("empty")
        assert data["task_baselines"] == {}


# ---------------------------------------------------------------------------
# RegressionChecker and run_regression_check
# ---------------------------------------------------------------------------


class TestRegressionChecker:
    def test_run_returns_0_when_no_regression(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.regression import RegressionChecker

        checker = RegressionChecker(baselines_dir=baselines_tmp)
        from beyondbench.eval.baseline import BaselineManager
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")

        code = checker.run(results_v2_improved, "v1", print_report=False)
        assert code == 0

    def test_run_returns_1_when_regression(self, results_v1, results_v2_regressed, baselines_tmp):
        from beyondbench.eval.regression import RegressionChecker

        checker = RegressionChecker(baselines_dir=baselines_tmp)
        from beyondbench.eval.baseline import BaselineManager
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")

        code = checker.run(results_v2_regressed, "v1", print_report=False)
        assert code == 1

    def test_run_returns_1_on_missing_baseline(self, results_v1, baselines_tmp):
        from beyondbench.eval.regression import RegressionChecker

        checker = RegressionChecker(baselines_dir=baselines_tmp)
        code = checker.run(results_v1, "ghost", print_report=False)
        assert code == 1

    def test_run_prints_report(self, results_v1, results_v2_improved, baselines_tmp, capsys):
        from beyondbench.eval.regression import RegressionChecker
        from beyondbench.eval.baseline import BaselineManager

        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")
        checker = RegressionChecker(baselines_dir=baselines_tmp)
        checker.run(results_v2_improved, "v1", print_report=True)
        captured = capsys.readouterr()
        assert "BEYONDBENCH REGRESSION REPORT" in captured.out

    def test_compare_delegates_to_baseline_manager(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.regression import RegressionChecker
        from beyondbench.eval.baseline import BaselineManager

        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")
        checker = RegressionChecker(baselines_dir=baselines_tmp)
        report = checker.compare(results_v2_improved, "v1")
        assert "task_deltas" in report
        assert "improved" in report


class TestRunRegressionCheck:
    def test_returns_0_no_regression(self, results_v1, results_v2_improved, baselines_tmp):
        from beyondbench.eval.regression import run_regression_check
        from beyondbench.eval.baseline import BaselineManager

        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")
        code = run_regression_check(
            results_v2_improved, "v1", baselines_dir=baselines_tmp, print_report=False
        )
        assert code == 0

    def test_returns_1_regression(self, results_v1, results_v2_regressed, baselines_tmp):
        from beyondbench.eval.regression import run_regression_check
        from beyondbench.eval.baseline import BaselineManager

        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")
        code = run_regression_check(
            results_v2_regressed, "v1", baselines_dir=baselines_tmp, print_report=False
        )
        assert code == 1

    def test_custom_thresholds(self, results_v1, baselines_tmp, tmp_path):
        from beyondbench.eval.regression import run_regression_check
        from beyondbench.eval.baseline import BaselineManager

        # Drop of 5% on sum; passes with 10% threshold, fails with 3%
        tasks = {
            "sum": [{"accuracy": 0.80, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
            "sorting": [{"accuracy": 0.6, "parsing_success_rate": 0.95, "generation_success_rate": 1.0}],
            "max_value": [{"accuracy": 0.7, "parsing_success_rate": 1.0, "generation_success_rate": 1.0}],
        }
        d = _write_results(tmp_path / "slight_drop3", tasks)
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")

        assert run_regression_check(d, "v1", accuracy_threshold=0.10,
                                    baselines_dir=baselines_tmp, print_report=False) == 0
        assert run_regression_check(d, "v1", accuracy_threshold=0.03,
                                    baselines_dir=baselines_tmp, print_report=False) == 1


# ---------------------------------------------------------------------------
# CLI: beyondbench baseline save
# ---------------------------------------------------------------------------


def _get_runner():
    import inspect
    try:
        params = inspect.signature(CliRunner).parameters
    except (TypeError, ValueError):
        params = {}
    if "mix_stderr" in params:
        return CliRunner(mix_stderr=False)
    return CliRunner()


def _stdout(result) -> str:
    s = getattr(result, "stdout", None)
    if s:
        return s
    return result.output


class TestCLIBaselineSave:
    def test_save_command_exists(self):
        from beyondbench.cli.main import main
        assert "baseline" in main.commands
        assert "save" in main.commands["baseline"].commands

    def test_save_command_succeeds(self, results_v1, baselines_tmp, monkeypatch):
        from beyondbench.cli.main import main
        from beyondbench.eval.baseline import BaselineManager

        monkeypatch.setattr(
            "beyondbench.eval.baseline._DEFAULT_BASELINES_DIR",
            baselines_tmp,
        )
        runner = _get_runner()
        result = runner.invoke(
            main,
            ["baseline", "save", "--results-dir", str(results_v1), "--name", "cli-test"],
        )
        assert result.exit_code == 0, f"exit_code={result.exit_code} output={result.output}"
        assert "cli-test" in _stdout(result)

    def test_save_command_creates_file(self, results_v1, baselines_tmp, monkeypatch):
        from beyondbench.cli.main import main

        monkeypatch.setattr(
            "beyondbench.eval.baseline._DEFAULT_BASELINES_DIR",
            baselines_tmp,
        )
        runner = _get_runner()
        runner.invoke(
            main,
            ["baseline", "save", "--results-dir", str(results_v1), "--name", "cli-file-test"],
        )
        assert (baselines_tmp / "cli-file-test.json").exists()

    def test_save_command_missing_results_dir_errors(self):
        from beyondbench.cli.main import main

        runner = _get_runner()
        result = runner.invoke(
            main,
            ["baseline", "save", "--results-dir", "/nonexistent/path/xyz", "--name", "bad"],
        )
        assert result.exit_code != 0

    def test_save_command_missing_name_errors(self, results_v1):
        from beyondbench.cli.main import main

        runner = _get_runner()
        result = runner.invoke(
            main,
            ["baseline", "save", "--results-dir", str(results_v1)],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: beyondbench baseline compare
# ---------------------------------------------------------------------------


class TestCLIBaselineCompare:
    def test_compare_command_exists(self):
        from beyondbench.cli.main import main
        assert "compare" in main.commands["baseline"].commands

    def test_compare_no_regression_exit_0(self, results_v1, results_v2_improved, baselines_tmp, monkeypatch):
        from beyondbench.cli.main import main

        monkeypatch.setattr("beyondbench.eval.baseline._DEFAULT_BASELINES_DIR", baselines_tmp)
        monkeypatch.setattr("beyondbench.eval.regression._DEFAULT_BASELINES_DIR", baselines_tmp, raising=False)

        from beyondbench.eval.baseline import BaselineManager
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")

        runner = _get_runner()
        result = runner.invoke(
            main,
            ["baseline", "compare", "--results-dir", str(results_v2_improved), "--baseline", "v1"],
        )
        # Should not exit with error when no --fail-on-regression
        assert result.exit_code == 0, f"exit_code={result.exit_code}\n{result.output}"

    def test_compare_regression_no_fail_flag_exit_0(self, results_v1, results_v2_regressed, baselines_tmp, monkeypatch):
        """Without --fail-on-regression, exit code should be 0 even on regression."""
        from beyondbench.cli.main import main

        monkeypatch.setattr("beyondbench.eval.baseline._DEFAULT_BASELINES_DIR", baselines_tmp)

        from beyondbench.eval.baseline import BaselineManager
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")

        runner = _get_runner()
        result = runner.invoke(
            main,
            ["baseline", "compare", "--results-dir", str(results_v2_regressed), "--baseline", "v1"],
        )
        assert result.exit_code == 0

    def test_compare_regression_with_fail_flag_exit_1(self, results_v1, results_v2_regressed, baselines_tmp, monkeypatch):
        from beyondbench.cli.main import main

        monkeypatch.setattr("beyondbench.eval.baseline._DEFAULT_BASELINES_DIR", baselines_tmp)

        from beyondbench.eval.baseline import BaselineManager
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")

        runner = _get_runner()
        result = runner.invoke(
            main,
            [
                "baseline", "compare",
                "--results-dir", str(results_v2_regressed),
                "--baseline", "v1",
                "--fail-on-regression",
            ],
        )
        assert result.exit_code == 1

    def test_compare_prints_report(self, results_v1, results_v2_improved, baselines_tmp, monkeypatch):
        from beyondbench.cli.main import main

        monkeypatch.setattr("beyondbench.eval.baseline._DEFAULT_BASELINES_DIR", baselines_tmp)

        from beyondbench.eval.baseline import BaselineManager
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "v1")

        runner = _get_runner()
        result = runner.invoke(
            main,
            ["baseline", "compare", "--results-dir", str(results_v2_improved), "--baseline", "v1"],
        )
        output = result.output
        assert "REGRESSION REPORT" in output or "PASS" in output or "FAIL" in output

    def test_compare_missing_baseline_errors(self, results_v1):
        from beyondbench.cli.main import main

        runner = _get_runner()
        result = runner.invoke(
            main,
            ["baseline", "compare", "--results-dir", str(results_v1), "--baseline", "does-not-exist"],
        )
        assert result.exit_code != 0 or "Error" in result.output or "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI: beyondbench baseline list
# ---------------------------------------------------------------------------


class TestCLIBaselineList:
    def test_list_command_exists(self):
        from beyondbench.cli.main import main
        assert "list" in main.commands["baseline"].commands

    def test_list_empty(self, baselines_tmp, monkeypatch):
        from beyondbench.cli.main import main

        monkeypatch.setattr("beyondbench.eval.baseline._DEFAULT_BASELINES_DIR", baselines_tmp)
        runner = _get_runner()
        result = runner.invoke(main, ["baseline", "list"])
        assert result.exit_code == 0
        assert "No baselines" in _stdout(result)

    def test_list_shows_saved_baselines(self, results_v1, baselines_tmp, monkeypatch):
        from beyondbench.cli.main import main

        monkeypatch.setattr("beyondbench.eval.baseline._DEFAULT_BASELINES_DIR", baselines_tmp)
        from beyondbench.eval.baseline import BaselineManager
        BaselineManager(baselines_dir=baselines_tmp).save(results_v1, "my-baseline-abc")

        runner = _get_runner()
        result = runner.invoke(main, ["baseline", "list"])
        assert "my-baseline-abc" in _stdout(result)


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


class TestImports:
    def test_eval_module_importable(self):
        import beyondbench.eval
        assert hasattr(beyondbench.eval, "BaselineManager")
        assert hasattr(beyondbench.eval, "RegressionChecker")
        assert hasattr(beyondbench.eval, "run_regression_check")

    def test_baseline_importable(self):
        from beyondbench.eval.baseline import BaselineManager
        assert BaselineManager is not None

    def test_regression_importable(self):
        from beyondbench.eval.regression import RegressionChecker, run_regression_check
        assert RegressionChecker is not None
        assert run_regression_check is not None

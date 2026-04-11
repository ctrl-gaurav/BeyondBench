"""
Unit tests for Phase 7 (Token & Cost Analytics) and Phase 8 (Reporting).

Tests cover:
- TokenCounter: counting, per-request tracking, analytics, per-task analytics
- CostTracker: pricing lookup, accumulation, estimation, reporting
- RequestLogger: logging, JSONL persistence, replay, summarize
- ReportGenerator: HTML, Markdown, LaTeX generation
- Visualizer: all 8 chart types (Plotly available), save/save_all
- `beyondbench report` CLI command
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_results() -> Dict[str, Any]:
    return {
        "model_id": "test-model",
        "timestamp": "2026-04-10T12:00:00",
        "summary": {
            "avg_accuracy": 0.72,
            "total_evaluations": 100,
            "total_tasks": 4,
            "total_duration": 60.0,
            "token_analytics": {
                "count": 100,
                "input": {"avg": 80.0, "min": 50, "max": 120, "p50": 78.0, "p95": 110.0, "total": 8000},
                "output": {"avg": 40.0, "min": 1, "max": 200, "p50": 35.0, "p95": 180.0, "total": 4000},
                "throughput": {"avg_tokens_per_second": 35.0, "avg_latency_ms": 1000.0},
            },
        },
        "task_results": {
            "sum": {"accuracy": 0.91, "avg_tokens": 35.0, "success_rate": 1.0},
            "sorting": {"accuracy": 0.82, "avg_tokens": 110.0, "success_rate": 1.0},
            "fibonacci_sequence": {"accuracy": 0.55, "avg_tokens": 280.0, "success_rate": 0.95},
            "sudoku_solving": {"accuracy": 0.20, "avg_tokens": 500.0, "success_rate": 0.80},
        },
    }


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ===========================================================================
# Phase 7 — Token & Cost Analytics
# ===========================================================================

class TestTokenCounter:
    def test_count_tokens_returns_positive_int(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        n = tc.count_tokens("Hello, world!")
        assert isinstance(n, int)
        assert n > 0

    def test_empty_string_returns_zero(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        assert tc.count_tokens("") == 0

    def test_count_is_reproducible(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        text = "The sum of [1, 2, 3] is 6."
        assert tc.count_tokens(text) == tc.count_tokens(text)

    def test_longer_text_has_more_tokens(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        short = tc.count_tokens("Hi")
        long = tc.count_tokens("Hi " * 100)
        assert long > short

    def test_record_request_returns_token_usage(self):
        from beyondbench.utils.token_counter import TokenCounter, TokenUsage
        tc = TokenCounter()
        usage = tc.record_request(
            prompt="What is 3+5?",
            response="8",
            latency_ms=100.0,
            task_name="sum",
            fold=0,
        )
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens
        assert usage.tokens_per_second > 0
        assert usage.task_name == "sum"
        assert usage.fold == 0
        assert usage.counting_method in ("tiktoken", "model_tokenizer", "estimate")

    def test_request_log_grows(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        assert len(tc.get_request_log()) == 0
        tc.record_request("p", "r", task_name="sum")
        tc.record_request("p", "r", task_name="mean")
        assert len(tc.get_request_log()) == 2

    def test_clear_log(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        tc.record_request("p", "r")
        tc.clear_log()
        assert len(tc.get_request_log()) == 0

    def test_compute_analytics_empty(self):
        from beyondbench.utils.token_counter import TokenCounter, TokenAnalytics
        tc = TokenCounter()
        analytics = tc.compute_analytics()
        assert isinstance(analytics, TokenAnalytics)
        assert analytics.count == 0

    def test_compute_analytics_statistics(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        for i in range(10):
            tc.record_request("x " * (50 + i), "y " * (10 + i), latency_ms=100.0 + i * 10)
        a = tc.compute_analytics()
        assert a.count == 10
        assert a.input_avg > 0
        assert a.input_min <= a.input_p50 <= a.input_max
        assert a.output_p50 <= a.output_p95
        assert a.total_input > 0
        assert a.total_output > 0
        assert a.avg_latency_ms > 0

    def test_compute_analytics_to_dict(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        tc.record_request("hello world", "hi", latency_ms=50.0)
        d = tc.compute_analytics().to_dict()
        assert "count" in d
        assert "input" in d
        assert "output" in d
        assert "throughput" in d
        assert "avg" in d["input"]
        assert "p95" in d["output"]

    def test_per_task_analytics(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        for _ in range(3):
            tc.record_request("sum prompt", "15", task_name="sum")
        for _ in range(5):
            tc.record_request("mean prompt text", "5.5", task_name="mean")
        task_analytics = tc.compute_task_analytics()
        assert "sum" in task_analytics
        assert "mean" in task_analytics
        assert task_analytics["sum"].count == 3
        assert task_analytics["mean"].count == 5

    def test_per_difficulty_analytics(self):
        from beyondbench.utils.token_counter import TokenCounter
        tc = TokenCounter()
        tc.record_request("p", "r", task_name="sum")
        tc.record_request("p", "r", task_name="sorting")
        tc.record_request("p", "r", task_name="sudoku_solving")
        difficulty_map = {"sum": "easy", "sorting": "easy", "sudoku_solving": "hard"}
        diff_analytics = tc.compute_difficulty_analytics(difficulty_map)
        assert "easy" in diff_analytics
        assert "hard" in diff_analytics
        assert diff_analytics["easy"].count == 2
        assert diff_analytics["hard"].count == 1


class TestCostTracker:
    def test_local_model_zero_cost(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="local")
        ct.record("sum", input_tokens=1000, output_tokens=500)
        report = ct.get_report()
        assert report.total_cost == 0.0

    def test_openai_gpt4o_pricing(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="openai", model_id="gpt-4o")
        ct.record("sum", input_tokens=1000, output_tokens=1000)
        report = ct.get_report()
        # $0.0025/1K input + $0.010/1K output = $0.0125 for 1K each
        assert abs(report.total_cost - 0.0125) < 1e-8

    def test_anthropic_pricing_lookup(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="anthropic", model_id="claude-sonnet-4-20250514")
        ct.record("task", input_tokens=2000, output_tokens=500)
        report = ct.get_report()
        expected = (2000 / 1000) * 0.003 + (500 / 1000) * 0.015
        assert abs(report.total_cost - expected) < 1e-8

    def test_unknown_model_uses_default(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="openai", model_id="gpt-future-model")
        ct.record("t", input_tokens=1000, output_tokens=0)
        report = ct.get_report()
        assert report.total_cost > 0  # falls back to _default pricing

    def test_custom_pricing(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(custom_input_per_1k=0.001, custom_output_per_1k=0.002)
        ct.record("t", input_tokens=1000, output_tokens=1000)
        report = ct.get_report()
        assert abs(report.total_cost - 0.003) < 1e-8

    def test_accumulate_multiple_tasks(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="openai", model_id="gpt-4o-mini")
        ct.record("sum", input_tokens=100, output_tokens=20)
        ct.record("sum", input_tokens=110, output_tokens=25)
        ct.record("mean", input_tokens=95, output_tokens=18)
        report = ct.get_report()
        assert report.per_task["sum"].request_count == 2
        assert report.per_task["mean"].request_count == 1
        assert report.total_input_tokens == 305
        assert report.total_output_tokens == 63

    def test_estimation(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="openai", model_id="gpt-4o")
        est = ct.estimate(avg_input_tokens=100, avg_output_tokens=50, n_tasks=10, datapoints_per_task=20)
        assert est["total_requests"] == 200
        assert est["estimated_input_tokens"] == 20000
        assert est["estimated_output_tokens"] == 10000
        assert est["estimated_cost_usd"] > 0
        assert "Estimated cost" in est["message"]

    def test_local_estimation_message(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="local")
        est = ct.estimate(100, 50, 10, 20)
        assert "no API cost" in est["message"].lower() or est["estimated_cost_usd"] == 0

    def test_attach_accuracy_cost_per_correct(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="openai", model_id="gpt-4o")
        ct.record("sum", input_tokens=1000, output_tokens=200)
        ct.record("sum", input_tokens=1000, output_tokens=200)
        ct.attach_accuracy({"sum": 0.5})
        report = ct.get_report()
        tc = report.per_task["sum"]
        # 2 requests at 0.5 accuracy = 1 correct; cost_per_correct = total_cost / 1
        assert tc.cost_per_correct > 0

    def test_report_to_dict(self):
        from beyondbench.utils.cost_tracker import CostTracker
        ct = CostTracker(provider="openai", model_id="gpt-4o")
        ct.record("sum", input_tokens=500, output_tokens=100)
        report = ct.get_report()
        d = report.to_dict()
        assert "totals" in d
        assert "per_task" in d
        assert "sum" in d["per_task"]
        assert "total_cost_usd" in d["totals"]

    def test_list_providers_and_models(self):
        from beyondbench.utils.cost_tracker import CostTracker
        providers = CostTracker.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        models = CostTracker.list_models("openai")
        assert "gpt-4o" in models


class TestRequestLogger:
    def test_log_creates_jsonl(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        logger = RequestLogger(output_dir=tmp_dir, enabled=True)
        logger.log("sum", "prompt", "response", input_tokens=8, output_tokens=2)
        logger.close()
        jsonl = Path(tmp_dir) / "requests.jsonl"
        assert jsonl.exists()
        assert jsonl.stat().st_size > 0

    def test_log_disabled_no_file(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        logger = RequestLogger(output_dir=tmp_dir, enabled=False)
        logger.log("sum", "p", "r")
        logger.close()
        jsonl = Path(tmp_dir) / "requests.jsonl"
        assert not jsonl.exists()

    def test_multiple_entries_multiple_lines(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        logger = RequestLogger(output_dir=tmp_dir)
        for i in range(5):
            logger.log("sum", f"prompt {i}", f"answer {i}", correct=True)
        logger.close()
        entries = RequestLogger.load(Path(tmp_dir) / "requests.jsonl")
        assert len(entries) == 5

    def test_entry_fields(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        logger = RequestLogger(output_dir=tmp_dir)
        logger.log(
            "mean",
            "What is mean of [1,2,3]?",
            "2",
            fold=1,
            input_tokens=12,
            output_tokens=1,
            latency_ms=95.0,
            correct=True,
            expected=2.0,
            parsed_answer=2.0,
            metadata={"list_size": 3},
        )
        logger.close()
        entries = RequestLogger.load(Path(tmp_dir) / "requests.jsonl")
        e = entries[0]
        assert e["task_name"] == "mean"
        assert e["fold"] == 1
        assert e["input_tokens"] == 12
        assert e["output_tokens"] == 1
        assert e["total_tokens"] == 13
        assert e["latency_ms"] == 95.0
        assert e["correct"] is True
        assert e["metadata"]["list_size"] == 3

    def test_context_manager(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        with RequestLogger(output_dir=tmp_dir) as logger:
            logger.log("t", "p", "r")
        # File should be closed and flushed
        entries = RequestLogger.load(Path(tmp_dir) / "requests.jsonl")
        assert len(entries) == 1

    def test_custom_filename(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        logger = RequestLogger(output_dir=tmp_dir, filename="mylog.jsonl")
        logger.log("t", "p", "r")
        logger.close()
        assert (Path(tmp_dir) / "mylog.jsonl").exists()

    def test_replay_prompts(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        with RequestLogger(output_dir=tmp_dir) as logger:
            logger.log("t", "prompt A", "ans A")
            logger.log("t", "prompt B", "ans B")
        prompts = RequestLogger.replay_prompts(Path(tmp_dir) / "requests.jsonl")
        assert prompts == ["prompt A", "prompt B"]

    def test_replay_responses(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        with RequestLogger(output_dir=tmp_dir) as logger:
            logger.log("t", "p", "resp X")
            logger.log("t", "p", "resp Y")
        responses = RequestLogger.replay_responses(Path(tmp_dir) / "requests.jsonl")
        assert responses == ["resp X", "resp Y"]

    def test_summarize(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        with RequestLogger(output_dir=tmp_dir) as logger:
            logger.log("sum", "p1", "r1", input_tokens=10, output_tokens=2, latency_ms=100.0, correct=True)
            logger.log("mean", "p2", "r2", input_tokens=12, output_tokens=3, latency_ms=80.0, correct=False)
        s = RequestLogger.summarize(Path(tmp_dir) / "requests.jsonl")
        assert s["total_requests"] == 2
        assert "sum" in s["tasks"]
        assert "mean" in s["tasks"]
        assert s["avg_input_tokens"] == 11.0
        assert s["correct_rate"] == 0.5

    def test_entry_count(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        logger = RequestLogger(output_dir=tmp_dir)
        logger.log("t", "p", "r")
        logger.log("t", "p", "r")
        assert logger.entry_count == 2
        logger.close()

    def test_path_property(self, tmp_dir):
        from beyondbench.utils.request_logger import RequestLogger
        logger = RequestLogger(output_dir=tmp_dir)
        assert logger.path is not None
        assert str(logger.path).endswith("requests.jsonl")
        logger.close()


# ===========================================================================
# Phase 8 — Result Visualization & Reporting
# ===========================================================================

class TestReportGenerator:
    def test_html_report_generated(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        path = gen.save_html_report(simple_results)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 5000

    def test_html_contains_plotly(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        html = gen.generate_html_report(simple_results)
        assert "plotly" in html.lower()

    def test_html_contains_model_id(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        html = gen.generate_html_report(simple_results)
        assert "test-model" in html

    def test_html_contains_task_names(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        html = gen.generate_html_report(simple_results)
        assert "sudoku_solving" in html
        assert "fibonacci_sequence" in html

    def test_html_token_analytics_section(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        html = gen.generate_html_report(simple_results)
        assert "Token Analysis" in html

    def test_html_suite_breakdown(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        html = gen.generate_html_report(simple_results)
        assert "Suite Breakdown" in html

    def test_markdown_generated(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        path = gen.save_markdown_report(simple_results)
        assert Path(path).exists()
        with open(path) as f:
            md = f.read()
        assert "|" in md  # has tables
        assert "test-model" in md
        assert "##" in md  # has headers

    def test_markdown_has_suite_breakdown(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        md = gen.generate_markdown_report(simple_results)
        assert "Suite Breakdown" in md
        assert "easy" in md
        assert "hard" in md

    def test_markdown_has_per_task_table(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        md = gen.generate_markdown_report(simple_results)
        assert "Per-Task Results" in md
        assert "| sum |" in md or "sum" in md
        assert "sorting" in md

    def test_markdown_token_analytics(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        md = gen.generate_markdown_report(simple_results)
        assert "Token Analysis" in md
        assert "p95" in md

    def test_latex_generated(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        tex = gen.generate_latex_report(simple_results)
        assert r"\begin{table}" in tex
        assert r"\toprule" in tex
        assert "sudoku_solving" in tex

    def test_save_latex(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        path = gen.save_latex_report(simple_results)
        assert Path(path).exists()
        assert path.endswith(".tex")

    def test_save_all_html_markdown(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_dir)
        paths = gen.save_all(simple_results, formats=["html", "markdown"])
        assert "html" in paths and paths["html"] is not None
        assert "markdown" in paths and paths["markdown"] is not None

    def test_comparison_in_markdown(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        other = dict(simple_results)
        other["model_id"] = "other-model"
        gen = ReportGenerator(output_dir=tmp_dir)
        md = gen.generate_markdown_report(simple_results, compare_results=[other])
        assert "Model Comparison" in md

    def test_pdf_skipped_gracefully_without_weasyprint(self, simple_results, tmp_dir):
        from beyondbench.utils.report_generator import ReportGenerator
        with patch.dict("sys.modules", {"weasyprint": None}):
            gen = ReportGenerator(output_dir=tmp_dir)
            path = gen.save_pdf_report(simple_results)
            assert path is None  # gracefully skipped

    def test_backward_compat_generate_final_report(self):
        from beyondbench.utils.report_generator import generate_final_report
        results = {
            "task_results": {
                "sum": {"accuracy": 0.9, "avg_tokens": 40},
                "mean": {"accuracy": 0.7, "avg_tokens": 60},
            }
        }
        report = generate_final_report(results)
        assert "summary" in report
        assert "efficiency_ranking" in report
        assert report["summary"]["total_tasks"] == 2


class TestVisualizer:
    """Tests for the Visualizer chart generation."""

    @pytest.fixture
    def viz(self):
        from beyondbench.utils.visualizer import Visualizer
        return Visualizer()

    @pytest.fixture
    def multi_results(self):
        return {
            "ModelA": {
                "task_results": {
                    "sum": {"accuracy": 0.9},
                    "sorting": {"accuracy": 0.85},
                    "sudoku_solving": {"accuracy": 0.2},
                }
            },
            "ModelB": {
                "task_results": {
                    "sum": {"accuracy": 0.75},
                    "sorting": {"accuracy": 0.70},
                    "sudoku_solving": {"accuracy": 0.1},
                }
            },
        }

    def test_accuracy_by_task_returns_figure(self, viz, simple_results):
        fig = viz.plot_accuracy_by_task(simple_results)
        assert fig is not None

    def test_accuracy_by_suite_returns_figure(self, viz, simple_results):
        fig = viz.plot_accuracy_by_suite(simple_results)
        assert fig is not None

    def test_accuracy_by_difficulty_returns_figure_or_none(self, viz, simple_results):
        results_list = [simple_results, simple_results]
        fig = viz.plot_accuracy_by_difficulty(results_list, list_sizes=[8, 16])
        # may be None if numpy not available, else a figure
        assert fig is None or hasattr(fig, "to_html") or hasattr(fig, "savefig")

    def test_token_distribution_from_list(self, viz):
        token_counts = [10, 20, 30, 40, 50, 60, 70, 80]
        fig = viz.plot_token_distribution(token_counts)
        assert fig is not None

    def test_latency_distribution_flat(self, viz):
        latencies = [100.0, 150.0, 200.0, 120.0, 180.0]
        fig = viz.plot_latency_distribution(latencies)
        assert fig is not None

    def test_latency_distribution_dict(self, viz):
        latencies = {"sum": [100.0, 110.0], "mean": [200.0, 210.0]}
        fig = viz.plot_latency_distribution(latencies)
        assert fig is not None

    def test_model_comparison_returns_figure(self, viz, multi_results):
        fig = viz.plot_model_comparison(multi_results)
        assert fig is not None

    def test_radar_chart_returns_figure(self, viz, multi_results):
        fig = viz.plot_radar_chart(multi_results)
        assert fig is not None

    def test_scaling_curve_returns_figure(self, viz, simple_results):
        results_by_size = {8: simple_results, 16: simple_results, 32: simple_results}
        fig = viz.plot_scaling_curve(results_by_size)
        assert fig is not None

    def test_empty_results_returns_none(self, viz):
        fig = viz.plot_accuracy_by_task({})
        assert fig is None

    def test_save_html(self, viz, simple_results, tmp_dir):
        fig = viz.plot_accuracy_by_task(simple_results)
        out = os.path.join(tmp_dir, "chart.html")
        path = viz.save(fig, out, fmt="html")
        assert path is not None
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100

    def test_save_none_returns_none(self, viz, tmp_dir):
        path = viz.save(None, os.path.join(tmp_dir, "test.html"))
        assert path is None

    def test_save_all(self, viz, simple_results, tmp_dir):
        figs = {
            "task": viz.plot_accuracy_by_task(simple_results),
            "suite": viz.plot_accuracy_by_suite(simple_results),
        }
        saved = viz.save_all(figs, tmp_dir, fmt="html")
        assert len(saved) == 2
        for name, path in saved.items():
            assert path is not None
            assert os.path.exists(path)


# ===========================================================================
# CLI integration tests
# ===========================================================================

class TestReportCLI:
    """Test `beyondbench report` CLI command."""

    @pytest.fixture
    def results_file(self, simple_results, tmp_dir):
        path = Path(tmp_dir) / "final_results.json"
        with open(path, "w") as f:
            json.dump(simple_results, f)
        return str(path)

    def test_report_html(self, results_file, tmp_dir):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        out_html = str(Path(tmp_dir) / "report.html")
        result = runner.invoke(main, [
            "report",
            "--results-file", results_file,
            "--format", "html",
            "--output", out_html,
        ])
        assert result.exit_code == 0, result.output
        assert Path(out_html).exists()
        assert "Report generation complete" in result.output

    def test_report_markdown(self, results_file, tmp_dir):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        out_md = str(Path(tmp_dir) / "report.md")
        result = runner.invoke(main, [
            "report",
            "--results-file", results_file,
            "--format", "markdown",
            "--output", out_md,
        ])
        assert result.exit_code == 0, result.output
        assert Path(out_md).exists()

    def test_report_all_formats(self, results_file, tmp_dir):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, [
            "report",
            "--results-file", results_file,
            "--format", "all",
        ])
        assert result.exit_code == 0, result.output
        # HTML and Markdown must be generated; PDF optional (weasyprint)
        assert "HTML" in result.output
        assert "MARKDOWN" in result.output

    def test_report_missing_file_exits_nonzero(self, tmp_dir):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, [
            "report",
            "--results-file", "/nonexistent/path/final_results.json",
        ])
        assert result.exit_code != 0

    def test_report_with_charts_dir(self, results_file, tmp_dir):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        charts_dir = str(Path(tmp_dir) / "charts")
        result = runner.invoke(main, [
            "report",
            "--results-file", results_file,
            "--format", "html",
            "--charts-dir", charts_dir,
        ])
        assert result.exit_code == 0, result.output
        assert Path(charts_dir).exists()
        chart_files = list(Path(charts_dir).glob("*.html"))
        assert len(chart_files) >= 1

    def test_report_auto_discovers_results(self, simple_results, tmp_dir):
        """report --results-dir should find the most recent final_results.json."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        path = Path(tmp_dir) / "run1" / "final_results.json"
        path.parent.mkdir()
        with open(path, "w") as f:
            json.dump(simple_results, f)

        runner = CliRunner()
        result = runner.invoke(main, [
            "report",
            "--results-dir", tmp_dir,
            "--format", "markdown",
        ])
        assert result.exit_code == 0, result.output

    def test_report_compare_flag(self, simple_results, tmp_dir):
        """--compare should include comparison in the report."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        file_a = Path(tmp_dir) / "a.json"
        file_b = Path(tmp_dir) / "b.json"
        other = dict(simple_results)
        other["model_id"] = "other-model"
        with open(file_a, "w") as f:
            json.dump(simple_results, f)
        with open(file_b, "w") as f:
            json.dump(other, f)

        runner = CliRunner()
        result = runner.invoke(main, [
            "report",
            "--results-file", str(file_a),
            "--compare", str(file_b),
            "--format", "markdown",
        ])
        assert result.exit_code == 0, result.output

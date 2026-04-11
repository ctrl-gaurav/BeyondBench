"""
Comprehensive reporting utilities for BeyondBench evaluations.

Generates HTML (with interactive Plotly charts), Markdown (with tables),
PDF (via WeasyPrint, optional), and LaTeX tables.

Report sections
---------------
1. Executive summary  — overall accuracy, top/bottom tasks, model info
2. Per-suite breakdown — per-suite accuracy table
3. Per-task details   — accuracy, parse success, token usage
4. Token analysis     — input/output distribution, efficiency metrics
5. Failure analysis   — most common parsing failures
6. Model comparison   — accuracy delta (when multiple results provided)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from importlib.metadata import version as _pkg_version, PackageNotFoundError

import numpy as np

try:
    _VERSION = _pkg_version("beyondbench")
except PackageNotFoundError:
    try:
        from beyondbench import __version__ as _VERSION
    except ImportError:
        _VERSION = "unknown"

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def calculate_efficiency_score(accuracy, tokens, all_tokens):
    """Harmonic mean of accuracy and token efficiency."""
    if not all_tokens or len(all_tokens) < 2:
        return accuracy
    min_tokens = min(all_tokens)
    max_tokens = max(all_tokens)
    if max_tokens == min_tokens:
        return accuracy
    normalized_tokens = (tokens - min_tokens) / (max_tokens - min_tokens)
    token_efficiency = 1 - normalized_tokens
    if accuracy + token_efficiency == 0:
        return 0
    return 2 * (accuracy * token_efficiency) / (accuracy + token_efficiency)


def calculate_overthinking_ratio(accuracy, tokens, baseline_tokens=50):
    """Ratio of tokens used vs baseline, adjusted for accuracy."""
    if baseline_tokens <= 0:
        return 1.0
    raw_ratio = tokens / baseline_tokens
    accuracy_weight = max(0.1, accuracy)
    return raw_ratio / accuracy_weight


def format_report_table(report_data: Dict[str, Any]) -> str:
    """Format task-level report data into an ASCII table."""
    if not TABULATE_AVAILABLE:
        return str(report_data)
    if not report_data:
        return "No data to display"
    headers = ["Task", "Accuracy", "Avg Tokens", "Efficiency", "Overthinking"]
    table_data = []
    all_tokens = [data.get("avg_tokens", 0) for data in report_data.values()]
    for task, data in report_data.items():
        acc = data.get("accuracy", 0)
        tokens = data.get("avg_tokens", 0)
        eff = calculate_efficiency_score(acc, tokens, all_tokens)
        over = calculate_overthinking_ratio(acc, tokens)
        table_data.append([task, f"{acc:.2%}", f"{tokens:.1f}", f"{eff:.3f}", f"{over:.2f}"])
    return tabulate(table_data, headers=headers, tablefmt="grid")


def generate_efficiency_ranking(report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate efficiency ranking from report data."""
    all_tokens = [data.get("avg_tokens", 0) for data in report_data.values()]
    rankings = []
    for task, data in report_data.items():
        acc = data.get("accuracy", 0)
        tokens = data.get("avg_tokens", 0)
        rankings.append({
            "task": task,
            "accuracy": acc,
            "tokens": tokens,
            "efficiency_score": calculate_efficiency_score(acc, tokens, all_tokens),
            "overthinking_ratio": calculate_overthinking_ratio(acc, tokens),
        })
    rankings.sort(key=lambda x: x["efficiency_score"], reverse=True)
    return rankings


def generate_final_report(results: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate legacy-format final report (kept for backward compatibility)."""
    task_results = results.get("task_results", {})
    total_accuracy = np.mean([t.get("accuracy", 0) for t in task_results.values()]) if task_results else 0
    total_success_rate = np.mean([t.get("success_rate", 0) for t in task_results.values()]) if task_results else 0
    efficiency_ranking = generate_efficiency_ranking(task_results)
    report = {
        "summary": {
            "total_tasks": len(task_results),
            "overall_accuracy": float(total_accuracy),
            "total_evaluations": results.get("total_evaluations", 0),
            "success_rate": float(total_success_rate),
            "avg_efficiency_score": float(np.mean([r["efficiency_score"] for r in efficiency_ranking])) if efficiency_ranking else 0,
            "avg_overthinking_ratio": float(np.mean([r["overthinking_ratio"] for r in efficiency_ranking])) if efficiency_ranking else 1,
        },
        "task_details": task_results,
        "efficiency_ranking": efficiency_ranking,
        "formatted_table": format_report_table(task_results),
        "metadata": {
            "model_id": results.get("model_id", "unknown"),
            "timestamp": results.get("timestamp", "unknown"),
            "evaluation_engine": "beyondbench",
            "version": _VERSION,
        },
    }
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------
# Internal data-extraction helpers
# ---------------------------------------------------------------------------

def _extract_accuracy(task_result: Dict[str, Any]) -> float:
    for key in ("accuracy", "avg_accuracy"):
        val = task_result.get(key)
        if val is not None:
            return float(val)
    summary = task_result.get("summary", {})
    for key in ("avg_accuracy", "accuracy"):
        val = summary.get(key)
        if val is not None:
            return float(val)
    return 0.0


def _extract_task_data(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {task_name: {accuracy, avg_tokens, suite}} from a results dict."""
    task_results = results.get("task_results", {})
    out: Dict[str, Dict[str, Any]] = {}

    EASY = {
        "sorting", "comparison", "sum", "multiplication", "odd_count",
        "even_count", "absolute_difference", "division", "find_maximum",
        "find_minimum", "mean", "median", "mode", "subtraction",
        "second_maximum", "range", "index_of_maximum", "count_negative",
        "count_unique", "count_multiples", "count_palindromic",
        "count_perfect_squares", "count_greater_than_previous",
        "alternating_sum", "sum_of_digits", "sum_of_max_indices",
        "local_maxima_count", "max_adjacent_difference",
        "longest_increasing_subsequence",
    }
    MEDIUM = {
        "fibonacci_sequence", "algebraic_sequence", "geometric_sequence",
        "prime_sequence", "complex_pattern",
    }

    for task, data in task_results.items():
        suite = "easy" if task in EASY else "medium" if task in MEDIUM else "hard"
        acc = _extract_accuracy(data)
        avg_tokens = data.get("avg_tokens", 0)
        out[task] = {"accuracy": acc, "avg_tokens": avg_tokens, "suite": suite}
    return out


def _suite_summary(task_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-suite mean accuracy."""
    from collections import defaultdict
    suites: Dict[str, List[float]] = defaultdict(list)
    for td in task_data.values():
        suites[td["suite"]].append(td["accuracy"])
    return {s: {"mean_accuracy": float(np.mean(v)), "task_count": len(v)} for s, v in suites.items()}


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _build_plotly_html(results: Dict[str, Any]) -> str:
    """Build embedded Plotly chart divs for the HTML report."""
    try:
        from .visualizer import Visualizer
        import plotly.graph_objects as go
        viz = Visualizer(theme="plotly_white")
        charts_html = ""

        # Chart 1: accuracy by task
        fig1 = viz.plot_accuracy_by_task(results)
        if fig1 is not None:
            charts_html += fig1.to_html(full_html=False, include_plotlyjs=False, div_id="chart_task")

        # Chart 2: accuracy by suite
        fig2 = viz.plot_accuracy_by_suite(results)
        if fig2 is not None:
            charts_html += fig2.to_html(full_html=False, include_plotlyjs=False, div_id="chart_suite")

        return charts_html
    except Exception as e:
        logger.debug(f"Plotly chart generation failed: {e}")
        return "<p><em>Charts unavailable (install plotly: pip install plotly)</em></p>"


def _plotly_cdn_script() -> str:
    return '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'


# ---------------------------------------------------------------------------
# ReportGenerator class
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generate evaluation reports in multiple formats.

    Supported formats: html, markdown, pdf (optional), latex.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        results: Dict[str, Any],
        compare_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a full HTML report with embedded Plotly charts."""
        task_data = _extract_task_data(results)
        summary = results.get("summary", {})
        model_id = results.get("model_id", summary.get("model_id", "unknown"))
        timestamp = results.get("timestamp", summary.get("timestamp", ""))

        avg_accuracy = summary.get("avg_accuracy") or (
            float(np.mean([td["accuracy"] for td in task_data.values()])) if task_data else 0.0
        )

        suite_summary = _suite_summary(task_data)
        top_tasks = sorted(task_data.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:5]
        bottom_tasks = sorted(task_data.items(), key=lambda x: x[1]["accuracy"])[:5]

        # Token analytics section
        token_analytics = (results.get("summary") or {}).get("token_analytics") or results.get("token_analytics") or {}

        # Charts
        charts_html = _build_plotly_html(results)
        plotly_script = _plotly_cdn_script()

        # Per-task table rows
        task_rows = ""
        for task_name, td in sorted(task_data.items()):
            acc = td["accuracy"]
            suite = td["suite"]
            tokens = td.get("avg_tokens", 0)
            color_class = "good" if acc >= 0.7 else "medium" if acc >= 0.4 else "poor"
            task_rows += f"""
            <tr>
                <td>{task_name}</td>
                <td class="badge badge-{suite}">{suite}</td>
                <td class="acc {color_class}">{acc:.1%}</td>
                <td>{tokens:.0f}</td>
            </tr>"""

        # Suite summary rows
        suite_rows = ""
        for suite_name, ss in sorted(suite_summary.items()):
            suite_rows += f"""
            <tr>
                <td class="badge badge-{suite_name}">{suite_name}</td>
                <td>{ss['task_count']}</td>
                <td>{ss['mean_accuracy']:.1%}</td>
            </tr>"""

        # Token analytics section HTML
        token_section = ""
        if token_analytics:
            inp = token_analytics.get("input", {})
            out = token_analytics.get("output", {})
            token_section = f"""
            <section>
              <h2>Token Analysis</h2>
              <table>
                <thead><tr><th>Metric</th><th>Input</th><th>Output</th></tr></thead>
                <tbody>
                  <tr><td>Average</td><td>{inp.get('avg', 0):.1f}</td><td>{out.get('avg', 0):.1f}</td></tr>
                  <tr><td>Min</td><td>{inp.get('min', 0)}</td><td>{out.get('min', 0)}</td></tr>
                  <tr><td>Max</td><td>{inp.get('max', 0)}</td><td>{out.get('max', 0)}</td></tr>
                  <tr><td>p50</td><td>{inp.get('p50', 0):.1f}</td><td>{out.get('p50', 0):.1f}</td></tr>
                  <tr><td>p95</td><td>{inp.get('p95', 0):.1f}</td><td>{out.get('p95', 0):.1f}</td></tr>
                  <tr><td>Total</td><td>{inp.get('total', 0):,}</td><td>{out.get('total', 0):,}</td></tr>
                </tbody>
              </table>
            </section>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BeyondBench Report — {model_id}</title>
  {plotly_script}
  <style>
    :root {{
      --green: #2ecc71; --yellow: #f39c12; --red: #e74c3c;
      --blue: #3498db; --dark: #1a1a2e; --card: #16213e;
      --text: #eaeaea; --muted: #888;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--dark); color: var(--text); padding: 24px; }}
    h1 {{ color: var(--blue); font-size: 2rem; margin-bottom: 4px; }}
    h2 {{ color: var(--blue); font-size: 1.25rem; margin: 24px 0 12px; }}
    .meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
    .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
    .card {{ background: var(--card); border-radius: 10px; padding: 20px; min-width: 160px; flex: 1; }}
    .card-value {{ font-size: 2rem; font-weight: bold; color: var(--green); }}
    .card-label {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; }}
    section {{ background: var(--card); border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th {{ background: #0f3460; color: var(--text); padding: 10px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #1e1e3a; }}
    tr:hover td {{ background: #1e1e3a; }}
    .badge {{ display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-weight: bold; }}
    .badge-easy {{ background: #1a5c38; color: var(--green); }}
    .badge-medium {{ background: #5c3d0a; color: var(--yellow); }}
    .badge-hard {{ background: #5c1a1a; color: var(--red); }}
    .acc.good {{ color: var(--green); font-weight: bold; }}
    .acc.medium {{ color: var(--yellow); }}
    .acc.poor {{ color: var(--red); }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>BeyondBench Evaluation Report</h1>
  <div class="meta">
    Model: <strong>{model_id}</strong> &nbsp;|&nbsp;
    Version: {_VERSION} &nbsp;|&nbsp;
    {timestamp}
  </div>

  <!-- Executive Summary Cards -->
  <div class="cards">
    <div class="card">
      <div class="card-value">{avg_accuracy:.1%}</div>
      <div class="card-label">Overall Accuracy</div>
    </div>
    <div class="card">
      <div class="card-value">{len(task_data)}</div>
      <div class="card-label">Tasks Evaluated</div>
    </div>
    <div class="card">
      <div class="card-value">{summary.get('total_evaluations', '—')}</div>
      <div class="card-label">Total Evaluations</div>
    </div>
    <div class="card">
      <div class="card-value">{summary.get('total_duration', 0):.0f}s</div>
      <div class="card-label">Duration</div>
    </div>
  </div>

  <!-- Charts -->
  <section>
    <h2>Charts</h2>
    <div class="chart-grid">
      {charts_html}
    </div>
  </section>

  <!-- Suite Breakdown -->
  <section>
    <h2>Suite Breakdown</h2>
    <table>
      <thead><tr><th>Suite</th><th>Tasks</th><th>Avg Accuracy</th></tr></thead>
      <tbody>{suite_rows}</tbody>
    </table>
  </section>

  <!-- Top & Bottom Tasks -->
  <section>
    <h2>Top 5 Tasks</h2>
    <table>
      <thead><tr><th>Task</th><th>Accuracy</th></tr></thead>
      <tbody>
        {"".join(f"<tr><td>{t}</td><td class='acc good'>{d['accuracy']:.1%}</td></tr>" for t, d in top_tasks)}
      </tbody>
    </table>
  </section>
  <section>
    <h2>Bottom 5 Tasks</h2>
    <table>
      <thead><tr><th>Task</th><th>Accuracy</th></tr></thead>
      <tbody>
        {"".join(f"<tr><td>{t}</td><td class='acc poor'>{d['accuracy']:.1%}</td></tr>" for t, d in bottom_tasks)}
      </tbody>
    </table>
  </section>

  <!-- Per-Task Results -->
  <section>
    <h2>Per-Task Results</h2>
    <table>
      <thead><tr><th>Task</th><th>Suite</th><th>Accuracy</th><th>Avg Tokens</th></tr></thead>
      <tbody>{task_rows}</tbody>
    </table>
  </section>

  {token_section}

  <div class="meta" style="margin-top:32px;">
    Generated by BeyondBench v{_VERSION}
  </div>
</body>
</html>"""
        return html

    def save_html_report(
        self,
        results: Dict[str, Any],
        filename: str = "report.html",
        compare_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        html = self.generate_html_report(results, compare_results=compare_results)
        report_file = self.output_dir / filename
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML report saved to {report_file}")
        return str(report_file)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        compare_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a GitHub-flavoured Markdown report."""
        task_data = _extract_task_data(results)
        summary = results.get("summary", {})
        model_id = results.get("model_id", summary.get("model_id", "unknown"))
        timestamp = results.get("timestamp", summary.get("timestamp", ""))
        avg_accuracy = summary.get("avg_accuracy") or (
            float(np.mean([td["accuracy"] for td in task_data.values()])) if task_data else 0.0
        )
        suite_summary = _suite_summary(task_data)
        token_analytics = (results.get("summary") or {}).get("token_analytics") or results.get("token_analytics") or {}

        lines = [
            f"# BeyondBench Evaluation Report",
            f"",
            f"**Model:** `{model_id}`  ",
            f"**Version:** {_VERSION}  ",
            f"**Timestamp:** {timestamp}",
            f"",
            f"## Executive Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Accuracy | **{avg_accuracy:.1%}** |",
            f"| Tasks Evaluated | {len(task_data)} |",
            f"| Total Evaluations | {summary.get('total_evaluations', '—')} |",
            f"| Duration | {summary.get('total_duration', 0):.0f}s |",
            f"",
            f"## Suite Breakdown",
            f"",
            f"| Suite | Tasks | Avg Accuracy |",
            f"|-------|-------|--------------|",
        ]
        for suite_name, ss in sorted(suite_summary.items()):
            lines.append(f"| {suite_name} | {ss['task_count']} | {ss['mean_accuracy']:.1%} |")

        lines += [
            f"",
            f"## Per-Task Results",
            f"",
            f"| Task | Suite | Accuracy | Avg Tokens |",
            f"|------|-------|----------|------------|",
        ]
        for task_name, td in sorted(task_data.items(), key=lambda x: x[1]["accuracy"], reverse=True):
            lines.append(
                f"| {task_name} | {td['suite']} | {td['accuracy']:.1%} | {td.get('avg_tokens', 0):.0f} |"
            )

        if token_analytics:
            inp = token_analytics.get("input", {})
            out = token_analytics.get("output", {})
            lines += [
                f"",
                f"## Token Analysis",
                f"",
                f"| Metric | Input | Output |",
                f"|--------|-------|--------|",
                f"| Average | {inp.get('avg', 0):.1f} | {out.get('avg', 0):.1f} |",
                f"| Min | {inp.get('min', 0)} | {out.get('min', 0)} |",
                f"| Max | {inp.get('max', 0)} | {out.get('max', 0)} |",
                f"| p50 | {inp.get('p50', 0):.1f} | {out.get('p50', 0):.1f} |",
                f"| p95 | {inp.get('p95', 0):.1f} | {out.get('p95', 0):.1f} |",
                f"| Total | {inp.get('total', 0):,} | {out.get('total', 0):,} |",
            ]

        # Model comparison section
        if compare_results:
            lines += ["", "## Model Comparison", ""]
            all_tasks = sorted(task_data.keys())
            header = "| Task |" + "".join(
                f" {r.get('model_id', r.get('summary', {}).get('model_id', f'Model{i}'))} |"
                for i, r in enumerate(compare_results)
            )
            sep = "|------|" + "------|" * len(compare_results)
            lines += [header, sep]
            for task in all_tasks:
                row = f"| {task} |"
                for r in compare_results:
                    td2 = _extract_task_data(r)
                    acc = td2.get(task, {}).get("accuracy", 0.0)
                    row += f" {acc:.1%} |"
                lines.append(row)

        lines += ["", f"---", f"*Generated by BeyondBench v{_VERSION}*"]
        return "\n".join(lines)

    def save_markdown_report(
        self,
        results: Dict[str, Any],
        filename: str = "report.md",
        compare_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        md = self.generate_markdown_report(results, compare_results=compare_results)
        report_file = self.output_dir / filename
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"Markdown report saved to {report_file}")
        return str(report_file)

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    def generate_latex_report(self, results: Dict[str, Any]) -> str:
        """Generate LaTeX tables for academic papers."""
        task_data = _extract_task_data(results)
        model_id = results.get("model_id", "unknown")
        avg_accuracy = float(np.mean([td["accuracy"] for td in task_data.values()])) if task_data else 0.0

        rows = ""
        for task_name, td in sorted(task_data.items(), key=lambda x: x[1]["accuracy"], reverse=True):
            rows += f"  {task_name} & {td['suite']} & {td['accuracy']:.1%} \\\\\n"

        return rf"""\begin{{table}}[ht]
  \centering
  \caption{{BeyondBench Results for {model_id}}}
  \label{{tab:beyondbench}}
  \begin{{tabular}}{{lll}}
    \toprule
    Task & Suite & Accuracy \\
    \midrule
{rows}    \midrule
    \textbf{{Overall}} & & \textbf{{{avg_accuracy:.1%}}} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""

    def save_latex_report(self, results: Dict[str, Any], filename: str = "report.tex") -> str:
        latex = self.generate_latex_report(results)
        report_file = self.output_dir / filename
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(latex)
        logger.info(f"LaTeX report saved to {report_file}")
        return str(report_file)

    # ------------------------------------------------------------------
    # PDF (optional WeasyPrint)
    # ------------------------------------------------------------------

    def save_pdf_report(
        self,
        results: Dict[str, Any],
        filename: str = "report.pdf",
    ) -> Optional[str]:
        """Generate PDF via WeasyPrint (optional dependency)."""
        try:
            from weasyprint import HTML  # type: ignore
        except ImportError:
            logger.error(
                "PDF generation requires WeasyPrint: pip install weasyprint. "
                "Skipping PDF export."
            )
            return None

        html_content = self.generate_html_report(results)
        report_file = self.output_dir / filename
        try:
            HTML(string=html_content).write_pdf(str(report_file))
            logger.info(f"PDF report saved to {report_file}")
            return str(report_file)
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Convenience: save all
    # ------------------------------------------------------------------

    def save_all(
        self,
        results: Dict[str, Any],
        formats: Optional[List[str]] = None,
        compare_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Save reports in multiple formats.

        Args:
            results: Evaluation results dict.
            formats: List of formats to generate. Default: ["html", "markdown"].
            compare_results: Optional list of additional results dicts for comparison.

        Returns:
            {format: output_path_or_None}
        """
        if formats is None:
            formats = ["html", "markdown"]

        paths: Dict[str, Optional[str]] = {}
        for fmt in formats:
            if fmt in ("html", "htm"):
                paths["html"] = self.save_html_report(results, compare_results=compare_results)
            elif fmt in ("markdown", "md"):
                paths["markdown"] = self.save_markdown_report(results, compare_results=compare_results)
            elif fmt == "latex":
                paths["latex"] = self.save_latex_report(results)
            elif fmt == "pdf":
                paths["pdf"] = self.save_pdf_report(results)
            else:
                logger.warning(f"Unknown report format: {fmt}")
        return paths

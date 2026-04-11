"""
Visualization library for BeyondBench evaluation results.

Produces 8 chart types using Plotly (interactive HTML) with Matplotlib fallback.
All charts can be saved as PNG, SVG, or interactive HTML.

Chart types
-----------
1. accuracy_by_task       — horizontal bar chart, one bar per task
2. accuracy_by_suite      — grouped bar or radar chart per suite
3. accuracy_by_difficulty — heatmap: list_size vs task accuracy
4. token_distribution     — histogram of output token counts
5. latency_distribution   — box plot of per-request latency
6. model_comparison       — grouped bar chart across multiple models
7. scaling_curve          — line chart: accuracy vs list_size
8. radar_chart            — radar / spider chart across task categories

Usage::

    from beyondbench.utils.visualizer import Visualizer
    viz = Visualizer()
    fig = viz.plot_accuracy_by_task(results)
    viz.save(fig, "accuracy.html", fmt="html")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_task_accuracy(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract {task_name: accuracy} from a results dict."""
    task_results = results.get("task_results", {})
    out: Dict[str, float] = {}
    for task, data in task_results.items():
        # Support various result shapes
        if isinstance(data, dict):
            acc = data.get("accuracy") or data.get("avg_accuracy")
            if acc is None:
                summary = data.get("summary", {})
                acc = summary.get("avg_accuracy") or summary.get("accuracy", 0.0)
            out[task] = float(acc or 0.0)
    return out


def _get_suite_for_task(task_name: str) -> str:
    EASY_TASKS = {
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
    MEDIUM_TASKS = {
        "fibonacci_sequence", "algebraic_sequence", "geometric_sequence",
        "prime_sequence", "complex_pattern",
    }
    if task_name in EASY_TASKS:
        return "easy"
    if task_name in MEDIUM_TASKS:
        return "medium"
    return "hard"


SUITE_COLORS = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c", "unknown": "#95a5a6"}


# ---------------------------------------------------------------------------
# Visualizer class
# ---------------------------------------------------------------------------

class Visualizer:
    """
    Generates evaluation charts.

    Prefer Plotly when available; falls back to Matplotlib.
    """

    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme
        if not PLOTLY_AVAILABLE:
            logger.warning(
                "Plotly not installed — charts will use Matplotlib. "
                "For interactive HTML: pip install plotly"
            )

    # ------------------------------------------------------------------
    # 1. Accuracy by task
    # ------------------------------------------------------------------

    def plot_accuracy_by_task(self, results: Dict[str, Any]) -> Any:
        """Horizontal bar chart of per-task accuracy."""
        acc_map = _extract_task_accuracy(results)
        if not acc_map:
            logger.warning("No task accuracy data found.")
            return None

        tasks = sorted(acc_map.keys(), key=lambda t: acc_map[t])
        accs = [acc_map[t] for t in tasks]
        colors = [SUITE_COLORS[_get_suite_for_task(t)] for t in tasks]

        if PLOTLY_AVAILABLE:
            fig = go.Figure(go.Bar(
                y=tasks,
                x=accs,
                orientation="h",
                marker_color=colors,
                text=[f"{a:.1%}" for a in accs],
                textposition="outside",
            ))
            fig.update_layout(
                title="Accuracy by Task",
                xaxis_title="Accuracy",
                yaxis_title="Task",
                template=self.theme,
                height=max(400, len(tasks) * 22),
                xaxis=dict(range=[0, 1.05]),
            )
            return fig

        # Matplotlib fallback
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, max(4, len(tasks) * 0.35)))
            y_pos = range(len(tasks))
            ax.barh(y_pos, accs, color=colors)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(tasks, fontsize=8)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Accuracy")
            ax.set_title("Accuracy by Task")
            plt.tight_layout()
            return fig
        return None

    # ------------------------------------------------------------------
    # 2. Accuracy by suite
    # ------------------------------------------------------------------

    def plot_accuracy_by_suite(self, results: Dict[str, Any]) -> Any:
        """Grouped bar chart showing average accuracy per suite."""
        acc_map = _extract_task_accuracy(results)
        if not acc_map:
            return None

        from collections import defaultdict
        suite_accs: Dict[str, List[float]] = defaultdict(list)
        for task, acc in acc_map.items():
            suite_accs[_get_suite_for_task(task)].append(acc)

        suites = list(suite_accs.keys())
        means = [sum(v) / len(v) for v in suite_accs.values()]
        colors = [SUITE_COLORS.get(s, "#95a5a6") for s in suites]

        if PLOTLY_AVAILABLE:
            fig = go.Figure(go.Bar(
                x=suites,
                y=means,
                marker_color=colors,
                text=[f"{m:.1%}" for m in means],
                textposition="outside",
            ))
            fig.update_layout(
                title="Average Accuracy by Suite",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1.1]),
                template=self.theme,
            )
            return fig
        return None

    # ------------------------------------------------------------------
    # 3. Accuracy by difficulty (heatmap)
    # ------------------------------------------------------------------

    def plot_accuracy_by_difficulty(
        self,
        results_list: List[Dict[str, Any]],
        list_sizes: Optional[List[int]] = None,
    ) -> Any:
        """
        Heatmap: rows = tasks, columns = list_sizes / difficulty levels.

        ``results_list`` should be a list of results dicts, one per list_size.
        ``list_sizes`` must match len(results_list).
        """
        if not results_list:
            return None

        sizes = list_sizes or list(range(len(results_list)))
        all_tasks = sorted({
            t for r in results_list for t in _extract_task_accuracy(r).keys()
        })

        z = []
        for task in all_tasks:
            row = [_extract_task_accuracy(r).get(task, 0.0) for r in results_list]
            z.append(row)

        if PLOTLY_AVAILABLE and NUMPY_AVAILABLE:
            fig = go.Figure(go.Heatmap(
                z=z,
                x=[str(s) for s in sizes],
                y=all_tasks,
                colorscale="RdYlGn",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Accuracy"),
            ))
            fig.update_layout(
                title="Accuracy by Difficulty (list size)",
                xaxis_title="List Size",
                yaxis_title="Task",
                template=self.theme,
                height=max(400, len(all_tasks) * 22),
            )
            return fig
        return None

    # ------------------------------------------------------------------
    # 4. Token distribution
    # ------------------------------------------------------------------

    def plot_token_distribution(
        self,
        token_data: Union[Dict[str, Any], List[int]],
        token_type: str = "output",
    ) -> Any:
        """
        Histogram of token counts.

        ``token_data`` can be:
          - A list of raw token counts.
          - A results dict with a ``token_analytics`` section.
        """
        tokens: List[int] = []

        if isinstance(token_data, list):
            tokens = [int(t) for t in token_data]
        elif isinstance(token_data, dict):
            analytics = token_data.get("token_analytics", {})
            # Try to reconstruct from summary stats (best effort)
            if "output" in analytics:
                avg = analytics["output"].get("avg", 0)
                tokens = [int(avg)] * analytics.get("count", 1)

        if not tokens:
            return None

        if PLOTLY_AVAILABLE:
            fig = go.Figure(go.Histogram(
                x=tokens,
                nbinsx=30,
                marker_color="#3498db",
                opacity=0.8,
            ))
            fig.update_layout(
                title=f"{token_type.capitalize()} Token Distribution",
                xaxis_title=f"{token_type.capitalize()} Tokens",
                yaxis_title="Count",
                template=self.theme,
            )
            return fig
        return None

    # ------------------------------------------------------------------
    # 5. Latency distribution
    # ------------------------------------------------------------------

    def plot_latency_distribution(
        self,
        latencies: Union[List[float], Dict[str, List[float]]],
    ) -> Any:
        """
        Box plot of per-request latency.

        ``latencies`` can be a flat list or {task_name: [latency_ms, ...]} dict.
        """
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            if isinstance(latencies, dict):
                for task, vals in latencies.items():
                    fig.add_trace(go.Box(y=vals, name=task, boxpoints="outliers"))
            else:
                fig.add_trace(go.Box(y=latencies, name="all tasks", boxpoints="outliers"))
            fig.update_layout(
                title="Latency Distribution (ms)",
                yaxis_title="Latency (ms)",
                template=self.theme,
            )
            return fig
        return None

    # ------------------------------------------------------------------
    # 6. Model comparison
    # ------------------------------------------------------------------

    def plot_model_comparison(
        self,
        results_map: Dict[str, Dict[str, Any]],
    ) -> Any:
        """
        Grouped bar chart comparing multiple models.

        Args:
            results_map: {model_name: results_dict}
        """
        if not results_map:
            return None

        all_tasks = sorted({
            t
            for r in results_map.values()
            for t in _extract_task_accuracy(r).keys()
        })

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            for model_name, res in results_map.items():
                acc_map = _extract_task_accuracy(res)
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=all_tasks,
                    y=[acc_map.get(t, 0.0) for t in all_tasks],
                ))
            fig.update_layout(
                title="Model Comparison by Task",
                xaxis_title="Task",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1.1]),
                barmode="group",
                template=self.theme,
                height=500,
            )
            return fig
        return None

    # ------------------------------------------------------------------
    # 7. Scaling curve
    # ------------------------------------------------------------------

    def plot_scaling_curve(
        self,
        results_by_size: Dict[int, Dict[str, Any]],
        tasks: Optional[List[str]] = None,
    ) -> Any:
        """
        Line chart of accuracy vs list_size.

        Args:
            results_by_size: {list_size: results_dict}
            tasks: Subset of tasks to plot (default: all).
        """
        if not results_by_size:
            return None

        sizes = sorted(results_by_size.keys())
        all_tasks = tasks or sorted({
            t
            for r in results_by_size.values()
            for t in _extract_task_accuracy(r).keys()
        })

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            for task in all_tasks:
                y_vals = [_extract_task_accuracy(results_by_size[s]).get(task, 0.0) for s in sizes]
                fig.add_trace(go.Scatter(
                    x=sizes,
                    y=y_vals,
                    mode="lines+markers",
                    name=task,
                ))
            fig.update_layout(
                title="Scaling Curve: Accuracy vs List Size",
                xaxis_title="List Size",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1.05]),
                template=self.theme,
            )
            return fig
        return None

    # ------------------------------------------------------------------
    # 8. Radar chart
    # ------------------------------------------------------------------

    def plot_radar_chart(
        self,
        results_map: Dict[str, Dict[str, Any]],
        categories: Optional[List[str]] = None,
    ) -> Any:
        """
        Radar / spider chart comparing models across task categories.

        Args:
            results_map: {model_name: results_dict}
            categories: Category names to use as axes (default: unique tasks).
        """
        if not results_map:
            return None

        all_tasks = categories or sorted({
            t
            for r in results_map.values()
            for t in _extract_task_accuracy(r).keys()
        })

        if not all_tasks:
            return None

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            for model_name, res in results_map.items():
                acc_map = _extract_task_accuracy(res)
                r_vals = [acc_map.get(t, 0.0) for t in all_tasks]
                # Close the polygon
                r_vals_closed = r_vals + [r_vals[0]]
                theta_closed = all_tasks + [all_tasks[0]]
                fig.add_trace(go.Scatterpolar(
                    r=r_vals_closed,
                    theta=theta_closed,
                    fill="toself",
                    name=model_name,
                    opacity=0.7,
                ))
            fig.update_layout(
                title="Radar Chart: Model Comparison",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template=self.theme,
            )
            return fig
        return None

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save(
        self,
        fig: Any,
        output_path: str | Path,
        fmt: str = "html",
    ) -> Optional[str]:
        """
        Save a figure to disk.

        Args:
            fig: Plotly Figure or Matplotlib Figure.
            output_path: Destination file path.
            fmt: "html" | "png" | "svg" | "pdf"

        Returns:
            Absolute path string if successful, else None.
        """
        if fig is None:
            logger.warning("Cannot save None figure.")
            return None

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if PLOTLY_AVAILABLE and hasattr(fig, "write_html"):
            if fmt == "html":
                fig.write_html(str(out), include_plotlyjs="cdn")
            elif fmt in ("png", "svg", "pdf"):
                try:
                    fig.write_image(str(out))
                except Exception as e:
                    logger.warning(
                        f"Plotly image export failed (install kaleido: pip install kaleido): {e}"
                    )
                    return None
            else:
                logger.warning(f"Unknown format: {fmt}. Using html.")
                fig.write_html(str(out), include_plotlyjs="cdn")
            logger.info(f"Saved chart to {out}")
            return str(out.resolve())

        # Matplotlib fallback
        if MATPLOTLIB_AVAILABLE and hasattr(fig, "savefig"):
            try:
                fig.savefig(str(out), bbox_inches="tight", dpi=150)
                plt.close(fig)
                return str(out.resolve())
            except Exception as e:
                logger.error(f"Matplotlib save failed: {e}")
                return None

        logger.error("No plotting library available.")
        return None

    def save_all(
        self,
        figures: Dict[str, Any],
        output_dir: str | Path,
        fmt: str = "html",
    ) -> Dict[str, Optional[str]]:
        """
        Save multiple figures.

        Args:
            figures: {name: figure}
            output_dir: Directory to save all files.
            fmt: Output format.

        Returns:
            {name: path_or_None}
        """
        out_dir = Path(output_dir)
        paths: Dict[str, Optional[str]] = {}
        for name, fig in figures.items():
            filename = f"{name}.{fmt}"
            path = self.save(fig, out_dir / filename, fmt=fmt)
            paths[name] = path
        return paths

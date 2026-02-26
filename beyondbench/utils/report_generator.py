"""Comprehensive reporting utilities for beyondbench package."""

import json
import os
import logging
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from importlib.metadata import version as _pkg_version, PackageNotFoundError

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
    logging.warning("tabulate not available. Install with: pip install tabulate")


def calculate_efficiency_score(accuracy, tokens, all_tokens):
    """
    Calculate efficiency score that balances accuracy and token efficiency.

    Efficiency Score = 2 × (Accuracy × Token_Efficiency) / (Accuracy + Token_Efficiency)

    Where:
    - Token_Efficiency = 1 - normalized_tokens
    - normalized_tokens = (tokens - min_tokens) / (max_tokens - min_tokens)

    Args:
        accuracy (float): Model accuracy (0-1)
        tokens (float): Average tokens for this model
        all_tokens (list): List of all token counts across models for normalization

    Returns:
        float: Efficiency score (0-1), higher is better
    """
    if not all_tokens or len(all_tokens) < 2:
        # If we only have one model or no token data, return accuracy
        return accuracy

    min_tokens = min(all_tokens)
    max_tokens = max(all_tokens)

    # Avoid division by zero
    if max_tokens == min_tokens:
        return accuracy

    # Normalize tokens (0 = most efficient, 1 = least efficient)
    normalized_tokens = (tokens - min_tokens) / (max_tokens - min_tokens)

    # Token efficiency (0 = least efficient, 1 = most efficient)
    token_efficiency = 1 - normalized_tokens

    # Harmonic mean of accuracy and token efficiency
    if accuracy + token_efficiency == 0:
        return 0

    efficiency_score = 2 * (accuracy * token_efficiency) / (accuracy + token_efficiency)
    return efficiency_score


def calculate_overthinking_ratio(accuracy, tokens, baseline_tokens=50):
    """
    Calculate overthinking ratio - how many extra tokens were used compared to baseline.

    Args:
        accuracy (float): Model accuracy (0-1)
        tokens (float): Average tokens used by model
        baseline_tokens (float): Baseline token count for reference

    Returns:
        float: Overthinking ratio (>1 means overthinking, <1 means efficient)
    """
    if baseline_tokens <= 0:
        return 1.0

    # Calculate raw overthinking ratio
    raw_ratio = tokens / baseline_tokens

    # Adjust for accuracy - if accuracy is low, penalize overthinking more
    accuracy_weight = max(0.1, accuracy)  # Minimum weight to avoid division issues
    adjusted_ratio = raw_ratio / accuracy_weight

    return adjusted_ratio


def format_report_table(report_data):
    """Format report data into a readable table."""
    if not TABULATE_AVAILABLE:
        return str(report_data)

    if not report_data:
        return "No data to display"

    headers = ["Task", "Accuracy", "Success Rate", "Avg Tokens", "Efficiency Score", "Overthinking Ratio"]
    table_data = []

    for task, data in report_data.items():
        table_data.append([
            task,
            f"{data.get('accuracy', 0):.2%}",
            f"{data.get('success_rate', 0):.2%}",
            f"{data.get('avg_tokens', 0):.1f}",
            f"{data.get('efficiency_score', 0):.3f}",
            f"{data.get('overthinking_ratio', 1):.2f}"
        ])

    return tabulate(table_data, headers=headers, tablefmt="grid")


def generate_efficiency_ranking(report_data):
    """Generate efficiency ranking from report data."""
    rankings = []
    all_tokens = [data.get('avg_tokens', 0) for data in report_data.values()]

    for task, data in report_data.items():
        accuracy = data.get('accuracy', 0)
        tokens = data.get('avg_tokens', 0)

        efficiency_score = calculate_efficiency_score(accuracy, tokens, all_tokens)
        overthinking_ratio = calculate_overthinking_ratio(accuracy, tokens)

        rankings.append({
            'task': task,
            'accuracy': accuracy,
            'tokens': tokens,
            'efficiency_score': efficiency_score,
            'overthinking_ratio': overthinking_ratio
        })

    # Sort by efficiency score (descending)
    rankings.sort(key=lambda x: x['efficiency_score'], reverse=True)
    return rankings


def generate_final_report(results: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive final report from evaluation results.

    Args:
        results: Evaluation results dictionary
        output_path: Optional path to save report

    Returns:
        Formatted report dictionary
    """
    task_results = results.get("task_results", {})

    # Calculate overall metrics
    total_accuracy = np.mean([task.get('accuracy', 0) for task in task_results.values()])
    total_success_rate = np.mean([task.get('success_rate', 0) for task in task_results.values()])

    # Generate efficiency ranking
    efficiency_ranking = generate_efficiency_ranking(task_results)

    report = {
        "summary": {
            "total_tasks": len(task_results),
            "overall_accuracy": total_accuracy,
            "total_evaluations": results.get("total_evaluations", 0),
            "success_rate": total_success_rate,
            "avg_efficiency_score": np.mean([r['efficiency_score'] for r in efficiency_ranking]) if efficiency_ranking else 0,
            "avg_overthinking_ratio": np.mean([r['overthinking_ratio'] for r in efficiency_ranking]) if efficiency_ranking else 1
        },
        "task_details": task_results,
        "efficiency_ranking": efficiency_ranking,
        "formatted_table": format_report_table(task_results),
        "metadata": {
            "model_id": results.get("model_id", "unknown"),
            "timestamp": results.get("timestamp", "unknown"),
            "evaluation_engine": "beyondbench",
            "version": _VERSION
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report


class ReportGenerator:
    """Generate comprehensive evaluation reports."""

    def __init__(self, output_dir: str):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)

    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report from results."""
        # Simplified HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>beyondbench Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .task-result {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; }}
            </style>
        </head>
        <body>
            <h1>🔥 beyondbench Evaluation Report</h1>
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Average Accuracy:</strong> {results.get('summary', {}).get('avg_accuracy', 0):.3f}</p>
                <p><strong>Success Rate:</strong> {results.get('summary', {}).get('success_rate', 0):.1%}</p>
                <p><strong>Total Tasks:</strong> {results.get('summary', {}).get('total_tasks', 0)}</p>
            </div>
            <h2>📋 Task Results</h2>
            <div class="task-results">
                {self._generate_task_results_html(results.get('task_results', {}))}
            </div>
        </body>
        </html>
        """
        return html_content

    def _generate_task_results_html(self, task_results: Dict[str, Any]) -> str:
        """Generate HTML for task results."""
        html_parts = []
        for task_name, result in task_results.items():
            summary = result.get('summary', {})
            accuracy = summary.get('avg_accuracy', 0)
            success_rate = summary.get('success_rate', 0)

            html_parts.append(f"""
            <div class="task-result">
                <h3>{task_name}</h3>
                <p>Accuracy: {accuracy:.3f} | Success Rate: {success_rate:.1%}</p>
            </div>
            """)

        return "".join(html_parts)

    def save_html_report(self, results: Dict[str, Any], filename: str = "report.html"):
        """Save HTML report to file."""
        html_content = self.generate_html_report(results)
        report_file = self.output_dir / filename

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(report_file)
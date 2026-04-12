"""optimize_prompts.py — Prompt optimization utility for BeyondBench (Phase 15).

For a given model and set of tasks, runs evaluation with each prompt style
(concise, detailed, few_shot) and compares accuracy and parse-success rate
per style.  Results are saved to a JSON file and a summary table is printed.

Usage (as a script)::

    python -m beyondbench.scripts.optimize_prompts \\
        --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --tasks sum sorting fibonacci_sequence \\
        --datapoints 20 \\
        --seed 42 \\
        --output-path prompt_optimization_results.json \\
        --gpu-memory-utilization 0.5

Usage (as a library)::

    from beyondbench.scripts.optimize_prompts import optimize_prompts
    results = optimize_prompts(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tasks=["sum", "sorting"],
        datapoints=20,
        seed=42,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

PROMPT_STYLES = ["concise", "detailed", "few_shot"]


def optimize_prompts(
    model_id: str,
    tasks: List[str],
    datapoints: int = 20,
    seed: int = 42,
    output_path: str = "prompt_optimization_results.json",
    backend: Optional[str] = None,
    gpu_memory_utilization: float = 0.5,
    cuda_device: str = "cuda:0",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 8192,
    suite: str = "all",
) -> Dict[str, Any]:
    """Run each prompt style on each task and compare results.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier (or API model name).
    tasks:
        List of task names to evaluate.
    datapoints:
        Number of data points per task per style.
    seed:
        Random seed for data generation.
    output_path:
        Where to write the JSON summary.
    backend:
        Force a specific backend (``"vllm"``, ``"transformers"``, etc.).
    gpu_memory_utilization:
        GPU memory fraction for vLLM.
    cuda_device:
        CUDA device string.
    temperature / top_p / max_tokens:
        Generation hyper-parameters.
    suite:
        Task suite filter (``"easy"``, ``"medium"``, ``"hard"``, or ``"all"``).

    Returns
    -------
    Dict with per-task, per-style accuracy / parse-success results and
    a ``best_style`` recommendation per task.
    """
    from ..models.model_handler import ModelHandler
    from ..core.evaluation_engine import EvaluationEngine

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("=" * 60)
    logger.info("BeyondBench Prompt Optimization Utility")
    logger.info(f"  model_id  : {model_id}")
    logger.info(f"  tasks     : {tasks}")
    logger.info(f"  datapoints: {datapoints}")
    logger.info(f"  styles    : {PROMPT_STYLES}")
    logger.info("=" * 60)

    # Build model handler once — shared across all style runs
    model_handler = ModelHandler(
        model_id=model_id,
        backend=backend,
        cuda_device=cuda_device,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    all_results: Dict[str, Dict[str, Any]] = {}  # task → style → metrics

    for style in PROMPT_STYLES:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Running with prompt_style='{style}'")
        logger.info(f"{'─'*50}")

        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"bb_opt_{style}_") as tmp_dir:
            engine = EvaluationEngine(
                model_handler=model_handler,
                output_dir=tmp_dir,
                store_details=False,
            )

            try:
                run_result = engine.run_evaluation(
                    suite=suite,
                    tasks=tasks,
                    datapoints=datapoints,
                    folds=1,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    prompt_style=style,
                )
            except Exception as exc:
                logger.error(f"Evaluation failed for style '{style}': {exc}")
                continue

            # Extract per-task metrics from the run result
            task_results = run_result.get("task_results", {})
            for task_name, task_data in task_results.items():
                if task_name not in all_results:
                    all_results[task_name] = {}

                # Aggregate metrics across folds / list-sizes
                fold_metrics = task_data if isinstance(task_data, list) else []
                if isinstance(task_data, dict):
                    fold_metrics = task_data.get("fold_metrics", [task_data])

                accuracies = []
                parse_rates = []
                for m in fold_metrics:
                    if isinstance(m, dict):
                        acc = m.get("accuracy", m.get("mean_accuracy", 0.0))
                        prate = m.get(
                            "instruction_followed_pct",
                            m.get("parse_success_rate", 0.0),
                        )
                        accuracies.append(float(acc))
                        parse_rates.append(float(prate))

                all_results[task_name][style] = {
                    "mean_accuracy": (
                        sum(accuracies) / len(accuracies) if accuracies else 0.0
                    ),
                    "parse_success_rate": (
                        sum(parse_rates) / len(parse_rates) if parse_rates else 0.0
                    ),
                    "num_folds": len(accuracies),
                }

    # Determine best style per task
    summary = {}
    for task_name, style_metrics in all_results.items():
        best_style = max(
            style_metrics,
            key=lambda s: style_metrics[s].get("mean_accuracy", 0.0),
            default=None,
        )
        summary[task_name] = {
            "styles": style_metrics,
            "best_style": best_style,
        }

    # Print summary table
    _print_summary_table(summary)

    # Persist results
    output = {
        "model_id": model_id,
        "tasks": tasks,
        "datapoints": datapoints,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": summary,
    }
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_path}")

    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary_table(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("PROMPT OPTIMIZATION SUMMARY")
    print("=" * 80)
    header = f"{'Task':<35} {'concise':>10} {'detailed':>10} {'few_shot':>10} {'BEST':>10}"
    print(header)
    print("-" * 80)
    for task_name, data in sorted(summary.items()):
        styles = data.get("styles", {})
        best = data.get("best_style", "?")
        row = f"{task_name:<35}"
        for s in PROMPT_STYLES:
            acc = styles.get(s, {}).get("mean_accuracy", float("nan"))
            row += f" {acc:>10.3f}"
        row += f" {best:>10}"
        print(row)
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BeyondBench Phase 15 — Prompt Optimization Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-id", required=True, help="HuggingFace model ID or API model name")
    p.add_argument("--tasks", nargs="+", default=[], help="Task names to evaluate (default: all)")
    p.add_argument("--suite", default="all", choices=["easy", "medium", "hard", "all"],
                   help="Task suite to run when --tasks is empty")
    p.add_argument("--datapoints", type=int, default=20, help="Data points per task per style")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output-path", default="prompt_optimization_results.json",
                   help="Output JSON file path")
    p.add_argument("--backend", default=None, help="Force backend (vllm/transformers/openai/gemini)")
    p.add_argument("--cuda-device", default="cuda:0", help="CUDA device")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                   help="GPU memory utilization fraction")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=8192)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    optimize_prompts(
        model_id=args.model_id,
        tasks=args.tasks or [],
        datapoints=args.datapoints,
        seed=args.seed,
        output_path=args.output_path,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cuda_device=args.cuda_device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        suite=args.suite,
    )

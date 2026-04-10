"""
Multi-model comparison workflow for BeyondBench.

Runs the same evaluation suite across N models simultaneously, one model per GPU.
Produces a comparison report: accuracy matrix, parse success matrix, and latency.

Usage (standalone):
    python -m beyondbench.scripts.model_comparison \\
        --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct" \\
        --suite easy --datapoints 20 --gpus 0,1

Usage (CLI):
    beyondbench compare \\
        --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct" \\
        --suite easy --datapoints 20 --gpus 0,1
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Per-model worker (runs in a subprocess)
# ============================================================================

def _worker_compare(
    gpu_id: int,
    model_id: str,
    model_kwargs: Dict[str, Any],
    suite: str,
    tasks: List[str],
    eval_params: Dict[str, Any],
    engine_params: Dict[str, Any],
    output_dir: str,
    result_queue: "mp.Queue",
) -> None:
    """
    Worker process: evaluate *model_id* on the given suite/tasks on GPU *gpu_id*.
    Puts (model_id, results_dict | error_str) into *result_queue*.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[compare GPU {gpu_id} | {model_id.split('/')[-1]}] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)

    try:
        from beyondbench.models.model_handler import ModelHandler
        from beyondbench.core.evaluation_engine import EvaluationEngine
        from beyondbench.parsers.core import get_unparseable_rate, reset_unparseable_stats

        reset_unparseable_stats()

        handler = ModelHandler(
            model_id=model_id,
            cuda_device="cuda:0",  # always 0 inside this process
            **model_kwargs,
        )

        safe_name = model_id.replace("/", "_").replace(".", "-")
        model_output_dir = str(Path(output_dir) / safe_name)

        engine = EvaluationEngine(
            model_handler=handler,
            output_dir=model_output_dir,
            **engine_params,
        )

        start = time.time()
        results = engine.run_evaluation(
            suite=suite,
            tasks=tasks if tasks else None,
            **eval_params,
        )
        elapsed = time.time() - start

        # Compute parse success stats
        unparseable_rate = get_unparseable_rate(model_id)

        results["_comparison_meta"] = {
            "model_id": model_id,
            "gpu_id": gpu_id,
            "elapsed_seconds": elapsed,
            "unparseable_rate": unparseable_rate,
        }

        result_queue.put((model_id, results))
        log.info("Done in %.1fs", elapsed)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("Worker failed: %s\n%s", exc, tb)
        result_queue.put((model_id, f"ERROR: {exc}\n{tb}"))


# ============================================================================
# ModelComparisonRunner
# ============================================================================

class ModelComparisonRunner:
    """
    Run the same evaluation across multiple models (one per GPU) and produce
    a side-by-side comparison report.
    """

    def __init__(
        self,
        models: List[str],
        suite: str = "easy",
        tasks: Optional[List[str]] = None,
        datapoints: int = 20,
        gpus: Optional[List[int]] = None,
        output_dir: str = "./beyondbench_comparison",
        backend: str = "vllm",
        gpu_memory_utilization: float = 0.45,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        store_details: bool = False,
    ):
        self.models = models
        self.suite = suite
        self.tasks = tasks or []
        self.datapoints = datapoints
        self.gpus = gpus
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.store_details = store_details

    def run(self) -> Dict[str, Any]:
        """
        Launch one worker process per model and wait for all to complete.

        Returns:
            Comparison report dict.
        """
        n_models = len(self.models)
        if n_models == 0:
            raise ValueError("No models specified.")

        # Determine GPU assignment
        gpu_ids = self._resolve_gpus(n_models)
        logger.info(
            "Comparing %d models on GPUs %s: %s",
            n_models, gpu_ids, self.models,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        result_queue: mp.Queue = mp.Queue()
        processes: List[mp.Process] = []

        model_kwargs = {
            "backend": self.backend,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        eval_params = {
            "datapoints": self.datapoints,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            eval_params["seed"] = self.seed

        engine_params = {"store_details": self.store_details}

        for model_id, gpu_id in zip(self.models, gpu_ids):
            p = mp.Process(
                target=_worker_compare,
                args=(
                    gpu_id, model_id, model_kwargs,
                    self.suite, self.tasks,
                    eval_params, engine_params,
                    str(self.output_dir), result_queue,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        # Collect results
        raw_results: Dict[str, Any] = {}
        for _ in range(n_models):
            model_id, data = result_queue.get(timeout=3600)
            raw_results[model_id] = data

        for p in processes:
            p.join(timeout=10)

        # Build comparison report
        report = self._build_report(raw_results)

        # Save report
        report_path = self.output_dir / "comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Comparison report saved to %s", report_path)

        return report

    # ------------------------------------------------------------------ #

    def _resolve_gpus(self, n_models: int) -> List[int]:
        if self.gpus and len(self.gpus) >= n_models:
            return self.gpus[:n_models]

        # Auto-detect free GPUs
        try:
            from beyondbench.core.gpu_scheduler import GPUScheduler
            scheduler = GPUScheduler()
            free = scheduler.get_free_gpus()
            if len(free) >= n_models:
                return free[:n_models]
        except Exception:
            pass

        # Fallback: assign GPUs 0..n-1
        return list(range(n_models))

    def _build_report(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert per-model result dicts into a structured comparison report."""
        models_summary: Dict[str, Any] = {}
        task_accuracy_matrix: Dict[str, Dict[str, Any]] = {}  # {task: {model: acc}}
        parse_success_matrix: Dict[str, float] = {}  # {model: parse_success_rate}

        for model_id, data in raw_results.items():
            if isinstance(data, str) and data.startswith("ERROR:"):
                models_summary[model_id] = {"error": data}
                continue

            meta = data.get("_comparison_meta", {})
            summary = data.get("summary", {})

            overall_acc = summary.get("avg_accuracy", 0.0)
            elapsed = meta.get("elapsed_seconds", 0.0)
            unparseable = meta.get("unparseable_rate", 0.0)
            parse_success = max(0.0, 1.0 - unparseable)

            models_summary[model_id] = {
                "overall_accuracy": overall_acc,
                "elapsed_seconds": elapsed,
                "parse_success_rate": parse_success,
                "unparseable_rate": unparseable,
                "total_tasks": summary.get("total_tasks", 0),
            }
            parse_success_matrix[model_id] = parse_success

            # Per-task breakdown
            task_results = data.get("task_results", {})
            for task_name, task_data in task_results.items():
                if task_name not in task_accuracy_matrix:
                    task_accuracy_matrix[task_name] = {}
                task_acc = task_data.get("accuracy", None)
                if task_acc is not None:
                    task_accuracy_matrix[task_name][model_id] = round(task_acc, 4)

        # Rank models by overall accuracy
        ranked = sorted(
            [(m, s.get("overall_accuracy", 0)) for m, s in models_summary.items()
             if "error" not in s],
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "suite": self.suite,
            "datapoints_per_task": self.datapoints,
            "models_evaluated": list(self.models),
            "models_summary": models_summary,
            "ranking": [{"rank": i + 1, "model": m, "accuracy": a} for i, (m, a) in enumerate(ranked)],
            "task_accuracy_matrix": task_accuracy_matrix,
            "parse_success_matrix": parse_success_matrix,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ============================================================================
# Rich table printer
# ============================================================================

def print_comparison_report(report: Dict[str, Any]) -> None:
    """Print a formatted comparison report to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table
        _print_rich(report, Console())
    except ImportError:
        _print_plain(report)


def _print_rich(report: Dict[str, Any], console) -> None:
    from rich.table import Table

    console.print(f"\n[bold cyan]BeyondBench Model Comparison[/bold cyan]  suite={report['suite']}")

    # Overall summary table
    summary_table = Table(title="Model Rankings", show_lines=True)
    summary_table.add_column("Rank", style="dim", justify="center")
    summary_table.add_column("Model", style="bold")
    summary_table.add_column("Accuracy", justify="right")
    summary_table.add_column("Parse Success", justify="right")
    summary_table.add_column("Latency (s)", justify="right")

    for entry in report.get("ranking", []):
        rank = str(entry["rank"])
        model = entry["model"]
        acc = f"{entry['accuracy']:.3f}"
        s = report["models_summary"].get(model, {})
        parse_ok = f"{s.get('parse_success_rate', 0):.1%}"
        latency = f"{s.get('elapsed_seconds', 0):.1f}"
        console.print()  # spacing
        summary_table.add_row(rank, model, acc, parse_ok, latency)

    console.print(summary_table)

    # Per-task accuracy matrix (top tasks)
    matrix = report.get("task_accuracy_matrix", {})
    if matrix:
        mat_table = Table(title="Per-Task Accuracy Matrix", show_lines=True)
        mat_table.add_column("Task", style="cyan")
        models = report["models_evaluated"]
        for m in models:
            mat_table.add_column(m.split("/")[-1], justify="right")

        for task, accs in sorted(matrix.items()):
            row = [task]
            for m in models:
                v = accs.get(m)
                row.append(f"{v:.3f}" if v is not None else "-")
            mat_table.add_row(*row)

        console.print(mat_table)


def _print_plain(report: Dict[str, Any]) -> None:
    print(f"\nBeyondBench Model Comparison — suite={report['suite']}")
    print("=" * 60)
    print("\nRankings:")
    for entry in report.get("ranking", []):
        s = report["models_summary"].get(entry["model"], {})
        print(
            f"  {entry['rank']}. {entry['model']:<45} "
            f"acc={entry['accuracy']:.3f}  "
            f"parse={s.get('parse_success_rate', 0):.1%}  "
            f"time={s.get('elapsed_seconds', 0):.1f}s"
        )
    print()


# ============================================================================
# CLI entry-point (standalone)
# ============================================================================

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare multiple models on BeyondBench tasks."
    )
    parser.add_argument("--models", required=True,
                        help="Comma-separated list of model IDs")
    parser.add_argument("--suite", default="easy",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--tasks", default="",
                        help="Comma-separated task names (default: all in suite)")
    parser.add_argument("--datapoints", type=int, default=20)
    parser.add_argument("--gpus", default="",
                        help="Comma-separated GPU IDs (default: auto-detect)")
    parser.add_argument("--output-dir", default="./beyondbench_comparison")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()] if args.gpus else None

    runner = ModelComparisonRunner(
        models=models,
        suite=args.suite,
        tasks=tasks,
        datapoints=args.datapoints,
        gpus=gpus,
        output_dir=args.output_dir,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    report = runner.run()
    print_comparison_report(report)


if __name__ == "__main__":
    _cli()

"""
Comprehensive CLI for beyondbench package.

Based on proven implementation patterns from BeyondBench codebase with enhanced
robustness and comprehensive parameter support.
"""

import click
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from importlib.metadata import version as _pkg_version, PackageNotFoundError

from ..utils.logging_utils import setup_logging, get_logger
from ..models.model_handler import ModelHandler
from ..core.evaluation_engine import EvaluationEngine

try:
    _VERSION = _pkg_version("beyondbench")
except PackageNotFoundError:
    from .. import __version__ as _VERSION


@click.group()
@click.version_option(version=_VERSION, prog_name="beyondbench")
def main():
    """BeyondBench: Comprehensive LLM Reasoning Evaluation Framework

    Evaluate language model reasoning capabilities across easy, medium, and hard
    computational tasks with contamination-resistant benchmarks.

    Features:
      29 easy suite tasks + 15 new implementations
      5 medium suite tasks with 49 variations
      10 hard suite tasks with 68 variations
      Multi-backend support (vLLM, Transformers, OpenAI, Gemini)
      Robust parsing and comprehensive metrics
    """
    pass


@main.command()
@click.option('--model-id', required=True, help='Model identifier (HuggingFace path or API model name)')
@click.option('--tasks', multiple=True, help='Tasks to evaluate (default: all)')
@click.option('--suite', type=click.Choice(['easy', 'medium', 'hard', 'all']), default='all', help='Task suite to run')

# Backend Configuration
@click.option('--backend', type=click.Choice(['vllm', 'transformers', 'openai', 'gemini']), help='Force specific backend')
@click.option('--api-provider', type=click.Choice(['openai', 'gemini', 'anthropic']), help='API provider for cloud models')
@click.option('--api-key', help='API key (or set OPENAI_API_KEY/GEMINI_API_KEY env var)')

# Hardware Configuration
@click.option('--cuda-device', default='cuda:0', help='CUDA device for local models')
@click.option('--tensor-parallel-size', type=int, default=1, help='Number of GPUs for tensor parallelism')
@click.option('--gpu-memory-utilization', type=float, default=0.96, help='GPU memory utilization ratio')
@click.option('--trust-remote-code', is_flag=True, help='Allow remote code execution')
@click.option('--no-chat-template', is_flag=True, help='Skip chat template application (raw prompt testing)')

# Generation Parameters
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature')
@click.option('--top-p', type=float, default=0.9, help='Nucleus sampling parameter')
@click.option('--max-tokens', type=int, default=32768, help='Maximum tokens to generate (default: 32768, falls back to 8192 on error)')
@click.option('--seed', type=int, help='Random seed for reproducibility')

# API-Specific Parameters
@click.option('--reasoning-effort', type=click.Choice(['minimal', 'low', 'medium', 'high']), default='medium', help='OpenAI GPT-5 reasoning effort')
@click.option('--thinking-budget', type=int, default=1024, help='Gemini thinking budget')

# Evaluation Parameters
@click.option('--datapoints', type=int, default=100, help='Number of datapoints per task/configuration')
@click.option('--folds', type=int, default=1, help='Number of cross-validation folds')
@click.option('--list-sizes', help='Comma-separated list sizes for scalable tasks (e.g., "8,16,32,64")')
@click.option('--range-min', type=int, default=-100, help='Minimum value for number generation')
@click.option('--range-max', type=int, default=100, help='Maximum value for number generation')

# Output Configuration
@click.option('--output-dir', default='./beyondbench_results', help='Output directory for results')
@click.option('--store-details', is_flag=True, help='Store detailed per-example results')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO', help='Logging level')

# Performance Options
@click.option('--batch-size', type=int, default=1, help='Batch size for local model inference')
@click.option('--max-retries', type=int, default=3, help='Maximum retries for failed operations')
@click.option('--timeout', type=int, default=300, help='Timeout for individual operations (seconds)')

# Parser Configuration
@click.option('--parser', type=click.Choice(['unified', 'legacy']), default='unified',
              help='Parser mode: unified (Phase 1 UnifiedParser) or legacy (original per-task parsers)')

# Multi-GPU Parallel Options
@click.option('--parallel', is_flag=True, default=False, help='Enable multi-GPU parallel evaluation')
@click.option('--gpus', default='auto', show_default=True,
              help='GPU IDs for parallel eval: "auto" (detect free GPUs), "0,1,2,3", or "0-7"')
@click.option('--strategy', type=click.Choice(['task_parallel', 'data_parallel', 'model_parallel', 'auto']),
              default='auto', show_default=True,
              help='Parallel strategy: task_parallel (tasks→GPUs), data_parallel (split data), auto')

def evaluate(**kwargs):
    """
    Run comprehensive evaluation on specified tasks.

    Examples:

    \b
    # Evaluate GPT-4o on easy suite
    beyondbench evaluate --model-id gpt-4o --api-provider openai --suite easy

    \b
    # Evaluate local model with vLLM
    beyondbench evaluate --model-id meta-llama/Llama-3.2-3B-Instruct --backend vllm

    \b
    # Evaluate specific tasks with custom parameters
    beyondbench evaluate --model-id gpt-5 --tasks sorting,comparison --datapoints 50 --reasoning-effort high
    """

    # Setup logging
    output_dir = Path(kwargs['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        output_dir=str(output_dir),
        log_level=kwargs['log_level'],
        enable_file_logging=True,
        enable_console_logging=True
    )

    logger = get_logger("CLI")
    logger.info("Starting beyondbench evaluation")

    # Print banner
    print_banner()

    # Validate configuration
    config = validate_and_prepare_config(kwargs)

    # Apply parser mode globally
    parser_mode = config.get('parser_mode', 'unified')
    try:
        from ..parsers.settings import set_parser_mode
        set_parser_mode(parser_mode)
        logger.info(f"Parser mode: {parser_mode}")
    except Exception:
        pass

    # Determine if parallel mode is requested
    use_parallel = config.get('parallel', False)

    try:
        if use_parallel:
            # ------------------------------------------------------------------ #
            # Parallel multi-GPU evaluation
            # ------------------------------------------------------------------ #
            from ..core.parallel_engine import ParallelEvaluationEngine
            from ..core.gpu_scheduler import GPUScheduler

            logger.info(f"[PARALLEL] Initialising parallel engine (gpus={config['gpus']}, strategy={config['strategy']})")

            # We still create a lightweight model_handler for metadata only.
            # Workers create their own handlers inside spawned processes.
            # Use a stub/api handler if available so parent process loads no GPU weights.
            model_handler = _create_metadata_handler(config['model_config'], logger)

            parallel_engine = ParallelEvaluationEngine(
                model_handler=model_handler,
                output_dir=str(output_dir),
                **config['engine_config']
            )

            scheduler = GPUScheduler()
            gpu_ids = scheduler.parse_gpu_spec(config['gpus'], config['model_id'])

            if not gpu_ids:
                logger.error("No suitable GPUs found. Check --gpus argument and GPU memory.")
                sys.exit(1)

            logger.info(f"[PARALLEL] Using GPUs: {gpu_ids}  strategy={config['strategy']}")

            results = parallel_engine.run_parallel_evaluation(
                suite=config['suite'],
                tasks=config['tasks'],
                gpu_ids=gpu_ids,
                strategy=config['strategy'],
                model_id=config['model_id'],
                model_kwargs=config['model_config'],
                **config['eval_config']
            )

        else:
            # ------------------------------------------------------------------ #
            # Standard single-GPU evaluation
            # ------------------------------------------------------------------ #
            logger.info(f"Initializing model handler for {config['model_id']}")
            model_handler = ModelHandler(**config['model_config'])

            logger.info("Initializing evaluation engine")
            evaluation_engine = EvaluationEngine(
                model_handler=model_handler,
                output_dir=str(output_dir),
                **config['engine_config']
            )

            logger.info(f"Running evaluation on {config['suite']} suite")
            results = evaluation_engine.run_evaluation(
                suite=config['suite'],
                tasks=config['tasks'],
                **config['eval_config']
            )

        # Print results summary
        print_results_summary(results)

        # Save final results (engine already saves, but ensure CLI output dir has them)
        results_file = output_dir / "final_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Evaluation completed successfully! Results saved to {output_dir}")

        # Print statistics (only for single-GPU mode where we have a handler)
        if not use_parallel and hasattr(model_handler, 'print_statistics_report'):
            model_handler.print_statistics_report()

    except KeyError as e:
        _handle_cli_error(e, "configuration")
    except ImportError as e:
        _handle_cli_error(e, "import")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


@main.command()
@click.option('--suite', type=click.Choice(['easy', 'medium', 'hard', 'all']), default='all', help='Task suite to list')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
def list_tasks(suite: str, output_format: str):
    """List available tasks in each suite."""

    logger = get_logger("CLI")

    try:
        from ..core.task_registry import TaskRegistry
        registry = TaskRegistry()

        tasks = registry.get_available_tasks(suite)

        if output_format == 'table':
            print_tasks_table(tasks)
        elif output_format == 'json':
            click.echo(json.dumps(tasks, indent=2))
        elif output_format == 'yaml':
            import yaml
            click.echo(yaml.dump(tasks, default_flow_style=False))

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
def run_config(config_file: str):
    """Run evaluation from a YAML or JSON configuration file.

    The config file should have top-level keys: model, evaluation, output,
    and optionally performance. Keys are flattened and mapped to CLI params.

    Example:

    \b
      beyondbench run-config beyondbench/configs/default.yaml
    """

    logger = get_logger("CLI")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Validate against schema if jsonschema is available
        _validate_config(config)

        logger.info(f"Running evaluation from config: {config_file}")

        # Flatten nested config into flat CLI-compatible kwargs
        flat = _flatten_config(config)

        # Invoke the evaluate command with flattened params
        ctx = click.Context(evaluate)
        ctx.invoke(evaluate, **flat)

    except click.exceptions.BadParameter as e:
        _handle_cli_error(e, "configuration")
    except Exception as e:
        logger.error(f"Failed to run config: {e}")
        sys.exit(1)


@main.command()
@click.argument('task_name')
def info(task_name: str):
    """Show detailed information about a specific task.

    Displays the task suite, description, and available parameters using
    a Rich formatted panel.

    Example:

    \b
      beyondbench info sorting
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        from ..core.task_registry import TaskRegistry

        console = Console()
        registry = TaskRegistry()
        task_info = registry.get_task_info(task_name)

        if task_info is None:
            console.print(
                f"[bold red]Error:[/bold red] Task '{task_name}' not found.\n"
                f"Run [bold]beyondbench list-tasks[/bold] to see all available tasks."
            )
            sys.exit(1)

        # Build info table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")

        table.add_row("Name", task_info["name"])
        table.add_row("Suite", task_info["suite"].upper())
        table.add_row("Type", task_info["type"])
        table.add_row("Available", "Yes" if task_info["available"] else "No")
        table.add_row("Description", task_info["description"])

        # Try to get additional info from the task class itself
        task_class = registry.get_task_class(task_name)
        if task_class and task_class.__doc__:
            doc = task_class.__doc__.strip().split('\n')[0]
            table.add_row("Class Doc", doc)

        console.print(Panel(table, title=f"Task: {task_name}", border_style="green"))

    except ImportError:
        # Fallback without Rich
        from ..core.task_registry import TaskRegistry
        registry = TaskRegistry()
        task_info = registry.get_task_info(task_name)
        if task_info is None:
            click.echo(f"Error: Task '{task_name}' not found.")
            sys.exit(1)
        click.echo(f"\nTask: {task_name}")
        click.echo(f"  Suite:       {task_info['suite'].upper()}")
        click.echo(f"  Type:        {task_info['type']}")
        click.echo(f"  Available:   {'Yes' if task_info['available'] else 'No'}")
        click.echo(f"  Description: {task_info['description']}")


# ------------------------------------------------------------------ #
# results command group
# ------------------------------------------------------------------ #
@main.group()
def results():
    """View and compare past evaluation results."""
    pass


@results.command("list")
@click.option('--output-dir', default='./beyondbench_results', help='Directory to scan for results')
def results_list(output_dir: str):
    """List past evaluation results found in the output directory.

    Scans recursively for final_results.json files.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        results_dir = Path(output_dir)

        if not results_dir.exists():
            console.print(f"[yellow]Output directory does not exist:[/yellow] {output_dir}")
            return

        result_files = sorted(results_dir.rglob("final_results.json"))
        if not result_files:
            console.print("[yellow]No evaluation results found.[/yellow]")
            return

        table = Table(title="Evaluation Results")
        table.add_column("#", style="dim")
        table.add_column("Path", style="cyan")
        table.add_column("Tasks", justify="right")
        table.add_column("Avg Accuracy", justify="right")
        table.add_column("Duration", justify="right")

        for idx, rf in enumerate(result_files, 1):
            try:
                with open(rf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                summary = data.get("summary", {})
                n_tasks = str(summary.get("total_tasks", "?"))
                avg_acc = f"{summary.get('avg_accuracy', 0):.3f}" if "avg_accuracy" in summary else "?"
                duration = f"{summary.get('total_duration', 0):.1f}s" if "total_duration" in summary else "?"
            except Exception:
                n_tasks = "?"
                avg_acc = "?"
                duration = "?"
            table.add_row(str(idx), str(rf), n_tasks, avg_acc, duration)

        console.print(table)

    except ImportError:
        # Fallback without Rich
        results_dir = Path(output_dir)
        if not results_dir.exists():
            click.echo(f"Output directory does not exist: {output_dir}")
            return
        result_files = sorted(results_dir.rglob("final_results.json"))
        if not result_files:
            click.echo("No evaluation results found.")
            return
        click.echo("\nEvaluation Results:")
        click.echo("-" * 60)
        for idx, rf in enumerate(result_files, 1):
            click.echo(f"  {idx}. {rf}")


@results.command("show")
@click.argument('path', type=click.Path(exists=True))
def results_show(path: str):
    """Show detailed results from a results file.

    PATH should point to a final_results.json file.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import print_json

        console = Console()

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        summary = data.get("summary", {})

        # Summary panel
        summary_lines = []
        for key in ["total_duration", "total_tasks", "completed_tasks", "failed_tasks",
                     "total_evaluations", "success_rate", "avg_accuracy"]:
            val = summary.get(key)
            if val is not None:
                if isinstance(val, float):
                    if key in ("success_rate", "avg_accuracy", "avg_success_rate"):
                        summary_lines.append(f"[bold]{key}:[/bold] {val:.3f}")
                    else:
                        summary_lines.append(f"[bold]{key}:[/bold] {val:.1f}")
                else:
                    summary_lines.append(f"[bold]{key}:[/bold] {val}")
        console.print(Panel("\n".join(summary_lines), title="Evaluation Summary", border_style="green"))

        # Per-task table
        task_results = data.get("task_results", {})
        if task_results:
            table = Table(title="Per-Task Results")
            table.add_column("Task", style="cyan")
            table.add_column("Accuracy", justify="right")
            table.add_column("Status", justify="center")

            for task_name, task_result in task_results.items():
                acc = _extract_accuracy(task_result)
                table.add_row(task_name, f"{acc:.3f}", "OK" if acc > 0 else "?")

            console.print(table)

    except ImportError:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        click.echo(json.dumps(data.get("summary", {}), indent=2))


@results.command("compare")
@click.argument('path1', type=click.Path(exists=True))
@click.argument('path2', type=click.Path(exists=True))
def results_compare(path1: str, path2: str):
    """Compare two evaluation results side by side.

    Both PATH1 and PATH2 should point to final_results.json files.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        with open(path1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(path2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)

        s1 = data1.get("summary", {})
        s2 = data2.get("summary", {})

        # Overall comparison
        table = Table(title="Results Comparison")
        table.add_column("Metric", style="bold")
        table.add_column("Result A", justify="right", style="cyan")
        table.add_column("Result B", justify="right", style="magenta")
        table.add_column("Delta", justify="right")

        for key in ["avg_accuracy", "success_rate", "total_duration", "total_evaluations",
                     "completed_tasks", "failed_tasks"]:
            v1 = s1.get(key, 0)
            v2 = s2.get(key, 0)
            if isinstance(v1, float):
                delta = v2 - v1
                sign = "+" if delta > 0 else ""
                style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
                table.add_row(key, f"{v1:.3f}", f"{v2:.3f}", f"[{style}]{sign}{delta:.3f}[/{style}]")
            else:
                delta = v2 - v1 if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else 0
                sign = "+" if delta > 0 else ""
                table.add_row(key, str(v1), str(v2), f"{sign}{delta}" if delta != 0 else "-")

        console.print(table)

        # Per-task comparison
        tr1 = data1.get("task_results", {})
        tr2 = data2.get("task_results", {})
        all_tasks = sorted(set(list(tr1.keys()) + list(tr2.keys())))

        if all_tasks:
            task_table = Table(title="Per-Task Accuracy Comparison")
            task_table.add_column("Task", style="bold")
            task_table.add_column("A", justify="right", style="cyan")
            task_table.add_column("B", justify="right", style="magenta")
            task_table.add_column("Delta", justify="right")

            for task_name in all_tasks:
                a1 = _extract_accuracy(tr1.get(task_name, {}))
                a2 = _extract_accuracy(tr2.get(task_name, {}))
                delta = a2 - a1
                sign = "+" if delta > 0 else ""
                style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
                task_table.add_row(
                    task_name,
                    f"{a1:.3f}" if task_name in tr1 else "-",
                    f"{a2:.3f}" if task_name in tr2 else "-",
                    f"[{style}]{sign}{delta:.3f}[/{style}]"
                )

            console.print(task_table)

    except ImportError:
        with open(path1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(path2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        click.echo("Result A summary:")
        click.echo(json.dumps(data1.get("summary", {}), indent=2))
        click.echo("\nResult B summary:")
        click.echo(json.dumps(data2.get("summary", {}), indent=2))


# ------------------------------------------------------------------ #
# compare command (Phase 4)
# ------------------------------------------------------------------ #

@main.command()
@click.option('--models', required=True,
              help='Comma-separated list of model IDs to compare')
@click.option('--suite', type=click.Choice(['easy', 'medium', 'hard', 'all']), default='easy',
              show_default=True, help='Task suite')
@click.option('--tasks', default='', help='Comma-separated task names (default: all in suite)')
@click.option('--datapoints', type=int, default=20, show_default=True,
              help='Datapoints per task per model')
@click.option('--gpus', default='',
              help='Comma-separated GPU IDs (e.g. "0,1,2").  Auto-detected if empty.')
@click.option('--output-dir', default='./beyondbench_comparison', show_default=True,
              help='Directory for comparison reports and per-model results')
@click.option('--backend', type=click.Choice(['vllm', 'transformers']), default='vllm',
              show_default=True, help='Local inference backend')
@click.option('--gpu-memory-utilization', type=float, default=0.45, show_default=True,
              help='GPU memory utilization per model (keep low when running many models in parallel)')
@click.option('--seed', type=int, default=None, help='Random seed for reproducibility')
@click.option('--store-details', is_flag=True, help='Store detailed per-example results')
def compare(
    models: str,
    suite: str,
    tasks: str,
    datapoints: int,
    gpus: str,
    output_dir: str,
    backend: str,
    gpu_memory_utilization: float,
    seed: Optional[int],
    store_details: bool,
):
    """Compare multiple models side-by-side on the same task suite.

    Runs one model per GPU in parallel and produces an accuracy matrix,
    parse success matrix, and ranked summary table.

    Examples:

    \b
    # Compare 4 Qwen variants on GPUs 0-3
    beyondbench compare \\
        --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct" \\
        --suite easy --datapoints 20 --gpus 0,1,2,3

    \b
    # Compare Llama and Qwen on easy suite
    beyondbench compare \\
        --models "meta-llama/Llama-3.2-3B-Instruct,Qwen/Qwen2.5-1.5B-Instruct" \\
        --suite easy --datapoints 50
    """
    from ..utils.logging_utils import setup_logging, get_logger

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir=str(output_path), log_level="INFO",
                  enable_file_logging=True, enable_console_logging=True)
    log = get_logger("CLI.compare")

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    gpu_list = [int(g.strip()) for g in gpus.split(",") if g.strip()] if gpus else None

    if not model_list:
        click.echo("Error: --models must specify at least one model.", err=True)
        sys.exit(1)

    log.info("Comparing %d models: %s", len(model_list), model_list)

    try:
        from ..scripts.model_comparison import ModelComparisonRunner, print_comparison_report

        runner = ModelComparisonRunner(
            models=model_list,
            suite=suite,
            tasks=task_list,
            datapoints=datapoints,
            gpus=gpu_list,
            output_dir=output_dir,
            backend=backend,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            store_details=store_details,
        )

        report = runner.run()
        print_comparison_report(report)

        click.echo(f"\nFull report saved to {output_path / 'comparison_report.json'}")

    except Exception as exc:
        log.error("Comparison failed: %s", exc)
        sys.exit(1)


# ------------------------------------------------------------------ #
# init command
# ------------------------------------------------------------------ #
@main.command()
@click.option('--output', '-o', default='beyondbench.yaml', help='Output config file path')
def init(output: str):
    """Create a beyondbench.yaml configuration file interactively.

    Prompts for model, backend, suite, datapoints, and output directory,
    then writes a ready-to-use config file.
    """
    try:
        from rich.console import Console
        from rich.prompt import Prompt, IntPrompt
        console = Console()

        console.print("[bold green]BeyondBench Configuration Wizard[/bold green]\n")

        model_id = Prompt.ask(
            "[cyan]Model ID[/cyan] (e.g. gpt-4o, meta-llama/Llama-3.2-3B-Instruct)",
            default="gpt-4o"
        )

        api_provider = Prompt.ask(
            "[cyan]API provider / backend[/cyan]",
            choices=["openai", "gemini", "anthropic", "vllm", "transformers"],
            default="openai"
        )

        # Determine backend vs api_provider
        if api_provider in ("vllm", "transformers"):
            backend = api_provider
            provider = None
        else:
            backend = None
            provider = api_provider

        suite = Prompt.ask(
            "[cyan]Task suite[/cyan]",
            choices=["easy", "medium", "hard", "all"],
            default="easy"
        )

        datapoints = IntPrompt.ask("[cyan]Datapoints per task[/cyan]", default=100)

        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="./beyondbench_results"
        )

    except ImportError:
        # Fallback without Rich
        click.echo("BeyondBench Configuration Wizard\n")

        model_id = click.prompt("Model ID", default="gpt-4o")
        api_provider = click.prompt(
            "API provider / backend (openai/gemini/anthropic/vllm/transformers)",
            default="openai"
        )
        if api_provider in ("vllm", "transformers"):
            backend = api_provider
            provider = None
        else:
            backend = None
            provider = api_provider

        suite = click.prompt("Task suite (easy/medium/hard/all)", default="easy")
        datapoints = click.prompt("Datapoints per task", default=100, type=int)
        output_dir = click.prompt("Output directory", default="./beyondbench_results")

    # Build config dict
    config = {"model": {"model_id": model_id}}
    if backend:
        config["model"]["backend"] = backend
    if provider:
        config["model"]["api_provider"] = provider

    config["evaluation"] = {
        "suite": suite,
        "datapoints": datapoints,
        "folds": 1,
        "list_sizes": [8, 16, 32],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 32768,
    }

    config["output"] = {
        "output_dir": output_dir,
        "store_details": False,
        "log_level": "INFO",
    }

    import yaml
    out_path = Path(output)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"\nConfiguration written to {out_path.resolve()}")
    click.echo(f"Run with: beyondbench run-config {output}")


# ------------------------------------------------------------------ #
# serve command (replaces stub)
# ------------------------------------------------------------------ #
@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to listen on')
@click.option('--reload', 'use_reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, use_reload: bool):
    """Start the BeyondBench API server.

    Launches a FastAPI application served by uvicorn with endpoints for
    task listing, evaluation, and result retrieval.

    Requires the [serve] extra: pip install beyondbench[serve]
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "Error: The 'serve' extra is required to run the API server.\n"
            "Install it with:\n\n"
            "  pip install beyondbench[serve]\n\n"
            "This installs FastAPI and uvicorn."
        )
        sys.exit(1)

    try:
        from ..serve.app import create_app  # noqa: F811 - verify import works
    except ImportError as exc:
        click.echo(f"Error: Could not import serve module: {exc}")
        sys.exit(1)

    click.echo(f"Starting BeyondBench API server on {host}:{port}")
    click.echo(f"API docs available at http://{host}:{port}/docs")

    uvicorn.run(
        "beyondbench.serve.app:create_app",
        host=host,
        port=port,
        reload=use_reload,
        factory=True,
    )


# ------------------------------------------------------------------ #
# wizard and chat (unchanged)
# ------------------------------------------------------------------ #
@main.command()
def wizard():
    """Launch the interactive BeyondBench wizard."""
    from .wizard import main_wizard
    main_wizard()


@main.command()
def chat():
    """Start an interactive chat session with a model (coming soon)."""
    click.echo(click.style("Interactive chat coming soon!", fg='yellow'))
    click.echo("For now, use the evaluate command to test models.")


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #

def print_banner():
    """Print stylized banner."""
    banner = f"""
 ____                            _ ____                  _
| __ )  ___ _   _  ___  _ __   __| | __ )  ___ _ __   ___| |__
|  _ \\ / _ \\ | | |/ _ \\| '_ \\ / _` |  _ \\ / _ \\ '_ \\ / __| '_ \\
| |_) |  __/ |_| | (_) | | | | (_| | |_) |  __/ | | | (__| | | |
|____/ \\___|\\__, |\\___/|_| |_|\\__,_|____/ \\___|_| |_|\\___|_| |_|
            |___/

BeyondBench Evaluation Framework v{_VERSION}
Comprehensive LLM Reasoning Evaluation
"""
    click.echo(click.style(banner, fg='cyan', bold=True))


def _auto_detect_api_provider(model_id: str) -> Optional[str]:
    """Auto-detect API provider from model_id prefix."""
    model_lower = model_id.lower()
    if model_lower.startswith('gpt-') or model_lower.startswith('o1-') or model_lower.startswith('o3-'):
        return 'openai'
    if model_lower.startswith('gemini-'):
        return 'gemini'
    if model_lower.startswith('claude-'):
        return 'anthropic'
    return None


def _auto_detect_api_key(provider: str) -> Optional[str]:
    """Auto-detect API key from standard environment variables."""
    env_vars = {
        'openai': 'OPENAI_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
    }
    var = env_vars.get(provider)
    if var:
        return os.environ.get(var)
    return None


def _parse_list_sizes(list_sizes_str: str) -> List[int]:
    """Parse list sizes supporting ranges and comma-separated values.

    Supports:
    - Comma-separated: "8,16,32,64"
    - Ranges: "8-64" -> [8, 16, 32, 64] (powers of 2)
    - Mixed: "4,8-32,128" -> [4, 8, 16, 32, 128]

    Raises:
        click.BadParameter: If format is invalid or values are not positive integers
    """
    sizes = []
    for part in list_sizes_str.split(','):
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # Range: "8-64" -> powers of 2 between 8 and 64
            try:
                start, end = [int(x.strip()) for x in part.split('-', 1)]
                if start <= 0 or end <= 0:
                    raise click.BadParameter(f"List sizes must be positive integers, got range {start}-{end}")
                val = start
                while val <= end:
                    sizes.append(val)
                    val *= 2
            except ValueError:
                raise click.BadParameter(f"Invalid range format: '{part}'. Use 'start-end' (e.g., '8-64')")
        else:
            try:
                val = int(part)
                if val <= 0:
                    raise click.BadParameter(f"List sizes must be positive integers, got {val}")
                sizes.append(val)
            except ValueError:
                raise click.BadParameter(f"Invalid list size: '{part}'. Use comma-separated positive integers")
    return sizes


def validate_and_prepare_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate CLI arguments and prepare configuration."""

    # Auto-detect API provider from model_id if not specified
    if not kwargs.get('api_provider') and not kwargs.get('backend'):
        detected = _auto_detect_api_provider(kwargs['model_id'])
        if detected:
            kwargs['api_provider'] = detected

    # Auto-detect API key from environment if not specified
    if kwargs.get('api_provider') and not kwargs.get('api_key'):
        detected_key = _auto_detect_api_key(kwargs['api_provider'])
        if detected_key:
            kwargs['api_key'] = detected_key

    # Parse list sizes with range support
    list_sizes = None
    if kwargs.get('list_sizes'):
        list_sizes = _parse_list_sizes(kwargs['list_sizes'])

    # Prepare model configuration
    model_config = {
        'model_id': kwargs['model_id'],
        'backend': kwargs.get('backend'),
        'api_provider': kwargs.get('api_provider'),
        'api_key': kwargs.get('api_key'),
        'cuda_device': kwargs['cuda_device'],
        'tensor_parallel_size': kwargs['tensor_parallel_size'],
        'gpu_memory_utilization': kwargs['gpu_memory_utilization'],
        'trust_remote_code': kwargs['trust_remote_code'],
        'reasoning_effort': kwargs['reasoning_effort'],
        'thinking_budget': kwargs['thinking_budget'],
        'skip_chat_template': kwargs.get('no_chat_template', False)
    }

    # Prepare engine configuration
    engine_config = {
        'max_retries': kwargs['max_retries'],
        'store_details': kwargs['store_details']
    }

    # Prepare evaluation configuration
    eval_config = {
        'datapoints': kwargs['datapoints'],
        'folds': kwargs['folds'],
        'list_sizes': list_sizes,
        'range_min': kwargs['range_min'],
        'range_max': kwargs['range_max'],
        'temperature': kwargs['temperature'],
        'top_p': kwargs['top_p'],
        'max_tokens': kwargs['max_tokens']
    }

    # Process tasks - handle both comma-separated strings and multiple values
    tasks = kwargs.get('tasks', [])
    if tasks:
        # Flatten and split comma-separated values
        processed_tasks = []
        for task in tasks:
            if ',' in task:
                processed_tasks.extend([t.strip() for t in task.split(',')])
            else:
                processed_tasks.append(task.strip())
        tasks = processed_tasks

    return {
        'model_id': kwargs['model_id'],
        'suite': kwargs['suite'],
        'tasks': tasks,
        'model_config': model_config,
        'engine_config': engine_config,
        'eval_config': eval_config,
        'parser_mode': kwargs.get('parser', 'unified'),
        # Parallel options
        'parallel': kwargs.get('parallel', False),
        'gpus': kwargs.get('gpus', 'auto'),
        'strategy': kwargs.get('strategy', 'auto'),
    }


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested YAML config dict into flat CLI-compatible kwargs.

    Sections (model, evaluation, output, performance) are merged into a single
    level dict with keys matching the evaluate command parameters.
    """
    flat: Dict[str, Any] = {}

    model = config.get("model", {})
    flat["model_id"] = model.get("model_id")
    for key in ("backend", "api_provider", "api_key", "cuda_device",
                "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code"):
        if key in model:
            flat[key.replace("-", "_")] = model[key]

    evaluation = config.get("evaluation", {})
    for key in ("suite", "tasks", "datapoints", "folds", "temperature", "top_p",
                "max_tokens", "seed", "reasoning_effort", "thinking_budget",
                "range_min", "range_max"):
        if key in evaluation:
            flat[key.replace("-", "_")] = evaluation[key]

    # list_sizes: convert list to comma-separated string for CLI compatibility
    if "list_sizes" in evaluation:
        ls = evaluation["list_sizes"]
        if isinstance(ls, list):
            flat["list_sizes"] = ",".join(str(x) for x in ls)
        else:
            flat["list_sizes"] = str(ls)

    output = config.get("output", {})
    for key in ("output_dir", "store_details", "log_level"):
        if key in output:
            flat[key.replace("-", "_")] = output[key]

    performance = config.get("performance", {})
    for key in ("batch_size", "max_retries", "timeout"):
        if key in performance:
            flat[key.replace("-", "_")] = performance[key]

    # Apply defaults for required fields the evaluate command expects
    defaults = {
        "suite": "all",
        "cuda_device": "cuda:0",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.96,
        "trust_remote_code": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 32768,
        "reasoning_effort": "medium",
        "thinking_budget": 1024,
        "datapoints": 100,
        "folds": 1,
        "range_min": -100,
        "range_max": 100,
        "output_dir": "./beyondbench_results",
        "store_details": False,
        "log_level": "INFO",
        "batch_size": 1,
        "max_retries": 3,
        "timeout": 300,
    }

    for key, default_val in defaults.items():
        flat.setdefault(key, default_val)

    # Convert tasks list to tuple for Click multiple option
    if "tasks" in flat and isinstance(flat["tasks"], list):
        flat["tasks"] = tuple(flat["tasks"])
    elif "tasks" not in flat:
        flat["tasks"] = ()

    return flat


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate a config dict against the BeyondBench JSON schema.

    Silently passes if jsonschema is not installed or schema file is missing.
    """
    try:
        import jsonschema
        schema_path = Path(__file__).parent.parent / "configs" / "schema.json"
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            jsonschema.validate(config, schema)
    except ImportError:
        pass
    except jsonschema.ValidationError as e:
        raise click.BadParameter(
            f"Config validation error: {e.message}\n"
            f"  Path: {' -> '.join(str(p) for p in e.absolute_path)}"
        )


def _extract_accuracy(task_result: Any) -> float:
    """Extract accuracy from a task result (handles various formats)."""
    if isinstance(task_result, list) and task_result:
        return sum(m.get('accuracy', 0) for m in task_result) / len(task_result)
    elif isinstance(task_result, dict):
        if 'summary' in task_result:
            return task_result['summary'].get('avg_accuracy', 0)
        elif 'overall_accuracy' in task_result:
            return task_result.get('overall_accuracy', 0)
    return 0.0


def _create_metadata_handler(model_config: Dict[str, Any], log=None):
    """Create a lightweight ModelHandler for the parent process.

    In parallel mode the parent process must NOT load GPU weights — that's
    the worker's job.  We create either an API handler (no GPU) or, for local
    models, a thin stub that provides get_model_info() / get_statistics() but
    skips model loading.
    """
    if model_config.get("api_provider"):
        # API models are lightweight — safe to init in parent
        return ModelHandler(**model_config)

    # Local model: create a stub to avoid loading weights in the parent
    class _MetaStub:
        def __init__(self, cfg):
            self.model_id = cfg.get("model_id", "")
            self.backend = cfg.get("backend", "vllm")
            self.api_provider = None
            self.stats = {
                "api_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "start_time": __import__("time").time(),
                "prompts_processed": [],
            }
        def get_model_info(self):
            return {"model_name": self.model_id, "backend": self.backend, "api_provider": None, "model_type": "local"}
        def get_statistics(self):
            return self.stats
        def print_statistics_report(self):
            pass
        def generate(self, *a, **kw):
            raise RuntimeError("Parent process stub — use worker processes for inference")

    return _MetaStub(model_config)


def _handle_cli_error(exc: Exception, error_type: str = "general") -> None:
    """Handle common CLI errors with helpful Rich-formatted messages."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console(stderr=True)

        if error_type == "import":
            msg = str(exc)
            suggestion = ""
            if "openai" in msg.lower():
                suggestion = "pip install beyondbench[openai]"
            elif "google" in msg.lower() or "genai" in msg.lower():
                suggestion = "pip install beyondbench[gemini]"
            elif "anthropic" in msg.lower():
                suggestion = "pip install beyondbench[anthropic]"
            elif "vllm" in msg.lower():
                suggestion = "pip install beyondbench[vllm]"
            elif "fastapi" in msg.lower() or "uvicorn" in msg.lower():
                suggestion = "pip install beyondbench[serve]"

            lines = [f"[bold red]Import Error:[/bold red] {exc}"]
            if suggestion:
                lines.append(f"\n[bold]Suggestion:[/bold] Install the required extra:\n  {suggestion}")
            console.print(Panel("\n".join(lines), title="Missing Dependency", border_style="red"))

        elif error_type == "configuration":
            msg = str(exc)
            lines = [f"[bold red]Configuration Error:[/bold red] {msg}"]
            if "api_key" in msg.lower() or "api-key" in msg.lower():
                lines.append(
                    "\n[bold]Suggestion:[/bold] Set your API key via environment variable:\n"
                    "  export OPENAI_API_KEY=sk-...\n"
                    "  export GEMINI_API_KEY=...\n"
                    "Or pass --api-key on the command line."
                )
            console.print(Panel("\n".join(lines), title="Configuration Error", border_style="red"))
        else:
            console.print(f"[bold red]Error:[/bold red] {exc}")

    except ImportError:
        click.echo(f"Error ({error_type}): {exc}", err=True)

    sys.exit(1)


def print_tasks_table(tasks: Dict[str, List[str]]):
    """Print tasks in a formatted table."""

    click.echo("\nAvailable Tasks by Suite\n")

    for suite_name, task_list in tasks.items():
        click.echo(click.style(f"{suite_name.upper()} SUITE ({len(task_list)} tasks)", fg='green', bold=True))
        click.echo("-" * 50)

        for i, task in enumerate(task_list, 1):
            click.echo(f"  {i:2d}. {task}")

        click.echo()


def print_results_summary(results: Dict[str, Any]):
    """Print evaluation results summary."""

    click.echo("\n" + "="*80)
    click.echo(click.style("EVALUATION RESULTS SUMMARY", fg='green', bold=True))
    click.echo("="*80)

    if 'summary' in results:
        summary = results['summary']

        click.echo(f"  Total Time: {summary.get('total_duration', 0):.1f}s")
        click.echo(f"  Total Evaluations: {summary.get('total_evaluations', 0):,}")
        click.echo(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        click.echo(f"  Average Accuracy: {summary.get('avg_accuracy', 0):.3f}")

        if 'task_results' in results:
            click.echo("\nPer-Task Results:")
            click.echo("-" * 50)

            for task_name, task_result in results['task_results'].items():
                accuracy = _extract_accuracy(task_result)

                # Compute success rate
                if isinstance(task_result, list) and task_result:
                    success_rate = sum(1 for m in task_result if m.get('generation_success', True)) / len(task_result)
                elif isinstance(task_result, dict):
                    if 'summary' in task_result:
                        success_rate = task_result['summary'].get('success_rate', 0)
                    elif 'overall_accuracy' in task_result:
                        fold_results = task_result.get('fold_results', [])
                        if fold_results:
                            total = sum(f.get('total_count', 0) for f in fold_results)
                            correct = sum(f.get('correct_count', 0) for f in fold_results)
                            success_rate = correct / total if total > 0 else 0
                        else:
                            success_rate = 0
                    else:
                        success_rate = 0
                else:
                    success_rate = 0

                click.echo(f"  {task_name:25s} | Accuracy: {accuracy:.3f} | Success: {success_rate:.1%}")

    click.echo("="*80)


# Make wizard the default when no command is specified
def cli():
    """Entry point that shows wizard if no args provided."""
    import sys
    if len(sys.argv) == 1:
        # No arguments - launch wizard
        from .wizard import main_wizard
        main_wizard()
    else:
        # Arguments provided - run normal CLI
        main()


if __name__ == '__main__':
    cli()

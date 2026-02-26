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
    """🔥 beyondbench: BeyondBench Evaluation Package

    Comprehensive evaluation framework for language model reasoning capabilities.

    Features:
    • 29 easy suite tasks + 15 new implementations
    • 5 medium suite tasks with 49 variations
    • 10 hard suite tasks with 68 variations
    • Multi-backend support (vLLM, Transformers, OpenAI, Gemini)
    • Robust parsing and comprehensive metrics
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
    logger.info("🚀 Starting beyondbench evaluation")

    # Print banner
    print_banner()

    # Validate configuration
    config = validate_and_prepare_config(kwargs)

    try:
        # Initialize model handler
        logger.info(f"🤖 Initializing model handler for {config['model_id']}")
        model_handler = ModelHandler(**config['model_config'])

        # Initialize evaluation engine
        logger.info("🔧 Initializing evaluation engine")
        evaluation_engine = EvaluationEngine(
            model_handler=model_handler,
            output_dir=str(output_dir),
            **config['engine_config']
        )

        # Run evaluation
        logger.info(f"📊 Running evaluation on {config['suite']} suite")
        results = evaluation_engine.run_evaluation(
            suite=config['suite'],
            tasks=config['tasks'],
            **config['eval_config']
        )

        # Print results summary
        print_results_summary(results)

        # Save final results (evaluation_engine already saves, but this ensures CLI output dir has them)
        results_file = output_dir / "final_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ Evaluation completed successfully! Results saved to {output_dir}")

        # Print statistics
        model_handler.print_statistics_report()

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
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
    """Run evaluation from configuration file."""

    logger = get_logger("CLI")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        logger.info(f"📋 Running evaluation from config: {config_file}")

        # Convert config to CLI arguments and run
        ctx = click.Context(evaluate)
        ctx.invoke(evaluate, **config)

    except Exception as e:
        logger.error(f"Failed to run config: {e}")
        sys.exit(1)


def print_banner():
    """Print stylized banner."""
    banner = f"""
██████╗ ██████╗       ███████╗██╗   ██╗ █████╗ ██╗
██╔══██╗██╔══██╗      ██╔════╝██║   ██║██╔══██╗██║
██████╔╝██████╔╝█████╗█████╗  ██║   ██║███████║██║
██╔══██╗██╔══██╗╚════╝██╔══╝  ╚██╗ ██╔╝██╔══██║██║
██████╔╝██████╔╝      ███████╗ ╚████╔╝ ██║  ██║███████╗
╚═════╝ ╚═════╝       ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝

🔥 BeyondBench Evaluation Framework v{_VERSION}
📊 Comprehensive LLM Reasoning Evaluation
"""
    click.echo(click.style(banner, fg='cyan', bold=True))


def validate_and_prepare_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate CLI arguments and prepare configuration."""

    # Parse list sizes
    list_sizes = None
    if kwargs.get('list_sizes'):
        try:
            list_sizes = [int(x.strip()) for x in kwargs['list_sizes'].split(',')]
        except ValueError:
            raise click.BadParameter("Invalid list-sizes format. Use comma-separated integers (e.g., '8,16,32,64')")

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
        'thinking_budget': kwargs['thinking_budget']
    }

    # Prepare engine configuration
    engine_config = {
        'max_retries': kwargs['max_retries'],
        'seed': kwargs.get('seed'),
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
        'eval_config': eval_config
    }


def print_tasks_table(tasks: Dict[str, List[str]]):
    """Print tasks in a formatted table."""

    click.echo("\n📋 Available Tasks by Suite\n")

    for suite_name, task_list in tasks.items():
        click.echo(click.style(f"🎯 {suite_name.upper()} SUITE ({len(task_list)} tasks)", fg='green', bold=True))
        click.echo("─" * 50)

        for i, task in enumerate(task_list, 1):
            click.echo(f"  {i:2d}. {task}")

        click.echo()


def print_results_summary(results: Dict[str, Any]):
    """Print evaluation results summary."""

    click.echo("\n" + "="*80)
    click.echo(click.style("📊 EVALUATION RESULTS SUMMARY", fg='green', bold=True))
    click.echo("="*80)

    if 'summary' in results:
        summary = results['summary']

        click.echo(f"⏱️  Total Time: {summary.get('total_duration', 0):.1f}s")
        click.echo(f"📊 Total Evaluations: {summary.get('total_evaluations', 0):,}")
        click.echo(f"✅ Success Rate: {summary.get('success_rate', 0):.1%}")
        click.echo(f"🎯 Average Accuracy: {summary.get('avg_accuracy', 0):.3f}")

        if 'task_results' in results:
            click.echo("\n📈 Per-Task Results:")
            click.echo("-" * 50)

            for task_name, task_result in results['task_results'].items():
                # Handle both list and dict return formats
                if isinstance(task_result, list):
                    # task_result is a list of metrics, compute averages
                    if task_result:
                        accuracy = sum(m.get('accuracy', 0) for m in task_result) / len(task_result)
                        success_rate = sum(1 for m in task_result if m.get('generation_success', True)) / len(task_result)
                    else:
                        accuracy = 0
                        success_rate = 0
                elif isinstance(task_result, dict):
                    if 'summary' in task_result:
                        accuracy = task_result['summary'].get('avg_accuracy', 0)
                        success_rate = task_result['summary'].get('success_rate', 0)
                    elif 'overall_accuracy' in task_result:
                        # Handle tasks with custom run_evaluation (e.g., RobustTowerHanoiTask)
                        accuracy = task_result.get('overall_accuracy', 0)
                        fold_results = task_result.get('fold_results', [])
                        if fold_results:
                            total = sum(f.get('total_count', 0) for f in fold_results)
                            correct = sum(f.get('correct_count', 0) for f in fold_results)
                            success_rate = correct / total if total > 0 else 0
                        else:
                            success_rate = 0
                    else:
                        accuracy = 0
                        success_rate = 0
                else:
                    accuracy = 0
                    success_rate = 0

                click.echo(f"  {task_name:25s} | Accuracy: {accuracy:.3f} | Success: {success_rate:.1%}")

    click.echo("="*80)


@main.command()
def wizard():
    """Launch the interactive BeyondBench wizard."""
    from .wizard import main_wizard
    main_wizard()


@main.command()
def serve():
    """Start the BeyondBench API server (coming soon)."""
    click.echo(click.style("🚧 API server coming soon!", fg='yellow'))
    click.echo("For now, use the CLI or Python API to run evaluations.")


@main.command()
def chat():
    """Start an interactive chat session with a model (coming soon)."""
    click.echo(click.style("🚧 Interactive chat coming soon!", fg='yellow'))
    click.echo("For now, use the evaluate command to test models.")


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
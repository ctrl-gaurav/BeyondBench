"""
Comprehensive CLI for beyondbench package.

Based on proven implementation patterns from BeyondBench codebase with enhanced
robustness and comprehensive parameter support.
"""

import click
import os
import sys
import time
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


class CLIContext:
    """Shared CLI state: verbosity, output mode, logger.

    Populated by the top-level group and consumed by subcommands via
    ``@click.pass_obj``. Subcommands that do not opt into the context object
    can still read it from ``click.get_current_context().obj``.
    """

    def __init__(self) -> None:
        self.verbose: bool = False
        self.quiet: bool = False
        self.json_mode: bool = False

    @property
    def log_level(self) -> str:
        if self.quiet:
            return "ERROR"
        if self.verbose:
            return "DEBUG"
        return "INFO"

    def echo(self, message: str, *, err: bool = False) -> None:
        """Print a human-facing line, suppressed in --quiet and --json modes."""
        if self.quiet or self.json_mode:
            return
        click.echo(message, err=err)

    def emit_json(self, payload: Any) -> None:
        """Emit a machine-readable JSON payload to stdout.

        Always prints regardless of --quiet; used by subcommands when
        --json is active to return structured output.
        """
        click.echo(json.dumps(payload, indent=2, default=str))


def _get_cli_ctx() -> "CLIContext":
    """Fetch the shared CLIContext from the active click context.

    Returns a fresh default if no context is active (e.g. unit tests
    calling helpers directly).
    """
    try:
        ctx = click.get_current_context(silent=True)
        if ctx is not None and isinstance(ctx.obj, CLIContext):
            return ctx.obj
    except RuntimeError:
        pass
    return CLIContext()


@click.group()
@click.version_option(version=_VERSION, prog_name="beyondbench")
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Enable verbose output (DEBUG log level, extra diagnostics).",
)
@click.option(
    "--quiet", "-q", is_flag=True, default=False,
    help="Suppress non-essential output (ERROR log level only).",
)
@click.option(
    "--json", "json_mode", is_flag=True, default=False,
    help="Machine-readable JSON output mode for supported commands.",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, quiet: bool, json_mode: bool) -> None:
    """BeyondBench: Comprehensive LLM Reasoning Evaluation Framework

    Evaluate language model reasoning capabilities across easy, medium, and hard
    computational tasks with contamination-resistant benchmarks.

    Features:
      29 easy suite tasks + 15 new implementations
      5 medium suite tasks with 49 variations
      10 hard suite tasks with 68 variations
      Multi-backend support (vLLM, Transformers, OpenAI, Gemini)
      Robust parsing and comprehensive metrics

    Global flags:
      --verbose / -v  more logging (DEBUG)
      --quiet / -q    suppress human-facing output
      --json          emit structured JSON for supported commands
    """
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive")

    cli_ctx = CLIContext()
    cli_ctx.verbose = verbose
    cli_ctx.quiet = quiet
    cli_ctx.json_mode = json_mode
    ctx.obj = cli_ctx

    # Propagate log level into the shared logger and the BEYONDBENCH_LOG_LEVEL
    # env var so any subprocess/worker we spawn inherits it.
    os.environ["BEYONDBENCH_LOG_LEVEL"] = cli_ctx.log_level
    try:
        setup_logging(level=cli_ctx.log_level)
    except Exception:
        # Logging init is best-effort; never crash the CLI on logger setup.
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

# Dashboard Options (Phase 6)
@click.option('--dashboard', 'launch_dashboard_flag', is_flag=True, default=False,
              help='Launch the Gradio observability dashboard alongside evaluation')
@click.option('--dashboard-port', type=int, default=7860, show_default=True,
              help='Port for the live dashboard (requires --dashboard)')

# Prompt Engineering (Phase 15)
@click.option('--prompt-style', type=click.Choice(['concise', 'detailed', 'few_shot', 'cot']),
              default=None,
              help='Prompt style variant for all tasks (concise/detailed/few_shot/cot). '
                   'Omit to use each task\'s built-in prompt (default behaviour).')

# Contamination Resistance (Phase 16)
@click.option('--contamination-resistance', type=click.Choice(['off', 'low', 'medium', 'high']),
              default='off', show_default=True,
              help='Contamination resistance level: noise injection + prompt rephrasing. '
                   'off = no changes (default), low = minor formatting, '
                   'medium = add context, high = distractors + heavy rephrasing.')

# Pre-flight Validation (Phase 9.2.1)
@click.option('--skip-validation', is_flag=True, default=False,
              help='Skip pre-flight model validation (HF existence, GPU memory, deps)')
@click.option('--validation-warn-only', is_flag=True, default=False,
              help='Treat all pre-flight validation issues as warnings instead of errors')

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

    # Pre-flight model validation (Phase 9.2.1)
    _run_preflight_validation(
        kwargs,
        skip=kwargs.get('skip_validation', False),
        warn_only=kwargs.get('validation_warn_only', False),
        logger=logger,
    )

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

    # ------------------------------------------------------------------ #
    # Dashboard setup (Phase 6) — optional, launched before evaluation
    # ------------------------------------------------------------------ #
    use_dashboard = kwargs.get('launch_dashboard_flag', False)
    dashboard_port = kwargs.get('dashboard_port', 7860)
    data_bridge = None
    _dashboard_app = None

    if use_dashboard:
        try:
            import gradio  # noqa: F401
            from ..dashboard.data_bridge import DataBridge, BridgeLogHandler
            from ..dashboard.app import create_dashboard

            data_bridge = DataBridge()

            # Attach bridge log handler so evaluation logs flow into dashboard
            bridge_log_handler = BridgeLogHandler(data_bridge)
            import logging as _logging
            _logging.getLogger().addHandler(bridge_log_handler)

            _dashboard_app = create_dashboard(data_bridge=data_bridge, results_dir=str(output_dir))
            _launch_extras = getattr(_dashboard_app, '_beyondbench_launch_extras', {})
            _dashboard_app.launch(server_port=dashboard_port, share=False, prevent_thread_lock=True, **_launch_extras)
            logger.info(f"Dashboard launched at http://0.0.0.0:{dashboard_port}")
            click.echo(f"Dashboard available at http://localhost:{dashboard_port}")
        except ImportError:
            click.echo(
                "Warning: --dashboard requires 'gradio'. Install with: pip install beyondbench[dashboard]\n"
                "Continuing without dashboard."
            )
            use_dashboard = False

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

            if data_bridge is not None:
                data_bridge.notify_evaluation_started(
                    total_tasks=0,
                    model_info=model_handler.get_model_info() if hasattr(model_handler, 'get_model_info') else {},
                )

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

            if data_bridge is not None:
                data_bridge.notify_evaluation_complete(results)

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

            # Wire up bridge callbacks if dashboard is active
            if data_bridge is not None:
                _wire_bridge_to_engine(evaluation_engine, data_bridge, config)

            logger.info(f"Running evaluation on {config['suite']} suite")
            results = evaluation_engine.run_evaluation(
                suite=config['suite'],
                tasks=config['tasks'],
                **config['eval_config']
            )

            if data_bridge is not None:
                data_bridge.notify_evaluation_complete(results)

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

        # Keep dashboard alive after evaluation completes
        if use_dashboard and _dashboard_app is not None:
            click.echo(f"\nEvaluation done. Dashboard still running at http://localhost:{dashboard_port}")
            click.echo("Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

    except KeyError as e:
        if data_bridge is not None:
            data_bridge.notify_error(str(e))
        _handle_cli_error(e, "configuration")
    except ImportError as e:
        if data_bridge is not None:
            data_bridge.notify_error(str(e))
        _handle_cli_error(e, "import")
    except Exception as e:
        if data_bridge is not None:
            data_bridge.notify_error(str(e))
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


@main.command()
@click.option('--suite', type=click.Choice(['easy', 'medium', 'hard', 'all']), default='all', help='Task suite to list')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.pass_obj
def list_tasks(cli_ctx: "CLIContext", suite: str, output_format: str):
    """List available tasks in each suite.

    When the global ``--json`` flag is set, output is forced to JSON
    regardless of ``--format``.
    """

    logger = get_logger("CLI")

    # Global --json overrides per-command --format
    if cli_ctx is not None and cli_ctx.json_mode:
        output_format = 'json'

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
@click.option('--skip-validation', is_flag=True, default=False,
              help='Skip pre-flight model validation (HF existence, GPU memory, deps)')
@click.option('--validation-warn-only', is_flag=True, default=False,
              help='Treat all pre-flight validation issues as warnings instead of errors')
def run_config(config_file: str, skip_validation: bool, validation_warn_only: bool):
    """Run evaluation from a YAML or JSON configuration file.

    The config file should have top-level keys: model, evaluation, output,
    and optionally performance. Keys are flattened and mapped to CLI params.

    Environment variables (``BEYONDBENCH_*``) and any ``.env`` file in the
    current directory are applied on top of the file values.

    Example:

    \b
      beyondbench run-config beyondbench/configs/default.yaml
    """

    logger = get_logger("CLI")

    try:
        # Use the env-var + .env-aware loader so overrides apply transparently.
        from ..configs import load_and_validate_config
        config = load_and_validate_config(config_file)

        logger.info(f"Running evaluation from config: {config_file}")

        # Flatten nested config into flat CLI-compatible kwargs
        flat = _flatten_config(config)
        # Propagate skip flags into the evaluate invocation
        flat['skip_validation'] = skip_validation
        flat['validation_warn_only'] = validation_warn_only

        # Invoke the evaluate command with flattened params
        ctx = click.Context(evaluate)
        ctx.invoke(evaluate, **flat)

    except click.exceptions.BadParameter as e:
        _handle_cli_error(e, "configuration")
    except Exception as e:
        logger.error(f"Failed to run config: {e}")
        sys.exit(1)


@main.command(name="task-info")
@click.argument('task_name')
def task_info_cmd(task_name: str):
    """Show detailed information about a specific task.

    Displays the task suite, description, and available parameters using
    a Rich formatted panel.

    Example:

    \b
      beyondbench task-info sorting
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
# info command — system information (Phase 10)
# ------------------------------------------------------------------ #

@main.command()
@click.pass_obj
def info(cli_ctx: "CLIContext"):
    """Show system information: GPUs, Python, backends, and package version.

    With the global ``--json`` flag, emits a structured JSON payload
    suitable for scripts and CI.

    Example:

    \b
      beyondbench info
      beyondbench --json info
    """
    json_mode = bool(cli_ctx and cli_ctx.json_mode)
    payload: Dict[str, Any] = {}

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box as rbox
        console = None if json_mode else Console()
    except ImportError:
        console = None  # type: ignore

    def _echo_section(title: str, rows: list, *, key: Optional[str] = None):
        # Always record into payload for JSON mode.
        if key is not None:
            # Strip Rich markup for the JSON payload so downstream tools
            # don't see leaked formatting codes.
            import re as _re
            clean = [
                (k, _re.sub(r"\[/?[a-zA-Z0-9 _]+\]", "", str(v)))
                for k, v in rows
            ]
            payload[key] = dict(clean)
        if json_mode:
            return
        if console:
            t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
            t.add_column("Key", style="bold cyan", min_width=28)
            t.add_column("Value")
            for k, v in rows:
                t.add_row(k, str(v))
            console.print(Panel(t, title=f"[bold]{title}[/bold]", border_style="blue"))
        else:
            click.echo(f"\n{title}")
            click.echo("-" * 40)
            for k, v in rows:
                click.echo(f"  {k:<28} {v}")

    # ── Package info ────────────────────────────────────────────────
    pkg_rows = [("Version", _VERSION)]
    try:
        import importlib.metadata as _im
        dist = _im.metadata("beyondbench")
        pkg_rows.append(("Author", dist.get("Author-email", "—")))
        pkg_rows.append(("License", dist.get("License", "—")))
    except Exception:
        pass
    _echo_section("BeyondBench", pkg_rows, key="package")

    # ── Python / platform ───────────────────────────────────────────
    import platform, sys as _sys
    py_rows = [
        ("Python", _sys.version.split()[0]),
        ("Platform", platform.platform()),
        ("Architecture", platform.machine()),
    ]
    _echo_section("Python / Platform", py_rows, key="platform")

    # ── GPU info ────────────────────────────────────────────────────
    gpu_rows = []
    try:
        import subprocess, re
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                idx, name, mem_total, mem_free, util = [x.strip() for x in line.split(",")]
                used_mb = int(mem_total) - int(mem_free)
                gpu_rows.append((
                    f"GPU {idx}: {name}",
                    f"{used_mb} / {mem_total} MiB  ({util}% util)"
                ))
        else:
            gpu_rows.append(("CUDA GPUs", "nvidia-smi not available"))
    except Exception as e:
        gpu_rows.append(("CUDA GPUs", f"Detection failed: {e}"))
    _echo_section("GPUs", gpu_rows, key="gpus")

    # ── Backend availability ─────────────────────────────────────────
    backend_rows = []
    for pkg, label in [
        ("vllm", "vLLM"),
        ("transformers", "Transformers (HuggingFace)"),
        ("openai", "OpenAI SDK"),
        ("google.generativeai", "Google Generative AI (Gemini)"),
        ("anthropic", "Anthropic SDK"),
        ("torch", "PyTorch"),
        ("gradio", "Gradio (dashboard)"),
        ("fastapi", "FastAPI (server)"),
        ("rich", "Rich (CLI formatting)"),
        ("jsonschema", "jsonschema (config validation)"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            backend_rows.append((label, f"[green]✓[/green] {ver}" if console else f"OK  {ver}"))
        except ImportError:
            backend_rows.append((label, "[red]✗ not installed[/red]" if console else "MISSING"))
    _echo_section("Backends & Dependencies", backend_rows, key="backends")

    # ── Config presets ──────────────────────────────────────────────
    from ..configs import list_presets
    preset_rows = [(name, str(path)) for name, path in list_presets().items()]
    if preset_rows:
        _echo_section("Available Config Presets", preset_rows, key="presets")

    # ── Env var overrides ──────────────────────────────────────────
    from ..configs import get_env_overrides
    overrides = get_env_overrides()
    if overrides:
        import json as _json
        env_rows = [("Active overrides", _json.dumps(overrides, indent=2))]
        _echo_section("Environment Variable Overrides (BEYONDBENCH_*)", env_rows, key="env_overrides")
        payload["env_overrides_raw"] = overrides

    if json_mode:
        click.echo(json.dumps(payload, indent=2, default=str))


# ------------------------------------------------------------------ #
# doctor command — diagnose issues (Phase 10)
# ------------------------------------------------------------------ #

@main.command()
@click.pass_obj
def doctor(cli_ctx: "CLIContext"):
    """Diagnose common setup issues: CUDA, vLLM, imports, API keys.

    Runs a series of checks and reports which pass or fail, with
    suggested fixes for any problems.

    With the global ``--json`` flag, emits a structured JSON payload and
    exits with status 1 if any checks fail.

    Example:

    \b
      beyondbench doctor
      beyondbench --json doctor
    """
    json_mode = bool(cli_ctx and cli_ctx.json_mode)
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box as rbox
        console = None if json_mode else Console()
        _OK    = "[bold green]PASS[/bold green]"
        _WARN  = "[bold yellow]WARN[/bold yellow]"
        _FAIL  = "[bold red]FAIL[/bold red]"
    except ImportError:
        console = None  # type: ignore
        _OK    = "PASS"
        _WARN  = "WARN"
        _FAIL  = "FAIL"

    checks: list[tuple[str, str, str]] = []  # (label, status, note)

    def _add(label: str, status: str, note: str = ""):
        checks.append((label, status, note))

    # ── CUDA ────────────────────────────────────────────────────────
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if r.returncode == 0:
            _add("nvidia-smi available", _OK)
        else:
            _add("nvidia-smi available", _FAIL, "Install NVIDIA drivers")
    except Exception:
        _add("nvidia-smi available", _FAIL, "NVIDIA drivers or nvidia-smi not found")

    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            _add("PyTorch CUDA", _OK, f"{n} GPU(s) visible")
        else:
            _add("PyTorch CUDA", _WARN,
                 "torch.cuda.is_available()=False — install CUDA toolkit or use CPU backend")
    except ImportError:
        _add("PyTorch CUDA", _WARN, "torch not installed — run: pip install torch")

    # ── vLLM ────────────────────────────────────────────────────────
    try:
        import vllm  # noqa: F401
        _add("vLLM import", _OK, getattr(vllm, "__version__", "installed"))
    except ImportError:
        _add("vLLM import", _WARN,
             "vLLM not installed — run: pip install beyondbench[vllm]  (required for local models)")

    # ── Transformers ────────────────────────────────────────────────
    try:
        import transformers  # noqa: F401
        _add("transformers import", _OK, transformers.__version__)
    except ImportError:
        _add("transformers import", _WARN,
             "transformers not installed — run: pip install transformers")

    # ── API packages ────────────────────────────────────────────────
    for pkg, label, extra in [
        ("openai", "OpenAI SDK", "openai"),
        ("google.generativeai", "Google GenerativeAI", "gemini"),
        ("anthropic", "Anthropic SDK", "anthropic"),
    ]:
        try:
            __import__(pkg)
            _add(f"{label} import", _OK)
        except ImportError:
            _add(f"{label} import", _WARN,
                 f"Optional — run: pip install beyondbench[{extra}]")

    # ── API keys in environment ──────────────────────────────────────
    for env_var, label in [
        ("OPENAI_API_KEY", "OpenAI API key"),
        ("GEMINI_API_KEY", "Gemini API key"),
        ("ANTHROPIC_API_KEY", "Anthropic API key"),
    ]:
        if os.environ.get(env_var):
            _add(f"{label} ({env_var})", _OK, "Set")
        else:
            _add(f"{label} ({env_var})", _WARN,
                 f"Not set — export {env_var}=<your-key>  (only needed for API models)")

    # ── Core imports ────────────────────────────────────────────────
    for dotted, label in [
        ("beyondbench.core.evaluation_engine", "EvaluationEngine"),
        ("beyondbench.models.model_handler", "ModelHandler"),
        ("beyondbench.core.task_registry", "TaskRegistry"),
        ("beyondbench.parsers.core", "UnifiedParser"),
    ]:
        try:
            __import__(dotted)
            _add(f"Core import: {label}", _OK)
        except Exception as exc:
            _add(f"Core import: {label}", _FAIL, str(exc))

    # ── Config schema ───────────────────────────────────────────────
    try:
        import jsonschema  # noqa: F401
        _add("jsonschema (config validation)", _OK)
    except ImportError:
        _add("jsonschema (config validation)", _WARN,
             "Optional — run: pip install jsonschema")

    # ── Rich (CLI formatting) ────────────────────────────────────────
    try:
        import rich  # noqa: F401
        _add("rich (CLI formatting)", _OK)
    except ImportError:
        _add("rich (CLI formatting)", _WARN, "run: pip install rich")

    # Normalize statuses for summary counts (strip rich markup)
    def _status_word(s: str) -> str:
        if "FAIL" in s:
            return "FAIL"
        if "WARN" in s:
            return "WARN"
        if "PASS" in s or "OK" in s:
            return "PASS"
        return s

    fail_count = sum(1 for _, s, _ in checks if _status_word(s) == "FAIL")
    warn_count = sum(1 for _, s, _ in checks if _status_word(s) == "WARN")
    pass_count = len(checks) - fail_count - warn_count

    # ── JSON output mode ─────────────────────────────────────────────
    if json_mode:
        click.echo(json.dumps(
            {
                "checks": [
                    {"label": label, "status": _status_word(status), "note": note}
                    for label, status, note in checks
                ],
                "summary": {
                    "total": len(checks),
                    "passed": pass_count,
                    "warnings": warn_count,
                    "failed": fail_count,
                },
            },
            indent=2,
        ))
        if fail_count:
            sys.exit(1)
        return

    # ── Print results ────────────────────────────────────────────────
    if console:
        from rich.panel import Panel
        table = Table(show_header=True, box=rbox.SIMPLE, padding=(0, 1))
        table.add_column("Check", style="bold", min_width=40)
        table.add_column("Status", justify="center", min_width=8)
        table.add_column("Notes")
        for label, status, note in checks:
            table.add_row(label, status, note)
        console.print(Panel(table, title="[bold]BeyondBench Doctor[/bold]", border_style="cyan"))
        summary_color = "red" if fail_count else ("yellow" if warn_count else "green")
        console.print(
            f"[{summary_color}]"
            f"Results: {len(checks)} checks — "
            f"{fail_count} failed, {warn_count} warnings, "
            f"{pass_count} passed"
            f"[/{summary_color}]"
        )
    else:
        click.echo("\nBeyondBench Doctor\n" + "=" * 50)
        for label, status, note in checks:
            click.echo(f"  [{_status_word(status)}] {label}" + (f"  — {note}" if note else ""))
        click.echo(f"\nResults: {len(checks)} checks — "
                   f"{fail_count} failed, {warn_count} warnings, {pass_count} passed")


# ------------------------------------------------------------------ #
# benchmark command — quick pre-configured run (Phase 10)
# ------------------------------------------------------------------ #

@main.command()
@click.option("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct", show_default=True,
              help="Model to benchmark")
@click.option("--backend", type=click.Choice(["vllm", "transformers"]), default="vllm",
              show_default=True, help="Inference backend")
@click.option("--cuda-device", default="cuda:0", show_default=True, help="GPU to use")
@click.option("--gpu-memory-utilization", type=float, default=0.45, show_default=True,
              help="GPU memory utilization (keep low for a quick benchmark)")
@click.option("--datapoints", type=int, default=50, show_default=True,
              help="Datapoints per task")
@click.option("--output-dir", default="./beyondbench_results/benchmark", show_default=True,
              help="Output directory")
@click.option("--store-details", is_flag=True, help="Store per-example details")
def benchmark(
    model_id: str,
    backend: str,
    cuda_device: str,
    gpu_memory_utilization: float,
    datapoints: int,
    output_dir: str,
    store_details: bool,
):
    """Run a quick pre-configured benchmark on the easy suite.

    Runs the easy task suite with sensible defaults and saves results.
    Useful for sanity-checking a new model or environment.

    Examples:

    \b
    # Quick benchmark with default Qwen model
    beyondbench benchmark

    \b
    # Benchmark a custom model
    beyondbench benchmark --model-id Qwen/Qwen2.5-3B-Instruct --gpu-memory-utilization 0.5
    """
    # Reuse the evaluate command via ctx.invoke
    ctx = click.get_current_context()
    ctx.invoke(
        evaluate,
        model_id=model_id,
        tasks=(),
        suite="easy",
        backend=backend,
        api_provider=None,
        api_key=None,
        cuda_device=cuda_device,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=False,
        no_chat_template=False,
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,
        seed=42,
        reasoning_effort="medium",
        thinking_budget=1024,
        datapoints=datapoints,
        folds=1,
        list_sizes=None,
        range_min=-100,
        range_max=100,
        output_dir=output_dir,
        store_details=store_details,
        log_level="INFO",
        batch_size=1,
        max_retries=3,
        timeout=300,
        parser="unified",
        parallel=False,
        gpus="auto",
        strategy="auto",
        launch_dashboard_flag=False,
        dashboard_port=7860,
    )


# ------------------------------------------------------------------ #
# export command — CSV / Excel export (Phase 10)
# ------------------------------------------------------------------ #

@main.command()
@click.argument("results_dir", default="./beyondbench_results",
                type=click.Path())
@click.option("--format", "export_format",
              type=click.Choice(["csv", "excel", "xlsx", "json"]),
              default="csv", show_default=True,
              help="Export format")
@click.option("--output", "-o", default="", help="Output file path (auto-named if omitted)")
def export(results_dir: str, export_format: str, output: str):
    """Export evaluation results to CSV, Excel, or JSON.

    RESULTS_DIR should contain a final_results.json file (searched recursively).

    Examples:

    \b
    # Export to CSV
    beyondbench export ./beyondbench_results --format csv

    \b
    # Export to Excel
    beyondbench export ./beyondbench_results --format excel -o my_results.xlsx
    """
    import json as _json
    from pathlib import Path as _Path

    rdir = _Path(results_dir)

    # Find results file
    if (rdir / "final_results.json").exists():
        results_path = rdir / "final_results.json"
    else:
        candidates = sorted(rdir.rglob("final_results.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            click.echo(f"Error: No final_results.json found in {results_dir}", err=True)
            sys.exit(1)
        results_path = candidates[0]

    with open(results_path, "r", encoding="utf-8") as f:
        data = _json.load(f)

    # Build flat rows from task_results
    task_results = data.get("task_results", {})
    summary = data.get("summary", {})
    model_id = data.get("model_id", summary.get("model_id", "unknown"))

    rows = []
    for task_name, task_result in task_results.items():
        row: Dict[str, Any] = {"model_id": model_id, "task": task_name}
        if isinstance(task_result, list) and task_result:
            accs = [r.get("accuracy", 0) for r in task_result]
            row["accuracy"] = sum(accs) / len(accs)
            row["n_folds"] = len(task_result)
            # Aggregate token usage across folds
            total_in = sum(r.get("total_input_tokens", 0) for r in task_result)
            total_out = sum(r.get("total_output_tokens", 0) for r in task_result)
            row["total_input_tokens"] = total_in
            row["total_output_tokens"] = total_out
            row["parse_success_rate"] = sum(
                r.get("parse_success_rate", 0) for r in task_result
            ) / len(task_result)
        elif isinstance(task_result, dict):
            ts = task_result.get("summary", task_result)
            row["accuracy"] = ts.get("avg_accuracy", ts.get("overall_accuracy", 0))
            row["n_folds"] = 1
            row["total_input_tokens"] = ts.get("total_input_tokens", 0)
            row["total_output_tokens"] = ts.get("total_output_tokens", 0)
            row["parse_success_rate"] = ts.get("parse_success_rate", 0)
        rows.append(row)

    if not rows:
        click.echo("No task results found to export.", err=True)
        sys.exit(1)

    # Determine output path
    fmt_norm = "xlsx" if export_format in ("excel", "xlsx") else export_format
    if output:
        out_path = _Path(output)
    else:
        out_path = results_path.parent / f"results.{fmt_norm}"

    if fmt_norm == "csv":
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        click.echo(f"Exported {len(rows)} rows to {out_path}")

    elif fmt_norm == "xlsx":
        try:
            import openpyxl
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "BeyondBench Results"
            headers = list(rows[0].keys())
            ws.append(headers)
            for row in rows:
                ws.append([row.get(h, "") for h in headers])
            # Auto-width columns
            for col_cells in ws.columns:
                max_len = max(len(str(c.value or "")) for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)
            wb.save(out_path)
            click.echo(f"Exported {len(rows)} rows to {out_path}")
        except ImportError:
            click.echo(
                "Error: openpyxl is required for Excel export.\n"
                "Install with: pip install openpyxl", err=True
            )
            sys.exit(1)

    elif fmt_norm == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
        click.echo(f"Exported {len(rows)} rows to {out_path}")


# ------------------------------------------------------------------ #
# reproduce command — recreate evaluation from fingerprint (Phase 17)
# ------------------------------------------------------------------ #

@main.command()
@click.argument('fingerprint')
@click.option('--results-dir', default='./beyondbench_results',
              help='Directory containing results with fingerprint')
def reproduce(fingerprint: str, results_dir: str):
    """Recreate an evaluation from its fingerprint.

    Searches for a results file containing the given FINGERPRINT and
    displays the configuration needed to reproduce it.

    Examples:

    \b
    # Find and display config for a fingerprint
    beyondbench reproduce abc123def456 --results-dir ./beyondbench_results
    """
    import glob as _glob

    results_path = Path(results_dir)
    if not results_path.exists():
        click.echo(f"Error: Results directory '{results_dir}' does not exist.", err=True)
        sys.exit(1)

    # Search all JSON files for a matching fingerprint
    json_files = sorted(results_path.rglob("*.json"))
    found = None
    for jf in json_files:
        try:
            with open(jf, encoding='utf-8') as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                fp = data.get('eval_fingerprint', '')
                if fp and fp.startswith(fingerprint):
                    found = (jf, data)
                    break
        except Exception:
            continue

    if found is None:
        click.echo(f"No results file found containing fingerprint '{fingerprint}' in '{results_dir}'.")
        click.echo("Make sure you ran the evaluation with beyondbench >= 0.3.0 (Phase 17).")
        sys.exit(1)

    result_file, result_data = found
    click.echo(f"\nFound results: {result_file}")
    click.echo(f"Fingerprint:   {result_data.get('eval_fingerprint', 'N/A')}\n")

    # Show environment info
    env = result_data.get('environment', {})
    if env:
        click.echo("Environment at time of evaluation:")
        for k, v in env.items():
            click.echo(f"  {k:<30} {v}")
        click.echo()

    # Show evaluation config
    eval_cfg = result_data.get('evaluation_config', {})
    model_info = result_data.get('model_info', {})
    summary = result_data.get('summary', {})

    model_name = model_info.get('model_name', model_info.get('model_id', 'unknown'))
    tasks = eval_cfg.get('tasks', [])
    suite = eval_cfg.get('suite', 'unknown')

    click.echo("Evaluation configuration:")
    click.echo(f"  model_id:     {model_name}")
    click.echo(f"  suite:        {suite}")
    click.echo(f"  tasks:        {', '.join(tasks) if tasks else '(all in suite)'}")
    click.echo(f"  duration:     {summary.get('total_duration', 0):.1f}s")
    click.echo()

    # Suggest a reproduce command
    task_flags = " ".join(f"--tasks {t}" for t in tasks) if tasks else ""
    click.echo("To reproduce this evaluation, run:")
    click.echo(
        f"  beyondbench evaluate"
        f" --model-id {model_name}"
        f" --backend vllm"
        f" --suite {suite}"
        f"{' ' + task_flags if task_flags else ''}"
        f" --seed <original-seed>"
        f" --temperature 0.0"
    )
    click.echo()
    click.echo("Note: Use --seed with the same value as the original run for full reproducibility.")


# ------------------------------------------------------------------ #
# install-completion command — shell completion (Phase 10)
# ------------------------------------------------------------------ #

@main.command(name="install-completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@click.option("--output", "-o", default="", help="Output file path (prints to stdout if omitted)")
def install_completion(shell: str, output: str):
    """Generate and optionally install shell completion scripts.

    Supported shells: bash, zsh, fish.

    Examples:

    \b
    # Print bash completion to stdout
    beyondbench install-completion bash

    \b
    # Install zsh completion
    beyondbench install-completion zsh -o ~/.zsh/completions/_beyondbench

    \b
    # Install fish completion
    beyondbench install-completion fish -o ~/.config/fish/completions/beyondbench.fish
    """
    scripts = {
        "bash": _BASH_COMPLETION,
        "zsh": _ZSH_COMPLETION,
        "fish": _FISH_COMPLETION,
    }
    script = scripts[shell]

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(script)
        click.echo(f"Completion script written to {out_path}")
        if shell == "bash":
            click.echo("Add to ~/.bashrc:  source " + str(out_path))
        elif shell == "zsh":
            click.echo("Add the directory to fpath in ~/.zshrc:  fpath=(" +
                       str(out_path.parent) + " $fpath)")
        elif shell == "fish":
            click.echo(f"Restart fish or run: source {out_path}")
    else:
        click.echo(script)


# Shell completion scripts
_BASH_COMPLETION = """\
# BeyondBench bash completion
# Source this file or add to ~/.bashrc:
#   source <(beyondbench install-completion bash)

_beyondbench_completion() {
    local IFS=$'\\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD \
               _BEYONDBENCH_COMPLETE=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"
        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

_beyondbench_completion_setup() {
    complete -o nosort -F _beyondbench_completion beyondbench
}

_beyondbench_completion_setup;
"""

_ZSH_COMPLETION = """\
#compdef beyondbench
# BeyondBench zsh completion
# Add to ~/.zshrc or place in a directory in $fpath

_beyondbench_completion() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[beyondbench] )) && return 1

    response=("${(@f)$(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) \
              _BEYONDBENCH_COMPLETE=zsh_complete beyondbench)}")

    for type message in ${response}; do
        if [[ "$type" == "dir" ]]; then
            _path_files -/
        elif [[ "$type" == "file" ]]; then
            _path_files -f
        elif [[ "$type" == "plain" ]]; then
            completions+=($message)
        elif [[ "$type" == "multiline" ]]; then
            completions_with_descriptions+=($message)
        fi
    done

    if [ -n "$completions_with_descriptions" ]; then
        _describe -V unsorted completions_with_descriptions -U
    fi

    if [ -n "$completions" ]; then
        compadd -U -V unsorted -a completions
    fi
}

if [[ $zsh_eval_context == *loadautofunc* ]]; then
    _beyondbench_completion "$@"
else
    compdef _beyondbench_completion beyondbench
fi
"""

_FISH_COMPLETION = """\
# BeyondBench fish completion
# Place in ~/.config/fish/completions/beyondbench.fish

function _beyondbench_completion
    set -l response (env _BEYONDBENCH_COMPLETE=fish_complete COMP_WORDS=(commandline -cp) \
                     COMP_CWORD=(commandline -t) beyondbench)

    for completion in $response
        set -l metadata (string split "," $completion)
        if test $metadata[1] = "dir"
            __fish_complete_directories $metadata[2]
        else if test $metadata[1] = "file"
            __fish_complete_path $metadata[2]
        else if test $metadata[1] = "plain"
            echo $metadata[2]
        end
    end
end

complete --no-files --command beyondbench --arguments "(_beyondbench_completion)"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "evaluate"    -d "Run evaluation"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "info"         -d "System information"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "doctor"       -d "Diagnose issues"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "benchmark"    -d "Quick benchmark"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "export"       -d "Export results"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "list-tasks"   -d "List tasks"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "task-info"    -d "Task details"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "compare"      -d "Compare models"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "report"       -d "Generate report"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "dashboard"    -d "Launch dashboard"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "serve"        -d "API server"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "wizard"       -d "Interactive wizard"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "run-config"   -d "Run from config file"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "init"         -d "Create config file"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "install-completion" -d "Install shell completion"
complete -f -c beyondbench -n "__fish_use_subcommand" -a "profile-model"    -d "Profile model output behavior"
"""


# ------------------------------------------------------------------ #
# profile-model command — run calibration and print/save a ModelProfile
# ------------------------------------------------------------------ #
@main.command(name="profile-model")
@click.option("--model-id", required=True,
              help="HuggingFace model path or API model name")
@click.option("--backend", type=click.Choice(["vllm", "transformers", "openai", "gemini", "anthropic"]),
              default="vllm", show_default=True,
              help="Model backend to use for calibration generation")
@click.option("--api-provider", type=click.Choice(["openai", "gemini", "anthropic"]),
              default=None, help="API provider (auto-detected from model-id if omitted)")
@click.option("--api-key", default=None, help="API key (or set OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY)")
@click.option("--cuda-device", default="cuda:0", show_default=True,
              help="CUDA device for local backends")
@click.option("--gpu-memory-utilization", type=float, default=0.45, show_default=True,
              help="vLLM GPU memory utilization ratio")
@click.option("--tensor-parallel-size", type=int, default=1, show_default=True,
              help="vLLM tensor-parallel size")
@click.option("--trust-remote-code", is_flag=True, default=False,
              help="Allow remote code execution when loading the model")
@click.option("--profile-dir", default=None,
              help="Directory to store profiles (default: ~/.beyondbench/profiles/)")
@click.option("--max-tokens", type=int, default=512, show_default=True,
              help="Token budget per calibration prompt")
@click.option("--force", is_flag=True, default=False,
              help="Re-run calibration even if a cached profile exists")
@click.option("--no-save", is_flag=True, default=False,
              help="Do not persist the profile to disk (analysis only)")
@click.pass_obj
def profile_model(
    cli_ctx: "CLIContext",
    model_id: str,
    backend: str,
    api_provider: Optional[str],
    api_key: Optional[str],
    cuda_device: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
    profile_dir: Optional[str],
    max_tokens: int,
    force: bool,
    no_save: bool,
):
    """Profile a model's output patterns for adaptive parsing.

    Runs a short calibration prompt set against the model and measures how
    it formats answers (boxed, explicit, code-block, CoT), language mix,
    and average response length. The resulting ``ModelProfile`` is saved
    to ``~/.beyondbench/profiles/`` (override with ``--profile-dir``) and
    used by the UnifiedParser to adapt parsing strategy order per model.

    Examples:

    \b
      # Profile a local HF model with vLLM
      beyondbench profile-model --model-id Qwen/Qwen2.5-3B-Instruct

    \b
      # Re-profile ignoring the cache, analysis-only
      beyondbench profile-model --model-id gpt-4o-mini --api-provider openai --force --no-save

    \b
      # Machine-readable JSON output
      beyondbench --json profile-model --model-id Qwen/Qwen2.5-1.5B-Instruct
    """
    from ..utils.model_profiler import ModelProfiler, ModelProfile
    from dataclasses import asdict

    json_mode = bool(cli_ctx and cli_ctx.json_mode)
    logger = get_logger("CLI.profile-model")

    # ── 1. Check cache unless --force ───────────────────────────────
    if not force:
        cached = ModelProfiler.load(model_id, profile_dir)
        if cached is not None:
            if json_mode:
                click.echo(json.dumps(
                    {"cached": True, "profile": asdict(cached)},
                    indent=2, default=str,
                ))
                return
            _print_profile_rich(cached, cached_from_disk=True, cli_ctx=cli_ctx)
            return

    # ── 2. Build model handler ──────────────────────────────────────
    try:
        from ..models.model_handler import ModelHandler
    except Exception as exc:
        _handle_cli_error(exc, "model-handler import")
        return

    # Auto-detect API provider from model-id if not given explicitly
    if api_provider is None and backend not in {"vllm", "transformers"}:
        api_provider = _auto_detect_api_provider(model_id)
    elif api_provider is None:
        detected = _auto_detect_api_provider(model_id)
        if detected is not None:
            # Switch to API backend implicitly
            api_provider = detected
            backend = detected

    # Resolve API key from env if needed
    if api_provider and not api_key:
        api_key = _auto_detect_api_key(api_provider)

    handler_kwargs: Dict[str, Any] = {
        "model_id": model_id,
        "backend": backend if backend in {"vllm", "transformers"} else None,
        "api_provider": api_provider,
        "api_key": api_key,
        "cuda_device": cuda_device,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    # Drop Nones so ModelHandler uses its own defaults.
    handler_kwargs = {k: v for k, v in handler_kwargs.items() if v is not None}

    if not json_mode:
        cli_ctx and cli_ctx.echo(
            f"[profile-model] Calibrating {model_id} "
            f"(backend={backend}, provider={api_provider or '-'}) ..."
        )

    try:
        handler = ModelHandler(**handler_kwargs)
    except Exception as exc:
        logger.error(f"Failed to initialize model handler: {exc}")
        if json_mode:
            click.echo(json.dumps({"error": str(exc), "model_id": model_id}, indent=2))
            sys.exit(1)
        _handle_cli_error(exc, "model handler")
        return

    # ── 3. Run calibration ──────────────────────────────────────────
    try:
        profiler = ModelProfiler(handler, model_id=model_id)
        profile = profiler.profile(max_tokens=max_tokens)
    except Exception as exc:
        logger.error(f"Calibration run failed: {exc}")
        if json_mode:
            click.echo(json.dumps({"error": str(exc), "model_id": model_id}, indent=2))
            sys.exit(1)
        _handle_cli_error(exc, "calibration")
        return

    # ── 4. Persist ──────────────────────────────────────────────────
    saved_path: Optional[Path] = None
    if not no_save:
        try:
            saved_path = profiler.save(profile, profile_dir)
        except Exception as exc:
            logger.warning(f"Could not save profile: {exc}")

    # ── 5. Report ───────────────────────────────────────────────────
    if json_mode:
        payload = {
            "cached": False,
            "saved_path": str(saved_path) if saved_path else None,
            "profile": asdict(profile),
        }
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    _print_profile_rich(profile, saved_path=saved_path, cli_ctx=cli_ctx)


def _print_profile_rich(
    profile,
    *,
    saved_path: Optional[Path] = None,
    cached_from_disk: bool = False,
    cli_ctx: Optional["CLIContext"] = None,
) -> None:
    """Pretty-print a ModelProfile using Rich (falls back to plain)."""
    if cli_ctx is None:
        cli_ctx = _get_cli_ctx()
    if cli_ctx.quiet:
        return
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box as rbox
        console = Console()
    except ImportError:
        # Plain fallback
        click.echo(f"\n=== Model Profile: {profile.model_id} ===")
        click.echo(f"  Boxed rate:          {profile.boxed_rate:.1%}")
        click.echo(f"  Explicit stmt rate:  {profile.explicit_rate:.1%}")
        click.echo(f"  Code block rate:     {profile.code_block_rate:.1%}")
        click.echo(f"  CoT rate:            {profile.cot_rate:.1%}")
        click.echo(f"  Non-Latin rate:      {profile.non_latin_rate:.1%}")
        click.echo(f"  Avg response length: {profile.avg_response_length:.0f} chars")
        click.echo(f"  Recommended order:   {profile.recommended_strategies}")
        if getattr(profile, "quirks", None):
            click.echo("  Quirks:")
            for q in profile.quirks:
                click.echo(f"    - {q}")
        if saved_path:
            click.echo(f"  Saved to: {saved_path}")
        click.echo()
        return

    table = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    table.add_column("Metric", style="bold cyan", min_width=22)
    table.add_column("Value", style="white")
    table.add_row("Model ID", profile.model_id)
    table.add_row("Boxed rate", f"{profile.boxed_rate:.1%}")
    table.add_row("Explicit-statement rate", f"{profile.explicit_rate:.1%}")
    table.add_row("Code-block rate", f"{profile.code_block_rate:.1%}")
    table.add_row("Chain-of-thought rate", f"{profile.cot_rate:.1%}")
    table.add_row("Non-Latin rate", f"{profile.non_latin_rate:.1%}")
    table.add_row("Avg response length", f"{profile.avg_response_length:.0f} chars")
    table.add_row("Avg token estimate", f"{profile.avg_token_estimate:.0f} tokens")
    table.add_row("Calibration samples", str(profile.num_calibration_samples))
    table.add_row(
        "Recommended strategies",
        ", ".join(profile.recommended_strategies) if profile.recommended_strategies else "—",
    )
    if getattr(profile, "quirks", None):
        table.add_row("Quirks", "\n".join(f"• {q}" for q in profile.quirks))
    if saved_path:
        table.add_row("Saved to", str(saved_path))
    elif cached_from_disk:
        table.add_row("Source", "[yellow]cached on disk[/yellow]")

    border = "yellow" if cached_from_disk else "cyan"
    title = f"[bold]Model Profile — {profile.model_id}[/bold]"
    console.print(Panel(table, title=title, border_style=border))


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
# dashboard command (Phase 6)
# ------------------------------------------------------------------ #
@main.command()
@click.option('--port', default=7860, type=int, show_default=True, help='Port to serve dashboard on')
@click.option('--results-dir', default='./beyondbench_results', show_default=True,
              help='Directory with past evaluation results to display')
@click.option('--compare', 'compare_files', default='', help='Comma-separated paths to two result files for overlay comparison')
@click.option('--share', is_flag=True, help='Create a public Gradio share link')
def dashboard(port: int, results_dir: str, compare_files: str, share: bool):
    """Launch the BeyondBench observability dashboard.

    Standalone viewer for past results:

    \b
      beyondbench dashboard --results-dir ./beyondbench_results

    Compare two runs:

    \b
      beyondbench dashboard --compare run_a/final_results.json,run_b/final_results.json

    Requires the [dashboard] extra: pip install beyondbench[dashboard]
    """
    try:
        import gradio  # noqa: F401
    except ImportError:
        click.echo(
            "Error: The 'dashboard' extra is required.\n"
            "Install it with:\n\n"
            "  pip install beyondbench[dashboard]\n"
        )
        sys.exit(1)

    try:
        from ..dashboard import launch_dashboard
    except ImportError as exc:
        click.echo(f"Error: Could not import dashboard module: {exc}")
        sys.exit(1)

    click.echo(f"Starting BeyondBench dashboard on http://0.0.0.0:{port}")
    click.echo(f"Viewing results from: {results_dir}")

    app = launch_dashboard(
        data_bridge=None,
        port=port,
        results_dir=results_dir,
        share=share,
    )

    # If compare mode was requested, pre-fill the compare tab (just notify user)
    if compare_files:
        paths = [p.strip() for p in compare_files.split(',') if p.strip()]
        if len(paths) >= 2:
            click.echo(f"Compare mode: {paths[0]} vs {paths[1]}")
            click.echo("Navigate to the 'Compare Results' tab and enter the paths above.")

    click.echo("Press Ctrl+C to stop the dashboard.")
    try:
        # Block until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")


# ------------------------------------------------------------------ #
# report command (Phase 8)
# ------------------------------------------------------------------ #
@main.command()
@click.option('--results-dir', default='./beyondbench_results', show_default=True,
              help='Directory containing final_results.json (searched recursively)')
@click.option('--results-file', default='', help='Direct path to a final_results.json file')
@click.option('--format', 'report_format',
              type=click.Choice(['html', 'markdown', 'md', 'pdf', 'latex', 'all']),
              default='html', show_default=True, help='Report output format')
@click.option('--output', default='', help='Output file path (auto-named if omitted)')
@click.option('--compare', 'compare_paths', default='',
              help='Comma-separated paths to additional final_results.json files for model comparison')
@click.option('--charts-dir', default='', help='Directory to save individual chart files (html by default)')
@click.option('--chart-format', type=click.Choice(['html', 'png', 'svg']), default='html',
              show_default=True, help='Format for individual chart files')
def report(
    results_dir: str,
    results_file: str,
    report_format: str,
    output: str,
    compare_paths: str,
    charts_dir: str,
    chart_format: str,
):
    """Generate a comprehensive evaluation report from past results.

    Reads a final_results.json produced by ``beyondbench evaluate`` and outputs
    HTML (with interactive charts), Markdown, PDF, or LaTeX.

    Examples:

    \b
    # Generate HTML report from the default results directory
    beyondbench report --results-dir ./beyondbench_results --format html

    \b
    # Generate Markdown report from a specific file
    beyondbench report --results-file ./results/final_results.json --format markdown

    \b
    # Generate all formats and save charts
    beyondbench report --results-dir ./results --format all --charts-dir ./charts

    \b
    # Compare two runs
    beyondbench report --results-dir ./run_a \\
        --compare ./run_b/final_results.json --format html
    """
    from ..utils.logging_utils import setup_logging, get_logger
    from ..utils.report_generator import ReportGenerator

    setup_logging(log_level="INFO", enable_console_logging=True, enable_file_logging=False)
    log = get_logger("CLI.report")

    # ------------------------------------------------------------------
    # Locate main results file
    # ------------------------------------------------------------------
    results_path: Optional[Path] = None

    if results_file:
        results_path = Path(results_file)
    else:
        rdir = Path(results_dir)
        if not rdir.exists():
            click.echo(f"Error: Results directory does not exist: {results_dir}", err=True)
            sys.exit(1)
        # Search for final_results.json
        candidates = sorted(rdir.rglob("final_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            click.echo(f"Error: No final_results.json found in {results_dir}", err=True)
            sys.exit(1)
        results_path = candidates[0]
        if len(candidates) > 1:
            log.info(f"Multiple result files found; using most recent: {results_path}")

    if not results_path.exists():
        click.echo(f"Error: Results file not found: {results_path}", err=True)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    log.info(f"Loaded results from {results_path}")

    # ------------------------------------------------------------------
    # Load comparison results
    # ------------------------------------------------------------------
    compare_results: List[Dict[str, Any]] = []
    if compare_paths:
        for cp in compare_paths.split(","):
            cp = cp.strip()
            if not cp:
                continue
            cpath = Path(cp)
            if cpath.is_dir():
                # Find final_results.json inside dir
                candidates = sorted(cpath.rglob("final_results.json"))
                if candidates:
                    cpath = candidates[0]
            if cpath.exists():
                with open(cpath, "r", encoding="utf-8") as f:
                    compare_results.append(json.load(f))
                log.info(f"Loaded comparison results from {cpath}")
            else:
                log.warning(f"Comparison file not found: {cpath}")

    # ------------------------------------------------------------------
    # Determine output directory and filenames
    # ------------------------------------------------------------------
    out_dir = Path(output).parent if output else results_path.parent
    generator = ReportGenerator(output_dir=str(out_dir))

    # ------------------------------------------------------------------
    # Generate reports
    # ------------------------------------------------------------------
    formats_to_generate = (
        ["html", "markdown", "latex", "pdf"] if report_format == "all" else [report_format]
    )

    generated: Dict[str, Optional[str]] = {}
    for fmt in formats_to_generate:
        fmt_norm = "markdown" if fmt == "md" else fmt
        # Determine output filename
        if output and report_format != "all":
            out_file = Path(output).name
        else:
            ext = {"html": "html", "markdown": "md", "latex": "tex", "pdf": "pdf"}.get(fmt_norm, fmt_norm)
            out_file = f"report.{ext}"

        if fmt_norm == "html":
            path = generator.save_html_report(
                results, filename=out_file, compare_results=compare_results or None
            )
        elif fmt_norm == "markdown":
            path = generator.save_markdown_report(
                results, filename=out_file, compare_results=compare_results or None
            )
        elif fmt_norm == "latex":
            path = generator.save_latex_report(results, filename=out_file)
        elif fmt_norm == "pdf":
            path = generator.save_pdf_report(results, filename=out_file)
        else:
            log.warning(f"Unknown format: {fmt}")
            continue

        generated[fmt_norm] = path
        if path:
            click.echo(f"  {fmt_norm.upper()}: {path}")

    # ------------------------------------------------------------------
    # Generate charts (optional)
    # ------------------------------------------------------------------
    if charts_dir:
        try:
            from ..utils.visualizer import Visualizer
            viz = Visualizer()
            charts_output = Path(charts_dir)
            charts_output.mkdir(parents=True, exist_ok=True)

            figs = {
                "accuracy_by_task": viz.plot_accuracy_by_task(results),
                "accuracy_by_suite": viz.plot_accuracy_by_suite(results),
            }
            if compare_results:
                results_map = {results.get("model_id", "Model A"): results}
                for i, cr in enumerate(compare_results):
                    results_map[cr.get("model_id", f"Model {i+2}")] = cr
                figs["model_comparison"] = viz.plot_model_comparison(results_map)
                figs["radar_chart"] = viz.plot_radar_chart(results_map)

            saved = viz.save_all(figs, charts_output, fmt=chart_format)
            for name, path in saved.items():
                if path:
                    click.echo(f"  Chart [{name}]: {path}")
        except Exception as e:
            log.warning(f"Chart generation failed: {e}")

    click.echo(f"\nReport generation complete. Files in: {out_dir.resolve()}")


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
    """Print stylized banner using Rich if available, else plain ASCII."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.align import Align
        from rich import box as rbox

        _banner_text = (
            "[bold cyan]\n"
            "██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗ \n"
            "██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗\n"
            "██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║\n"
            "██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║\n"
            "██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝\n"
            "╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝ \n"
            "\n"
            "██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗\n"
            "██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║\n"
            "██████╔╝█████╗  ██╔██╗ ██║██║     ███████║\n"
            "██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║\n"
            "██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║\n"
            "╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝\n"
            "[/bold cyan]\n"
            f"[dim]Contamination-Resistant LLM Reasoning Evaluation — v{_VERSION}[/dim]"
        )
        console = Console()
        console.print(Panel(Align.center(_banner_text), box=rbox.DOUBLE,
                            border_style="cyan", padding=(0, 2)))
    except ImportError:
        banner = (
            f"\n"
            f" ____                            _ ____                  _\n"
            f"| __ )  ___ _   _  ___  _ __   __| | __ )  ___ _ __   ___| |__\n"
            f"|  _ \\ / _ \\ | | |/ _ \\| '_ \\ / _` |  _ \\ / _ \\ '_ \\ / __| '_ \\\n"
            f"| |_) |  __/ |_| | (_) | | | | (_| | |_) |  __/ | | | (__| | | |\n"
            f"|____/ \\___|\\__, |\\___/|_| |_|\\__,_|____/ \\___|_| |_|\\___|_| |_|\n"
            f"            |___/\n"
            f"\nBeyondBench Evaluation Framework v{_VERSION}\n"
        )
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
    """Validate CLI arguments and prepare configuration.

    Priority: CLI args > BEYONDBENCH_* env vars > defaults
    """
    # Apply env var overrides for fields that were not explicitly set via CLI.
    # Click sets defaults automatically, so we detect "env-var candidates" by
    # checking a set of BEYONDBENCH_* variables and applying them only when
    # the CLI value matches the Click default (i.e. the user didn't override it).
    _env_defaults = {
        'model_id':              ('BEYONDBENCH_MODEL_ID', str, None),
        'backend':               ('BEYONDBENCH_BACKEND', str, None),
        'api_key':               ('BEYONDBENCH_API_KEY', str, None),
        'api_provider':          ('BEYONDBENCH_API_PROVIDER', str, None),
        'cuda_device':           ('BEYONDBENCH_CUDA_DEVICE', str, 'cuda:0'),
        'gpu_memory_utilization':('BEYONDBENCH_GPU_MEMORY_UTIL', float, 0.96),
        'output_dir':            ('BEYONDBENCH_OUTPUT_DIR', str, './beyondbench_results'),
        'log_level':             ('BEYONDBENCH_LOG_LEVEL', str, 'INFO'),
        'datapoints':            ('BEYONDBENCH_DATAPOINTS', int, 100),
        'suite':                 ('BEYONDBENCH_SUITE', str, 'all'),
        'seed':                  ('BEYONDBENCH_SEED', int, None),
        'temperature':           ('BEYONDBENCH_TEMPERATURE', float, 0.7),
        'max_tokens':            ('BEYONDBENCH_MAX_TOKENS', int, 32768),
    }
    for field, (env_var, cast, click_default) in _env_defaults.items():
        raw = os.environ.get(env_var)
        if raw is None:
            continue
        current = kwargs.get(field)
        # Only apply if CLI value matches the default (user didn't explicitly set it)
        if current == click_default or current is None:
            try:
                kwargs[field] = cast(raw)
            except (ValueError, TypeError):
                pass

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
        'max_tokens': kwargs['max_tokens'],
        'seed': kwargs.get('seed'),
        'prompt_style': kwargs.get('prompt_style', None),
        'contamination_resistance': kwargs.get('contamination_resistance', 'off'),
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


def _run_preflight_validation(
    kwargs: Dict[str, Any],
    *,
    skip: bool = False,
    warn_only: bool = False,
    logger: Any = None,
) -> None:
    """Run Phase 9.2.1 pre-flight validation on a flat CLI kwargs dict.

    Translates CLI kwargs back into the nested config shape that
    :class:`ModelValidator` expects, runs every enabled check, and logs the
    findings. When ``warn_only`` is False and the validator reports any
    error-severity issues, aborts via :func:`click.BadParameter` so the
    user sees a clean error message and a non-zero exit code.

    ``skip=True`` short-circuits the whole thing — useful for CI runs where
    network egress is blocked or GPU info is unreliable.
    """
    if skip:
        if logger is not None:
            logger.info("Pre-flight validation skipped (--skip-validation).")
        return

    try:
        from ..configs.model_validator import ModelValidator
    except ImportError:
        # Validator isn't available for some reason — don't block the run.
        if logger is not None:
            logger.warning("Pre-flight validator unavailable; skipping checks.")
        return

    nested = {
        "model": {
            "model_id":               kwargs.get("model_id"),
            "backend":                kwargs.get("backend"),
            "api_provider":           kwargs.get("api_provider"),
            "api_key":                kwargs.get("api_key"),
            "cuda_device":            kwargs.get("cuda_device"),
            "tensor_parallel_size":   kwargs.get("tensor_parallel_size"),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization"),
        },
        "evaluation": {
            "temperature": kwargs.get("temperature"),
            "max_tokens":  kwargs.get("max_tokens"),
            "datapoints":  kwargs.get("datapoints"),
            "suite":       kwargs.get("suite"),
        },
    }

    try:
        result = ModelValidator().validate(nested)
    except Exception as e:  # pragma: no cover — the validator is defensive
        if logger is not None:
            logger.warning(f"Pre-flight validation raised unexpectedly: {e}")
        return

    # Emit every issue to the logger at the appropriate level
    _level_for_severity = {"error": "error", "warning": "warning", "info": "info"}
    if logger is not None:
        if not result.issues:
            logger.info("Pre-flight validation: all checks passed.")
        for issue in result.issues:
            level = _level_for_severity.get(issue.severity, "info")
            msg = f"[pre-flight] [{issue.check}] {issue.message}"
            if issue.suggestion:
                msg += f"  -> {issue.suggestion}"
            getattr(logger, level)(msg)

    if result.suggested_settings and logger is not None:
        hints = {k: v for k, v in result.suggested_settings.items() if k != "note"}
        if hints:
            logger.info(f"[pre-flight] Suggested settings for this model: {hints}")

    # Fail fast on errors unless the user asked for warn-only mode
    if not result.ok and not warn_only:
        bullets = "\n".join(
            f"  - [{i.check}] {i.message}"
            + (f"\n      hint: {i.suggestion}" if i.suggestion else "")
            for i in result.errors
        )
        raise click.BadParameter(
            "Pre-flight model validation failed:\n"
            f"{bullets}\n\n"
            "Re-run with --skip-validation to bypass, or --validation-warn-only "
            "to downgrade these to warnings."
        )


def _wire_bridge_to_engine(engine, data_bridge, config: Dict[str, Any]) -> None:
    """Monkey-patch EvaluationEngine.run_evaluation to emit DataBridge events.

    This is a non-invasive approach: we wrap the existing run_evaluation method
    so that existing code paths continue to work unchanged.
    """
    from ..core.task_registry import TaskRegistry

    # Determine task list so we can tell the bridge the total count up-front
    registry = TaskRegistry()
    suite = config.get('suite', 'all')
    tasks = config.get('tasks') or []
    if tasks:
        task_list = list(tasks)
    else:
        task_list = registry.get_tasks_for_suite(suite)

    model_info = {}
    if hasattr(engine, 'model_handler') and hasattr(engine.model_handler, 'get_model_info'):
        try:
            model_info = engine.model_handler.get_model_info()
        except Exception:
            pass

    data_bridge.notify_evaluation_started(
        total_tasks=len(task_list),
        model_info=model_info,
    )

    original_run = engine.run_evaluation

    def _patched_run_evaluation(suite='all', tasks=None, **kwargs):
        # Resolve the actual task list the engine will run
        _tasks = tasks or registry.get_tasks_for_suite(suite)

        # Wrap the inner per-task loop by patching _aggregate_results to intercept results
        original_aggregate = engine._aggregate_results

        def _patched_aggregate(task_results, overall_stats, total_duration):
            # Emit per-task events for each completed task
            for task_name, result in task_results.items():
                acc = _extract_accuracy(result)
                data_bridge.notify_task_completed(
                    task_name=task_name,
                    accuracy=acc,
                    suite='easy',  # default; bridge groups by suite
                )
            return original_aggregate(task_results, overall_stats, total_duration)

        engine._aggregate_results = _patched_aggregate
        try:
            return original_run(suite=suite, tasks=tasks, **kwargs)
        finally:
            engine._aggregate_results = original_aggregate

    engine.run_evaluation = _patched_run_evaluation


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

        msg = str(exc)
        msg_lower = msg.lower()

        if error_type == "import":
            suggestion = ""
            if "openai" in msg_lower:
                suggestion = "pip install beyondbench[openai]"
            elif "google" in msg_lower or "genai" in msg_lower:
                suggestion = "pip install beyondbench[gemini]"
            elif "anthropic" in msg_lower:
                suggestion = "pip install beyondbench[anthropic]"
            elif "vllm" in msg_lower:
                suggestion = (
                    "pip install beyondbench[vllm]\n\n"
                    "  If vLLM fails to start, try:\n"
                    "    beyondbench evaluate ... --backend transformers\n"
                    "  or diagnose with:  beyondbench doctor"
                )
            elif "fastapi" in msg_lower or "uvicorn" in msg_lower:
                suggestion = "pip install beyondbench[serve]"
            elif "cuda" in msg_lower or "no cuda gpus" in msg_lower:
                suggestion = (
                    "No CUDA GPUs detected. Options:\n"
                    "  1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads\n"
                    "  2. Use CPU (slow): --backend transformers\n"
                    "  3. Use a cloud API: --api-provider openai\n"
                    "  Diagnose with:  beyondbench doctor"
                )

            lines = [f"[bold red]Import Error:[/bold red] {exc}"]
            if suggestion:
                lines.append(f"\n[bold]Suggestion:[/bold]\n  {suggestion}")
            console.print(Panel("\n".join(lines), title="Missing Dependency", border_style="red"))

        elif error_type == "configuration":
            lines = [f"[bold red]Configuration Error:[/bold red] {msg}"]
            if "api_key" in msg_lower or "api-key" in msg_lower:
                lines.append(
                    "\n[bold]Suggestion:[/bold] Set your API key via environment variable:\n"
                    "  export OPENAI_API_KEY=sk-...\n"
                    "  export GEMINI_API_KEY=...\n"
                    "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                    "Or pass --api-key on the command line."
                )
            elif "model" in msg_lower and "not found" in msg_lower:
                lines.append(
                    "\n[bold]Suggestion:[/bold] Check the model ID on HuggingFace:\n"
                    "  https://huggingface.co/models\n"
                    "  Common models: Qwen/Qwen2.5-1.5B-Instruct, meta-llama/Llama-3.2-3B-Instruct"
                )
            console.print(Panel("\n".join(lines), title="Configuration Error", border_style="red"))

        else:
            # Generic error — try to give actionable hints
            lines = [f"[bold red]Error:[/bold red] {exc}"]
            if "cuda" in msg_lower and "out of memory" in msg_lower:
                lines.append(
                    "\n[bold]GPU OOM.[/bold] Try:\n"
                    "  --gpu-memory-utilization 0.4\n"
                    "  --tensor-parallel-size 2  (use multiple GPUs)\n"
                    "  Use a smaller model"
                )
            elif "vllm" in msg_lower and ("failed" in msg_lower or "error" in msg_lower):
                lines.append(
                    "\n[bold]vLLM failed.[/bold] Try:\n"
                    "  beyondbench evaluate ... --backend transformers\n"
                    "  pip install --upgrade vllm\n"
                    "  beyondbench doctor"
                )
            elif "not found" in msg_lower and ("model" in msg_lower or "repo" in msg_lower):
                lines.append(
                    "\n[bold]Model not found.[/bold] Check spelling or try:\n"
                    "  Qwen/Qwen2.5-1.5B-Instruct\n"
                    "  meta-llama/Llama-3.2-3B-Instruct\n"
                    "  Ensure you have access to gated models (HF_TOKEN env var)"
                )
            console.print(Panel("\n".join(lines), title="Error", border_style="red"))

    except ImportError:
        click.echo(f"Error ({error_type}): {exc}", err=True)

    sys.exit(1)


_DIFFICULTY_STYLE = {
    "easy": ("Easy", "green"),
    "medium": ("Medium", "yellow"),
    "hard": ("Hard", "red"),
}


def print_tasks_table(tasks: Dict[str, List[str]]):
    """Print tasks as Rich tables with description + difficulty columns.

    Falls back to plain click.echo output when Rich is unavailable so the CLI
    remains functional on minimal installs.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box as rbox
    except ImportError:
        click.echo("\nAvailable Tasks by Suite\n")
        for suite_name, task_list in tasks.items():
            click.echo(click.style(
                f"{suite_name.upper()} SUITE ({len(task_list)} tasks)",
                fg='green', bold=True,
            ))
            click.echo("-" * 50)
            for i, task in enumerate(task_list, 1):
                click.echo(f"  {i:2d}. {task}")
            click.echo()
        return

    # Lazy import the registry so we can fetch per-task descriptions.
    try:
        from ..core.task_registry import TaskRegistry
        registry = TaskRegistry()
    except Exception:
        registry = None

    console = Console()
    console.print()
    console.print("[bold cyan]Available Tasks by Suite[/bold cyan]")

    for suite_name, task_list in tasks.items():
        diff_label, diff_color = _DIFFICULTY_STYLE.get(
            suite_name.lower(), (suite_name.title(), "cyan"),
        )
        title = (
            f"[bold {diff_color}]{suite_name.upper()} SUITE[/bold {diff_color}] "
            f"([dim]{len(task_list)} tasks[/dim])"
        )

        table = Table(
            title=title,
            box=rbox.SIMPLE_HEAD,
            header_style=f"bold {diff_color}",
            show_lines=False,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Task", style="bold white", min_width=24, overflow="fold")
        table.add_column("Difficulty", justify="center", width=12)
        table.add_column("Description", style="white", overflow="fold")

        for i, task in enumerate(task_list, 1):
            description = ""
            if registry is not None:
                try:
                    info = registry.get_task_info(task) or {}
                    description = str(info.get("description") or "").strip()
                except Exception:
                    description = ""
            if not description:
                description = f"{task.replace('_', ' ').title()} task"

            table.add_row(
                str(i),
                task,
                f"[{diff_color}]{diff_label}[/{diff_color}]",
                description,
            )

        console.print(table)
        console.print()


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


# ---------------------------------------------------------------------------
# Baseline management commands (Phase 19)
# ---------------------------------------------------------------------------


@main.group()
def baseline():
    """Manage evaluation baselines for regression testing.

    Baselines capture per-task accuracy and parser metrics from a completed
    evaluation run so that future runs can be automatically checked for
    regressions.

    \b
    Typical workflow:
      1. Run an evaluation and save a baseline:
           beyondbench baseline save --results-dir ./results --name v0.2.0-qwen
      2. Run a new evaluation and compare:
           beyondbench baseline compare --results-dir ./new_results --baseline v0.2.0-qwen
    """
    pass


@baseline.command("save")
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the results directory containing final_results.json",
)
@click.option(
    "--name",
    required=True,
    help="Baseline name (e.g. v0.2.0-qwen-1.5b). Used as the filename stem.",
)
def baseline_save(results_dir: str, name: str) -> None:
    """Save a new baseline from an evaluation results directory.

    Reads final_results.json from RESULTS_DIR and stores per-task accuracy
    and parser success rates as a named baseline that can later be used for
    regression comparison.
    """
    from ..eval.baseline import BaselineManager

    try:
        mgr = BaselineManager()
        out_path = mgr.save(results_dir, name)
        click.echo(f"Baseline '{name}' saved to: {out_path}")
        # Echo the tasks that were captured
        baseline_data = mgr.load(name)
        task_count = len(baseline_data.get("task_baselines", {}))
        click.echo(f"Captured metrics for {task_count} task(s).")
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    except ValueError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:
        raise click.ClickException(f"Unexpected error saving baseline: {exc}")


@baseline.command("compare")
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the current results directory containing final_results.json",
)
@click.option(
    "--baseline",
    "baseline_name",
    required=True,
    help="Baseline name to compare against (as given to 'baseline save')",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    default=False,
    help="Exit with code 1 if any regression is detected",
)
@click.option(
    "--accuracy-threshold",
    type=float,
    default=0.10,
    show_default=True,
    help="Maximum allowed accuracy drop before a regression is declared",
)
@click.option(
    "--parser-threshold",
    type=float,
    default=0.05,
    show_default=True,
    help="Maximum allowed parser success-rate drop before a regression is declared",
)
def baseline_compare(
    results_dir: str,
    baseline_name: str,
    fail_on_regression: bool,
    accuracy_threshold: float,
    parser_threshold: float,
) -> None:
    """Compare current results against a stored baseline.

    Prints a detailed per-task breakdown showing which tasks improved,
    regressed, or stayed the same relative to the saved baseline.
    """
    from ..eval.regression import RegressionChecker

    try:
        checker = RegressionChecker()
        exit_code = checker.run(
            current_results_dir=results_dir,
            baseline_name=baseline_name,
            accuracy_threshold=accuracy_threshold,
            parser_threshold=parser_threshold,
            print_report=True,
        )
        if exit_code != 0 and fail_on_regression:
            raise SystemExit(1)
        elif exit_code != 0:
            click.echo("Error: regression or baseline not found. Use --fail-on-regression to fail CI.")
    except SystemExit:
        raise
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    except ValueError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:
        raise click.ClickException(f"Unexpected error during comparison: {exc}")


@baseline.command("list")
def baseline_list() -> None:
    """List all saved baselines."""
    from ..eval.baseline import BaselineManager

    mgr = BaselineManager()
    names = mgr.list_baselines()
    if not names:
        click.echo("No baselines saved yet. Use 'beyondbench baseline save' to create one.")
        return
    click.echo(f"Saved baselines ({len(names)}):")
    for name in names:
        try:
            data = mgr.load(name)
            ts = data.get("timestamp", "unknown")
            tasks = len(data.get("task_baselines", {}))
            model = data.get("model_info", {}).get("model_name", "unknown")
            click.echo(f"  {name:<40s}  {tasks} tasks  model={model}  saved={ts}")
        except Exception:
            click.echo(f"  {name}")


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

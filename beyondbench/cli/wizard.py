"""
BeyondBench Interactive CLI Wizard

A beautiful, interactive command-line interface for running BeyondBench evaluations
with rich spinners, progress bars, and animations.
"""

import sys
import time
import os
from typing import Optional, List, Dict, Any
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    _VERSION = _pkg_version("beyondbench")
except PackageNotFoundError:
    try:
        from beyondbench import __version__ as _VERSION
    except ImportError:
        _VERSION = "unknown"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.live import Live
    from rich.layout import Layout
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not installed. Install with: pip install rich")

# Initialize console
console = Console()

# ASCII Art Banner
BANNER = """
[bold cyan]
██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗
██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗
██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║
██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║
██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝
╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝

██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
[/bold cyan]
[dim]Contamination-Resistant LLM Reasoning Evaluation[/dim]
[dim]Version """ + _VERSION + """[/dim]
"""


def print_banner():
    """Print the BeyondBench banner."""
    console.print(Panel(
        Align.center(BANNER),
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2)
    ))


def print_menu():
    """Print the main menu."""
    menu_table = Table(
        show_header=False,
        box=box.ROUNDED,
        border_style="blue",
        padding=(0, 2)
    )
    menu_table.add_column("Option", style="cyan bold", width=6)
    menu_table.add_column("Description", style="white")

    menu_table.add_row("1", "🚀 Run Evaluation - Evaluate a model on BeyondBench tasks")
    menu_table.add_row("2", "📋 List Tasks - Show available evaluation tasks")
    menu_table.add_row("3", "⚙️  Configure - Set up API keys and preferences")
    menu_table.add_row("4", "📊 View Results - Browse previous evaluation results")
    menu_table.add_row("5", "📖 Documentation - View help and documentation")
    menu_table.add_row("6", "🚪 Exit - Exit the wizard")

    console.print("\n")
    console.print(Panel(menu_table, title="[bold white]Main Menu[/bold white]", border_style="blue"))


def select_model_type() -> str:
    """Interactive model type selection."""
    console.print("\n[bold cyan]Select Model Provider:[/bold cyan]\n")

    options = [
        ("1", "OpenAI", "GPT-4o, GPT-4o-mini, GPT-5, GPT-5-mini", "green"),
        ("2", "Gemini", "Gemini 2.5 Pro, Gemini 2.5 Flash", "yellow"),
        ("3", "Anthropic", "Claude Sonnet 4, Claude Opus 4", "magenta"),
        ("4", "Local", "vLLM or Transformers (Llama, Qwen, Mistral, etc.)", "blue"),
    ]

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("", width=4)
    table.add_column("Provider", width=12)
    table.add_column("Models")

    for opt, provider, models, color in options:
        table.add_row(f"[{color}]{opt}[/{color}]", f"[bold {color}]{provider}[/bold {color}]", f"[dim]{models}[/dim]")

    console.print(table)

    choice = Prompt.ask("\n[bold]Enter your choice[/bold]", choices=["1", "2", "3", "4"], default="1")

    providers = {"1": "openai", "2": "gemini", "3": "anthropic", "4": "local"}
    return providers[choice]


def select_model_id(provider: str) -> str:
    """Interactive model ID selection based on provider."""
    console.print(f"\n[bold cyan]Select {provider.title()} Model:[/bold cyan]\n")

    model_options = {
        "openai": [
            ("1", "gpt-4o", "Latest GPT-4o model"),
            ("2", "gpt-4o-mini", "Smaller, faster GPT-4o"),
            ("3", "gpt-5", "GPT-5 with reasoning capabilities"),
            ("4", "gpt-5-mini", "Smaller GPT-5"),
            ("5", "gpt-5-nano", "Smallest GPT-5"),
            ("6", "custom", "Enter custom model ID"),
        ],
        "gemini": [
            ("1", "gemini-2.5-pro", "Most capable Gemini model"),
            ("2", "gemini-2.5-flash", "Fast Gemini model"),
            ("3", "gemini-2.5-flash-lite", "Lightweight Gemini"),
            ("4", "custom", "Enter custom model ID"),
        ],
        "anthropic": [
            ("1", "claude-sonnet-4-20250514", "Claude Sonnet 4"),
            ("2", "claude-opus-4-20250514", "Claude Opus 4 (Most capable)"),
            ("3", "custom", "Enter custom model ID"),
        ],
        "local": [
            ("1", "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct"),
            ("2", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct"),
            ("3", "Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B Instruct"),
            ("4", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct"),
            ("5", "custom", "Enter custom model ID"),
        ],
    }

    options = model_options.get(provider, model_options["local"])

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("", width=4)
    table.add_column("Model", width=40)
    table.add_column("Description")

    for opt, model, desc in options:
        table.add_row(f"[cyan]{opt}[/cyan]", f"[bold]{model}[/bold]", f"[dim]{desc}[/dim]")

    console.print(table)

    choices = [str(i) for i in range(1, len(options) + 1)]
    choice = Prompt.ask("\n[bold]Enter your choice[/bold]", choices=choices, default="1")

    selected = options[int(choice) - 1]
    if selected[1] == "custom":
        return Prompt.ask("[bold]Enter custom model ID[/bold]")
    return selected[1]


def select_suite() -> str:
    """Interactive suite selection."""
    console.print("\n[bold cyan]Select Evaluation Suite:[/bold cyan]\n")

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("", width=4)
    table.add_column("Suite", width=12)
    table.add_column("Tasks")
    table.add_column("Complexity")

    table.add_row("[green]1[/green]", "[bold green]Easy[/bold green]", "29 tasks", "[dim]Basic arithmetic, statistics, counting[/dim]")
    table.add_row("[yellow]2[/yellow]", "[bold yellow]Medium[/bold yellow]", "5 tasks (49 variations)", "[dim]Sequence patterns, recursive reasoning[/dim]")
    table.add_row("[red]3[/red]", "[bold red]Hard[/bold red]", "10 tasks (68 variations)", "[dim]NP-complete problems, constraint satisfaction[/dim]")
    table.add_row("[magenta]4[/magenta]", "[bold magenta]All[/bold magenta]", "44 tasks (117 variations)", "[dim]Complete evaluation across all suites[/dim]")

    console.print(table)

    choice = Prompt.ask("\n[bold]Enter your choice[/bold]", choices=["1", "2", "3", "4"], default="4")

    suites = {"1": "easy", "2": "medium", "3": "hard", "4": "all"}
    return suites[choice]


def configure_parameters() -> Dict[str, Any]:
    """Interactive parameter configuration."""
    console.print("\n[bold cyan]Configure Evaluation Parameters:[/bold cyan]\n")

    params = {}

    # Datapoints
    params['datapoints'] = IntPrompt.ask(
        "[bold]Number of datapoints per task[/bold]",
        default=100
    )

    # Temperature
    temp_str = Prompt.ask(
        "[bold]Temperature (0.0-1.0)[/bold]",
        default="0.1"
    )
    params['temperature'] = float(temp_str)

    # Max tokens
    params['max_tokens'] = IntPrompt.ask(
        "[bold]Maximum tokens to generate[/bold]",
        default=32768
    )

    # Seed
    params['seed'] = IntPrompt.ask(
        "[bold]Random seed[/bold]",
        default=42
    )

    # Output directory
    params['output_dir'] = Prompt.ask(
        "[bold]Output directory[/bold]",
        default="./results"
    )

    return params


def check_api_key(provider: str) -> Optional[str]:
    """Check and prompt for API key."""
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    if provider == "local":
        return None

    env_var = env_vars.get(provider)
    api_key = os.environ.get(env_var)

    if api_key:
        console.print(f"\n[green]✅ {env_var} found in environment[/green]")
        return api_key

    console.print(f"\n[yellow]⚠️  {env_var} not found in environment[/yellow]")
    if Confirm.ask("[bold]Would you like to enter your API key now?[/bold]"):
        api_key = Prompt.ask(f"[bold]Enter your {provider.title()} API key[/bold]", password=True)
        return api_key

    console.print("[red]❌ API key required for this provider[/red]")
    return None


def run_evaluation_wizard():
    """Main evaluation wizard flow."""
    console.print("\n[bold green]🚀 Starting Evaluation Wizard[/bold green]\n")

    # Step 1: Select provider
    provider = select_model_type()

    # Step 2: Check API key
    api_key = check_api_key(provider)
    if provider != "local" and not api_key:
        return

    # Step 3: Select model
    model_id = select_model_id(provider)

    # Step 4: Select suite
    suite = select_suite()

    # Step 5: Configure parameters
    if Confirm.ask("\n[bold]Would you like to customize evaluation parameters?[/bold]", default=False):
        params = configure_parameters()
    else:
        params = {
            'datapoints': 100,
            'temperature': 0.1,
            'max_tokens': 32768,
            'seed': 42,
            'output_dir': './results'
        }

    # Summary
    console.print("\n")
    summary_table = Table(title="[bold]Evaluation Configuration[/bold]", box=box.ROUNDED, border_style="green")
    summary_table.add_column("Setting", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Provider", provider)
    summary_table.add_row("Model", model_id)
    summary_table.add_row("Suite", suite)
    summary_table.add_row("Datapoints", str(params['datapoints']))
    summary_table.add_row("Temperature", str(params['temperature']))
    summary_table.add_row("Max Tokens", str(params['max_tokens']))
    summary_table.add_row("Seed", str(params['seed']))
    summary_table.add_row("Output Directory", params['output_dir'])

    console.print(summary_table)

    if not Confirm.ask("\n[bold]Start evaluation with these settings?[/bold]", default=True):
        console.print("[yellow]Evaluation cancelled.[/yellow]")
        return

    try:
        from ..models.model_handler import ModelHandler
        from ..core.evaluation_engine import EvaluationEngine

        # Build model handler kwargs
        model_kwargs = {
            'model_id': model_id,
            'api_provider': provider if provider != "local" else None,
            'api_key': api_key,
        }

        console.print("\n[cyan]Initializing model handler...[/cyan]")
        model_handler = ModelHandler(**model_kwargs)

        console.print("[cyan]Initializing evaluation engine...[/cyan]")
        engine = EvaluationEngine(
            model_handler=model_handler,
            output_dir=params['output_dir'],
        )

        console.print("[cyan]Running evaluation...[/cyan]\n")
        results = engine.run_evaluation(
            suite=suite,
            datapoints=params['datapoints'],
            temperature=params['temperature'],
            max_tokens=params['max_tokens'],
        )

        console.print("\n[bold green]Evaluation complete![/bold green]")
        console.print(results)

    except Exception as e:
        console.print(f"\n[bold red]Evaluation failed: {e}[/bold red]")


def list_tasks_wizard():
    """Display available tasks."""
    console.print("\n[bold cyan]📋 Available Tasks[/bold cyan]\n")

    # Easy suite
    easy_table = Table(title="[bold green]Easy Suite (29 tasks)[/bold green]", box=box.ROUNDED)
    easy_table.add_column("#", style="dim", width=4)
    easy_table.add_column("Task", style="green")
    easy_table.add_column("Description", style="dim")

    easy_tasks = [
        ("sorting", "Sort a list of numbers"),
        ("comparison", "Compare two numbers"),
        ("sum", "Calculate sum of numbers"),
        ("multiplication", "Multiply numbers"),
        ("odd_count", "Count odd numbers"),
        ("even_count", "Count even numbers"),
        ("absolute_difference", "Calculate absolute difference"),
        ("division", "Divide two numbers"),
        ("find_maximum", "Find maximum in list"),
        ("find_minimum", "Find minimum in list"),
        ("mean", "Calculate arithmetic mean"),
        ("median", "Calculate median"),
        ("mode", "Find mode of numbers"),
        ("subtraction", "Subtract numbers"),
        ("second_maximum", "Find second largest"),
        ("range", "Calculate range"),
        ("index_of_maximum", "Find index of maximum"),
        ("count_negative", "Count negative numbers"),
        ("count_unique", "Count unique elements"),
    ]

    for i, (task, desc) in enumerate(easy_tasks[:15], 1):
        easy_table.add_row(str(i), task, desc)

    console.print(easy_table)

    # Medium suite
    console.print("\n")
    medium_table = Table(title="[bold yellow]Medium Suite (5 tasks, 49 variations)[/bold yellow]", box=box.ROUNDED)
    medium_table.add_column("#", style="dim", width=4)
    medium_table.add_column("Task", style="yellow")
    medium_table.add_column("Variations", style="dim")

    medium_tasks = [
        ("fibonacci_sequence", "6 variations"),
        ("algebraic_sequence", "10 variations"),
        ("geometric_sequence", "10 variations"),
        ("prime_sequence", "11 variations"),
        ("complex_pattern", "12 variations"),
    ]

    for i, (task, variations) in enumerate(medium_tasks, 1):
        medium_table.add_row(str(i), task, variations)

    console.print(medium_table)

    # Hard suite
    console.print("\n")
    hard_table = Table(title="[bold red]Hard Suite (10 tasks, 68 variations)[/bold red]", box=box.ROUNDED)
    hard_table.add_column("#", style="dim", width=4)
    hard_table.add_column("Task", style="red")
    hard_table.add_column("Complexity", style="dim")

    hard_tasks = [
        ("tower_hanoi", "NP-complete, 6 variations"),
        ("n_queens", "NP-complete, 4 variations"),
        ("graph_coloring", "NP-complete, 10 variations"),
        ("boolean_sat", "NP-complete, 5 variations"),
        ("sudoku_solving", "Constraint satisfaction, 8 variations"),
        ("cryptarithmetic", "Constraint satisfaction, 12 variations"),
        ("matrix_chain_multiplication", "Dynamic programming, 5 variations"),
        ("modular_systems", "Number theory, 5 variations"),
        ("constraint_optimization", "Optimization, 5 variations"),
        ("logic_grid_puzzles", "Deductive reasoning, 8 variations"),
    ]

    for i, (task, complexity) in enumerate(hard_tasks, 1):
        hard_table.add_row(str(i), task, complexity)

    console.print(hard_table)


def show_documentation():
    """Display documentation."""
    doc = """
# BeyondBench Documentation

## Quick Start

```bash
# Run evaluation with OpenAI
beyondbench evaluate --model-id gpt-4o --api-provider openai --suite easy

# Run with local model
beyondbench evaluate --model-id meta-llama/Llama-3.2-3B-Instruct --backend vllm

# List available tasks
beyondbench list-tasks
```

## Environment Variables

Set these before running evaluations:

- `OPENAI_API_KEY` - For OpenAI models
- `GEMINI_API_KEY` - For Google Gemini models
- `ANTHROPIC_API_KEY` - For Anthropic Claude models

## Evaluation Suites

- **Easy Suite**: 29 basic arithmetic and statistical tasks
- **Medium Suite**: 5 sequence pattern recognition tasks (49 variations)
- **Hard Suite**: 10 NP-complete and constraint satisfaction tasks (68 variations)

## For More Information

- Contact: gks@vt.edu
- Paper: See main.tex in this repository
    """

    console.print(Panel(Markdown(doc), title="[bold]Documentation[/bold]", border_style="blue"))


def main_wizard():
    """Main wizard entry point."""
    if not RICH_AVAILABLE:
        print("Error: Rich library required for interactive wizard.")
        print("Install with: pip install rich")
        sys.exit(1)

    while True:
        print_banner()
        print_menu()

        choice = Prompt.ask("\n[bold]Enter your choice[/bold]", choices=["1", "2", "3", "4", "5", "6"], default="1")

        if choice == "1":
            run_evaluation_wizard()
        elif choice == "2":
            list_tasks_wizard()
        elif choice == "3":
            console.print("\n[yellow]Configuration wizard coming soon...[/yellow]")
        elif choice == "4":
            console.print("\n[yellow]Results viewer coming soon...[/yellow]")
        elif choice == "5":
            show_documentation()
        elif choice == "6":
            console.print("\n[bold green]Thank you for using BeyondBench! 👋[/bold green]\n")
            break

        if choice != "6":
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    main_wizard()

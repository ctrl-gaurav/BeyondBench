"""
BeyondBench Parser Debugger
============================
Paste any model response and see exactly how each parsing strategy processes it —
value extracted, confidence, raw match, and which strategy "wins".

Run with:
    python examples/gradio_parser_debugger.py

Requirements:
    pip install gradio
    pip install beyondbench  # or: pip install -e .
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Parser strategy helpers
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    "boxed",
    "explicit_statement",
    "code_block",
    "latex_math",
    "json",
    "list",
    "grid",
    "comparison",
    "sequence",
    "fallback",
]

STRATEGY_DESCRIPTIONS = {
    "boxed": "LaTeX \\boxed{} and variants, <answer> tags, bold final-answer patterns.",
    "explicit_statement": "'The answer is X', 'Therefore X', 'Hence X', '= X' patterns.",
    "code_block": "Extracts the last numeric expression from ```code blocks```.",
    "latex_math": "Inline $ and $$ math environments, \\( \\) delimiters.",
    "json": "Parses JSON objects/arrays and extracts relevant fields.",
    "list": "Extracts Python-style or comma-separated lists of numbers.",
    "grid": "Parses multi-line grids (matrices, sudoku, etc.).",
    "comparison": "Extracts comparison results (greater/less/equal, True/False).",
    "sequence": "Extracts comma/semicolon-separated numeric sequences.",
    "fallback": "Scans for the last number in the entire response as a last resort.",
}


def _available_task_names() -> List[str]:
    try:
        from beyondbench.core.task_registry import TaskRegistry
        reg = TaskRegistry()
        tasks = []
        for suite in ("easy", "medium", "hard"):
            tasks.extend(reg.get_tasks_for_suite(suite))
        return sorted(tasks)
    except Exception:
        return sorted([
            "sum", "sorting", "comparison", "multiplication", "even_count", "odd_count",
            "find_maximum", "find_minimum", "mean", "median", "mode",
            "fibonacci_sequence", "algebraic_sequence", "geometric_sequence",
            "tower_hanoi", "n_queens", "graph_coloring",
        ])


def _run_single_strategy(strategy_name: str, response: str, task_name: str) -> Dict[str, Any]:
    """Run a single parsing strategy on the response."""
    try:
        strategy_module_map = {
            "boxed": ("beyondbench.parsers.strategies.boxed_strategy", "BoxedStrategy"),
            "explicit_statement": ("beyondbench.parsers.strategies.explicit_statement_strategy", "ExplicitStatementStrategy"),
            "code_block": ("beyondbench.parsers.strategies.code_block_strategy", "CodeBlockStrategy"),
            "latex_math": ("beyondbench.parsers.strategies.latex_math_strategy", "LatexMathStrategy"),
            "json": ("beyondbench.parsers.strategies.json_strategy", "JsonStrategy"),
            "list": ("beyondbench.parsers.strategies.list_strategy", "ListStrategy"),
            "grid": ("beyondbench.parsers.strategies.grid_strategy", "GridStrategy"),
            "comparison": ("beyondbench.parsers.strategies.comparison_strategy", "ComparisonStrategy"),
            "sequence": ("beyondbench.parsers.strategies.sequence_strategy", "SequenceStrategy"),
            "fallback": ("beyondbench.parsers.strategies.fallback_strategy", "FallbackStrategy"),
        }
        if strategy_name not in strategy_module_map:
            return {"value": None, "confidence": 0.0, "raw_match": "N/A", "error": "Unknown strategy"}

        mod_path, cls_name = strategy_module_map[strategy_name]
        import importlib
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        strategy = cls()

        from beyondbench.parsers.task_configs import get_task_config
        try:
            config = get_task_config(task_name)
        except Exception:
            from beyondbench.parsers.task_configs import ParserConfig
            config = ParserConfig(
                expected_type="int",
                strategies=ALL_STRATEGIES,
                tolerance=0,
                post_processors=[],
                description="Generic task",
            )

        result = strategy.parse(response, config)
        return {
            "value": result.value,
            "confidence": result.confidence,
            "raw_match": result.raw_match if hasattr(result, "raw_match") else "N/A",
            "error": None,
        }
    except Exception as e:
        return {
            "value": None,
            "confidence": 0.0,
            "raw_match": "N/A",
            "error": str(e),
        }


def _run_unified_parser(response: str, task_name: str) -> Dict[str, Any]:
    try:
        from beyondbench.parsers.core import UnifiedParser
        from beyondbench.parsers.task_configs import get_task_config
        config = get_task_config(task_name)
        parser = UnifiedParser(config)
        result = parser.parse(response)
        return {
            "value": result.value,
            "confidence": result.confidence,
            "strategy_used": result.strategy_used,
            "error": None,
        }
    except Exception as e:
        return {
            "value": None,
            "confidence": 0.0,
            "strategy_used": "none",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def debug_response(response: str, task_name: str, run_all_strategies: bool):
    """Main callback: parse response and return per-strategy table + summary."""
    if not response.strip():
        empty_rows = [[s, "—", "—", "—", "—"] for s in ALL_STRATEGIES]
        return empty_rows, "Enter a model response above.", ""

    # Run unified parser (winner)
    unified = _run_unified_parser(response, task_name)

    # Build per-strategy rows
    rows = []
    strategies_to_run = ALL_STRATEGIES if run_all_strategies else (
        _get_task_strategies(task_name) or ALL_STRATEGIES
    )

    for strat in ALL_STRATEGIES:
        if strat in strategies_to_run:
            r = _run_single_strategy(strat, response, task_name)
            val = str(r["value"]) if r["value"] is not None else "—"
            conf = f"{r['confidence']:.2f}" if r["confidence"] else "0.00"
            raw = str(r["raw_match"])[:60] if r["raw_match"] and r["raw_match"] != "N/A" else "—"
            err = r["error"] or ""
            is_winner = "YES" if strat == unified.get("strategy_used") else ""
        else:
            val, conf, raw, err, is_winner = "—", "—", "—", "(skipped)", ""
        rows.append([strat, val, conf, raw[:60], is_winner])

    # Summary
    summary_parts = [
        f"**Winning strategy:** `{unified.get('strategy_used', 'none')}`",
        f"**Parsed value:** `{unified.get('value')}`",
        f"**Confidence:** `{unified.get('confidence', 0):.2f}`",
    ]
    if unified.get("error"):
        summary_parts.append(f"**Error:** {unified['error']}")
    summary = "\n\n".join(summary_parts)

    # Build details text
    desc = STRATEGY_DESCRIPTIONS.get(unified.get("strategy_used", ""), "")
    detail = f"**{unified.get('strategy_used', 'none')} strategy:** {desc}" if desc else ""

    return rows, summary, detail


def _get_task_strategies(task_name: str) -> List[str]:
    try:
        from beyondbench.parsers.task_configs import get_task_config
        config = get_task_config(task_name)
        return config.strategies
    except Exception:
        return ALL_STRATEGIES


def load_task_strategies(task_name: str) -> str:
    """Show which strategies this task uses."""
    strategies = _get_task_strategies(task_name)
    return f"Strategies for **{task_name}**: " + ", ".join(f"`{s}`" for s in strategies)


def load_example(task_name: str) -> str:
    """Load an example model response for the selected task."""
    examples = {
        "sum": "Let me calculate the sum step by step.\n\n1 + 2 + 3 + 4 + 5 = 15\n\nTherefore the answer is \\boxed{15}",
        "sorting": "To sort [3, 1, 4, 1, 5], I arrange in ascending order.\n\n```python\nsorted_list = [1, 1, 3, 4, 5]\n```\n\nThe answer is \\boxed{[1, 1, 3, 4, 5]}",
        "comparison": "Comparing 42 and 17: clearly 42 is greater.\n\nThe answer is **greater**.\n\n\\boxed{greater}",
        "fibonacci_sequence": "The Fibonacci sequence starts: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34\n\nThe first 10 terms are: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34",
        "tower_hanoi": "To solve Tower of Hanoi with 3 disks:\n1. Move disk 1: A → C\n2. Move disk 2: A → B\n3. Move disk 1: C → B\n4. Move disk 3: A → C\n5. Move disk 1: B → A\n6. Move disk 2: B → C\n7. Move disk 1: A → C\n\nTotal moves: 7",
        "mean": "The numbers are [10, 20, 30, 40, 50].\nSum = 150, Count = 5.\nMean = 150 / 5 = 30\n\n\\boxed{30}",
        "even_count": "Counting even numbers in [1, 2, 3, 4, 5, 6]:\n- 2: even\n- 4: even\n- 6: even\n\nThere are 3 even numbers. \\boxed{3}",
    }
    return examples.get(task_name, f"Paste a model response for task '{task_name}' here...")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="BeyondBench — Parser Debugger", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# BeyondBench Parser Debugger
Paste any model response and see how each parsing strategy handles it.
Useful for developing new parsers or debugging existing ones.
"""
        )

        with gr.Row():
            task_dd = gr.Dropdown(
                label="Task (determines which strategies are active)",
                choices=_available_task_names(),
                value="sum",
                allow_custom_value=True,
            )
            run_all_cb = gr.Checkbox(
                label="Run ALL strategies (not just task defaults)",
                value=False,
            )
            load_example_btn = gr.Button("Load Example Response", variant="secondary")

        task_info_md = gr.Markdown(load_task_strategies("sum"))

        response_box = gr.Textbox(
            label="Model Response",
            lines=8,
            placeholder="Paste the model's response here...",
            value="Let me calculate: 1 + 2 + 3 + 4 + 5 = 15\n\nThe final answer is \\boxed{15}",
        )

        debug_btn = gr.Button("Debug Response", variant="primary")

        summary_md = gr.Markdown("Click 'Debug Response' to see results.")
        detail_md = gr.Markdown("")

        results_table = gr.Dataframe(
            label="Per-Strategy Results",
            headers=["Strategy", "Extracted Value", "Confidence", "Raw Match (truncated)", "Winner"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        gr.Markdown(
            """
---
### Strategy Reference
| Strategy | Description |
|----------|-------------|
| `boxed` | LaTeX `\\boxed{}`, `<answer>` tags, bold final-answer |
| `explicit_statement` | "The answer is X", "Therefore X", "= X" |
| `code_block` | Last numeric expression from ` ```code``` ` blocks |
| `latex_math` | Inline `$...$` and `$$...$$` math environments |
| `json` | JSON objects / arrays with relevant numeric fields |
| `list` | Python-style or comma-separated number lists |
| `grid` | Multi-line grids (matrices, sudoku) |
| `comparison` | greater / less / equal, True / False answers |
| `sequence` | Comma/semicolon-separated numeric sequences |
| `fallback` | Last number anywhere in the response |
"""
        )

        # Wire up
        task_dd.change(fn=load_task_strategies, inputs=[task_dd], outputs=[task_info_md])
        load_example_btn.click(fn=load_example, inputs=[task_dd], outputs=[response_box])

        debug_btn.click(
            fn=debug_response,
            inputs=[response_box, task_dd, run_all_cb],
            outputs=[results_table, summary_md, detail_md],
        )

        # Auto-debug on response change
        response_box.change(
            fn=debug_response,
            inputs=[response_box, task_dd, run_all_cb],
            outputs=[results_table, summary_md, detail_md],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)

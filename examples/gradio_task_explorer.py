"""
BeyondBench Task Explorer
==========================
Browse all 79 tasks by category, generate sample instances with expected
outputs, and try custom inputs.

Run with:
    python examples/gradio_task_explorer.py

Requirements:
    pip install gradio
    pip install beyondbench  # or: pip install -e .
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS: Dict[str, str] = {
    # Easy
    "sum": "Sum all numbers in a list.",
    "sorting": "Sort a list of numbers in ascending order.",
    "comparison": "Compare two numbers and determine which is greater.",
    "multiplication": "Multiply a list of numbers together.",
    "odd_count": "Count how many odd numbers are in a list.",
    "even_count": "Count how many even numbers are in a list.",
    "absolute_difference": "Find the absolute difference between two numbers.",
    "division": "Divide two numbers.",
    "find_maximum": "Find the maximum value in a list.",
    "find_minimum": "Find the minimum value in a list.",
    "mean": "Compute the arithmetic mean of a list.",
    "median": "Find the median of a list.",
    "mode": "Find the most frequently occurring number.",
    "subtraction": "Subtract one number from another.",
    "second_maximum": "Find the second largest number in a list.",
    "range": "Find the range (max - min) of a list.",
    "index_of_maximum": "Find the 0-based index of the maximum value.",
    "count_negative": "Count how many negative numbers are in a list.",
    "count_unique": "Count unique values in a list.",
    "max_adjacent_difference": "Find the maximum absolute difference between adjacent elements.",
    "count_greater_than_previous": "Count elements greater than the previous element.",
    "sum_of_max_indices": "Sum the indices of the maximum element occurrences.",
    "count_palindromic": "Count palindromic numbers in a list.",
    "longest_increasing_subsequence": "Find the length of the longest increasing subsequence.",
    "sum_of_digits": "Sum all digits of all numbers in a list.",
    "count_perfect_squares": "Count how many numbers in a list are perfect squares.",
    "alternating_sum": "Compute alternating sum (+/-) of a list.",
    "count_multiples": "Count multiples of a given number in a list.",
    "local_maxima_count": "Count local maxima (greater than both neighbors).",
    "weighted_sum": "Compute weighted sum of a list with given weights.",
    "running_average": "Compute running averages of a list.",
    "parity_check": "Classify each number as odd or even.",
    "cumulative_sum": "Compute cumulative sum of a list.",
    "reverse_list": "Reverse a list of numbers.",
    "rotate_list": "Rotate a list by k positions.",
    "interleave_lists": "Interleave two lists element by element.",
    "set_intersection": "Find common elements between two sets.",
    "set_difference": "Find elements in one set but not the other.",
    "moving_average": "Compute moving average with window k.",
    "element_frequency": "Count the frequency of each unique element.",
    "second_minimum": "Find the second smallest value.",
    "variance": "Compute the population variance.",
    "standard_deviation": "Compute the population standard deviation.",
    "dot_product": "Compute the dot product of two vectors.",
    # Medium
    "fibonacci_sequence": "Generate the first N terms of the Fibonacci sequence.",
    "algebraic_sequence": "Identify and extend an algebraic sequence.",
    "geometric_sequence": "Generate a geometric sequence given first term and ratio.",
    "prime_sequence": "Generate the first N prime numbers.",
    "complex_pattern": "Identify and continue a complex numerical pattern.",
    "arithmetic_progression": "Find terms of an arithmetic progression.",
    "harmonic_sequence": "Generate terms of the harmonic series.",
    "collatz_sequence": "Generate the Collatz sequence for a given start.",
    "polynomial_evaluation": "Evaluate a polynomial at given x values.",
    "matrix_operations": "Perform matrix addition, subtraction, or multiplication.",
    "number_base_conversion": "Convert numbers between different bases.",
    "logical_operations": "Apply logical operators (AND, OR, XOR) to bit strings.",
    "pattern_completion": "Identify the rule and complete a number pattern.",
    "gcd_lcm": "Compute GCD and LCM of two numbers.",
    "combinatorics": "Compute combinations and permutations.",
    # Hard
    "tower_hanoi": "Determine moves to solve Tower of Hanoi with N disks.",
    "n_queens": "Place N queens on an N×N board so none attack each other.",
    "graph_coloring": "Color a graph with minimum colors so no adjacent nodes share one.",
    "boolean_sat": "Determine if a boolean formula is satisfiable.",
    "sudoku_solving": "Solve a 9×9 Sudoku puzzle.",
    "cryptarithmetic": "Solve a cryptarithmetic puzzle (e.g. SEND+MORE=MONEY).",
    "matrix_chain_multiplication": "Find optimal parenthesization for matrix chain multiplication.",
    "modular_systems": "Solve systems of modular equations.",
    "constraint_optimization": "Find optimal assignment satisfying constraints.",
    "logic_grid_puzzles": "Solve a logic grid puzzle from clues.",
    "shortest_path": "Find the shortest path in a weighted graph.",
    "knapsack": "Solve the 0/1 knapsack problem.",
    "traveling_salesman": "Find the shortest tour visiting all cities.",
    "longest_common_subsequence": "Find the LCS of two sequences.",
    "minimax_game": "Compute the minimax value of a game tree.",
    "regex_matching": "Determine if a string matches a pattern.",
    "topological_sort": "Produce a valid topological ordering of a DAG.",
    "interval_scheduling": "Select maximum non-overlapping intervals.",
    "coin_change": "Find the minimum coins to make a target amount.",
    "edit_distance": "Compute the Levenshtein edit distance between two strings.",
}


def _get_tasks_by_suite() -> Dict[str, List[str]]:
    try:
        from beyondbench.core.task_registry import TaskRegistry
        reg = TaskRegistry()
        return {s: reg.get_tasks_for_suite(s) for s in ("easy", "medium", "hard")}
    except Exception:
        easy = [k for k in TASK_DESCRIPTIONS if k not in ("fibonacci_sequence",
                "algebraic_sequence", "geometric_sequence", "prime_sequence", "complex_pattern",
                "arithmetic_progression", "harmonic_sequence", "collatz_sequence",
                "polynomial_evaluation", "matrix_operations", "number_base_conversion",
                "logical_operations", "pattern_completion", "gcd_lcm", "combinatorics",
                "tower_hanoi", "n_queens", "graph_coloring", "boolean_sat", "sudoku_solving",
                "cryptarithmetic", "matrix_chain_multiplication", "modular_systems",
                "constraint_optimization", "logic_grid_puzzles", "shortest_path", "knapsack",
                "traveling_salesman", "longest_common_subsequence", "minimax_game",
                "regex_matching", "topological_sort", "interval_scheduling", "coin_change",
                "edit_distance")]
        medium = ["fibonacci_sequence", "algebraic_sequence", "geometric_sequence",
                  "prime_sequence", "complex_pattern", "arithmetic_progression",
                  "harmonic_sequence", "collatz_sequence", "polynomial_evaluation",
                  "matrix_operations", "number_base_conversion", "logical_operations",
                  "pattern_completion", "gcd_lcm", "combinatorics"]
        hard = ["tower_hanoi", "n_queens", "graph_coloring", "boolean_sat", "sudoku_solving",
                "cryptarithmetic", "matrix_chain_multiplication", "modular_systems",
                "constraint_optimization", "logic_grid_puzzles", "shortest_path", "knapsack",
                "traveling_salesman", "longest_common_subsequence", "minimax_game",
                "regex_matching", "topological_sort", "interval_scheduling", "coin_change",
                "edit_distance"]
        return {"easy": easy, "medium": medium, "hard": hard}


TASKS_BY_SUITE = _get_tasks_by_suite()


def _all_task_rows() -> List[List[str]]:
    rows = []
    for suite, tasks in TASKS_BY_SUITE.items():
        for t in tasks:
            desc = TASK_DESCRIPTIONS.get(t, "")
            rows.append([t, suite, desc])
    return rows


def _generate_example(task_name: str, seed: int) -> Tuple[str, str]:
    """Return (prompt, ground_truth) for a task."""
    try:
        from beyondbench.core.task_registry import TaskRegistry
        reg = TaskRegistry()
        task_cls = reg.get_task_class(task_name)
        if task_cls:
            task = task_cls(model_handler=None, num_samples=1, seed=seed)
            data = task.generate_data_point()
            prompt = task.create_prompt(data)
            gt = str(task.get_ground_truth(data))
            return prompt, gt
    except Exception:
        pass

    # Fallback examples
    rng = random.Random(seed)
    nums = [rng.randint(1, 50) for _ in range(6)]
    fallbacks: Dict[str, Tuple[str, str]] = {
        "sum": (f"Sum of {nums}?", str(sum(nums))),
        "sorting": (f"Sort {nums}?", str(sorted(nums))),
        "find_maximum": (f"Max of {nums}?", str(max(nums))),
        "find_minimum": (f"Min of {nums}?", str(min(nums))),
        "even_count": (f"Count evens in {nums}?", str(sum(1 for n in nums if n % 2 == 0))),
        "odd_count": (f"Count odds in {nums}?", str(sum(1 for n in nums if n % 2 != 0))),
        "mean": (f"Mean of {nums}?", f"{sum(nums)/len(nums):.2f}"),
        "fibonacci_sequence": (f"First 8 Fibonacci numbers?", "0, 1, 1, 2, 3, 5, 8, 13"),
        "tower_hanoi": (f"Solve Tower of Hanoi with 3 disks.", "Move disk 1 A→C, disk 2 A→B, disk 1 C→B, disk 3 A→C, disk 1 B→A, disk 2 B→C, disk 1 A→C"),
    }
    if task_name in fallbacks:
        p, gt = fallbacks[task_name]
        prompt = f"{p}\n\nProvide the answer in \\boxed{{answer}} format."
        return prompt, gt

    prompt = (
        f"Compute the result for task '{task_name}' on input {nums}.\n"
        f"Provide your final answer in \\boxed{{answer}} format."
    )
    return prompt, "(depends on task logic)"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def filter_tasks(category: str, search: str) -> List[List[str]]:
    rows = _all_task_rows()
    if category != "All":
        rows = [r for r in rows if r[1] == category]
    if search.strip():
        q = search.strip().lower()
        rows = [r for r in rows if q in r[0] or q in r[2].lower()]
    return rows


def load_task_details(evt: gr.SelectData, df_data) -> Tuple[str, str, str, str, str]:
    """Called when user clicks a row in the task table."""
    try:
        # evt.index is (row, col)
        row_idx = evt.index[0]
        task_name = df_data[row_idx][0]
        suite = df_data[row_idx][1]
        desc = df_data[row_idx][2]
    except Exception:
        return "", "", "", "", ""
    prompt, gt = _generate_example(task_name, seed=42)
    difficulty_label = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}.get(suite, suite)
    header = f"## {task_name.replace('_', ' ').title()}\n**Suite:** {difficulty_label} | **Description:** {desc}"
    return header, task_name, prompt, gt, str(42)


def regenerate_example(task_name: str, seed_str: str):
    try:
        seed = int(seed_str)
    except Exception:
        seed = random.randint(0, 9999)
    if not task_name:
        return "", "", str(seed)
    prompt, gt = _generate_example(task_name, seed=seed)
    return prompt, gt, str(seed)


def try_custom_input(task_name: str, custom_prompt: str):
    """Evaluate a custom prompt using the task's parser."""
    if not task_name or not custom_prompt.strip():
        return "Please select a task and enter a response to evaluate."
    try:
        from beyondbench.parsers.task_configs import get_task_config
        from beyondbench.parsers.core import UnifiedParser
        config = get_task_config(task_name)
        parser = UnifiedParser(config)
        result = parser.parse(custom_prompt)
        return (
            f"**Parsed value:** `{result.value}`\n"
            f"**Confidence:** `{result.confidence:.2f}`\n"
            f"**Strategy:** `{result.strategy_used}`\n"
        )
    except Exception as e:
        return f"Parser not available for this task or error occurred: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="BeyondBench — Task Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# BeyondBench Task Explorer
Browse all 79 tasks across Easy (44), Medium (15), and Hard (20) suites.
Click a task to see a sample instance and try your own inputs.
"""
        )

        # Current selected task state
        selected_task_state = gr.State("")

        with gr.Row():
            category_filter = gr.Radio(
                label="Category",
                choices=["All", "easy", "medium", "hard"],
                value="All",
            )
            search_box = gr.Textbox(
                label="Search tasks",
                placeholder="Type to filter by name or description...",
                scale=2,
            )

        task_table = gr.Dataframe(
            value=_all_task_rows(),
            headers=["Task Name", "Suite", "Description"],
            datatype=["str", "str", "str"],
            interactive=False,
            wrap=True,
            height=300,
            label="Task Catalog (click a row to select)",
        )

        task_header = gr.Markdown("## Select a task from the table above")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Sample Instance")
                prompt_box = gr.Textbox(
                    label="Task Prompt",
                    lines=6,
                    interactive=False,
                    placeholder="Click a task to see a sample prompt...",
                )
                gt_box = gr.Textbox(
                    label="Ground Truth",
                    interactive=False,
                    placeholder="Expected answer...",
                )
                seed_box = gr.Textbox(label="Seed", value="42", scale=1)
                regen_btn = gr.Button("New Sample", variant="secondary")

            with gr.Column():
                gr.Markdown("### Try Custom Response")
                custom_input = gr.Textbox(
                    label="Paste a model response here",
                    lines=6,
                    placeholder="E.g.: The sum is \\boxed{42}",
                )
                parse_btn = gr.Button("Evaluate Response", variant="primary")
                parse_result = gr.Markdown("Parser output will appear here.")

        # Wire up
        def _filter(cat, search):
            return filter_tasks(cat, search)

        category_filter.change(fn=_filter, inputs=[category_filter, search_box], outputs=[task_table])
        search_box.change(fn=_filter, inputs=[category_filter, search_box], outputs=[task_table])

        task_table.select(
            fn=load_task_details,
            inputs=[task_table],
            outputs=[task_header, selected_task_state, prompt_box, gt_box, seed_box],
        )

        regen_btn.click(
            fn=regenerate_example,
            inputs=[selected_task_state, seed_box],
            outputs=[prompt_box, gt_box, seed_box],
        )

        parse_btn.click(
            fn=try_custom_input,
            inputs=[selected_task_state, custom_input],
            outputs=[parse_result],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)

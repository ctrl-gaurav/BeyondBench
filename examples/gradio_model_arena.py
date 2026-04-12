"""
BeyondBench Model Arena
========================
Side-by-side model comparison with human voting and a leaderboard.

Run with:
    python examples/gradio_model_arena.py

Requirements:
    pip install gradio
    pip install beyondbench  # or: pip install -e .
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Task registry helpers
# ---------------------------------------------------------------------------

def _get_tasks_by_suite() -> Dict[str, List[str]]:
    try:
        from beyondbench.core.task_registry import TaskRegistry
        reg = TaskRegistry()
        return {s: reg.get_tasks_for_suite(s) for s in ("easy", "medium", "hard")}
    except Exception:
        return {
            "easy": ["sum", "sorting", "comparison", "even_count", "find_maximum"],
            "medium": ["fibonacci_sequence", "algebraic_sequence", "geometric_sequence"],
            "hard": ["tower_hanoi", "n_queens", "graph_coloring"],
        }


TASKS_BY_SUITE = _get_tasks_by_suite()
ALL_TASKS: List[str] = [t for tasks in TASKS_BY_SUITE.values() for t in tasks]

LOCAL_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2b-it",
]

# ---------------------------------------------------------------------------
# Leaderboard state (in-memory)
# ---------------------------------------------------------------------------

_leaderboard: Dict[str, Dict[str, int]] = defaultdict(
    lambda: {"wins": 0, "losses": 0, "ties": 0, "total_votes": 0}
)
_lb_lock = threading.Lock()

# Current arena state (last generated comparison)
_arena: Dict[str, Any] = {
    "prompt": "",
    "response_a": "",
    "response_b": "",
    "model_a": "",
    "model_b": "",
    "ground_truth": "",
    "task": "",
}


# ---------------------------------------------------------------------------
# Task prompt generation
# ---------------------------------------------------------------------------

def _generate_task_instance(task_name: str, seed: Optional[int] = None) -> Tuple[str, str]:
    """Return (prompt, ground_truth_str) for a given task."""
    if seed is None:
        seed = random.randint(0, 10000)
    try:
        from beyondbench.core.task_registry import TaskRegistry
        reg = TaskRegistry()
        task_cls = reg.get_task_class(task_name)
        if task_cls is None:
            raise ValueError("Task not found")
        task = task_cls(model_handler=None, num_samples=1, seed=seed)
        data = task.generate_data_point()  # type: ignore[attr-defined]
        prompt = task.create_prompt(data)
        gt = str(task.get_ground_truth(data))
        return prompt, gt
    except Exception:
        pass

    # Fallback: generate a simple arithmetic prompt
    rng = random.Random(seed)
    nums = [rng.randint(1, 100) for _ in range(5)]
    if task_name == "sum":
        return (
            f"What is the sum of {nums}? Answer in \\boxed{{answer}}.",
            str(sum(nums)),
        )
    if task_name == "find_maximum":
        return (
            f"What is the maximum value in {nums}? Answer in \\boxed{{answer}}.",
            str(max(nums)),
        )
    if task_name == "even_count":
        cnt = sum(1 for n in nums if n % 2 == 0)
        return (
            f"How many even numbers are in {nums}? Answer in \\boxed{{answer}}.",
            str(cnt),
        )
    return (
        f"Compute the result for task '{task_name}' on input {nums}. Answer in \\boxed{{answer}}.",
        "(varies)",
    )


def _simulate_response(model_id: str, prompt: str, task_name: str, seed: int) -> str:
    """Simulate a model response (used when vLLM is not available)."""
    rng = random.Random(hash(model_id + prompt) % 100000 + seed)
    # Simulate chain-of-thought style
    lines = [
        f"Let me solve this step by step.",
        "",
        f"Looking at the problem: {prompt[:80]}...",
        "",
    ]
    # Sometimes give correct-ish answer, sometimes wrong
    if rng.random() < 0.65:
        lines.append("After careful computation, my answer is:")
        lines.append(f"\\boxed{{{rng.randint(1, 200)}}}")
    else:
        lines.append(f"The result is approximately {rng.randint(1, 200)}.")
    return "\n".join(lines)


def _get_model_response(model_id: str, prompt: str) -> str:
    try:
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler(model_id=model_id, backend="vllm")
        responses = handler.generate([prompt], max_tokens=256)
        return responses[0] if responses else "(empty response)"
    except Exception as e:
        return _simulate_response(model_id, prompt, "", 42)


# ---------------------------------------------------------------------------
# Arena callbacks
# ---------------------------------------------------------------------------

def generate_comparison(model_a: str, model_b: str, task_name: str, seed_val: int):
    """Generate a task instance and get responses from both models."""
    if model_a == model_b:
        return (
            gr.update(value="⚠️ Please select two different models."),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        )

    prompt, gt = _generate_task_instance(task_name, seed=int(seed_val))
    _arena["prompt"] = prompt
    _arena["ground_truth"] = gt
    _arena["model_a"] = model_a
    _arena["model_b"] = model_b
    _arena["task"] = task_name

    # Get responses (in parallel via threads for speed)
    resp_a_container = [None]
    resp_b_container = [None]

    def fetch_a():
        resp_a_container[0] = _get_model_response(model_a, prompt)

    def fetch_b():
        resp_b_container[0] = _get_model_response(model_b, prompt)

    t_a = threading.Thread(target=fetch_a, daemon=True)
    t_b = threading.Thread(target=fetch_b, daemon=True)
    t_a.start(); t_b.start()
    t_a.join(timeout=30); t_b.join(timeout=30)

    resp_a = resp_a_container[0] or "(timeout)"
    resp_b = resp_b_container[0] or "(timeout)"
    _arena["response_a"] = resp_a
    _arena["response_b"] = resp_b

    return (
        gr.update(value=prompt),
        gr.update(value=resp_a),
        gr.update(value=resp_b),
        gr.update(value=f"Ground truth: **{gt}**"),
        gr.update(value="Vote for the better response below."),
    )


def _record_vote(winner: str, loser: str, tie: bool = False):
    with _lb_lock:
        if tie:
            _leaderboard[winner]["ties"] += 1
            _leaderboard[loser]["ties"] += 1
            _leaderboard[winner]["total_votes"] += 1
            _leaderboard[loser]["total_votes"] += 1
        else:
            _leaderboard[winner]["wins"] += 1
            _leaderboard[loser]["losses"] += 1
            _leaderboard[winner]["total_votes"] += 1
            _leaderboard[loser]["total_votes"] += 1


def vote_a():
    m_a = _arena.get("model_a", "")
    m_b = _arena.get("model_b", "")
    if not m_a or not m_b:
        return _get_leaderboard_df(), "No comparison generated yet."
    _record_vote(m_a, m_b)
    return _get_leaderboard_df(), f"Voted for {m_a}"


def vote_b():
    m_a = _arena.get("model_a", "")
    m_b = _arena.get("model_b", "")
    if not m_a or not m_b:
        return _get_leaderboard_df(), "No comparison generated yet."
    _record_vote(m_b, m_a)
    return _get_leaderboard_df(), f"Voted for {m_b}"


def vote_tie():
    m_a = _arena.get("model_a", "")
    m_b = _arena.get("model_b", "")
    if not m_a or not m_b:
        return _get_leaderboard_df(), "No comparison generated yet."
    _record_vote(m_a, m_b, tie=True)
    return _get_leaderboard_df(), "Recorded as tie."


def _get_leaderboard_df() -> List[List[Any]]:
    with _lb_lock:
        rows = []
        for model, stats in sorted(
            _leaderboard.items(),
            key=lambda x: x[1]["wins"],
            reverse=True,
        ):
            total = stats["total_votes"]
            win_rate = f"{stats['wins'] / total * 100:.0f}%" if total else "—"
            rows.append([
                model.split("/")[-1],
                stats["wins"],
                stats["losses"],
                stats["ties"],
                total,
                win_rate,
            ])
        return rows


def refresh_leaderboard():
    return _get_leaderboard_df()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="BeyondBench — Model Arena", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# BeyondBench Model Arena
Select two models, generate the same task instance for both, and vote for the better response.
"""
        )

        with gr.Row():
            model_a_dd = gr.Dropdown(
                label="Model A",
                choices=LOCAL_MODELS,
                value=LOCAL_MODELS[0],
                allow_custom_value=True,
            )
            model_b_dd = gr.Dropdown(
                label="Model B",
                choices=LOCAL_MODELS,
                value=LOCAL_MODELS[1],
                allow_custom_value=True,
            )

        with gr.Row():
            task_dd = gr.Dropdown(
                label="Task",
                choices=ALL_TASKS,
                value="sum",
            )
            seed_num = gr.Number(label="Seed", value=42, precision=0)
            gen_btn = gr.Button("Generate & Compare", variant="primary")

        prompt_box = gr.Textbox(
            label="Task Prompt",
            lines=4,
            interactive=False,
            placeholder="Task prompt will appear here after clicking 'Generate & Compare'",
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model A Response")
                resp_a_box = gr.Textbox(
                    lines=8, interactive=False,
                    placeholder="Model A response will appear here...",
                )
            with gr.Column():
                gr.Markdown("### Model B Response")
                resp_b_box = gr.Textbox(
                    lines=8, interactive=False,
                    placeholder="Model B response will appear here...",
                )

        gt_box = gr.Markdown("Ground truth will appear here.")
        vote_status = gr.Markdown("")

        with gr.Row():
            vote_a_btn = gr.Button("Model A is better", variant="secondary")
            tie_btn = gr.Button("Tie / Both wrong", variant="secondary")
            vote_b_btn = gr.Button("Model B is better", variant="secondary")

        gr.Markdown("---")
        gr.Markdown("## Leaderboard")

        lb_table = gr.Dataframe(
            label="Model Leaderboard",
            headers=["Model", "Wins", "Losses", "Ties", "Total Votes", "Win Rate"],
            datatype=["str", "number", "number", "number", "number", "str"],
            interactive=False,
        )

        refresh_lb_btn = gr.Button("Refresh Leaderboard")

        # Wire up
        gen_btn.click(
            fn=generate_comparison,
            inputs=[model_a_dd, model_b_dd, task_dd, seed_num],
            outputs=[prompt_box, resp_a_box, resp_b_box, gt_box, vote_status],
        )
        vote_a_btn.click(fn=vote_a, outputs=[lb_table, vote_status])
        tie_btn.click(fn=vote_tie, outputs=[lb_table, vote_status])
        vote_b_btn.click(fn=vote_b, outputs=[lb_table, vote_status])
        refresh_lb_btn.click(fn=refresh_leaderboard, outputs=[lb_table])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)

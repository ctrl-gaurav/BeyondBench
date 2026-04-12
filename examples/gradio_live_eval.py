"""
BeyondBench Live Evaluation Dashboard
======================================
Gradio app that lets you configure and launch an evaluation run, then watch
results populate in real-time as tasks complete.

Run with:
    python examples/gradio_live_eval.py

Requirements:
    pip install gradio
    pip install beyondbench  # or: pip install -e .
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _detect_gpus() -> List[str]:
    """Return list of GPU labels available on this machine."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return ["CPU only"]
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(",", 1)
            if len(parts) == 2:
                idx, name = parts[0].strip(), parts[1].strip()
                gpus.append(f"GPU {idx}: {name}")
        return gpus if gpus else ["CPU only"]
    except Exception:
        return ["CPU only"]


def _query_gpu_stats() -> List[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_used_mb": int(parts[2]),
                    "mem_total_mb": int(parts[3]),
                    "util_pct": int(parts[4]),
                    "temp_c": int(parts[5]),
                })
            except (ValueError, IndexError):
                continue
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Task registry helpers
# ---------------------------------------------------------------------------

def _get_tasks_by_suite() -> Dict[str, List[str]]:
    try:
        from beyondbench.core.task_registry import TaskRegistry
        reg = TaskRegistry()
        return {
            "easy": reg.get_tasks_for_suite("easy"),
            "medium": reg.get_tasks_for_suite("medium"),
            "hard": reg.get_tasks_for_suite("hard"),
        }
    except Exception:
        return {
            "easy": ["sum", "sorting", "comparison"],
            "medium": ["fibonacci_sequence", "algebraic_sequence"],
            "hard": ["tower_hanoi", "n_queens"],
        }


TASKS_BY_SUITE = _get_tasks_by_suite()
ALL_TASKS = [t for tasks in TASKS_BY_SUITE.values() for t in tasks]
GPU_LIST = _detect_gpus()

# ---------------------------------------------------------------------------
# Known local models
# ---------------------------------------------------------------------------

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
# Shared run state
# ---------------------------------------------------------------------------

_run_state: Dict[str, Any] = {
    "running": False,
    "rows": [],           # list of [task, accuracy, parsed, total, time_s] rows
    "log": [],
    "results_json": None,
    "stop_flag": False,
}
_lock = threading.Lock()


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _lock:
        _run_state["log"].append(line)
        if len(_run_state["log"]) > 500:
            _run_state["log"] = _run_state["log"][-400:]


# ---------------------------------------------------------------------------
# Simulated / real evaluation runner
# ---------------------------------------------------------------------------

def _run_evaluation_thread(
    model_id: str,
    suite: str,
    datapoints: int,
    gpu_selection: str,
    output_dir: str,
):
    """Run evaluation in a background thread, updating _run_state."""
    with _lock:
        _run_state["running"] = True
        _run_state["rows"] = []
        _run_state["log"] = []
        _run_state["results_json"] = None
        _run_state["stop_flag"] = False

    _log(f"Starting evaluation: model={model_id}, suite={suite}, datapoints={datapoints}")

    # Determine CUDA_VISIBLE_DEVICES from selection
    cuda_env = os.environ.copy()
    if gpu_selection and gpu_selection != "CPU only" and gpu_selection.startswith("GPU "):
        gpu_idx = gpu_selection.split(":")[0].replace("GPU ", "").strip()
        cuda_env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        _log(f"Using GPU {gpu_idx}")

    tasks = TASKS_BY_SUITE.get(suite, []) if suite != "all" else ALL_TASKS

    # Try to use real evaluation engine; fall back to simulation
    try:
        from beyondbench.core.task_registry import TaskRegistry
        from beyondbench.models.model_handler import ModelHandler
        registry = TaskRegistry()
        _log("Initializing model handler...")
        handler = ModelHandler(model_id=model_id, backend="vllm")

        for task_name in tasks:
            if _run_state["stop_flag"]:
                _log("Stopped by user.")
                break
            _log(f"Running task: {task_name}")
            t0 = time.time()
            try:
                task_cls = registry.get_task_class(task_name)
                task = task_cls(
                    model_handler=handler,
                    num_samples=datapoints,
                    seed=42,
                )
                result = task.run()
                elapsed = time.time() - t0
                acc = result.get("accuracy", 0.0)
                parsed = result.get("parsed", datapoints)
                row = [task_name, f"{acc*100:.1f}%", parsed, datapoints, f"{elapsed:.1f}s"]
                with _lock:
                    _run_state["rows"].append(row)
                _log(f"  {task_name}: accuracy={acc*100:.1f}%, time={elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                row = [task_name, "ERROR", 0, datapoints, f"{elapsed:.1f}s"]
                with _lock:
                    _run_state["rows"].append(row)
                _log(f"  {task_name}: ERROR — {e}")

    except ImportError:
        # Simulate results for demo purposes
        _log("Model handler not available — running simulation mode.")
        import random
        rng = random.Random(42)
        for task_name in tasks:
            if _run_state["stop_flag"]:
                _log("Stopped by user.")
                break
            _log(f"[SIM] Running task: {task_name}")
            sim_time = rng.uniform(0.5, 3.0)
            time.sleep(min(sim_time, 1.0))
            acc = rng.uniform(0.4, 0.95)
            parsed = int(acc * datapoints)
            row = [task_name, f"{acc*100:.1f}%", parsed, datapoints, f"{sim_time:.1f}s"]
            with _lock:
                _run_state["rows"].append(row)
            _log(f"  [SIM] {task_name}: accuracy={acc*100:.1f}%")

    # Aggregate results
    rows = _run_state["rows"]
    if rows:
        accs = []
        for r in rows:
            try:
                accs.append(float(r[1].rstrip("%")) / 100)
            except Exception:
                pass
        avg = sum(accs) / len(accs) if accs else 0.0
        results = {
            "model_id": model_id,
            "suite": suite,
            "datapoints": datapoints,
            "overall_accuracy": avg,
            "task_results": {r[0]: r[1] for r in rows},
        }
        with _lock:
            _run_state["results_json"] = json.dumps(results, indent=2)
        _log(f"Evaluation complete. Overall accuracy: {avg*100:.1f}%")
    else:
        _log("No results collected.")

    with _lock:
        _run_state["running"] = False


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def start_evaluation(model_id, suite, datapoints, gpu_selection, output_dir):
    with _lock:
        if _run_state["running"]:
            return (
                gr.update(),
                gr.update(),
                gr.update(value="⚠️ Evaluation already running. Stop it first."),
            )

    output_dir = output_dir.strip() or "./beyondbench_results"
    t = threading.Thread(
        target=_run_evaluation_thread,
        args=(model_id, suite, datapoints, gpu_selection, output_dir),
        daemon=True,
    )
    t.start()
    time.sleep(0.1)
    return (
        gr.update(value=[], headers=["Task", "Accuracy", "Parsed", "Total", "Time"]),
        gr.update(value=""),
        gr.update(value="Evaluation started..."),
    )


def stop_evaluation():
    with _lock:
        _run_state["stop_flag"] = True
    return gr.update(value="Stop signal sent.")


def poll_results():
    """Called periodically by the auto-refresh timer."""
    with _lock:
        rows = list(_run_state["rows"])
        log_lines = list(_run_state["log"])
        running = _run_state["running"]

    status = "Running..." if running else ("Idle" if not rows else "Complete")
    log_text = "\n".join(log_lines[-60:])
    return rows, log_text, status


def download_results():
    with _lock:
        json_str = _run_state.get("results_json")
        rows = list(_run_state["rows"])

    if not rows:
        return None, None

    # Build CSV
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Task", "Accuracy", "Parsed", "Total", "Time"])
    writer.writerows(rows)

    csv_path = "/tmp/beyondbench_results.csv"
    json_path = "/tmp/beyondbench_results.json"
    with open(csv_path, "w") as f:
        f.write(buf.getvalue())
    if json_str:
        with open(json_path, "w") as f:
            f.write(json_str)
    else:
        json_path = None

    return csv_path, json_path


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="BeyondBench — Live Evaluation Dashboard",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# BeyondBench Live Evaluation Dashboard
Configure and launch an evaluation run, then watch results populate in real time.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    label="Model",
                    choices=LOCAL_MODELS,
                    value=LOCAL_MODELS[1],
                    allow_custom_value=True,
                )
                suite_radio = gr.Radio(
                    label="Suite",
                    choices=["easy", "medium", "hard", "all"],
                    value="easy",
                )
                dp_slider = gr.Slider(
                    label="Datapoints per task",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=10,
                )
                gpu_dd = gr.Dropdown(
                    label="GPU",
                    choices=GPU_LIST,
                    value=GPU_LIST[0] if GPU_LIST else "CPU only",
                )
                output_dir_box = gr.Textbox(
                    label="Output directory",
                    value="./beyondbench_results",
                    placeholder="./beyondbench_results",
                )
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")
                status_box = gr.Textbox(label="Status", value="Idle", interactive=False)

            with gr.Column(scale=2):
                results_table = gr.Dataframe(
                    label="Task Results",
                    headers=["Task", "Accuracy", "Parsed", "Total", "Time"],
                    datatype=["str", "str", "number", "number", "str"],
                    interactive=False,
                    wrap=True,
                )

        log_box = gr.Textbox(
            label="Live Log",
            lines=12,
            max_lines=20,
            interactive=False,
            autoscroll=True,
        )

        with gr.Row():
            dl_csv_btn = gr.Button("Download CSV")
            dl_json_btn = gr.Button("Download JSON")
            csv_file = gr.File(label="CSV download", visible=True)
            json_file = gr.File(label="JSON download", visible=True)

        # Auto-refresh timer (every 1.5 s)
        timer = gr.Timer(1.5)

        # Wire up
        start_btn.click(
            fn=start_evaluation,
            inputs=[model_dd, suite_radio, dp_slider, gpu_dd, output_dir_box],
            outputs=[results_table, log_box, status_box],
        )
        stop_btn.click(fn=stop_evaluation, outputs=[status_box])
        timer.tick(fn=poll_results, outputs=[results_table, log_box, status_box])

        def _dl():
            csv_p, json_p = download_results()
            return csv_p, json_p

        dl_csv_btn.click(fn=_dl, outputs=[csv_file, json_file])
        dl_json_btn.click(fn=_dl, outputs=[csv_file, json_file])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

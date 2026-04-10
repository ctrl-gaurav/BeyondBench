"""
BeyondBench Live Observability Dashboard (Gradio Blocks)

Tabs:
  1. Overview      — progress bar, suite accuracies, model info, elapsed time
  2. Per-Task      — sortable/filterable results table
  3. GPU Monitor   — nvidia-smi live stats + memory timeline
  4. Token Analytics — input/output bar charts, throughput
  5. Live Log      — streaming log with level filter and search

This module imports Gradio at the top level — callers must guard with
  try: from beyondbench.dashboard.app import ...
  except ImportError: ...
or use beyondbench.dashboard.launch_dashboard() which does the guard.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .data_bridge import DataBridge

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _query_gpu_stats() -> List[Dict[str, Any]]:
    """Run nvidia-smi and return per-GPU stats as a list of dicts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "mem_used_mb": int(parts[2]),
                        "mem_total_mb": int(parts[3]),
                        "util_pct": int(parts[4]),
                        "temp_c": int(parts[5]),
                    }
                )
            except (ValueError, IndexError):
                continue
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_results_dir(results_dir: str) -> List[Dict[str, Any]]:
    """Scan a directory for final_results.json files."""
    p = Path(results_dir)
    if not p.exists():
        return []
    files = []
    for f in sorted(p.rglob("final_results.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["_source_file"] = str(f)
            files.append(data)
        except Exception:
            continue
    return files


def _pct(n: float) -> str:
    return f"{n * 100:.1f}%"


def _suite_avg(state: Dict[str, Any], suite: str) -> float:
    vals = state.get("suite_accuracies", {}).get(suite, [])
    return sum(vals) / len(vals) if vals else 0.0


def _elapsed_str(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _eta_str(state: Dict[str, Any], elapsed: Optional[float]) -> str:
    total = state.get("total_tasks", 0)
    done = state.get("completed_tasks", 0) + state.get("failed_tasks", 0)
    if elapsed is None or done == 0 or total == 0:
        return "—"
    rate = done / elapsed  # tasks/s
    remaining = (total - done) / rate
    return _elapsed_str(remaining)


# ---------------------------------------------------------------------------
# GPU monitoring background thread
# ---------------------------------------------------------------------------

class _GPUMonitor:
    """Background thread that polls GPU stats and pushes to bridge."""

    def __init__(self, bridge: DataBridge, interval: float = 5.0):
        self._bridge = bridge
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-monitor")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            stats = _query_gpu_stats()
            if stats:
                self._bridge.update_gpu_stats(stats)
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------

def create_dashboard(
    data_bridge: Optional[DataBridge] = None,
    results_dir: Optional[str] = None,
    poll_interval: float = 2.0,
) -> gr.Blocks:
    """
    Build and return a gr.Blocks dashboard.

    Args:
        data_bridge: Live DataBridge (for real-time mode).  If None, static viewer.
        results_dir:  Optional path to a directory of past results to pre-populate.
        poll_interval: Seconds between UI refreshes.

    Returns:
        gr.Blocks instance (not yet launched).
    """
    # If a bridge is provided, also start a GPU monitor that feeds into it
    gpu_monitor: Optional[_GPUMonitor] = None
    if data_bridge is not None:
        gpu_monitor = _GPUMonitor(data_bridge)
        gpu_monitor.start()

    # Pre-load static results if a dir was given
    static_results: List[Dict[str, Any]] = []
    if results_dir:
        static_results = _load_results_dir(results_dir)

    # ------------------------------------------------------------------ #
    # Helper: get current state snapshot
    # ------------------------------------------------------------------ #
    def _get_state() -> Dict[str, Any]:
        if data_bridge is not None:
            return data_bridge.get_state()
        # Static mode: synthesize a completed state from the first result file
        if static_results:
            r = static_results[0]
            summary = r.get("summary", {})
            task_results = r.get("task_results", {})
            task_rows = {}
            for tn, tr in task_results.items():
                acc = 0.0
                if isinstance(tr, list) and tr:
                    acc = sum(m.get("accuracy", 0) for m in tr) / len(tr)
                elif isinstance(tr, dict):
                    acc = tr.get("summary", {}).get("avg_accuracy", 0) or tr.get("overall_accuracy", 0)
                task_rows[tn] = {"accuracy": acc, "suite": "easy", "tokens_in": 0, "tokens_out": 0}
            return {
                "status": "completed",
                "model_info": r.get("model_info", {}),
                "total_tasks": summary.get("total_tasks", len(task_rows)),
                "completed_tasks": summary.get("completed_tasks", len(task_rows)),
                "failed_tasks": summary.get("failed_tasks", 0),
                "current_task": None,
                "start_time": None,
                "end_time": None,
                "task_results": task_rows,
                "suite_accuracies": {"easy": [], "medium": [], "hard": []},
                "log_lines": [],
                "gpu_stats": _query_gpu_stats(),
                "token_stats": {"total_input": 0, "total_output": 0, "per_task": {}},
            }
        return {
            "status": "idle",
            "model_info": {},
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_task": None,
            "start_time": None,
            "end_time": None,
            "task_results": {},
            "suite_accuracies": {"easy": [], "medium": [], "hard": []},
            "log_lines": [],
            "gpu_stats": _query_gpu_stats(),
            "token_stats": {"total_input": 0, "total_output": 0, "per_task": {}},
        }

    # ------------------------------------------------------------------ #
    # Per-tab refresh functions
    # ------------------------------------------------------------------ #

    def refresh_overview():
        state = _get_state()
        total = state["total_tasks"]
        done = state["completed_tasks"]
        failed = state["failed_tasks"]
        progress_pct = done / max(1, total)

        elapsed = (data_bridge.elapsed_seconds() if data_bridge else None)
        status_color = {
            "idle": "gray",
            "running": "green",
            "completed": "blue",
            "failed": "red",
        }.get(state["status"], "gray")
        status_md = f"**Status:** <span style='color:{status_color}'>{state['status'].upper()}</span>"

        model_info = state["model_info"]
        model_md = "\n".join(
            f"- **{k}:** {v}" for k, v in model_info.items() if v
        ) or "_No model info_"

        easy_acc = _suite_avg(state, "easy")
        med_acc = _suite_avg(state, "medium")
        hard_acc = _suite_avg(state, "hard")

        overview_md = (
            f"### Evaluation Progress\n\n"
            f"{status_md}  \n"
            f"**Tasks:** {done} / {total} completed, {failed} failed  \n"
            f"**Elapsed:** {_elapsed_str(elapsed)}  \n"
            f"**ETA:** {_eta_str(state, elapsed)}\n\n"
            f"---\n"
            f"### Suite Accuracies\n\n"
            f"- **Easy:** {_pct(easy_acc)}\n"
            f"- **Medium:** {_pct(med_acc)}\n"
            f"- **Hard:** {_pct(hard_acc)}\n\n"
            f"---\n"
            f"### Model Info\n\n{model_md}"
        )
        return overview_md, progress_pct

    def refresh_task_table(suite_filter: str, status_filter: str):
        state = _get_state()
        task_results = state["task_results"]
        rows = []
        for task_name, info in task_results.items():
            suite = info.get("suite", "easy")
            accuracy = info.get("accuracy", 0.0)
            failed = info.get("failed", False)

            if suite_filter != "all" and suite != suite_filter:
                continue
            if status_filter == "completed" and failed:
                continue
            if status_filter == "failed" and not failed:
                continue

            acc_pct = f"{accuracy * 100:.1f}%"
            parse_sr = f"{info.get('parse_success_rate', 1.0) * 100:.1f}%"
            latency = f"{info.get('latency_s', 0.0):.2f}s"
            tokens = str(info.get("tokens_in", 0) + info.get("tokens_out", 0))
            status = "FAILED" if failed else "OK"

            rows.append([task_name, suite.upper(), acc_pct, parse_sr, latency, tokens, status])

        if not rows:
            return [["—", "—", "—", "—", "—", "—", "—"]]
        return sorted(rows, key=lambda r: r[2], reverse=True)

    def refresh_gpu_monitor():
        state = _get_state()
        gpu_stats = state.get("gpu_stats") or _query_gpu_stats()
        if not gpu_stats:
            return "No GPU data available (nvidia-smi not found or no GPUs)", []

        lines = []
        rows = []
        for g in gpu_stats:
            used = g["mem_used_mb"]
            total = g["mem_total_mb"]
            util = g["util_pct"]
            temp = g["temp_c"]
            mem_pct = used / max(1, total) * 100
            alert = " ⚠️ HIGH" if mem_pct > 90 else ""
            lines.append(
                f"**GPU {g['index']} — {g['name']}**  \n"
                f"  Memory: {used} / {total} MB ({mem_pct:.1f}%){alert}  \n"
                f"  Utilization: {util}%  |  Temperature: {temp}°C"
            )
            rows.append([
                str(g["index"]),
                g["name"],
                f"{used}/{total} MB",
                f"{mem_pct:.1f}%",
                f"{util}%",
                f"{temp}°C",
            ])

        md = "\n\n".join(lines)
        return md, rows

    def refresh_token_analytics():
        state = _get_state()
        ts = state.get("token_stats", {})
        total_in = ts.get("total_input", 0)
        total_out = ts.get("total_output", 0)
        per_task = ts.get("per_task", {})
        elapsed = data_bridge.elapsed_seconds() if data_bridge else None

        tps = (total_in + total_out) / max(1, elapsed) if elapsed else 0

        summary_md = (
            f"### Token Summary\n\n"
            f"- **Total Input Tokens:** {total_in:,}\n"
            f"- **Total Output Tokens:** {total_out:,}\n"
            f"- **Total Tokens:** {total_in + total_out:,}\n"
            f"- **Throughput:** {tps:.1f} tokens/s\n"
        )

        # Build per-task table
        rows = []
        for task_name, td in per_task.items():
            rows.append([task_name, str(td.get("input", 0)), str(td.get("output", 0))])
        if not rows:
            rows = [["—", "—", "—"]]

        # Build chart data for gr.BarPlot
        chart_data = []
        for task_name, td in per_task.items():
            chart_data.append({"task": task_name, "type": "input", "tokens": td.get("input", 0)})
            chart_data.append({"task": task_name, "type": "output", "tokens": td.get("output", 0)})

        return summary_md, rows, chart_data

    def refresh_log(level_filter: str, search_query: str):
        state = _get_state()
        log_lines = state.get("log_lines", [])
        level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        min_level = level_order.get(level_filter.upper(), 0)

        filtered = []
        for ts, level, msg in log_lines:
            if level_order.get(level.upper(), 0) < min_level:
                continue
            if search_query and search_query.lower() not in msg.lower():
                continue
            t_str = time.strftime("%H:%M:%S", time.localtime(ts))
            filtered.append(f"[{t_str}] [{level}] {msg}")

        # Return last 200 lines
        return "\n".join(filtered[-200:]) if filtered else "No log entries yet."

    # ------------------------------------------------------------------ #
    # Compare mode helpers
    # ------------------------------------------------------------------ #

    def load_compare_files(file1_path: str, file2_path: str):
        """Load two result files and return comparison table rows."""
        rows = []
        try:
            with open(file1_path, "r") as f:
                d1 = json.load(f)
            with open(file2_path, "r") as f:
                d2 = json.load(f)
        except Exception as e:
            return [[f"Error loading files: {e}", "", "", ""]]

        tr1 = d1.get("task_results", {})
        tr2 = d2.get("task_results", {})

        def _acc(tr, name):
            r = tr.get(name, {})
            if isinstance(r, list) and r:
                return sum(m.get("accuracy", 0) for m in r) / len(r)
            if isinstance(r, dict):
                return r.get("summary", {}).get("avg_accuracy", 0) or r.get("overall_accuracy", 0)
            return 0.0

        all_tasks = sorted(set(list(tr1.keys()) + list(tr2.keys())))
        for tn in all_tasks:
            a1 = _acc(tr1, tn)
            a2 = _acc(tr2, tn)
            delta = a2 - a1
            sign = "+" if delta > 0 else ""
            rows.append([tn, f"{a1 * 100:.1f}%", f"{a2 * 100:.1f}%", f"{sign}{delta * 100:.1f}%"])
        return rows if rows else [["—", "—", "—", "—"]]

    # ------------------------------------------------------------------ #
    # Build the Gradio UI
    # ------------------------------------------------------------------ #

    # Gradio 6+ moved theme/css from Blocks() to launch(); detect and adapt
    _blocks_kwargs = {"title": "BeyondBench Dashboard"}
    _launch_extras = {}
    _major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 4
    if _major >= 6:
        _launch_extras["theme"] = gr.themes.Soft()
        _launch_extras["css"] = (
            ".progress-bar-container { margin-top: 8px; } "
            ".status-badge { font-size: 1.1em; font-weight: bold; }"
        )
    else:
        _blocks_kwargs["theme"] = gr.themes.Soft()
        _blocks_kwargs["css"] = (
            ".progress-bar-container { margin-top: 8px; } "
            ".status-badge { font-size: 1.1em; font-weight: bold; }"
        )

    with gr.Blocks(**_blocks_kwargs) as demo:
        # Stash launch extras so callers can pass them to launch()
        demo._beyondbench_launch_extras = _launch_extras

        gr.Markdown("# BeyondBench Live Observability Dashboard")
        gr.Markdown("_Real-time monitoring of LLM evaluation runs_")

        with gr.Tabs():
            # ----------------------------------------------------------------
            # TAB 1: Overview
            # ----------------------------------------------------------------
            with gr.Tab("Overview"):
                overview_md = gr.Markdown("_Loading..._")
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.001,
                    label="Overall Progress",
                    interactive=False,
                )

            # ----------------------------------------------------------------
            # TAB 2: Per-Task Results
            # ----------------------------------------------------------------
            with gr.Tab("Per-Task Results"):
                with gr.Row():
                    suite_filter = gr.Dropdown(
                        choices=["all", "easy", "medium", "hard"],
                        value="all",
                        label="Filter by Suite",
                    )
                    status_filter = gr.Dropdown(
                        choices=["all", "completed", "failed"],
                        value="all",
                        label="Filter by Status",
                    )

                task_table = gr.Dataframe(
                    headers=["Task", "Suite", "Accuracy", "Parse Success", "Avg Latency", "Total Tokens", "Status"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    label="Task Results (sorted by accuracy)",
                    interactive=False,
                    wrap=True,
                )

            # ----------------------------------------------------------------
            # TAB 3: GPU Monitor
            # ----------------------------------------------------------------
            with gr.Tab("GPU Monitor"):
                gpu_md = gr.Markdown("_Loading GPU info..._")
                gpu_table = gr.Dataframe(
                    headers=["GPU", "Name", "Memory", "Mem %", "Utilization", "Temperature"],
                    datatype=["str"] * 6,
                    label="GPU Stats",
                    interactive=False,
                )

            # ----------------------------------------------------------------
            # TAB 4: Token Analytics
            # ----------------------------------------------------------------
            with gr.Tab("Token Analytics"):
                token_summary_md = gr.Markdown("_Loading token stats..._")
                token_table = gr.Dataframe(
                    headers=["Task", "Input Tokens", "Output Tokens"],
                    datatype=["str", "str", "str"],
                    label="Per-Task Token Breakdown",
                    interactive=False,
                )

            # ----------------------------------------------------------------
            # TAB 5: Live Log
            # ----------------------------------------------------------------
            with gr.Tab("Live Log"):
                with gr.Row():
                    log_level_filter = gr.Dropdown(
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        value="INFO",
                        label="Min Log Level",
                    )
                    log_search = gr.Textbox(
                        label="Search",
                        placeholder="Filter log lines...",
                        max_lines=1,
                    )
                log_output = gr.Textbox(
                    label="Log Output",
                    lines=30,
                    max_lines=30,
                    interactive=False,
                    autoscroll=True,
                )
                download_btn = gr.Button("Download Log")

            # ----------------------------------------------------------------
            # TAB 6: Compare Results
            # ----------------------------------------------------------------
            with gr.Tab("Compare Results"):
                gr.Markdown("### Compare Two Evaluation Runs")
                with gr.Row():
                    compare_file1 = gr.Textbox(label="Result File A (path to final_results.json)")
                    compare_file2 = gr.Textbox(label="Result File B (path to final_results.json)")
                compare_btn = gr.Button("Compare", variant="primary")
                compare_table = gr.Dataframe(
                    headers=["Task", "A Accuracy", "B Accuracy", "Delta"],
                    datatype=["str", "str", "str", "str"],
                    label="Accuracy Comparison",
                    interactive=False,
                )

        # ----------------------------------------------------------------
        # Auto-refresh timer (every poll_interval seconds)
        # ----------------------------------------------------------------
        timer = gr.Timer(value=poll_interval)

        # Overview refresh
        timer.tick(
            fn=refresh_overview,
            outputs=[overview_md, progress_bar],
        )

        # Task table refresh (also reacts to filter dropdowns)
        timer.tick(
            fn=refresh_task_table,
            inputs=[suite_filter, status_filter],
            outputs=[task_table],
        )
        suite_filter.change(fn=refresh_task_table, inputs=[suite_filter, status_filter], outputs=[task_table])
        status_filter.change(fn=refresh_task_table, inputs=[suite_filter, status_filter], outputs=[task_table])

        # GPU monitor refresh (every tick)
        timer.tick(
            fn=refresh_gpu_monitor,
            outputs=[gpu_md, gpu_table],
        )

        # Token analytics refresh
        def _token_refresh():
            md, rows, _ = refresh_token_analytics()
            return md, rows

        timer.tick(fn=_token_refresh, outputs=[token_summary_md, token_table])

        # Log refresh (also reacts to filter/search changes)
        timer.tick(
            fn=refresh_log,
            inputs=[log_level_filter, log_search],
            outputs=[log_output],
        )
        log_level_filter.change(fn=refresh_log, inputs=[log_level_filter, log_search], outputs=[log_output])
        log_search.change(fn=refresh_log, inputs=[log_level_filter, log_search], outputs=[log_output])

        # Log download
        def _make_log_text():
            state = _get_state()
            lines = state.get("log_lines", [])
            result = []
            for ts, level, msg in lines:
                t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                result.append(f"[{t_str}] [{level}] {msg}")
            return "\n".join(result)

        download_btn.click(fn=_make_log_text, outputs=[log_output])

        # Compare results
        compare_btn.click(
            fn=lambda f1, f2: load_compare_files(f1, f2),
            inputs=[compare_file1, compare_file2],
            outputs=[compare_table],
        )

        # ----------------------------------------------------------------
        # Initial load (populate on first render)
        # ----------------------------------------------------------------
        demo.load(fn=refresh_overview, outputs=[overview_md, progress_bar])
        demo.load(fn=lambda: refresh_task_table("all", "all"), outputs=[task_table])
        demo.load(fn=refresh_gpu_monitor, outputs=[gpu_md, gpu_table])
        demo.load(fn=_token_refresh, outputs=[token_summary_md, token_table])
        demo.load(fn=lambda: refresh_log("INFO", ""), outputs=[log_output])

    return demo

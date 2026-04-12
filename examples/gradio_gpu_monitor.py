"""
BeyondBench GPU Monitor
========================
Real-time GPU statistics for all available GPUs with memory usage timeline,
temperature, utilization, and configurable alert thresholds.

Run with:
    python examples/gradio_gpu_monitor.py

Requirements:
    pip install gradio
    pip install beyondbench  # or: pip install -e .
"""

from __future__ import annotations

import collections
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# nvidia-smi helpers
# ---------------------------------------------------------------------------

MAX_HISTORY = 120  # data points to retain per GPU (~2 min at 1s interval)


# Per-GPU rolling history: {gpu_idx: deque of {ts, mem_used_mb, util_pct, temp_c}}
_history: Dict[int, collections.deque] = {}


def _query_gpu_stats() -> List[Dict[str, Any]]:
    """Query all GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                "temperature.gpu,power.draw,power.limit",
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
                power_draw = float(parts[6]) if len(parts) > 6 and parts[6] not in ("N/A", "[N/A]") else 0.0
                power_limit = float(parts[7]) if len(parts) > 7 and parts[7] not in ("N/A", "[N/A]") else 0.0
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_used_mb": int(parts[2]),
                    "mem_total_mb": int(parts[3]),
                    "util_pct": int(parts[4]),
                    "temp_c": int(parts[5]),
                    "power_draw_w": power_draw,
                    "power_limit_w": power_limit,
                })
            except (ValueError, IndexError):
                continue
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _update_history(gpus: List[Dict[str, Any]]) -> None:
    ts = time.time()
    for gpu in gpus:
        idx = gpu["index"]
        if idx not in _history:
            _history[idx] = collections.deque(maxlen=MAX_HISTORY)
        _history[idx].append({
            "ts": ts,
            "mem_used_mb": gpu["mem_used_mb"],
            "util_pct": gpu["util_pct"],
            "temp_c": gpu["temp_c"],
        })


def _format_bar(value: float, total: float, width: int = 20) -> str:
    """Return an ASCII progress bar."""
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(int(value / total * width), width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:.0f}/{total:.0f}"


def _mem_pct(gpu: Dict[str, Any]) -> float:
    if gpu["mem_total_mb"] <= 0:
        return 0.0
    return gpu["mem_used_mb"] / gpu["mem_total_mb"] * 100


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def _check_alerts(gpus: List[Dict[str, Any]], mem_thresh: float, util_thresh: float, temp_thresh: float) -> List[str]:
    alerts = []
    for gpu in gpus:
        idx = gpu["index"]
        mem_p = _mem_pct(gpu)
        if mem_p >= mem_thresh:
            alerts.append(f"GPU {idx}: Memory {mem_p:.0f}% >= threshold {mem_thresh:.0f}%")
        if gpu["util_pct"] >= util_thresh:
            alerts.append(f"GPU {idx}: Utilization {gpu['util_pct']}% >= threshold {util_thresh:.0f}%")
        if gpu["temp_c"] >= temp_thresh:
            alerts.append(f"GPU {idx}: Temperature {gpu['temp_c']}°C >= threshold {temp_thresh:.0f}°C")
    return alerts


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def refresh_stats(mem_thresh: float, util_thresh: float, temp_thresh: float):
    """Fetch current GPU stats and return UI updates."""
    gpus = _query_gpu_stats()
    _update_history(gpus)

    if not gpus:
        no_gpu_rows = [["—", "No GPUs detected", "—", "—", "—", "—", "—", "—"]]
        return (
            no_gpu_rows,
            "No nvidia-smi GPUs detected. Check your CUDA installation.",
            _build_mem_timeline_plot(),
            _build_util_timeline_plot(),
        )

    # Build status table
    rows = []
    for gpu in gpus:
        mem_p = _mem_pct(gpu)
        mem_str = f"{gpu['mem_used_mb']}MB / {gpu['mem_total_mb']}MB ({mem_p:.0f}%)"
        pow_str = (
            f"{gpu['power_draw_w']:.0f}W / {gpu['power_limit_w']:.0f}W"
            if gpu["power_limit_w"] > 0
            else f"{gpu['power_draw_w']:.0f}W"
        )
        rows.append([
            f"GPU {gpu['index']}",
            gpu["name"],
            mem_str,
            f"{gpu['util_pct']}%",
            f"{gpu['temp_c']}°C",
            pow_str,
            _status_icon(mem_p, gpu["util_pct"], gpu["temp_c"], mem_thresh, util_thresh, temp_thresh),
            time.strftime("%H:%M:%S"),
        ])

    # Alerts
    alerts = _check_alerts(gpus, mem_thresh, util_thresh, temp_thresh)
    alert_text = "\n".join(alerts) if alerts else "No alerts."

    return (
        rows,
        alert_text,
        _build_mem_timeline_plot(),
        _build_util_timeline_plot(),
    )


def _status_icon(mem_p, util_p, temp_c, mem_thresh, util_thresh, temp_thresh) -> str:
    if mem_p >= mem_thresh or util_p >= util_thresh or temp_c >= temp_thresh:
        return "ALERT"
    if mem_p >= mem_thresh * 0.8 or util_p >= util_thresh * 0.8 or temp_c >= temp_thresh * 0.8:
        return "WARN"
    return "OK"


def _build_mem_timeline_plot():
    """Build memory usage timeline as a Gradio-compatible line plot dataset."""
    if not _history:
        return None
    try:
        import numpy as np
        # Build {gpu_idx: [values]}
        datasets = {}
        for idx, hist in sorted(_history.items()):
            datasets[f"GPU {idx}"] = [h["mem_used_mb"] for h in hist]
        # Normalize lengths
        max_len = max(len(v) for v in datasets.values())
        # Create x-axis (seconds ago)
        xs = list(range(-max_len + 1, 1))

        # Return as list of (x, y, label) tuples — Gradio LinePlot compatible
        data = []
        for label, vals in datasets.items():
            # Pad shorter histories
            pad = max_len - len(vals)
            padded_x = xs[pad:]
            for xi, yi in zip(padded_x, vals):
                data.append({"t": xi, "mem_mb": yi, "gpu": label})
        return data
    except Exception:
        return None


def _build_util_timeline_plot():
    """Build utilization timeline."""
    if not _history:
        return None
    try:
        datasets = {}
        for idx, hist in sorted(_history.items()):
            datasets[f"GPU {idx}"] = [h["util_pct"] for h in hist]
        max_len = max(len(v) for v in datasets.values())
        xs = list(range(-max_len + 1, 1))
        data = []
        for label, vals in datasets.items():
            pad = max_len - len(vals)
            padded_x = xs[pad:]
            for xi, yi in zip(padded_x, vals):
                data.append({"t": xi, "util_pct": yi, "gpu": label})
        return data
    except Exception:
        return None


def get_gpu_summary() -> str:
    """Quick text summary for the header."""
    gpus = _query_gpu_stats()
    if not gpus:
        return "No GPUs detected."
    parts = []
    for gpu in gpus:
        mp = _mem_pct(gpu)
        parts.append(
            f"**GPU {gpu['index']}** ({gpu['name'].strip()}): "
            f"Mem {mp:.0f}% | Util {gpu['util_pct']}% | Temp {gpu['temp_c']}°C"
        )
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="BeyondBench — GPU Monitor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# BeyondBench GPU Monitor\nReal-time GPU statistics with alerts and memory timeline.")

        with gr.Row():
            mem_thresh_sl = gr.Slider(
                label="Memory Alert Threshold (%)",
                minimum=50, maximum=100, step=5, value=90,
            )
            util_thresh_sl = gr.Slider(
                label="Utilization Alert Threshold (%)",
                minimum=50, maximum=100, step=5, value=95,
            )
            temp_thresh_sl = gr.Slider(
                label="Temperature Alert Threshold (°C)",
                minimum=50, maximum=100, step=5, value=85,
            )

        manual_refresh_btn = gr.Button("Refresh Now", variant="secondary")

        gpu_table = gr.Dataframe(
            label="GPU Status",
            headers=["GPU", "Model", "Memory", "Utilization", "Temp", "Power", "Status", "Updated"],
            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        alert_box = gr.Textbox(
            label="Alerts",
            lines=4,
            interactive=False,
            value="No alerts.",
        )

        gr.Markdown("### Memory Usage Timeline (MB)")
        mem_plot = gr.LinePlot(
            x="t",
            y="mem_mb",
            color="gpu",
            x_title="Seconds ago",
            y_title="Memory (MB)",
            tooltip=["gpu", "t", "mem_mb"],
            height=250,
        )

        gr.Markdown("### GPU Utilization Timeline (%)")
        util_plot = gr.LinePlot(
            x="t",
            y="util_pct",
            color="gpu",
            x_title="Seconds ago",
            y_title="Utilization (%)",
            y_lim=[0, 100],
            tooltip=["gpu", "t", "util_pct"],
            height=250,
        )

        # Auto-refresh timer (every 2 s)
        timer = gr.Timer(2.0)

        def _refresh(mt, ut, tt):
            return refresh_stats(mt, ut, tt)

        timer.tick(
            fn=_refresh,
            inputs=[mem_thresh_sl, util_thresh_sl, temp_thresh_sl],
            outputs=[gpu_table, alert_box, mem_plot, util_plot],
        )
        manual_refresh_btn.click(
            fn=_refresh,
            inputs=[mem_thresh_sl, util_thresh_sl, temp_thresh_sl],
            outputs=[gpu_table, alert_box, mem_plot, util_plot],
        )

        # Trigger one refresh on page load
        demo.load(
            fn=_refresh,
            inputs=[mem_thresh_sl, util_thresh_sl, temp_thresh_sl],
            outputs=[gpu_table, alert_box, mem_plot, util_plot],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7864, share=False)

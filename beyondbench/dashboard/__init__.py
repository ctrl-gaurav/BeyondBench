"""
BeyondBench Live Observability Dashboard

Optional Gradio-based real-time monitoring dashboard for evaluations.
Install with: pip install beyondbench[dashboard]
"""

__all__ = ["DataBridge", "DashboardEvent", "EventType"]


def _check_gradio():
    """Check if Gradio is available."""
    try:
        import gradio  # noqa: F401
        return True
    except ImportError:
        return False


def launch_dashboard(data_bridge=None, port: int = 7860, results_dir: str = None, share: bool = False):
    """Launch the Gradio dashboard.

    Args:
        data_bridge: DataBridge instance for live evaluation data (or None for static viewer)
        port: Port to serve dashboard on
        results_dir: Directory with past results (for standalone viewer mode)
        share: Create a public Gradio share link

    Raises:
        ImportError: If gradio is not installed
    """
    if not _check_gradio():
        raise ImportError(
            "Gradio is required for the dashboard. Install with:\n"
            "  pip install beyondbench[dashboard]"
        )
    from .app import create_dashboard
    app = create_dashboard(data_bridge=data_bridge, results_dir=results_dir)
    launch_kwargs = getattr(app, '_beyondbench_launch_extras', {})
    app.launch(server_port=port, share=share, prevent_thread_lock=True, **launch_kwargs)
    return app


from .data_bridge import DataBridge, DashboardEvent, EventType  # noqa: E402

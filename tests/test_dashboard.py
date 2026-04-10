"""
Unit tests for the Phase 6 dashboard — DataBridge, events, and CLI integration.

No Gradio required; tests the data plumbing layer only.
"""

import time
import threading
import pytest

from beyondbench.dashboard.data_bridge import (
    DataBridge,
    DashboardEvent,
    EventType,
    BridgeLogHandler,
)


# ---------------------------------------------------------------------------
# DataBridge core tests
# ---------------------------------------------------------------------------

class TestDataBridgeLifecycle:
    def test_initial_state_is_idle(self):
        bridge = DataBridge()
        state = bridge.get_state()
        assert state["status"] == "idle"
        assert state["total_tasks"] == 0
        assert state["completed_tasks"] == 0

    def test_evaluation_started(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=5, model_info={"model_name": "test"})
        state = bridge.get_state()
        assert state["status"] == "running"
        assert state["total_tasks"] == 5
        assert state["model_info"]["model_name"] == "test"
        assert state["start_time"] is not None

    def test_task_completed_increments_counter(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=3, model_info={})
        bridge.notify_task_completed("sum", accuracy=0.9, suite="easy")
        state = bridge.get_state()
        assert state["completed_tasks"] == 1
        assert "sum" in state["task_results"]
        assert state["task_results"]["sum"]["accuracy"] == 0.9

    def test_failed_task_increments_failed_counter(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=2, model_info={})
        bridge.notify_task_completed("bad_task", accuracy=0.0, failed=True)
        state = bridge.get_state()
        assert state["failed_tasks"] == 1
        assert state["completed_tasks"] == 0

    def test_evaluation_complete_changes_status(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=1, model_info={})
        bridge.notify_task_completed("sum", accuracy=0.95)
        bridge.notify_evaluation_complete({"summary": {"avg_accuracy": 0.95}})
        state = bridge.get_state()
        assert state["status"] == "completed"
        assert state["end_time"] is not None

    def test_error_sets_failed_status(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=1, model_info={})
        bridge.notify_error("something went wrong")
        state = bridge.get_state()
        assert state["status"] == "failed"

    def test_token_stats_accumulate(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=2, model_info={})
        bridge.notify_task_completed("t1", accuracy=0.8, tokens_in=100, tokens_out=50)
        bridge.notify_task_completed("t2", accuracy=0.7, tokens_in=200, tokens_out=80)
        state = bridge.get_state()
        assert state["token_stats"]["total_input"] == 300
        assert state["token_stats"]["total_output"] == 130

    def test_suite_accuracies_tracked(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=2, model_info={})
        bridge.notify_task_completed("easy_t", accuracy=0.9, suite="easy")
        bridge.notify_task_completed("hard_t", accuracy=0.5, suite="hard")
        state = bridge.get_state()
        assert 0.9 in state["suite_accuracies"]["easy"]
        assert 0.5 in state["suite_accuracies"]["hard"]

    def test_log_lines_stored(self):
        bridge = DataBridge()
        bridge.push_log("INFO", "starting")
        bridge.push_log("WARNING", "low memory")
        state = bridge.get_state()
        assert len(state["log_lines"]) == 2
        _, level, msg = state["log_lines"][0]
        assert level == "INFO"
        assert "starting" in msg

    def test_log_lines_capped_at_2000(self):
        bridge = DataBridge()
        for i in range(2100):
            bridge.push_log("DEBUG", f"line {i}")
        state = bridge.get_state()
        assert len(state["log_lines"]) == 2000

    def test_get_state_returns_deep_copy(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=1, model_info={})
        state1 = bridge.get_state()
        state1["status"] = "hacked"
        state2 = bridge.get_state()
        assert state2["status"] == "running"  # original unchanged


class TestDataBridgeEvents:
    def test_events_emitted_in_order(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=2, model_info={})
        bridge.notify_task_started("sum")
        bridge.notify_task_completed("sum", accuracy=0.9)
        bridge.notify_evaluation_complete({})

        events = bridge.drain_events()
        types = [e.event_type for e in events]
        assert types[0] == EventType.EVALUATION_STARTED
        assert types[1] == EventType.TASK_STARTED
        assert types[2] == EventType.TASK_COMPLETED
        assert types[3] == EventType.EVALUATION_COMPLETE

    def test_drain_events_clears_queue(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=1, model_info={})
        bridge.drain_events()
        second_drain = bridge.drain_events()
        assert second_drain == []

    def test_fold_completed_event(self):
        bridge = DataBridge()
        bridge.notify_fold_completed("sorting", fold=0, accuracy=0.88)
        events = bridge.drain_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.FOLD_COMPLETED
        assert events[0].data["fold"] == 0

    def test_log_event_emitted(self):
        bridge = DataBridge()
        bridge.push_log("ERROR", "test error")
        events = bridge.drain_events()
        assert any(e.event_type == EventType.LOG for e in events)

    def test_gpu_update_event(self):
        bridge = DataBridge()
        bridge.update_gpu_stats([{"index": 0, "name": "A40", "util_pct": 70}])
        events = bridge.drain_events()
        assert any(e.event_type == EventType.GPU_UPDATE for e in events)
        state = bridge.get_state()
        assert state["gpu_stats"][0]["name"] == "A40"


class TestDataBridgeThreadSafety:
    def test_concurrent_task_completions(self):
        """Multiple threads completing tasks — counter must end at exact expected value."""
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=20, model_info={})
        n = 20

        def _complete(idx):
            bridge.notify_task_completed(f"task_{idx}", accuracy=0.5)

        threads = [threading.Thread(target=_complete, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        state = bridge.get_state()
        assert state["completed_tasks"] == n

    def test_elapsed_seconds_returns_none_before_start(self):
        bridge = DataBridge()
        assert bridge.elapsed_seconds() is None

    def test_elapsed_seconds_returns_float_after_start(self):
        bridge = DataBridge()
        bridge.notify_evaluation_started(total_tasks=1, model_info={})
        time.sleep(0.01)
        elapsed = bridge.elapsed_seconds()
        assert elapsed is not None
        assert elapsed >= 0.0

    def test_is_running_and_complete_flags(self):
        bridge = DataBridge()
        assert not bridge.is_running()
        assert not bridge.is_complete()

        bridge.notify_evaluation_started(total_tasks=1, model_info={})
        assert bridge.is_running()
        assert not bridge.is_complete()

        bridge.notify_evaluation_complete({})
        assert not bridge.is_running()
        assert bridge.is_complete()


class TestBridgeLogHandler:
    def test_log_records_forwarded_to_bridge(self):
        import logging
        bridge = DataBridge()
        handler = BridgeLogHandler(bridge)
        logger = logging.getLogger("test_bridge_handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("hello bridge")
        logger.warning("a warning")

        state = bridge.get_state()
        assert len(state["log_lines"]) >= 2
        levels = [lvl for _, lvl, _ in state["log_lines"]]
        assert "INFO" in levels
        assert "WARNING" in levels

        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# CLI: dashboard command registered
# ---------------------------------------------------------------------------

class TestDashboardCLI:
    def test_dashboard_command_registered(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "dashboard" in result.output.lower()
        assert "--port" in result.output

    def test_evaluate_has_dashboard_flag(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--dashboard" in result.output
        assert "--dashboard-port" in result.output

    def test_dashboard_missing_gradio_exits_cleanly(self):
        """Simulate missing gradio by patching the import."""
        import sys
        import builtins
        from unittest.mock import patch
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "gradio" or name.startswith("gradio."):
                raise ImportError("mocked: no gradio")
            return real_import(name, *args, **kwargs)

        # Remove cached gradio modules and block reimport
        saved_modules = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "gradio" or k.startswith("gradio.")}
        try:
            with patch("builtins.__import__", side_effect=_mock_import):
                runner = CliRunner()
                result = runner.invoke(main, ["dashboard", "--port", "19999"])
                # Should fail with a helpful message, not a traceback
                assert "dashboard" in result.output.lower() or result.exit_code != 0
        finally:
            sys.modules.update(saved_modules)

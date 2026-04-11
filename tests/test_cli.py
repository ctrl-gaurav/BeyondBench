"""Test CLI functionality."""

import pytest
from click.testing import CliRunner
from beyondbench.cli.main import main, cli


def _runner() -> CliRunner:
    """Return a CliRunner that keeps stdout/stderr separate.

    Several BeyondBench commands emit INFO logs to stderr during module
    import (e.g. the TaskRegistry's "44 tasks" banner). With older Click
    (<8.3) those lines would bleed into ``result.output`` unless we pass
    ``mix_stderr=False``. Click 8.3+ always keeps the streams separate
    and removed that keyword, so we feature-detect and adapt.
    """
    import inspect
    try:
        params = inspect.signature(CliRunner).parameters
    except (TypeError, ValueError):
        params = {}
    if "mix_stderr" in params:
        return CliRunner(mix_stderr=False)
    return CliRunner()


def _stdout(result) -> str:
    """Return clean stdout from a CliRunner result.

    Click 8.3 exposes ``result.stdout`` as the split stdout stream even
    though ``result.output`` still contains merged stdout+stderr. Older
    Click versions only have ``result.output`` (already split when the
    runner was constructed with ``mix_stderr=False``). Prefer stdout
    when present; fall back to output otherwise.
    """
    stdout = getattr(result, "stdout", None)
    if stdout:
        return stdout
    return result.output


class TestCLIStructure:
    def test_main_is_click_group(self):
        import click
        assert isinstance(main, click.core.Group)

    def test_evaluate_command_exists(self):
        assert "evaluate" in main.commands

    def test_list_tasks_command_exists(self):
        assert "list-tasks" in main.commands

    def test_wizard_command_exists(self):
        assert "wizard" in main.commands

    def test_serve_command_exists(self):
        assert "serve" in main.commands

    def test_run_config_command_exists(self):
        assert "run-config" in main.commands


class TestCLIHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "beyondbench" in result.output.lower()

    def test_evaluate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model-id" in result.output

    def test_list_tasks_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks", "--help"])
        assert result.exit_code == 0

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0


class TestListTasks:
    def test_list_all_tasks(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks"])
        assert result.exit_code == 0
        assert "EASY" in result.output.upper() or "easy" in result.output.lower()

    def test_list_easy_tasks(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks", "--suite", "easy"])
        assert result.exit_code == 0

    def test_list_tasks_json(self):
        runner = _runner()
        result = runner.invoke(main, ["list-tasks", "--format", "json"])
        assert result.exit_code == 0
        import json
        data = json.loads(_stdout(result).strip())
        assert "easy" in data


class TestEvaluateValidation:
    def test_evaluate_requires_model_id(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate"])
        assert result.exit_code != 0
        assert "model-id" in result.output.lower() or "required" in result.output.lower() or "missing" in result.output.lower() or "error" in result.output.lower()


class TestAutoDetection:
    """Test CLI auto-detection features."""

    def test_auto_detect_openai(self):
        from beyondbench.cli.main import _auto_detect_api_provider
        assert _auto_detect_api_provider("gpt-4o") == "openai"
        assert _auto_detect_api_provider("gpt-5-mini") == "openai"
        assert _auto_detect_api_provider("o1-preview") == "openai"

    def test_auto_detect_gemini(self):
        from beyondbench.cli.main import _auto_detect_api_provider
        assert _auto_detect_api_provider("gemini-2.5-pro") == "gemini"
        assert _auto_detect_api_provider("gemini-2.5-flash") == "gemini"

    def test_auto_detect_anthropic(self):
        from beyondbench.cli.main import _auto_detect_api_provider
        assert _auto_detect_api_provider("claude-sonnet-4-20250514") == "anthropic"
        assert _auto_detect_api_provider("claude-opus-4-20250514") == "anthropic"

    def test_auto_detect_local(self):
        from beyondbench.cli.main import _auto_detect_api_provider
        assert _auto_detect_api_provider("Qwen/Qwen2.5-1.5B-Instruct") is None
        assert _auto_detect_api_provider("meta-llama/Llama-3.2-3B") is None


class TestListSizesParsing:
    """Test list_sizes parsing with ranges."""

    def test_comma_separated(self):
        from beyondbench.cli.main import _parse_list_sizes
        assert _parse_list_sizes("8,16,32,64") == [8, 16, 32, 64]

    def test_range(self):
        from beyondbench.cli.main import _parse_list_sizes
        result = _parse_list_sizes("8-64")
        assert result == [8, 16, 32, 64]

    def test_range_non_power_of_2(self):
        from beyondbench.cli.main import _parse_list_sizes
        result = _parse_list_sizes("4-50")
        assert result == [4, 8, 16, 32]

    def test_single_value(self):
        from beyondbench.cli.main import _parse_list_sizes
        assert _parse_list_sizes("16") == [16]

    def test_mixed(self):
        from beyondbench.cli.main import _parse_list_sizes
        result = _parse_list_sizes("4,8-32,128")
        assert result == [4, 8, 16, 32, 128]

    def test_invalid_raises(self):
        import click
        from beyondbench.cli.main import _parse_list_sizes
        with pytest.raises(click.BadParameter):
            _parse_list_sizes("abc")

    def test_negative_raises(self):
        import click
        from beyondbench.cli.main import _parse_list_sizes
        with pytest.raises(click.BadParameter):
            _parse_list_sizes("-5")


class TestNoChatTemplateFlag:
    """Test --no-chat-template CLI flag."""

    def test_flag_in_evaluate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert "--no-chat-template" in result.output


class TestCLIEntry:
    def test_cli_function_exists(self):
        assert callable(cli)


# ---------------------------------------------------------------------- #
# Phase 10 polish — global flags, profile-model, Rich list-tasks, GPU util
# ---------------------------------------------------------------------- #

class TestGlobalFlags:
    """Tests for the top-level --verbose / --quiet / --json flags."""

    def test_help_shows_global_flags(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output
        assert "--quiet" in result.output
        assert "--json" in result.output

    def test_verbose_and_quiet_mutually_exclusive(self):
        runner = CliRunner()
        result = runner.invoke(main, ["-v", "-q", "list-tasks", "--suite", "hard"])
        # UsageError triggers exit code 2
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or "usage" in result.output.lower()

    def test_json_flag_forces_list_tasks_json(self):
        runner = _runner()
        result = runner.invoke(main, ["--json", "list-tasks", "--suite", "medium"])
        assert result.exit_code == 0, result.output
        import json
        # Output may contain trailing newline; json.loads handles whitespace
        data = json.loads(_stdout(result).strip())
        assert "medium" in data
        assert isinstance(data["medium"], list)
        assert len(data["medium"]) > 0

    def test_quiet_suppresses_info_logging(self):
        runner = _runner()
        result = runner.invoke(main, ["-q", "list-tasks", "--format", "json"])
        assert result.exit_code == 0
        # The `Task registry initialized` line is INFO — must not appear
        # (stderr is captured separately via mix_stderr=False)
        full = result.output + (result.stderr or "")
        assert "Task registry initialized" not in full

    def test_cli_context_log_level(self):
        from beyondbench.cli.main import CLIContext
        ctx = CLIContext()
        assert ctx.log_level == "INFO"
        ctx.verbose = True
        assert ctx.log_level == "DEBUG"
        ctx.verbose = False
        ctx.quiet = True
        assert ctx.log_level == "ERROR"


class TestProfileModelCommand:
    """Tests for the `beyondbench profile-model` subcommand."""

    def test_profile_model_command_exists(self):
        assert "profile-model" in main.commands

    def test_profile_model_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["profile-model", "--help"])
        assert result.exit_code == 0
        assert "--model-id" in result.output
        assert "--backend" in result.output
        assert "--force" in result.output
        assert "--no-save" in result.output

    def test_profile_model_requires_model_id(self):
        runner = CliRunner()
        result = runner.invoke(main, ["profile-model"])
        assert result.exit_code != 0
        assert "model-id" in result.output.lower() or "required" in result.output.lower()

    def test_print_profile_rich_renders(self):
        """_print_profile_rich should not raise on a synthetic ModelProfile."""
        from beyondbench.utils.model_profiler import ModelProfile
        from beyondbench.cli.main import _print_profile_rich, CLIContext
        p = ModelProfile(
            model_id="unit/test-model",
            boxed_rate=0.5, explicit_rate=0.3, code_block_rate=0.1,
            cot_rate=0.4, avg_response_length=200, avg_token_estimate=50,
            non_latin_rate=0.0,
            recommended_strategies=["boxed", "explicit_statement", "fallback"],
            quirks=[], num_calibration_samples=20,
        )
        # Should not raise with or without cli_ctx
        _print_profile_rich(p, cli_ctx=CLIContext())
        _print_profile_rich(p, cli_ctx=None)

    def test_profile_model_cached_json(self, tmp_path):
        """Saving a profile then loading it with --json should round-trip."""
        from beyondbench.utils.model_profiler import ModelProfile, ModelProfiler
        from dataclasses import asdict
        import json

        profile = ModelProfile(
            model_id="unit/cached-model",
            boxed_rate=0.6, explicit_rate=0.2, code_block_rate=0.05,
            cot_rate=0.3, avg_response_length=150, avg_token_estimate=37,
            non_latin_rate=0.0,
            recommended_strategies=["boxed", "explicit_statement", "fallback"],
            quirks=["test quirk"], num_calibration_samples=20,
        )
        # Save directly using the ModelProfiler helper (no handler needed).
        ModelProfiler.__init__  # just ensure import
        out_path = tmp_path / "unit_cached-model.json"
        out_path.write_text(json.dumps(asdict(profile), indent=2))

        # Invoke the CLI with --json; it should hit the cache and emit JSON.
        runner = _runner()
        result = runner.invoke(
            main,
            ["--json", "profile-model",
             "--model-id", "unit/cached-model",
             "--profile-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(_stdout(result).strip())
        assert data["cached"] is True
        assert data["profile"]["model_id"] == "unit/cached-model"
        assert data["profile"]["boxed_rate"] == pytest.approx(0.6)


class TestRichListTasks:
    """The upgraded list-tasks table should include description + difficulty."""

    def test_table_mentions_difficulty_label(self):
        runner = CliRunner()
        # Strip ANSI so we can do substring matches
        result = runner.invoke(main, ["list-tasks", "--suite", "hard"], color=False)
        assert result.exit_code == 0
        # The Rich table title renders as "HARD SUITE"
        assert "HARD SUITE" in result.output
        # Difficulty column label (not per-row) — "Difficulty" is a header
        # in the new table. Fallback mode doesn't have it, so test only
        # when rich is available.
        try:
            import rich  # noqa: F401
            assert "Difficulty" in result.output
            assert "Description" in result.output
        except ImportError:
            pytest.skip("rich not installed — fallback path used")


class TestGPUUtilSampler:
    """Tests for the GPU utilization sampler / progress postfix."""

    def test_sampler_available_is_bool(self):
        from beyondbench.core.result_aggregator import GPUUtilizationSampler
        s = GPUUtilizationSampler()
        assert isinstance(s.available, bool)

    def test_sample_once_returns_dict(self):
        from beyondbench.core.result_aggregator import GPUUtilizationSampler
        s = GPUUtilizationSampler()
        snap = s.sample_once()
        assert isinstance(snap, dict)
        # If sampler is unavailable (no nvidia-smi), expect empty dict
        if not s.available:
            assert snap == {}
        else:
            # Keys must be ints, values must have util/mem fields
            for gid, info in snap.items():
                assert isinstance(gid, int)
                assert "util" in info
                assert "mem_used_mib" in info
                assert "mem_total_mib" in info

    def test_summarize_empty(self):
        from beyondbench.core.result_aggregator import summarize_gpu_utilization
        assert summarize_gpu_utilization({}) == ""

    def test_summarize_non_empty(self):
        from beyondbench.core.result_aggregator import summarize_gpu_utilization
        snap = {
            0: {"util": 73, "mem_used_mib": 18542, "mem_total_mib": 24564},
            1: {"util": 5, "mem_used_mib": 4096, "mem_total_mib": 24564},
        }
        s = summarize_gpu_utilization(snap)
        assert "GPU0" in s
        assert "73%" in s
        assert "GPU1" in s

    def test_progress_gpu_postfix_env_disable(self, monkeypatch):
        """BEYONDBENCH_DISABLE_GPU_POSTFIX=1 must force-disable sampling."""
        from beyondbench.core import base_task as _bt
        monkeypatch.setenv("BEYONDBENCH_DISABLE_GPU_POSTFIX", "1")
        assert _bt._progress_gpu_postfix() == ""

    def test_progress_tracker_gpu_snapshot_shape(self):
        """ProgressTracker.gpu_snapshot() must return a dict (possibly empty)."""
        from beyondbench.core.result_aggregator import ProgressTracker
        pt = ProgressTracker([0], sample_gpu_util=True)
        snap = pt.gpu_snapshot()
        assert isinstance(snap, dict)

    def test_progress_tracker_sampler_opt_out(self):
        """sample_gpu_util=False should skip the sampler entirely."""
        from beyondbench.core.result_aggregator import ProgressTracker
        pt = ProgressTracker([0], sample_gpu_util=False)
        assert pt.gpu_snapshot() == {}


class TestDoctorJSON:
    """doctor --json must emit structured output and never leak Rich markup."""

    def test_doctor_json_mode(self):
        runner = _runner()
        result = runner.invoke(main, ["--json", "doctor"])
        # Exit code 0 or 1 (fail if any check failed) — both are acceptable
        assert result.exit_code in (0, 1)
        import json
        data = json.loads(_stdout(result).strip())
        assert "checks" in data
        assert "summary" in data
        assert "total" in data["summary"]
        assert "passed" in data["summary"]
        # Status values must be normalized words, not Rich markup
        for check in data["checks"]:
            assert check["status"] in ("PASS", "WARN", "FAIL")


class TestInfoJSON:
    """info --json must emit structured system information."""

    def test_info_json_mode(self):
        runner = _runner()
        result = runner.invoke(main, ["--json", "info"])
        assert result.exit_code == 0, result.output
        import json
        data = json.loads(_stdout(result).strip())
        # Expect at minimum package + platform + backends keys
        assert "package" in data
        assert "platform" in data
        assert "backends" in data
        # Ensure Rich markup didn't leak into the JSON payload
        payload_str = json.dumps(data)
        assert "[green]" not in payload_str
        assert "[/green]" not in payload_str
        assert "[red]" not in payload_str

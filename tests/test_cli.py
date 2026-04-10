"""Test CLI functionality."""

import pytest
from click.testing import CliRunner
from beyondbench.cli.main import main, cli


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
        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks", "--format", "json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
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

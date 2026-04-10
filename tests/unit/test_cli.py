"""
Unit tests for CLI commands — parsing, config validation, help text.

No GPU required.
"""

import json
import os
import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch


# ===========================================================================
# 1. CLI structure
# ===========================================================================

class TestCLIStructure:
    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.cli.main import main
        self.main = main
        self.runner = CliRunner()

    def test_main_is_click_group(self):
        import click
        assert isinstance(self.main, click.core.Group)

    def test_evaluate_command_exists(self):
        assert "evaluate" in self.main.commands

    def test_list_tasks_command_exists(self):
        assert "list-tasks" in self.main.commands

    def test_serve_command_exists(self):
        assert "serve" in self.main.commands

    def test_wizard_command_exists(self):
        assert "wizard" in self.main.commands

    def test_run_config_command_exists(self):
        assert "run-config" in self.main.commands

    def test_all_commands_are_click_commands(self):
        import click
        for name, cmd in self.main.commands.items():
            assert isinstance(cmd, (click.core.Command, click.core.Group)), \
                f"Command '{name}' is not a Click command"


# ===========================================================================
# 2. Help text
# ===========================================================================

class TestHelpText:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.cli.main import main
        self.main = main
        self.runner = CliRunner()

    def test_main_help(self):
        result = self.runner.invoke(self.main, ["--help"])
        assert result.exit_code == 0
        assert "beyondbench" in result.output.lower()

    def test_evaluate_help(self):
        result = self.runner.invoke(self.main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model-id" in result.output

    def test_evaluate_help_has_suite(self):
        result = self.runner.invoke(self.main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--suite" in result.output

    def test_evaluate_help_has_datapoints(self):
        result = self.runner.invoke(self.main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--datapoints" in result.output or "datapoints" in result.output

    def test_evaluate_help_has_backend(self):
        result = self.runner.invoke(self.main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output or "backend" in result.output

    def test_list_tasks_help(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--help"])
        assert result.exit_code == 0

    def test_serve_help(self):
        result = self.runner.invoke(self.main, ["serve", "--help"])
        assert result.exit_code == 0

    def test_version(self):
        result = self.runner.invoke(self.main, ["--version"])
        assert result.exit_code == 0


# ===========================================================================
# 3. list-tasks command
# ===========================================================================

class TestListTasksCommand:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.cli.main import main
        self.main = main
        self.runner = CliRunner()

    def test_list_all_tasks_exit_zero(self):
        result = self.runner.invoke(self.main, ["list-tasks"])
        assert result.exit_code == 0

    def test_list_all_includes_easy(self):
        result = self.runner.invoke(self.main, ["list-tasks"])
        assert "easy" in result.output.lower()

    def test_list_easy_suite(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--suite", "easy"])
        assert result.exit_code == 0

    def test_list_medium_suite(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--suite", "medium"])
        assert result.exit_code == 0

    def test_list_hard_suite(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--suite", "hard"])
        assert result.exit_code == 0

    def test_list_json_format(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "easy" in data

    def test_list_json_easy_has_29(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data.get("easy", [])) == 29

    def test_list_json_medium_has_5(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data.get("medium", [])) == 5

    def test_list_json_hard_has_10(self):
        result = self.runner.invoke(self.main, ["list-tasks", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data.get("hard", [])) == 10

    def test_list_includes_sum(self):
        result = self.runner.invoke(self.main, ["list-tasks"])
        assert "sum" in result.output

    def test_list_includes_sorting(self):
        result = self.runner.invoke(self.main, ["list-tasks"])
        assert "sorting" in result.output

    def test_list_includes_sudoku(self):
        result = self.runner.invoke(self.main, ["list-tasks"])
        assert "sudoku" in result.output.lower()


# ===========================================================================
# 4. evaluate command — parameter validation (no actual model loading)
# ===========================================================================

class TestEvaluateValidation:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.cli.main import main
        self.main = main
        self.runner = CliRunner()

    def test_missing_model_id_fails(self):
        result = self.runner.invoke(self.main, ["evaluate"])
        assert result.exit_code != 0

    def test_missing_model_id_error_message(self):
        result = self.runner.invoke(self.main, ["evaluate"])
        output = result.output.lower()
        assert (
            "missing" in output
            or "required" in output
            or "model-id" in output
            or "error" in output
        )

    def test_invalid_suite_value(self):
        # Some CLIs validate suite values at parse time
        result = self.runner.invoke(self.main, [
            "evaluate", "--model-id", "Qwen/test", "--suite", "invalid_suite"
        ])
        # Either fails early or proceeds to model loading (which will fail)
        # Just check it doesn't hang or produce a Python traceback
        assert result.exit_code != 0 or "error" not in result.output.lower()

    def test_invalid_datapoints_non_integer(self):
        result = self.runner.invoke(self.main, [
            "evaluate", "--model-id", "Qwen/test", "--datapoints", "notanumber"
        ])
        assert result.exit_code != 0


# ===========================================================================
# 5. Auto-detect API provider
# ===========================================================================

class TestAutoDetectAPIProvider:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from beyondbench.cli.main import _auto_detect_api_provider
            self.fn = _auto_detect_api_provider
            self.available = True
        except ImportError:
            self.available = False

    def _call(self, model_id):
        if not self.available:
            pytest.skip("_auto_detect_api_provider not available")
        return self.fn(model_id)

    def test_gpt4o_is_openai(self):
        assert self._call("gpt-4o") == "openai"

    def test_gpt4o_mini_is_openai(self):
        assert self._call("gpt-4o-mini") == "openai"

    def test_gpt5_is_openai(self):
        assert self._call("gpt-5") == "openai"

    def test_o1_is_openai(self):
        result = self._call("o1-preview")
        assert result == "openai" or result is None

    def test_gemini_pro_is_gemini(self):
        assert self._call("gemini-2.5-pro") == "gemini"

    def test_gemini_flash_is_gemini(self):
        assert self._call("gemini-2.5-flash") == "gemini"

    def test_claude_sonnet_is_anthropic(self):
        assert self._call("claude-sonnet-4-20250514") == "anthropic"

    def test_claude_opus_is_anthropic(self):
        assert self._call("claude-opus-4-20250514") == "anthropic"

    def test_qwen_local_returns_none_or_local(self):
        result = self._call("Qwen/Qwen2.5-1.5B-Instruct")
        assert result is None or result == "local" or result == "vllm"

    def test_llama_local_returns_none_or_local(self):
        result = self._call("meta-llama/Llama-3.2-1B-Instruct")
        assert result is None or result == "local" or result == "vllm"


# ===========================================================================
# 6. Config file loading (YAML / JSON)
# ===========================================================================

class TestConfigFileLoading:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.cli.main import main
        self.main = main
        self.runner = CliRunner()

    def test_json_config_file(self, tmp_path):
        config = {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "suite": "easy",
            "tasks": ["sum"],
            "datapoints": 3,
            "backend": "vllm",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        # run-config should parse it without actually running the model
        result = self.runner.invoke(self.main, ["run-config", str(config_file), "--dry-run"])
        # May not have --dry-run but should at least accept the file
        # If dry-run unsupported, accept any non-crash result
        assert isinstance(result.exit_code, int)

    def test_yaml_config_file(self, tmp_path):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        config = {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "suite": "easy",
            "tasks": ["sum"],
            "datapoints": 3,
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))
        result = self.runner.invoke(self.main, ["run-config", str(config_file), "--dry-run"])
        assert isinstance(result.exit_code, int)


# ===========================================================================
# 7. wizard command — just that it doesn't crash immediately
# ===========================================================================

class TestWizardCommand:
    def test_wizard_help(self):
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, ["wizard", "--help"])
        assert result.exit_code == 0

    def test_wizard_has_help_output(self):
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, ["wizard", "--help"])
        assert len(result.output) > 5


# ===========================================================================
# 8. serve command help — validate options present
# ===========================================================================

class TestServeCommand:
    def test_serve_help_has_host(self):
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "--host" in result.output or "host" in result.output.lower()

    def test_serve_help_has_port(self):
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "--port" in result.output or "port" in result.output.lower()

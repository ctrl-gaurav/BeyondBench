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


class TestCLIEntry:
    def test_cli_function_exists(self):
        assert callable(cli)

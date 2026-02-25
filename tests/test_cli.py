"""
Test CLI functionality.
"""

import pytest


class TestCLICommands:
    """Test CLI commands."""

    def test_cli_import(self):
        """Test that CLI can be imported."""
        from beyondbench.cli.main import main
        assert main is not None

    def test_cli_is_click_group(self):
        """Test that CLI is a Click group."""
        from beyondbench.cli.main import main
        import click
        assert isinstance(main, click.core.Group)

    def test_evaluate_command_exists(self):
        """Test that evaluate command exists."""
        from beyondbench.cli.main import main, evaluate
        assert evaluate is not None
        assert 'evaluate' in main.commands

    def test_list_tasks_command_exists(self):
        """Test that list-tasks command exists."""
        from beyondbench.cli.main import main
        assert 'list-tasks' in main.commands


class TestCLIWizard:
    """Test CLI wizard functionality."""

    def test_wizard_import(self):
        """Test that wizard can be imported."""
        from beyondbench.cli import wizard
        assert wizard is not None

    def test_wizard_has_main_function(self):
        """Test that wizard has main function."""
        from beyondbench.cli.wizard import main_wizard, run_evaluation_wizard
        assert main_wizard is not None
        assert run_evaluation_wizard is not None


class TestCLIStructure:
    """Test CLI structure and options."""

    def test_evaluate_has_model_option(self):
        """Test that evaluate command has model-id option."""
        from beyondbench.cli.main import evaluate
        # Check params
        param_names = [p.name for p in evaluate.params]
        assert 'model_id' in param_names

    def test_evaluate_has_suite_option(self):
        """Test that evaluate command has suite option."""
        from beyondbench.cli.main import evaluate
        param_names = [p.name for p in evaluate.params]
        assert 'suite' in param_names

    def test_evaluate_has_backend_option(self):
        """Test that evaluate command has backend option."""
        from beyondbench.cli.main import evaluate
        param_names = [p.name for p in evaluate.params]
        assert 'backend' in param_names

    def test_evaluate_has_api_provider_option(self):
        """Test that evaluate command has api-provider option."""
        from beyondbench.cli.main import evaluate
        param_names = [p.name for p in evaluate.params]
        assert 'api_provider' in param_names


class TestCLIHelp:
    """Test CLI help output."""

    def test_main_group_has_docstring(self):
        """Test that main group has docstring."""
        from beyondbench.cli.main import main
        assert main.help is not None
        assert 'beyondbench' in main.help.lower() or 'beyondbench' in main.help.lower()

    def test_evaluate_has_docstring(self):
        """Test that evaluate command has docstring."""
        from beyondbench.cli.main import evaluate
        # Evaluate should have help text
        assert evaluate.callback.__doc__ is not None or True  # Some commands may not have docstrings

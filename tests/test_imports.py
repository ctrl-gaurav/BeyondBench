"""
Test module imports and package structure.
"""

import pytest


class TestCoreImports:
    """Test that all core modules can be imported."""

    def test_import_beyondbench(self):
        """Test main package import."""
        import beyondbench
        assert hasattr(beyondbench, "__version__")

    def test_import_evaluation_engine(self):
        """Test EvaluationEngine import."""
        from beyondbench.core.evaluation_engine import EvaluationEngine
        assert EvaluationEngine is not None

    def test_import_task_registry(self):
        """Test TaskRegistry import."""
        from beyondbench.core.task_registry import TaskRegistry
        assert TaskRegistry is not None

    def test_import_base_task(self):
        """Test BaseTask import."""
        from beyondbench.core.base_task import BaseTask
        assert BaseTask is not None

    def test_import_model_handler(self):
        """Test ModelHandler import."""
        from beyondbench.models.model_handler import ModelHandler
        assert ModelHandler is not None


class TestParsingImports:
    """Test parsing module imports."""

    def test_import_utils_parsing(self):
        """Test utils.parsing import."""
        from beyondbench.utils import parsing
        assert hasattr(parsing, "parse_boxed_answer")
        assert hasattr(parsing, "extract_number")
        assert hasattr(parsing, "extract_list")

    def test_import_parsers_module(self):
        """Test parsers module import."""
        from beyondbench import parsers
        assert parsers is not None
        # Check for task-specific parsers
        assert hasattr(parsers, "parse_sum_answer")


class TestTaskImports:
    """Test task module imports."""

    def test_import_easy_tasks(self):
        """Test easy tasks module."""
        from beyondbench.tasks import easy
        assert easy is not None

    def test_import_medium_tasks(self):
        """Test medium tasks module."""
        from beyondbench.tasks import medium
        assert medium is not None

    def test_import_hard_tasks(self):
        """Test hard tasks module."""
        from beyondbench.tasks import hard
        assert hard is not None


class TestCLIImports:
    """Test CLI module imports."""

    def test_import_cli_main(self):
        """Test CLI main import."""
        from beyondbench.cli.main import main
        # main is a Click group
        import click
        assert isinstance(main, click.core.Group)

    def test_import_cli_wizard(self):
        """Test CLI wizard import."""
        from beyondbench.cli import wizard
        from beyondbench.cli.wizard import main_wizard
        assert wizard is not None
        assert main_wizard is not None

"""Test module imports and package structure."""

import pytest


class TestCoreImports:
    def test_import_package(self):
        import beyondbench
        assert hasattr(beyondbench, "__version__")
        assert hasattr(beyondbench, "EvaluationEngine")
        assert hasattr(beyondbench, "TaskRegistry")
        assert hasattr(beyondbench, "ModelHandler")

    def test_import_evaluation_engine(self):
        from beyondbench.core.evaluation_engine import EvaluationEngine
        assert EvaluationEngine is not None

    def test_import_task_registry(self):
        from beyondbench.core.task_registry import TaskRegistry
        assert TaskRegistry is not None

    def test_import_base_task(self):
        from beyondbench.core.base_task import BaseTask
        assert BaseTask is not None

    def test_import_model_handler(self):
        from beyondbench.models.model_handler import ModelHandler
        assert ModelHandler is not None


class TestParsingImports:
    def test_import_parsing(self):
        from beyondbench.utils.parsing import (
            parse_boxed_answer, extract_number, extract_list,
            parse_number, parse_number_list, parse_boolean, parse_grid,
            robust_extract_answer, extract_all_numbers, is_valid_number,
            validate_arithmetic_result,
        )

    def test_import_parsing_utils(self):
        from beyondbench.utils.parsing_utils import AnswerParser
        assert AnswerParser is not None

    def test_import_sequence_parsing(self):
        from beyondbench.utils.sequence_parsing_utils import SequenceAnswerParser
        assert SequenceAnswerParser is not None


class TestTaskImports:
    def test_import_easy_tasks(self):
        from beyondbench.tasks import easy
        assert easy is not None

    def test_import_medium_tasks(self):
        from beyondbench.tasks import medium
        assert medium is not None

    def test_import_hard_tasks(self):
        from beyondbench.tasks import hard
        assert hard is not None

    def test_import_specific_easy_task(self):
        from beyondbench.tasks.easy.sum_task import SumTask
        assert SumTask is not None

    def test_import_specific_medium_task(self):
        from beyondbench.tasks.medium.fibonacci_sequence_task import FibonacciSequenceTask
        assert FibonacciSequenceTask is not None


class TestUtilImports:
    def test_import_utils(self):
        from beyondbench.utils import get_logger, setup_logging
        assert callable(get_logger)
        assert callable(setup_logging)

    def test_import_error_handler(self):
        from beyondbench.utils.error_handler import ErrorHandler
        assert ErrorHandler is not None

    def test_import_stats_tracker(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        assert StatsTracker is not None

    def test_import_token_counter(self):
        from beyondbench.utils.token_counter import TokenCounter
        assert TokenCounter is not None

    def test_import_shared_utils(self):
        from beyondbench.utils.shared_utils import (
            values_are_close, round_if_close_to_int, is_valid_number,
            safe_divide, clamp, gcd, lcm, fibonacci, is_prime, factorial,
        )


class TestCLIImports:
    def test_import_cli_main(self):
        from beyondbench.cli.main import main, cli
        import click
        assert isinstance(main, click.core.Group)
        assert callable(cli)

    def test_import_wizard(self):
        from beyondbench.cli.wizard import main_wizard
        assert callable(main_wizard)

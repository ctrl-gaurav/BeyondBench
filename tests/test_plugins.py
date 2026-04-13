"""
Tests for Phase 22: Plugin System & Custom Task SDK.

Covers:
- PluginManager discovery, loading, and validation
- Invalid plugin rejection with clear error messages
- create-task scaffolding
- TaskMetadata creation and defaults
- list-tasks --detailed CLI output
- Plugin name conflict detection
- Loading plugins from a local directory
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import textwrap
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_handler():
    """Return a mock ModelHandler."""
    handler = MagicMock()
    handler.backend = "mock"
    handler.api_provider = None
    handler.model_id = "test-model"
    handler.tokenizer = None
    handler.get_model_info.return_value = {"model_name": "test-model"}
    handler.generate.return_value = [r"\boxed{42}"]
    return handler


def _minimal_task_class(name: str, suite: str = "easy"):
    """
    Dynamically create a minimal but fully valid BaseTask subclass.
    """
    from beyondbench.core.base_task import BaseTask

    class _MinimalTask(BaseTask):
        @property
        def task_name(self):
            return name

        def generate_data(self, **kwargs):
            return [1, 2, 3]

        def create_prompt(self, data_point):
            return f"What is {data_point}?"

        def evaluate_response(self, response, data_point):
            return {"accuracy": 1.0, "instruction_followed": True}

    _MinimalTask.__name__ = f"{name.title().replace('_', '')}Task"
    _MinimalTask.__qualname__ = _MinimalTask.__name__
    return _MinimalTask


# ===========================================================================
# 1. TaskMetadata
# ===========================================================================


class TestTaskMetadata:
    def test_default_fields(self):
        from beyondbench.plugins import TaskMetadata

        meta = TaskMetadata(name="sorting")
        assert meta.name == "sorting"
        assert meta.difficulty == "easy"
        assert meta.category == "general"
        assert meta.author == ""
        assert meta.version == "1.0.0"
        assert meta.description == ""

    def test_custom_fields(self):
        from beyondbench.plugins import TaskMetadata

        meta = TaskMetadata(
            name="my_task",
            description="Test task",
            difficulty="hard",
            category="graph",
            author="Alice",
            version="2.3.1",
        )
        assert meta.difficulty == "hard"
        assert meta.category == "graph"
        assert meta.author == "Alice"
        assert meta.version == "2.3.1"

    def test_invalid_difficulty_raises(self):
        from beyondbench.plugins import TaskMetadata

        with pytest.raises(ValueError, match="Invalid difficulty"):
            TaskMetadata(name="x", difficulty="extreme")

    def test_invalid_category_raises(self):
        from beyondbench.plugins import TaskMetadata

        with pytest.raises(ValueError, match="Invalid category"):
            TaskMetadata(name="x", category="nonsense_category")

    def test_to_dict(self):
        from beyondbench.plugins import TaskMetadata

        meta = TaskMetadata(name="foo", author="Bob", version="0.1.0")
        d = meta.to_dict()
        assert d["name"] == "foo"
        assert d["author"] == "Bob"
        assert "tags" in d

    def test_valid_categories(self):
        from beyondbench.plugins import VALID_CATEGORIES, TaskMetadata

        for cat in VALID_CATEGORIES:
            m = TaskMetadata(name="t", category=cat)
            assert m.category == cat

    def test_valid_difficulties(self):
        from beyondbench.plugins import VALID_DIFFICULTIES, TaskMetadata

        for diff in VALID_DIFFICULTIES:
            m = TaskMetadata(name="t", difficulty=diff)
            assert m.difficulty == diff


# ===========================================================================
# 2. get_task_metadata
# ===========================================================================


class TestGetTaskMetadata:
    def test_reads_class_attribute(self):
        from beyondbench.plugins import TaskMetadata, get_task_metadata

        cls = _minimal_task_class("my_task")
        cls.metadata = TaskMetadata(
            name="my_task", description="Custom desc", difficulty="hard",
            category="graph",
        )
        meta = get_task_metadata(cls, suite="easy")
        assert meta.description == "Custom desc"
        assert meta.difficulty == "hard"

    def test_reads_get_metadata_classmethod(self):
        from beyondbench.plugins import TaskMetadata, get_task_metadata

        cls = _minimal_task_class("cm_task")

        @classmethod
        def _get_meta(klass):
            return TaskMetadata(name="cm_task", category="arithmetic")

        cls.get_metadata = _get_meta
        meta = get_task_metadata(cls, suite="medium")
        assert meta.category == "arithmetic"

    def test_auto_generates_when_no_metadata(self):
        from beyondbench.plugins import get_task_metadata

        cls = _minimal_task_class("sum_task")
        meta = get_task_metadata(cls, suite="easy")
        assert meta.difficulty == "easy"
        # category should be inferred as arithmetic from name
        assert meta.category in ("arithmetic", "general")

    def test_falls_back_gracefully(self):
        """get_task_metadata should never raise for a well-formed task class."""
        from beyondbench.plugins import get_task_metadata

        cls = _minimal_task_class("unknown_xyz_task")
        meta = get_task_metadata(cls, suite="hard")
        assert meta is not None
        assert meta.difficulty == "hard"


# ===========================================================================
# 3. validate_plugin_class
# ===========================================================================


class TestValidatePluginClass:
    def test_valid_class_passes(self):
        from beyondbench.plugins import validate_plugin_class

        cls = _minimal_task_class("valid_task")
        ok, reason = validate_plugin_class(cls)
        assert ok is True
        assert reason == ""

    def test_non_class_rejected(self):
        from beyondbench.plugins import validate_plugin_class

        ok, reason = validate_plugin_class("not a class")  # type: ignore[arg-type]
        assert ok is False
        assert "class" in reason.lower()

    def test_non_basetask_rejected(self):
        from beyondbench.plugins import validate_plugin_class

        class NotATask:
            task_name = "nope"

        ok, reason = validate_plugin_class(NotATask)
        assert ok is False
        assert "BaseTask" in reason

    def test_missing_task_name_rejected(self):
        from beyondbench.core.base_task import BaseTask
        from beyondbench.plugins import validate_plugin_class

        class NoNameTask(BaseTask):
            def generate_data(self, **kwargs):
                return []

            def create_prompt(self, dp):
                return ""

            def evaluate_response(self, r, dp):
                return {}

        ok, reason = validate_plugin_class(NoNameTask)
        assert ok is False
        assert "task_name" in reason.lower()

    def test_missing_generate_data_rejected(self):
        from beyondbench.core.base_task import BaseTask
        from beyondbench.plugins import validate_plugin_class

        class IncompleteTask(BaseTask):
            @property
            def task_name(self):
                return "incomplete"

            def create_prompt(self, dp):
                return ""

            def evaluate_response(self, r, dp):
                return {}

        ok, reason = validate_plugin_class(IncompleteTask)
        assert ok is False
        # generate_data is abstract → still fails validation
        assert "generate_data" in reason

    def test_name_conflict_rejected(self):
        from beyondbench.plugins import validate_plugin_class

        cls = _minimal_task_class("sorting")  # 'sorting' is a built-in task
        fake_existing = {"sorting": object()}
        ok, reason = validate_plugin_class(cls, existing_tasks=fake_existing)
        assert ok is False
        assert "conflict" in reason.lower()


# ===========================================================================
# 4. PluginManager — basic construction
# ===========================================================================


class TestPluginManagerConstruction:
    def test_creates_with_default_registry(self):
        from beyondbench.plugins import PluginManager

        pm = PluginManager()
        assert pm.registry is not None
        assert pm.loaded_plugins == {}
        assert pm.errors == []

    def test_creates_with_custom_registry(self):
        from beyondbench.core.task_registry import TaskRegistry
        from beyondbench.plugins import PluginManager

        reg = TaskRegistry()
        pm = PluginManager(registry=reg)
        assert pm.registry is reg

    def test_creates_with_plugin_dir(self, tmp_path):
        from beyondbench.plugins import PluginManager

        pm = PluginManager(plugin_dir=tmp_path)
        assert pm.plugin_dir == tmp_path

    def test_default_suite(self):
        from beyondbench.plugins import PluginManager

        pm = PluginManager(default_suite="medium")
        assert pm.default_suite == "medium"


# ===========================================================================
# 5. PluginManager — entry point discovery (mocked)
# ===========================================================================


class TestPluginManagerEntryPoints:
    def _mock_ep(self, name: str, task_class):
        """Create a mock entry point that loads *task_class*."""
        ep = MagicMock()
        ep.name = name
        ep.load.return_value = task_class
        return ep

    def test_loads_valid_entry_point(self):
        from beyondbench.plugins import PluginManager
        from beyondbench.plugins.discovery import ENTRY_POINT_GROUP

        cls = _minimal_task_class("ep_task_valid")
        ep = self._mock_ep("ep_task_valid", cls)

        with patch(
            "beyondbench.plugins.discovery._iter_entry_points",
            return_value=[("ep_task_valid", ep)],
        ):
            pm = PluginManager()
            loaded = pm.load_from_entry_points()

        assert "ep_task_valid" in loaded
        assert pm.errors == []

    def test_rejects_non_basetask_entry_point(self):
        from beyondbench.plugins import PluginManager

        class BadPlugin:
            task_name = "bad"

        ep = MagicMock()
        ep.name = "bad_ep"
        ep.load.return_value = BadPlugin

        with patch(
            "beyondbench.plugins.discovery._iter_entry_points",
            return_value=[("bad_ep", ep)],
        ):
            pm = PluginManager()
            loaded = pm.load_from_entry_points()

        assert "bad_ep" not in loaded
        assert len(pm.errors) == 1
        assert "BaseTask" in pm.errors[0]

    def test_handles_import_error_gracefully(self):
        from beyondbench.plugins import PluginManager

        ep = MagicMock()
        ep.name = "broken_ep"
        ep.load.side_effect = ImportError("module missing")

        with patch(
            "beyondbench.plugins.discovery._iter_entry_points",
            return_value=[("broken_ep", ep)],
        ):
            pm = PluginManager()
            loaded = pm.load_from_entry_points()

        assert loaded == {}
        assert len(pm.errors) == 1
        assert "broken_ep" in pm.errors[0]

    def test_name_conflict_detected(self):
        from beyondbench.plugins import PluginManager

        # 'sorting' is already a built-in
        cls = _minimal_task_class("sorting")
        ep = self._mock_ep("sorting", cls)

        with patch(
            "beyondbench.plugins.discovery._iter_entry_points",
            return_value=[("sorting", ep)],
        ):
            pm = PluginManager()
            loaded = pm.load_from_entry_points()

        assert "sorting" not in loaded
        assert len(pm.errors) >= 1
        assert "conflict" in pm.errors[0].lower()


# ===========================================================================
# 6. PluginManager — local directory loading
# ===========================================================================


class TestPluginManagerDirectoryLoading:
    def _write_valid_plugin(self, directory: Path, task_name: str) -> Path:
        """Write a minimal valid plugin .py file to *directory*."""
        content = textwrap.dedent(f"""\
            from beyondbench.core.base_task import BaseTask

            class {task_name.title().replace("_", "")}Task(BaseTask):
                @property
                def task_name(self):
                    return "{task_name}"

                def generate_data(self, **kwargs):
                    return [1, 2, 3]

                def create_prompt(self, data_point):
                    return f"What is {{data_point}}?"

                def evaluate_response(self, response, data_point):
                    return {{"accuracy": 1.0, "instruction_followed": True}}
        """)
        py_file = directory / f"{task_name}_task.py"
        py_file.write_text(content)
        return py_file

    def test_loads_valid_py_file(self, tmp_path):
        from beyondbench.plugins import PluginManager

        self._write_valid_plugin(tmp_path, "dir_test_task")
        pm = PluginManager(plugin_dir=tmp_path)
        loaded = pm.load_from_directory(tmp_path)

        assert "dir_test_task" in loaded
        assert pm.errors == []

    def test_nonexistent_directory_logged(self, tmp_path):
        from beyondbench.plugins import PluginManager

        nonexistent = tmp_path / "does_not_exist"
        pm = PluginManager()
        # Should not raise — just logs a warning
        loaded = pm.load_from_directory(nonexistent)
        assert loaded == {}

    def test_invalid_python_file_gives_error(self, tmp_path):
        from beyondbench.plugins import PluginManager

        bad_file = tmp_path / "bad_syntax_task.py"
        bad_file.write_text("this is not valid python )(!")

        pm = PluginManager()
        loaded = pm.load_from_directory(tmp_path)
        # bad file should not be in loaded
        assert loaded == {}
        assert len(pm.errors) == 1

    def test_task_registered_in_registry(self, tmp_path):
        from beyondbench.core.task_registry import TaskRegistry
        from beyondbench.plugins import PluginManager

        self._write_valid_plugin(tmp_path, "reg_check_task")
        reg = TaskRegistry()
        pm = PluginManager(registry=reg)
        pm.load_from_directory(tmp_path)

        assert reg.get_task_class("reg_check_task") is not None

    def test_loads_from_package_directory(self, tmp_path):
        """A directory with __init__.py should be importable as a package."""
        from beyondbench.plugins import PluginManager

        pkg_dir = tmp_path / "my_pkg_task"
        pkg_dir.mkdir()

        init_content = textwrap.dedent("""\
            from beyondbench.core.base_task import BaseTask

            class MyPkgTask(BaseTask):
                @property
                def task_name(self):
                    return "my_pkg_task"

                def generate_data(self, **kwargs):
                    return [10]

                def create_prompt(self, dp):
                    return str(dp)

                def evaluate_response(self, r, dp):
                    return {"accuracy": 0.5, "instruction_followed": True}
        """)
        (pkg_dir / "__init__.py").write_text(init_content)

        pm = PluginManager(plugin_dir=tmp_path)
        loaded = pm.load_from_directory(tmp_path)

        assert "my_pkg_task" in loaded


# ===========================================================================
# 7. create-task scaffolding
# ===========================================================================


class TestCreateTaskScaffold:
    def test_creates_expected_files(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="my_test_task",
            output_dir=str(tmp_path),
            suite="easy",
            author="Test Author",
            description="A test task",
        )

        assert project_dir.exists()
        expected_files = [
            "__init__.py",
            "task.py",
            "parser.py",
            "test_task.py",
            "pyproject.toml",
            "README.md",
        ]
        for fname in expected_files:
            fpath = project_dir / fname
            assert fpath.exists(), f"Missing file: {fname}"
            assert fpath.stat().st_size > 0, f"Empty file: {fname}"

    def test_task_py_is_valid_python(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="valid_py_task",
            output_dir=str(tmp_path),
        )
        source = (project_dir / "task.py").read_text()
        ast.parse(source)  # Must not raise

    def test_parser_py_is_valid_python(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="parser_py_task",
            output_dir=str(tmp_path),
        )
        source = (project_dir / "parser.py").read_text()
        ast.parse(source)

    def test_init_py_is_valid_python(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="init_py_task",
            output_dir=str(tmp_path),
        )
        source = (project_dir / "__init__.py").read_text()
        ast.parse(source)

    def test_test_py_is_valid_python(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="test_py_task",
            output_dir=str(tmp_path),
        )
        source = (project_dir / "test_task.py").read_text()
        ast.parse(source)

    def test_pyproject_toml_has_entry_point(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="ep_check_task",
            output_dir=str(tmp_path),
        )
        toml_content = (project_dir / "pyproject.toml").read_text()
        assert "beyondbench.tasks" in toml_content
        assert "ep_check_task" in toml_content

    def test_class_name_in_task_py(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="class_name_task",
            output_dir=str(tmp_path),
        )
        source = (project_dir / "task.py").read_text()
        assert "ClassNameTask" in source

    def test_error_on_existing_directory(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        create_task_scaffold(task_name="duplicate_task", output_dir=str(tmp_path))
        with pytest.raises(FileExistsError):
            create_task_scaffold(task_name="duplicate_task", output_dir=str(tmp_path))

    def test_invalid_name_raises(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        with pytest.raises(ValueError):
            create_task_scaffold(task_name="123invalid", output_dir=str(tmp_path))

    def test_hyphenated_name_normalised(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="my-task-name",
            output_dir=str(tmp_path),
        )
        # Should be normalised to my_task_name
        assert project_dir.name == "my_task_name"

    def test_author_appears_in_files(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="author_task",
            output_dir=str(tmp_path),
            author="Jane Doe",
        )
        toml = (project_dir / "pyproject.toml").read_text()
        assert "Jane Doe" in toml
        task_src = (project_dir / "task.py").read_text()
        assert "Jane Doe" in task_src

    def test_suite_appears_in_pyproject(self, tmp_path):
        from beyondbench.plugins.scaffold import create_task_scaffold

        project_dir = create_task_scaffold(
            task_name="hard_task",
            output_dir=str(tmp_path),
            suite="hard",
        )
        task_src = (project_dir / "task.py").read_text()
        assert "hard" in task_src


# ===========================================================================
# 8. CLI — create-task command
# ===========================================================================


class TestCreateTaskCLI:
    def test_create_task_command_exists(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["create-task", "--help"])
        assert result.exit_code == 0
        assert "TASK_NAME" in result.output

    def test_create_task_creates_directory(self, tmp_path):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "create-task", "cli_test_task",
                "--output-dir", str(tmp_path),
                "--author", "CLI Tester",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cli_test_task").exists()

    def test_create_task_shows_next_steps(self, tmp_path):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["create-task", "steps_task", "--output-dir", str(tmp_path)],
        )
        assert "Next steps" in result.output

    def test_create_task_duplicate_fails(self, tmp_path):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        runner.invoke(main, ["create-task", "dup_task2", "--output-dir", str(tmp_path)])
        result = runner.invoke(main, ["create-task", "dup_task2", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_create_task_invalid_name_fails(self, tmp_path):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["create-task", "123badname", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code != 0


# ===========================================================================
# 9. CLI — list-tasks --detailed
# ===========================================================================


class TestListTasksDetailed:
    def test_detailed_flag_exists(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks", "--help"])
        assert result.exit_code == 0
        assert "--detailed" in result.output

    def test_detailed_table_output(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks", "--suite", "easy", "--detailed"])
        assert result.exit_code == 0, result.output
        # Should output something meaningful
        assert len(result.output.strip()) > 0

    def test_detailed_json_output(self):
        import json
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        # Use --quiet to suppress log messages from mixing with JSON output
        result = runner.invoke(
            main,
            ["--quiet", "list-tasks", "--suite", "easy", "--detailed", "--format", "json"],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0
        # Each entry should have metadata fields
        first = data[0]
        for field in ("name", "suite", "category", "difficulty", "author", "version", "description"):
            assert field in first, f"Missing field: {field}"

    def test_detailed_contains_task_names(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks", "--suite", "easy", "--detailed"])
        assert result.exit_code == 0, result.output
        # 'sorting' is always in easy suite
        assert "sorting" in result.output

    def test_regular_list_tasks_still_works(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["list-tasks"])
        assert result.exit_code == 0, result.output


# ===========================================================================
# 10. load_all_plugins combines both sources
# ===========================================================================


class TestLoadAllPlugins:
    def _write_valid_plugin(self, directory: Path, task_name: str) -> None:
        content = textwrap.dedent(f"""\
            from beyondbench.core.base_task import BaseTask

            class {task_name.title().replace("_", "")}Task(BaseTask):
                @property
                def task_name(self):
                    return "{task_name}"

                def generate_data(self, **kwargs):
                    return [1]

                def create_prompt(self, dp):
                    return str(dp)

                def evaluate_response(self, r, dp):
                    return {{"accuracy": 1.0, "instruction_followed": True}}
        """)
        (directory / f"{task_name}_task.py").write_text(content)

    def test_load_all_combines_sources(self, tmp_path):
        from beyondbench.plugins import PluginManager
        from beyondbench.plugins.discovery import ENTRY_POINT_GROUP

        self._write_valid_plugin(tmp_path, "combined_dir_task")

        ep_task_cls = _minimal_task_class("combined_ep_task")
        ep = MagicMock()
        ep.name = "combined_ep_task"
        ep.load.return_value = ep_task_cls

        with patch(
            "beyondbench.plugins.discovery._iter_entry_points",
            return_value=[("combined_ep_task", ep)],
        ):
            pm = PluginManager(plugin_dir=tmp_path)
            loaded = pm.load_all_plugins()

        assert "combined_dir_task" in loaded
        assert "combined_ep_task" in loaded


# ===========================================================================
# 11. Imports smoke test
# ===========================================================================


class TestImports:
    def test_plugin_module_importable(self):
        import beyondbench.plugins
        assert hasattr(beyondbench.plugins, "PluginManager")
        assert hasattr(beyondbench.plugins, "TaskMetadata")

    def test_discovery_importable(self):
        from beyondbench.plugins.discovery import PluginManager, validate_plugin_class, ENTRY_POINT_GROUP
        assert ENTRY_POINT_GROUP == "beyondbench.tasks"

    def test_metadata_importable(self):
        from beyondbench.plugins.metadata import TaskMetadata, get_task_metadata
        assert callable(get_task_metadata)

    def test_scaffold_importable(self):
        from beyondbench.plugins.scaffold import create_task_scaffold
        assert callable(create_task_scaffold)

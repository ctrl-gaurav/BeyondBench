"""Test TaskRegistry functionality."""

import pytest
from beyondbench.core.task_registry import TaskRegistry


class TestTaskRegistryInit:
    @pytest.fixture
    def registry(self):
        return TaskRegistry()

    def test_initialization(self, registry):
        assert registry is not None
        assert hasattr(registry, '_tasks')
        assert hasattr(registry, '_suite_mapping')

    def test_suite_mapping_has_all_suites(self, registry):
        assert 'easy' in registry._suite_mapping
        assert 'medium' in registry._suite_mapping
        assert 'hard' in registry._suite_mapping

    def test_easy_task_count(self, registry):
        tasks = registry.get_tasks_for_suite("easy")
        assert len(tasks) == 29

    def test_medium_task_count(self, registry):
        tasks = registry.get_tasks_for_suite("medium")
        assert len(tasks) == 5

    def test_hard_task_count(self, registry):
        tasks = registry.get_tasks_for_suite("hard")
        assert len(tasks) == 10

    def test_all_tasks_count(self, registry):
        tasks = registry.get_tasks_for_suite("all")
        assert len(tasks) == 44

    def test_invalid_suite_returns_empty(self, registry):
        tasks = registry.get_tasks_for_suite("nonexistent")
        assert tasks == []


class TestTaskClassImports:
    @pytest.fixture
    def registry(self):
        return TaskRegistry()

    def test_easy_tasks_imported(self, registry):
        """Verify all 29 easy task classes were actually imported."""
        easy_tasks = registry.get_tasks_for_suite("easy")
        imported = sum(1 for t in easy_tasks if t in registry._tasks)
        assert imported == 29, f"Only {imported}/29 easy tasks imported"

    def test_medium_tasks_imported(self, registry):
        medium_tasks = registry.get_tasks_for_suite("medium")
        imported = sum(1 for t in medium_tasks if t in registry._tasks)
        assert imported == 5, f"Only {imported}/5 medium tasks imported"

    def test_hard_tasks_imported(self, registry):
        hard_tasks = registry.get_tasks_for_suite("hard")
        imported = sum(1 for t in hard_tasks if t in registry._tasks)
        assert imported == 10, f"Only {imported}/10 hard tasks imported"

    def test_get_task_class_returns_class(self, registry):
        task_class = registry.get_task_class("sum")
        assert task_class is not None

    def test_get_task_class_unknown_returns_none(self, registry):
        task_class = registry.get_task_class("nonexistent_task_xyz")
        assert task_class is None

    def test_validate_task_name(self, registry):
        assert registry.validate_task_name("sum") is True
        assert registry.validate_task_name("sorting") is True
        assert registry.validate_task_name("fake_task") is False

    def test_get_task_info(self, registry):
        info = registry.get_task_info("sum")
        assert info is not None
        assert info["suite"] == "easy"
        assert info["name"] == "sum"

    def test_get_task_info_unknown_returns_none(self, registry):
        info = registry.get_task_info("nonexistent_task_xyz")
        assert info is None

    def test_get_available_tasks(self, registry):
        available = registry.get_available_tasks("all")
        assert "easy" in available
        assert "medium" in available
        assert "hard" in available

    def test_get_available_tasks_single_suite(self, registry):
        available = registry.get_available_tasks("easy")
        assert "easy" in available
        assert "medium" not in available

    def test_get_available_tasks_unknown_suite(self, registry):
        available = registry.get_available_tasks("nonexistent")
        assert available == {}

    def test_get_suite_stats(self, registry):
        stats = registry.get_suite_stats()
        assert stats["easy"] == 29
        assert stats["medium"] == 5
        assert stats["hard"] == 10

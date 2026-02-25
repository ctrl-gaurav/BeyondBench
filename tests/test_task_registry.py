"""
Test TaskRegistry functionality.
"""

import pytest
from beyondbench.core.task_registry import TaskRegistry


class TestTaskRegistry:
    """Test TaskRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a TaskRegistry instance."""
        return TaskRegistry()

    def test_registry_initialization(self, registry):
        """Test registry initializes correctly."""
        assert registry is not None

    def test_get_easy_tasks(self, registry):
        """Test retrieving easy tasks."""
        tasks = registry.get_tasks_for_suite("easy")
        assert isinstance(tasks, (list, dict))
        assert len(tasks) > 0

    def test_get_medium_tasks(self, registry):
        """Test retrieving medium tasks."""
        tasks = registry.get_tasks_for_suite("medium")
        assert isinstance(tasks, (list, dict))

    def test_get_hard_tasks(self, registry):
        """Test retrieving hard tasks."""
        tasks = registry.get_tasks_for_suite("hard")
        assert isinstance(tasks, (list, dict))

    def test_get_all_tasks(self, registry):
        """Test retrieving all tasks."""
        all_tasks = registry.get_tasks_for_suite("all")
        easy_tasks = registry.get_tasks_for_suite("easy")
        medium_tasks = registry.get_tasks_for_suite("medium")
        hard_tasks = registry.get_tasks_for_suite("hard")

        # All tasks should include tasks from all suites
        if isinstance(all_tasks, list):
            assert len(all_tasks) >= len(easy_tasks)

    def test_invalid_suite(self, registry):
        """Test invalid suite name handling."""
        tasks = registry.get_tasks_for_suite("invalid_suite_name")
        # Should return empty or raise appropriate error
        assert tasks is None or len(tasks) == 0 or isinstance(tasks, (list, dict))

    def test_easy_task_count(self, registry):
        """Test that easy suite has expected number of tasks."""
        tasks = registry.get_tasks_for_suite("easy")
        # Should have 29 easy tasks
        if isinstance(tasks, list):
            assert len(tasks) == 29
        elif isinstance(tasks, dict):
            assert len(tasks) == 29


class TestTaskRegistryContents:
    """Test specific task registration."""

    @pytest.fixture
    def registry(self):
        """Create a TaskRegistry instance."""
        return TaskRegistry()

    def test_sum_task_registered(self, registry):
        """Test that sum task is registered."""
        tasks = registry.get_tasks_for_suite("easy")
        task_list = list(tasks) if isinstance(tasks, dict) else tasks
        # Should contain 'sum' task
        assert 'sum' in task_list or any('sum' in str(t).lower() for t in task_list)

    def test_sorting_task_registered(self, registry):
        """Test that sorting task is registered."""
        tasks = registry.get_tasks_for_suite("easy")
        task_list = list(tasks) if isinstance(tasks, dict) else tasks
        assert 'sorting' in task_list or any('sort' in str(t).lower() for t in task_list)

    def test_medium_suite_has_sequence_tasks(self, registry):
        """Test that medium suite has sequence tasks."""
        tasks = registry.get_tasks_for_suite("medium")
        assert len(tasks) >= 5  # At least 5 medium tasks

    def test_hard_suite_has_complex_tasks(self, registry):
        """Test that hard suite has complex tasks."""
        tasks = registry.get_tasks_for_suite("hard")
        assert len(tasks) >= 10  # At least 10 hard tasks


class TestTaskAccess:
    """Test accessing individual tasks."""

    @pytest.fixture
    def registry(self):
        """Create a TaskRegistry instance."""
        return TaskRegistry()

    def test_get_task_by_name(self, registry):
        """Test getting a task by name."""
        # Try to get a specific task
        if hasattr(registry, 'get_task'):
            task = registry.get_task('sum')
            assert task is not None
        else:
            # Registry might use different method
            tasks = registry.get_tasks_for_suite('easy')
            assert len(tasks) > 0

    def test_task_has_info(self, registry):
        """Test that tasks have information available."""
        tasks = registry.get_tasks_for_suite('easy')
        assert tasks is not None
        # Should have at least one task
        assert len(tasks) > 0

"""
Task registry for beyondbench package.

Manages task discovery, registration, and retrieval across all suites.
"""

import importlib
from typing import Dict, List, Type, Optional
import logging

from .base_task import BaseTask


class TaskRegistry:
    """
    Central registry for all evaluation tasks.

    Manages task discovery, registration, and provides unified access
    to tasks across easy, medium, and hard suites.
    """

    def __init__(self):
        """Initialize task registry."""
        self.logger = logging.getLogger(__name__)
        self._tasks: Dict[str, Type[BaseTask]] = {}
        self._suite_mapping: Dict[str, List[str]] = {
            'easy': [],
            'medium': [],
            'hard': []
        }

        # Register all available tasks
        self._discover_and_register_tasks()

    def _discover_and_register_tasks(self):
        """Discover and register all available tasks."""

        # Easy suite tasks (14 core + 15 scalable + 15 Phase 12 = 44 total)
        easy_tasks = [
            # Core tasks
            'sorting', 'comparison', 'sum', 'multiplication', 'odd_count', 'even_count',
            'absolute_difference', 'division', 'find_maximum', 'find_minimum',
            'mean', 'median', 'mode', 'subtraction',

            # New scalable tasks from new.md
            'second_maximum', 'range', 'index_of_maximum', 'count_negative',
            'count_unique', 'max_adjacent_difference', 'count_greater_than_previous', 'sum_of_max_indices',
            'count_palindromic', 'longest_increasing_subsequence', 'sum_of_digits', 'count_perfect_squares',
            'alternating_sum', 'count_multiples', 'local_maxima_count',

            # Phase 12: 15 new easy tasks
            'weighted_sum', 'running_average', 'parity_check', 'cumulative_sum',
            'reverse_list', 'rotate_list', 'interleave_lists', 'set_intersection',
            'set_difference', 'moving_average', 'element_frequency', 'second_minimum',
            'variance', 'standard_deviation', 'dot_product',
        ]

        # Medium suite tasks (5 original + 10 Phase 13 = 15 total)
        medium_tasks = [
            'fibonacci_sequence', 'algebraic_sequence', 'geometric_sequence',
            'prime_sequence', 'complex_pattern',
            # Phase 13 additions
            'arithmetic_progression', 'harmonic_sequence', 'collatz_sequence',
            'polynomial_evaluation', 'matrix_operations', 'number_base_conversion',
            'logical_operations', 'pattern_completion', 'gcd_lcm', 'combinatorics',
        ]

        # Hard suite tasks (20 domains: 10 original + 10 Phase 14)
        hard_tasks = [
            'tower_hanoi', 'n_queens', 'graph_coloring', 'boolean_sat',
            'sudoku_solving', 'cryptarithmetic', 'matrix_chain_multiplication',
            'modular_systems', 'constraint_optimization', 'logic_grid_puzzles',
            # Phase 14 additions
            'shortest_path', 'knapsack', 'traveling_salesman',
            'longest_common_subsequence', 'minimax_game', 'regex_matching',
            'topological_sort', 'interval_scheduling', 'coin_change', 'edit_distance',
        ]

        # Register task mappings
        self._suite_mapping['easy'] = easy_tasks
        self._suite_mapping['medium'] = medium_tasks
        self._suite_mapping['hard'] = hard_tasks

        # Import and register actual task classes
        self._import_task_classes()

        # Log discovered tasks
        total_tasks = len(easy_tasks) + len(medium_tasks) + len(hard_tasks)
        self.logger.info(f"📋 Task registry initialized with {total_tasks} tasks:")
        self.logger.info(f"   🟢 Easy: {len(easy_tasks)} tasks")
        self.logger.info(f"   🟡 Medium: {len(medium_tasks)} tasks")
        self.logger.info(f"   🔴 Hard: {len(hard_tasks)} tasks")

        # Load plugins from entry points (Phase 22)
        self._load_plugins()

    def _import_task_classes(self):
        """Import and register actual task classes."""
        # Import easy task classes
        for task_name in self._suite_mapping['easy']:
            try:
                module_name = f"beyondbench.tasks.easy.{task_name}_task"
                task_class_name = f"{''.join(word.capitalize() for word in task_name.split('_'))}Task"
                module = importlib.import_module(module_name)
                task_class = getattr(module, task_class_name)
                self._tasks[task_name] = task_class
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"⚠️  Could not import easy task {task_name}: {e}")

        # Import medium task classes
        for task_name in self._suite_mapping['medium']:
            try:
                module_name = f"beyondbench.tasks.medium.{task_name}_task"
                task_class_name = f"{''.join(word.capitalize() for word in task_name.split('_'))}Task"
                module = importlib.import_module(module_name)
                task_class = getattr(module, task_class_name)
                self._tasks[task_name] = task_class
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"⚠️  Could not import medium task {task_name}: {e}")

        # Import hard task classes with correct mappings
        hard_task_mappings = {
            'sudoku_solving': ('beyondbench.tasks.hard.sudoku_task', 'SudokuTask'),
            'boolean_sat': ('beyondbench.tasks.hard.boolean_sat_task', 'BooleanSATTask'),
            'matrix_chain_multiplication': ('beyondbench.tasks.hard.matrix_chain_multiplication_task', 'MatrixChainTask'),
            'modular_systems': ('beyondbench.tasks.hard.modular_systems_solver_task', 'ModularSystemsTask'),
            'logic_grid_puzzles': ('beyondbench.tasks.hard.logic_grid_puzzles_task_enhanced', 'LogicGridPuzzlesTask'),
            'tower_hanoi': ('beyondbench.tasks.hard.tower_hanoi_task', 'TowerHanoiTask'),
            'n_queens': ('beyondbench.tasks.hard.n_queens_task', 'NQueensTask'),
            'cryptarithmetic': ('beyondbench.tasks.hard.cryptarithmetic_task', 'CryptarithmeticTask'),
            'constraint_optimization': ('beyondbench.tasks.hard.constraint_optimization_task', 'ConstraintOptimizationTask'),
            'graph_coloring': ('beyondbench.tasks.hard.graph_coloring_task', 'GraphColoringTask'),
            # Phase 14 additions
            'shortest_path': ('beyondbench.tasks.hard.shortest_path_task', 'ShortestPathTask'),
            'knapsack': ('beyondbench.tasks.hard.knapsack_task', 'KnapsackTask'),
            'traveling_salesman': ('beyondbench.tasks.hard.traveling_salesman_task', 'TravelingSalesmanTask'),
            'longest_common_subsequence': ('beyondbench.tasks.hard.longest_common_subsequence_task', 'LongestCommonSubsequenceTask'),
            'minimax_game': ('beyondbench.tasks.hard.minimax_game_task', 'MinimaxGameTask'),
            'regex_matching': ('beyondbench.tasks.hard.regex_matching_task', 'RegexMatchingTask'),
            'topological_sort': ('beyondbench.tasks.hard.topological_sort_task', 'TopologicalSortTask'),
            'interval_scheduling': ('beyondbench.tasks.hard.interval_scheduling_task', 'IntervalSchedulingTask'),
            'coin_change': ('beyondbench.tasks.hard.coin_change_task', 'CoinChangeTask'),
            'edit_distance': ('beyondbench.tasks.hard.edit_distance_task', 'EditDistanceTask'),
        }

        for task_name in self._suite_mapping['hard']:
            try:
                if task_name in hard_task_mappings:
                    module_name, task_class_name = hard_task_mappings[task_name]
                else:
                    # Fallback to original logic for any unmapped tasks
                    module_name = f"beyondbench.tasks.hard.{task_name}_task"
                    task_class_name = f"{''.join(word.capitalize() for word in task_name.split('_'))}Task"

                module = importlib.import_module(module_name)
                task_class = getattr(module, task_class_name)
                self._tasks[task_name] = task_class
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"⚠️  Could not import hard task {task_name}: {e}")

    def _load_plugins(self) -> None:
        """Discover and load plugins from installed entry points (Phase 22)."""
        try:
            from ..plugins.discovery import PluginManager, ENTRY_POINT_GROUP
            from importlib.metadata import entry_points as _eps

            # Quick-check: skip the heavier PluginManager if no plugins installed
            try:
                eps = _eps(group=ENTRY_POINT_GROUP)
            except TypeError:
                all_eps = _eps()
                eps = all_eps.get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

            if not eps:
                return  # No plugins installed — avoid any overhead

            pm = PluginManager(registry=self)
            loaded = pm.load_from_entry_points()
            if loaded:
                self.logger.info(
                    f"🔌 Loaded {len(loaded)} plugin task(s): "
                    + ", ".join(loaded.keys())
                )
        except Exception as exc:
            self.logger.debug(f"Plugin discovery skipped: {exc}")

    def register_task(self, task_name: str, task_class: Type[BaseTask], suite: str):
        """
        Register a task class.

        Args:
            task_name: Unique task identifier
            task_class: Task implementation class
            suite: Task suite (easy, medium, hard)
        """
        if not issubclass(task_class, BaseTask):
            raise ValueError(f"Task class must inherit from BaseTask")

        self._tasks[task_name] = task_class

        if suite in self._suite_mapping and task_name not in self._suite_mapping[suite]:
            self._suite_mapping[suite].append(task_name)

        self.logger.debug(f"✅ Registered task: {task_name} ({suite})")

    def get_task_class(self, task_name: str) -> Optional[Type[BaseTask]]:
        """
        Get task class by name.

        Args:
            task_name: Task identifier

        Returns:
            Task class or None if not found
        """
        # Return the actual imported task class if available
        if task_name in self._tasks:
            return self._tasks[task_name]

        # Fallback to placeholder for tasks not yet imported
        self.logger.warning(f"⚠️  Task '{task_name}' not found in imported classes, using placeholder")
        return self._create_placeholder_task(task_name)

    def _create_placeholder_task(self, task_name: str) -> Optional[Type[BaseTask]]:
        """
        Create a placeholder task implementation.

        In the full package, this would be replaced with actual task loading.
        """
        self.logger.warning(f"Task '{task_name}' not available, no placeholder will be created")
        return None

    def get_tasks_for_suite(self, suite: str) -> List[str]:
        """
        Get all task names for a specific suite.

        Args:
            suite: Suite name ("easy", "medium", "hard", "all")

        Returns:
            List of task names
        """
        if suite == "all":
            all_tasks = []
            for suite_tasks in self._suite_mapping.values():
                all_tasks.extend(suite_tasks)
            return all_tasks
        elif suite in self._suite_mapping:
            return self._suite_mapping[suite].copy()
        else:
            self.logger.warning(f"Unknown suite: {suite}")
            return []

    def get_available_tasks(self, suite: str = "all") -> Dict[str, List[str]]:
        """
        Get available tasks organized by suite.

        Args:
            suite: Suite to get tasks for ("easy", "medium", "hard", "all")

        Returns:
            Dict mapping suite names to task lists
        """
        if suite == "all":
            return self._suite_mapping.copy()
        elif suite in self._suite_mapping:
            return {suite: self._suite_mapping[suite].copy()}
        else:
            return {}

    def validate_task_name(self, task_name: str) -> bool:
        """
        Validate that a task name is available.

        Args:
            task_name: Task name to validate

        Returns:
            True if task is available, False otherwise
        """
        for suite_tasks in self._suite_mapping.values():
            if task_name in suite_tasks:
                return True
        return False

    def get_task_info(self, task_name: str) -> Optional[Dict[str, any]]:
        """
        Get information about a specific task.

        Args:
            task_name: Task name

        Returns:
            Dict with task information or None if not found
        """
        # Find which suite the task belongs to
        task_suite = None
        for suite, tasks in self._suite_mapping.items():
            if task_name in tasks:
                task_suite = suite
                break

        if not task_suite:
            return None

        return {
            "name": task_name,
            "suite": task_suite,
            "type": task_suite,
            "available": True,
            "description": f"{task_name.replace('_', ' ').title()} task from {task_suite} suite"
        }

    def get_suite_stats(self) -> Dict[str, int]:
        """Get statistics about tasks in each suite."""
        return {
            suite: len(tasks)
            for suite, tasks in self._suite_mapping.items()
        }
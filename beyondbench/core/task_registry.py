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

        # Easy suite tasks (14 core + 15 new = 29 total)
        easy_tasks = [
            # Core tasks
            'sorting', 'comparison', 'sum', 'multiplication', 'odd_count', 'even_count',
            'absolute_difference', 'division', 'find_maximum', 'find_minimum',
            'mean', 'median', 'mode', 'subtraction',

            # New scalable tasks from new.md
            'second_maximum', 'range', 'index_of_maximum', 'count_negative',
            'count_unique', 'max_adjacent_difference', 'count_greater_than_previous', 'sum_of_max_indices',
            'count_palindromic', 'longest_increasing_subsequence', 'sum_of_digits', 'count_perfect_squares',
            'alternating_sum', 'count_multiples', 'local_maxima_count'
        ]

        # Medium suite tasks (5 domains)
        medium_tasks = [
            'fibonacci_sequence', 'algebraic_sequence', 'geometric_sequence',
            'prime_sequence', 'complex_pattern'
        ]

        # Hard suite tasks (10 domains)
        hard_tasks = [
            'tower_hanoi', 'n_queens', 'graph_coloring', 'boolean_sat',
            'sudoku_solving', 'cryptarithmetic', 'matrix_chain_multiplication',
            'modular_systems', 'constraint_optimization', 'logic_grid_puzzles'
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
            'graph_coloring': ('beyondbench.tasks.hard.graph_coloring_task', 'GraphColoringTask')
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
        # Get reference to the registry instance for suite mapping
        suite_mapping = self._suite_mapping

        class PlaceholderTask(BaseTask):
            @property
            def task_name(self):
                return task_name

            @property
            def task_type(self):
                # Determine suite based on task name
                for suite, tasks in suite_mapping.items():
                    if task_name in tasks:
                        return suite
                return "unknown"

            def generate_data(self, **kwargs):
                """Placeholder data generation."""
                import random
                # Always return a list of data points
                data = []
                num_samples = getattr(self, 'num_samples', 2)  # Default to 2 samples

                if self.task_type == "easy":
                    # Determine if this is a list-based task or comparison task based on task name
                    list_based_tasks = {
                        'sorting', 'sum', 'multiplication', 'find_maximum', 'find_minimum',
                        'second_maximum', 'count_negative', 'odd_count', 'even_count',
                        'mean', 'median', 'mode', 'sum_of_digits', 'index_of_maximum',
                        'max_adjacent_difference'
                    }

                    if task_name in list_based_tasks:
                        # List-based task - generate lists of numbers
                        list_size = kwargs.get("list_size", 8)  # Default to 8 for reasonable list size
                        for _ in range(num_samples):
                            numbers = [random.randint(-100, 100) for _ in range(list_size)]
                            ground_truth = self._calculate_ground_truth(numbers)
                            data.append({"numbers": numbers, "ground_truth": ground_truth})
                    else:
                        # Comparison or other fixed tasks - generate pairs of numbers
                        for _ in range(num_samples):
                            a, b = random.randint(-100, 100), random.randint(-100, 100)
                            ground_truth = self._calculate_comparison_ground_truth(a, b)
                            data.append({"a": a, "b": b, "ground_truth": ground_truth})
                elif self.task_type == "medium":
                    # Medium tasks - generate meaningful problems
                    for _ in range(num_samples):
                        if task_name == "fibonacci_sequence":
                            # Generate a fibonacci sequence problem
                            n = random.randint(5, 12)
                            fib_sequence = [1, 1]
                            while len(fib_sequence) < n:
                                fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
                            shown_length = n - 2
                            shown_sequence = fib_sequence[:shown_length]
                            next_term = fib_sequence[shown_length]
                            data.append({
                                "sequence": shown_sequence,
                                "question": f"What is the next term in this sequence: {shown_sequence}?",
                                "ground_truth": next_term
                            })
                        else:
                            # Generic medium task
                            data.append({"problem": f"Solve this {task_name} problem", "ground_truth": "unknown"})
                else:
                    # Hard tasks - generate meaningful problems
                    for _ in range(num_samples):
                        if task_name == "sudoku_solving":
                            # Generate a simple sudoku problem
                            data.append({
                                "problem": "Solve this 4x4 Sudoku: Fill each row, column, and 2x2 box with digits 1-4",
                                "grid": "1 _ 3 _\n_ 3 _ 1\n3 _ 1 _\n_ 1 _ 3",
                                "ground_truth": "1 2 3 4\n4 3 2 1\n3 4 1 2\n2 1 4 3"
                            })
                        else:
                            # Generic hard task
                            data.append({"problem": f"Solve this {task_name} problem", "ground_truth": "unknown"})

                return data

            def _calculate_ground_truth(self, numbers):
                """Calculate ground truth for easy tasks."""
                if task_name == "sorting":
                    return sorted(numbers)
                elif task_name == "sum":
                    return sum(numbers)
                elif task_name == "find_maximum":
                    return max(numbers)
                elif task_name == "second_maximum":
                    return sorted(numbers, reverse=True)[1]
                elif task_name == "count_negative":
                    return sum(1 for x in numbers if x < 0)
                # Add more as needed
                return 0

            def _calculate_comparison_ground_truth(self, a, b):
                """Calculate ground truth for comparison tasks."""
                if a > b:
                    return "greater than"
                elif a < b:
                    return "less than"
                else:
                    return "equal to"

            def create_prompt(self, data_point):
                """Placeholder prompt creation."""
                if self.task_type == "easy":
                    # Handle both dictionary and string data points
                    if isinstance(data_point, dict):
                        if "numbers" in data_point:
                            # List-based task
                            numbers = data_point["numbers"]
                            if task_name == "sorting":
                                return f"Sort these numbers in ascending order: {numbers}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                            elif task_name == "sum":
                                return f"Calculate the sum of these numbers: {numbers}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                            elif task_name == "find_maximum":
                                return f"Find the maximum number in this list: {numbers}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                            elif task_name == "second_maximum":
                                return f"Find the second largest element in the list: {numbers}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                            elif task_name == "count_negative":
                                return f"Count how many negative numbers are in the list: {numbers}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                            else:
                                # Generic list task
                                return f"Process these numbers for {task_name}: {numbers}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                        elif "a" in data_point and "b" in data_point:
                            # Comparison task
                            a, b = data_point["a"], data_point["b"]
                            return f"Compare these two numbers: {a} and {b}. Is {a} greater than, less than, or equal to {b}?\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                        else:
                            # Generic dict data point
                            problem = data_point.get('problem', str(data_point))
                            return f"Solve this {task_name} problem: {problem}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                    else:
                        # String data point
                        problem = str(data_point)
                        return f"Solve this {task_name} problem: {problem}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                elif self.task_type == "medium":
                    # Medium tasks with specific handling
                    if isinstance(data_point, dict):
                        if task_name == "fibonacci_sequence" and "sequence" in data_point:
                            # Fibonacci sequence task
                            sequence = data_point["sequence"]
                            return f"What is the next term in this sequence: {sequence}?\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                        else:
                            # Generic medium task
                            problem = data_point.get('problem', str(data_point))
                            return f"{problem}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                    else:
                        problem = str(data_point)
                        return f"Solve this {task_name} problem: {problem}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                else:
                    # Hard tasks with specific handling
                    if isinstance(data_point, dict):
                        if task_name == "sudoku_solving" and "grid" in data_point:
                            # Sudoku task
                            problem = data_point.get("problem", "Solve this Sudoku puzzle")
                            grid = data_point["grid"]
                            return f"{problem}:\n\n{grid}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                        else:
                            # Generic hard task
                            problem = data_point.get('problem', str(data_point))
                            return f"{problem}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."
                    else:
                        problem = str(data_point)
                        return f"Solve this {task_name} problem: {problem}\n\nYour final answer must be in the format \\boxed{{answer}} at the end of your response."

            def evaluate_response(self, response, data_point):
                """Placeholder response evaluation."""
                from ..utils.parsing_utils import AnswerParser

                parser = AnswerParser()
                extracted = parser.parse_answer(response)

                # Handle both dictionary and string data points
                if isinstance(data_point, dict):
                    ground_truth = data_point.get("ground_truth", "unknown")
                else:
                    ground_truth = str(data_point)

                # Smart accuracy calculation with format normalization
                accuracy = self._calculate_accuracy(extracted, ground_truth, task_name)

                return {
                    "accuracy": accuracy,
                    "extracted_answer": extracted,
                    "ground_truth": ground_truth,
                    "response_length": len(response),
                    "parsing_successful": extracted is not None
                }

            def _calculate_accuracy(self, extracted, ground_truth, task_name):
                """Calculate accuracy with format normalization."""
                if extracted is None:
                    return 0.0

                # Direct equality check first
                if extracted == ground_truth:
                    return 1.0

                # Normalize and compare for specific task types
                try:
                    if task_name == "sorting":
                        return self._compare_sorting_answers(extracted, ground_truth)
                    elif task_name in ["sudoku_solving", "n_queens", "tower_hanoi"]:
                        return self._compare_structured_answers(extracted, ground_truth)
                    elif task_name in ["comparison"]:
                        return self._compare_comparison_answers(extracted, ground_truth)
                    else:
                        # Generic comparison with type conversion
                        return self._compare_generic_answers(extracted, ground_truth)
                except Exception:
                    # Fallback to string comparison
                    return 1.0 if str(extracted).strip() == str(ground_truth).strip() else 0.0

            def _compare_sorting_answers(self, extracted, ground_truth):
                """Compare sorting answers, handling list vs string formats."""
                import re

                # Normalize extracted answer (remove brackets, split by commas)
                if isinstance(extracted, str):
                    # Handle formats like "-94, 0, 20, 22" or "[-94, 0, 20, 22]"
                    extracted_clean = re.sub(r'[\[\]]', '', extracted)
                    extracted_list = [int(x.strip()) for x in extracted_clean.split(',')]
                elif isinstance(extracted, list):
                    extracted_list = extracted
                else:
                    return 0.0

                # Normalize ground truth
                if isinstance(ground_truth, list):
                    ground_truth_list = ground_truth
                elif isinstance(ground_truth, str):
                    ground_truth_clean = re.sub(r'[\[\]]', '', ground_truth)
                    ground_truth_list = [int(x.strip()) for x in ground_truth_clean.split(',')]
                else:
                    return 0.0

                # Compare the actual values
                return 1.0 if extracted_list == ground_truth_list else 0.0

            def _compare_structured_answers(self, extracted, ground_truth):
                """Compare structured answers like sudoku grids."""
                import re

                # Normalize both answers to comparable format
                def normalize_grid(answer):
                    if isinstance(answer, str):
                        # Remove extra formatting, normalize separators
                        normalized = re.sub(r'[;\n]', ' ', answer)  # Replace ; and \n with space
                        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
                        normalized = normalized.strip()
                        # Extract just the numbers
                        numbers = re.findall(r'-?\d+', normalized)
                        return ' '.join(numbers)
                    return str(answer)

                extracted_norm = normalize_grid(extracted)
                ground_truth_norm = normalize_grid(ground_truth)

                return 1.0 if extracted_norm == ground_truth_norm else 0.0

            def _compare_comparison_answers(self, extracted, ground_truth):
                """Compare comparison task answers with semantic understanding."""
                # Normalize comparison answers
                comparison_mappings = {
                    'greater than': ['greater than', 'greater', '>', 'is greater than'],
                    'less than': ['less than', 'less', '<', 'is less than'],
                    'equal to': ['equal to', 'equal', '=', '==', 'is equal to']
                }

                def normalize_comparison(answer):
                    answer_str = str(answer).lower().strip()
                    for standard, variants in comparison_mappings.items():
                        if any(variant in answer_str for variant in variants):
                            return standard
                    return answer_str

                extracted_norm = normalize_comparison(extracted)
                ground_truth_norm = normalize_comparison(ground_truth)

                return 1.0 if extracted_norm == ground_truth_norm else 0.0

            def _compare_generic_answers(self, extracted, ground_truth):
                """Generic answer comparison with type conversion."""
                # Try numeric comparison
                try:
                    extracted_num = float(extracted)
                    ground_truth_num = float(ground_truth)
                    return 1.0 if abs(extracted_num - ground_truth_num) < 1e-6 else 0.0
                except (ValueError, TypeError):
                    pass

                # String comparison
                return 1.0 if str(extracted).strip().lower() == str(ground_truth).strip().lower() else 0.0

        return PlaceholderTask

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
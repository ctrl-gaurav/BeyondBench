"""
Core evaluation engine for beyondbench package.

Orchestrates the complete evaluation process across multiple tasks and suites.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .task_registry import TaskRegistry
from ..utils.logging_utils import get_logger
from ..utils.stats_tracker import StatsTracker


class EvaluationEngine:
    """
    Core evaluation engine that orchestrates evaluation across tasks.

    Manages task execution, result aggregation, and comprehensive reporting.
    """

    def __init__(
        self,
        model_handler,
        output_dir: str = "./beyondbench_results",
        store_details: bool = False,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize evaluation engine.

        Args:
            model_handler: Model handler for generating responses
            output_dir: Directory for storing results
            store_details: Whether to store detailed per-example results
            max_retries: Maximum retries for failed operations
            **kwargs: Additional configuration parameters
        """
        self.model_handler = model_handler
        self.output_dir = Path(output_dir)
        self.store_details = store_details
        self.max_retries = max_retries

        self.logger = get_logger("EvaluationEngine")
        self.task_registry = TaskRegistry()
        self.stats_tracker = StatsTracker()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"🚀 Evaluation engine initialized with output dir: {output_dir}")

    def run_evaluation(
        self,
        suite: str = "all",
        tasks: Optional[List[str]] = None,
        datapoints: int = 100,
        folds: int = 1,
        **eval_params
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.

        Args:
            suite: Task suite to evaluate ("easy", "medium", "hard", "all")
            tasks: Specific tasks to evaluate (None for all in suite)
            datapoints: Number of datapoints per task
            folds: Number of cross-validation folds
            **eval_params: Additional evaluation parameters

        Returns:
            Dict containing comprehensive evaluation results
        """
        start_time = time.time()
        self.logger.info(f"🎯 Starting evaluation: suite={suite}, tasks={tasks}, datapoints={datapoints}")

        # Get tasks to evaluate
        if tasks:
            task_list = tasks
        else:
            task_list = self.task_registry.get_tasks_for_suite(suite)

        if not task_list:
            raise ValueError(f"No tasks found for suite '{suite}' with tasks '{tasks}'")

        self.logger.info(f"📋 Evaluating {len(task_list)} tasks: {task_list}")

        # Run evaluation on each task
        task_results = {}
        overall_stats = {
            'total_tasks': len(task_list),
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_evaluations': 0,
            'successful_evaluations': 0
        }

        for task_name in task_list:
            try:
                self.logger.info(f"🔄 Evaluating task: {task_name}")

                # Get task implementation
                task_class = self.task_registry.get_task_class(task_name)
                if not task_class:
                    self.logger.error(f"❌ Task implementation not found: {task_name}")
                    overall_stats['failed_tasks'] += 1
                    continue

                # Initialize task - detect constructor signature to handle
                # different task types (easy/medium use min_val/max_val,
                # hard tasks have custom parameters like board_sizes, etc.)
                import inspect
                init_sig = inspect.signature(task_class.__init__)
                init_params = set(init_sig.parameters.keys()) - {'self'}

                # Common parameters shared by all tasks
                common_kwargs = {
                    'model_handler': self.model_handler,
                    'output_dir': str(self.output_dir / "task_results"),
                    'num_folds': folds,
                    'num_samples': datapoints,
                    'store_details': self.store_details,
                    'temperature': eval_params.get('temperature', 0.7),
                    'top_p': eval_params.get('top_p', 0.9),
                    'max_tokens': eval_params.get('max_tokens', 32768),
                    'seed': eval_params.get('seed'),
                    'prompt_style': eval_params.get('prompt_style', None),
                    'contamination_resistance': eval_params.get('contamination_resistance', 'off'),
                }

                # Check if task uses **kwargs (accepts arbitrary params)
                accepts_kwargs = 'kwargs' in init_params

                # Add max_retries if the task accepts it (explicitly or via **kwargs)
                if 'max_retries' in init_params or accepts_kwargs:
                    common_kwargs['max_retries'] = self.max_retries

                # Standard easy/medium tasks use min_val/max_val
                # Also add for tasks that accept **kwargs (they pass through to BaseTask)
                if 'min_val' in init_params or accepts_kwargs:
                    common_kwargs['min_val'] = eval_params.get('range_min', 1)
                    common_kwargs['max_val'] = eval_params.get('range_max', 100)

                # Hard task-specific parameters with sensible defaults
                hard_task_defaults = {
                    'board_sizes': [4, 5, 6, 8],
                    'num_disks_list': [3, 4, 5],
                    'num_variables_list': [4, 6, 8],
                    'sat_types_list': ['random_k'],
                    'clause_ratios_list': [3.0],
                    'graph_types': ['random', 'planar'],
                    'num_vertices_list': [5, 8, 10],
                    'grid_sizes': [3, 4],
                    'difficulty_levels': ['easy', 'medium', 'hard'],
                    'word_lengths': [3, 4, 5],
                    'puzzle_types': ['addition'],
                    'num_equations_list': [2, 3],
                    'constraint_types': ['standard'],
                    'matrix_counts_list': [3, 4, 5],
                    'dimension_patterns': ['uniform'],
                    'num_projects_list': [3, 4, 5],
                    'technique_levels': ['basic'],
                }

                for param_name, default_val in hard_task_defaults.items():
                    if param_name in init_params:
                        common_kwargs[param_name] = eval_params.get(param_name, default_val)

                # Only pass parameters the constructor actually accepts
                # If the task uses **kwargs, pass all common params through
                if 'kwargs' in init_params:
                    filtered_kwargs = common_kwargs
                else:
                    filtered_kwargs = {k: v for k, v in common_kwargs.items() if k in init_params}

                try:
                    task_instance = task_class(**filtered_kwargs)
                except TypeError as init_error:
                    self.logger.error(f"❌ Failed to initialize task {task_name}: {init_error}")
                    self.logger.debug(f"   Constructor params: {init_params}")
                    self.logger.debug(f"   Provided kwargs: {set(filtered_kwargs.keys())}")
                    overall_stats['failed_tasks'] += 1
                    continue

                # Run task evaluation
                # Tasks have different run_evaluation signatures:
                # - Some accept list_sizes (easy list-based, medium, hard)
                # - Some accept no parameters (comparison, division, etc.)
                # We try with list_sizes first, then fallback to no args
                list_sizes = eval_params.get('list_sizes') or [8, 16, 32]
                sig = inspect.signature(task_instance.run_evaluation)
                params = sig.parameters

                if 'list_sizes' in params:
                    task_result = task_instance.run_evaluation(list_sizes=list_sizes)
                elif len(params) > 0 and list(params.keys())[0] != 'self':
                    # Has positional args, try passing list_sizes positionally
                    try:
                        task_result = task_instance.run_evaluation(list_sizes)
                    except TypeError:
                        task_result = task_instance.run_evaluation()
                else:
                    task_result = task_instance.run_evaluation()

                task_results[task_name] = task_result
                overall_stats['completed_tasks'] += 1

                # Update overall statistics
                # Handle both list and dict return formats
                if isinstance(task_result, list):
                    # task_result is a list of metrics, compute summary
                    if task_result:
                        avg_accuracy = sum(m.get('accuracy', 0) for m in task_result) / len(task_result)
                        overall_stats['total_evaluations'] += len(task_result)
                        overall_stats['successful_evaluations'] += sum(1 for m in task_result if m.get('accuracy', 0) > 0)
                    else:
                        avg_accuracy = 0
                elif isinstance(task_result, dict) and 'summary' in task_result:
                    summary = task_result['summary']
                    overall_stats['total_evaluations'] += summary.get('total_datapoints', 0)
                    overall_stats['successful_evaluations'] += summary.get('successful_datapoints', 0)
                    avg_accuracy = summary.get('avg_accuracy', 0)
                elif isinstance(task_result, dict) and 'overall_accuracy' in task_result:
                    # Handle tasks with custom run_evaluation (e.g., RobustTowerHanoiTask)
                    avg_accuracy = task_result.get('overall_accuracy', 0)
                    # Count fold results as evaluations
                    fold_results = task_result.get('fold_results', [])
                    if fold_results:
                        for fold in fold_results:
                            total = fold.get('total_count', 0)
                            correct = fold.get('correct_count', 0)
                            overall_stats['total_evaluations'] += total
                            overall_stats['successful_evaluations'] += correct
                else:
                    avg_accuracy = 0

                self.logger.info(f"✅ Task {task_name} completed: accuracy={avg_accuracy:.3f}")

            except Exception as e:
                self.logger.error(f"❌ Task {task_name} failed: {e}")
                overall_stats['failed_tasks'] += 1
                continue

        # Calculate final statistics
        end_time = time.time()
        total_duration = end_time - start_time

        # Aggregate results
        final_results = self._aggregate_results(task_results, overall_stats, total_duration)

        # Add reproducibility metadata (Phase 17)
        try:
            from .eval_fingerprint import compute_eval_fingerprint, get_environment_info
            from importlib.metadata import version as _pkg_version, PackageNotFoundError
            try:
                beyondbench_version = _pkg_version("beyondbench")
            except PackageNotFoundError:
                from beyondbench import __version__ as beyondbench_version  # type: ignore[assignment]

            final_results['eval_fingerprint'] = compute_eval_fingerprint(
                model_id=str(self.model_handler.get_model_info().get('model_name', '')),
                config=eval_params,
                seed=eval_params.get('seed'),
                task_list=task_list,
                beyondbench_version=beyondbench_version,
            )
            final_results['environment'] = get_environment_info()
            self.logger.info(f"🔑 Eval fingerprint: {final_results['eval_fingerprint']}")
        except Exception as _fp_err:
            self.logger.warning(f"Could not compute eval fingerprint: {_fp_err}")

        # Save results
        self._save_results(final_results)

        self.logger.info(f"🎉 Evaluation completed in {total_duration:.1f}s")
        self.logger.info(f"📊 Results: {overall_stats['completed_tasks']}/{overall_stats['total_tasks']} tasks completed")

        return final_results

    def _aggregate_results(
        self,
        task_results: Dict[str, Any],
        overall_stats: Dict[str, Any],
        total_duration: float
    ) -> Dict[str, Any]:
        """Aggregate results from all tasks."""

        # Calculate overall metrics
        total_tasks = len(task_results)
        if total_tasks == 0:
            return {
                'summary': {'error': 'No tasks completed successfully'},
                'task_results': {},
                'overall_stats': overall_stats
            }

        # Aggregate accuracy across all tasks
        accuracies = []
        success_rates = []
        token_counts = []

        for task_name, result in task_results.items():
            if isinstance(result, list) and result:
                # List of per-fold/per-config metrics
                task_acc = sum(m.get('accuracy', 0) for m in result) / len(result)
                accuracies.append(task_acc)
                task_success = sum(1 for m in result if m.get('successful_generations', 0) > 0) / len(result)
                success_rates.append(task_success)
            elif isinstance(result, dict) and 'summary' in result:
                summary = result['summary']
                if 'avg_accuracy' in summary:
                    accuracies.append(summary['avg_accuracy'])
                if 'success_rate' in summary:
                    success_rates.append(summary['success_rate'])
                if 'total_tokens' in summary:
                    token_counts.append(summary.get('total_tokens', 0))
            elif isinstance(result, dict) and 'overall_accuracy' in result:
                # Handle tasks with custom run_evaluation (e.g., RobustTowerHanoiTask)
                accuracies.append(result['overall_accuracy'])

        # Calculate aggregate metrics
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        total_tokens = sum(token_counts)

        # Create comprehensive summary
        summary = {
            'total_duration': total_duration,
            'total_tasks': total_tasks,
            'completed_tasks': overall_stats['completed_tasks'],
            'failed_tasks': overall_stats['failed_tasks'],
            'total_evaluations': overall_stats['total_evaluations'],
            'successful_evaluations': overall_stats['successful_evaluations'],
            'success_rate': overall_stats['successful_evaluations'] / max(1, overall_stats['total_evaluations']),
            'avg_accuracy': avg_accuracy,
            'avg_success_rate': avg_success_rate,
            'total_tokens': total_tokens,
            'evaluations_per_second': overall_stats['total_evaluations'] / max(1, total_duration),
        }

        return {
            'summary': summary,
            'task_results': task_results,
            'overall_stats': overall_stats,
            'model_info': self.model_handler.get_model_info(),
            'evaluation_config': {
                'suite': 'multiple' if len(task_results) > 1 else 'single',
                'tasks': list(task_results.keys()),
                'output_dir': str(self.output_dir)
            }
        }

    @staticmethod
    def _json_serializer(obj):
        """Handle non-serializable types for JSON output."""
        import numpy as np
        from enum import Enum
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return str(obj)

    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive results to files."""

        try:
            # Save main results
            results_file = self.output_dir / "final_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=self._json_serializer)

            # Save summary only
            summary_file = self.output_dir / "evaluation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results['summary'], f, indent=2, ensure_ascii=False, default=self._json_serializer)

            # Save model statistics
            if hasattr(self.model_handler, 'get_statistics'):
                stats_file = self.output_dir / "model_statistics.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(self.model_handler.get_statistics(), f, indent=2, ensure_ascii=False, default=self._json_serializer)

            self.logger.info(f"💾 Results saved to {self.output_dir}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")

    def get_available_tasks(self, suite: str = "all") -> Dict[str, List[str]]:
        """Get available tasks for a suite."""
        return self.task_registry.get_available_tasks(suite)

    def validate_configuration(self) -> bool:
        """Validate evaluation configuration."""
        try:
            # Check model handler
            if not self.model_handler:
                self.logger.error("No model handler provided")
                return False

            # Check output directory
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)

            # Check task registry
            if not self.task_registry:
                self.logger.error("Task registry not initialized")
                return False

            self.logger.info("✅ Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
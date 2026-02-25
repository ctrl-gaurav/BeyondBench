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

                # Initialize task with all required parameters
                task_instance = task_class(
                    model_handler=self.model_handler,
                    output_dir=str(self.output_dir / "task_results"),
                    min_val=eval_params.get('range_min', 1),
                    max_val=eval_params.get('range_max', 100),
                    num_folds=folds,
                    num_samples=datapoints,
                    store_details=self.store_details,
                    temperature=eval_params.get('temperature', 0.7),
                    top_p=eval_params.get('top_p', 0.9),
                    max_tokens=eval_params.get('max_tokens', 1024),
                    seed=eval_params.get('seed'),
                    max_retries=self.max_retries
                )

                # Run task evaluation
                # BaseTask.run_evaluation only accepts list_sizes parameter
                list_sizes = eval_params.get('list_sizes', [8, 16, 32])
                task_result = task_instance.run_evaluation(list_sizes=list_sizes)

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
            if 'summary' in result:
                summary = result['summary']
                if 'avg_accuracy' in summary:
                    accuracies.append(summary['avg_accuracy'])
                if 'success_rate' in summary:
                    success_rates.append(summary['success_rate'])
                if 'total_tokens' in summary:
                    token_counts.append(summary.get('total_tokens', 0))

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

    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive results to files."""

        try:
            # Save main results
            results_file = self.output_dir / "final_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Save summary only
            summary_file = self.output_dir / "evaluation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results['summary'], f, indent=2, ensure_ascii=False)

            # Save model statistics
            if hasattr(self.model_handler, 'get_statistics'):
                stats_file = self.output_dir / "model_statistics.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(self.model_handler.get_statistics(), f, indent=2, ensure_ascii=False)

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
"""
Interval Scheduling Task

Evaluates LLM's ability to find the maximum number of non-overlapping intervals.
Uses the greedy algorithm (sort by end time) for ground truth.

Data point: {'intervals': [(s1,e1), ...], 'answer': max_count}
Output: int
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from ...core.base_task import BaseTask


class IntervalSchedulingSolver:
    """Greedy interval scheduling solver."""

    @staticmethod
    def max_non_overlapping(intervals: List[Tuple[int, int]]) -> int:
        """Returns max count of non-overlapping intervals (greedy by end time)."""
        if not intervals:
            return 0
        sorted_intervals = sorted(intervals, key=lambda x: x[1])
        count = 1
        last_end = sorted_intervals[0][1]
        for start, end in sorted_intervals[1:]:
            if start >= last_end:  # non-overlapping: start >= previous end
                count += 1
                last_end = end
        return count


class IntervalSchedulingTask(BaseTask):
    """Maximum non-overlapping interval scheduling task."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=7, **kwargs):
        self._task_name = "interval_scheduling"
        self.problem_size = problem_size
        super().__init__(
            model_handler=model_handler, output_dir=output_dir,
            min_val=min_val, max_val=max_val, num_folds=num_folds,
            num_samples=num_samples, store_details=store_details,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            seed=seed, **kwargs
        )

    @property
    def task_name(self):
        return self._task_name

    def generate_data(self, **kwargs):
        list_size = kwargs.get('list_size', None)
        n_intervals = list_size if list_size else self.problem_size

        test_cases = []
        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 61)
            n = rng.randint(5, min(10, max(5, n_intervals)))
            intervals = []
            for _ in range(n):
                start = rng.randint(0, 18)
                end = start + rng.randint(1, 6)
                intervals.append((start, end))
            answer = IntervalSchedulingSolver.max_non_overlapping(intervals)
            test_cases.append({
                'intervals': intervals,
                'num_intervals': n,
                'answer': answer,
                'ground_truth': answer,
            })
        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        intervals = data_point['intervals']

        rows = []
        for idx, (s, e) in enumerate(intervals, 1):
            rows.append(f"  Interval {idx}: start={s}, end={e}")
        interval_text = "\n".join(rows)

        prompt = f"""You are solving the Interval Scheduling Maximization problem.

Given intervals (each with a start and end time):
{interval_text}

Find the maximum number of non-overlapping intervals you can select.

Two intervals [s1, e1) and [s2, e2) overlap if s2 < e1 (i.e., they share time).
They are non-overlapping if s2 >= e1.

Greedy algorithm: Sort by end time. Always pick the interval that ends earliest
and doesn't conflict with the last selected interval.

Show your step-by-step selection, then state the maximum count.

Express your final answer as \\boxed{{max_count}}."""
        return prompt

    def evaluate_response(self, response: str, data_point: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = data_point['answer']
        predicted = self._parse_answer(response)

        accuracy = 1 if (predicted is not None and int(predicted) == int(ground_truth)) else 0
        instruction_followed = 1 if predicted is not None else 0

        return {
            'accuracy': accuracy,
            'instruction_followed': instruction_followed,
            'ground_truth': ground_truth,
            'predicted_answer': predicted,
        }

    def _parse_answer(self, response: str) -> Optional[int]:
        import re
        m = re.search(r'\\boxed\{(\d+)\}', response)
        if m:
            return int(m.group(1))
        m = re.search(r'(?:maximum|max|select)[^\d]*(\d+)\s*(?:interval|non-overlap)', response, re.IGNORECASE)
        if m:
            return int(m.group(1))
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return int(nums[-1])
        return None

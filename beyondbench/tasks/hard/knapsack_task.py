"""
0/1 Knapsack Task

Evaluates LLM's ability to solve the 0/1 knapsack problem using dynamic programming.

Data point: {'items': [(weight, value), ...], 'capacity': W, 'answer': max_value}
Output: int (maximum achievable value)
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from ...core.base_task import BaseTask


class KnapsackSolver:
    """DP-based 0/1 knapsack solver."""

    @staticmethod
    def solve(items: List[Tuple[int, int]], capacity: int) -> int:
        """Returns maximum value achievable within given capacity."""
        n = len(items)
        # dp[w] = max value with capacity w
        dp = [0] * (capacity + 1)
        for weight, value in items:
            for w in range(capacity, weight - 1, -1):
                dp[w] = max(dp[w], dp[w - weight] + value)
        return dp[capacity]


class KnapsackTask(BaseTask):
    """0/1 Knapsack optimization task."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=6, **kwargs):
        self._task_name = "knapsack"
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
        """Generate evaluation data points."""
        list_size = kwargs.get('list_size', None)
        n_items = list_size if list_size else self.problem_size

        test_cases = []
        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 17)
            n = rng.randint(4, min(8, max(4, n_items)))
            items = [(rng.randint(1, 15), rng.randint(1, 30)) for _ in range(n)]
            capacity = rng.randint(10, 30)
            answer = KnapsackSolver.solve(items, capacity)
            test_cases.append({
                'items': items,
                'capacity': capacity,
                'num_items': n,
                'answer': answer,
                'ground_truth': answer,
            })
        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        items = data_point['items']
        capacity = data_point['capacity']

        item_lines = []
        for idx, (w, v) in enumerate(items, 1):
            item_lines.append(f"  Item {idx}: weight={w}, value={v}")
        items_text = "\n".join(item_lines)

        prompt = f"""You are solving a 0/1 Knapsack problem. Each item can be taken at most once.

Items (weight, value):
{items_text}

Knapsack capacity: {capacity}

Find the maximum total value you can achieve without exceeding the capacity.

Use dynamic programming: dp[w] = max value achievable with capacity w.

Show your work step by step, then express your final answer as \\boxed{{max_value}}."""
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
        m = re.search(r'(?:maximum|max|total)\s+(?:value|profit)[^\d]*(\d+)', response, re.IGNORECASE)
        if m:
            return int(m.group(1))
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return int(nums[-1])
        return None

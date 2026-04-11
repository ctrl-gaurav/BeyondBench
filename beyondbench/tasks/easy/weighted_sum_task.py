"""
Weighted Sum Task - sum of i * a[i-1] for 1-indexed positions.
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_sum


class WeightedSumTask(BaseTask):
    """Compute weighted sum: sum of position * element (1-indexed)."""

    @property
    def task_name(self):
        return "weighted_sum"

    def generate_data(self, list_size=8):
        if self.seed is not None:
            random.seed(self.seed)
        pool = range(self.min_val, self.max_val + 1)
        if list_size > len(pool):
            return [[random.randint(self.min_val, self.max_val) for _ in range(list_size)]
                    for _ in range(self.num_samples)]
        return [random.sample(pool, list_size) for _ in range(self.num_samples)]

    def create_prompt(self, data_point):
        return (
            f"Calculate the weighted sum of the list {data_point}. "
            f"The weighted sum is defined as: multiply each element by its 1-indexed position, "
            f"then sum all results. For example, for [a, b, c]: 1*a + 2*b + 3*c.\n\n"
            f"Your final answer must be in the format \\boxed{{answer}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = sum((i + 1) * v for i, v in enumerate(data_point))
        parsed = parse_sum(response)
        instruction_followed = parsed is not None
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed == ground_truth else 0
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed,
        }

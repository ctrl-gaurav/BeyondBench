"""
Cumulative Sum Task - return prefix sum list.
"""

import random
from ...core.base_task import BaseTask
from ...parsers.cumulative_sum_parsing import parse_cumulative_sum_answer


class CumulativeSumTask(BaseTask):
    """Compute cumulative (prefix) sum list."""

    @property
    def task_name(self):
        return "cumulative_sum"

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
            f"Calculate the cumulative (prefix) sum of the list {data_point}. "
            f"The cumulative sum at position i is the sum of all elements from index 0 to i (inclusive). "
            f"Return the full list of cumulative sums.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = []
        cumsum = 0
        for v in data_point:
            cumsum += v
            ground_truth.append(cumsum)

        parsed, instruction_followed = parse_cumulative_sum_answer(response)
        accuracy = 0
        if parsed is not None and len(parsed) == len(ground_truth):
            if all(int(round(a)) == b for a, b in zip(parsed, ground_truth)):
                accuracy = 1
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

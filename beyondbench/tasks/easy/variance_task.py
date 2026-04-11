"""
Variance Task - population variance mean((x - mu)^2).
"""

import random
from ...core.base_task import BaseTask
from ...utils.parsing import parse_statistical_result


class VarianceTask(BaseTask):
    """Compute population variance of a list of numbers."""

    @property
    def task_name(self):
        return "variance"

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
            f"Calculate the population variance of the list {data_point}. "
            f"Population variance = mean of (x - mean)^2 for all x in the list. "
            f"Round your answer to 2 decimal places.\n\n"
            f"Your final answer must be in the format \\boxed{{answer}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        mu = sum(data_point) / len(data_point)
        ground_truth = sum((x - mu) ** 2 for x in data_point) / len(data_point)
        ground_truth = round(ground_truth, 2)

        parsed = parse_statistical_result(response)
        instruction_followed = parsed is not None
        accuracy = 0
        if instruction_followed:
            try:
                accuracy = 1 if abs(float(parsed) - ground_truth) < 1e-2 else 0
            except (TypeError, ValueError):
                pass
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

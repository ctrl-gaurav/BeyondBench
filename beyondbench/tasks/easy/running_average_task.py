"""
Running Average Task - return cumulative running average list.
"""

import random
from ...core.base_task import BaseTask
from ...parsers.running_average_parsing import parse_running_average_answer


class RunningAverageTask(BaseTask):
    """Compute running (cumulative) average at each index."""

    @property
    def task_name(self):
        return "running_average"

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
            f"Calculate the running average of the list {data_point}. "
            f"The running average at position i is the mean of all elements from index 0 to i (inclusive). "
            f"Return the full list of running averages as a list of floats rounded to 4 decimal places.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        n = len(data_point)
        ground_truth = []
        cumsum = 0
        for i, v in enumerate(data_point):
            cumsum += v
            ground_truth.append(round(cumsum / (i + 1), 4))

        parsed, instruction_followed = parse_running_average_answer(response)
        accuracy = 0
        if parsed is not None and len(parsed) == len(ground_truth):
            if all(abs(a - b) < 1e-4 for a, b in zip(parsed, ground_truth)):
                accuracy = 1
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

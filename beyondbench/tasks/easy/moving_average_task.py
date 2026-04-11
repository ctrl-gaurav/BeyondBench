"""
Moving Average Task - k-window moving average.
"""

import random
from ...core.base_task import BaseTask
from ...parsers.moving_average_parsing import parse_moving_average_answer


class MovingAverageTask(BaseTask):
    """Compute k-window moving average (list of length n-k+1)."""

    @property
    def task_name(self):
        return "moving_average"

    def generate_data(self, list_size=8, window=3):
        if self.seed is not None:
            random.seed(self.seed)
        k = window
        data = []
        for _ in range(self.num_samples):
            pool = range(self.min_val, self.max_val + 1)
            if list_size > len(pool):
                lst = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            else:
                lst = random.sample(pool, list_size)
            answer = [round(sum(lst[i:i+k]) / k, 4) for i in range(len(lst) - k + 1)]
            data.append({"list": lst, "window": k, "answer": answer})
        return data

    def create_prompt(self, data_point):
        lst = data_point["list"]
        k = data_point["window"]
        return (
            f"Calculate the {k}-element moving average of the list {lst}. "
            f"For each window of {k} consecutive elements starting at index 0, compute the average. "
            f"The result has {len(lst) - k + 1} elements. Round each value to 4 decimal places.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point["answer"]
        parsed, instruction_followed = parse_moving_average_answer(response)
        accuracy = 0
        if parsed is not None and len(parsed) == len(ground_truth):
            if all(abs(a - b) < 1e-4 for a, b in zip(parsed, ground_truth)):
                accuracy = 1
        return {
            "input_list": data_point["list"],
            "window": data_point["window"],
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

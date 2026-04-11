"""
Second Minimum Task - second smallest distinct element.
"""

import random
from ...core.base_task import BaseTask
from ...utils.parsing import parse_number


class SecondMinimumTask(BaseTask):
    """Find the second smallest distinct element in a list."""

    @property
    def task_name(self):
        return "second_minimum"

    def generate_data(self, list_size=8):
        if self.seed is not None:
            random.seed(self.seed)
        data = []
        for _ in range(self.num_samples):
            # Ensure at least 2 distinct values
            while True:
                pool = range(self.min_val, self.max_val + 1)
                if list_size > len(pool):
                    lst = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
                else:
                    lst = random.sample(pool, list_size)
                if len(set(lst)) >= 2:
                    break
            data.append(lst)
        return data

    def create_prompt(self, data_point):
        return (
            f"Find the second smallest distinct value in the list {data_point}. "
            f"Consider only unique values. For example, in [3, 1, 2, 1], the distinct values are 1, 2, 3, "
            f"so the second smallest is 2.\n\n"
            f"Your final answer must be in the format \\boxed{{answer}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        distinct = sorted(set(data_point))
        ground_truth = distinct[1]
        parsed = parse_number(response)
        instruction_followed = parsed is not None
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if int(round(parsed)) == ground_truth else 0
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

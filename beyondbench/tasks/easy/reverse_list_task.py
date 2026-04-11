"""
Reverse List Task - reverse the list.
"""

import random
from ...core.base_task import BaseTask
from ...parsers.reverse_list_parsing import parse_reverse_list_answer


class ReverseListTask(BaseTask):
    """Reverse a list of integers."""

    @property
    def task_name(self):
        return "reverse_list"

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
            f"Reverse the list {data_point}. Return the elements in reverse order.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = list(reversed(data_point))
        parsed, instruction_followed = parse_reverse_list_answer(response)
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

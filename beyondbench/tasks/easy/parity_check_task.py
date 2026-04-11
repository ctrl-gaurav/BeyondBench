"""
Parity Check Task - determine if product of all elements is even or odd.
"""

import random
from ...core.base_task import BaseTask
from ...parsers.parity_check_parsing import parse_parity_check_answer


class ParityCheckTask(BaseTask):
    """Determine whether the product of all list elements is even or odd."""

    @property
    def task_name(self):
        return "parity_check"

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
            f"Determine whether the product of all elements in the list {data_point} is even or odd.\n\n"
            f"Answer with exactly one word: 'even' if the product is even, 'odd' if the product is odd.\n"
            f"Your final answer must be in the format \\boxed{{even}} or \\boxed{{odd}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        product = 1
        for v in data_point:
            product *= v
        ground_truth = "even" if product % 2 == 0 else "odd"

        parsed, instruction_followed = parse_parity_check_answer(response)
        accuracy = 0
        if parsed is not None:
            accuracy = 1 if parsed.lower() == ground_truth else 0
        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

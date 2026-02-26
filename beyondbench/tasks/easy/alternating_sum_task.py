"""
Alternating Sum Task - Calculate alternating sum (add odd indices, subtract even indices)
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class AlternatingSumTask(BaseTask):
    """Implementation of the alternating sum task"""

    @property
    def task_name(self):
        return "alternating_sum"

    def generate_data(self, list_size=8):
        """Generate random lists of numbers within specified range"""
        if self.seed is not None:
            random.seed(self.seed)

        pool = range(self.min_val, self.max_val + 1)
        if list_size > len(pool):
            return [[random.randint(self.min_val, self.max_val) for _ in range(list_size)]
                    for _ in range(self.num_samples)]
        return [random.sample(pool, list_size)
                for _ in range(self.num_samples)]

    def create_prompt(self, data_point):
        """Create prompt for alternating sum task"""
        return (f"Calculate the alternating sum of the list {data_point}. "
                f"Start by adding the first element, then subtract the second, add the third, etc.\n\n"
                f"Your final answer must be in the format \\boxed{{sum}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for alternating sum task"""
        # Calculate ground truth
        ground_truth = sum(data_point[i] if i % 2 == 0 else -data_point[i] for i in range(len(data_point)))

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
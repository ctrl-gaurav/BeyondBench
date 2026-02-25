"""
Range Task - Calculate the difference between maximum and minimum values
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class RangeTask(BaseTask):
    """Implementation of the range (max - min) task"""

    @property
    def task_name(self):
        return "range"

    def generate_data(self, list_size=8):
        """Generate random lists of numbers within specified range"""
        if self.seed is not None:
            random.seed(self.seed)

        return [random.sample(range(self.min_val, self.max_val + 1), list_size)
                for _ in range(self.num_samples)]

    def create_prompt(self, data_point):
        """Create prompt for range task"""
        return (f"Calculate the range (difference between maximum and minimum) of the following "
                f"list of numbers: {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{range}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for range task"""
        # Calculate ground truth
        ground_truth = max(data_point) - min(data_point)

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
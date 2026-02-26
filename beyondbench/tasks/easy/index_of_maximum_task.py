"""
Index of Maximum Task - Find the index of the first occurrence of the maximum element
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_index


class IndexOfMaximumTask(BaseTask):
    """Implementation of the index of maximum task"""

    @property
    def task_name(self):
        return "index_of_maximum"

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
        """Create prompt for index of maximum task"""
        return (f"Find the index (0-based position) of the maximum element in the list {data_point}. "
                f"If there are multiple maximum elements, return the index of the first occurrence.\n\n"
                f"Your final answer must be in the format \\boxed{{index}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for index of maximum task"""
        # Calculate ground truth
        max_value = max(data_point)
        ground_truth = data_point.index(max_value)

        # Parse model response
        parsed_answer = parse_index(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "max_value": max_value,
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
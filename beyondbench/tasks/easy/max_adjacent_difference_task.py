"""
Maximum Adjacent Difference Task - Find the maximum absolute difference between adjacent elements
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class MaxAdjacentDifferenceTask(BaseTask):
    """Implementation of the maximum adjacent difference task"""

    @property
    def task_name(self):
        return "max_adjacent_difference"

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
        """Create prompt for maximum adjacent difference task"""
        return (f"Find the maximum absolute difference between any two adjacent elements in the list {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{difference}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for maximum adjacent difference task"""
        # Calculate ground truth
        if len(data_point) < 2:
            ground_truth = 0
        else:
            ground_truth = max(abs(data_point[i+1] - data_point[i]) for i in range(len(data_point)-1))

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
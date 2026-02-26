"""
Count Greater Than Previous Task - Count elements that are greater than the previous element
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class CountGreaterThanPreviousTask(BaseTask):
    """Implementation of the count greater than previous task"""

    @property
    def task_name(self):
        return "count_greater_than_previous"

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
        """Create prompt for count greater than previous task"""
        return (f"Count how many elements in the list are greater than the element that comes "
                f"immediately before them: {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{count}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for count greater than previous task"""
        # Calculate ground truth
        ground_truth = sum(1 for i in range(1, len(data_point)) if data_point[i] > data_point[i-1])

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
"""
Sum of Indices of Maximum Elements Task - Find sum of all indices where maximum value occurs
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class SumOfMaxIndicesTask(BaseTask):
    """Implementation of the sum of indices of maximum elements task"""

    @property
    def task_name(self):
        return "sum_of_max_indices"

    def generate_data(self, list_size=8):
        """Generate random lists with controlled maximum duplicates"""
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            # Create list where maximum appears 1-3 times
            max_occurrences = random.randint(1, 3)
            max_value = random.randint(self.min_val + 10, self.max_val)

            # Fill list with smaller values
            other_values = [random.randint(self.min_val, max_value - 1)
                          for _ in range(list_size - max_occurrences)]

            # Add maximum values
            numbers = other_values + [max_value] * max_occurrences
            random.shuffle(numbers)
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for sum of max indices task"""
        return (f"Find the sum of all indices (0-based) where the maximum value occurs in the list {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{sum}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for sum of max indices task"""
        # Calculate ground truth
        max_value = max(data_point)
        max_indices = [i for i, x in enumerate(data_point) if x == max_value]
        ground_truth = sum(max_indices)

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "max_value": max_value,
            "max_indices": max_indices,
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
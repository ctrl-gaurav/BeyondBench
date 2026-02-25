"""
Second Maximum Task - Find the second largest element in a list
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class SecondMaximumTask(BaseTask):
    """Implementation of the second maximum task"""

    @property
    def task_name(self):
        return "second_maximum"

    def generate_data(self, list_size=8):
        """Generate random lists of numbers within specified range"""
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            # Generate list ensuring we have at least 2 unique values
            while True:
                numbers = random.sample(range(self.min_val, self.max_val + 1), list_size)
                if len(set(numbers)) >= 2:  # Ensure at least 2 unique values
                    break
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for second maximum task"""
        return (f"Find the second maximum number from the given list of numbers. "
                f"List = {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{second_maximum}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for second maximum task"""
        # Calculate ground truth
        sorted_unique = sorted(set(data_point), reverse=True)
        ground_truth = sorted_unique[1] if len(sorted_unique) >= 2 else None

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed and ground_truth is not None:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
"""
Sum of Digits Task - Calculate the sum of all digits in all numbers in the list
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class SumOfDigitsTask(BaseTask):
    """Implementation of the sum of digits task"""

    @property
    def task_name(self):
        return "sum_of_digits"

    def generate_data(self, list_size=8):
        """Generate random lists of numbers within specified range"""
        if self.seed is not None:
            random.seed(self.seed)

        # Generate numbers with multiple digits for more interesting sum
        data = []
        for _ in range(self.num_samples):
            numbers = []
            for _ in range(list_size):
                # Generate numbers between 10 and max_val for multi-digit numbers
                min_val = max(10, self.min_val)
                numbers.append(random.randint(min_val, self.max_val))
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for sum of digits task"""
        return (f"Calculate the sum of all digits in all numbers in the list {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{sum}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for sum of digits task"""
        # Calculate ground truth
        ground_truth = sum(sum(int(digit) for digit in str(abs(num))) for num in data_point)

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
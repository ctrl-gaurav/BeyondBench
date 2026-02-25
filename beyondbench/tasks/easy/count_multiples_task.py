"""
Count Multiples of K Task - Count elements that are multiples of a given number K
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class CountMultiplesTask(BaseTask):
    """Implementation of the count multiples of K task"""

    @property
    def task_name(self):
        return "count_multiples"

    def generate_data(self, list_size=8):
        """Generate random lists with some multiples of 3"""
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        k = 3  # We'll use 3 as the fixed divisor for consistency

        for _ in range(self.num_samples):
            numbers = []

            # Add some multiples of k
            multiples_count = random.randint(list_size // 3, 2 * list_size // 3)
            for _ in range(multiples_count):
                multiple = random.randint(max(1, self.min_val // k), self.max_val // k) * k
                if self.min_val <= multiple <= self.max_val:
                    numbers.append(multiple)

            # Fill remaining with non-multiples
            remaining = list_size - len(numbers)
            for _ in range(remaining):
                while True:
                    num = random.randint(self.min_val, self.max_val)
                    if num % k != 0:  # Not a multiple of k
                        numbers.append(num)
                        break

            random.shuffle(numbers)
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for count multiples task"""
        return (f"Count how many numbers in the list {data_point} are multiples of 3.\n\n"
                f"Your final answer must be in the format \\boxed{{count}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for count multiples task"""
        # Calculate ground truth (multiples of 3)
        k = 3
        ground_truth = sum(1 for x in data_point if x % k == 0)

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "k": k,
            "multiples": [x for x in data_point if x % k == 0],
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
"""
Count Perfect Squares Task - Count how many perfect squares are in the list
"""

import random
import math
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class CountPerfectSquaresTask(BaseTask):
    """Implementation of the count perfect squares task"""

    @property
    def task_name(self):
        return "count_perfect_squares"

    def generate_data(self, list_size=8):
        """Generate random lists with some perfect squares"""
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            numbers = []

            # Add some perfect squares
            perfect_squares_count = random.randint(1, list_size // 2)
            perfect_squares = [i*i for i in range(1, int(math.sqrt(self.max_val)) + 1)
                             if self.min_val <= i*i <= self.max_val]

            for _ in range(min(perfect_squares_count, len(perfect_squares))):
                if perfect_squares:
                    numbers.append(random.choice(perfect_squares))

            # Fill remaining with non-perfect squares
            remaining = list_size - len(numbers)
            for _ in range(remaining):
                while True:
                    num = random.randint(self.min_val, self.max_val)
                    if int(math.sqrt(num))**2 != num:  # Not a perfect square
                        numbers.append(num)
                        break

            random.shuffle(numbers)
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for count perfect squares task"""
        return (f"Count how many perfect squares are in the list {data_point}. "
                f"A perfect square is an integer that is the square of another integer.\n\n"
                f"Your final answer must be in the format \\boxed{{count}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for count perfect squares task"""
        # Calculate ground truth
        ground_truth = sum(1 for x in data_point if x >= 0 and int(math.sqrt(x))**2 == x)

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "perfect_squares": [x for x in data_point if x >= 0 and int(math.sqrt(x))**2 == x],
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
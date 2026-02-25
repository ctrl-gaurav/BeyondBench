"""
Count Palindromic Numbers Task - Count how many numbers in the list are palindromes
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class CountPalindromicTask(BaseTask):
    """Implementation of the count palindromic numbers task"""

    @property
    def task_name(self):
        return "count_palindromic"

    def generate_data(self, list_size=8):
        """Generate random lists with some palindromic numbers"""
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            numbers = []

            # Add some palindromic numbers
            palindromic_count = random.randint(1, list_size // 2)
            for _ in range(palindromic_count):
                # Generate palindromic numbers
                if random.choice([True, False]):
                    # Single digit (always palindromic)
                    numbers.append(random.randint(1, 9))
                else:
                    # Multi-digit palindromes
                    palindromes = [11, 22, 33, 44, 55, 66, 77, 88, 99,
                                 101, 111, 121, 131, 141, 151, 161, 171, 181, 191,
                                 202, 212, 222, 232, 242, 252, 262, 272, 282, 292,
                                 303, 313, 323, 333, 343, 353, 363, 373, 383, 393]
                    # Filter based on range
                    valid_palindromes = [p for p in palindromes if self.min_val <= p <= self.max_val]
                    if valid_palindromes:
                        numbers.append(random.choice(valid_palindromes))
                    else:
                        numbers.append(random.randint(1, 9))

            # Fill remaining with non-palindromic numbers
            remaining = list_size - len(numbers)
            for _ in range(remaining):
                while True:
                    num = random.randint(max(10, self.min_val), self.max_val)
                    if str(num) != str(num)[::-1]:  # Not palindromic
                        numbers.append(num)
                        break

            random.shuffle(numbers)
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for count palindromic task"""
        return (f"Count how many palindromic numbers are in the list {data_point}. "
                f"A palindromic number reads the same forwards and backwards.\n\n"
                f"Your final answer must be in the format \\boxed{{count}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for count palindromic task"""
        # Calculate ground truth
        ground_truth = sum(1 for x in data_point if str(x) == str(x)[::-1])

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "palindromic_numbers": [x for x in data_point if str(x) == str(x)[::-1]],
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
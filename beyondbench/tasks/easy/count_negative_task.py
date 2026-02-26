"""
Count Negative Numbers Task - Count how many negative numbers are in the list
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class CountNegativeTask(BaseTask):
    """Implementation of the count negative numbers task"""

    @property
    def task_name(self):
        return "count_negative"

    def generate_data(self, list_size=8):
        """Generate random lists of numbers within specified range (including negatives)"""
        if self.seed is not None:
            random.seed(self.seed)

        # Ensure we have a good mix of positive and negative numbers
        data = []
        for _ in range(self.num_samples):
            # Generate roughly half negative and half positive
            neg_upper = min(-1, self.max_val)
            neg_lower = min(self.min_val, neg_upper)
            pos_lower = max(1, self.min_val)
            pos_upper = max(self.max_val, pos_lower)
            if neg_lower <= neg_upper:
                negatives = [random.randint(neg_lower, neg_upper) for _ in range(list_size // 2)]
            else:
                negatives = []
            positives = [random.randint(pos_lower, pos_upper) for _ in range(list_size - len(negatives))]
            numbers = negatives + positives
            random.shuffle(numbers)
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for count negative task"""
        return (f"Count how many negative numbers are in the list {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{count}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for count negative task"""
        # Calculate ground truth
        ground_truth = sum(1 for x in data_point if x < 0)

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
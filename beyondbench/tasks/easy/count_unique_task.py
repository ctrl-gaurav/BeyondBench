"""
Count Unique Elements Task - Count the number of unique/distinct elements in the list
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class CountUniqueTask(BaseTask):
    """Implementation of the count unique elements task"""

    @property
    def task_name(self):
        return "count_unique"

    def generate_data(self, list_size=8):
        """Generate random lists with controlled duplicates"""
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            # Generate list with some duplicates
            pool_size = self.max_val - self.min_val + 1
            unique_count = random.randint(max(1, min(list_size // 2, pool_size)), min(list_size, pool_size))
            unique_values = random.sample(range(self.min_val, self.max_val + 1), unique_count)

            # Create list with duplicates
            numbers = []
            for _ in range(list_size):
                numbers.append(random.choice(unique_values))

            random.shuffle(numbers)
            data.append(numbers)

        return data

    def create_prompt(self, data_point):
        """Create prompt for count unique task"""
        return (f"Count the number of unique (distinct) elements in the list {data_point}.\n\n"
                f"Your final answer must be in the format \\boxed{{count}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for count unique task"""
        # Calculate ground truth
        ground_truth = len(set(data_point))

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        return {
            "input_list": data_point,
            "unique_elements": sorted(set(data_point)),
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
"""
Local Maxima Count Task - Count elements that are greater than both neighbors
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class LocalMaximaCountTask(BaseTask):
    """Implementation of the local maxima count task"""

    @property
    def task_name(self):
        return "local_maxima_count"

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
        """Create prompt for local maxima count task"""
        return (f"Count how many local maxima exist in the list {data_point}. "
                f"A local maximum is an element that is greater than both its immediate neighbors.\n\n"
                f"Your final answer must be in the format \\boxed{{count}} at the end of your response.")

    def evaluate_response(self, response, data_point):
        """Evaluate model response for local maxima count task"""
        # Calculate ground truth
        ground_truth = sum(1 for i in range(1, len(data_point)-1)
                          if data_point[i] > data_point[i-1] and data_point[i] > data_point[i+1])

        # Parse model response
        parsed_answer = parse_count(response)
        instruction_followed = parsed_answer is not None

        # Calculate accuracy
        accuracy = 0
        if instruction_followed:
            accuracy = 1 if parsed_answer == ground_truth else 0

        # Find local maxima for debugging
        local_maxima_indices = [i for i in range(1, len(data_point)-1)
                               if data_point[i] > data_point[i-1] and data_point[i] > data_point[i+1]]
        local_maxima_values = [data_point[i] for i in local_maxima_indices]

        return {
            "input_list": data_point,
            "local_maxima_indices": local_maxima_indices,
            "local_maxima_values": local_maxima_values,
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
"""
Longest Increasing Subsequence Length Task - Find the length of the longest increasing subsequence
"""

import random
import logging
from ...core.base_task import BaseTask
from ...utils.parsing import parse_count


class LongestIncreasingSubsequenceTask(BaseTask):
    """Implementation of the longest increasing subsequence length task"""

    @property
    def task_name(self):
        return "longest_increasing_subsequence"

    def generate_data(self, list_size=8):
        """Generate random lists of numbers within specified range"""
        if self.seed is not None:
            random.seed(self.seed)

        return [random.sample(range(self.min_val, self.max_val + 1), list_size)
                for _ in range(self.num_samples)]

    def create_prompt(self, data_point):
        """Create prompt for longest increasing subsequence task"""
        return (f"Find the length of the longest increasing subsequence in {data_point}. "
                f"A subsequence maintains relative order but elements don't need to be consecutive.\n\n"
                f"Your final answer must be in the format \\boxed{{length}} at the end of your response.")

    def _calculate_lis_length(self, arr):
        """Calculate the length of the longest increasing subsequence using dynamic programming"""
        if not arr:
            return 0

        n = len(arr)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if arr[j] < arr[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    def evaluate_response(self, response, data_point):
        """Evaluate model response for longest increasing subsequence task"""
        # Calculate ground truth
        ground_truth = self._calculate_lis_length(data_point)

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
"""
GCD / LCM Task

Compute GCD or LCM of two or three integers.
"""

import random
import re
import math
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class GcdLcmTask(BaseTask):
    """Compute GCD or LCM of 2-3 integers."""

    @property
    def task_name(self):
        return "gcd_lcm"

    def _gcd_many(self, nums):
        result = nums[0]
        for n in nums[1:]:
            result = math.gcd(result, n)
        return result

    def _lcm_two(self, a, b):
        return abs(a * b) // math.gcd(a, b)

    def _lcm_many(self, nums):
        result = nums[0]
        for n in nums[1:]:
            result = self._lcm_two(result, n)
        return result

    def generate_data(self, sequence_length=1):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            op = random.choice(['gcd', 'lcm'])
            count = random.choice([2, 3])
            nums = [random.randint(2, 60) for _ in range(count)]
            if op == 'gcd':
                answer = self._gcd_many(nums)
            else:
                answer = self._lcm_many(nums)
                # Keep LCM manageable
                if answer > 10000:
                    # regenerate with smaller numbers
                    nums = [random.randint(2, 20) for _ in range(count)]
                    answer = self._lcm_many(nums)
            data.append({
                'operation': op,
                'numbers': nums,
                'answer': answer,
            })
        return data

    def create_prompt(self, data_point):
        op = data_point['operation'].upper()
        nums_str = ', '.join(map(str, data_point['numbers']))
        return (
            f"Compute the {op} of the following numbers: {nums_str}\n\n"
            f"{'GCD (Greatest Common Divisor) is the largest number that divides all given numbers.' if data_point['operation']=='gcd' else 'LCM (Least Common Multiple) is the smallest number divisible by all given numbers.'}\n\n"
            f"Provide your final answer as \\boxed{{value}}."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point['answer']
        parsed = self._parse_answer(response)
        instruction_followed = parsed is not None
        accuracy = 0
        if parsed is not None:
            try:
                accuracy = 1 if abs(float(parsed) - float(ground_truth)) < 0.5 else 0
            except Exception:
                accuracy = 0
        return {
            'ground_truth': ground_truth,
            'predicted_answer': parsed,
            'accuracy': accuracy,
            'instruction_followed': int(instruction_followed),
            'operation': data_point['operation'],
            'numbers': data_point['numbers'],
            'prompt': self.create_prompt(data_point),
            'model_response': response,
        }

    def _parse_answer(self, response):
        try:
            from ...utils.parsing import parse_sequence_result
            val = parse_sequence_result(response)
            if val is not None:
                return int(round(float(val)))
        except Exception:
            pass
        m = re.search(r'\\boxed\{(-?\d+)\}', response)
        if m:
            return int(m.group(1))
        m = re.search(r'(?:answer|result|gcd|lcm)[^\d-]*(-?\d+)', response, re.I)
        if m:
            return int(m.group(1))
        return None

    def run_evaluation(self, list_sizes):
        all_metrics = []
        for size in list_sizes:
            data = self.generate_data(size)
            for fold in range(self.num_folds):
                metrics = self.run_fold(data, size, fold)
                all_metrics.append(metrics)
        return all_metrics

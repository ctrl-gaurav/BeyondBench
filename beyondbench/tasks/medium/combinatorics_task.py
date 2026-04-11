"""
Combinatorics Task

Compute permutations P(n,r) or combinations C(n,r) with 1 ≤ r ≤ n ≤ 12.
"""

import random
import re
import math
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class CombinatoricsTask(BaseTask):
    """Compute P(n,r) or C(n,r)."""

    @property
    def task_name(self):
        return "combinatorics"

    def generate_data(self, sequence_length=1):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            op = random.choice(['permutation', 'combination'])
            n = random.randint(3, 12)
            r = random.randint(1, n)
            if op == 'permutation':
                answer = math.perm(n, r)
            else:
                answer = math.comb(n, r)
            data.append({
                'operation': op,
                'n': n,
                'r': r,
                'answer': answer,
            })
        return data

    def create_prompt(self, data_point):
        op = data_point['operation']
        n, r = data_point['n'], data_point['r']
        if op == 'permutation':
            formula_note = f"P(n,r) = n! / (n-r)! = the number of ways to arrange {r} items from {n}."
            notation = f"P({n},{r})"
        else:
            formula_note = f"C(n,r) = n! / (r!(n-r)!) = the number of ways to choose {r} items from {n}."
            notation = f"C({n},{r})"
        return (
            f"Compute {notation}.\n\n"
            f"{formula_note}\n\n"
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
            'n': data_point['n'],
            'r': data_point['r'],
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
        m = re.search(r'(?:answer|result|value)[^\d-]*(-?\d+)', response, re.I)
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

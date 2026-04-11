"""
Arithmetic Progression Task

Generate an arithmetic progression with random first term and common difference.
Show k terms; ask for the next one.
"""

import random
import re
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class ArithmeticProgressionTask(BaseTask):
    """Next term in an arithmetic progression."""

    @property
    def task_name(self):
        return "arithmetic_progression"

    def generate_data(self, sequence_length=6):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            first_term = random.randint(-20, 20)
            common_diff = random.choice(list(range(-10, 0)) + list(range(1, 11)))
            sequence = [first_term + i * common_diff for i in range(sequence_length + 1)]
            shown = sequence[:sequence_length]
            next_term = sequence[sequence_length]
            data.append({
                'shown_sequence': shown,
                'first_term': first_term,
                'common_diff': common_diff,
                'answer': next_term,
                'next_term': next_term,
            })
        return data

    def create_prompt(self, data_point):
        seq_str = ', '.join(map(str, data_point['shown_sequence']))
        return (
            f"Find the next term in the following arithmetic progression:\n\n"
            f"{seq_str}, ?\n\n"
            f"An arithmetic progression has a constant difference between consecutive terms. "
            f"Identify that difference, then compute the next term.\n\n"
            f"Provide your final answer as \\boxed{{next_term}}."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point['next_term']
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
            'shown_sequence': data_point['shown_sequence'],
            'common_diff': data_point['common_diff'],
            'prompt': self.create_prompt(data_point),
            'model_response': response,
        }

    def _parse_answer(self, response):
        try:
            from ...utils.parsing import parse_sequence_result
            return parse_sequence_result(response)
        except Exception:
            pass
        # fallback
        m = re.search(r'\\boxed\{(-?\d+)\}', response)
        if m:
            return int(m.group(1))
        m = re.search(r'(?:answer|next term)[^\d-]*(-?\d+)', response, re.I)
        if m:
            return int(m.group(1))
        return None

    def run_evaluation(self, list_sizes):
        all_metrics = []
        for seq_length in list_sizes:
            data = self.generate_data(seq_length)
            for fold in range(self.num_folds):
                metrics = self.run_fold(data, seq_length, fold)
                metrics['sequence_length'] = seq_length
                all_metrics.append(metrics)
        return all_metrics

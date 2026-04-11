"""
Collatz Sequence Task

Collatz rule: if n even → n/2, else → 3n+1.
Generate a starting n, compute k terms, show first k-1, ask for the k-th.
"""

import random
import re
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class CollatzSequenceTask(BaseTask):
    """Next term in a Collatz sequence."""

    @property
    def task_name(self):
        return "collatz_sequence"

    def _collatz_step(self, n: int) -> int:
        return n // 2 if n % 2 == 0 else 3 * n + 1

    def _collatz_sequence(self, start: int, length: int) -> List[int]:
        seq = [start]
        for _ in range(length - 1):
            seq.append(self._collatz_step(seq[-1]))
        return seq

    def generate_data(self, sequence_length=6):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            # Pick a starting value that gives non-trivial sequences
            start = random.randint(6, 50)
            sequence = self._collatz_sequence(start, sequence_length + 1)
            shown = sequence[:sequence_length]
            next_term = sequence[sequence_length]
            data.append({
                'start': start,
                'shown_sequence': shown,
                'answer': next_term,
                'next_term': next_term,
            })
        return data

    def create_prompt(self, data_point):
        seq_str = ', '.join(map(str, data_point['shown_sequence']))
        return (
            f"The following is a Collatz sequence. The rule is: if a term is even, "
            f"the next term is half of it; if a term is odd, the next term is 3 times it plus 1.\n\n"
            f"Sequence: {seq_str}, ?\n\n"
            f"Apply the Collatz rule to the last term to find the next term.\n\n"
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
            'prompt': self.create_prompt(data_point),
            'model_response': response,
        }

    def _parse_answer(self, response):
        try:
            from ...utils.parsing import parse_sequence_result
            return parse_sequence_result(response)
        except Exception:
            pass
        m = re.search(r'\\boxed\{(-?\d+)\}', response)
        if m:
            return int(m.group(1))
        m = re.search(r'(?:next term|answer)[^\d-]*(-?\d+)', response, re.I)
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

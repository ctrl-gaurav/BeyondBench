"""
Pattern Completion Task

Numeric pattern completion: squares, cubes, triangular, powers of 2, factorial, Pascal diagonal.
"""

import random
import re
import math
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class PatternCompletionTask(BaseTask):
    """Complete a standard numeric pattern."""

    @property
    def task_name(self):
        return "pattern_completion"

    _PATTERNS = ['squares', 'cubes', 'triangular', 'powers_of_2', 'factorial', 'pascal_diagonal']

    def _generate_pattern(self, pattern_type, start_offset, length):
        """Generate sequence terms."""
        seq = []
        for k in range(length + 1):
            i = start_offset + k
            if pattern_type == 'squares':
                seq.append(i * i)
            elif pattern_type == 'cubes':
                seq.append(i * i * i)
            elif pattern_type == 'triangular':
                seq.append(i * (i + 1) // 2)
            elif pattern_type == 'powers_of_2':
                seq.append(2 ** i)
            elif pattern_type == 'factorial':
                seq.append(math.factorial(i))
            elif pattern_type == 'pascal_diagonal':
                # C(i+k-1, k-1) for k-th diagonal — use diagonal 2 (triangular) but with offset
                # Simplify: just use C(i, 2) for diagonal 2
                seq.append(math.comb(i, 2))
        return seq

    def generate_data(self, sequence_length=6):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            pattern_type = random.choice(self._PATTERNS)
            # Avoid overly large factorials
            if pattern_type == 'factorial':
                start_offset = random.randint(1, 6)
            elif pattern_type == 'pascal_diagonal':
                start_offset = random.randint(2, 8)
            elif pattern_type == 'squares':
                start_offset = random.randint(1, 10)
            elif pattern_type == 'cubes':
                start_offset = random.randint(1, 6)
            elif pattern_type == 'powers_of_2':
                start_offset = random.randint(0, 5)
            else:
                start_offset = random.randint(1, 8)

            sequence = self._generate_pattern(pattern_type, start_offset, sequence_length)
            shown = sequence[:sequence_length]
            next_term = sequence[sequence_length]
            data.append({
                'pattern_type': pattern_type,
                'start_offset': start_offset,
                'shown_sequence': shown,
                'answer': next_term,
                'next_term': next_term,
            })
        return data

    def create_prompt(self, data_point):
        seq_str = ', '.join(map(str, data_point['shown_sequence']))
        return (
            f"Complete the following numeric pattern:\n\n"
            f"{seq_str}, ?\n\n"
            f"Identify the pattern (e.g., perfect squares, cubes, powers of 2, triangular numbers, "
            f"factorials) and compute the next term.\n\n"
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
            'pattern_type': data_point['pattern_type'],
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

"""
Harmonic Sequence Task

Generate reciprocal sequences 1/a, 1/(a+d), 1/(a+2d), ...
Ask for the next term in fraction or decimal form.
"""

import random
import re
import math
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class HarmonicSequenceTask(BaseTask):
    """Next term in a harmonic (reciprocal) sequence."""

    @property
    def task_name(self):
        return "harmonic_sequence"

    def generate_data(self, sequence_length=5):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            # a >= 1, d >= 1 to avoid zero denominators
            a = random.randint(1, 8)
            d = random.randint(1, 5)
            # generate sequence_length + 1 terms
            denominators = [a + i * d for i in range(sequence_length + 1)]
            shown_fracs = [f"1/{den}" for den in denominators[:sequence_length]]
            next_den = denominators[sequence_length]
            g = math.gcd(1, next_den)
            answer_fraction = f"1/{next_den // g}"
            answer_decimal = round(1.0 / next_den, 6)
            data.append({
                'shown_sequence': shown_fracs,
                'a': a,
                'd': d,
                'answer_fraction': answer_fraction,
                'answer_decimal': answer_decimal,
                'answer': answer_decimal,  # primary key for compatibility
                'next_denominator': next_den,
            })
        return data

    def create_prompt(self, data_point):
        seq_str = ', '.join(data_point['shown_sequence'])
        return (
            f"Find the next term in the following harmonic (reciprocal) sequence:\n\n"
            f"{seq_str}, ?\n\n"
            f"The denominators form an arithmetic progression. Find the next denominator and "
            f"give the reciprocal as a simplified fraction.\n\n"
            f"Provide your final answer as \\boxed{{1/n}} (e.g. \\boxed{{1/7}}) "
            f"or as a decimal \\boxed{{0.142857}}."
        )

    def evaluate_response(self, response, data_point):
        answer_decimal = data_point['answer_decimal']
        answer_fraction = data_point['answer_fraction']
        parsed = self._parse_answer(response)
        instruction_followed = parsed is not None
        accuracy = 0
        if parsed is not None:
            try:
                accuracy = 1 if abs(float(parsed) - float(answer_decimal)) < 1e-3 else 0
            except Exception:
                accuracy = 0
        return {
            'ground_truth': answer_fraction,
            'predicted_answer': parsed,
            'accuracy': accuracy,
            'instruction_followed': int(instruction_followed),
            'shown_sequence': data_point['shown_sequence'],
            'prompt': self.create_prompt(data_point),
            'model_response': response,
        }

    def _parse_answer(self, response):
        try:
            from ...parsers.harmonic_sequence_parsing import parse_harmonic_answer
            return parse_harmonic_answer(response)
        except Exception:
            pass
        return _fallback_harmonic_parse(response)

    def run_evaluation(self, list_sizes):
        all_metrics = []
        for seq_length in list_sizes:
            data = self.generate_data(seq_length)
            for fold in range(self.num_folds):
                metrics = self.run_fold(data, seq_length, fold)
                metrics['sequence_length'] = seq_length
                all_metrics.append(metrics)
        return all_metrics


def _fallback_harmonic_parse(response):
    """Fallback harmonic parser — tries boxed fraction/decimal."""
    m = re.search(r'\\boxed\{([^}]+)\}', response)
    if m:
        raw = m.group(1).strip()
        # fraction: 1/7
        frac = re.match(r'^(-?\d+)/(\d+)$', raw)
        if frac:
            num, den = int(frac.group(1)), int(frac.group(2))
            if den != 0:
                return num / den
        try:
            return float(raw)
        except ValueError:
            pass
    return None

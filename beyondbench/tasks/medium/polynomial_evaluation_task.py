"""
Polynomial Evaluation Task

Generate a random polynomial with integer coefficients (degree 2-4) and a random
integer x; ask for p(x).
"""

import random
import re
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class PolynomialEvaluationTask(BaseTask):
    """Evaluate a polynomial at a given integer."""

    @property
    def task_name(self):
        return "polynomial_evaluation"

    def _poly_to_str(self, coeffs: List[int]) -> str:
        """Format polynomial from highest to lowest degree."""
        degree = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if c == 0:
                continue
            if power == 0:
                terms.append(str(c))
            elif power == 1:
                if c == 1:
                    terms.append('x')
                elif c == -1:
                    terms.append('-x')
                else:
                    terms.append(f'{c}x')
            else:
                if c == 1:
                    terms.append(f'x^{power}')
                elif c == -1:
                    terms.append(f'-x^{power}')
                else:
                    terms.append(f'{c}x^{power}')
        if not terms:
            return '0'
        result = terms[0]
        for t in terms[1:]:
            if t.startswith('-'):
                result += f' - {t[1:]}'
            else:
                result += f' + {t}'
        return result

    def _evaluate(self, coeffs: List[int], x: int) -> int:
        result = 0
        degree = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            result += c * (x ** (degree - i))
        return result

    def generate_data(self, sequence_length=1):
        """sequence_length unused; kept for interface compatibility."""
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            degree = random.randint(2, 4)
            # Coefficients from -5 to 5, leading coeff nonzero
            coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4])] + \
                     [random.randint(-5, 5) for _ in range(degree)]
            x = random.randint(-6, 6)
            answer = self._evaluate(coeffs, x)
            poly_str = self._poly_to_str(coeffs)
            data.append({
                'coefficients': coeffs,
                'x': x,
                'polynomial_str': poly_str,
                'answer': answer,
            })
        return data

    def create_prompt(self, data_point):
        return (
            f"Evaluate the following polynomial at x = {data_point['x']}:\n\n"
            f"p(x) = {data_point['polynomial_str']}\n\n"
            f"Substitute x = {data_point['x']} and compute p({data_point['x']}).\n\n"
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
            'polynomial_str': data_point['polynomial_str'],
            'x': data_point['x'],
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

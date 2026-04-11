"""
Matrix Operations Task

Performs one of: 2x2 matrix multiplication, determinant, or inverse.
"""

import random
import re
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class MatrixOperationsTask(BaseTask):
    """2x2 matrix operation (multiply, determinant, inverse)."""

    @property
    def task_name(self):
        return "matrix_operations"

    def _det(self, m):
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    def _mat_mul(self, a, b):
        return [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]],
        ]

    def _mat_inv(self, m):
        d = self._det(m)
        if d == 0:
            return None
        return [
            [m[1][1]/d, -m[0][1]/d],
            [-m[1][0]/d, m[0][0]/d],
        ]

    def generate_data(self, sequence_length=1):
        if self.seed is not None:
            random.seed(self.seed)

        ops = ['multiply', 'determinant', 'inverse']
        data = []
        for _ in range(self.num_samples):
            op = random.choice(ops)
            while True:
                a = [[random.randint(-4, 4) for _ in range(2)] for _ in range(2)]
                b = [[random.randint(-4, 4) for _ in range(2)] for _ in range(2)]
                if op == 'inverse':
                    if self._det(a) == 0:
                        continue
                break
            if op == 'multiply':
                answer = self._mat_mul(a, b)
            elif op == 'determinant':
                answer = self._det(a)
            else:
                answer = self._mat_inv(a)
            data.append({
                'operation': op,
                'matrix_a': a,
                'matrix_b': b if op == 'multiply' else None,
                'answer': answer,
            })
        return data

    def _mat_str(self, m):
        rows = ['  [' + ', '.join(str(x) for x in row) + ']' for row in m]
        return '[\n' + ',\n'.join(rows) + '\n]'

    def create_prompt(self, data_point):
        op = data_point['operation']
        a = data_point['matrix_a']
        a_str = self._mat_str(a)

        if op == 'multiply':
            b = data_point['matrix_b']
            b_str = self._mat_str(b)
            return (
                f"Compute the product of the following two 2x2 matrices:\n\n"
                f"A = {a_str}\n\nB = {b_str}\n\n"
                f"Give the result as a 2x2 matrix in the format \\boxed{{[[a,b],[c,d]]}}."
            )
        elif op == 'determinant':
            return (
                f"Compute the determinant of the following 2x2 matrix:\n\n"
                f"A = {a_str}\n\n"
                f"The determinant of [[a,b],[c,d]] is ad - bc.\n\n"
                f"Provide your final answer as \\boxed{{value}}."
            )
        else:  # inverse
            return (
                f"Compute the inverse of the following 2x2 matrix:\n\n"
                f"A = {a_str}\n\n"
                f"For a 2x2 matrix [[a,b],[c,d]], the inverse is (1/det)*[[d,-b],[-c,a]].\n\n"
                f"Give the result as \\boxed{{[[a,b],[c,d]]}} with decimal or fractional entries."
            )

    def evaluate_response(self, response, data_point):
        op = data_point['operation']
        ground_truth = data_point['answer']

        try:
            from ...parsers.matrix_operations_parsing import parse_matrix_answer
            parsed = parse_matrix_answer(response, op)
        except Exception:
            parsed = None

        instruction_followed = parsed is not None
        accuracy = 0

        if parsed is not None:
            try:
                if op == 'determinant':
                    accuracy = 1 if abs(float(parsed) - float(ground_truth)) < 0.5 else 0
                else:  # matrix answer
                    if isinstance(parsed, list) and len(parsed) == 2:
                        tol = 1e-3
                        close = all(
                            abs(float(parsed[i][j]) - float(ground_truth[i][j])) < tol
                            for i in range(2) for j in range(2)
                        )
                        accuracy = 1 if close else 0
            except Exception:
                accuracy = 0

        return {
            'ground_truth': ground_truth,
            'predicted_answer': parsed,
            'accuracy': accuracy,
            'instruction_followed': int(instruction_followed),
            'operation': op,
            'prompt': self.create_prompt(data_point),
            'model_response': response,
        }

    def run_evaluation(self, list_sizes):
        all_metrics = []
        for size in list_sizes:
            data = self.generate_data(size)
            for fold in range(self.num_folds):
                metrics = self.run_fold(data, size, fold)
                all_metrics.append(metrics)
        return all_metrics

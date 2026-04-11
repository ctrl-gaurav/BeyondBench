"""
Number Base Conversion Task

Convert between bases: decimal↔binary, decimal↔hex.
"""

import random
import re
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class NumberBaseConversionTask(BaseTask):
    """Convert numbers between different bases."""

    @property
    def task_name(self):
        return "number_base_conversion"

    _OPS = ['dec_to_bin', 'bin_to_dec', 'dec_to_hex', 'hex_to_dec']

    def generate_data(self, sequence_length=1):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            op = random.choice(self._OPS)
            if op == 'dec_to_bin':
                n = random.randint(1, 255)
                data.append({
                    'operation': op, 'input_value': str(n),
                    'from_base': 10, 'to_base': 2,
                    'answer': bin(n)[2:],
                })
            elif op == 'bin_to_dec':
                n = random.randint(1, 255)
                data.append({
                    'operation': op, 'input_value': bin(n)[2:],
                    'from_base': 2, 'to_base': 10,
                    'answer': str(n),
                })
            elif op == 'dec_to_hex':
                n = random.randint(1, 255)
                data.append({
                    'operation': op, 'input_value': str(n),
                    'from_base': 10, 'to_base': 16,
                    'answer': hex(n)[2:].upper(),
                })
            else:  # hex_to_dec
                n = random.randint(1, 255)
                hex_str = hex(n)[2:].upper()
                data.append({
                    'operation': op, 'input_value': hex_str,
                    'from_base': 16, 'to_base': 10,
                    'answer': str(n),
                })
        return data

    def create_prompt(self, data_point):
        op = data_point['operation']
        val = data_point['input_value']
        if op == 'dec_to_bin':
            return (
                f"Convert the decimal number {val} to binary.\n\n"
                f"Provide your final answer as \\boxed{{binary_number}} "
                f"(e.g. \\boxed{{11010110}})."
            )
        elif op == 'bin_to_dec':
            return (
                f"Convert the binary number {val} to decimal.\n\n"
                f"Provide your final answer as \\boxed{{decimal_number}}."
            )
        elif op == 'dec_to_hex':
            return (
                f"Convert the decimal number {val} to hexadecimal.\n\n"
                f"Use uppercase letters (A–F). "
                f"Provide your final answer as \\boxed{{hex_number}} "
                f"(e.g. \\boxed{{1A3F}})."
            )
        else:
            return (
                f"Convert the hexadecimal number {val} to decimal.\n\n"
                f"Provide your final answer as \\boxed{{decimal_number}}."
            )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point['answer']
        try:
            from ...parsers.number_base_conversion_parsing import parse_base_conversion_answer
            parsed = parse_base_conversion_answer(response, data_point['to_base'])
        except Exception:
            parsed = None

        instruction_followed = parsed is not None
        accuracy = 0
        if parsed is not None:
            accuracy = 1 if str(parsed).upper() == str(ground_truth).upper() else 0
        return {
            'ground_truth': ground_truth,
            'predicted_answer': parsed,
            'accuracy': accuracy,
            'instruction_followed': int(instruction_followed),
            'operation': data_point['operation'],
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

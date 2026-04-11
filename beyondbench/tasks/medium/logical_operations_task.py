"""
Logical Operations Task

Evaluate random boolean expressions with AND/OR/NOT/XOR over 3-5 variables.
"""

import random
import re
from typing import List, Dict, Any, Optional

from ...core.base_task import BaseTask


class LogicalOperationsTask(BaseTask):
    """Evaluate a boolean expression given variable assignments."""

    @property
    def task_name(self):
        return "logical_operations"

    _VAR_NAMES = ['A', 'B', 'C', 'D', 'E']

    def _make_expr(self, vars_used, depth=0):
        """Recursively generate a boolean expression."""
        if depth >= 2 or (depth > 0 and random.random() < 0.5):
            return random.choice(vars_used)
        op = random.choice(['AND', 'OR', 'XOR'])
        left = self._make_expr(vars_used, depth + 1)
        right = self._make_expr(vars_used, depth + 1)
        expr = f'({left} {op} {right})'
        if random.random() < 0.3:
            return f'(NOT {expr})'
        return expr

    def _eval_expr(self, expr, vals):
        """Safely evaluate expression with given variable values."""
        py_expr = expr
        py_expr = py_expr.replace('AND', 'and').replace('OR', 'or')
        py_expr = py_expr.replace('NOT', 'not').replace('XOR', '^')
        # Replace ^ with Python XOR for booleans
        namespace = {k: v for k, v in vals.items()}
        try:
            return bool(eval(py_expr, {"__builtins__": {}}, namespace))
        except Exception:
            return None

    def generate_data(self, sequence_length=1):
        if self.seed is not None:
            random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            n_vars = random.randint(3, 5)
            var_names = self._VAR_NAMES[:n_vars]
            values = {v: random.choice([True, False]) for v in var_names}
            # Generate a valid expression
            for _ in range(20):
                expr = self._make_expr(var_names)
                result = self._eval_expr(expr, values)
                if result is not None:
                    break
            else:
                expr = 'A'
                result = values['A']
            data.append({
                'expression': expr,
                'variables': values,
                'answer': result,
            })
        return data

    def create_prompt(self, data_point):
        var_strs = ', '.join(
            f"{k} = {'True' if v else 'False'}"
            for k, v in data_point['variables'].items()
        )
        return (
            f"Evaluate the following boolean expression:\n\n"
            f"Expression: {data_point['expression']}\n\n"
            f"Variable values: {var_strs}\n\n"
            f"Rules: AND = logical and, OR = logical or, "
            f"NOT = logical negation, XOR = exclusive or.\n\n"
            f"Provide your final answer as \\boxed{{True}} or \\boxed{{False}}."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point['answer']
        try:
            from ...parsers.logical_operations_parsing import parse_logical_answer
            parsed = parse_logical_answer(response)
        except Exception:
            parsed = None

        instruction_followed = parsed is not None
        accuracy = 0
        if parsed is not None:
            accuracy = 1 if (bool(parsed) == bool(ground_truth)) else 0
        return {
            'ground_truth': ground_truth,
            'predicted_answer': parsed,
            'accuracy': accuracy,
            'instruction_followed': int(instruction_followed),
            'expression': data_point['expression'],
            'variables': data_point['variables'],
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

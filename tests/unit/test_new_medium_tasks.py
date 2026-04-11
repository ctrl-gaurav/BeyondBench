"""
Unit tests for the 10 new Phase 13 medium tasks.
Tests: instantiation, data generation, prompt creation, evaluation, parser robustness.
"""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_handler(tmp_path):
    from unittest.mock import MagicMock
    handler = MagicMock()
    handler.backend = "mock"
    handler.api_provider = None
    handler.model_id = "test-model"
    handler.tokenizer = None
    handler.generate = MagicMock(return_value=[r"\boxed{42}"])
    handler.generate_batch = handler.generate
    handler.get_model_info = MagicMock(return_value={"model_name": "test", "backend": "mock", "api_provider": None, "model_type": "mock"})
    handler.get_statistics = MagicMock(return_value={"total_time_seconds": 1.0, "api_calls": 1, "total_input_tokens": 10, "total_output_tokens": 5})
    handler.count_tokens = MagicMock(return_value=10)
    return handler


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return str(d)


def make_task(cls, handler, out_dir, **kwargs):
    defaults = dict(
        model_handler=handler,
        output_dir=out_dir,
        min_val=1, max_val=20,
        num_folds=1, num_samples=5,
        store_details=False,
        temperature=0.1, top_p=0.9, max_tokens=512,
        seed=42,
    )
    defaults.update(kwargs)
    return cls(**defaults)


def _boxed(val):
    return rf"\boxed{{{val}}}"


# ===========================================================================
# 1. arithmetic_progression
# ===========================================================================

class TestArithmeticProgression:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.arithmetic_progression_task import ArithmeticProgressionTask
        return make_task(ArithmeticProgressionTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "arithmetic_progression"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(sequence_length=6)
        assert len(data) == 5
        for dp in data:
            assert 'shown_sequence' in dp
            assert 'answer' in dp
            assert len(dp['shown_sequence']) == 6
            # Verify AP: differences are constant
            diffs = [dp['shown_sequence'][i+1] - dp['shown_sequence'][i]
                     for i in range(len(dp['shown_sequence'])-1)]
            assert len(set(diffs)) == 1, "Not an arithmetic progression"
            # Verify next term
            assert dp['answer'] == dp['shown_sequence'][-1] + diffs[0]

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        prompt = t.create_prompt(data[0])
        assert r'\boxed' in prompt

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'shown_sequence': [2, 5, 8, 11, 14], 'first_term': 2, 'common_diff': 3,
              'answer': 17, 'next_term': 17}
        result = t.evaluate_response(_boxed(17), dp)
        assert result['accuracy'] == 1
        assert result['instruction_followed'] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'shown_sequence': [2, 5, 8, 11, 14], 'first_term': 2, 'common_diff': 3,
              'answer': 17, 'next_term': 17}
        result = t.evaluate_response(_boxed(99), dp)
        assert result['accuracy'] == 0

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.tasks.medium.arithmetic_progression_task import ArithmeticProgressionTask
        t = make_task(ArithmeticProgressionTask, mock_handler, out_dir)
        dp = {'shown_sequence': [1, 4, 7, 10], 'first_term': 1, 'common_diff': 3,
              'answer': 13, 'next_term': 13}
        formats = [
            r"\boxed{13}",
            r"The answer is \boxed{13}.",
            r"The next term is 13.",
            "Answer: 13",
            r"Therefore the answer is \boxed{13}",
            r"$\boxed{13}$",
            r"$$\boxed{13}$$",
            "So the next term is **13**",
            r"\boxed{13.0}",
            "The progression continues to 13",
        ]
        for fmt in formats:
            result = t.evaluate_response(fmt, dp)
            assert result['accuracy'] == 1, f"Failed for: {fmt!r}"


# ===========================================================================
# 2. harmonic_sequence
# ===========================================================================

class TestHarmonicSequence:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.harmonic_sequence_task import HarmonicSequenceTask
        return make_task(HarmonicSequenceTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "harmonic_sequence"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(sequence_length=5)
        assert len(data) == 5
        for dp in data:
            assert 'shown_sequence' in dp
            assert 'answer_fraction' in dp
            assert 'answer_decimal' in dp
            assert len(dp['shown_sequence']) == 5
            # Verify fraction format
            assert dp['answer_fraction'].startswith('1/')

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        prompt = t.create_prompt(data[0])
        assert r'\boxed' in prompt

    def test_evaluate_fraction_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'shown_sequence': ['1/1','1/2','1/3','1/4','1/5'],
              'a': 1, 'd': 1, 'answer_fraction': '1/6', 'answer_decimal': 1/6,
              'answer': 1/6, 'next_denominator': 6}
        result = t.evaluate_response(r'\boxed{1/6}', dp)
        assert result['accuracy'] == 1

    def test_evaluate_decimal_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'shown_sequence': ['1/1','1/2','1/3','1/4','1/5'],
              'a': 1, 'd': 1, 'answer_fraction': '1/6', 'answer_decimal': 1/6,
              'answer': 1/6, 'next_denominator': 6}
        result = t.evaluate_response(r'\boxed{0.166667}', dp)
        assert result['accuracy'] == 1

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.parsers.harmonic_sequence_parsing import parse_harmonic_answer
        # 1/7 ≈ 0.142857
        expected = 1/7
        cases = [
            (r'\boxed{1/7}', expected),
            (r'\boxed{0.142857}', expected),
            ('The next term is 1/7', expected),
            ('The answer is 0.142857', expected),
            (r'Therefore \boxed{1/7}', expected),
            (r'$\boxed{0.142857}$', expected),
            ('next term = 1/7', expected),
            ('The reciprocal is 0.142857', expected),
            (r'$$\boxed{1/7}$$', expected),
            ('So we get 1/7', expected),
        ]
        for response, exp in cases:
            result = parse_harmonic_answer(response)
            assert result is not None, f"Failed to parse: {response!r}"
            assert abs(result - exp) < 1e-3, f"Wrong value for {response!r}: {result}"


# ===========================================================================
# 3. collatz_sequence
# ===========================================================================

class TestCollatzSequence:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.collatz_sequence_task import CollatzSequenceTask
        return make_task(CollatzSequenceTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "collatz_sequence"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(sequence_length=6)
        assert len(data) == 5
        for dp in data:
            assert 'shown_sequence' in dp
            assert 'answer' in dp
            # Verify Collatz step
            last = dp['shown_sequence'][-1]
            expected_next = last // 2 if last % 2 == 0 else 3 * last + 1
            assert dp['answer'] == expected_next

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert r'\boxed' in t.create_prompt(data[0])

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        # 6 → 3 → 10 → 5 → 16 → 8 → 4
        dp = {'shown_sequence': [6, 3, 10, 5, 16, 8], 'start': 6,
              'answer': 4, 'next_term': 4}
        result = t.evaluate_response(_boxed(4), dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'shown_sequence': [6, 3, 10, 5, 16, 8], 'start': 6,
              'answer': 4, 'next_term': 4}
        result = t.evaluate_response(_boxed(5), dp)
        assert result['accuracy'] == 0

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.tasks.medium.collatz_sequence_task import CollatzSequenceTask
        t = make_task(CollatzSequenceTask, mock_handler, out_dir)
        dp = {'shown_sequence': [6, 3, 10, 5, 16, 8], 'start': 6,
              'answer': 4, 'next_term': 4}
        formats = [
            r"\boxed{4}", r"The answer is \boxed{4}.",
            "Next term is 4", "Answer: 4",
            r"Therefore \boxed{4}", r"$\boxed{4}$",
            r"$$\boxed{4}$$", "The next term is **4**",
            r"\boxed{4.0}", "So 8 is even, 8/2 = 4",
        ]
        for fmt in formats:
            result = t.evaluate_response(fmt, dp)
            assert result['accuracy'] == 1, f"Failed: {fmt!r}"


# ===========================================================================
# 4. polynomial_evaluation
# ===========================================================================

class TestPolynomialEvaluation:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.polynomial_evaluation_task import PolynomialEvaluationTask
        return make_task(PolynomialEvaluationTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "polynomial_evaluation"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert len(data) == 5
        for dp in data:
            assert 'coefficients' in dp and 'x' in dp and 'answer' in dp
            # Verify the polynomial evaluation
            coeffs = dp['coefficients']
            x = dp['x']
            deg = len(coeffs) - 1
            expected = sum(c * (x ** (deg - i)) for i, c in enumerate(coeffs))
            assert dp['answer'] == expected

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert r'\boxed' in t.create_prompt(data[0])

    def test_evaluate_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        # p(x) = x^2 + 3x - 5 at x=2 → 4 + 6 - 5 = 5
        dp = {'coefficients': [1, 3, -5], 'x': 2, 'polynomial_str': 'x^2 + 3x - 5', 'answer': 5}
        result = t.evaluate_response(_boxed(5), dp)
        assert result['accuracy'] == 1

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.tasks.medium.polynomial_evaluation_task import PolynomialEvaluationTask
        t = make_task(PolynomialEvaluationTask, mock_handler, out_dir)
        dp = {'coefficients': [1, 3, -5], 'x': 2, 'polynomial_str': 'x^2+3x-5', 'answer': 5}
        formats = [
            r"\boxed{5}", r"The answer is \boxed{5}.", "Answer: 5",
            r"Therefore \boxed{5}", r"$\boxed{5}$", r"$$\boxed{5}$$",
            "p(2) = 5", "The result is 5", "5.0", r"\boxed{5.0}",
        ]
        for fmt in formats:
            result = t.evaluate_response(fmt, dp)
            assert result['accuracy'] == 1, f"Failed: {fmt!r}"


# ===========================================================================
# 5. matrix_operations
# ===========================================================================

class TestMatrixOperations:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.matrix_operations_task import MatrixOperationsTask
        return make_task(MatrixOperationsTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "matrix_operations"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert len(data) == 5
        for dp in data:
            assert 'operation' in dp
            assert 'matrix_a' in dp
            assert 'answer' in dp
            assert dp['operation'] in ('multiply', 'determinant', 'inverse')

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        for dp in data:
            assert r'\boxed' in t.create_prompt(dp)

    def test_evaluate_determinant_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        # det([[1,2],[3,4]]) = 4-6 = -2
        dp = {'operation': 'determinant', 'matrix_a': [[1,2],[3,4]], 'matrix_b': None, 'answer': -2}
        result = t.evaluate_response(r'\boxed{-2}', dp)
        assert result['accuracy'] == 1

    def test_evaluate_multiply_correct(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        # [[1,0],[0,1]] * [[2,3],[4,5]] = [[2,3],[4,5]]
        dp = {'operation': 'multiply', 'matrix_a': [[1,0],[0,1]],
              'matrix_b': [[2,3],[4,5]], 'answer': [[2,3],[4,5]]}
        result = t.evaluate_response(r'\boxed{[[2,3],[4,5]]}', dp)
        assert result['accuracy'] == 1

    def test_matrix_parser_formats(self):
        from beyondbench.parsers.matrix_operations_parsing import parse_matrix_answer
        # determinant scalar
        cases = [
            (r'\boxed{-2}', 'determinant', -2.0),
            ('The determinant is -2', 'determinant', -2.0),
            ('det = -2.0', 'determinant', -2.0),
        ]
        for response, op, expected in cases:
            result = parse_matrix_answer(response, op)
            assert result is not None, f"Failed: {response!r}"
            assert abs(float(result) - expected) < 0.5, f"Wrong: {result} != {expected}"

        # matrix answers
        mat_cases = [
            (r'\boxed{[[2,3],[4,5]]}', 'multiply', [[2,3],[4,5]]),
            ('[[2,3],[4,5]]', 'multiply', [[2,3],[4,5]]),
            ('  [2, 3]\n  [4, 5]', 'multiply', [[2,3],[4,5]]),
        ]
        for response, op, expected in mat_cases:
            result = parse_matrix_answer(response, op)
            assert result is not None, f"Failed: {response!r}"
            for i in range(2):
                for j in range(2):
                    assert abs(float(result[i][j]) - expected[i][j]) < 1e-3


# ===========================================================================
# 6. number_base_conversion
# ===========================================================================

class TestNumberBaseConversion:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.number_base_conversion_task import NumberBaseConversionTask
        return make_task(NumberBaseConversionTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "number_base_conversion"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert len(data) == 5
        for dp in data:
            assert 'operation' in dp and 'input_value' in dp and 'answer' in dp
            assert dp['operation'] in ('dec_to_bin', 'bin_to_dec', 'dec_to_hex', 'hex_to_dec')

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        for dp in data:
            assert r'\boxed' in t.create_prompt(dp)

    def test_evaluate_dec_to_bin(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'operation': 'dec_to_bin', 'input_value': '10', 'from_base': 10, 'to_base': 2, 'answer': '1010'}
        result = t.evaluate_response(r'\boxed{1010}', dp)
        assert result['accuracy'] == 1

    def test_evaluate_dec_to_hex(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'operation': 'dec_to_hex', 'input_value': '255', 'from_base': 10, 'to_base': 16, 'answer': 'FF'}
        result = t.evaluate_response(r'\boxed{FF}', dp)
        assert result['accuracy'] == 1

    def test_parser_formats(self):
        from beyondbench.parsers.number_base_conversion_parsing import parse_base_conversion_answer
        # binary
        bin_cases = [
            r'\boxed{1010}', 'The binary is 1010', r'binary: \boxed{1010}',
            'Answer: 1010', r'$\boxed{1010}$', 'result is 1010',
        ]
        for resp in bin_cases:
            result = parse_base_conversion_answer(resp, 2)
            assert result == '1010', f"Failed: {resp!r} → {result}"

        # hex
        hex_cases = [
            r'\boxed{FF}', r'\boxed{ff}', 'The hex is FF',
            'Answer: FF', r'$\boxed{FF}$', '0xFF',
        ]
        for resp in hex_cases:
            result = parse_base_conversion_answer(resp, 16)
            assert result == 'FF', f"Failed: {resp!r} → {result}"

        # decimal
        dec_cases = [
            r'\boxed{255}', 'The decimal is 255', 'Answer: 255',
            r'$\boxed{255}$', 'result = 255',
        ]
        for resp in dec_cases:
            result = parse_base_conversion_answer(resp, 10)
            assert result == '255', f"Failed: {resp!r} → {result}"


# ===========================================================================
# 7. logical_operations
# ===========================================================================

class TestLogicalOperations:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.logical_operations_task import LogicalOperationsTask
        return make_task(LogicalOperationsTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "logical_operations"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert len(data) == 5
        for dp in data:
            assert 'expression' in dp and 'variables' in dp and 'answer' in dp
            assert isinstance(dp['answer'], bool)

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        for dp in data:
            assert r'\boxed' in t.create_prompt(dp)

    def test_evaluate_true(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'expression': '(A AND B)', 'variables': {'A': True, 'B': True}, 'answer': True}
        result = t.evaluate_response(r'\boxed{True}', dp)
        assert result['accuracy'] == 1

    def test_evaluate_false(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'expression': '(A AND B)', 'variables': {'A': True, 'B': False}, 'answer': False}
        result = t.evaluate_response(r'\boxed{False}', dp)
        assert result['accuracy'] == 1

    def test_parser_formats(self):
        from beyondbench.parsers.logical_operations_parsing import parse_logical_answer
        true_cases = [
            r'\boxed{True}', r'\boxed{true}', r'\boxed{1}',
            r'\boxed{yes}', 'The answer is True', 'result is true',
            'True', r'$\boxed{True}$', r'$$\boxed{True}$$', 'yes',
        ]
        for resp in true_cases:
            result = parse_logical_answer(resp)
            assert result is True, f"Expected True for: {resp!r}, got {result}"

        false_cases = [
            r'\boxed{False}', r'\boxed{false}', r'\boxed{0}',
            r'\boxed{no}', 'The answer is False', 'result is false',
            'False', r'$\boxed{False}$', r'$$\boxed{False}$$', 'no',
        ]
        for resp in false_cases:
            result = parse_logical_answer(resp)
            assert result is False, f"Expected False for: {resp!r}, got {result}"


# ===========================================================================
# 8. pattern_completion
# ===========================================================================

class TestPatternCompletion:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.pattern_completion_task import PatternCompletionTask
        return make_task(PatternCompletionTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "pattern_completion"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data(sequence_length=5)
        assert len(data) == 5
        for dp in data:
            assert 'pattern_type' in dp and 'shown_sequence' in dp and 'answer' in dp
            assert dp['pattern_type'] in ('squares', 'cubes', 'triangular', 'powers_of_2', 'factorial', 'pascal_diagonal')

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        for dp in data:
            assert r'\boxed' in t.create_prompt(dp)

    def test_evaluate_squares(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'pattern_type': 'squares', 'start_offset': 1, 'shown_sequence': [1,4,9,16,25], 'answer': 36, 'next_term': 36}
        result = t.evaluate_response(_boxed(36), dp)
        assert result['accuracy'] == 1

    def test_evaluate_powers_of_2(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'pattern_type': 'powers_of_2', 'start_offset': 0, 'shown_sequence': [1,2,4,8,16], 'answer': 32, 'next_term': 32}
        result = t.evaluate_response(_boxed(32), dp)
        assert result['accuracy'] == 1

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.tasks.medium.pattern_completion_task import PatternCompletionTask
        t = make_task(PatternCompletionTask, mock_handler, out_dir)
        dp = {'pattern_type': 'squares', 'start_offset': 1, 'shown_sequence': [1,4,9,16,25], 'answer': 36, 'next_term': 36}
        formats = [
            r"\boxed{36}", r"The answer is \boxed{36}.", "Next: 36",
            r"Therefore \boxed{36}", r"$\boxed{36}$", r"$$\boxed{36}$$",
            "The pattern gives 36", r"\boxed{36.0}", "6^2 = 36",
            "Answer: 36",
        ]
        for fmt in formats:
            result = t.evaluate_response(fmt, dp)
            assert result['accuracy'] == 1, f"Failed: {fmt!r}"


# ===========================================================================
# 9. gcd_lcm
# ===========================================================================

class TestGcdLcm:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.gcd_lcm_task import GcdLcmTask
        return make_task(GcdLcmTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "gcd_lcm"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert len(data) == 5
        for dp in data:
            assert 'operation' in dp and 'numbers' in dp and 'answer' in dp
            assert dp['operation'] in ('gcd', 'lcm')
            # Verify answer
            nums = dp['numbers']
            if dp['operation'] == 'gcd':
                expected = nums[0]
                for n in nums[1:]:
                    expected = math.gcd(expected, n)
            else:
                expected = nums[0]
                for n in nums[1:]:
                    expected = abs(expected * n) // math.gcd(expected, n)
            assert dp['answer'] == expected

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        for dp in data:
            assert r'\boxed' in t.create_prompt(dp)

    def test_evaluate_gcd(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'operation': 'gcd', 'numbers': [24, 36, 60], 'answer': 12}
        result = t.evaluate_response(_boxed(12), dp)
        assert result['accuracy'] == 1

    def test_evaluate_lcm(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'operation': 'lcm', 'numbers': [4, 6], 'answer': 12}
        result = t.evaluate_response(_boxed(12), dp)
        assert result['accuracy'] == 1

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.tasks.medium.gcd_lcm_task import GcdLcmTask
        t = make_task(GcdLcmTask, mock_handler, out_dir)
        dp = {'operation': 'gcd', 'numbers': [24, 36, 60], 'answer': 12}
        formats = [
            r"\boxed{12}", r"The GCD is \boxed{12}.", "Answer: 12",
            r"Therefore \boxed{12}", r"$\boxed{12}$", r"$$\boxed{12}$$",
            "GCD = 12", "The result is 12", r"\boxed{12.0}",
            "So GCD(24,36,60) = 12",
        ]
        for fmt in formats:
            result = t.evaluate_response(fmt, dp)
            assert result['accuracy'] == 1, f"Failed: {fmt!r}"


# ===========================================================================
# 10. combinatorics
# ===========================================================================

class TestCombinatorics:
    def _task(self, handler, out_dir):
        from beyondbench.tasks.medium.combinatorics_task import CombinatoricsTask
        return make_task(CombinatoricsTask, handler, out_dir)

    def test_instantiable(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        assert t.task_name == "combinatorics"

    def test_generate_data(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        assert len(data) == 5
        for dp in data:
            assert 'operation' in dp and 'n' in dp and 'r' in dp and 'answer' in dp
            assert dp['operation'] in ('permutation', 'combination')
            n, r = dp['n'], dp['r']
            assert 1 <= r <= n <= 12
            if dp['operation'] == 'permutation':
                assert dp['answer'] == math.perm(n, r)
            else:
                assert dp['answer'] == math.comb(n, r)

    def test_create_prompt_has_boxed(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        data = t.generate_data()
        for dp in data:
            assert r'\boxed' in t.create_prompt(dp)

    def test_evaluate_combination(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'operation': 'combination', 'n': 8, 'r': 3, 'answer': 56}
        result = t.evaluate_response(_boxed(56), dp)
        assert result['accuracy'] == 1

    def test_evaluate_permutation(self, mock_handler, out_dir):
        t = self._task(mock_handler, out_dir)
        dp = {'operation': 'permutation', 'n': 5, 'r': 2, 'answer': 20}
        result = t.evaluate_response(_boxed(20), dp)
        assert result['accuracy'] == 1

    def test_parser_formats(self, mock_handler, out_dir):
        from beyondbench.tasks.medium.combinatorics_task import CombinatoricsTask
        t = make_task(CombinatoricsTask, mock_handler, out_dir)
        dp = {'operation': 'combination', 'n': 8, 'r': 3, 'answer': 56}
        formats = [
            r"\boxed{56}", r"The answer is \boxed{56}.", "Answer: 56",
            r"Therefore \boxed{56}", r"$\boxed{56}$", r"$$\boxed{56}$$",
            "C(8,3) = 56", "The result is 56", r"\boxed{56.0}",
            "So there are 56 ways",
        ]
        for fmt in formats:
            result = t.evaluate_response(fmt, dp)
            assert result['accuracy'] == 1, f"Failed: {fmt!r}"


# ===========================================================================
# Registry check
# ===========================================================================

def test_medium_task_count():
    from beyondbench.core.task_registry import TaskRegistry
    registry = TaskRegistry()
    medium = registry.get_tasks_for_suite('medium')
    assert len(medium) == 15, f"Expected 15 medium tasks, got {len(medium)}: {medium}"


def test_all_new_tasks_importable():
    new_tasks = [
        ('arithmetic_progression', 'ArithmeticProgressionTask'),
        ('harmonic_sequence', 'HarmonicSequenceTask'),
        ('collatz_sequence', 'CollatzSequenceTask'),
        ('polynomial_evaluation', 'PolynomialEvaluationTask'),
        ('matrix_operations', 'MatrixOperationsTask'),
        ('number_base_conversion', 'NumberBaseConversionTask'),
        ('logical_operations', 'LogicalOperationsTask'),
        ('pattern_completion', 'PatternCompletionTask'),
        ('gcd_lcm', 'GcdLcmTask'),
        ('combinatorics', 'CombinatoricsTask'),
    ]
    import importlib
    for task_name, class_name in new_tasks:
        mod = importlib.import_module(f'beyondbench.tasks.medium.{task_name}_task')
        cls = getattr(mod, class_name)
        assert cls is not None


def test_parser_configs_have_new_tasks():
    from beyondbench.parsers.task_configs import TASK_PARSER_CONFIGS
    new_tasks = [
        'arithmetic_progression', 'harmonic_sequence', 'collatz_sequence',
        'polynomial_evaluation', 'matrix_operations', 'number_base_conversion',
        'logical_operations', 'pattern_completion', 'gcd_lcm', 'combinatorics',
    ]
    for task in new_tasks:
        assert task in TASK_PARSER_CONFIGS, f"Missing parser config for: {task}"

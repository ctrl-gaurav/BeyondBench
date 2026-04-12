"""
Regex Matching Task

Evaluates LLM's ability to determine whether a string fully matches a regex pattern.
Uses Python's re.fullmatch for ground truth.

Data point: {'pattern': p, 'test_string': s, 'answer': True/False}
Output: True or False (boxed as 'true' or 'false')
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ...core.base_task import BaseTask


class RegexMatchingSolver:
    """Ground truth using Python's re.fullmatch."""

    @staticmethod
    def matches(pattern: str, test_string: str) -> bool:
        try:
            return bool(re.fullmatch(pattern, test_string))
        except re.error:
            return False

    # Pre-defined patterns and string generation for valid, interesting cases
    PATTERN_TEMPLATES = [
        # (pattern_template, matching_examples, non_matching_examples)
        ('a+b', ['ab', 'aab', 'aaab'], ['b', 'a', 'ba', 'abc']),
        ('a*b', ['b', 'ab', 'aab'], ['a', 'ba', 'abc']),
        ('[abc]+', ['a', 'bc', 'abc', 'bca'], ['d', 'abcd', '']),
        ('\\d+', ['0', '42', '123'], ['', 'a', '1a', 'a1']),
        ('[a-z]{3}', ['abc', 'xyz', 'aaa'], ['ab', 'abcd', 'ABC', '']),
        ('ab?c', ['ac', 'abc'], ['abbc', 'bc', 'a']),
        ('(ab)+', ['ab', 'abab', 'ababab'], ['a', 'b', 'aba', '']),
        ('[0-9]{2}', ['00', '12', '99'], ['0', '123', 'ab']),
        ('a|b', ['a', 'b'], ['ab', 'c', '']),
        ('[a-c]*d', ['d', 'ad', 'abd', 'abcd'], ['e', 'da', 'abcde']),
        ('\\w+', ['hello', 'a1b2', '_test'], ['', 'hello world', 'a-b']),
        ('[aeiou]+', ['a', 'aei', 'uoiea'], ['b', 'ae1', 'AEIOU']),
    ]


class RegexMatchingTask(BaseTask):
    """Regex full-match determination task."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=5, **kwargs):
        self._task_name = "regex_matching"
        self.problem_size = problem_size
        super().__init__(
            model_handler=model_handler, output_dir=output_dir,
            min_val=min_val, max_val=max_val, num_folds=num_folds,
            num_samples=num_samples, store_details=store_details,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            seed=seed, **kwargs
        )

    @property
    def task_name(self):
        return self._task_name

    def generate_data(self, **kwargs):
        test_cases = []
        templates = RegexMatchingSolver.PATTERN_TEMPLATES

        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 41)
            tmpl = templates[i % len(templates)]
            pattern, matching, non_matching = tmpl

            # Alternate between match and non-match
            if rng.random() < 0.5 and matching:
                test_str = rng.choice(matching)
                expected = True
            elif non_matching:
                test_str = rng.choice(non_matching)
                expected = False
            else:
                test_str = rng.choice(matching)
                expected = True

            # Verify with Python re
            actual = RegexMatchingSolver.matches(pattern, test_str)
            answer = actual  # Use Python as ground truth

            test_cases.append({
                'pattern': pattern,
                'test_string': test_str,
                'answer': answer,
                'ground_truth': answer,
            })
        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        pattern = data_point['pattern']
        test_str = data_point['test_string']

        prompt = f"""Determine if the following string FULLY matches the given regular expression pattern.

Pattern: {pattern}
String: "{test_str}"

Rules:
- The ENTIRE string must match (as if using fullmatch, not search).
- Common regex elements:
  - `.` matches any character
  - `*` means zero or more of previous
  - `+` means one or more of previous
  - `?` means zero or one of previous
  - `[abc]` matches a, b, or c
  - `[a-z]` matches any lowercase letter
  - `\\d` matches a digit (0-9)
  - `\\w` matches word character (letter, digit, underscore)
  - `|` means OR
  - `(ab)+` means one or more occurrences of "ab"
  - `{{3}}` means exactly 3 of previous

Analyze the pattern carefully and check the string character by character.

Answer True if the full string matches, False if it does not.
Express your final answer as \\boxed{{true}} or \\boxed{{false}}."""
        return prompt

    def evaluate_response(self, response: str, data_point: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = data_point['answer']
        predicted = self._parse_answer(response)

        accuracy = 1 if (predicted is not None and bool(predicted) == bool(ground_truth)) else 0
        instruction_followed = 1 if predicted is not None else 0

        return {
            'accuracy': accuracy,
            'instruction_followed': instruction_followed,
            'ground_truth': ground_truth,
            'predicted_answer': predicted,
        }

    def _parse_answer(self, response: str) -> Optional[bool]:
        import re as re_module
        # Boxed true/false
        m = re_module.search(r'\\boxed\{(true|false)\}', response, re_module.IGNORECASE)
        if m:
            return m.group(1).lower() == 'true'
        # Yes/No / True/False in final line
        last_line = response.strip().split('\n')[-1].lower()
        if 'true' in last_line or 'yes' in last_line or 'matches' in last_line:
            return True
        if 'false' in last_line or 'no' in last_line or 'does not match' in last_line:
            return False
        # Search anywhere
        if re_module.search(r'\btrue\b', response, re_module.IGNORECASE):
            return True
        if re_module.search(r'\bfalse\b', response, re_module.IGNORECASE):
            return False
        return None

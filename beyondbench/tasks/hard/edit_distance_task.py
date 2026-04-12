"""
Edit Distance (Levenshtein) Task

Evaluates LLM's ability to compute the edit distance (minimum insertions, deletions,
substitutions) between two strings using DP.

Data point: {'string1': s1, 'string2': s2, 'answer': distance}
Output: int
"""

import random
from typing import Any, Dict, Optional

from ...core.base_task import BaseTask


class EditDistanceSolver:
    """DP-based Levenshtein distance solver."""

    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]


class EditDistanceTask(BaseTask):
    """Levenshtein edit distance between two strings."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=7, **kwargs):
        self._task_name = "edit_distance"
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

    def _random_string(self, length: int, rng: random.Random, alphabet: str = 'abcde') -> str:
        return ''.join(rng.choice(alphabet) for _ in range(length))

    def generate_data(self, **kwargs):
        list_size = kwargs.get('list_size', None)
        base_len = list_size if list_size else self.problem_size

        test_cases = []
        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 79)
            l1 = rng.randint(4, min(8, max(4, base_len)))
            l2 = rng.randint(4, min(8, max(4, base_len)))
            s1 = self._random_string(l1, rng)
            s2 = self._random_string(l2, rng)
            answer = EditDistanceSolver.edit_distance(s1, s2)
            test_cases.append({
                'string1': s1,
                'string2': s2,
                'answer': answer,
                'ground_truth': answer,
            })
        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        s1 = data_point['string1']
        s2 = data_point['string2']

        prompt = f"""Compute the Edit Distance (Levenshtein distance) between two strings.

String 1: "{s1}"  (length {len(s1)})
String 2: "{s2}"  (length {len(s2)})

The edit distance is the minimum number of single-character operations (insertions, deletions, substitutions) to transform String 1 into String 2.

Use dynamic programming:
- dp[i][j] = edit distance between s1[0..i-1] and s2[0..j-1]
- dp[i][0] = i (delete all chars from s1)
- dp[0][j] = j (insert all chars of s2)
- If s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
- Else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

Fill the full DP table, row by row.

Express your final answer as \\boxed{{edit_distance}}."""
        return prompt

    def evaluate_response(self, response: str, data_point: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = data_point['answer']
        predicted = self._parse_answer(response)

        accuracy = 1 if (predicted is not None and int(predicted) == int(ground_truth)) else 0
        instruction_followed = 1 if predicted is not None else 0

        return {
            'accuracy': accuracy,
            'instruction_followed': instruction_followed,
            'ground_truth': ground_truth,
            'predicted_answer': predicted,
        }

    def _parse_answer(self, response: str) -> Optional[int]:
        import re
        m = re.search(r'\\boxed\{(\d+)\}', response)
        if m:
            return int(m.group(1))
        m = re.search(r'(?:edit distance|levenshtein|answer)[^\d]*(\d+)', response, re.IGNORECASE)
        if m:
            return int(m.group(1))
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return int(nums[-1])
        return None

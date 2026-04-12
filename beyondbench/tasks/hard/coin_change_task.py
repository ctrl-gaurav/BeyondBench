"""
Coin Change Task

Evaluates LLM's ability to find the minimum number of coins to make a target amount.
Always includes coin denomination 1 to ensure feasibility.

Data point: {'coins': [1, 5, 10, ...], 'target': amount, 'answer': min_coins}
Output: int
"""

import random
from typing import Any, Dict, List, Optional

from ...core.base_task import BaseTask


class CoinChangeSolver:
    """DP-based minimum coin change solver."""

    @staticmethod
    def min_coins(coins: List[int], target: int) -> int:
        """Returns minimum number of coins to make target (inf if impossible)."""
        dp = [float('inf')] * (target + 1)
        dp[0] = 0
        for amount in range(1, target + 1):
            for coin in coins:
                if coin <= amount and dp[amount - coin] + 1 < dp[amount]:
                    dp[amount] = dp[amount - coin] + 1
        return int(dp[target]) if dp[target] < float('inf') else -1


class CoinChangeTask(BaseTask):
    """Minimum coin change task."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=5, **kwargs):
        self._task_name = "coin_change"
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
        list_size = kwargs.get('list_size', None)
        n_coins = list_size if list_size else self.problem_size

        test_cases = []
        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 71)
            # Always include 1 for feasibility
            extra = rng.randint(1, min(4, max(1, n_coins - 1)))
            coin_set = set([1])
            candidates = [2, 3, 5, 6, 7, 10, 11, 15, 20, 25]
            rng.shuffle(candidates)
            for c in candidates[:extra]:
                coin_set.add(c)
            coins = sorted(coin_set)
            target = rng.randint(10, 30)
            answer = CoinChangeSolver.min_coins(coins, target)
            test_cases.append({
                'coins': coins,
                'target': target,
                'answer': answer,
                'ground_truth': answer,
            })
        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        coins = data_point['coins']
        target = data_point['target']
        coins_str = ", ".join(str(c) for c in coins)

        prompt = f"""You are solving the Coin Change problem.

Available coin denominations: [{coins_str}]
Target amount: {target}

Find the MINIMUM number of coins needed to make exactly {target}.
You have unlimited coins of each denomination.

Use dynamic programming:
- dp[0] = 0 (zero coins needed for amount 0)
- For each amount a from 1 to {target}:
  - dp[a] = min(dp[a - coin] + 1) for each coin where coin <= a

Fill the DP table and find dp[{target}].

Show your work step by step, then state the minimum number of coins.

Express your final answer as \\boxed{{min_coins}}."""
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
        m = re.search(r'(?:minimum|min|answer)[^\d]*(\d+)\s*coin', response, re.IGNORECASE)
        if m:
            return int(m.group(1))
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return int(nums[-1])
        return None

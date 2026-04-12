"""
Traveling Salesman Task (small TSP)

Evaluates LLM's ability to find the minimum tour cost visiting all cities and returning
to start. Uses brute-force over permutations since n is small (4-6 cities).

Data point: {'distance_matrix': [[...]], 'num_cities': n, 'answer': min_cost}
Output: int (minimum tour cost)
"""

import random
from itertools import permutations
from typing import Any, Dict, List, Optional

from ...core.base_task import BaseTask


class TSPSolver:
    """Brute-force TSP solver for small instances."""

    @staticmethod
    def solve(dist: List[List[int]]) -> int:
        """Returns minimum tour cost (starting and ending at city 0)."""
        n = len(dist)
        cities = list(range(1, n))
        min_cost = float('inf')
        for perm in permutations(cities):
            tour = [0] + list(perm) + [0]
            cost = sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
            if cost < min_cost:
                min_cost = cost
        return int(min_cost)


class TravelingSalesmanTask(BaseTask):
    """Small Traveling Salesman Problem task."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=5, **kwargs):
        self._task_name = "traveling_salesman"
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

    def _make_distance_matrix(self, n: int, rng: random.Random) -> List[List[int]]:
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                w = rng.randint(5, 30)
                dist[i][j] = w
                dist[j][i] = w
        return dist

    def generate_data(self, **kwargs):
        list_size = kwargs.get('list_size', None)
        n_cities = list_size if list_size else self.problem_size

        test_cases = []
        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 13)
            n = rng.randint(4, min(6, max(4, n_cities)))
            dist = self._make_distance_matrix(n, rng)
            answer = TSPSolver.solve(dist)
            test_cases.append({
                'distance_matrix': dist,
                'num_cities': n,
                'answer': answer,
                'ground_truth': answer,
            })
        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        dist = data_point['distance_matrix']
        n = data_point['num_cities']
        city_names = [str(i) for i in range(n)]

        # Header row
        header = "     " + "  ".join(f"C{i:2d}" for i in range(n))
        rows = [header]
        for i in range(n):
            row = f"C{i:2d} | " + "  ".join(f"{dist[i][j]:4d}" for j in range(n))
            rows.append(row)
        matrix_text = "\n".join(rows)

        prompt = f"""You are solving a Traveling Salesman Problem (TSP) with {n} cities (C0, C1, ..., C{n-1}).

Distance matrix (symmetric):
{matrix_text}

Find the minimum cost tour that:
- Starts at city C0
- Visits every other city exactly once
- Returns to city C0

Try all permutations of cities C1..C{n-1} and compute each tour's total distance.

Show your reasoning, then express your final answer as \\boxed{{min_cost}}."""
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
        m = re.search(r'(?:minimum|min|optimal|total)\s+(?:cost|tour|distance)[^\d]*(\d+)', response, re.IGNORECASE)
        if m:
            return int(m.group(1))
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return int(nums[-1])
        return None

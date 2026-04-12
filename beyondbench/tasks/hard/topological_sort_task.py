"""
Topological Sort Task

Evaluates LLM's ability to produce a valid topological ordering of a DAG.

Data point: {'edges': [(u,v), ...], 'nodes': [...], 'answer': one_valid_ordering, 'all_valid': [...]}
Output: list of nodes (any valid topological ordering accepted)
"""

import random
from collections import deque
from itertools import permutations
from typing import Any, Dict, List, Optional, Set, Tuple

from ...core.base_task import BaseTask


class TopologicalSortSolver:
    """Topological sort and validation utilities."""

    @staticmethod
    def kahn_sort(nodes: List[int], edges: List[Tuple[int, int]]) -> List[int]:
        """Kahn's algorithm returns ONE valid topological order."""
        in_degree = {n: 0 for n in nodes}
        adj = {n: [] for n in nodes}
        for u, v in edges:
            adj[u].append(v)
            in_degree[v] += 1

        queue = deque(sorted([n for n in nodes if in_degree[n] == 0]))
        result = []
        while queue:
            u = queue.popleft()
            result.append(u)
            for v in sorted(adj[u]):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return result if len(result) == len(nodes) else []

    @staticmethod
    def is_valid_topological_order(ordering: List[int], nodes: List[int], edges: List[Tuple[int, int]]) -> bool:
        """Check if a given ordering is a valid topological sort."""
        if set(ordering) != set(nodes) or len(ordering) != len(nodes):
            return False
        pos = {n: i for i, n in enumerate(ordering)}
        for u, v in edges:
            if pos[u] >= pos[v]:
                return False
        return True

    @staticmethod
    def all_valid_orderings(nodes: List[int], edges: List[Tuple[int, int]]) -> List[List[int]]:
        """Enumerate all valid topological orderings (only for small graphs)."""
        valid = []
        for perm in permutations(nodes):
            if TopologicalSortSolver.is_valid_topological_order(list(perm), nodes, edges):
                valid.append(list(perm))
        return valid


class TopologicalSortTask(BaseTask):
    """Topological sort of a DAG task."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=6, **kwargs):
        self._task_name = "topological_sort"
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

    def _generate_dag(self, n: int, rng: random.Random) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Generate a random DAG with n nodes by only adding edges i->j where i<j."""
        nodes = list(range(n))
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    edges.append((i, j))
        return nodes, edges

    def generate_data(self, **kwargs):
        list_size = kwargs.get('list_size', None)
        n_nodes = list_size if list_size else self.problem_size

        test_cases = []
        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 53)
            n = rng.randint(5, min(7, max(5, n_nodes)))
            nodes, edges = self._generate_dag(n, rng)
            one_valid = TopologicalSortSolver.kahn_sort(nodes, edges)
            # For small n, compute all valid (n<=6 is feasible)
            if n <= 6:
                all_valid = TopologicalSortSolver.all_valid_orderings(nodes, edges)
            else:
                all_valid = [one_valid]

            test_cases.append({
                'nodes': nodes,
                'edges': edges,
                'answer': one_valid,
                'ground_truth': one_valid,
                'all_valid': all_valid,
            })
        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        nodes = data_point['nodes']
        edges = data_point['edges']

        node_list = ", ".join(str(n) for n in nodes)
        edge_list = ", ".join(f"({u},{v})" for u, v in edges) if edges else "(none)"

        prompt = f"""You are given a Directed Acyclic Graph (DAG).

Nodes: {{{node_list}}}
Directed edges (u -> v means u must come before v): {edge_list}

Find a valid topological ordering of all nodes.

A topological ordering lists every node such that for every directed edge (u, v),
node u appears before node v in the list.

Show your reasoning (e.g., compute in-degrees, process nodes with in-degree 0 first).

Express your final answer as \\boxed{{[n1, n2, n3, ...]}} listing all nodes in order."""
        return prompt

    def evaluate_response(self, response: str, data_point: Dict[str, Any]) -> Dict[str, Any]:
        nodes = data_point['nodes']
        edges = data_point['edges']
        ground_truth = data_point['answer']
        predicted = self._parse_answer(response, nodes)

        if predicted is not None:
            valid = TopologicalSortSolver.is_valid_topological_order(predicted, nodes, edges)
            accuracy = 1 if valid else 0
            instruction_followed = 1
        else:
            accuracy = 0
            instruction_followed = 0

        return {
            'accuracy': accuracy,
            'instruction_followed': instruction_followed,
            'ground_truth': ground_truth,
            'predicted_answer': predicted,
        }

    def _parse_answer(self, response: str, nodes: List[int]) -> Optional[List[int]]:
        import re
        # Try \boxed{[...]}
        m = re.search(r'\\boxed\{\[([^\]]+)\]\}', response)
        if m:
            try:
                return [int(x.strip()) for x in m.group(1).split(',')]
            except (ValueError, AttributeError):
                pass
        # Try any [...] with integers
        matches = re.findall(r'\[([0-9,\s]+)\]', response)
        for match in reversed(matches):
            try:
                lst = [int(x.strip()) for x in match.split(',')]
                if set(lst) == set(nodes):
                    return lst
            except ValueError:
                pass
        # Try comma-separated integers at end
        lines = response.strip().split('\n')
        for line in reversed(lines):
            parts = re.findall(r'\b(\d+)\b', line)
            if len(parts) == len(nodes):
                try:
                    lst = [int(p) for p in parts]
                    if set(lst) == set(nodes):
                        return lst
                except ValueError:
                    pass
        return None

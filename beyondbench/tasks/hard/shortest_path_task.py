"""
Shortest Path Task

Evaluates LLM's ability to find the shortest path cost in a weighted directed graph
using Dijkstra's algorithm reasoning.

Data point: {'graph': {node: [(neighbor, weight), ...]}, 'source': s, 'target': t, 'answer': cost}
Output: int (the minimum cost)
"""

import random
import heapq
from typing import Any, Dict, List, Optional, Tuple

from ...core.base_task import BaseTask


class ShortestPathSolver:
    """Correct Dijkstra solver for ground-truth."""

    @staticmethod
    def dijkstra(graph: Dict[int, List[Tuple[int, int]]], source: int, target: int, nodes: List[int]) -> Optional[int]:
        dist = {n: float('inf') for n in nodes}
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == target:
                return int(dist[target])
            for v, w in graph.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        result = dist[target]
        return int(result) if result < float('inf') else None


class ShortestPathTask(BaseTask):
    """Shortest path in a weighted directed graph."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=6, **kwargs):
        self._task_name = "shortest_path"
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

    def _generate_reachable_graph(self, n_nodes: int, rng: random.Random) -> Tuple[Dict, List, int, int, int]:
        """Generate a random directed weighted graph guaranteed to have a path from source to target."""
        nodes = list(range(n_nodes))
        source = 0
        target = n_nodes - 1

        # Build a path to guarantee reachability
        path_nodes = rng.sample(nodes[1:], min(n_nodes - 1, n_nodes - 1))
        path = [source] + path_nodes[:target] + [target]
        # simpler: just guarantee source->target via chain
        path = nodes[:]
        rng.shuffle(path[1:-1])  # keep source=0, target=n-1 fixed

        graph = {n: [] for n in nodes}
        used_edges = set()

        # Guaranteed path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            w = rng.randint(1, 20)
            graph[u].append((v, w))
            used_edges.add((u, v))

        # Random extra edges
        extra = rng.randint(n_nodes // 2, n_nodes)
        for _ in range(extra):
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            if u != v and (u, v) not in used_edges:
                w = rng.randint(1, 20)
                graph[u].append((v, w))
                used_edges.add((u, v))

        return graph, nodes, source, target

    def generate_data(self, **kwargs):
        """Generate evaluation data points."""
        list_size = kwargs.get('list_size', None)
        n_nodes = list_size if list_size else self.problem_size

        rng = random.Random(self.seed)
        test_cases = []

        for i in range(self.num_samples):
            local_seed = (self.seed or 0) + i * 31
            rng2 = random.Random(local_seed)
            n = rng2.randint(5, min(8, max(5, n_nodes)))
            graph, nodes, source, target = self._generate_reachable_graph(n, rng2)
            answer = ShortestPathSolver.dijkstra(graph, source, target, nodes)
            if answer is None:
                # fallback simple graph
                graph = {0: [(1, 3), (2, 5)], 1: [(3, 2)], 2: [(3, 1)], 3: []}
                nodes = [0, 1, 2, 3]
                source, target = 0, 3
                answer = ShortestPathSolver.dijkstra(graph, source, target, nodes)

            test_cases.append({
                'graph': graph,
                'nodes': nodes,
                'source': source,
                'target': target,
                'answer': answer,
                'ground_truth': answer,
            })

        return test_cases

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        graph = data_point['graph']
        nodes = data_point['nodes']
        source = data_point['source']
        target = data_point['target']

        # Build edge list display
        edge_lines = []
        for u in sorted(graph.keys()):
            for v, w in graph[u]:
                edge_lines.append(f"  {u} -> {v}  (weight {w})")

        edge_text = "\n".join(edge_lines) if edge_lines else "  (no edges)"
        node_list = ", ".join(str(n) for n in sorted(nodes))

        prompt = f"""You are given a weighted directed graph with nodes {{{node_list}}}.

Directed edges (from -> to, weight):
{edge_text}

Find the minimum cost (shortest path) from node {source} to node {target}.

Use Dijkstra's algorithm or any shortest-path method. Show your reasoning step by step.

Express your final answer as \\boxed{{cost}} where cost is the integer minimum path cost."""
        return prompt

    def evaluate_response(self, response: str, data_point: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = data_point['answer']
        predicted = self._parse_answer(response)

        accuracy = 1 if (predicted is not None and int(predicted) == int(ground_truth)) else 0
        instruction_followed = 1 if (predicted is not None) else 0

        return {
            'accuracy': accuracy,
            'instruction_followed': instruction_followed,
            'ground_truth': ground_truth,
            'predicted_answer': predicted,
        }

    def _parse_answer(self, response: str) -> Optional[int]:
        import re
        # Try \boxed{N}
        m = re.search(r'\\boxed\{(\d+)\}', response)
        if m:
            return int(m.group(1))
        # Try "answer is N" / "= N" at end
        m = re.search(r'(?:answer|cost|shortest path)[^\d]*(\d+)', response, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # Last integer in response
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return int(nums[-1])
        return None

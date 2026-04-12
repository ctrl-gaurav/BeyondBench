"""
Minimax Game Task

Evaluates LLM's ability to compute the optimal value for the maximizing player
in a binary game tree using the minimax algorithm.

Data point: {'tree': nested_list, 'depth': d, 'answer': optimal_value}
Output: int
"""

import random
from typing import Any, Dict, List, Optional, Union

from ...core.base_task import BaseTask

# Type alias: a tree node is either an int (leaf) or [left, right]
TreeNode = Union[int, List]


class MinimaxSolver:
    """Recursive minimax solver."""

    @staticmethod
    def minimax(node: TreeNode, is_maximizing: bool) -> int:
        if isinstance(node, (int, float)):
            return int(node)
        left, right = node[0], node[1]
        left_val = MinimaxSolver.minimax(left, not is_maximizing)
        right_val = MinimaxSolver.minimax(right, not is_maximizing)
        if is_maximizing:
            return max(left_val, right_val)
        else:
            return min(left_val, right_val)

    @staticmethod
    def generate_tree(depth: int, rng: random.Random) -> TreeNode:
        """Generate a binary tree of given depth with random leaf values."""
        if depth == 0:
            return rng.randint(1, 20)
        left = MinimaxSolver.generate_tree(depth - 1, rng)
        right = MinimaxSolver.generate_tree(depth - 1, rng)
        return [left, right]

    @staticmethod
    def tree_to_str(node: TreeNode, indent: int = 0) -> str:
        prefix = "  " * indent
        if isinstance(node, (int, float)):
            return f"{prefix}{int(node)}"
        left_str = MinimaxSolver.tree_to_str(node[0], indent + 1)
        right_str = MinimaxSolver.tree_to_str(node[1], indent + 1)
        return f"{prefix}[\n{left_str},\n{right_str}\n{prefix}]"

    @staticmethod
    def get_leaves(node: TreeNode) -> List[int]:
        if isinstance(node, (int, float)):
            return [int(node)]
        return MinimaxSolver.get_leaves(node[0]) + MinimaxSolver.get_leaves(node[1])


class MinimaxGameTask(BaseTask):
    """Minimax game tree optimal value task."""

    def __init__(self, model_handler, output_dir, min_val=1, max_val=100,
                 num_folds=1, num_samples=5, store_details=False,
                 temperature=0.1, top_p=0.9, max_tokens=2048,
                 seed=None, problem_size=3, **kwargs):
        self._task_name = "minimax_game"
        self.problem_size = problem_size  # tree depth
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
        base_depth = list_size if list_size else self.problem_size

        test_cases = []
        for i in range(self.num_samples):
            rng = random.Random((self.seed or 0) + i * 37)
            depth = rng.randint(2, min(4, max(2, base_depth)))
            tree = MinimaxSolver.generate_tree(depth, rng)
            answer = MinimaxSolver.minimax(tree, is_maximizing=True)
            test_cases.append({
                'tree': tree,
                'depth': depth,
                'answer': answer,
                'ground_truth': answer,
                'leaves': MinimaxSolver.get_leaves(tree),
            })
        return test_cases

    def _format_tree(self, node: TreeNode, depth: int, is_max: bool, level: int = 0) -> str:
        """Format tree for display."""
        role = "MAX" if is_max else "MIN"
        indent = "  " * level
        if isinstance(node, (int, float)):
            return f"{indent}Leaf: {int(node)}"
        left_str = self._format_tree(node[0], depth - 1, not is_max, level + 1)
        right_str = self._format_tree(node[1], depth - 1, not is_max, level + 1)
        return f"{indent}{role} node:\n{left_str}\n{right_str}"

    def create_prompt(self, data_point: Dict[str, Any]) -> str:
        tree = data_point['tree']
        depth = data_point['depth']
        leaves = data_point['leaves']

        tree_display = self._format_tree(tree, depth, is_max=True)
        leaves_str = ", ".join(str(v) for v in leaves)

        prompt = f"""You are playing a two-player zero-sum game with a binary game tree of depth {depth}.

The root is the MAXIMIZING player's turn. Players alternate (MAX, MIN, MAX, ...).
Leaf values are: [{leaves_str}]

Tree structure (root at top, leaves at depth {depth}):
{tree_display}

Use the minimax algorithm:
- MAX nodes choose the MAXIMUM of their children's values.
- MIN nodes choose the MINIMUM of their children's values.
- Leaf nodes return their given value.

Compute the optimal value bottom-up, layer by layer.

Express your final answer as \\boxed{{optimal_value}}."""
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
        m = re.search(r'(?:optimal|minimax|root|answer)[^\d]*(\d+)', response, re.IGNORECASE)
        if m:
            return int(m.group(1))
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return int(nums[-1])
        return None

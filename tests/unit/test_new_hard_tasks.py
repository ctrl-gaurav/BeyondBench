"""
Unit tests for Phase 14 hard tasks (10 new tasks).

Tests cover:
- Task instantiation
- generate_data structure
- Ground-truth solver correctness (known cases)
- create_prompt has \\boxed
- evaluate_response accuracy=1 on correct, 0 on wrong
- Parser handles multiple formats
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================================
# Solver imports for unit testing (no model_handler needed)
# ============================================================================

from beyondbench.tasks.hard.shortest_path_task import ShortestPathSolver, ShortestPathTask
from beyondbench.tasks.hard.knapsack_task import KnapsackSolver, KnapsackTask
from beyondbench.tasks.hard.traveling_salesman_task import TSPSolver, TravelingSalesmanTask
from beyondbench.tasks.hard.longest_common_subsequence_task import LCSSolver, LongestCommonSubsequenceTask
from beyondbench.tasks.hard.minimax_game_task import MinimaxSolver, MinimaxGameTask
from beyondbench.tasks.hard.regex_matching_task import RegexMatchingSolver, RegexMatchingTask
from beyondbench.tasks.hard.topological_sort_task import TopologicalSortSolver, TopologicalSortTask
from beyondbench.tasks.hard.interval_scheduling_task import IntervalSchedulingSolver, IntervalSchedulingTask
from beyondbench.tasks.hard.coin_change_task import CoinChangeSolver, CoinChangeTask
from beyondbench.tasks.hard.edit_distance_task import EditDistanceSolver, EditDistanceTask


# ============================================================================
# Shared mock for model_handler
# ============================================================================

class MockModelHandler:
    def get_model_info(self):
        return {'model_name': 'mock', 'backend': 'mock'}
    def generate(self, prompt, **kwargs):
        return "mock response"


@pytest.fixture
def mock_handler(tmp_path):
    return MockModelHandler(), str(tmp_path)


def make_task(TaskClass, tmp_path, **kwargs):
    handler = MockModelHandler()
    return TaskClass(
        model_handler=handler,
        output_dir=str(tmp_path),
        num_samples=3,
        seed=42,
        **kwargs
    )


# ============================================================================
# 1. ShortestPathTask
# ============================================================================

class TestShortestPathSolver:
    def test_simple_graph(self):
        # 0->1 (3), 0->2 (5), 1->3 (2), 2->3 (1)
        graph = {0: [(1, 3), (2, 5)], 1: [(3, 2)], 2: [(3, 1)], 3: []}
        nodes = [0, 1, 2, 3]
        result = ShortestPathSolver.dijkstra(graph, 0, 3, nodes)
        assert result == 5, f"Expected 5, got {result}"  # 0->1->3 = 5

    def test_direct_edge(self):
        graph = {0: [(1, 10)], 1: [(2, 1)], 2: []}
        nodes = [0, 1, 2]
        result = ShortestPathSolver.dijkstra(graph, 0, 2, nodes)
        assert result == 11

    def test_longer_path(self):
        # multiple paths
        graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
        nodes = [0, 1, 2, 3]
        # 0->2->1->3 = 1+2+1 = 4, 0->1->3 = 4+1=5
        result = ShortestPathSolver.dijkstra(graph, 0, 3, nodes)
        assert result == 4


class TestShortestPathTask:
    def test_instantiable(self, tmp_path):
        task = make_task(ShortestPathTask, tmp_path)
        assert task.task_name == "shortest_path"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(ShortestPathTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'graph' in dp
            assert 'source' in dp
            assert 'target' in dp
            assert 'answer' in dp
            assert isinstance(dp['answer'], int)

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(ShortestPathTask, tmp_path)
        data = task.generate_data()
        prompt = task.create_prompt(data[0])
        assert '\\boxed' in prompt

    def test_evaluate_correct(self, tmp_path):
        task = make_task(ShortestPathTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        answer = dp['answer']
        response = f"The shortest path cost is \\boxed{{{answer}}}"
        result = task.evaluate_response(response, dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(ShortestPathTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        wrong = dp['answer'] + 999
        response = f"\\boxed{{{wrong}}}"
        result = task.evaluate_response(response, dp)
        assert result['accuracy'] == 0

    def test_parser_formats(self, tmp_path):
        task = make_task(ShortestPathTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        ans = dp['answer']
        formats = [
            f"\\boxed{{{ans}}}",
            f"The answer is {ans}",
            f"shortest path cost = {ans}",
            f"answer: {ans}",
        ]
        for fmt in formats:
            parsed = task._parse_answer(fmt)
            assert parsed == ans, f"Format '{fmt}' -> {parsed}, expected {ans}"


# ============================================================================
# 2. KnapsackTask
# ============================================================================

class TestKnapsackSolver:
    def test_basic(self):
        items = [(2, 3), (3, 4), (4, 5), (5, 6)]
        assert KnapsackSolver.solve(items, 5) == 7  # take (2,3)+(3,4)=7

    def test_single_item_fits(self):
        items = [(3, 10), (5, 7)]
        assert KnapsackSolver.solve(items, 3) == 10

    def test_capacity_zero(self):
        items = [(1, 5), (2, 10)]
        assert KnapsackSolver.solve(items, 0) == 0

    def test_all_items(self):
        items = [(1, 2), (1, 3), (1, 4)]
        assert KnapsackSolver.solve(items, 3) == 9


class TestKnapsackTask:
    def test_instantiable(self, tmp_path):
        task = make_task(KnapsackTask, tmp_path)
        assert task.task_name == "knapsack"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(KnapsackTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'items' in dp
            assert 'capacity' in dp
            assert 'answer' in dp
            assert isinstance(dp['answer'], int)
            assert dp['answer'] >= 0

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(KnapsackTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(KnapsackTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        response = f"\\boxed{{{dp['answer']}}}"
        result = task.evaluate_response(response, dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(KnapsackTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        response = f"\\boxed{{{dp['answer'] + 999}}}"
        result = task.evaluate_response(response, dp)
        assert result['accuracy'] == 0

    def test_parser_formats(self, tmp_path):
        task = make_task(KnapsackTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        ans = dp['answer']
        for fmt in [f"\\boxed{{{ans}}}", f"max value is {ans}", f"answer: {ans}"]:
            assert task._parse_answer(fmt) == ans


# ============================================================================
# 3. TravelingSalesmanTask
# ============================================================================

class TestTSPSolver:
    def test_4_cities(self):
        dist = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        # Best tour: 0->1->3->2->0 = 10+25+30+15 = 80
        result = TSPSolver.solve(dist)
        assert result == 80

    def test_symmetric(self):
        dist = [[0, 5, 9], [5, 0, 6], [9, 6, 0]]
        # 0->1->2->0 = 5+6+9=20, 0->2->1->0=9+6+5=20
        result = TSPSolver.solve(dist)
        assert result == 20

    def test_4_uniform(self):
        dist = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        result = TSPSolver.solve(dist)
        assert result == 4


class TestTravelingSalesmanTask:
    def test_instantiable(self, tmp_path):
        task = make_task(TravelingSalesmanTask, tmp_path)
        assert task.task_name == "traveling_salesman"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(TravelingSalesmanTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'distance_matrix' in dp
            assert 'num_cities' in dp
            assert 'answer' in dp
            n = dp['num_cities']
            assert len(dp['distance_matrix']) == n

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(TravelingSalesmanTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(TravelingSalesmanTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer']}}}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(TravelingSalesmanTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer'] + 999}}}", dp)
        assert result['accuracy'] == 0


# ============================================================================
# 4. LongestCommonSubsequenceTask
# ============================================================================

class TestLCSSolver:
    def test_abc_ac(self):
        assert LCSSolver.lcs_length("abc", "ac") == 2

    def test_abcdef_ace(self):
        assert LCSSolver.lcs_length("abcdef", "ace") == 3

    def test_identical(self):
        assert LCSSolver.lcs_length("abcd", "abcd") == 4

    def test_disjoint(self):
        assert LCSSolver.lcs_length("abc", "def") == 0

    def test_empty(self):
        assert LCSSolver.lcs_length("", "abc") == 0


class TestLongestCommonSubsequenceTask:
    def test_instantiable(self, tmp_path):
        task = make_task(LongestCommonSubsequenceTask, tmp_path)
        assert task.task_name == "longest_common_subsequence"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(LongestCommonSubsequenceTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'string1' in dp and 'string2' in dp and 'answer' in dp
            assert isinstance(dp['answer'], int)
            assert 0 <= dp['answer'] <= min(len(dp['string1']), len(dp['string2']))

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(LongestCommonSubsequenceTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(LongestCommonSubsequenceTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer']}}}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(LongestCommonSubsequenceTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer'] + 100}}}", dp)
        assert result['accuracy'] == 0


# ============================================================================
# 5. MinimaxGameTask
# ============================================================================

class TestMinimaxSolver:
    def test_depth1_max(self):
        # Depth 1: [3, 5] -> MAX picks 5
        tree = [3, 5]
        assert MinimaxSolver.minimax(tree, is_maximizing=True) == 5

    def test_depth2(self):
        # Root MAX, children MIN
        # Left MIN picks min(3, 5) = 3
        # Right MIN picks min(2, 9) = 2
        # Root MAX picks max(3, 2) = 3
        tree = [[3, 5], [2, 9]]
        assert MinimaxSolver.minimax(tree, is_maximizing=True) == 3

    def test_depth3(self):
        # Root MAX -> depth1 MIN -> depth2 MAX leaves
        # Left subtree MIN: max(3,5)=5, max(2,9)=9 -> MIN picks 5
        # Right subtree MIN: max(1,7)=7, max(4,6)=6 -> MIN picks 6
        # Root MAX: max(5,6) = 6
        tree = [[[3, 5], [2, 9]], [[1, 7], [4, 6]]]
        assert MinimaxSolver.minimax(tree, is_maximizing=True) == 6

    def test_leaf(self):
        assert MinimaxSolver.minimax(7, is_maximizing=True) == 7


class TestMinimaxGameTask:
    def test_instantiable(self, tmp_path):
        task = make_task(MinimaxGameTask, tmp_path)
        assert task.task_name == "minimax_game"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(MinimaxGameTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'tree' in dp and 'depth' in dp and 'answer' in dp
            assert isinstance(dp['answer'], int)

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(MinimaxGameTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(MinimaxGameTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer']}}}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(MinimaxGameTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer'] + 100}}}", dp)
        assert result['accuracy'] == 0


# ============================================================================
# 6. RegexMatchingTask
# ============================================================================

class TestRegexMatchingSolver:
    def test_simple_match(self):
        assert RegexMatchingSolver.matches('a+b', 'aaab') == True

    def test_simple_no_match(self):
        assert RegexMatchingSolver.matches('a+b', 'b') == False

    def test_digit_match(self):
        assert RegexMatchingSolver.matches(r'\d+', '123') == True

    def test_digit_no_match(self):
        assert RegexMatchingSolver.matches(r'\d+', 'abc') == False

    def test_fullmatch_not_search(self):
        # fullmatch requires whole string to match
        assert RegexMatchingSolver.matches('[abc]+', 'abcd') == False
        assert RegexMatchingSolver.matches('[abc]+', 'abc') == True


class TestRegexMatchingTask:
    def test_instantiable(self, tmp_path):
        task = make_task(RegexMatchingTask, tmp_path)
        assert task.task_name == "regex_matching"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(RegexMatchingTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'pattern' in dp and 'test_string' in dp and 'answer' in dp
            assert isinstance(dp['answer'], bool)

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(RegexMatchingTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct_true(self, tmp_path):
        task = make_task(RegexMatchingTask, tmp_path)
        dp = {'pattern': 'a+b', 'test_string': 'aab', 'answer': True, 'ground_truth': True}
        result = task.evaluate_response("\\boxed{true}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_correct_false(self, tmp_path):
        task = make_task(RegexMatchingTask, tmp_path)
        dp = {'pattern': 'a+b', 'test_string': 'b', 'answer': False, 'ground_truth': False}
        result = task.evaluate_response("\\boxed{false}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(RegexMatchingTask, tmp_path)
        dp = {'pattern': 'a+b', 'test_string': 'aab', 'answer': True, 'ground_truth': True}
        result = task.evaluate_response("\\boxed{false}", dp)
        assert result['accuracy'] == 0

    def test_parser_formats(self, tmp_path):
        task = make_task(RegexMatchingTask, tmp_path)
        assert task._parse_answer("\\boxed{true}") == True
        assert task._parse_answer("\\boxed{false}") == False
        assert task._parse_answer("The answer is True") == True
        assert task._parse_answer("The answer is False") == False
        assert task._parse_answer("Yes, it matches") == True
        assert task._parse_answer("No, it does not match") == False


# ============================================================================
# 7. TopologicalSortTask
# ============================================================================

class TestTopologicalSortSolver:
    def test_simple_chain(self):
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        result = TopologicalSortSolver.kahn_sort(nodes, edges)
        assert result == [0, 1, 2]
        assert TopologicalSortSolver.is_valid_topological_order(result, nodes, edges)

    def test_no_edges(self):
        nodes = [0, 1, 2]
        edges = []
        result = TopologicalSortSolver.kahn_sort(nodes, edges)
        assert set(result) == set(nodes)
        assert TopologicalSortSolver.is_valid_topological_order(result, nodes, edges)

    def test_diamond(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        result = TopologicalSortSolver.kahn_sort(nodes, edges)
        assert TopologicalSortSolver.is_valid_topological_order(result, nodes, edges)

    def test_invalid_order(self):
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        assert not TopologicalSortSolver.is_valid_topological_order([2, 1, 0], nodes, edges)


class TestTopologicalSortTask:
    def test_instantiable(self, tmp_path):
        task = make_task(TopologicalSortTask, tmp_path)
        assert task.task_name == "topological_sort"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(TopologicalSortTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'nodes' in dp and 'edges' in dp and 'answer' in dp
            assert isinstance(dp['answer'], list)

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(TopologicalSortTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(TopologicalSortTask, tmp_path)
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        dp = {'nodes': nodes, 'edges': edges, 'answer': [0, 1, 2], 'ground_truth': [0, 1, 2]}
        result = task.evaluate_response("\\boxed{[0, 1, 2]}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(TopologicalSortTask, tmp_path)
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        dp = {'nodes': nodes, 'edges': edges, 'answer': [0, 1, 2], 'ground_truth': [0, 1, 2]}
        result = task.evaluate_response("\\boxed{[2, 1, 0]}", dp)
        assert result['accuracy'] == 0

    def test_parser_list(self, tmp_path):
        task = make_task(TopologicalSortTask, tmp_path)
        nodes = [0, 1, 2, 3]
        parsed = task._parse_answer("The answer is \\boxed{[0, 1, 2, 3]}", nodes)
        assert parsed == [0, 1, 2, 3]

        parsed2 = task._parse_answer("[0, 1, 2, 3]", nodes)
        assert parsed2 == [0, 1, 2, 3]


# ============================================================================
# 8. IntervalSchedulingTask
# ============================================================================

class TestIntervalSchedulingSolver:
    def test_classic(self):
        intervals = [(1, 4), (2, 6), (5, 8), (7, 9)]
        # Sort by end: (1,4),(2,6),(5,8),(7,9)
        # Pick (1,4), skip (2,6), pick (5,8), pick (7,9) -> 3? No: 7>=8 is False
        # Pick (1,4), skip (2,6)[start=2<4], pick (5,8)[start=5>=4], skip (7,9)[start=7<8] -> 2
        assert IntervalSchedulingSolver.max_non_overlapping(intervals) == 2

    def test_no_overlap(self):
        intervals = [(1, 2), (3, 4), (5, 6)]
        assert IntervalSchedulingSolver.max_non_overlapping(intervals) == 3

    def test_all_overlap(self):
        intervals = [(1, 5), (2, 6), (3, 7)]
        assert IntervalSchedulingSolver.max_non_overlapping(intervals) == 1

    def test_single(self):
        assert IntervalSchedulingSolver.max_non_overlapping([(1, 3)]) == 1


class TestIntervalSchedulingTask:
    def test_instantiable(self, tmp_path):
        task = make_task(IntervalSchedulingTask, tmp_path)
        assert task.task_name == "interval_scheduling"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(IntervalSchedulingTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'intervals' in dp and 'answer' in dp
            assert isinstance(dp['answer'], int)
            assert dp['answer'] >= 1

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(IntervalSchedulingTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(IntervalSchedulingTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer']}}}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(IntervalSchedulingTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer'] + 100}}}", dp)
        assert result['accuracy'] == 0


# ============================================================================
# 9. CoinChangeTask
# ============================================================================

class TestCoinChangeSolver:
    def test_basic(self):
        assert CoinChangeSolver.min_coins([1, 5, 10], 11) == 2  # 10+1

    def test_quarters(self):
        assert CoinChangeSolver.min_coins([1, 5, 10, 25], 30) == 2  # 25+5

    def test_no_coin_1(self):
        assert CoinChangeSolver.min_coins([2, 5], 3) == -1  # impossible without 1

    def test_exact_coin(self):
        assert CoinChangeSolver.min_coins([1, 5, 10], 10) == 1

    def test_all_ones(self):
        assert CoinChangeSolver.min_coins([1], 7) == 7


class TestCoinChangeTask:
    def test_instantiable(self, tmp_path):
        task = make_task(CoinChangeTask, tmp_path)
        assert task.task_name == "coin_change"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(CoinChangeTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'coins' in dp and 'target' in dp and 'answer' in dp
            assert 1 in dp['coins'], "coin 1 must always be present"
            assert dp['answer'] >= 0

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(CoinChangeTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(CoinChangeTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer']}}}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(CoinChangeTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer'] + 999}}}", dp)
        assert result['accuracy'] == 0


# ============================================================================
# 10. EditDistanceTask
# ============================================================================

class TestEditDistanceSolver:
    def test_empty_to_abc(self):
        assert EditDistanceSolver.edit_distance("", "abc") == 3

    def test_abc_to_abc(self):
        assert EditDistanceSolver.edit_distance("abc", "abc") == 0

    def test_kitten_sitting(self):
        assert EditDistanceSolver.edit_distance("kitten", "sitting") == 3

    def test_sunday_saturday(self):
        assert EditDistanceSolver.edit_distance("sunday", "saturday") == 3

    def test_single_chars(self):
        assert EditDistanceSolver.edit_distance("a", "b") == 1
        assert EditDistanceSolver.edit_distance("a", "a") == 0


class TestEditDistanceTask:
    def test_instantiable(self, tmp_path):
        task = make_task(EditDistanceTask, tmp_path)
        assert task.task_name == "edit_distance"

    def test_generate_data_structure(self, tmp_path):
        task = make_task(EditDistanceTask, tmp_path)
        data = task.generate_data()
        assert len(data) == 3
        for dp in data:
            assert 'string1' in dp and 'string2' in dp and 'answer' in dp
            assert isinstance(dp['answer'], int)
            assert dp['answer'] >= 0

    def test_create_prompt_has_boxed(self, tmp_path):
        task = make_task(EditDistanceTask, tmp_path)
        data = task.generate_data()
        assert '\\boxed' in task.create_prompt(data[0])

    def test_evaluate_correct(self, tmp_path):
        task = make_task(EditDistanceTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer']}}}", dp)
        assert result['accuracy'] == 1

    def test_evaluate_wrong(self, tmp_path):
        task = make_task(EditDistanceTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        result = task.evaluate_response(f"\\boxed{{{dp['answer'] + 100}}}", dp)
        assert result['accuracy'] == 0

    def test_parser_formats(self, tmp_path):
        task = make_task(EditDistanceTask, tmp_path)
        data = task.generate_data()
        dp = data[0]
        ans = dp['answer']
        formats = [
            f"\\boxed{{{ans}}}",
            f"edit distance is {ans}",
            f"Levenshtein distance: {ans}",
            f"answer = {ans}",
        ]
        for fmt in formats:
            parsed = task._parse_answer(fmt)
            assert parsed == ans, f"Failed for format: {fmt}"


# ============================================================================
# Registry test
# ============================================================================

class TestRegistryPhase14:
    def test_hard_tasks_count(self):
        from beyondbench.core.task_registry import TaskRegistry
        registry = TaskRegistry()
        hard_tasks = registry.get_tasks_for_suite('hard')
        assert len(hard_tasks) == 20, f"Expected 20 hard tasks, got {len(hard_tasks)}"

    def test_all_phase14_tasks_importable(self):
        from beyondbench.core.task_registry import TaskRegistry
        registry = TaskRegistry()
        phase14_tasks = [
            'shortest_path', 'knapsack', 'traveling_salesman',
            'longest_common_subsequence', 'minimax_game', 'regex_matching',
            'topological_sort', 'interval_scheduling', 'coin_change', 'edit_distance',
        ]
        for task_name in phase14_tasks:
            cls = registry.get_task_class(task_name)
            assert cls is not None, f"Task '{task_name}' not found in registry"

    def test_parser_configs_exist(self):
        from beyondbench.parsers.task_configs import get_task_config
        phase14_tasks = [
            'shortest_path', 'knapsack', 'traveling_salesman',
            'longest_common_subsequence', 'minimax_game', 'regex_matching',
            'topological_sort', 'interval_scheduling', 'coin_change', 'edit_distance',
        ]
        for task_name in phase14_tasks:
            config = get_task_config(task_name)
            assert config is not None, f"No parser config for '{task_name}'"

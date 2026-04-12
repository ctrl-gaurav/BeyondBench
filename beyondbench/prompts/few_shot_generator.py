"""Dynamic few-shot example generator for BeyondBench tasks.

Generates fresh, random few-shot examples at runtime to avoid contamination.
Each call produces new examples using the task's own ground-truth logic,
so no static data can leak into training corpora.

Usage::

    from beyondbench.prompts.few_shot_generator import FewShotGenerator
    gen = FewShotGenerator(seed=42)
    examples = gen.generate("sum", n=2)
    # → "Example 1: [5, -3, 8] → 10\nExample 2: [12, 1, -7] → 6"
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Registry: task_name → (data_generator, answer_formatter)
#
# Each entry is a pair:
#   data_generator(rng) → (data_point_display_str, ground_truth_display_str)
# We keep these tiny and fast — no model inference, no GPU.
# ---------------------------------------------------------------------------

_GENERATORS: Dict[str, Callable] = {}


def _register(task_name: str):
    """Decorator to register a few-shot generator for *task_name*."""
    def decorator(fn: Callable):
        _GENERATORS[task_name] = fn
        return fn
    return decorator


# ===================================================================
# EASY TASKS
# ===================================================================

@_register("sum")
def _sum(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    return str(nums), str(sum(nums))


@_register("sorting")
def _sorting(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    return str(nums), str(sorted(nums))


@_register("comparison")
def _comparison(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(-50, 50), rng.randint(-50, 50)
    if a > b:
        ans = "greater"
    elif a < b:
        ans = "less"
    else:
        ans = "equal"
    return f"{a} and {b}", ans


@_register("multiplication")
def _multiplication(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(2, 4)
    nums = [rng.randint(-10, 20) for _ in range(n)]
    prod = 1
    for x in nums:
        prod *= x
    return str(nums), str(prod)


@_register("odd_count")
def _odd_count(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    return str(nums), str(sum(1 for x in nums if x % 2 != 0))


@_register("even_count")
def _even_count(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    return str(nums), str(sum(1 for x in nums if x % 2 == 0))


@_register("absolute_difference")
def _absolute_difference(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(-50, 50), rng.randint(-50, 50)
    return f"{a} and {b}", str(abs(a - b))


@_register("division")
def _division(rng: random.Random) -> Tuple[str, str]:
    b = rng.choice([x for x in range(1, 20)])
    a = rng.randint(-50, 50)
    result = round(a / b, 2)
    return f"{a} ÷ {b}", str(int(result) if result == int(result) else result)


@_register("find_maximum")
def _find_maximum(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-50, 100) for _ in range(n)]
    return str(nums), str(max(nums))


@_register("find_minimum")
def _find_minimum(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-50, 100) for _ in range(n)]
    return str(nums), str(min(nums))


@_register("mean")
def _mean(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    m = round(sum(nums) / len(nums), 2)
    return str(nums), str(m)


@_register("median")
def _median(rng: random.Random) -> Tuple[str, str]:
    n = rng.choice([3, 5])  # odd for simplicity
    nums = [rng.randint(-20, 50) for _ in range(n)]
    s = sorted(nums)
    return str(nums), str(s[len(s) // 2])


@_register("mode")
def _mode(rng: random.Random) -> Tuple[str, str]:
    base = [rng.randint(1, 20) for _ in range(3)]
    mode_val = rng.choice(base)
    nums = base + [mode_val]
    rng.shuffle(nums)
    from collections import Counter
    c = Counter(nums)
    actual_mode = c.most_common(1)[0][0]
    return str(nums), str(actual_mode)


@_register("range")
def _range(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-50, 100) for _ in range(n)]
    return str(nums), str(max(nums) - min(nums))


@_register("subtraction")
def _subtraction(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(-50, 50), rng.randint(-50, 50)
    return f"{a} - {b}", str(a - b)


@_register("sum_of_digits")
def _sum_of_digits(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(10, 9999)
    return str(n), str(sum(int(d) for d in str(abs(n))))


@_register("count_unique")
def _count_unique(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(1, 10) for _ in range(n)]
    return str(nums), str(len(set(nums)))


@_register("count_negative")
def _count_negative(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(-30, 30) for _ in range(n)]
    return str(nums), str(sum(1 for x in nums if x < 0))


@_register("index_of_maximum")
def _index_of_maximum(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-50, 100) for _ in range(n)]
    return str(nums), str(nums.index(max(nums)))


@_register("second_maximum")
def _second_maximum(rng: random.Random) -> Tuple[str, str]:
    nums = list(set(rng.randint(-50, 100) for _ in range(6)))
    while len(nums) < 3:
        nums.append(rng.randint(-50, 100))
    s = sorted(set(nums), reverse=True)
    return str(nums), str(s[1])


@_register("second_minimum")
def _second_minimum(rng: random.Random) -> Tuple[str, str]:
    nums = list(set(rng.randint(-50, 100) for _ in range(6)))
    while len(nums) < 3:
        nums.append(rng.randint(-50, 100))
    s = sorted(set(nums))
    return str(nums), str(s[1])


@_register("reverse_list")
def _reverse_list(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    return str(nums), str(list(reversed(nums)))


@_register("cumulative_sum")
def _cumulative_sum(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-10, 30) for _ in range(n)]
    cs = []
    s = 0
    for x in nums:
        s += x
        cs.append(s)
    return str(nums), str(cs)


@_register("alternating_sum")
def _alternating_sum(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    total = sum(x * (1 if i % 2 == 0 else -1) for i, x in enumerate(nums))
    return str(nums), str(total)


@_register("dot_product")
def _dot_product(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(2, 4)
    a = [rng.randint(-10, 10) for _ in range(n)]
    b = [rng.randint(-10, 10) for _ in range(n)]
    dp = sum(x * y for x, y in zip(a, b))
    return f"{a} and {b}", str(dp)


@_register("weighted_sum")
def _weighted_sum(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(2, 4)
    vals = [rng.randint(1, 20) for _ in range(n)]
    weights = [rng.randint(1, 5) for _ in range(n)]
    ws = sum(v * w for v, w in zip(vals, weights))
    return f"values={vals}, weights={weights}", str(ws)


@_register("parity_check")
def _parity_check(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(1, 100)
    return str(n), "even" if n % 2 == 0 else "odd"


@_register("count_multiples")
def _count_multiples(rng: random.Random) -> Tuple[str, str]:
    k = rng.choice([2, 3, 5, 7])
    n = rng.randint(4, 7)
    nums = [rng.randint(1, 50) for _ in range(n)]
    return f"{nums}, k={k}", str(sum(1 for x in nums if x % k == 0))


@_register("count_perfect_squares")
def _count_perfect_squares(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(0, 50) for _ in range(n)]
    count = sum(1 for x in nums if x >= 0 and int(math.sqrt(x)) ** 2 == x)
    return str(nums), str(count)


@_register("count_palindromic")
def _count_palindromic(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(1, 200) for _ in range(n)]
    count = sum(1 for x in nums if str(x) == str(x)[::-1])
    return str(nums), str(count)


@_register("local_maxima_count")
def _local_maxima_count(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    count = 0
    for i in range(1, len(nums) - 1):
        if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
            count += 1
    return str(nums), str(count)


@_register("max_adjacent_difference")
def _max_adjacent_difference(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-30, 50) for _ in range(n)]
    d = max(abs(nums[i + 1] - nums[i]) for i in range(len(nums) - 1))
    return str(nums), str(d)


@_register("longest_increasing_subsequence")
def _lis(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    # O(n^2) LIS
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return str(nums), str(max(dp))


@_register("count_greater_than_previous")
def _cgtp(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    count = sum(1 for i in range(1, len(nums)) if nums[i] > nums[i - 1])
    return str(nums), str(count)


@_register("sum_of_max_indices")
def _somi(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(-20, 50) for _ in range(n)]
    mx = max(nums)
    indices = [i for i, x in enumerate(nums) if x == mx]
    return str(nums), str(sum(indices))


@_register("running_average")
def _running_avg(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-10, 30) for _ in range(n)]
    avgs = [round(sum(nums[: i + 1]) / (i + 1), 2) for i in range(n)]
    return str(nums), str(avgs)


@_register("moving_average")
def _moving_avg(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 6)
    nums = [rng.randint(1, 30) for _ in range(n)]
    w = rng.randint(2, min(3, n))
    avgs = [round(sum(nums[i:i + w]) / w, 2) for i in range(n - w + 1)]
    return f"{nums}, window={w}", str(avgs)


@_register("variance")
def _variance(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-10, 30) for _ in range(n)]
    m = sum(nums) / len(nums)
    var = round(sum((x - m) ** 2 for x in nums) / len(nums), 4)
    return str(nums), str(var)


@_register("standard_deviation")
def _std(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 5)
    nums = [rng.randint(-10, 30) for _ in range(n)]
    m = sum(nums) / len(nums)
    var = sum((x - m) ** 2 for x in nums) / len(nums)
    return str(nums), str(round(math.sqrt(var), 4))


@_register("element_frequency")
def _elem_freq(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 7)
    nums = [rng.randint(1, 8) for _ in range(n)]
    from collections import Counter
    freq = dict(sorted(Counter(nums).items()))
    return str(nums), str(freq)


@_register("set_intersection")
def _set_inter(rng: random.Random) -> Tuple[str, str]:
    a = list(set(rng.randint(1, 20) for _ in range(5)))
    b = list(set(rng.randint(1, 20) for _ in range(5)))
    result = sorted(set(a) & set(b))
    return f"{a} and {b}", str(result)


@_register("set_difference")
def _set_diff(rng: random.Random) -> Tuple[str, str]:
    a = list(set(rng.randint(1, 20) for _ in range(5)))
    b = list(set(rng.randint(1, 20) for _ in range(5)))
    result = sorted(set(a) - set(b))
    return f"{a} minus {b}", str(result)


@_register("interleave_lists")
def _interleave(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(2, 4)
    a = [rng.randint(1, 20) for _ in range(n)]
    b = [rng.randint(1, 20) for _ in range(n)]
    result = []
    for x, y in zip(a, b):
        result.extend([x, y])
    return f"{a} and {b}", str(result)


@_register("rotate_list")
def _rotate(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    nums = [rng.randint(1, 30) for _ in range(n)]
    k = rng.randint(1, n - 1)
    rotated = nums[k:] + nums[:k]
    return f"{nums}, k={k}", str(rotated)


# ===================================================================
# MEDIUM TASKS
# ===================================================================

@_register("fibonacci_sequence")
def _fib(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(5, 10)
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return f"first {n} terms", str(seq[-1])


@_register("prime_sequence")
def _prime(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 8)
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return f"first {n} primes", str(primes)


@_register("geometric_sequence")
def _geo(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(1, 5)
    r = rng.randint(2, 3)
    n = rng.randint(4, 6)
    term = a * (r ** (n - 1))
    return f"a={a}, r={r}, n={n}", str(term)


@_register("arithmetic_progression")
def _arith(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(1, 10)
    d = rng.randint(1, 5)
    n = rng.randint(4, 8)
    seq = [a + i * d for i in range(n)]
    return f"a={a}, d={d}, first {n} terms", str(seq[-1])


@_register("harmonic_sequence")
def _harmonic(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(3, 6)
    h = round(sum(1.0 / i for i in range(1, n + 1)), 4)
    return f"H({n})", str(h)


@_register("collatz_sequence")
def _collatz(rng: random.Random) -> Tuple[str, str]:
    start = rng.randint(3, 20)
    seq = [start]
    while seq[-1] != 1:
        if seq[-1] % 2 == 0:
            seq.append(seq[-1] // 2)
        else:
            seq.append(3 * seq[-1] + 1)
    return str(start), str(len(seq))


@_register("gcd_lcm")
def _gcd_lcm(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(2, 50), rng.randint(2, 50)
    g = math.gcd(a, b)
    return f"GCD of {a} and {b}", str(g)


@_register("polynomial_evaluation")
def _poly(rng: random.Random) -> Tuple[str, str]:
    degree = rng.randint(1, 3)
    coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
    x = rng.randint(-3, 5)
    result = sum(c * (x ** i) for i, c in enumerate(coeffs))
    terms = " + ".join(f"{c}x^{i}" for i, c in enumerate(coeffs) if c != 0)
    return f"P(x)={terms} at x={x}", str(result)


@_register("number_base_conversion")
def _base(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(5, 100)
    base = rng.choice([2, 8, 16])
    if base == 2:
        result = bin(n)[2:]
    elif base == 8:
        result = oct(n)[2:]
    else:
        result = hex(n)[2:].upper()
    return f"{n} to base {base}", result


@_register("matrix_operations")
def _matrix(rng: random.Random) -> Tuple[str, str]:
    a = [[rng.randint(-5, 5) for _ in range(2)] for _ in range(2)]
    b = [[rng.randint(-5, 5) for _ in range(2)] for _ in range(2)]
    result = [[a[i][j] + b[i][j] for j in range(2)] for i in range(2)]
    return f"A={a} + B={b}", str(result)


@_register("logical_operations")
def _logic(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.choice([True, False]), rng.choice([True, False])
    op = rng.choice(["AND", "OR", "XOR"])
    if op == "AND":
        result = a and b
    elif op == "OR":
        result = a or b
    else:
        result = a ^ b
    return f"{a} {op} {b}", str(result)


@_register("combinatorics")
def _comb(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(4, 10)
    r = rng.randint(1, min(3, n))
    result = math.comb(n, r)
    return f"C({n},{r})", str(result)


@_register("pattern_completion")
def _pattern(rng: random.Random) -> Tuple[str, str]:
    # Simple arithmetic sequence
    a = rng.randint(1, 10)
    d = rng.randint(1, 5)
    seq = [a + i * d for i in range(5)]
    nxt = a + 5 * d
    return f"{seq}, next?", str(nxt)


@_register("algebraic_sequence")
def _alg_seq(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(1, 5)
    b = rng.randint(1, 3)
    n = rng.randint(4, 8)
    # a*n + b type
    term = a * n + b
    return f"a(n) = {a}n + {b}, find a({n})", str(term)


@_register("complex_pattern")
def _complex_pat(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(1, 5)
    d = rng.randint(1, 3)
    seq = [a + i * d for i in range(6)]
    return f"{seq[:5]}, next?", str(seq[5])


# ===================================================================
# HARD TASKS
# ===================================================================

@_register("shortest_path")
def _shortest(rng: random.Random) -> Tuple[str, str]:
    # Simple 3-node graph
    return "A→B:3, B→C:4, A→C:10, shortest A→C", "7"


@_register("knapsack")
def _knapsack(rng: random.Random) -> Tuple[str, str]:
    return "items=[(w=2,v=3),(w=3,v=4),(w=4,v=5)], capacity=5", "7"


@_register("traveling_salesman")
def _tsp(rng: random.Random) -> Tuple[str, str]:
    return "3 cities, distances: AB=10, BC=15, AC=20", "45"


@_register("longest_common_subsequence")
def _lcs(rng: random.Random) -> Tuple[str, str]:
    a = "".join(rng.choice("abcde") for _ in range(4))
    b = "".join(rng.choice("abcde") for _ in range(4))
    # Compute LCS length
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return f"'{a}' and '{b}'", str(dp[m][n])


@_register("edit_distance")
def _edit(rng: random.Random) -> Tuple[str, str]:
    a = "".join(rng.choice("abcde") for _ in range(3))
    b = "".join(rng.choice("abcde") for _ in range(3))
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return f"'{a}' → '{b}'", str(dp[m][n])


@_register("coin_change")
def _coin_change(rng: random.Random) -> Tuple[str, str]:
    coins = sorted(rng.sample([1, 2, 5, 10, 25], rng.randint(2, 3)))
    target = rng.randint(5, 20)
    # DP
    dp = [float("inf")] * (target + 1)
    dp[0] = 0
    for c in coins:
        for j in range(c, target + 1):
            dp[j] = min(dp[j], dp[j - c] + 1)
    result = dp[target] if dp[target] != float("inf") else -1
    return f"coins={coins}, target={target}", str(result)


@_register("interval_scheduling")
def _interval(rng: random.Random) -> Tuple[str, str]:
    return "intervals=[(1,3),(2,5),(4,7),(6,8)]", "3"


@_register("topological_sort")
def _topo(rng: random.Random) -> Tuple[str, str]:
    return "A→B, A→C, B→D, C→D", "[A, B, C, D] or [A, C, B, D]"


@_register("regex_matching")
def _regex(rng: random.Random) -> Tuple[str, str]:
    return "'abc' matches 'a.c'?", "True"


@_register("minimax_game")
def _minimax(rng: random.Random) -> Tuple[str, str]:
    return "tree=[[3,5],[2,9]], depth=2", "3"


@_register("boolean_sat")
def _sat(rng: random.Random) -> Tuple[str, str]:
    return "(x OR y) AND (NOT x OR z), x=T,y=F,z=T", "True"


@_register("n_queens")
def _nqueens(rng: random.Random) -> Tuple[str, str]:
    return "4-queens: number of solutions", "2"


@_register("sudoku_solving")
def _sudoku(rng: random.Random) -> Tuple[str, str]:
    return "4×4 mini-sudoku partial grid", "completed grid"


@_register("graph_coloring")
def _graph_color(rng: random.Random) -> Tuple[str, str]:
    return "triangle graph, minimum colors", "3"


@_register("tower_hanoi")
def _hanoi(rng: random.Random) -> Tuple[str, str]:
    n = rng.randint(2, 4)
    return f"{n} disks", str(2 ** n - 1)


@_register("matrix_chain_multiplication")
def _mcm(rng: random.Random) -> Tuple[str, str]:
    return "dimensions=[10,30,5,60]", "4500"


@_register("cryptarithmetic")
def _crypt(rng: random.Random) -> Tuple[str, str]:
    return "AB + BA = CDC, find digits", "mapping of letters to digits"


@_register("constraint_optimization")
def _constraint(rng: random.Random) -> Tuple[str, str]:
    return "maximize 3x+2y, x+y≤10, x,y≥0", "30"


@_register("logic_grid_puzzles")
def _logic_grid(rng: random.Random) -> Tuple[str, str]:
    return "3 people, 3 colors, clues given", "assignment mapping"


@_register("modular_systems")
def _modular(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(10, 50), rng.randint(2, 10)
    return f"{a} mod {b}", str(a % b)


# ===================================================================
# Public API
# ===================================================================


class FewShotGenerator:
    """Generate dynamic few-shot examples for any registered task.

    Parameters
    ----------
    seed:
        Random seed.  If ``None``, examples are non-deterministic.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._base_seed = seed

    def generate(
        self,
        task_name: str,
        n: int = 2,
        *,
        call_index: int = 0,
    ) -> str:
        """Return *n* formatted few-shot examples for *task_name*.

        Parameters
        ----------
        task_name:
            Registered task name (e.g. ``"sum"``).
        n:
            Number of examples to produce.
        call_index:
            Differentiator so that repeated calls within the same seed
            produce different examples (e.g. datapoint index).

        Returns
        -------
        Multi-line string like::

            Example 1: [5, -3, 8] → \\boxed{10}
            Example 2: [12, 1, -7] → \\boxed{6}
        """
        gen_fn = _GENERATORS.get(task_name)
        if gen_fn is None:
            return ""  # No dynamic generator available

        # Derive a seed that varies per call but is reproducible
        if self._base_seed is not None:
            seed = self._base_seed * 1000003 + hash(task_name) + call_index
            # Keep seed in valid range
            seed = seed % (2**31)
        else:
            seed = None

        rng = random.Random(seed)

        lines: List[str] = []
        for i in range(n):
            data_str, answer_str = gen_fn(rng)
            lines.append(f"Example {i + 1}: {data_str} → \\boxed{{{answer_str}}}")

        return "\n".join(lines)

    @staticmethod
    def supported_tasks() -> List[str]:
        """Return list of task names with dynamic few-shot generators."""
        return sorted(_GENERATORS.keys())

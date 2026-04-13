# BeyondBench Task Reference

Complete reference for all 84 tasks organized by suite. Each task generates problems dynamically so model answers cannot be memorized.

**Total tasks: 84** (44 easy + 15 medium + 20 hard + 5 extra medium variants)

> **Parser note**: All tasks use the unified parser by default (`--parser unified`). Answers are extracted from `\boxed{answer}` notation or structured output. Pass `--parser legacy` to revert to the original per-task parsers.

---

## Easy Suite (44 Tasks)

Easy tasks operate on lists of numbers or simple arithmetic. They test basic numerical reasoning and list processing.

**Common input format**: A list of integers in the range `[range_min, range_max]` (default -100 to 100) with a configurable list size.

**Common output format**: A single number or list, placed in `\boxed{answer}`.

---

### Core Arithmetic Tasks (14 tasks)

| Task Name | Description | Input | Output |
|-----------|-------------|-------|--------|
| `sum` | Compute the sum of all numbers in a list | List of integers | Integer sum |
| `subtraction` | Compute the result of a sequence of subtractions | List of integers | Integer |
| `multiplication` | Multiply all numbers together | List of integers | Integer product |
| `division` | Compute integer division result | Two integers | Integer |
| `mean` | Compute the arithmetic mean | List of numbers | Float (rounded) |
| `median` | Find the median value | List of numbers | Number |
| `mode` | Find the most frequently occurring value | List of numbers | Number |
| `comparison` | Compare two numbers and determine which is larger | Two integers | "first", "second", or "equal" |
| `find_maximum` | Find the maximum value in a list | List of numbers | Number |
| `find_minimum` | Find the minimum value in a list | List of numbers | Number |
| `absolute_difference` | Compute absolute difference between two numbers | Two integers | Non-negative integer |
| `odd_count` | Count the number of odd integers in a list | List of integers | Integer count |
| `even_count` | Count the number of even integers in a list | List of integers | Integer count |
| `sorting` | Sort a list in ascending order | List of integers | Sorted list |

---

### Scalable List Tasks (15 tasks)

These tasks scale with list size, controlled via `--list-sizes`.

| Task Name | Description | Input | Output |
|-----------|-------------|-------|--------|
| `second_maximum` | Find the second largest value | List of numbers | Number |
| `second_minimum` | Find the second smallest value | List of numbers | Number |
| `range` | Compute the range (max minus min) | List of numbers | Non-negative number |
| `index_of_maximum` | Find the 0-based index of the maximum value | List of numbers | Integer index |
| `count_negative` | Count how many values are negative | List of numbers | Integer count |
| `count_unique` | Count the number of distinct values | List of numbers | Integer count |
| `max_adjacent_difference` | Find the largest absolute difference between adjacent elements | List of numbers | Number |
| `count_greater_than_previous` | Count elements larger than their predecessor | List of numbers | Integer count |
| `sum_of_max_indices` | Sum the indices of all maximum-value occurrences | List of numbers | Integer |
| `count_palindromic` | Count integers that are palindromes (same forwards and backwards) | List of integers | Integer count |
| `longest_increasing_subsequence` | Find the length of the longest strictly increasing subsequence | List of integers | Integer |
| `sum_of_digits` | Sum all digits of all numbers in the list | List of integers | Integer |
| `count_perfect_squares` | Count numbers that are perfect squares | List of non-negative integers | Integer count |
| `alternating_sum` | Compute sum with alternating +/- signs (first element positive) | List of numbers | Number |
| `count_multiples` | Count multiples of a given divisor | List of integers + divisor | Integer count |
| `local_maxima_count` | Count elements greater than both neighbors | List of numbers | Integer count |

---

### Phase 12 Tasks (15 tasks)

Additional list-processing and statistical tasks added in Phase 12.

| Task Name | Description | Input | Output |
|-----------|-------------|-------|--------|
| `weighted_sum` | Compute weighted sum using position-based weights | List of numbers | Number |
| `running_average` | Compute the running (cumulative) average at each position | List of numbers | List of floats |
| `parity_check` | Determine if the sum of the list is even or odd | List of integers | "even" or "odd" |
| `cumulative_sum` | Return the cumulative sum at each position | List of numbers | List of numbers |
| `reverse_list` | Reverse the order of elements | List | Reversed list |
| `rotate_list` | Rotate the list by k positions (right rotation) | List + k | Rotated list |
| `interleave_lists` | Interleave two equal-length lists element by element | Two lists | Interleaved list |
| `set_intersection` | Find elements common to both lists | Two lists | List of common elements |
| `set_difference` | Find elements in the first list but not the second | Two lists | List of elements |
| `moving_average` | Compute the moving average with a sliding window | List of numbers + window size | List of floats |
| `element_frequency` | Return a frequency count of each distinct element | List | Dict/list of (element, count) |
| `variance` | Compute the population variance of the list | List of numbers | Float |
| `standard_deviation` | Compute the population standard deviation | List of numbers | Float |
| `dot_product` | Compute the dot product of two equal-length vectors | Two numeric lists | Number |

---

## Medium Suite (15 Tasks)

Medium tasks require multi-step mathematical reasoning: sequence prediction, number theory, pattern recognition, and algebraic computation.

**Common format**: A partial sequence or expression is given; the model must identify the rule and compute the next term or evaluate the expression.

---

### Original Medium Tasks (5 tasks)

| Task Name | Description | What It Tests | Output |
|-----------|-------------|---------------|--------|
| `fibonacci_sequence` | Identify and complete Fibonacci-like recursive sequences (classic, Lucas, Tribonacci, modified recursive, alternating) | Pattern recognition, recursive reasoning | Next k terms |
| `algebraic_sequence` | Complete sequences defined by algebraic relationships and multi-step expressions | Symbolic reasoning, algebraic manipulation | Next term |
| `geometric_sequence` | Complete geometric progressions and exponential sequences | Multiplicative pattern recognition | Next term or value |
| `prime_sequence` | Complete sequences involving prime numbers and number-theoretic functions | Number theory knowledge, prime arithmetic | Next element |
| `complex_pattern` | Complete sequences with multi-layered, combined mathematical patterns | Deep pattern recognition | Next term |

---

### Phase 13 Tasks (10 tasks)

| Task Name | Description | What It Tests | Output |
|-----------|-------------|---------------|--------|
| `arithmetic_progression` | Generate arithmetic progression with random first term and common difference; predict the next term | Linear sequence reasoning | Next term |
| `harmonic_sequence` | Generate reciprocal sequences 1/a, 1/(a+d), ...; predict the next fraction | Fraction arithmetic | Fraction or decimal |
| `collatz_sequence` | Apply Collatz rule (even→n/2, odd→3n+1); given partial sequence, compute next term | Conditional rule application | Next integer |
| `polynomial_evaluation` | Evaluate a polynomial p(x) with random integer coefficients (degree 2–4) at a given x | Polynomial arithmetic | Integer |
| `matrix_operations` | Perform 2×2 matrix multiplication, compute determinant, or compute inverse | Linear algebra | Matrix or scalar |
| `number_base_conversion` | Convert between decimal, binary, and hexadecimal | Base arithmetic, bit manipulation | Converted value |
| `logical_operations` | Evaluate boolean expressions (AND/OR/NOT/XOR) over 3–5 variables | Boolean algebra | True/False |
| `pattern_completion` | Complete well-known numeric patterns: squares, cubes, triangular numbers, powers of 2, factorials, Pascal's triangle diagonals | Mathematical pattern knowledge | Next term |
| `gcd_lcm` | Compute GCD or LCM of two or three integers | Number theory, Euclidean algorithm | Integer |
| `combinatorics` | Compute permutations P(n,r) or combinations C(n,r) for 1 ≤ r ≤ n ≤ 12 | Combinatorial reasoning | Integer |

---

## Hard Suite (20 Tasks)

Hard tasks involve combinatorial optimization, constraint satisfaction, dynamic programming, and graph problems. These require multi-step algorithmic reasoning.

**Output format**: Varies by task — integers, lists, truth assignments, or grid solutions. All placed in `\boxed{answer}` or provided in a structured format.

---

### Original Hard Tasks (10 tasks)

| Task Name | Description | Algorithm Required | Input → Output |
|-----------|-------------|-------------------|----------------|
| `tower_hanoi` | Solve Tower of Hanoi: move all disks from source peg to destination peg following the rules | Recursive divide-and-conquer | n disks → ordered move sequence |
| `n_queens` | Place N queens on an N×N chessboard with no two queens attacking each other | Backtracking / constraint satisfaction | N → valid queen placement |
| `graph_coloring` | Color graph vertices so no two adjacent vertices share a color, minimizing colors used | Chromatic number reasoning | Graph adjacency → color assignment |
| `boolean_sat` | Find a truth assignment satisfying a Boolean formula in CNF (includes Horn, XOR, random-k-SAT variants) | SAT reasoning / DPLL-style | CNF formula → variable assignments |
| `sudoku_solving` | Solve partially filled 4×4, 6×6, or 9×9 Sudoku grids | Constraint propagation + backtracking | Partial grid → completed grid |
| `cryptarithmetic` | Find digit assignments to letters so that an arithmetic equation holds (e.g., SEND + MORE = MONEY) | Constraint satisfaction | Equation → letter-to-digit mapping |
| `matrix_chain_multiplication` | Find optimal parenthesization of a matrix chain to minimize scalar multiplications | Dynamic programming | Chain of matrix dimensions → min cost |
| `modular_systems` | Solve systems of modular arithmetic equations using the Chinese Remainder Theorem | CRT, Extended Euclidean | System of congruences → solution x |
| `constraint_optimization` | Solve multi-constraint resource allocation problems (integer linear programming style) | ILP / multi-objective optimization | Resource constraints → allocation |
| `logic_grid_puzzles` | Solve logic grid / Einstein riddle puzzles from clue sets | Logical deduction, constraint propagation | Clue set → entity–attribute assignments |

---

### Phase 14 Tasks (10 tasks)

| Task Name | Description | Algorithm Required | Input → Output |
|-----------|-------------|-------------------|----------------|
| `shortest_path` | Find the minimum cost path in a weighted directed graph from source to target | Dijkstra's algorithm | Weighted graph + source/target → integer cost |
| `knapsack` | Solve the 0/1 knapsack problem: maximize total value without exceeding capacity | Dynamic programming | Items (weight, value) + capacity → max value |
| `traveling_salesman` | Find the minimum-cost tour visiting all cities and returning to start (4–6 cities) | Brute-force permutation over small instances | Distance matrix → minimum tour cost |
| `longest_common_subsequence` | Compute the length of the longest common subsequence of two strings | Dynamic programming | Two strings → LCS length (integer) |
| `minimax_game` | Compute the optimal value for the maximizing player in a binary game tree | Minimax algorithm | Nested game tree + depth → optimal integer |
| `regex_matching` | Determine whether a string fully matches a given regular expression pattern | Regular expression reasoning | Pattern + string → True/False |
| `topological_sort` | Produce a valid topological ordering of a directed acyclic graph (DAG) | Kahn's algorithm / DFS | DAG edges → valid node ordering |
| `interval_scheduling` | Find the maximum number of non-overlapping intervals | Greedy (sort by end time) | Intervals (start, end) → max count |
| `coin_change` | Find the minimum number of coins to make a target amount (denomination 1 always included) | Dynamic programming | Coin denominations + target → min coins |
| `edit_distance` | Compute the Levenshtein edit distance (insertions, deletions, substitutions) between two strings | Dynamic programming | Two strings → integer distance |

---

## Task Configuration Tips

### List Sizes

Easy and medium tasks support variable list lengths. The default is determined by `--list-sizes`. For publication results, use multiple sizes:

```bash
beyondbench evaluate --model-id ... --suite easy --list-sizes "4,8,16,32,64"
```

### Hard Task Complexity

Hard tasks adjust complexity via their own internal parameters:

- **sudoku_solving**: tests 4×4, 6×6, and 9×9 grids
- **n_queens**: board sizes 4–8
- **tower_hanoi**: disk counts typically 3–6
- **boolean_sat**: formula size and clause types vary
- **traveling_salesman**: 4–6 cities (kept small for feasibility)

### Checking Available Tasks

```bash
beyondbench list-tasks --suite easy
beyondbench list-tasks --suite hard
beyondbench list-tasks  # all suites
```

### Running a Single Task

```bash
beyondbench evaluate \
  --model-id gpt-4o \
  --api-provider openai \
  --tasks knapsack \
  --datapoints 50
```

### Running Multiple Specific Tasks

```bash
beyondbench evaluate \
  --model-id gpt-4o \
  --api-provider openai \
  --tasks sum --tasks mean --tasks sorting \
  --datapoints 100
```

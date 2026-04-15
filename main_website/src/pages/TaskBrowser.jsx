import { useState, useMemo } from 'react'
import {
  Search, ChevronDown, ChevronUp, BookOpen, Layers, Cpu,
  Brain, Zap, FlaskConical, Filter, X,
} from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

// ── Task Data ────────────────────────────────────────────────────────────────

const TASK_DATA = {
  easy: {
    label: 'Easy Suite',
    description: '44 fundamental reasoning tasks covering arithmetic, statistics, counting, extrema, sequences, list operations, and set operations.',
    icon: Zap,
    color: 'bb-teal',
    accentDark: '#0ea5e9',
    accentLight: '#0284c7',
    tasks: [
      { name: 'sum', category: 'Arithmetic', variations: 1, description: 'Compute the sum of a list of numbers.', exampleInput: '[3, 7, 12, -2, 5]', exampleOutput: '25' },
      { name: 'multiplication', category: 'Arithmetic', variations: 1, description: 'Compute the product of all elements in a list.', exampleInput: '[2, 3, 4]', exampleOutput: '24' },
      { name: 'subtraction', category: 'Arithmetic', variations: 1, description: 'Subtract subsequent elements from the first.', exampleInput: '[10, 3, 2]', exampleOutput: '5' },
      { name: 'division', category: 'Arithmetic', variations: 1, description: 'Sequentially divide elements starting from the first.', exampleInput: '[120, 4, 5]', exampleOutput: '6' },
      { name: 'absolute_difference', category: 'Arithmetic', variations: 1, description: 'Find the absolute difference between the maximum and minimum values.', exampleInput: '[4, -1, 7, 2]', exampleOutput: '8' },
      { name: 'mean', category: 'Statistics', variations: 1, description: 'Compute the arithmetic mean of a list.', exampleInput: '[10, 20, 30]', exampleOutput: '20.0' },
      { name: 'median', category: 'Statistics', variations: 1, description: 'Find the median value of a list.', exampleInput: '[3, 1, 4, 1, 5]', exampleOutput: '3' },
      { name: 'mode', category: 'Statistics', variations: 1, description: 'Find the most frequently occurring element.', exampleInput: '[1, 2, 2, 3, 3, 3]', exampleOutput: '3' },
      { name: 'odd_count', category: 'Counting', variations: 1, description: 'Count how many odd numbers appear in the list.', exampleInput: '[1, 2, 3, 4, 5]', exampleOutput: '3' },
      { name: 'even_count', category: 'Counting', variations: 1, description: 'Count how many even numbers appear in the list.', exampleInput: '[1, 2, 3, 4, 5]', exampleOutput: '2' },
      { name: 'count_negative', category: 'Counting', variations: 1, description: 'Count the number of negative values in the list.', exampleInput: '[1, -2, 3, -4, 0]', exampleOutput: '2' },
      { name: 'count_unique', category: 'Counting', variations: 1, description: 'Count the number of distinct values.', exampleInput: '[1, 2, 2, 3, 1]', exampleOutput: '3' },
      { name: 'count_greater_than_previous', category: 'Counting', variations: 1, description: 'Count elements strictly greater than their predecessor.', exampleInput: '[1, 3, 2, 4, 4]', exampleOutput: '2' },
      { name: 'count_palindromic', category: 'Counting', variations: 1, description: 'Count numbers that read the same forwards and backwards.', exampleInput: '[121, 13, 131, 22]', exampleOutput: '3' },
      { name: 'count_perfect_squares', category: 'Counting', variations: 1, description: 'Count integers that are perfect squares.', exampleInput: '[1, 2, 4, 9, 10]', exampleOutput: '3' },
      { name: 'count_multiples', category: 'Counting', variations: 1, description: 'Count elements that are multiples of a given number.', exampleInput: '[3, 6, 7, 9, 12] (multiples of 3)', exampleOutput: '4' },
      { name: 'local_maxima_count', category: 'Counting', variations: 1, description: 'Count elements greater than both immediate neighbors.', exampleInput: '[1, 3, 2, 4, 1]', exampleOutput: '2' },
      { name: 'find_maximum', category: 'Extrema', variations: 1, description: 'Find the largest element in the list.', exampleInput: '[5, 2, 8, 1, 9]', exampleOutput: '9' },
      { name: 'find_minimum', category: 'Extrema', variations: 1, description: 'Find the smallest element in the list.', exampleInput: '[5, 2, 8, 1, 9]', exampleOutput: '1' },
      { name: 'second_maximum', category: 'Extrema', variations: 1, description: 'Find the second largest distinct value.', exampleInput: '[5, 2, 8, 1, 9]', exampleOutput: '8' },
      { name: 'range', category: 'Extrema', variations: 1, description: 'Compute max minus min.', exampleInput: '[3, 7, 1, 9]', exampleOutput: '8' },
      { name: 'index_of_maximum', category: 'Extrema', variations: 1, description: 'Return the zero-based index of the maximum element.', exampleInput: '[2, 7, 4, 7, 1]', exampleOutput: '1' },
      { name: 'max_adjacent_difference', category: 'Extrema', variations: 1, description: 'Find the largest absolute difference between adjacent elements.', exampleInput: '[1, 5, 2, 9, 3]', exampleOutput: '7' },
      { name: 'sum_of_max_indices', category: 'Extrema', variations: 1, description: 'Sum the indices of all occurrences of the maximum value.', exampleInput: '[2, 5, 3, 5, 1]', exampleOutput: '4' },
      { name: 'sorting', category: 'Sequences', variations: 1, description: 'Return the list in ascending sorted order.', exampleInput: '[3, 1, 4, 1, 5]', exampleOutput: '[1, 1, 3, 4, 5]' },
      { name: 'longest_increasing_subsequence', category: 'Sequences', variations: 1, description: 'Find the length of the longest strictly increasing subsequence.', exampleInput: '[10, 9, 2, 5, 3, 7, 101]', exampleOutput: '4' },
      { name: 'alternating_sum', category: 'Sequences', variations: 1, description: 'Compute a[0] - a[1] + a[2] - … for the list.', exampleInput: '[3, 1, 4, 1, 5]', exampleOutput: '10' },
      { name: 'sum_of_digits', category: 'Sequences', variations: 1, description: 'Sum all digits of all numbers in the list.', exampleInput: '[12, 34, 5]', exampleOutput: '15' },
      { name: 'comparison', category: 'Comparison', variations: 1, description: 'Compare two values and return >, <, or =.', exampleInput: '17 vs 23', exampleOutput: '<' },
      // Phase 12: 15 new easy tasks
      { name: 'weighted_sum', category: 'Arithmetic', variations: 1, description: 'Sum elements multiplied by their 1-indexed position.', exampleInput: '[3, 1, 4]', exampleOutput: '3*1 + 1*2 + 4*3 = 17' },
      { name: 'running_average', category: 'Statistics', variations: 1, description: 'Compute the running average at each position.', exampleInput: '[4, 2, 6]', exampleOutput: '[4.0, 3.0, 4.0]' },
      { name: 'parity_check', category: 'Arithmetic', variations: 1, description: 'Determine if the product of all elements is even or odd.', exampleInput: '[3, 5, 7]', exampleOutput: 'odd' },
      { name: 'cumulative_sum', category: 'Sequences', variations: 1, description: 'Compute the cumulative sum list.', exampleInput: '[1, 2, 3, 4]', exampleOutput: '[1, 3, 6, 10]' },
      { name: 'reverse_list', category: 'List Operations', variations: 1, description: 'Reverse the given list.', exampleInput: '[1, 2, 3, 4, 5]', exampleOutput: '[5, 4, 3, 2, 1]' },
      { name: 'rotate_list', category: 'List Operations', variations: 1, description: 'Rotate list by k positions.', exampleInput: '[1, 2, 3, 4, 5], k=2', exampleOutput: '[4, 5, 1, 2, 3]' },
      { name: 'interleave_lists', category: 'List Operations', variations: 1, description: 'Interleave two lists element by element.', exampleInput: '[1, 3, 5] and [2, 4, 6]', exampleOutput: '[1, 2, 3, 4, 5, 6]' },
      { name: 'set_intersection', category: 'Set Operations', variations: 1, description: 'Find common elements between two lists.', exampleInput: '[1, 2, 3, 4] and [3, 4, 5, 6]', exampleOutput: '[3, 4]' },
      { name: 'set_difference', category: 'Set Operations', variations: 1, description: 'Find elements in first list not in second.', exampleInput: '[1, 2, 3, 4] and [3, 4, 5]', exampleOutput: '[1, 2]' },
      { name: 'moving_average', category: 'Statistics', variations: 1, description: 'Compute k-window moving average.', exampleInput: '[1, 2, 3, 4, 5], k=3', exampleOutput: '[2.0, 3.0, 4.0]' },
      { name: 'element_frequency', category: 'Counting', variations: 1, description: 'Count frequency of each element.', exampleInput: '[1, 2, 2, 3, 3, 3]', exampleOutput: '{1: 1, 2: 2, 3: 3}' },
      { name: 'second_minimum', category: 'Extrema', variations: 1, description: 'Find the second smallest distinct value.', exampleInput: '[5, 2, 8, 1, 9]', exampleOutput: '2' },
      { name: 'variance', category: 'Statistics', variations: 1, description: 'Compute the variance of a list.', exampleInput: '[2, 4, 4, 4, 5, 5, 7, 9]', exampleOutput: '4.0' },
      { name: 'standard_deviation', category: 'Statistics', variations: 1, description: 'Compute the standard deviation of a list.', exampleInput: '[2, 4, 4, 4, 5, 5, 7, 9]', exampleOutput: '2.0' },
      { name: 'dot_product', category: 'Arithmetic', variations: 1, description: 'Compute the dot product of two lists.', exampleInput: '[1, 2, 3] and [4, 5, 6]', exampleOutput: '32' },
    ],
  },
  medium: {
    label: 'Medium Suite',
    description: '15 tasks covering sequence extrapolation, pattern recognition, number theory, matrix operations, and combinatorics.',
    icon: Brain,
    color: 'bb-purple',
    accentDark: '#a855f7',
    accentLight: '#9333ea',
    tasks: [
      {
        name: 'fibonacci_sequence',
        category: 'Sequence Extrapolation',
        variations: 6,
        description: 'Given a prefix of a Fibonacci-family sequence, predict the next K terms. Includes Tribonacci, Lucas numbers, and modified recursive variants.',
        exampleInput: '1, 1, 2, 3, 5, 8, 13, 21 → next 3',
        exampleOutput: '34, 55, 89',
      },
      {
        name: 'algebraic_sequence',
        category: 'Sequence Extrapolation',
        variations: 10,
        description: 'Identify and continue polynomial, arithmetic, and quadratic sequences. Requires inferring hidden algebraic rules.',
        exampleInput: '2, 5, 10, 17, 26, 37 → next 2',
        exampleOutput: '50, 65',
      },
      {
        name: 'geometric_sequence',
        category: 'Sequence Extrapolation',
        variations: 10,
        description: 'Continue exponential growth, compound growth, or factorial-based sequences.',
        exampleInput: '3, 6, 12, 24, 48 → next 3',
        exampleOutput: '96, 192, 384',
      },
      {
        name: 'prime_sequence',
        category: 'Number Theory',
        variations: 11,
        description: 'Work with prime gaps, twin prime pairs, Sophie Germain primes, and other prime-derived sequences.',
        exampleInput: 'Primes: 2, 3, 5, 7, 11 → gaps between consecutive',
        exampleOutput: '1, 2, 2, 4',
      },
      {
        name: 'complex_pattern',
        category: 'Pattern Recognition',
        variations: 12,
        description: 'Multi-rule patterns including interleaved sequences, conditional logic, and sequences governed by multiple simultaneous rules.',
        exampleInput: '1, 4, 2, 8, 3, 12, 4, 16 → next 4',
        exampleOutput: '5, 20, 6, 24',
      },
      // Phase 13: 10 new medium tasks
      {
        name: 'arithmetic_progression',
        category: 'Sequence Extrapolation',
        variations: 1,
        description: 'Predict the next term in an arithmetic progression with varying common differences.',
        exampleInput: '3, 7, 11, 15, 19 → next',
        exampleOutput: '23',
      },
      {
        name: 'harmonic_sequence',
        category: 'Sequence Extrapolation',
        variations: 1,
        description: 'Work with reciprocal sequences like 1/1, 1/2, 1/3, etc.',
        exampleInput: '1, 1/2, 1/3, 1/4 → next 2',
        exampleOutput: '1/5, 1/6',
      },
      {
        name: 'collatz_sequence',
        category: 'Number Theory',
        variations: 1,
        description: 'Predict the next terms in a Collatz sequence (if even: n/2, if odd: 3n+1).',
        exampleInput: 'Start: 6 → next 5 terms',
        exampleOutput: '3, 10, 5, 16, 8',
      },
      {
        name: 'polynomial_evaluation',
        category: 'Algebra',
        variations: 1,
        description: 'Evaluate a polynomial at a given point.',
        exampleInput: 'f(x) = 2x² + 3x - 5, x = 4',
        exampleOutput: '39',
      },
      {
        name: 'matrix_operations',
        category: 'Linear Algebra',
        variations: 1,
        description: '2x2 matrix multiplication, determinant, or inverse computation.',
        exampleInput: 'det([[3, 1], [2, 4]])',
        exampleOutput: '10',
      },
      {
        name: 'number_base_conversion',
        category: 'Number Theory',
        variations: 1,
        description: 'Convert between decimal, binary, and hexadecimal representations.',
        exampleInput: '255 → binary',
        exampleOutput: '11111111',
      },
      {
        name: 'logical_operations',
        category: 'Logic',
        variations: 1,
        description: 'Evaluate boolean expressions with AND, OR, NOT, and XOR operators.',
        exampleInput: '(True AND False) OR (NOT False)',
        exampleOutput: 'True',
      },
      {
        name: 'pattern_completion',
        category: 'Pattern Recognition',
        variations: 1,
        description: 'Complete visual or numeric patterns by inferring the underlying rule.',
        exampleInput: '2, 6, 12, 20, ? → next',
        exampleOutput: '30',
      },
      {
        name: 'gcd_lcm',
        category: 'Number Theory',
        variations: 1,
        description: 'Compute the Greatest Common Divisor and Least Common Multiple of number pairs.',
        exampleInput: 'GCD(12, 18)',
        exampleOutput: '6',
      },
      {
        name: 'combinatorics',
        category: 'Combinatorics',
        variations: 1,
        description: 'Compute permutations and combinations (nPr, nCr).',
        exampleInput: 'C(10, 3)',
        exampleOutput: '120',
      },
    ],
  },
  hard: {
    label: 'Hard Suite',
    description: '20 tasks covering NP-complete problems, combinatorial search, constraint satisfaction, dynamic programming, and algorithm design.',
    icon: FlaskConical,
    color: 'bb-accent',
    accentDark: '#00e676',
    accentLight: '#00a852',
    tasks: [
      {
        name: 'tower_hanoi',
        category: 'Combinatorial Search',
        variations: 6,
        description: 'Solve the Tower of Hanoi puzzle and output the sequence of moves. Variations include different peg configurations and move count minimization.',
        exampleInput: '3 disks: A→C',
        exampleOutput: 'Move disk 1 A→C, disk 2 A→B, disk 1 C→B, disk 3 A→C, disk 1 B→A, disk 2 B→C, disk 1 A→C',
      },
      {
        name: 'n_queens',
        category: 'Constraint Satisfaction',
        variations: 4,
        description: 'Place N queens on an N×N board so no two attack each other. Return a valid board configuration.',
        exampleInput: 'N=4',
        exampleOutput: '[2, 4, 1, 3]',
      },
      {
        name: 'graph_coloring',
        category: 'NP-Complete',
        variations: 10,
        description: 'Color graph vertices with minimum colors such that no adjacent vertices share a color. Variations include planar graphs, random sparse/dense graphs.',
        exampleInput: 'Triangle graph: 3 vertices, 3 edges',
        exampleOutput: 'Chromatic number: 3',
      },
      {
        name: 'boolean_sat',
        category: 'NP-Complete',
        variations: 5,
        description: 'Determine satisfiability of propositional logic formulas in CNF. Find a satisfying assignment or prove unsatisfiable.',
        exampleInput: '(A ∨ B) ∧ (¬A ∨ C) ∧ (¬B ∨ ¬C)',
        exampleOutput: 'A=false, B=true, C=true',
      },
      {
        name: 'sudoku',
        category: 'Constraint Satisfaction',
        variations: 8,
        description: 'Solve 9×9 Sudoku puzzles of varying difficulty. Variations include different given-cell counts and specialized puzzle structures.',
        exampleInput: '9×9 grid (17 given cells)',
        exampleOutput: 'Completed 9×9 grid',
      },
      {
        name: 'cryptarithmetic',
        category: 'Constraint Satisfaction',
        variations: 12,
        description: 'Solve letter-digit puzzles like SEND+MORE=MONEY. Each letter maps to a unique digit 0-9.',
        exampleInput: 'SEND + MORE = MONEY',
        exampleOutput: 'S=9, E=5, N=6, D=7, M=1, O=0, R=8, Y=2',
      },
      {
        name: 'matrix_chain_multiplication',
        category: 'Dynamic Programming',
        variations: 5,
        description: 'Find the optimal parenthesization of a matrix chain product to minimize scalar multiplications.',
        exampleInput: 'Dimensions: [10, 30, 5, 60]',
        exampleOutput: 'Min multiplications: 4500; Order: (A(BC))',
      },
      {
        name: 'modular_systems_solver',
        category: 'Number Theory',
        variations: 5,
        description: 'Solve systems of linear congruences using the Chinese Remainder Theorem and related number theory.',
        exampleInput: 'x ≡ 2 (mod 3), x ≡ 3 (mod 5)',
        exampleOutput: 'x ≡ 8 (mod 15)',
      },
      {
        name: 'constraint_optimization',
        category: 'Operations Research',
        variations: 5,
        description: 'Solve linear and combinatorial optimization problems: scheduling, bin packing, knapsack, assignment problems.',
        exampleInput: 'Knapsack: capacity=10, items=[(w=6,v=6),(w=4,v=5),(w=3,v=4)]',
        exampleOutput: 'Max value: 10 (items 2+3)',
      },
      {
        name: 'logic_grid_puzzles',
        category: 'Deductive Reasoning',
        variations: 8,
        description: 'Solve logic grid puzzles: given clues about attributes of entities, determine the full attribute assignment.',
        exampleInput: '3 people, 3 attributes, 5 clues',
        exampleOutput: 'Alice: red, dog; Bob: blue, cat; Carol: green, fish',
      },
      // Phase 14: 9 new hard tasks
      {
        name: 'shortest_path',
        category: 'Graph Algorithms',
        variations: 1,
        description: 'Find the shortest path in a weighted graph using Dijkstra\'s algorithm.',
        exampleInput: 'Graph: A→B(4), A→C(2), C→B(1); start=A, end=B',
        exampleOutput: 'Path: A→C→B, cost: 3',
      },
      {
        name: 'knapsack',
        category: 'Dynamic Programming',
        variations: 1,
        description: 'Solve the 0/1 knapsack problem: maximize value within weight capacity.',
        exampleInput: 'Capacity=10, items=[(w=6,v=6),(w=4,v=5),(w=3,v=4)]',
        exampleOutput: 'Max value: 9 (items 2+3)',
      },
      {
        name: 'traveling_salesman',
        category: 'NP-Complete',
        variations: 1,
        description: 'Find the shortest tour visiting all cities exactly once (small instances of 5-8 cities).',
        exampleInput: '4 cities with distance matrix',
        exampleOutput: 'Tour: A→B→D→C→A, distance: 21',
      },
      {
        name: 'longest_common_subsequence',
        category: 'Dynamic Programming',
        variations: 1,
        description: 'Find the longest common subsequence of two strings.',
        exampleInput: '"ABCBDAB" and "BDCAB"',
        exampleOutput: 'LCS: "BCAB", length: 4',
      },
      {
        name: 'minimax_game',
        category: 'Game Theory',
        variations: 1,
        description: 'Determine the optimal move in a simple game tree using minimax algorithm.',
        exampleInput: 'Game tree with depth 3, branching factor 2',
        exampleOutput: 'Optimal value: 5',
      },
      {
        name: 'regex_matching',
        category: 'String Algorithms',
        variations: 1,
        description: 'Determine if a string matches a given regular expression pattern.',
        exampleInput: 'Pattern: "a*b.c", String: "aabxc"',
        exampleOutput: 'True',
      },
      {
        name: 'topological_sort',
        category: 'Graph Algorithms',
        variations: 1,
        description: 'Find a valid topological ordering of a directed acyclic graph.',
        exampleInput: 'DAG: A→B, A→C, B→D, C→D',
        exampleOutput: '[A, B, C, D] or [A, C, B, D]',
      },
      {
        name: 'interval_scheduling',
        category: 'Greedy Algorithms',
        variations: 1,
        description: 'Find the maximum number of non-overlapping intervals.',
        exampleInput: '[(1,4), (2,3), (3,5), (0,6), (5,7)]',
        exampleOutput: '3 intervals: (2,3), (3,5), (5,7)',
      },
      {
        name: 'coin_change',
        category: 'Dynamic Programming',
        variations: 1,
        description: 'Find the minimum number of coins needed to make a target amount.',
        exampleInput: 'Coins: [1, 5, 10, 25], Target: 36',
        exampleOutput: '3 coins (25+10+1)',
      },
      {
        name: 'edit_distance',
        category: 'Dynamic Programming',
        variations: 1,
        description: 'Compute the Levenshtein edit distance between two strings.',
        exampleInput: '"kitten" and "sitting"',
        exampleOutput: '3',
      },
    ],
  },
}

const CATEGORIES = {
  easy: ['All', 'Arithmetic', 'Statistics', 'Counting', 'Extrema', 'Sequences', 'Comparison', 'List Operations', 'Set Operations'],
  medium: ['All', 'Sequence Extrapolation', 'Number Theory', 'Pattern Recognition', 'Algebra', 'Linear Algebra', 'Logic', 'Combinatorics'],
  hard: ['All', 'Combinatorial Search', 'Constraint Satisfaction', 'NP-Complete', 'Dynamic Programming', 'Number Theory', 'Operations Research', 'Deductive Reasoning', 'Graph Algorithms', 'Game Theory', 'String Algorithms', 'Greedy Algorithms'],
}

// ── TaskCard ─────────────────────────────────────────────────────────────────

function TaskCard({ task, accentDark, accentLight, isDark }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`rounded-xl border transition-all duration-200 overflow-hidden ${
        isDark
          ? 'bg-bb-dark-300/50 border-bb-dark-50/20 hover:border-bb-dark-50/40'
          : 'bg-white/60 border-bb-light-300/60 hover:border-bb-light-300 shadow-sm'
      }`}
    >
      <button
        className="w-full text-left p-4 flex items-start justify-between gap-3"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex items-center gap-3 min-w-0">
          <div
            className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-xs font-mono font-bold"
            style={{
              backgroundColor: (isDark ? accentDark : accentLight) + '18',
              color: isDark ? accentDark : accentLight,
            }}
          >
            {task.variations}
          </div>
          <div className="min-w-0">
            <span className={`font-mono font-semibold text-sm ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {task.name}
            </span>
            <span
              className="ml-2 text-[10px] font-medium px-1.5 py-0.5 rounded-full"
              style={{
                backgroundColor: (isDark ? accentDark : accentLight) + '18',
                color: isDark ? accentDark : accentLight,
              }}
            >
              {task.category}
            </span>
          </div>
        </div>
        <div className="flex-shrink-0 mt-0.5">
          {expanded
            ? <ChevronUp className={`w-4 h-4 ${isDark ? 'text-gray-400' : 'text-gray-500'}`} />
            : <ChevronDown className={`w-4 h-4 ${isDark ? 'text-gray-400' : 'text-gray-500'}`} />
          }
        </div>
      </button>

      {expanded && (
        <div className={`px-4 pb-4 border-t ${isDark ? 'border-bb-dark-50/15' : 'border-bb-light-300/40'}`}>
          <p className={`text-sm mt-3 leading-relaxed ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
            {task.description}
          </p>
          <div className={`mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3`}>
            <div className={`rounded-lg p-3 ${isDark ? 'bg-bb-dark-500/50' : 'bg-bb-light-100/80'}`}>
              <div className={`text-[10px] font-semibold mb-1.5 uppercase tracking-wide ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                Example Input
              </div>
              <code className={`text-xs font-mono ${isDark ? 'text-bb-teal' : 'text-blue-600'}`}>
                {task.exampleInput}
              </code>
            </div>
            <div className={`rounded-lg p-3 ${isDark ? 'bg-bb-dark-500/50' : 'bg-bb-light-100/80'}`}>
              <div className={`text-[10px] font-semibold mb-1.5 uppercase tracking-wide ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                Expected Output
              </div>
              <code
                className="text-xs font-mono font-bold"
                style={{ color: isDark ? accentDark : accentLight }}
              >
                {task.exampleOutput}
              </code>
            </div>
          </div>
          <div className={`mt-2 flex items-center gap-1.5 text-[11px] ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
            <Layers className="w-3 h-3" />
            {task.variations === 1 ? '1 variation' : `${task.variations} variations`}
          </div>
        </div>
      )}
    </div>
  )
}

// ── SuiteSection ─────────────────────────────────────────────────────────────

function SuiteSection({ suiteKey, suite }) {
  const { isDark } = useTheme()
  const [open, setOpen] = useState(true)
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('All')

  const Icon = suite.icon
  const accent = isDark ? suite.accentDark : suite.accentLight
  const categories = CATEGORIES[suiteKey]

  const filtered = useMemo(() => {
    let tasks = suite.tasks
    if (category !== 'All') tasks = tasks.filter(t => t.category === category)
    if (search.trim()) {
      const q = search.toLowerCase()
      tasks = tasks.filter(t => t.name.includes(q) || t.category.toLowerCase().includes(q) || t.description.toLowerCase().includes(q))
    }
    return tasks
  }, [search, category, suite.tasks])

  return (
    <div className={`rounded-2xl border overflow-hidden mb-6 ${
      isDark
        ? 'bg-bb-dark-300/40 border-bb-dark-50/25'
        : 'bg-white/60 border-bb-light-300/60 shadow-sm'
    }`}>
      {/* Suite header */}
      <button
        className={`w-full flex items-center justify-between p-5 transition-colors ${
          isDark ? 'hover:bg-bb-dark-300/30' : 'hover:bg-bb-light-100/50'
        }`}
        onClick={() => setOpen(o => !o)}
      >
        <div className="flex items-center gap-4">
          <div className="p-2.5 rounded-xl" style={{ backgroundColor: accent + '18' }}>
            <Icon className="w-6 h-6" style={{ color: accent }} />
          </div>
          <div className="text-left">
            <h2 className={`text-lg font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {suite.label}
            </h2>
            <p className={`text-xs mt-0.5 max-w-lg ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
              {suite.description}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 ml-4">
          <span
            className="text-xs font-mono font-bold px-2.5 py-1 rounded-full"
            style={{ backgroundColor: accent + '18', color: accent }}
          >
            {suite.tasks.length} tasks
          </span>
          {open
            ? <ChevronUp className={`w-5 h-5 ${isDark ? 'text-gray-400' : 'text-gray-500'}`} />
            : <ChevronDown className={`w-5 h-5 ${isDark ? 'text-gray-400' : 'text-gray-500'}`} />
          }
        </div>
      </button>

      {open && (
        <div className={`border-t ${isDark ? 'border-bb-dark-50/15' : 'border-bb-light-300/40'}`}>
          {/* Filters */}
          <div className={`px-5 py-3 flex flex-wrap gap-2 items-center border-b ${isDark ? 'border-bb-dark-50/10' : 'border-bb-light-300/30'}`}>
            {/* Search */}
            <div className={`relative flex items-center rounded-lg border text-sm ${
              isDark ? 'bg-bb-dark-500/50 border-bb-dark-50/20' : 'bg-bb-light-100 border-bb-light-300'
            }`}>
              <Search className={`w-3.5 h-3.5 ml-2.5 flex-shrink-0 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
              <input
                type="text"
                placeholder="Search tasks…"
                value={search}
                onChange={e => setSearch(e.target.value)}
                className={`bg-transparent pl-1.5 pr-8 py-1.5 text-xs outline-none w-36 ${isDark ? 'text-white placeholder-gray-600' : 'text-gray-900 placeholder-gray-400'}`}
              />
              {search && (
                <button onClick={() => setSearch('')} className="absolute right-2">
                  <X className={`w-3 h-3 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
                </button>
              )}
            </div>

            {/* Category filter chips */}
            <div className="flex flex-wrap gap-1.5">
              {categories.map(cat => (
                <button
                  key={cat}
                  onClick={() => setCategory(cat)}
                  className={`text-[11px] font-medium px-2.5 py-1 rounded-full border transition-all ${
                    category === cat
                      ? 'border-transparent'
                      : isDark
                        ? 'border-bb-dark-50/20 text-gray-400 hover:text-gray-200'
                        : 'border-bb-light-300 text-gray-500 hover:text-gray-700'
                  }`}
                  style={
                    category === cat
                      ? { backgroundColor: accent + '20', color: accent, borderColor: accent + '40' }
                      : {}
                  }
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>

          {/* Task list */}
          <div className="p-4 grid gap-2">
            {filtered.length > 0 ? (
              filtered.map(task => (
                <TaskCard
                  key={task.name}
                  task={task}
                  accentDark={suite.accentDark}
                  accentLight={suite.accentLight}
                  isDark={isDark}
                />
              ))
            ) : (
              <div className={`text-center py-10 text-sm ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                No tasks match your search.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function TaskBrowser() {
  const { isDark } = useTheme()
  const [globalSearch, setGlobalSearch] = useState('')

  const totalTasks = Object.values(TASK_DATA).reduce((s, suite) => s + suite.tasks.length, 0)
  const totalVariations = Object.values(TASK_DATA).reduce(
    (s, suite) => s + suite.tasks.reduce((vs, t) => vs + t.variations, 0),
    0
  )

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-3 mb-3">
          <div className={`p-2 rounded-xl ${isDark ? 'bg-bb-teal/10' : 'bg-blue-50'}`}>
            <BookOpen className={`w-7 h-7 ${isDark ? 'text-bb-teal' : 'text-blue-600'}`} />
          </div>
          <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Task Browser
          </h1>
        </div>
        <p className={`text-base max-w-2xl ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Explore all {totalTasks} reasoning tasks across 3 difficulty suites with {totalVariations} total variations.
          Click any task to see its description and an example problem.
        </p>

        {/* Summary stats */}
        <div className="flex flex-wrap gap-3 mt-5">
          {[
            { label: 'Easy Tasks', value: '44', icon: Zap, color: '#0ea5e9' },
            { label: 'Medium Tasks', value: '15', sub: '59 variations', icon: Brain, color: '#a855f7' },
            { label: 'Hard Tasks', value: '20', sub: '79 variations', icon: FlaskConical, color: '#00e676' },
            { label: 'Total Instances', value: '>10¹⁵', icon: Cpu, color: '#f59e0b' },
          ].map(({ label, value, sub, icon: Icon, color }) => (
            <div
              key={label}
              className={`flex items-center gap-3 px-4 py-2.5 rounded-xl border ${
                isDark
                  ? 'bg-bb-dark-300/50 border-bb-dark-50/20'
                  : 'bg-white/60 border-bb-light-300/50 shadow-sm'
              }`}
            >
              <Icon className="w-4 h-4 flex-shrink-0" style={{ color }} />
              <div>
                <span className={`text-base font-bold font-mono ${isDark ? 'text-white' : 'text-gray-900'}`}>{value}</span>
                <span className={`text-xs ml-1.5 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{label}</span>
                {sub && <span className={`text-[10px] block ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{sub}</span>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Suite sections */}
      {Object.entries(TASK_DATA).map(([key, suite]) => (
        <SuiteSection key={key} suiteKey={key} suite={suite} />
      ))}

      {/* Footer note */}
      <div className={`mt-6 rounded-xl border p-4 flex gap-3 items-start ${
        isDark
          ? 'bg-bb-dark-300/30 border-bb-dark-50/15'
          : 'bg-bb-light-100/60 border-bb-light-300/40'
      }`}>
        <Filter className={`w-4 h-4 flex-shrink-0 mt-0.5 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
        <p className={`text-xs leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
          <strong className={isDark ? 'text-gray-300' : 'text-gray-600'}>Contamination resistance</strong>: all tasks generate problems dynamically with a problem space exceeding 10¹⁵ unique instances, making memorization infeasible. Each evaluation run uses fresh, novel problem instances.
        </p>
      </div>
    </div>
  )
}

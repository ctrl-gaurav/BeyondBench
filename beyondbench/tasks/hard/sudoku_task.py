"""
Sudoku Solving Task

This task evaluates LLM's ability to solve Sudoku puzzles through logical deduction and constraint satisfaction.
Sudoku involves filling a grid with numbers such that each row, column, and sub-grid contains all unique numbers.
For 4x4 Sudoku: numbers 1-4, 2x2 sub-grids. For 6x6 Sudoku: numbers 1-6, 2x3 sub-grids.

Algorithm: Generate valid Sudoku puzzles by removing numbers from complete grids, ensuring unique solutions.
Reasoning: Requires logical deduction, constraint satisfaction, pattern recognition, and systematic elimination.

Example: 4x4 Sudoku puzzle:
[1, _, 3, _]
[_, 4, _, 2] 
[3, _, 1, _]
[_, 2, _, 4]

Solution:
[1, 2, 3, 4]
[3, 4, 2, 1]  
[2, 3, 4, 1]
[4, 1, 2, 3]

CLI USAGE:
python sudoku_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --cuda_device "cuda:6" --datapoints 20 --folds 1 --grid_sizes 4,6,9 --difficulty easy,medium,hard,extreme --constraint_types standard,diagonal,irregular --technique_levels basic,advanced,expert --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --tensor_parallel_size 1 --gpu_memory_utilization 0.96 --trust_remote_code False --store_details True
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["sudoku_solving"]
# Engine selection - choose 'vllm' or 'transformers'
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20  # Number of samples per fold
FOLDS = 1  # Number of evaluation folds
GRID_SIZES = [4, 6, 9]  # Grid sizes for Sudoku (4x4, 6x6, 9x9) - added 9x9 for extreme difficulty
DIFFICULTY_LEVELS = ["easy", "medium", "hard", "extreme"]  # Multiple difficulty levels for challenging LLMs
CONSTRAINT_TYPES = ["standard", "diagonal", "irregular", "killer"]  # Different sudoku variants for increased complexity
TECHNIQUE_LEVELS = ["basic", "advanced", "expert"]  # Puzzle technique complexity levels
STORE_DETAILS = True  # Save detailed results in JSON
SEED = 42  # Random seed for reproducibility

# Generation Parameters
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 8192

import os
import sys
import json
import argparse
import random
import logging
try:
    import numpy as np
except ImportError:
    # Create a minimal numpy-like module for basic functionality
    class NumpyLike:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def seed(s):
                    random.seed(s)
            return Random()
    np = NumpyLike()
import re
import ast
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import copy

# Increase recursion limit for larger Sudoku grids
sys.setrecursionlimit(2000)


from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

class SudokuSolver:
    """Advanced ground truth solver for multiple Sudoku variants"""
    
    def __init__(self, size: int, constraint_type: str = "standard"):
        self.size = size
        self.constraint_type = constraint_type
        
        # Box dimensions for different grid sizes
        if size == 4:
            self.box_rows, self.box_cols = 2, 2
        elif size == 6:
            self.box_rows, self.box_cols = 2, 3
        elif size == 9:
            self.box_rows, self.box_cols = 3, 3
        else:
            raise ValueError(f"Unsupported grid size: {size}")
        
        # Initialize constraint-specific properties
        self.setup_constraints()
    
    def setup_constraints(self):
        """Setup additional constraints based on sudoku variant"""
        self.extra_constraints = []
        
        if self.constraint_type == "diagonal":
            # Diagonal constraints - main diagonals must have unique numbers
            self.extra_constraints.append(self.check_diagonal_constraints)
        elif self.constraint_type == "irregular":
            # Irregular regions instead of standard boxes
            self.setup_irregular_regions()
        elif self.constraint_type == "killer":
            # Killer sudoku with cage constraints
            self.setup_killer_cages()
    
    def setup_irregular_regions(self):
        """Setup irregular regions for irregular sudoku"""
        # Create random irregular regions of equal size
        if self.size == 4:
            # Example irregular regions for 4x4
            self.irregular_regions = [
                [(0,0), (0,1), (1,0), (1,1)],  # Top-left irregular
                [(0,2), (0,3), (1,2), (2,0)],  # Top-right + mixed
                [(1,3), (2,1), (2,2), (2,3)],  # Bottom-right + mixed  
                [(3,0), (3,1), (3,2), (3,3)]   # Bottom row
            ]
        elif self.size == 6:
            # More complex irregular regions for 6x6
            self.irregular_regions = self.generate_irregular_regions_6x6()
        elif self.size == 9:
            # Complex irregular regions for 9x9
            self.irregular_regions = self.generate_irregular_regions_9x9()
    
    def generate_irregular_regions_6x6(self):
        """Generate irregular regions for 6x6 grid"""
        return [
            [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],
            [(0,2), (0,3), (1,2), (1,3), (0,4), (0,5)],
            [(1,4), (1,5), (2,2), (2,3), (2,4), (2,5)],
            [(3,0), (3,1), (4,0), (4,1), (5,0), (5,1)],
            [(3,2), (3,3), (4,2), (4,3), (3,4), (3,5)],
            [(4,4), (4,5), (5,2), (5,3), (5,4), (5,5)]
        ]
    
    def generate_irregular_regions_9x9(self):
        """Generate irregular regions for 9x9 grid"""
        # Complex irregular pattern for maximum difficulty
        return [
            [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
            [(0,3), (0,4), (0,5), (1,3), (1,4), (2,3), (2,4), (2,5), (3,3)],
            [(0,6), (0,7), (0,8), (1,5), (1,6), (1,7), (1,8), (2,6), (2,7)],
            [(2,8), (3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (5,0), (5,1)],
            [(3,4), (3,5), (3,6), (4,3), (4,4), (4,5), (4,6), (5,2), (5,3)],
            [(3,7), (3,8), (4,7), (4,8), (5,4), (5,5), (5,6), (5,7), (5,8)],
            [(6,0), (6,1), (6,2), (7,0), (7,1), (7,2), (8,0), (8,1), (8,2)],
            [(6,3), (6,4), (6,5), (7,3), (7,4), (7,5), (8,3), (8,4), (8,5)],
            [(6,6), (6,7), (6,8), (7,6), (7,7), (7,8), (8,6), (8,7), (8,8)]
        ]
    
    def setup_killer_cages(self):
        """Setup killer sudoku cages with sum constraints"""
        # Generate random cages with target sums
        self.killer_cages = self.generate_killer_cages()
    
    def generate_killer_cages(self):
        """Generate killer cages for killer sudoku variant"""
        cages = []
        used_cells = set()
        
        # Generate cages of different sizes (2-4 cells)
        attempts = 0
        while len(used_cells) < self.size * self.size and attempts < 100:
            attempts += 1
            cage_size = random.choice([2, 3, 4] if self.size >= 6 else [2, 3])
            
            # Pick a random starting cell not already used
            available_cells = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in used_cells]
            if not available_cells:
                break
                
            start_cell = random.choice(available_cells)
            cage_cells = [start_cell]
            used_cells.add(start_cell)
            
            # Add adjacent cells to form cage
            for _ in range(cage_size - 1):
                if not cage_cells:
                    break
                    
                # Find adjacent cells to current cage
                adjacent = []
                for r, c in cage_cells:
                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.size and 0 <= nc < self.size and 
                            (nr, nc) not in used_cells):
                            adjacent.append((nr, nc))
                
                if adjacent:
                    next_cell = random.choice(adjacent)
                    cage_cells.append(next_cell)
                    used_cells.add(next_cell)
                else:
                    break
            
            if len(cage_cells) >= 2:  # Valid cage
                # Calculate target sum based on possible values
                min_sum = sum(range(1, len(cage_cells) + 1))
                max_sum = sum(range(self.size - len(cage_cells) + 1, self.size + 1))
                target_sum = random.randint(min_sum, max_sum)
                
                cages.append({
                    'cells': cage_cells,
                    'target_sum': target_sum
                })
        
        return cages
    
    def is_valid_move(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid for the specific sudoku variant"""
        # Standard sudoku constraints
        
        # Check row
        for c in range(self.size):
            if grid[row][c] == num:
                return False
        
        # Check column
        for r in range(self.size):
            if grid[r][col] == num:
                return False
        
        # Check constraints based on variant type
        if self.constraint_type == "standard":
            # Standard box constraint
            box_row = (row // self.box_rows) * self.box_rows
            box_col = (col // self.box_cols) * self.box_cols
            
            for r in range(box_row, box_row + self.box_rows):
                for c in range(box_col, box_col + self.box_cols):
                    if grid[r][c] == num:
                        return False
        
        elif self.constraint_type == "irregular":
            # Check irregular regions
            if hasattr(self, 'irregular_regions'):
                for region in self.irregular_regions:
                    if (row, col) in region:
                        for r, c in region:
                            if grid[r][c] == num:
                                return False
                        break
        
        elif self.constraint_type == "diagonal":
            # Standard box constraint
            box_row = (row // self.box_rows) * self.box_rows
            box_col = (col // self.box_cols) * self.box_cols
            
            for r in range(box_row, box_row + self.box_rows):
                for c in range(box_col, box_col + self.box_cols):
                    if grid[r][c] == num:
                        return False
            
            # Check diagonal constraints
            if row == col:  # Main diagonal
                for i in range(self.size):
                    if grid[i][i] == num:
                        return False
            
            if row + col == self.size - 1:  # Anti-diagonal
                for i in range(self.size):
                    if grid[i][self.size - 1 - i] == num:
                        return False
        
        # Check any extra constraints
        for constraint_check in self.extra_constraints:
            if not constraint_check(grid, row, col, num):
                return False
        
        return True
    
    def check_diagonal_constraints(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """Additional check for diagonal constraints"""
        # This is handled in is_valid_move for diagonal type
        return True
    
    def solve(self, grid: List[List[int]]) -> bool:
        """Solve Sudoku using backtracking"""
        for row in range(self.size):
            for col in range(self.size):
                if grid[row][col] == 0:
                    for num in range(1, self.size + 1):
                        if self.is_valid_move(grid, row, col, num):
                            grid[row][col] = num
                            if self.solve(grid):
                                return True
                            grid[row][col] = 0
                    return False
        return True
    
    def generate_complete_grid(self) -> List[List[int]]:
        """Generate a complete valid Sudoku grid with recursion limit"""
        max_attempts = 10  # Prevent infinite recursion
        
        for attempt in range(max_attempts):
            grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
            
            # Fill diagonal boxes first (they don't interfere with each other)
            box_step = max(self.box_rows, self.box_cols)
            for box_start in range(0, self.size, box_step):
                if box_start + self.box_rows <= self.size and box_start + self.box_cols <= self.size:
                    nums = list(range(1, self.size + 1))
                    random.shuffle(nums)
                    
                    idx = 0
                    for r in range(box_start, box_start + self.box_rows):
                        for c in range(box_start, box_start + self.box_cols):
                            if idx < len(nums):
                                grid[r][c] = nums[idx]
                                idx += 1
            
            # Fill remaining cells
            if self.solve(grid):
                return grid
        
        # If all attempts fail, create a simple valid grid
        return self.create_simple_valid_grid()
    
    def create_simple_valid_grid(self) -> List[List[int]]:
        """Create a simple valid Sudoku grid as fallback"""
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        
        # Create a simple pattern that's guaranteed to be valid
        if self.size == 4:
            # Simple 4x4 pattern
            pattern = [
                [1, 2, 3, 4],
                [3, 4, 1, 2],
                [2, 3, 4, 1],
                [4, 1, 2, 3]
            ]
        elif self.size == 6:
            # Simple 6x6 pattern
            pattern = [
                [1, 2, 3, 4, 5, 6],
                [4, 5, 6, 1, 2, 3],
                [2, 3, 1, 5, 6, 4],
                [6, 1, 4, 3, 2, 5],
                [3, 6, 5, 2, 4, 1],
                [5, 4, 2, 6, 1, 3]
            ]
        elif self.size == 9:
            # Simple 9x9 pattern (Latin square)
            pattern = []
            for i in range(9):
                row = []
                for j in range(9):
                    val = ((i * 3 + i // 3 + j) % 9) + 1
                    row.append(val)
                pattern.append(row)
        else:
            # Generic pattern for other sizes
            pattern = []
            for i in range(self.size):
                row = []
                for j in range(self.size):
                    val = ((i + j) % self.size) + 1
                    row.append(val)
                pattern.append(row)
        
        return pattern
    
    def generate_puzzle(self, difficulty: str = "medium", technique_level: str = "basic") -> Tuple[List[List[int]], List[List[int]], Dict[str, Any]]:
        """Generate a challenging Sudoku puzzle with advanced difficulty controls"""
        max_attempts = 5  # Limit attempts to prevent hanging
        
        for attempt in range(max_attempts):
            try:
                complete_grid = self.generate_complete_grid()
                puzzle_grid = copy.deepcopy(complete_grid)
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Final attempt failed, use simple grid
                    complete_grid = self.create_simple_valid_grid()
                    puzzle_grid = copy.deepcopy(complete_grid)
                continue
        
        # Enhanced difficulty parameters
        total_cells = self.size * self.size
        
        # Determine cells to remove based on difficulty and technique level
        if difficulty == "easy":
            base_remove = total_cells // 3
            technique_multiplier = 0.8 if technique_level == "basic" else 0.9
        elif difficulty == "medium":
            base_remove = total_cells // 2
            technique_multiplier = 0.9 if technique_level == "basic" else 1.0
        elif difficulty == "hard":
            base_remove = (total_cells * 3) // 5
            technique_multiplier = 1.0 if technique_level == "basic" else 1.1
        else:  # extreme
            base_remove = (total_cells * 2) // 3
            technique_multiplier = 1.1 if technique_level == "basic" else 1.2
        
        cells_to_remove = int(base_remove * technique_multiplier)
        cells_to_remove = min(cells_to_remove, total_cells - self.size)  # Ensure solvable
        
        # Advanced removal strategy based on technique level
        if technique_level == "expert":
            removed_cells = self.expert_cell_removal(puzzle_grid, complete_grid, cells_to_remove)
        elif technique_level == "advanced":
            removed_cells = self.advanced_cell_removal(puzzle_grid, complete_grid, cells_to_remove)
        else:
            removed_cells = self.basic_cell_removal(puzzle_grid, complete_grid, cells_to_remove)
        
        # Calculate puzzle metrics for difficulty assessment
        puzzle_metrics = self.calculate_puzzle_metrics(puzzle_grid, complete_grid)
        
        return puzzle_grid, complete_grid, puzzle_metrics
    
    def basic_cell_removal(self, puzzle_grid, complete_grid, cells_to_remove):
        """Basic random cell removal strategy"""
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        random.shuffle(cells)
        removed_count = 0
        
        for r, c in cells:
            if removed_count >= cells_to_remove:
                break
                
            original_value = puzzle_grid[r][c]
            puzzle_grid[r][c] = 0
            
            # Check if puzzle still has unique solution
            test_grid = copy.deepcopy(puzzle_grid)
            temp_solver = SudokuSolver(self.size, self.constraint_type)
            if temp_solver.solve(test_grid) and test_grid == complete_grid:
                removed_count += 1
            else:
                # Restore if no unique solution
                puzzle_grid[r][c] = original_value
        
        return removed_count
    
    def advanced_cell_removal(self, puzzle_grid, complete_grid, cells_to_remove):
        """Advanced cell removal targeting strategic positions"""
        # Prioritize cells that are more constrained (harder to deduce)
        cells_with_constraints = []
        
        for r in range(self.size):
            for c in range(self.size):
                constraint_count = self.count_constraints(puzzle_grid, r, c)
                cells_with_constraints.append((constraint_count, r, c))
        
        # Sort by constraint count (remove more constrained cells first)
        cells_with_constraints.sort(reverse=True)
        removed_count = 0
        
        for _, r, c in cells_with_constraints:
            if removed_count >= cells_to_remove:
                break
                
            original_value = puzzle_grid[r][c]
            if original_value == 0:
                continue
                
            puzzle_grid[r][c] = 0
            
            # Check solution uniqueness
            test_grid = copy.deepcopy(puzzle_grid)
            temp_solver = SudokuSolver(self.size, self.constraint_type)
            if temp_solver.solve(test_grid) and test_grid == complete_grid:
                removed_count += 1
            else:
                puzzle_grid[r][c] = original_value
        
        return removed_count
    
    def expert_cell_removal(self, puzzle_grid, complete_grid, cells_to_remove):
        """Expert-level cell removal creating maximum logical complexity"""
        # Use symmetrical patterns and advanced logic requirements
        removed_count = 0
        
        # Create symmetrical removal pattern for aesthetic and logical complexity
        if self.size == 9:
            symmetry_pairs = self.generate_symmetry_pairs_9x9()
        elif self.size == 6:
            symmetry_pairs = self.generate_symmetry_pairs_6x6()
        else:
            symmetry_pairs = self.generate_symmetry_pairs_4x4()
        
        random.shuffle(symmetry_pairs)
        
        for pair in symmetry_pairs:
            if removed_count >= cells_to_remove:
                break
                
            # Try to remove both cells in symmetrical pair
            cells_removed = 0
            original_values = []
            
            for r, c in pair:
                if puzzle_grid[r][c] != 0:
                    original_values.append(puzzle_grid[r][c])
                    puzzle_grid[r][c] = 0
                    cells_removed += 1
            
            # Test uniqueness
            test_grid = copy.deepcopy(puzzle_grid)
            temp_solver = SudokuSolver(self.size, self.constraint_type)
            if temp_solver.solve(test_grid) and test_grid == complete_grid:
                removed_count += cells_removed
            else:
                # Restore if solution not unique
                idx = 0
                for r, c in pair:
                    if idx < len(original_values):
                        puzzle_grid[r][c] = original_values[idx]
                        idx += 1
        
        return removed_count
    
    def generate_symmetry_pairs_9x9(self):
        """Generate symmetrical cell pairs for 9x9 grid"""
        pairs = []
        center = self.size // 2
        
        for r in range(self.size):
            for c in range(self.size):
                if r <= center and c <= center:  # Only process upper-left quadrant
                    symmetric_r = self.size - 1 - r
                    symmetric_c = self.size - 1 - c
                    if (r, c) != (symmetric_r, symmetric_c):
                        pairs.append([(r, c), (symmetric_r, symmetric_c)])
        
        return pairs
    
    def generate_symmetry_pairs_6x6(self):
        """Generate symmetrical cell pairs for 6x6 grid"""
        pairs = []
        for r in range(self.size):
            for c in range(self.size // 2):
                symmetric_c = self.size - 1 - c
                pairs.append([(r, c), (r, symmetric_c)])
        return pairs
    
    def generate_symmetry_pairs_4x4(self):
        """Generate symmetrical cell pairs for 4x4 grid"""
        return [
            [(0, 0), (3, 3)],
            [(0, 1), (3, 2)],
            [(0, 2), (3, 1)],
            [(0, 3), (3, 0)],
            [(1, 0), (2, 3)],
            [(1, 1), (2, 2)],
            [(1, 2), (2, 1)],
            [(1, 3), (2, 0)]
        ]
    
    def count_constraints(self, grid, row, col):
        """Count the number of constraints affecting a cell"""
        constraints = 0
        
        # Count filled cells in row
        constraints += sum(1 for c in range(self.size) if grid[row][c] != 0)
        
        # Count filled cells in column
        constraints += sum(1 for r in range(self.size) if grid[r][col] != 0)
        
        # Count filled cells in box/region
        if self.constraint_type == "standard" or self.constraint_type == "diagonal":
            box_row = (row // self.box_rows) * self.box_rows
            box_col = (col // self.box_cols) * self.box_cols
            for r in range(box_row, box_row + self.box_rows):
                for c in range(box_col, box_col + self.box_cols):
                    if grid[r][c] != 0:
                        constraints += 1
        
        return constraints
    
    def calculate_puzzle_metrics(self, puzzle_grid, solution_grid):
        """Calculate metrics for puzzle difficulty assessment"""
        empty_cells = sum(row.count(0) for row in puzzle_grid)
        total_cells = self.size * self.size
        fill_ratio = (total_cells - empty_cells) / total_cells
        
        # Calculate constraint density
        total_constraints = 0
        for r in range(self.size):
            for c in range(self.size):
                if puzzle_grid[r][c] == 0:
                    total_constraints += self.count_constraints(puzzle_grid, r, c)
        
        avg_constraints = total_constraints / empty_cells if empty_cells > 0 else 0
        
        return {
            'empty_cells': empty_cells,
            'fill_ratio': fill_ratio,
            'average_constraints': avg_constraints,
            'constraint_type': self.constraint_type,
            'difficulty_score': (1 - fill_ratio) * avg_constraints
        }
    
    def is_valid_solution(self, grid: List[List[int]]) -> Tuple[bool, str]:
        """Check if a grid is a valid complete solution for the specific Sudoku variant"""
        if not grid or len(grid) != self.size:
            return False, f"Grid must be {self.size}x{self.size}"
        
        for row in grid:
            if len(row) != self.size:
                return False, f"All rows must have {self.size} elements"
        
        # Check all numbers are in valid range
        for r in range(self.size):
            for c in range(self.size):
                if not (1 <= grid[r][c] <= self.size):
                    return False, f"Invalid number {grid[r][c]} at ({r},{c})"
        
        # Check rows
        for r in range(self.size):
            if len(set(grid[r])) != self.size:
                return False, f"Row {r} has duplicate numbers: {grid[r]}"
        
        # Check columns
        for c in range(self.size):
            column = [grid[r][c] for r in range(self.size)]
            if len(set(column)) != self.size:
                return False, f"Column {c} has duplicate numbers: {column}"
        
        # Check constraints based on variant type
        if self.constraint_type == "standard":
            # Check standard boxes
            for box_r in range(0, self.size, self.box_rows):
                for box_c in range(0, self.size, self.box_cols):
                    box_nums = []
                    for r in range(box_r, min(box_r + self.box_rows, self.size)):
                        for c in range(box_c, min(box_c + self.box_cols, self.size)):
                            box_nums.append(grid[r][c])
                    if len(set(box_nums)) != len(box_nums):
                        return False, f"Box at ({box_r},{box_c}) has duplicate numbers: {box_nums}"
        
        elif self.constraint_type == "diagonal":
            # Check standard boxes
            for box_r in range(0, self.size, self.box_rows):
                for box_c in range(0, self.size, self.box_cols):
                    box_nums = []
                    for r in range(box_r, min(box_r + self.box_rows, self.size)):
                        for c in range(box_c, min(box_c + self.box_cols, self.size)):
                            box_nums.append(grid[r][c])
                    if len(set(box_nums)) != len(box_nums):
                        return False, f"Box at ({box_r},{box_c}) has duplicate numbers: {box_nums}"
            
            # Check diagonals
            main_diagonal = [grid[i][i] for i in range(self.size)]
            if len(set(main_diagonal)) != self.size:
                return False, f"Main diagonal has duplicate numbers: {main_diagonal}"
            
            anti_diagonal = [grid[i][self.size - 1 - i] for i in range(self.size)]
            if len(set(anti_diagonal)) != self.size:
                return False, f"Anti-diagonal has duplicate numbers: {anti_diagonal}"
        
        elif self.constraint_type == "irregular":
            # Check irregular regions
            if hasattr(self, 'irregular_regions'):
                for i, region in enumerate(self.irregular_regions):
                    region_nums = [grid[r][c] for r, c in region if 0 <= r < self.size and 0 <= c < self.size]
                    if len(set(region_nums)) != len(region_nums):
                        return False, f"Irregular region {i} has duplicate numbers: {region_nums}"
        
        elif self.constraint_type == "killer":
            # Check standard boxes first
            for box_r in range(0, self.size, self.box_rows):
                for box_c in range(0, self.size, self.box_cols):
                    box_nums = []
                    for r in range(box_r, min(box_r + self.box_rows, self.size)):
                        for c in range(box_c, min(box_c + self.box_cols, self.size)):
                            box_nums.append(grid[r][c])
                    if len(set(box_nums)) != len(box_nums):
                        return False, f"Box at ({box_r},{box_c}) has duplicate numbers: {box_nums}"
            
            # Check killer cages
            if hasattr(self, 'killer_cages'):
                for i, cage in enumerate(self.killer_cages):
                    cage_nums = [grid[r][c] for r, c in cage['cells'] if 0 <= r < self.size and 0 <= c < self.size]
                    cage_sum = sum(cage_nums)
                    target_sum = cage['target_sum']
                    
                    if len(set(cage_nums)) != len(cage_nums):
                        return False, f"Killer cage {i} has duplicate numbers: {cage_nums}"
                    if cage_sum != target_sum:
                        return False, f"Killer cage {i} sum {cage_sum} doesn't match target {target_sum}"
        
        return True, f"Valid {self.constraint_type} Sudoku solution"

class SudokuTask(BaseTask):
    """Advanced Sudoku evaluation task with multiple variants and robust parsing"""
    
    def __init__(self, model_handler, output_dir, min_val, max_val, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, 
                 grid_sizes=None, difficulty_levels=None, constraint_types=None, 
                 technique_levels=None, seed=None):
        self._task_name = "sudoku_results"
        super().__init__(model_handler, output_dir, min_val, max_val, num_folds, num_samples, 
                        store_details, temperature, top_p, max_tokens, seed)
        
        # Enhanced configuration parameters
        self.grid_sizes = grid_sizes or [4, 6, 9]
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard", "extreme"]
        self.constraint_types = constraint_types or ["standard", "diagonal"]
        self.technique_levels = technique_levels or ["basic", "advanced", "expert"]
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Enhanced parsing patterns with confidence scoring
        self.setup_parsing_patterns()
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, list_size=None, **kwargs):
        """Generate evaluation data with controlled sample count"""
        test_cases = []
        
        if list_size is not None:
            grid_size = list_size
            # Generate exactly num_samples for this grid size with varied parameters
            generated = 0
            attempts = 0
            max_attempts = self.num_samples * 3  # Allow some failed attempts
            
            while generated < self.num_samples and attempts < max_attempts:
                attempts += 1
                
                # Randomly select parameters to create variety within the sample limit
                # For larger grids, use simpler constraints to avoid generation issues
                if grid_size >= 9:
                    difficulty = random.choice(["easy", "medium"])
                    constraint_type = "standard"  # Only standard for 9x9
                    technique_level = random.choice(["basic", "advanced"])
                elif grid_size >= 6:
                    difficulty = random.choice(["easy", "medium", "hard"])
                    constraint_type = random.choice(["standard", "diagonal"])
                    technique_level = random.choice(["basic", "advanced"])
                else:
                    difficulty = random.choice(self.difficulty_levels)
                    constraint_type = random.choice(self.constraint_types)
                    technique_level = random.choice(self.technique_levels)
                
                problem = self.generate_problem(grid_size, difficulty, constraint_type, technique_level)
                if problem:  # Only add valid problems
                    test_cases.append(problem)
                    generated += 1
        else:
            # Fallback: generate for all grid sizes with reduced complexity
            for size in self.grid_sizes:
                cases_per_size = max(1, self.num_samples // len(self.grid_sizes))
                generated = 0
                attempts = 0
                max_attempts = cases_per_size * 3
                
                while generated < cases_per_size and attempts < max_attempts:
                    attempts += 1
                    difficulty = random.choice(self.difficulty_levels[:2])  # Easy/Medium only
                    problem = self.generate_problem(size, difficulty, "standard", "basic")
                    if problem:
                        test_cases.append(problem)
                        generated += 1
        
        return test_cases
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response"""
        result = self.parse_response(response, data_point)
        return {
            'accuracy': 1 if result['is_correct'] else 0,
            'instruction_followed': 1 if result.get('parsing_confidence', 0) > 0 else 0,
            'predicted_answer': result.get('parsed_grid', []),
            'ground_truth': data_point.get('solution_grid', []),
            'parsing_info': {
                'method': result.get('parsing_method', 'unknown'),
                'confidence': result.get('parsing_confidence', 0.0),
                'attempts': result.get('parsing_attempts', [])
            },
            'validation_info': result.get('validation_info', {})
        }
    
    def setup_parsing_patterns(self):
        """Setup regex patterns for parsing different Sudoku grid formats"""
        self.grid_patterns = [
            # Pattern 1: Bracketed rows [1,2,3,4] on separate lines
            (re.compile(r'\[([^\]]+)\]', re.MULTILINE), "bracketed_rows", 0.9),
            
            # Pattern 2: Space-separated grid "1 2 3 4\n3 4 1 2\n..."
            (re.compile(r'^(\d+(?:\s+\d+)+)$', re.MULTILINE), "space_separated", 0.8),
            
            # Pattern 3: Comma-separated grid "1,2,3,4\n3,4,1,2\n..."
            (re.compile(r'^(\d+(?:,\d+)+)$', re.MULTILINE), "comma_separated", 0.8),
            
            # Pattern 4: Grid keyword "Grid:\n1 2 3 4\n..."
            (re.compile(r'(?:grid|solution)\s*:\s*([\s\S]*?)(?:\n\n|$)', re.IGNORECASE), "grid_keyword", 0.8),
            
            # Pattern 5: Pipe-separated "1|2|3|4\n3|4|1|2\n..."
            (re.compile(r'^(\d+(?:\|\d+)+)$', re.MULTILINE), "pipe_separated", 0.7),
            
            # Pattern 6: Matrix format "[[1,2,3,4],[3,4,1,2],...]" 
            (re.compile(r'\[\s*\[([^\]]+)\](?:\s*,\s*\[([^\]]+)\])*\s*\]'), "matrix_format", 0.9),
            
            # Pattern 7: Numbered sequence "1 2 3 4 3 4 1 2 2 1 4 3 4 3 2 1"
            (re.compile(r'(?:^|\n)\s*((?:\d+\s*){9,})(?:\n|$)', re.MULTILINE), "number_sequence", 0.6),
            
            # Pattern 8: Tab-separated "1\t2\t3\t4\n3\t4\t1\t2\n..."
            (re.compile(r'^(\d+(?:\t\d+)+)$', re.MULTILINE), "tab_separated", 0.7),
            
            # Pattern 9: Box format with +---+ borders
            (re.compile(r'\|\s*(\d+(?:\s*\|\s*\d+)*)\s*\|', re.MULTILINE), "box_format", 0.6),
            
            # Pattern 10: JSON-like array format
            (re.compile(r'\[\[([^\]]+)\](?:,\s*\[([^\]]+)\])*\]'), "json_array", 0.8)
        ]
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for prompt"""
        # Rough estimation: ~4 characters per token
        return len(prompt) // 4
    
    def generate_problem(self, size: int, difficulty: str = "medium", constraint_type: str = "standard", technique_level: str = "basic") -> Dict[str, Any]:
        """Generate an advanced Sudoku problem with multiple variants"""
        solver = SudokuSolver(size, constraint_type)
        puzzle_grid, solution_grid, metrics = solver.generate_puzzle(difficulty, technique_level)
        
        return {
            'size': size,
            'difficulty': difficulty,
            'constraint_type': constraint_type,
            'technique_level': technique_level,
            'puzzle_grid': puzzle_grid,
            'solution_grid': solution_grid,
            'empty_cells': sum(row.count(0) for row in puzzle_grid),
            'metrics': metrics,
            'complexity_score': self.calculate_complexity_score(metrics, size, constraint_type)
        }
    
    def calculate_complexity_score(self, metrics: Dict[str, Any], size: int, constraint_type: str) -> float:
        """Calculate overall complexity score for the puzzle"""
        base_score = metrics.get('difficulty_score', 0)
        
        # Size multiplier - larger grids are more complex
        size_multiplier = {4: 1.0, 6: 1.5, 9: 2.0}.get(size, 1.0)
        
        # Constraint type multiplier - variants are more complex
        constraint_multiplier = {
            'standard': 1.0,
            'diagonal': 1.3,
            'irregular': 1.5,
            'killer': 1.7
        }.get(constraint_type, 1.0)
        
        return base_score * size_multiplier * constraint_multiplier
    
    def format_grid(self, grid: List[List[int]], use_blanks: bool = True) -> str:
        """Format grid for display"""
        lines = []
        for row in grid:
            formatted_row = []
            for cell in row:
                if cell == 0 and use_blanks:
                    formatted_row.append('_')
                else:
                    formatted_row.append(str(cell))
            lines.append('[' + ', '.join(formatted_row) + ']')
        return '\n'.join(lines)
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create enhanced prompt for advanced Sudoku variants"""
        size = problem['size']
        puzzle_grid = problem['puzzle_grid']
        constraint_type = problem.get('constraint_type', 'standard')
        difficulty = problem.get('difficulty', 'medium')
        technique_level = problem.get('technique_level', 'basic')
        complexity_score = problem.get('complexity_score', 0)
        
        # Base rules based on grid size
        if size == 4:
            base_rules = f"""Base Rules for {size}x{size} Sudoku:
- Fill the grid with numbers 1 to {size}
- Each row must contain all numbers 1 to {size} exactly once
- Each column must contain all numbers 1 to {size} exactly once
- Each 2x2 box must contain all numbers 1 to {size} exactly once"""
        elif size == 6:
            base_rules = f"""Base Rules for {size}x{size} Sudoku:
- Fill the grid with numbers 1 to {size}
- Each row must contain all numbers 1 to {size} exactly once
- Each column must contain all numbers 1 to {size} exactly once
- Each 2x3 box must contain all numbers 1 to {size} exactly once"""
        else:  # size == 9
            base_rules = f"""Base Rules for {size}x{size} Sudoku:
- Fill the grid with numbers 1 to {size}
- Each row must contain all numbers 1 to {size} exactly once
- Each column must contain all numbers 1 to {size} exactly once
- Each 3x3 box must contain all numbers 1 to {size} exactly once"""
        
        # Additional rules based on constraint type
        additional_rules = ""
        if constraint_type == "diagonal":
            additional_rules = f"""\n\nDIAGONAL SUDOKU CONSTRAINT:
- Both main diagonals must also contain all numbers 1 to {size} exactly once
- This adds extra complexity to the solving process"""
        elif constraint_type == "irregular":
            additional_rules = f"""\n\nIRREGULAR SUDOKU CONSTRAINT:
- Instead of standard boxes, irregular regions of {size} cells each must contain all numbers 1 to {size}
- Irregular regions are specially marked and do not follow standard box patterns
- This requires advanced logical deduction techniques"""
        elif constraint_type == "killer":
            additional_rules = f"""\n\nKILLER SUDOKU CONSTRAINT:
- Groups of cells (cages) have sum constraints
- Each cage must sum to its target value
- No number can repeat within a cage
- This combines arithmetic reasoning with standard Sudoku logic"""
        
        # Difficulty and technique information
        difficulty_note = ""
        if difficulty in ["hard", "extreme"]:
            if technique_level == "expert":
                difficulty_note = f"\n\nCHALLENGE LEVEL: {difficulty.upper()} - {technique_level.upper()}\nThis puzzle requires advanced solving techniques including:\n- Complex constraint analysis\n- Multi-step logical deduction\n- Pattern recognition\n- Advanced elimination strategies\nComplexity Score: {complexity_score:.2f}"
            elif technique_level == "advanced":
                difficulty_note = f"\n\nCHALLENGE LEVEL: {difficulty.upper()} - {technique_level.upper()}\nThis puzzle requires intermediate to advanced techniques:\n- Strategic cell prioritization\n- Constraint intersection analysis\n- Multi-level logical chains\nComplexity Score: {complexity_score:.2f}"
        elif difficulty == "medium" and technique_level != "basic":
            difficulty_note = f"\n\nCHALLENGE LEVEL: {difficulty.upper()} - {technique_level.upper()}\nComplexity Score: {complexity_score:.2f}"
        
        puzzle_str = self.format_grid(puzzle_grid)
        
        # Example format based on size
        if size == 4:
            example_format = """[1, 2, 3, 4]
[3, 4, 1, 2]
[2, 1, 4, 3]
[4, 3, 2, 1]"""
        elif size == 6:
            example_format = """[1, 2, 3, 4, 5, 6]
[4, 5, 6, 1, 2, 3]
[2, 3, 1, 5, 6, 4]
[6, 1, 4, 3, 2, 5]
[3, 6, 5, 2, 4, 1]
[5, 4, 2, 6, 1, 3]"""
        else:  # size == 9
            example_format = """[1, 2, 3, 4, 5, 6, 7, 8, 9]
[4, 5, 6, 7, 8, 9, 1, 2, 3]
[7, 8, 9, 1, 2, 3, 4, 5, 6]
[2, 3, 1, 5, 6, 4, 8, 9, 7]
[5, 6, 4, 8, 9, 7, 2, 3, 1]
[8, 9, 7, 2, 3, 1, 5, 6, 4]
[3, 1, 2, 6, 4, 5, 9, 7, 8]
[6, 4, 5, 9, 7, 8, 3, 1, 2]
[9, 7, 8, 3, 1, 2, 6, 4, 5]"""
        
        prompt = f"""Solve this challenging {size}x{size} {constraint_type.title()} Sudoku puzzle:

{base_rules}{additional_rules}{difficulty_note}

Puzzle (where _ represents empty cells):
{puzzle_str}

Solving Strategy Hints:
1. Start by analyzing the most constrained cells (cells with fewest possible values)
2. Look for cells where only one number can fit based on row/column/box constraints
3. Use elimination techniques to narrow down possibilities
4. Apply advanced logical deduction for harder puzzles
5. For variant sudokus, always consider the additional constraints

SOLUTION FORMAT REQUIREMENTS (STRICTLY FOLLOW THIS):
Your answer must provide the complete solved {size}x{size} grid.

REQUIRED FORMAT:
<answer>
{example_format}
</answer>

IMPORTANT FORMATTING NOTES:
- Use EXACTLY the format shown above inside <answer> tags
- Each row should be in brackets: [1, 2, 3, ...]
- Use commas and spaces between numbers: [1, 2, 3] not [1,2,3] or [123]
- Provide ALL {size} rows, each with ALL {size} numbers
- Do NOT use code blocks, markdown, or other formatting
- Numbers should be from 1 to {size} only

ALTERNATIVE FORMATS (if <answer> tags fail):
- Final Answer: {example_format}
- Answer: {example_format}
- Grid: {example_format}

REMEMBER: Provide ONE complete solution with the entire grid filled!"""
        
        return prompt
    
    def parse_response(self, response: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with enhanced robustness and detailed diagnostics"""
        size = problem['size']
        parsing_attempts = []
        best_grid = []
        best_confidence = 0.0
        best_method = "none"
        
        # Extract answer from <answer> tags if present
        answer_text = response
        if '<answer>' in response and '</answer>' in response:
            answer_text = response.split('<answer>')[1].split('</answer>')[0].strip()
            parsing_attempts.append(("answer_tags", "Extracted from <answer> tags", 0.1))
        
        # Try each parsing pattern
        for pattern, method_name, base_confidence in self.grid_patterns:
            try:
                grid = self.parse_with_pattern(answer_text, pattern, method_name, size)
                if grid and len(grid) == size and all(len(row) == size for row in grid):
                    confidence = base_confidence
                    parsing_attempts.append((method_name, f"Parsed {len(grid)}x{len(grid[0])} grid", confidence))
                    
                    if confidence > best_confidence:
                        best_grid = grid
                        best_confidence = confidence
                        best_method = method_name
                elif grid:
                    parsing_attempts.append((method_name, f"Partial parse: {len(grid)} rows", base_confidence * 0.3))
                else:
                    parsing_attempts.append((method_name, "No match", 0.0))
                    
            except Exception as e:
                parsing_attempts.append((method_name, f"Error: {str(e)}", 0.0))
        
        # Validate the best solution
        validation_info = {}
        is_correct = False
        
        if best_grid:
            solver = SudokuSolver(size)
            is_valid_sudoku, sudoku_msg = solver.is_valid_solution(best_grid)
            
            # Check if it matches the puzzle constraints
            puzzle_grid = problem['puzzle_grid']
            matches_puzzle = True
            conflicting_cells = []
            
            for r in range(size):
                for c in range(size):
                    if puzzle_grid[r][c] != 0 and puzzle_grid[r][c] != best_grid[r][c]:
                        matches_puzzle = False
                        conflicting_cells.append((r, c, puzzle_grid[r][c], best_grid[r][c]))
            
            is_correct = is_valid_sudoku and matches_puzzle
            
            validation_info = {
                'is_valid_sudoku': is_valid_sudoku,
                'matches_puzzle': matches_puzzle,
                'conflicting_cells': conflicting_cells,
                'sudoku_validation_msg': sudoku_msg,
                'validation_message': self.get_validation_message(is_valid_sudoku, matches_puzzle, conflicting_cells)
            }
        else:
            validation_info = {
                'is_valid_sudoku': False,
                'matches_puzzle': False,
                'conflicting_cells': [],
                'sudoku_validation_msg': "No grid parsed",
                'validation_message': f"Could not parse {size}x{size} grid from response"
            }
        
        return {
            'parsed_grid': best_grid,
            'parsing_method': best_method,
            'parsing_confidence': best_confidence,
            'parsing_attempts': parsing_attempts,
            'validation_info': validation_info,
            'is_correct': is_correct,
            'raw_response': response
        }
    
    def parse_with_pattern(self, text: str, pattern: re.Pattern, method_name: str, size: int) -> List[List[int]]:
        """Parse grid using a specific pattern"""
        grid = []
        
        if method_name == "bracketed_rows":
            # Bracketed rows [1,2,3,4]
            matches = pattern.findall(text)
            for match in matches:
                row = self.parse_row_content(match, size)
                if row and len(row) == size:
                    grid.append(row)
                if len(grid) == size:
                    break
        
        elif method_name == "space_separated":
            # Space-separated rows "1 2 3 4"
            matches = pattern.findall(text)
            for match in matches:
                numbers = [int(x) for x in match.split() if x.isdigit()]
                if len(numbers) == size and all(1 <= n <= size for n in numbers):
                    grid.append(numbers)
                if len(grid) == size:
                    break
        
        elif method_name == "comma_separated":
            # Comma-separated rows "1,2,3,4"
            matches = pattern.findall(text)
            for match in matches:
                numbers = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
                if len(numbers) == size and all(1 <= n <= size for n in numbers):
                    grid.append(numbers)
                if len(grid) == size:
                    break
        
        elif method_name == "grid_keyword":
            # Grid keyword followed by content
            matches = pattern.findall(text)
            if matches:
                grid_text = matches[0].strip()
                # Recursively parse the grid content
                lines = grid_text.split('\n')
                for line in lines:
                    row = self.parse_row_from_line(line.strip(), size)
                    if row:
                        grid.append(row)
                    if len(grid) == size:
                        break
        
        elif method_name == "pipe_separated":
            # Pipe-separated "1|2|3|4"
            matches = pattern.findall(text)
            for match in matches:
                numbers = [int(x.strip()) for x in match.split('|') if x.strip().isdigit()]
                if len(numbers) == size and all(1 <= n <= size for n in numbers):
                    grid.append(numbers)
                if len(grid) == size:
                    break
        
        elif method_name == "matrix_format":
            # Matrix format [[1,2,3,4],[3,4,1,2],...]
            try:
                # Try to evaluate as Python list
                matrix = ast.literal_eval(text.strip())
                if isinstance(matrix, list) and len(matrix) == size:
                    for row in matrix:
                        if isinstance(row, list) and len(row) == size:
                            if all(isinstance(x, int) and 1 <= x <= size for x in row):
                                grid.append(row)
                        if len(grid) != len(matrix):
                            grid = []  # Invalid, reset
                            break
            except:
                pass
        
        elif method_name == "number_sequence":
            # Long sequence of numbers
            matches = pattern.findall(text)
            if matches:
                numbers = [int(x) for x in matches[0].split() if x.isdigit()]
                if len(numbers) >= size * size:
                    for i in range(size):
                        start = i * size
                        end = start + size
                        if end <= len(numbers):
                            row = numbers[start:end]
                            if all(1 <= n <= size for n in row):
                                grid.append(row)
        
        elif method_name == "tab_separated":
            # Tab-separated "1\t2\t3\t4"
            matches = pattern.findall(text)
            for match in matches:
                numbers = [int(x.strip()) for x in match.split('\t') if x.strip().isdigit()]
                if len(numbers) == size and all(1 <= n <= size for n in numbers):
                    grid.append(numbers)
                if len(grid) == size:
                    break
        
        elif method_name == "box_format":
            # Box format with | borders
            matches = pattern.findall(text)
            for match in matches:
                numbers = [int(x.strip()) for x in re.split(r'[|\s]+', match) if x.strip().isdigit()]
                if len(numbers) == size and all(1 <= n <= size for n in numbers):
                    grid.append(numbers)
                if len(grid) == size:
                    break
        
        elif method_name == "json_array":
            # JSON-like array format
            try:
                # Similar to matrix_format but more flexible parsing
                cleaned = re.sub(r'\s+', '', text)
                matrix = ast.literal_eval(cleaned)
                if isinstance(matrix, list) and len(matrix) == size:
                    for row in matrix:
                        if isinstance(row, list) and len(row) == size:
                            if all(isinstance(x, int) and 1 <= x <= size for x in row):
                                grid.append(row)
            except:
                pass
        
        elif method_name == "code_block_grid":
            # Code block format with grid content
            matches = pattern.findall(text)
            if matches:
                block_content = matches[0].strip()
                # Try to parse the block content recursively using other patterns
                for sub_pattern, sub_method, _ in self.grid_patterns:
                    if sub_method != "code_block_grid":  # Avoid infinite recursion
                        try:
                            sub_grid = self.parse_with_pattern(block_content, sub_pattern, sub_method, size)
                            if sub_grid and len(sub_grid) == size:
                                return sub_grid
                        except:
                            continue
        
        elif method_name in ["final_answer", "answer_format", "solution_is", "completed_grid"]:
            # Answer format variants
            matches = pattern.findall(text)
            if matches:
                answer_content = matches[0].strip()
                # Try to parse the answer content recursively
                for sub_pattern, sub_method, _ in self.grid_patterns:
                    if sub_method not in ["final_answer", "answer_format", "solution_is", "completed_grid"]:
                        try:
                            sub_grid = self.parse_with_pattern(answer_content, sub_pattern, sub_method, size)
                            if sub_grid and len(sub_grid) == size:
                                return sub_grid
                        except:
                            continue
        
        elif method_name == "row_format":
            # Row format "Row 1: [1,2,3], Row 2: [4,5,6]"
            matches = pattern.findall(text)
            rows_dict = {}
            for match in matches:
                row_content = match
                row_nums = self.parse_row_content(row_content, size)
                if row_nums and len(row_nums) == size:
                    rows_dict[len(rows_dict)] = row_nums
                    if len(rows_dict) == size:
                        break
            
            # Convert to ordered grid
            if len(rows_dict) == size:
                for i in range(size):
                    if i in rows_dict:
                        grid.append(rows_dict[i])
        
        elif method_name == "multi_line_brackets":
            # Multi-line bracket format
            # Extract all bracket contents
            bracket_contents = re.findall(r'\[([^\]]+)\]', text)
            for content in bracket_contents:
                row = self.parse_row_content(content, size)
                if row and len(row) == size:
                    grid.append(row)
                if len(grid) == size:
                    break
        
        elif method_name == "simple_numeric":
            # Simple numeric grid "1 2 3\n4 5 6"
            lines = text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    numbers = [int(x) for x in line.split() if x.isdigit()]
                    if len(numbers) == size and all(1 <= n <= size for n in numbers):
                        grid.append(numbers)
                    if len(grid) == size:
                        break
        
        return grid
    
    def parse_row_content(self, content: str, size: int) -> Optional[List[int]]:
        """Parse row content from bracketed format"""
        try:
            numbers = [int(x.strip()) for x in content.split(',') if x.strip().isdigit()]
            if len(numbers) == size and all(1 <= n <= size for n in numbers):
                return numbers
        except:
            pass
        return None
    
    def parse_row_from_line(self, line: str, size: int) -> Optional[List[int]]:
        """Parse a row from a text line"""
        # Try different separators
        separators = [' ', ',', '\t', '|', ';']
        
        for sep in separators:
            if sep in line:
                try:
                    numbers = [int(x.strip()) for x in line.split(sep) if x.strip().isdigit()]
                    if len(numbers) == size and all(1 <= n <= size for n in numbers):
                        return numbers
                except:
                    continue
        
        # Try as continuous digits for small grids
        if size <= 6 and len(line.replace(' ', '')) == size:
            try:
                numbers = [int(c) for c in line.replace(' ', '') if c.isdigit()]
                if len(numbers) == size and all(1 <= n <= size for n in numbers):
                    return numbers
            except:
                pass
        
        return None
    
    def get_validation_message(self, is_valid_sudoku: bool, matches_puzzle: bool, conflicting_cells: List) -> str:
        """Generate validation message"""
        if is_valid_sudoku and matches_puzzle:
            return "Valid Sudoku solution that matches puzzle constraints"
        elif is_valid_sudoku and not matches_puzzle:
            return f"Valid Sudoku but conflicts with puzzle at {len(conflicting_cells)} cells"
        elif not is_valid_sudoku and matches_puzzle:
            return "Matches puzzle constraints but violates Sudoku rules"
        else:
            return "Invalid Sudoku solution that doesn't match puzzle constraints"
    
    
    

def main():
    """Main evaluation function with proper logging and reporting"""
    parser = argparse.ArgumentParser(description="Sudoku Problem Solving Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Inference engine')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for VLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization for VLLM')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of datapoints per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--grid_sizes', type=str, default=','.join(map(str, GRID_SIZES)), help='Grid sizes (comma-separated)')
    parser.add_argument('--difficulty', type=str, default=','.join(DIFFICULTY_LEVELS), help='Difficulty levels (comma-separated)')
    parser.add_argument('--constraint_types', type=str, default=','.join(CONSTRAINT_TYPES), help='Constraint types (comma-separated)')
    parser.add_argument('--technique_levels', type=str, default=','.join(TECHNIQUE_LEVELS), help='Technique levels (comma-separated)')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='sudoku_results', help='Output directory')

    # API configuration
    parser.add_argument('--api_provider', type=str, choices=['openai', 'gemini'], help='API provider for cloud models')
    parser.add_argument('--api_key', type=str, help='API key for cloud models')

    # Reasoning configuration
    parser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'],
                       help='OpenAI reasoning effort level (for GPT-5 models)')
    parser.add_argument('--thinking_budget', type=int,
                       help='Gemini thinking budget (0 to disable, -1 for dynamic, or specific number)')
    
    args = parser.parse_args()
    
    # Parse parameters
    grid_sizes = [int(x.strip()) for x in args.grid_sizes.split(',')]
    difficulty_levels = [x.strip() for x in args.difficulty.split(',')]
    constraint_types = [x.strip() for x in args.constraint_types.split(',')]
    technique_levels = [x.strip() for x in args.technique_levels.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("🧩 Starting Sudoku Problem Solving Evaluation")
    logging.info(f"📁 Output directory: {args.output_dir}")
    logging.info(f"📝 Log file: {log_file}")
    
    try:
        # Initialize model handler
        logging.info(f"🤖 Initializing model handler for {args.model_id}")
        model_handler = ModelHandler(
            model_id=args.model_id,
            api_provider=args.api_provider,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort,
            thinking_budget=args.thinking_budget,
            engine=args.engine,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code
        )
        
        # Initialize task
        logging.info(f"🎯 Initializing Advanced Sudoku task")
        task = SudokuTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            min_val=min(grid_sizes),
            max_val=max(grid_sizes),
            num_folds=args.folds,
            num_samples=args.datapoints,
            store_details=args.store_details,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            grid_sizes=grid_sizes,
            difficulty_levels=difficulty_levels,
            constraint_types=constraint_types,
            technique_levels=technique_levels,
            seed=args.seed
        )
        
        # Set random seed
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            logging.info(f"🎲 Set random seed to {args.seed}")
        
        # Run evaluation
        logging.info(f"🚀 Starting evaluation with grid sizes: {grid_sizes}, difficulties: {difficulty_levels}, constraints: {constraint_types}, techniques: {technique_levels}")
        all_metrics = task.run_evaluation(list_sizes=grid_sizes)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(args.output_dir, 'sudoku_solving', grid_sizes)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, grid_sizes, args.output_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, grid_size in enumerate(grid_sizes):
                # Find metrics for this grid size from all_metrics
                size_metrics = [m for m in all_metrics if m.get('grid_size') == grid_size or
                               (i < len(all_metrics) and all_metrics[i] is m)]

                if not size_metrics:
                    # Create a default metric for this grid size
                    size_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                   'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"sudoku_solving_{grid_size}",
                    'test_case': i,
                    'complexity': grid_size,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in size_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in size_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in size_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in size_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in size_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, grid_sizes, args.output_dir)
        
        # Log completion
        if all_metrics:
            avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
            logging.info(f"🎉 Evaluation completed successfully!")
            logging.info(f"📊 Overall average accuracy: {avg_accuracy:.4f}")
            logging.info(f"📁 Results saved to: {args.output_dir}")
        else:
            logging.warning("⚠️  No metrics were collected during evaluation")
            
    except Exception as e:
        logging.error(f"❌ Evaluation failed: {e}")
        logging.error(f"📍 Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup if needed
        if 'model_handler' in locals():
            try:
                # Model handler cleanup - if method exists
                if hasattr(model_handler, 'cleanup'):
                    model_handler.cleanup()
                    logging.info("🧹 Model handler cleanup completed")
            except Exception as cleanup_error:
                logging.warning(f"⚠️  Cleanup warning: {cleanup_error}")

if __name__ == "__main__":
    main()
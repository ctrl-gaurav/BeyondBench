"""
Enhanced N-Queens Problem Solving Task - Robust Implementation

This enhanced version includes:
- Multiple parsing patterns for different response formats
- Confidence scoring for parsed solutions
- Better error recovery and diagnostics
- Enhanced validation with detailed feedback
- Support for various queen position representations

Algorithm: Generate N-Queens problems (N=4 to 8), ask for valid queen placements.
Reasoning: Requires constraint satisfaction, backtracking, spatial reasoning, and conflict detection.
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False
DATAPOINTS = 20
FOLDS = 1
BOARD_SIZES = [4, 5, 6, 8]
STORE_DETAILS = True
SEED = 42
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 8192

import os
import sys
import json
import argparse
import random
import logging
import numpy as np
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

class ParseResult(Enum):
    """Parse result types"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID_FORMAT = "invalid_format"
    EMPTY_RESPONSE = "empty_response"

@dataclass
class ParsedSolution:
    """Parsed N-Queens solution with confidence"""
    positions: Optional[List[int]]
    confidence: float
    format_used: str
    original_text: str
    validation_errors: List[str]
    
    def is_valid(self, n: int) -> bool:
        """Check if solution is valid for board size n"""
        if not self.positions or len(self.positions) != n:
            return False
        return all(0 <= pos < n for pos in self.positions)

@dataclass
class ParsingResult:
    """Complete parsing result with diagnostics"""
    solution: Optional[List[int]]
    result_type: ParseResult
    confidence: float
    parsing_attempts: List[ParsedSolution]
    diagnostics: Dict[str, Any]
    raw_response: str

class NQueensResponseParser:
    """Enhanced parser with multiple strategies for N-Queens solutions"""
    
    def __init__(self):
        # Multiple parsing patterns for robustness
        self.solution_patterns = [
            # List format: [1, 3, 0, 2]
            (re.compile(r'\[\s*([0-9,\s]+)\s*\]'), "list_brackets"),
            
            # Tuple format: (1, 3, 0, 2)
            (re.compile(r'\(\s*([0-9,\s]+)\s*\)'), "tuple_parentheses"),
            
            # Solution format: "Solution: 1, 3, 0, 2"
            (re.compile(r'solution\s*:?\s*\[?\s*([0-9,\s]+)\s*\]?', re.IGNORECASE), "solution_keyword"),
            
            # Answer format: "Answer: [1, 3, 0, 2]" or "Answer: 1, 3, 0, 2"
            (re.compile(r'answer\s*:?\s*\[?\s*([0-9,\s]+)\s*\]?', re.IGNORECASE), "answer_keyword"),
            
            # Positions format: "Positions: 1, 3, 0, 2"
            (re.compile(r'positions?\s*:?\s*\[?\s*([0-9,\s]+)\s*\]?', re.IGNORECASE), "positions_keyword"),
            
            # Queens format: "Queens at columns: 1, 3, 0, 2"
            (re.compile(r'queens?\s+(?:at\s+)?columns?\s*:?\s*\[?\s*([0-9,\s]+)\s*\]?', re.IGNORECASE), "queens_columns"),
            
            # Column format: "columns: 1, 3, 0, 2" 
            (re.compile(r'columns?\s*:?\s*\[?\s*([0-9,\s]+)\s*\]?', re.IGNORECASE), "columns_keyword"),
            
            # Result format: "Result: [1, 3, 0, 2]"
            (re.compile(r'result\s*:?\s*\[?\s*([0-9,\s]+)\s*\]?', re.IGNORECASE), "result_keyword"),
            
            # Row-by-row format: "Row 0: column 1, Row 1: column 3..."
            (re.compile(r'row\s+\d+\s*:\s*column\s+(\d+)', re.IGNORECASE), "row_by_row"),
            
            # Just comma-separated numbers in square brackets or parentheses
            (re.compile(r'[\[\(]\s*(\d+(?:\s*,\s*\d+)*)\s*[\]\)]'), "bracketed_numbers"),
            
            # Just comma-separated numbers
            (re.compile(r'\b(\d+(?:\s*,\s*\d+){2,})\b'), "comma_separated"),
            
            # Space-separated numbers (at least 3 numbers)
            (re.compile(r'\b(\d+(?:\s+\d+){2,})\b'), "space_separated"),
            
            # Numbers with "and" separators: "1 and 3 and 0 and 2"
            (re.compile(r'(\d+(?:\s+and\s+\d+)+)', re.IGNORECASE), "and_separated"),
        ]
    
    def parse_response(self, response: str, n: int) -> ParsingResult:
        """Parse LLM response with multiple strategies"""
        if not response or not response.strip():
            return ParsingResult(
                solution=None, result_type=ParseResult.EMPTY_RESPONSE,
                confidence=0.0, parsing_attempts=[], 
                diagnostics={'error': 'Empty response'}, raw_response=response
            )
        
        # Try to extract from <answer> tags first, then other possible formats
        answer_text = response.strip()
        
        # Priority extraction order
        if '<answer>' in response and '</answer>' in response:
            answer_text = response.split('<answer>')[1].split('</answer>')[0].strip()
        elif '```' in response:
            # Extract from code blocks
            code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', response, re.DOTALL)
            if code_blocks:
                answer_text = code_blocks[0].strip()
        elif 'ANSWER:' in response.upper():
            # Extract after "ANSWER:" (case-insensitive)
            parts = re.split(r'answer\s*:', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer_text = parts[1].strip()
        elif 'SOLUTION:' in response.upper():
            # Extract after "SOLUTION:" (case-insensitive)
            parts = re.split(r'solution\s*:', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer_text = parts[1].strip()
        
        parsing_attempts = []
        
        # Try each pattern
        for pattern, format_name in self.solution_patterns:
            if format_name == "row_by_row":
                # Special handling for row-by-row format
                matches = pattern.findall(answer_text)
                if len(matches) == n:
                    try:
                        positions = [int(col) for col in matches]
                        confidence = self._calculate_confidence(format_name, positions, n, answer_text)
                        
                        parsed = ParsedSolution(
                            positions=positions,
                            confidence=confidence,
                            format_used=format_name,
                            original_text=answer_text,
                            validation_errors=self._validate_positions(positions, n)
                        )
                        parsing_attempts.append(parsed)
                    except ValueError:
                        continue
            else:
                # Standard pattern matching
                matches = pattern.findall(answer_text)
                for match in matches:
                    try:
                        # Clean and split the match - handle multiple separators
                        if format_name == "and_separated":
                            # Handle "1 and 3 and 0 and 2"
                            numbers_str = re.sub(r'\s+and\s+', ',', match, flags=re.IGNORECASE)
                        else:
                            numbers_str = match
                        
                        # Clean the string - keep only digits, commas, and spaces
                        numbers_str = re.sub(r'[^\d,\s]', '', numbers_str)
                        
                        # Extract numbers
                        if ',' in numbers_str:
                            positions = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
                        else:
                            positions = [int(x) for x in numbers_str.split() if x.isdigit()]
                        
                        # Only accept if we have the expected number of positions
                        if len(positions) == n:
                            confidence = self._calculate_confidence(format_name, positions, n, match)
                            
                            parsed = ParsedSolution(
                                positions=positions,
                                confidence=confidence,
                                format_used=format_name,
                                original_text=match,
                                validation_errors=self._validate_positions(positions, n)
                            )
                            parsing_attempts.append(parsed)
                    except (ValueError, IndexError):
                        continue
        
        # Fallback: try to find any sequence of exactly n numbers
        if not parsing_attempts:
            # Last resort: find all numbers and try different subsequences
            all_numbers = re.findall(r'\b\d+\b', answer_text)
            if all_numbers:
                numbers = [int(x) for x in all_numbers if 0 <= int(x) < n]
                
                # Try different subsequences of length n
                for i in range(len(numbers) - n + 1):
                    subseq = numbers[i:i+n]
                    if len(subseq) == n:
                        confidence = 0.3  # Low confidence for fallback
                        
                        parsed = ParsedSolution(
                            positions=subseq,
                            confidence=confidence,
                            format_used="fallback_sequence",
                            original_text=f"Numbers found: {subseq}",
                            validation_errors=self._validate_positions(subseq, n)
                        )
                        parsing_attempts.append(parsed)
        
        # Select best solution
        if not parsing_attempts:
            return ParsingResult(
                solution=None, result_type=ParseResult.FAILED,
                confidence=0.0, parsing_attempts=[],
                diagnostics={'error': 'No valid patterns found'}, raw_response=response
            )
        
        # Sort by confidence - focus on parsing success, not validation
        best_attempt = max(parsing_attempts, key=lambda x: x.confidence)
        
        # Determine result type based on parsing success, not validation
        if best_attempt.is_valid(n) and not best_attempt.validation_errors:
            result_type = ParseResult.SUCCESS
        elif best_attempt.is_valid(n):  # Parsed correctly but has conflicts
            result_type = ParseResult.PARTIAL  
        else:
            result_type = ParseResult.FAILED
        
        diagnostics = {
            'total_attempts': len(parsing_attempts),
            'valid_attempts': len([p for p in parsing_attempts if p.is_valid(n)]),
            'best_format': best_attempt.format_used,
            'validation_errors': best_attempt.validation_errors
        }
        
        return ParsingResult(
            solution=best_attempt.positions,
            result_type=result_type,
            confidence=best_attempt.confidence,
            parsing_attempts=parsing_attempts,
            diagnostics=diagnostics,
            raw_response=response
        )
    
    def _calculate_confidence(self, format_name: str, positions: List[int], n: int, text: str) -> float:
        """Calculate confidence score"""
        base_confidences = {
            "list_brackets": 0.95,
            "tuple_parentheses": 0.90,
            "answer_keyword": 0.92,
            "solution_keyword": 0.88,
            "positions_keyword": 0.87,
            "queens_columns": 0.85,
            "columns_keyword": 0.84,
            "result_keyword": 0.83,
            "bracketed_numbers": 0.82,
            "row_by_row": 0.80,
            "comma_separated": 0.75,
            "space_separated": 0.70,
            "and_separated": 0.68,
            "fallback_sequence": 0.30
        }
        
        confidence = base_confidences.get(format_name, 0.60)
        
        # Adjust based on validation
        validation_errors = self._validate_positions(positions, n)
        if not validation_errors:
            confidence += 0.05
        else:
            confidence *= (1.0 - len(validation_errors) * 0.1)
        
        # Adjust based on context
        text_lower = text.lower()
        if any(word in text_lower for word in ['queens', 'solution', 'answer']):
            confidence += 0.03
        
        return min(1.0, max(0.0, confidence))
    
    def _validate_positions(self, positions: List[int], n: int) -> List[str]:
        """Validate queen positions and return error list"""
        errors = []
        
        if not positions:
            errors.append("No positions provided")
            return errors
        
        if len(positions) != n:
            errors.append(f"Wrong number of positions: {len(positions)}, expected {n}")
        
        for i, pos in enumerate(positions):
            if not (0 <= pos < n):
                errors.append(f"Position {pos} at row {i} is out of bounds [0, {n-1}]")
        
        # Check for conflicts
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i] == positions[j]:
                    errors.append(f"Column conflict: rows {i} and {j} both have queen in column {positions[i]}")
                elif abs(i - j) == abs(positions[i] - positions[j]):
                    errors.append(f"Diagonal conflict: queens at ({i},{positions[i]}) and ({j},{positions[j]})")
        
        return errors

class NQueensSolver:
    """Enhanced N-Queens solver with detailed validation"""
    
    @staticmethod
    def is_safe(board: List[int], row: int, col: int) -> bool:
        """Check if placing a queen at (row, col) is safe"""
        for i in range(row):
            if (board[i] == col or 
                board[i] - i == col - row or
                board[i] + i == col + row):
                return False
        return True
    
    @staticmethod
    def solve_nqueens(n: int) -> List[List[int]]:
        """Find all solutions to N-Queens problem"""
        solutions = []
        
        def backtrack(board: List[int], row: int):
            if row == n:
                solutions.append(board[:])
                return
            
            for col in range(n):
                if NQueensSolver.is_safe(board, row, col):
                    board[row] = col
                    backtrack(board, row + 1)
                    board[row] = -1
        
        board = [-1] * n
        backtrack(board, 0)
        return solutions
    
    @staticmethod
    def validate_solution_detailed(solution: List[int], n: int) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced validation with detailed diagnostics"""
        diagnostics = {
            'n': n,
            'solution_length': len(solution) if solution else 0,
            'conflicts': [],
            'out_of_bounds': [],
            'total_conflicts': 0
        }
        
        if not solution:
            return False, "No solution provided", diagnostics
        
        if len(solution) != n:
            return False, f"Wrong number of queens: {len(solution)}, expected {n}", diagnostics
        
        # Check bounds
        for i, col in enumerate(solution):
            if not (0 <= col < n):
                diagnostics['out_of_bounds'].append((i, col))
        
        if diagnostics['out_of_bounds']:
            return False, f"Queens out of bounds: {diagnostics['out_of_bounds']}", diagnostics
        
        # Check conflicts
        for i in range(n):
            for j in range(i + 1, n):
                if solution[i] == solution[j]:
                    diagnostics['conflicts'].append(f"Column conflict: rows {i},{j} both in column {solution[i]}")
                elif abs(i - j) == abs(solution[i] - solution[j]):
                    diagnostics['conflicts'].append(f"Diagonal conflict: ({i},{solution[i]}) and ({j},{solution[j]})")
        
        diagnostics['total_conflicts'] = len(diagnostics['conflicts'])
        
        if diagnostics['total_conflicts'] > 0:
            return False, f"Found {diagnostics['total_conflicts']} conflicts", diagnostics
        
        return True, "Valid N-Queens solution", diagnostics

class NQueensTask(BaseTask):
    """N-Queens task that properly inherits from BaseTask for consistent evaluation"""
    
    def __init__(self, model_handler, output_dir, board_sizes, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        self._task_name = "n_queens_results"
        super().__init__(model_handler, output_dir, min(board_sizes), max(board_sizes), num_folds, num_samples, 
                        store_details, temperature, top_p, max_tokens, seed)
        self.board_sizes = board_sizes
        self.parser = NQueensResponseParser()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, list_size=None, **kwargs):
        """Generate evaluation data
        
        Args:
            list_size: Board size (4, 5, 6, 8, etc.)
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        board_size = list_size if list_size is not None else random.choice(self.board_sizes)
        data = []
        for _ in range(self.num_samples):
            problem = self.generate_problem(board_size)
            if problem['has_solution']:  # Only include solvable problems
                data.append(problem)
        
        return data
    
    def evaluate_response(self, response, data_point):
        """Evaluate the N-Queens solution"""
        try:
            # Use the robust parser to extract solution
            parsing_result = self.parser.parse_response(response, data_point['n'])
            
            # Check if we successfully extracted a solution (even if invalid)
            if parsing_result.solution is not None:
                predicted_solution = parsing_result.solution
                reference_solutions = data_point.get('all_solutions', [])
                
                # Check if solution is correct
                is_valid, validation_msg, validation_diagnostics = NQueensSolver.validate_solution_detailed(
                    predicted_solution, data_point['n']
                )
                
                # Check if it matches any known solution
                matches_reference = predicted_solution in reference_solutions if reference_solutions else False
                is_correct = is_valid and (matches_reference or len(reference_solutions) == 0)
                
                # Check instruction following (proper format) - be more lenient
                instruction_followed = parsing_result.confidence > 0.5
                
                return {
                    'accuracy': 1 if is_correct else 0,
                    'instruction_followed': 1 if instruction_followed else 0,
                    'predicted_answer': predicted_solution,  # Always include what was parsed
                    'ground_truth': data_point.get('reference_solution', []),
                    'parsing_confidence': parsing_result.confidence,
                    'parsing_diagnostics': parsing_result.diagnostics,
                    'validation_diagnostics': validation_diagnostics,
                    'is_valid_placement': is_valid,
                    'matches_reference': matches_reference,
                    'parsing_result_type': parsing_result.result_type.value
                }
            else:
                # Parsing completely failed
                return {
                    'accuracy': 0,
                    'instruction_followed': 0,
                    'predicted_answer': None,
                    'ground_truth': data_point.get('reference_solution', []),
                    'parsing_confidence': parsing_result.confidence,
                    'parsing_diagnostics': parsing_result.diagnostics,
                    'parsing_error': f"Failed to parse: {parsing_result.result_type.value}",
                    'raw_response': response[:200] + '...' if len(response) > 200 else response
                }
                
        except Exception as e:
            return {
                'accuracy': 0,
                'instruction_followed': 0,
                'predicted_answer': None,
                'ground_truth': data_point.get('reference_solution', []),
                'evaluation_error': str(e)
            }
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for prompt"""
        return len(prompt) // 4
    
    def generate_problem(self, n: int) -> Dict[str, Any]:
        """Generate an N-Queens problem"""
        solutions = NQueensSolver.solve_nqueens(n)
        total_solutions = len(solutions)
        reference_solution = solutions[0] if solutions else None
        
        return {
            'n': n,
            'total_solutions': total_solutions,
            'reference_solution': reference_solution,
            'has_solution': reference_solution is not None,
            'all_solutions': solutions[:10] if len(solutions) > 10 else solutions,  # Store up to 10 for validation
            'answer': reference_solution  # For BaseTask compatibility
        }
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create enhanced prompt with multiple format examples"""
        n = problem['n']
        
        prompt = f"""Solve the {n}-Queens problem: Place {n} queens on a {n}×{n} chessboard so that no two queens attack each other.

RULES:
- Queens attack horizontally, vertically, and diagonally
- No two queens can be in the same row, column, or diagonal
- You must place exactly {n} queens on the board (one queen per row)

CRITICAL: Your answer must contain exactly {n} numbers, each representing the column position (0 to {n-1}) of the queen in that row.

SOLUTION FORMAT:
Provide your solution as a list of {n} column numbers. The first number is the column for row 0, the second for row 1, etc.

REQUIRED FORMAT - Use one of these:
- [col0, col1, col2, ...] ← PREFERRED
- Answer: [1, 3, 0, 2]
- Solution: [1, 3, 0, 2]

EXAMPLE for 4-Queens:
If queens are at positions (row 0, col 1), (row 1, col 3), (row 2, col 0), (row 3, col 2):
Answer: [1, 3, 0, 2]

IMPORTANT:
- Your answer must have exactly {n} numbers
- Each number must be between 0 and {n-1}
- Use the format: [num1, num2, num3, ...]

<answer>
[your {n} column numbers here]
</answer>

Think step by step to find a valid configuration where no queens attack each other."""
        
        return prompt
    
    
    def format_board_visualization(self, solution: List[int], n: int) -> str:
        """Create visual representation of the board"""
        if not solution or len(solution) != n:
            return "Invalid solution"
        
        board = []
        for row in range(n):
            line = []
            for col in range(n):
                if col == solution[row]:
                    line.append('Q')
                else:
                    line.append('.')
            board.append(' '.join(line))
        
        return '\n'.join(board)


def main():
    """Main evaluation function with proper logging and reporting"""
    parser = argparse.ArgumentParser(description="N-Queens Problem Solving Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Engine to use')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration  
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of data points per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--board_sizes', type=str, default=','.join(map(str, BOARD_SIZES)), help='Comma-separated list of board sizes')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='n_queens_results', help='Output directory')

    # API configuration
    parser.add_argument('--api_provider', type=str, choices=['openai', 'gemini'], help='API provider for cloud models')
    parser.add_argument('--api_key', type=str, help='API key for cloud models')

    # Reasoning configuration
    parser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'],
                       help='OpenAI reasoning effort level (for GPT-5 models)')
    parser.add_argument('--thinking_budget', type=int,
                       help='Gemini thinking budget (0 to disable, -1 for dynamic, or specific number)')
    
    args = parser.parse_args()
    
    # Parse board sizes
    board_sizes = [int(x.strip()) for x in args.board_sizes.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("♛ Starting N-Queens Problem Solving Evaluation")
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
        logging.info(f"🎯 Initializing N-Queens task")
        task = NQueensTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            board_sizes=board_sizes,
            num_folds=args.folds,
            num_samples=args.datapoints,
            store_details=args.store_details,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed
        )
        
        # Set random seeds
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            logging.info(f"🎲 Set random seed to {args.seed}")
        
        # Run evaluation
        logging.info(f"🚀 Starting evaluation with board sizes: {board_sizes}")
        all_metrics = task.run_evaluation(list_sizes=board_sizes)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(args.output_dir, 'n_queens', board_sizes)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, board_sizes, args.output_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, board_size in enumerate(board_sizes):
                # Find metrics for this board size from all_metrics
                size_metrics = [m for m in all_metrics if m.get('board_size') == board_size or
                               (i < len(all_metrics) and all_metrics[i] is m)]

                if not size_metrics:
                    # Create a default metric for this board size
                    size_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                   'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"n_queens_{board_size}",
                    'test_case': i,
                    'complexity': board_size,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in size_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in size_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in size_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in size_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in size_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, board_sizes, args.output_dir)
        
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
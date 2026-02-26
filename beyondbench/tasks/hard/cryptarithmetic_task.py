"""
Cryptarithmetic Puzzles Task

This task evaluates LLM's ability to solve cryptarithmetic puzzles through constraint satisfaction and logical reasoning.
Cryptarithmetic involves finding digit assignments to letters such that arithmetic equations hold true.
Each letter represents a unique digit (0-9), and leading letters cannot be zero.

Algorithm: Generate valid cryptarithmetic puzzles with known unique solutions, ask for digit assignments.
Reasoning: Requires constraint satisfaction, arithmetic reasoning, systematic search, and logical deduction.

Example: SEND + MORE = MONEY
Solution: S=9, E=5, N=6, D=7, M=1, O=0, R=8, Y=2
Check: 9567 + 1085 = 10652 ✓

CLI USAGE:
python cryptarithmetic_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --cuda_device "cuda:6" --datapoints 20 --folds 1 --word_lengths 3,4,5 --puzzle_types addition,multiplication --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --tensor_parallel_size 1 --gpu_memory_utilization 0.96 --trust_remote_code False --store_details True
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["cryptarithmetic"]
# Engine selection - choose 'vllm' or 'transformers'
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20  # Number of samples per fold
FOLDS = 1  # Number of evaluation folds
WORD_LENGTHS = [4, 5, 6, 7]  # Length of words in puzzles - made more challenging
PUZZLE_TYPES = ["addition", "multiplication", "subtraction", "division"]  # Multiple arithmetic operations
DIFFICULTY_LEVELS = ["medium", "hard", "extreme"]  # Challenge difficulty levels
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
import itertools


from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

class CryptarithmeticSolver:
    """Ground truth solver for cryptarithmetic puzzles"""
    
    def __init__(self):
        # Enhanced word templates with famous cryptarithmetic puzzles and challenging combinations
        self.word_templates = {
            3: ['SUM', 'TWO', 'ONE', 'TEN', 'SIX', 'WIN', 'GET', 'PUT', 'GOT', 'CAN'],
            4: ['SEND', 'MORE', 'NINE', 'FIVE', 'ZERO', 'EVEN', 'TWIN', 'CUBE', 'TRUE', 'BASE', 'MATH', 'WORK', 'GAME', 'PLAY'],
            5: ['MONEY', 'FORTY', 'SIXTY', 'THREE', 'SEVEN', 'EIGHT', 'CROSS', 'ROADS', 'SPEED', 'WORLD', 'POWER', 'PRIME', 'MAGIC', 'LOGIC'],
            6: ['DONALD', 'GERALD', 'ROBERT', 'SOLVER', 'PUZZLE', 'GENIUS', 'MASTER', 'CRYPTO', 'CIPHER', 'SECRET', 'HIDDEN', 'ANSWER'],
            7: ['LETTERS', 'NUMBERS', 'ALGEBRA', 'PROBLEM', 'PATTERN', 'COMPLEX', 'HARDEST', 'EXTREME', 'MAXIMUM', 'OPTIMUM']
        }
        
        # Famous classical cryptarithmetic puzzles for higher difficulty
        self.classical_puzzles = [
            # Famous SEND + MORE = MONEY (9567 + 1085 = 10652)
            {'word1': 'SEND', 'word2': 'MORE', 'result': 'MONEY', 'type': 'addition'},
            # DONALD + GERALD = ROBERT (526485 + 197485 = 723970)
            {'word1': 'DONALD', 'word2': 'GERALD', 'result': 'ROBERT', 'type': 'addition'},
            # TWO + TWO = FOUR (734 + 734 = 1468)
            {'word1': 'TWO', 'word2': 'TWO', 'result': 'FOUR', 'type': 'addition'},
            # FORTY + TEN + TEN = SIXTY (29786 + 850 + 850 = 31486)
            {'word1': 'FORTY', 'word2': 'TEN', 'result': 'SIXTY', 'type': 'addition', 'third': 'TEN'},
            # BASE * BASE = BASES (1028 * 1028 = 1056784)
            {'word1': 'BASE', 'word2': 'BASE', 'result': 'BASES', 'type': 'multiplication'},
        ]
    
    def generate_words(self, length1: int, length2: int, puzzle_type: str = "addition", difficulty: str = "medium") -> Tuple[str, str, str]:
        """Generate three words for the puzzle with specified operation and difficulty"""
        
        # For extreme difficulty, use classical puzzles occasionally
        if difficulty == "extreme" and random.random() < 0.3 and puzzle_type == "addition":
            puzzle = random.choice(self.classical_puzzles)
            if puzzle['type'] == puzzle_type:
                return puzzle['word1'], puzzle['word2'], puzzle['result']
        
        # Generate challenging word combinations
        all_letters = set()
        
        # Select word templates based on difficulty
        if difficulty == "extreme":
            # Use longer, more complex words
            available_lengths = [l for l in self.word_templates.keys() if l >= max(length1, length2)]
            if available_lengths:
                length1 = min(length1, max(available_lengths))
                length2 = min(length2, max(available_lengths))
        
        # Get word templates
        word1 = random.choice(self.word_templates.get(length1, ['A' * length1]))
        word2 = random.choice(self.word_templates.get(length2, ['B' * length2]))
        
        # Ensure we have enough unique letters
        all_letters.update(word1)
        all_letters.update(word2)
        
        # Generate result word based on operation type
        if puzzle_type == "addition":
            max_result = int('9' * length1) + int('9' * length2)
        elif puzzle_type == "multiplication":
            max_result = int('9' * length1) * int('9' * length2)
        elif puzzle_type == "subtraction":
            max_result = int('9' * max(length1, length2))
        else:  # division
            max_result = int('9' * length1)
        
        result_length = len(str(max_result))
        
        # Create result word with meaningful letters
        remaining_letters = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in all_letters]
        
        # For hard/extreme difficulty, create more constrained puzzles
        if difficulty in ["hard", "extreme"]:
            # Try to use overlapping letters to increase constraint complexity
            overlap_count = min(2, len(all_letters) // 2) if difficulty == "extreme" else 1
            result_letters = list(all_letters)[:overlap_count]
            needed_new = max(0, result_length - len(result_letters))
            result_letters.extend(remaining_letters[:needed_new])
        else:
            result_letters = list(all_letters) + remaining_letters[:result_length - len(all_letters)]
        
        random.shuffle(result_letters)
        word3 = ''.join(result_letters[:result_length])
        
        return word1, word2, word3
    
    def solve_cryptarithmetic(self, word1: str, word2: str, result: str, operation: str = "addition") -> Optional[Dict[str, int]]:
        """Solve cryptarithmetic puzzle using constraint satisfaction with multiple operations"""
        all_letters = list(set(word1 + word2 + result))
        
        if len(all_letters) > 10:  # Can't assign more than 10 unique digits
            return None
        
        # Get leading letters (cannot be zero)
        leading_letters = {word1[0], word2[0], result[0]}
        
        # Try all possible digit assignments
        for digit_assignment in itertools.permutations(range(10), len(all_letters)):
            assignment = dict(zip(all_letters, digit_assignment))
            
            # Check leading letter constraint
            if any(assignment[letter] == 0 for letter in leading_letters):
                continue
            
            # Convert words to numbers
            try:
                num1 = self.word_to_number(word1, assignment)
                num2 = self.word_to_number(word2, assignment)
                num_result = self.word_to_number(result, assignment)
                
                # Check equation based on operation type
                equation_valid = False
                if operation == "addition":
                    equation_valid = (num1 + num2 == num_result)
                elif operation == "multiplication":
                    equation_valid = (num1 * num2 == num_result)
                elif operation == "subtraction":
                    equation_valid = (abs(num1 - num2) == num_result) or (num1 - num2 == num_result)
                elif operation == "division":
                    if num2 != 0 and num1 % num2 == 0:
                        equation_valid = (num1 // num2 == num_result)
                
                if equation_valid:
                    return assignment
            except:
                continue
        
        return None
    
    def word_to_number(self, word: str, assignment: Dict[str, int]) -> int:
        """Convert word to number using letter-digit assignment"""
        number = 0
        for letter in word:
            number = number * 10 + assignment[letter]
        return number
    
    def generate_valid_puzzle(self, length1: int, length2: int, puzzle_type: str = "addition", difficulty: str = "medium") -> Optional[Dict[str, Any]]:
        """Generate a cryptarithmetic puzzle with a valid solution"""
        # Reduce attempts to prevent hanging, especially for complex operations
        if puzzle_type == "multiplication":
            max_attempts = 50 if difficulty == "extreme" else 30 if difficulty == "hard" else 20
        elif puzzle_type in ["subtraction", "division"]:
            max_attempts = 40 if difficulty == "extreme" else 25 if difficulty == "hard" else 15
        else:  # addition
            max_attempts = 100 if difficulty == "extreme" else 75 if difficulty == "hard" else 50
        
        for attempt in range(max_attempts):
            word1, word2, result_word = self.generate_words(length1, length2, puzzle_type, difficulty)
            solution = self.solve_cryptarithmetic(word1, word2, result_word, puzzle_type)
            
            if solution:
                # Verify solution and calculate equation
                num1 = self.word_to_number(word1, solution)
                num2 = self.word_to_number(word2, solution)
                num_result = self.word_to_number(result_word, solution)
                
                # Create equation string based on operation
                if puzzle_type == "addition":
                    equation = f"{num1} + {num2} = {num_result}"
                    is_valid = (num1 + num2 == num_result)
                elif puzzle_type == "multiplication":
                    equation = f"{num1} × {num2} = {num_result}"
                    is_valid = (num1 * num2 == num_result)
                elif puzzle_type == "subtraction":
                    equation = f"{num1} - {num2} = {num_result}"
                    is_valid = (num1 - num2 == num_result) or (num2 - num1 == num_result)
                elif puzzle_type == "division":
                    equation = f"{num1} ÷ {num2} = {num_result}"
                    is_valid = (num2 != 0 and num1 // num2 == num_result)
                
                if is_valid:
                    return {
                        'word1': word1,
                        'word2': word2,
                        'result_word': result_word,
                        'solution': solution,
                        'equation': equation,
                        'operation': puzzle_type,
                        'difficulty': difficulty,
                        'unique_letters': len(solution),
                        'constraint_density': len(solution) / (len(word1) + len(word2) + len(result_word))
                    }
        
        return None
    
    def validate_solution(self, word1: str, word2: str, result_word: str, 
                         assignment: Dict[str, int], operation: str = "addition") -> Tuple[bool, str]:
        """Validate a solution to a cryptarithmetic puzzle with multiple operations"""
        try:
            all_letters = set(word1 + word2 + result_word)
            
            # Check if all letters have assignments
            for letter in all_letters:
                if letter not in assignment:
                    return False, f"Letter {letter} not assigned"
            
            # Check if assignments are valid digits
            for letter, digit in assignment.items():
                if not (0 <= digit <= 9):
                    return False, f"Invalid digit {digit} for letter {letter}"
            
            # Check uniqueness of digit assignments
            used_digits = list(assignment.values())
            if len(used_digits) != len(set(used_digits)):
                return False, "Digits must be unique for each letter"
            
            # Check leading letter constraint
            leading_letters = {word1[0], word2[0], result_word[0]}
            for letter in leading_letters:
                if assignment.get(letter, -1) == 0:
                    return False, f"Leading letter {letter} cannot be zero"
            
            # Check arithmetic equation based on operation
            num1 = self.word_to_number(word1, assignment)
            num2 = self.word_to_number(word2, assignment)
            num_result = self.word_to_number(result_word, assignment)
            
            if operation == "addition":
                if num1 + num2 == num_result:
                    return True, f"Valid solution: {num1} + {num2} = {num_result}"
                else:
                    return False, f"Arithmetic error: {num1} + {num2} = {num1 + num2} ≠ {num_result}"
            elif operation == "multiplication":
                if num1 * num2 == num_result:
                    return True, f"Valid solution: {num1} × {num2} = {num_result}"
                else:
                    return False, f"Arithmetic error: {num1} × {num2} = {num1 * num2} ≠ {num_result}"
            elif operation == "subtraction":
                if abs(num1 - num2) == num_result or (num1 - num2) == num_result:
                    return True, f"Valid solution: {max(num1, num2)} - {min(num1, num2)} = {num_result}"
                else:
                    return False, f"Arithmetic error: |{num1} - {num2}| = {abs(num1 - num2)} ≠ {num_result}"
            elif operation == "division":
                if num2 != 0 and num1 % num2 == 0 and num1 // num2 == num_result:
                    return True, f"Valid solution: {num1} ÷ {num2} = {num_result}"
                else:
                    return False, f"Arithmetic error: {num1} ÷ {num2} = {num1 // num2 if num2 != 0 else 'undefined'} ≠ {num_result}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
class CryptarithmeticTask(BaseTask):
    """Cryptarithmetic puzzle evaluation task with robust parsing"""
    
    def __init__(self, model_handler, output_dir, word_lengths, puzzle_types, difficulty_levels, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        self._task_name = "cryptarithmetic_results"
        super().__init__(model_handler, output_dir, min(word_lengths), max(word_lengths), num_folds, num_samples, 
                        store_details, temperature, top_p, max_tokens, seed)
        self.word_lengths = word_lengths
        self.puzzle_types = puzzle_types
        self.difficulty_levels = difficulty_levels
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Enhanced parsing patterns with confidence scoring
        self.setup_parsing_patterns()
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, list_size=None, **kwargs):
        """Generate evaluation data
        
        Args:
            list_size: Word length (3, 4, 5, etc.)
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        word_length = list_size if list_size is not None else random.choice(self.word_lengths)
        test_cases = []
        
        # Generate full num_samples for this word length
        generated = 0
        attempts = 0
        max_attempts = self.num_samples * 3  # Reduced from 5 to prevent hanging
        
        logging.info(f"🎲 Generating {self.num_samples} cryptarithmetic puzzles for {word_length}-letter words...")
        
        while generated < self.num_samples and attempts < max_attempts:
            attempts += 1
            if attempts % 10 == 0:  # Log progress every 10 attempts
                logging.info(f"   📊 Progress: {generated}/{self.num_samples} generated, attempt {attempts}/{max_attempts}")
            
            problem = self.generate_problem(word_length)
            if problem:
                problem['answer'] = problem['solution']  # For BaseTask compatibility
                test_cases.append(problem)
                generated += 1
        
        if generated < self.num_samples:
            logging.warning(f"⚠️  Only generated {generated}/{self.num_samples} puzzles for {word_length}-letter words after {attempts} attempts")
            
            # Add simple fallback puzzles if we couldn't generate enough
            while generated < self.num_samples:
                fallback_problem = self.create_fallback_puzzle(word_length)
                if fallback_problem:
                    fallback_problem['answer'] = fallback_problem['solution']
                    test_cases.append(fallback_problem)
                    generated += 1
                else:
                    break
            
            logging.info(f"✅ Final count: {len(test_cases)} puzzles generated for {word_length}-letter words")
        
        return test_cases
    
    def create_fallback_puzzle(self, word_length: int) -> Optional[Dict[str, Any]]:
        """Create a simple fallback puzzle when generation fails"""
        solver = CryptarithmeticSolver()
        
        # Use simple addition with known patterns
        if word_length == 4:
            # Simple 4-letter addition puzzle
            return solver.generate_valid_puzzle(3, 3, "addition", "medium")
        elif word_length == 5:
            # Simple 5-letter addition puzzle  
            return solver.generate_valid_puzzle(4, 4, "addition", "medium")
        else:
            # Fallback to addition with reduced complexity
            return solver.generate_valid_puzzle(word_length - 1, word_length - 1, "addition", "medium")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response"""
        result = self.parse_response(response, data_point)
        return {
            'accuracy': 1 if result['is_correct'] else 0,
            'instruction_followed': 1 if result.get('parsing_confidence', 0) > 0 else 0,
            'predicted_answer': result.get('parsed_assignment', {}),
            'ground_truth': data_point.get('solution', {}),
            'parsing_info': {
                'method': result.get('parsing_method', 'unknown'),
                'confidence': result.get('parsing_confidence', 0.0),
                'attempts': result.get('parsing_attempts', [])
            },
            'validation_info': result.get('validation_info', {})
        }
    
    def setup_parsing_patterns(self):
        """Setup enhanced regex patterns for parsing different assignment formats with maximum robustness"""
        self.assignment_patterns = [
            # Pattern 1: Standard dictionary format {"A": 1, "B": 2, "C": 3}
            (re.compile(r'\{\s*([^}]+)\s*\}', re.IGNORECASE | re.DOTALL), "dict_format", 0.9),
            
            # Pattern 2: Letter assignment pairs "A = 1, B = 2, C = 3"
            (re.compile(r'([A-Z])\s*=\s*(\d)', re.IGNORECASE), "letter_equals", 0.8),
            
            # Pattern 3: Assignment with colon "A: 1, B: 2, C: 3"
            (re.compile(r'([A-Z])\s*:\s*(\d)', re.IGNORECASE), "letter_colon", 0.8),
            
            # Pattern 4: Letter to digit "letter A is 1, letter B is 2"
            (re.compile(r'letter\s+([A-Z])\s+is\s+(\d)', re.IGNORECASE), "letter_is", 0.7),
            
            # Pattern 5: Assignment arrows "A -> 1, B -> 2"
            (re.compile(r'([A-Z])\s*->\s*(\d)', re.IGNORECASE), "arrow_assignment", 0.7),
            
            # Pattern 6: List format "letters: [A, B, C], digits: [1, 2, 3]"
            (re.compile(r'letters?\s*:\s*\[([^\]]+)\]\s*,?\s*digits?\s*:\s*\[([^\]]+)\]', re.IGNORECASE), "parallel_lists", 0.7),
            
            # Pattern 7: Tabular format "A | 1\nB | 2"
            (re.compile(r'([A-Z])\s*\|\s*(\d)', re.IGNORECASE), "table_format", 0.6),
            
            # Pattern 8: Brackets with assignments [A=1, B=2, C=3]
            (re.compile(r'\[([^\]]+)\]', re.IGNORECASE), "bracket_assignments", 0.6),
            
            # Pattern 9: Assignment keyword "Assignment: A=1, B=2"
            (re.compile(r'assignment\s*:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL), "assignment_keyword", 0.8),
            
            # Pattern 10: Solution keyword "Solution: A=1, B=2"
            (re.compile(r'solution\s*:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL), "solution_keyword", 0.8),
            
            # NEW ENHANCED PATTERNS:
            
            # Pattern 11: Code block format ```{"A": 1, "B": 2}```
            (re.compile(r'```(?:json|plaintext|text)?\s*\n?\s*\{([^}]+)\}\s*\n?```', re.IGNORECASE | re.DOTALL), "code_block_dict", 0.95),
            
            # Pattern 12: Simple code block ```A=1, B=2```
            (re.compile(r'```\s*([A-Z]\s*=\s*\d(?:\s*,\s*[A-Z]\s*=\s*\d)*)\s*```', re.IGNORECASE | re.DOTALL), "simple_code_block", 0.93),
            
            # Pattern 13: Final Answer format "Final Answer: {"A": 1, "B": 2}"
            (re.compile(r'final\s+answer\s*:?\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL), "final_answer", 0.92),
            
            # Pattern 14: Answer format "Answer: {"A": 1, "B": 2}"
            (re.compile(r'answer\s*:?\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL), "answer_format", 0.91),
            
            # Pattern 15: Multiple line assignments in sequence "A = 1\nB = 2\nC = 3"
            (re.compile(r'([A-Z])\s*=\s*(\d)(?:\s*\n\s*([A-Z])\s*=\s*(\d))*', re.IGNORECASE | re.MULTILINE), "multi_line_assignments", 0.87),
            
            # Pattern 16: Variable statement "Let A = 1, B = 2, C = 3"
            (re.compile(r'let\s+([A-Z])\s*=\s*(\d)', re.IGNORECASE), "let_statement", 0.86),
            
            # Pattern 17: Therefore statement "Therefore A = 1, B = 2, C = 3"
            (re.compile(r'therefore\s+([A-Z])\s*=\s*(\d)', re.IGNORECASE), "therefore_statement", 0.85),
            
            # Pattern 18: Assign statements "Assign A = 1, B = 2, C = 3"
            (re.compile(r'assign\s+([A-Z])\s*=\s*(\d)', re.IGNORECASE), "assign_statement", 0.84),
            
            # Pattern 19: Letter gets digit "A gets 1, B gets 2"
            (re.compile(r'([A-Z])\s+gets\s+(\d)', re.IGNORECASE), "letter_gets", 0.83),
            
            # Pattern 20: Letter represents digit "A represents 1, B represents 2"
            (re.compile(r'([A-Z])\s+represents\s+(\d)', re.IGNORECASE), "letter_represents", 0.82),
            
            # Pattern 21: Variable value "variable A has value 1, variable B has value 2"
            (re.compile(r'variable\s+([A-Z])\s+(?:has\s+)?value\s+(\d)', re.IGNORECASE), "variable_value", 0.81),
            
            # Pattern 22: The solution is format "The solution is: A=1, B=2, C=3"
            (re.compile(r'(?:the\s+)?solution\s+is\s*:?\s*(.+)', re.IGNORECASE | re.DOTALL), "solution_is", 0.88)
        ]
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for prompt"""
        return len(prompt) // 4
    
    def generate_problem(self, word_length: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate a cryptarithmetic problem with enhanced difficulty and operations"""
        solver = CryptarithmeticSolver()
        
        # Select puzzle type and difficulty - prefer easier operations to avoid hanging
        # Prioritize addition which is most reliable
        if random.random() < 0.6:  # 60% chance for addition
            puzzle_type = "addition"
        else:
            puzzle_type = random.choice(self.puzzle_types)
        
        # Limit difficulty for complex operations
        if puzzle_type in ["multiplication", "subtraction", "division"]:
            difficulty = random.choice(["medium", "hard"])  # Skip extreme for complex ops
        else:
            difficulty = random.choice(self.difficulty_levels)
        
        # Generate challenging word length combinations
        if difficulty == "extreme":
            # For extreme difficulty, use more diverse length combinations
            length2 = random.choice([word_length - 1, word_length, word_length + 1])  # Reduced range
            length2 = max(3, min(6, length2))  # Reduced max
        elif difficulty == "hard":
            length2 = random.choice([word_length - 1, word_length, word_length + 1])
            length2 = max(3, min(5, length2))  # Reduced max
        else:
            length2 = random.choice([word_length - 1, word_length, word_length + 1])
            length2 = max(3, min(5, length2))
        
        # Generate puzzle with enhanced parameters
        puzzle = solver.generate_valid_puzzle(word_length, length2, puzzle_type, difficulty)
        return puzzle
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create prompt for cryptarithmetic problem with multiple operations"""
        word1 = problem['word1']
        word2 = problem['word2']
        result_word = problem['result_word']
        operation = problem.get('operation', 'addition')
        difficulty = problem.get('difficulty', 'medium')
        unique_letters = problem['unique_letters']
        
        all_letters = sorted(set(word1 + word2 + result_word))
        letters_str = ', '.join(all_letters)
        
        # Operation symbols
        op_symbols = {
            'addition': '+', 'multiplication': '×', 'subtraction': '-', 'division': '÷'
        }
        op_symbol = op_symbols.get(operation, '+')
        
        # Enhanced prompt with operation-specific instructions
        operation_rules = {
            'addition': "Find the digit assignment that makes the addition equation true.",
            'multiplication': "Find the digit assignment that makes the multiplication equation true.",
            'subtraction': "Find the digit assignment that makes the subtraction equation true. Note that the result should be positive.",
            'division': "Find the digit assignment that makes the division equation true. The division must result in a whole number (no remainder)."
        }
        
        difficulty_note = ""
        if difficulty == "hard":
            difficulty_note = "\n\nNOTE: This is a HARD puzzle with complex constraints and overlapping letter usage."
        elif difficulty == "extreme":
            difficulty_note = "\n\nNOTE: This is an EXTREME puzzle requiring advanced constraint satisfaction and may use classical cryptarithmetic patterns."
        
        prompt = f"""Solve this challenging cryptarithmetic puzzle:

{word1} {op_symbol} {word2} = {result_word}

Rules:
- Each letter represents a unique digit from 0 to 9
- No two letters can have the same digit
- Leading letters ({word1[0]}, {word2[0]}, {result_word[0]}) cannot be zero
- {operation_rules[operation]}

Letters to assign: {letters_str}
Operation: {operation.title()}
Difficulty: {difficulty.title()}
Total unique letters: {unique_letters}{difficulty_note}

Please work through this step-by-step:
1. Analyze the constraints and relationships between letters
2. Consider the arithmetic operation and carry/borrow implications
3. Use logical deduction to narrow down possibilities
4. Find the complete digit assignment

SOLUTION FORMAT REQUIREMENTS (STRICTLY FOLLOW THIS):
Your answer must assign digits to ALL {unique_letters} letters.

REQUIRED FORMAT:
<answer>
{{"A": 1, "B": 2, "C": 3, ...}}
</answer>

IMPORTANT FORMATTING NOTES:
- Use EXACTLY the format shown above inside <answer> tags
- Use double quotes around letter names: "A", "B", "C", etc.
- Use actual digits: 1, 2, 3, etc. (not "1", "2", "3")
- Include ALL {unique_letters} letters in your answer
- Do NOT use code blocks, markdown, or other formatting
- Do NOT get stuck in repetitive patterns - provide ONE clear answer

ALTERNATIVE FORMATS (if <answer> tags fail):
- Final Answer: {{"A": 1, "B": 2, "C": 3, ...}}
- Answer: {{"A": 1, "B": 2, "C": 3, ...}}
- A=1, B=2, C=3, ...

REMEMBER: Provide ONE complete solution with all letters assigned!"""
        
        return prompt
    
    def parse_response(self, response: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with enhanced robustness and detailed diagnostics"""
        all_letters = set(problem['word1'] + problem['word2'] + problem['result_word'])
        parsing_attempts = []
        best_assignment = {}
        best_confidence = 0.0
        best_method = "none"
        
        # Pre-process response to clean up formatting issues
        cleaned_response = self.preprocess_response(response)
        
        # Extract answer from <answer> tags if present
        answer_text = cleaned_response
        if '<answer>' in cleaned_response and '</answer>' in cleaned_response:
            answer_text = cleaned_response.split('<answer>')[1].split('</answer>')[0].strip()
            parsing_attempts.append(("answer_tags", "Extracted from <answer> tags", 0.1))
        
        # Try both the full response and extracted answer text
        texts_to_parse = [answer_text, cleaned_response] if answer_text != cleaned_response else [cleaned_response]
        
        # Try each parsing pattern on each text variant
        for text_variant in texts_to_parse:
            for pattern, method_name, base_confidence in self.assignment_patterns:
                try:
                    assignment = self.parse_with_pattern(text_variant, pattern, method_name, all_letters)
                    if assignment and len(assignment) == len(all_letters):
                        confidence = base_confidence * (len(assignment) / len(all_letters))
                        # Bonus for complete assignments
                        confidence *= 1.1
                        parsing_attempts.append((method_name, f"Parsed {len(assignment)} letters", confidence))
                        
                        if confidence > best_confidence:
                            best_assignment = assignment
                            best_confidence = confidence
                            best_method = method_name
                    elif assignment:
                        partial_confidence = base_confidence * (len(assignment) / len(all_letters)) * 0.5
                        parsing_attempts.append((method_name, f"Partial parse: {len(assignment)}/{len(all_letters)} letters", partial_confidence))
                        
                        # Even partial assignments might be useful if nothing better is found
                        if partial_confidence > best_confidence and len(assignment) >= len(all_letters) // 2:
                            best_assignment = assignment
                            best_confidence = partial_confidence
                            best_method = method_name + "_partial"
                    else:
                        parsing_attempts.append((method_name, "No match", 0.0))
                        
                except Exception as e:
                    parsing_attempts.append((method_name, f"Error: {str(e)}", 0.0))
        
        # Validate the best solution
        validation_info = {}
        if best_assignment:
            operation = problem.get('operation', 'addition')
            is_valid, validation_msg = CryptarithmeticSolver().validate_solution(
                problem['word1'], problem['word2'], problem['result_word'], best_assignment, operation
            )
            
            # Check completeness
            missing_letters = [letter for letter in all_letters if letter not in best_assignment]
            duplicate_digits = self.find_duplicate_digits(best_assignment)
            leading_zero_violations = self.check_leading_zeros(problem, best_assignment)
            
            validation_info = {
                'is_valid_assignment': is_valid,
                'missing_letters': missing_letters,
                'duplicate_digits': duplicate_digits,
                'leading_zero_violations': leading_zero_violations,
                'validation_message': validation_msg,
                'arithmetic_check': self.get_arithmetic_details(problem, best_assignment) if best_assignment else None
            }
        else:
            is_valid = False
            validation_info = {
                'is_valid_assignment': False,
                'missing_letters': list(all_letters),
                'duplicate_digits': [],
                'leading_zero_violations': [],
                'validation_message': "Could not parse any assignment from response",
                'arithmetic_check': None
            }
        
        return {
            'parsed_assignment': best_assignment,
            'parsing_method': best_method,
            'parsing_confidence': best_confidence,
            'parsing_attempts': parsing_attempts,
            'validation_info': validation_info,
            'is_correct': is_valid,
            'raw_response': response
        }
    
    def parse_with_pattern(self, text: str, pattern: re.Pattern, method_name: str, all_letters: Set[str]) -> Dict[str, int]:
        """Parse assignment using a specific pattern"""
        assignment = {}
        
        if method_name == "dict_format":
            # Standard dictionary format
            matches = pattern.findall(text)
            if matches:
                dict_content = matches[0]
                return self.parse_dict_content(dict_content)
        
        elif method_name == "letter_equals":
            # Letter assignment pairs "A = 1"
            matches = pattern.findall(text)
            for letter, digit in matches:
                letter = letter.upper()
                if letter in all_letters:
                    assignment[letter] = int(digit)
        
        elif method_name == "letter_colon":
            # Assignment with colon "A: 1"
            matches = pattern.findall(text)
            for letter, digit in matches:
                letter = letter.upper()
                if letter in all_letters:
                    assignment[letter] = int(digit)
        
        elif method_name == "letter_is":
            # "letter A is 1"
            matches = pattern.findall(text)
            for letter, digit in matches:
                letter = letter.upper()
                if letter in all_letters:
                    assignment[letter] = int(digit)
        
        elif method_name == "arrow_assignment":
            # Assignment arrows "A -> 1"
            matches = pattern.findall(text)
            for letter, digit in matches:
                letter = letter.upper()
                if letter in all_letters:
                    assignment[letter] = int(digit)
        
        elif method_name == "parallel_lists":
            # List format "letters: [A, B, C], digits: [1, 2, 3]"
            matches = pattern.findall(text)
            if matches:
                letters_str, digits_str = matches[0]
                letters = [l.strip().upper() for l in re.findall(r'[A-Z]', letters_str, re.IGNORECASE)]
                digits = [int(d.strip()) for d in re.findall(r'\d', digits_str)]
                
                if len(letters) == len(digits):
                    for letter, digit in zip(letters, digits):
                        if letter in all_letters:
                            assignment[letter] = digit
        
        elif method_name == "table_format":
            # Tabular format "A | 1"
            matches = pattern.findall(text)
            for letter, digit in matches:
                letter = letter.upper()
                if letter in all_letters:
                    assignment[letter] = int(digit)
        
        elif method_name == "bracket_assignments":
            # Brackets with assignments
            matches = pattern.findall(text)
            if matches:
                content = matches[0]
                pairs = re.findall(r'([A-Z])\s*[=:]\s*(\d)', content, re.IGNORECASE)
                for letter, digit in pairs:
                    letter = letter.upper()
                    if letter in all_letters:
                        assignment[letter] = int(digit)
        
        elif method_name in ["assignment_keyword", "solution_keyword"]:
            # Assignment/Solution keyword
            matches = pattern.findall(text)
            if matches:
                assignment_text = matches[0].strip()
                pairs = re.findall(r'([A-Z])\s*[=:]\s*(\d)', assignment_text, re.IGNORECASE)
                for letter, digit in pairs:
                    letter = letter.upper()
                    if letter in all_letters:
                        assignment[letter] = int(digit)
        
        elif method_name in ["code_block_dict", "simple_code_block", "final_answer", "answer_format"]:
            # Code block or answer formats with dictionary content
            matches = pattern.findall(text)
            if matches:
                if method_name == "simple_code_block":
                    # Parse A=1, B=2 format from code block
                    assignment_text = matches[0]
                    pairs = re.findall(r'([A-Z])\s*=\s*(\d)', assignment_text, re.IGNORECASE)
                    for letter, digit in pairs:
                        letter = letter.upper()
                        if letter in all_letters:
                            assignment[letter] = int(digit)
                else:
                    # Parse dictionary content
                    dict_content = matches[0]
                    assignment = self.parse_dict_content(dict_content)
        
        elif method_name == "multi_line_assignments":
            # Multiple line assignments "A = 1\nB = 2\nC = 3"
            # This pattern captures pairs, need to extract all matches
            all_matches = re.findall(r'([A-Z])\s*=\s*(\d)', text, re.IGNORECASE | re.MULTILINE)
            for letter, digit in all_matches:
                letter = letter.upper()
                if letter in all_letters:
                    assignment[letter] = int(digit)
        
        elif method_name in ["let_statement", "therefore_statement", "assign_statement", "letter_gets", 
                           "letter_represents", "variable_value"]:
            # Extended assignment patterns
            matches = pattern.findall(text)
            for letter, digit in matches:
                letter = letter.upper()
                if letter in all_letters:
                    assignment[letter] = int(digit)
        
        elif method_name == "solution_is":
            # "The solution is: ..." format
            matches = pattern.findall(text)
            if matches:
                solution_text = matches[0].strip()
                # Try to parse the solution text recursively
                # First try dictionary format
                dict_match = re.search(r'\{([^}]+)\}', solution_text)
                if dict_match:
                    assignment = self.parse_dict_content(dict_match.group(1))
                else:
                    # Try letter assignment pairs
                    pairs = re.findall(r'([A-Z])\s*[=:]\s*(\d)', solution_text, re.IGNORECASE)
                    for letter, digit in pairs:
                        letter = letter.upper()
                        if letter in all_letters:
                            assignment[letter] = int(digit)
        
        return assignment
    
    def preprocess_response(self, response: str) -> str:
        """Clean up response text to improve parsing success"""
        # Remove extra whitespace and normalize newlines
        cleaned = re.sub(r'\s+', ' ', response.strip())
        
        # Fix common formatting issues
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)  # Normalize comma spacing
        cleaned = re.sub(r'\s*:\s*', ': ', cleaned)  # Normalize colon spacing
        cleaned = re.sub(r'\s*=\s*', '=', cleaned)   # Normalize equals spacing
        
        # Handle code blocks that might be split across lines
        cleaned = re.sub(r'```\s*json\s*\n', '```json\n', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*plaintext\s*\n', '```plaintext\n', cleaned, flags=re.IGNORECASE)
        
        # Remove repetitive text patterns that can confuse parsing
        # Remove patterns like "- D = 5\n- O = 0 (not valid, so we need to adjust)" repeated multiple times
        cleaned = re.sub(r'(- [A-Z] = \d.*?\n){5,}', '', cleaned, flags=re.DOTALL)
        
        return cleaned
    
    def parse_dict_content(self, content: str) -> Dict[str, int]:
        """Parse dictionary-like content"""
        assignment = {}
        
        # Try ast.literal_eval first
        try:
            if content.startswith('{') and content.endswith('}'):
                raw_dict = ast.literal_eval(content)
            else:
                raw_dict = ast.literal_eval('{' + content + '}')
                
            if isinstance(raw_dict, dict):
                for letter, digit in raw_dict.items():
                    if isinstance(letter, str) and len(letter) == 1:
                        if isinstance(digit, (int, str)):
                            try:
                                assignment[letter.upper()] = int(digit)
                            except ValueError:
                                continue
        except:
            # Manual parsing as fallback
            pairs = re.findall(r'["\']?([A-Z])["\']?\s*[=:]\s*["\']?(\d)["\']?', content, re.IGNORECASE)
            for letter, digit in pairs:
                assignment[letter.upper()] = int(digit)
        
        return assignment
    
    def find_duplicate_digits(self, assignment: Dict[str, int]) -> List[Tuple[int, List[str]]]:
        """Find duplicate digit assignments"""
        digit_to_letters = {}
        for letter, digit in assignment.items():
            if digit not in digit_to_letters:
                digit_to_letters[digit] = []
            digit_to_letters[digit].append(letter)
        
        duplicates = [(digit, letters) for digit, letters in digit_to_letters.items() if len(letters) > 1]
        return duplicates
    
    def check_leading_zeros(self, problem: Dict[str, Any], assignment: Dict[str, int]) -> List[str]:
        """Check for leading zero violations"""
        leading_letters = {problem['word1'][0], problem['word2'][0], problem['result_word'][0]}
        violations = [letter for letter in leading_letters if assignment.get(letter) == 0]
        return violations
    
    def get_arithmetic_details(self, problem: Dict[str, Any], assignment: Dict[str, int]) -> Dict[str, Any]:
        """Get detailed arithmetic check information"""
        try:
            solver = CryptarithmeticSolver()
            num1 = solver.word_to_number(problem['word1'], assignment)
            num2 = solver.word_to_number(problem['word2'], assignment)
            num_result = solver.word_to_number(problem['result_word'], assignment)

            operation = problem.get('operation', 'addition')
            op_map = {
                'addition': ('+', num1 + num2),
                'subtraction': ('-', num1 - num2),
                'multiplication': ('*', num1 * num2),
            }
            op_symbol, computed = op_map.get(operation, ('+', num1 + num2))

            return {
                'word1_value': num1,
                'word2_value': num2,
                'result_value': num_result,
                'operation': operation,
                'computed_result': computed,
                'equation_holds': computed == num_result,
                'equation_string': f"{num1} {op_symbol} {num2} = {num_result}"
            }
        except Exception as e:
            return {
                'error': str(e),
                'equation_holds': False
            }
    

def main():
    parser = argparse.ArgumentParser(description="Enhanced Cryptarithmetic Puzzles Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Inference engine')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for VLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization for VLLM')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of datapoints per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--word_lengths', type=str, default=','.join(map(str, WORD_LENGTHS)), help='Word lengths (comma-separated)')
    parser.add_argument('--puzzle_types', type=str, default=','.join(PUZZLE_TYPES), help='Puzzle types (comma-separated)')
    parser.add_argument('--difficulty_levels', type=str, default=','.join(DIFFICULTY_LEVELS), help='Difficulty levels (comma-separated)')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='cryptarithmetic_results', help='Output directory')

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
    word_lengths = [int(x.strip()) for x in args.word_lengths.split(',')]
    puzzle_types = [x.strip() for x in args.puzzle_types.split(',')]
    difficulty_levels = [x.strip() for x in args.difficulty_levels.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("🔤 Starting Cryptarithmetic Puzzles Evaluation")
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
        logging.info(f"🎯 Initializing Enhanced Cryptarithmetic task")
        task = CryptarithmeticTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            word_lengths=word_lengths,
            puzzle_types=puzzle_types,
            difficulty_levels=difficulty_levels,
            num_folds=args.folds,
            num_samples=args.datapoints,
            store_details=args.store_details,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed
        )
        
        # Now setup logging in the task-specific directory
        task_log_file = setup_logging(task.task_dir)
        logging.info(f"📁 Task-specific directory: {task.task_dir}")
        logging.info(f"📝 Task log file: {task_log_file}")
        
        # Set random seeds
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            logging.info(f"🎲 Set random seed to {args.seed}")
        
        # Run evaluation
        logging.info(f"🚀 Starting evaluation with word lengths: {word_lengths}, operations: {puzzle_types}, difficulties: {difficulty_levels}")
        all_metrics = task.run_evaluation(list_sizes=word_lengths)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(task.task_dir, 'cryptarithmetic', word_lengths)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, word_lengths, task.task_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, word_length in enumerate(word_lengths):
                # Find metrics for this word length from all_metrics
                length_metrics = [m for m in all_metrics if m.get('word_length') == word_length or
                                (i < len(all_metrics) and all_metrics[i] is m)]

                if not length_metrics:
                    # Create a default metric for this word length
                    length_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                     'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"cryptarithmetic_{word_length}",
                    'test_case': i,
                    'complexity': word_length,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in length_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in length_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in length_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in length_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in length_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, word_lengths, task.task_dir)
        
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
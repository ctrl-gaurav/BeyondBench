"""
Advanced Modular Systems Solver Task

This task evaluates LLM's ability to solve complex systems of modular arithmetic equations 
with additional mathematical constraints. It requires deep understanding of:
- Chinese Remainder Theorem
- Extended Euclidean Algorithm  
- Modular arithmetic operations
- Number theory (primality, divisibility)
- Multi-step logical reasoning
- Constraint satisfaction

PROBLEM TYPES:
- Basic modular systems (3-5 equations)
- Systems with primality constraints
- Systems with range and divisibility constraints
- Systems with additional number-theoretic conditions
- Mixed constraint types requiring multi-step reasoning

Algorithm: Uses Chinese Remainder Theorem for coprime moduli, Extended Euclidean Algorithm
for finding modular inverses, and sophisticated constraint satisfaction for additional conditions.
Ground truth is generated using mathematically rigorous algorithms ensuring correctness.

Reasoning: Requires advanced mathematical reasoning combining modular arithmetic, number theory,
constraint satisfaction, and multi-step logical inference.

Example: Find x such that:
- x ≡ 3 (mod 7)  
- x ≡ 5 (mod 11)
- x ≡ 2 (mod 13)
- x > 500 and x < 2000
- x is prime

CLI USAGE:
python modular_systems_solver_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --datapoints 20 --folds 1 --num_equations 3,4,5 --constraint_types "basic,prime,range,divisible,mixed" --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --store_details True
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["modular_systems_solver"]
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20
FOLDS = 1
NUM_EQUATIONS = [3, 4, 5]  # Number of modular equations in system
CONSTRAINT_TYPES = ["basic", "prime", "range", "divisible", "mixed"]  # Types of additional constraints
STORE_DETAILS = True
SEED = 42

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
import numpy as np
import re
import ast
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import math
from dataclasses import dataclass


from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

@dataclass
class ModularEquation:
    """Represents a modular equation x ≡ a (mod m)"""
    remainder: int  # a
    modulus: int    # m
    
    def __str__(self):
        return f"x ≡ {self.remainder} (mod {self.modulus})"

@dataclass
class AdditionalConstraint:
    """Represents additional constraints on the solution"""
    constraint_type: str  # "prime", "range", "divisible", "composite", etc.
    parameters: Dict[str, Any]
    
    def __str__(self):
        if self.constraint_type == "prime":
            return "x is prime"
        elif self.constraint_type == "composite":
            return "x is composite"
        elif self.constraint_type == "range":
            return f"{self.parameters['min']} < x < {self.parameters['max']}"
        elif self.constraint_type == "divisible":
            return f"x is divisible by {self.parameters['divisor']}"
        elif self.constraint_type == "not_divisible":
            return f"x is not divisible by {self.parameters['divisor']}"
        else:
            return str(self.parameters)

class NumberTheoryUtils:
    """Utility functions for number theory operations"""
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Compute greatest common divisor using Euclidean algorithm"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm: returns gcd, and coefficients x, y such that ax + by = gcd"""
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = NumberTheoryUtils.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    @staticmethod
    def modular_inverse(a: int, m: int) -> Optional[int]:
        """Find modular inverse of a modulo m using extended Euclidean algorithm"""
        gcd, x, _ = NumberTheoryUtils.extended_gcd(a, m)
        if gcd != 1:
            return None  # Inverse doesn't exist
        return (x % m + m) % m
    
    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Compute least common multiple"""
        return abs(a * b) // NumberTheoryUtils.gcd(a, b)
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if number is prime using trial division"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def next_prime(n: int) -> int:
        """Find next prime number >= n"""
        if n < 2:
            return 2
        while not NumberTheoryUtils.is_prime(n):
            n += 1
        return n
    
    @staticmethod
    def generate_primes_up_to(limit: int) -> List[int]:
        """Generate all primes up to limit using sieve of Eratosthenes"""
        if limit < 2:
            return []
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]

class ModularSystemSolver:
    """Solver for systems of modular equations with constraints"""
    
    @staticmethod
    def solve_basic_system(equations: List[ModularEquation]) -> Optional[int]:
        """
        Solve system of modular equations using Chinese Remainder Theorem
        Assumes moduli are pairwise coprime
        """
        if not equations:
            return None
        
        # Check if moduli are pairwise coprime
        moduli = [eq.modulus for eq in equations]
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if NumberTheoryUtils.gcd(moduli[i], moduli[j]) != 1:
                    # Moduli are not coprime, need more sophisticated approach
                    return ModularSystemSolver.solve_general_system(equations)
        
        # Apply Chinese Remainder Theorem
        M = 1
        for eq in equations:
            M *= eq.modulus
        
        result = 0
        for eq in equations:
            Mi = M // eq.modulus
            yi = NumberTheoryUtils.modular_inverse(Mi, eq.modulus)
            if yi is None:
                return None  # Should not happen if coprime
            result += eq.remainder * Mi * yi
        
        return result % M
    
    @staticmethod
    def solve_general_system(equations: List[ModularEquation]) -> Optional[int]:
        """
        Solve general system of modular equations (moduli not necessarily coprime)
        Uses successive substitution method
        """
        if not equations:
            return None
        if len(equations) == 1:
            return equations[0].remainder
        
        # Start with first equation: x ≡ a1 (mod m1)
        current_remainder = equations[0].remainder
        current_modulus = equations[0].modulus
        
        # Process each subsequent equation
        for eq in equations[1:]:
            # Solve: x ≡ current_remainder (mod current_modulus) AND x ≡ eq.remainder (mod eq.modulus)
            gcd = NumberTheoryUtils.gcd(current_modulus, eq.modulus)
            
            # Check compatibility
            if (current_remainder - eq.remainder) % gcd != 0:
                return None  # No solution exists
            
            # Apply the general solution formula
            m1, m2 = current_modulus, eq.modulus
            a1, a2 = current_remainder, eq.remainder
            
            # Find solution to: m1*s + m2*t = gcd
            _, s, t = NumberTheoryUtils.extended_gcd(m1, m2)
            
            # New solution parameters
            lcm = NumberTheoryUtils.lcm(m1, m2)
            new_remainder = (a1 + m1 * s * (a2 - a1) // gcd) % lcm
            
            current_remainder = new_remainder
            current_modulus = lcm
        
        return current_remainder
    
    @staticmethod
    def satisfies_constraints(x: int, constraints: List[AdditionalConstraint]) -> bool:
        """Check if solution x satisfies all additional constraints"""
        for constraint in constraints:
            if constraint.constraint_type == "prime":
                if not NumberTheoryUtils.is_prime(x):
                    return False
            elif constraint.constraint_type == "composite":
                if NumberTheoryUtils.is_prime(x) or x <= 1:
                    return False
            elif constraint.constraint_type == "range":
                min_val = constraint.parameters.get("min", float('-inf'))
                max_val = constraint.parameters.get("max", float('inf'))
                if not (min_val < x < max_val):
                    return False
            elif constraint.constraint_type == "divisible":
                divisor = constraint.parameters["divisor"]
                if x % divisor != 0:
                    return False
            elif constraint.constraint_type == "not_divisible":
                divisor = constraint.parameters["divisor"]
                if x % divisor == 0:
                    return False
        
        return True
    
    @staticmethod
    def solve_with_constraints(equations: List[ModularEquation], 
                             constraints: List[AdditionalConstraint],
                             search_limit: int = 10000) -> Optional[int]:
        """
        Solve modular system with additional constraints
        Uses basic solution + constraint checking with periodic search
        """
        # Get basic solution
        basic_solution = ModularSystemSolver.solve_basic_system(equations)
        if basic_solution is None:
            return None
        
        # Calculate period (LCM of all moduli)
        period = 1
        for eq in equations:
            period = NumberTheoryUtils.lcm(period, eq.modulus)
        
        # Search for solution satisfying all constraints
        for k in range(search_limit // period + 1):
            candidate = basic_solution + k * period
            if ModularSystemSolver.satisfies_constraints(candidate, constraints):
                return candidate
        
        return None

class ModularSystemGenerator:
    """Generator for modular systems with various constraint types"""
    
    @staticmethod
    def generate_basic_system(num_equations: int, max_modulus: int = 50) -> Tuple[List[ModularEquation], List[AdditionalConstraint]]:
        """Generate basic modular system with no additional constraints"""
        equations = []
        used_moduli = set()
        
        # Generate pairwise coprime moduli for easier solving
        primes = NumberTheoryUtils.generate_primes_up_to(max_modulus)
        random.shuffle(primes)
        
        for i in range(num_equations):
            if i < len(primes):
                modulus = primes[i]
            else:
                # Use composite numbers if we run out of primes
                modulus = random.randint(3, max_modulus)
                while modulus in used_moduli or any(NumberTheoryUtils.gcd(modulus, m) > 1 for m in used_moduli):
                    modulus = random.randint(3, max_modulus)
            
            remainder = random.randint(0, modulus - 1)
            equations.append(ModularEquation(remainder, modulus))
            used_moduli.add(modulus)
        
        return equations, []
    
    @staticmethod
    def generate_prime_constraint_system(num_equations: int) -> Tuple[List[ModularEquation], List[AdditionalConstraint]]:
        """Generate system with primality constraint"""
        equations, _ = ModularSystemGenerator.generate_basic_system(num_equations)
        constraints = [AdditionalConstraint("prime", {})]
        return equations, constraints
    
    @staticmethod
    def generate_range_constraint_system(num_equations: int) -> Tuple[List[ModularEquation], List[AdditionalConstraint]]:
        """Generate system with range constraint"""
        equations, _ = ModularSystemGenerator.generate_basic_system(num_equations)
        
        # Calculate basic solution to set reasonable range
        basic_sol = ModularSystemSolver.solve_basic_system(equations)
        if basic_sol is None:
            basic_sol = 100
        
        min_val = max(50, basic_sol - 500)
        max_val = basic_sol + 1000
        
        constraints = [AdditionalConstraint("range", {"min": min_val, "max": max_val})]
        return equations, constraints
    
    @staticmethod
    def generate_divisible_constraint_system(num_equations: int) -> Tuple[List[ModularEquation], List[AdditionalConstraint]]:
        """Generate system with divisibility constraint"""
        equations, _ = ModularSystemGenerator.generate_basic_system(num_equations)
        
        # Choose a divisor that's not one of the moduli
        used_moduli = {eq.modulus for eq in equations}
        divisor = random.choice([6, 10, 15, 21, 35])
        while divisor in used_moduli:
            divisor = random.randint(3, 20)
        
        constraints = [AdditionalConstraint("divisible", {"divisor": divisor})]
        return equations, constraints
    
    @staticmethod
    def generate_mixed_constraint_system(num_equations: int) -> Tuple[List[ModularEquation], List[AdditionalConstraint]]:
        """Generate system with multiple types of constraints"""
        equations, _ = ModularSystemGenerator.generate_basic_system(num_equations)
        
        constraints = []
        
        # Add range constraint
        basic_sol = ModularSystemSolver.solve_basic_system(equations)
        if basic_sol is None:
            basic_sol = 200
        min_val = max(100, basic_sol - 300)
        max_val = basic_sol + 800
        constraints.append(AdditionalConstraint("range", {"min": min_val, "max": max_val}))
        
        # Add either primality or divisibility constraint
        if random.choice([True, False]):
            constraints.append(AdditionalConstraint("prime", {}))
        else:
            divisor = random.choice([3, 5, 7])
            constraints.append(AdditionalConstraint("divisible", {"divisor": divisor}))
        
        return equations, constraints

class ModularSystemsTask(BaseTask):
    """Modular Systems Solver evaluation task with sophisticated parsing"""
    
    def __init__(self, model_handler, output_dir, num_equations_list, constraint_types, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        # Initialize base class with proper min/max values
        self._task_name = "modular_systems"
        super().__init__(model_handler, output_dir, min(num_equations_list), max(num_equations_list), 
                        num_folds, num_samples, store_details, temperature, top_p, max_tokens, seed)
        self.num_equations_list = num_equations_list
        self.constraint_types = constraint_types
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Setup parsing patterns
        self.setup_parsing_patterns()
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, **kwargs):
        """Generate evaluation data
        
        Args:
            list_size: Number of equations in the system
        """
        list_size = kwargs.get('list_size', None)
        if self.seed is not None:
            random.seed(self.seed)
        
        num_equations = list_size if list_size is not None else random.choice(self.num_equations_list)
        test_cases = []
        
        # Generate exactly num_samples total, distributed across constraint types
        samples_per_type = max(1, self.num_samples // len(self.constraint_types))
        remaining_samples = self.num_samples - (samples_per_type * len(self.constraint_types))
        
        for i, constraint_type in enumerate(self.constraint_types):
            # Add extra sample to first types if needed to reach exact num_samples
            samples_for_this_type = samples_per_type + (1 if i < remaining_samples else 0)
            
            for _ in range(samples_for_this_type):
                problem = self.generate_problem(constraint_type, num_equations)
                if problem['reference_solution'] is not None:
                    problem['answer'] = problem['reference_solution']  # For BaseTask compatibility
                    test_cases.append(problem)
        
        # Ensure we have exactly num_samples by padding if necessary
        while len(test_cases) < self.num_samples:
            constraint_type = random.choice(self.constraint_types)
            problem = self.generate_problem(constraint_type, num_equations)
            if problem['reference_solution'] is not None:
                problem['answer'] = problem['reference_solution']
                test_cases.append(problem)
        
        return test_cases[:self.num_samples]  # Ensure exact count
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response with robust handling"""
        result = self.parse_response(response, data_point)
        
        # Always include parsed result
        parsed_solution = result.get('parsed_solution', None)
        is_correct = result.get('is_correct', False)
        parsing_confidence = result.get('parsing_confidence', 0.0)
        
        # More lenient instruction following - consider successful if we parsed something numerical
        instruction_followed = 1 if parsed_solution is not None or parsing_confidence > 0.3 else 0
        
        return {
            'accuracy': 1 if is_correct else 0,
            'instruction_followed': instruction_followed,
            'predicted_answer': parsed_solution,
            'ground_truth': data_point.get('reference_solution'),
            'parsing_info': {
                'method': result.get('parsing_method', 'unknown'),
                'confidence': parsing_confidence,
                'attempts': result.get('parsing_attempts', []),
                'raw_response_preview': response[:200] + '...' if len(response) > 200 else response
            },
            'validation_info': result.get('validation_info', {})
        }
    
    def setup_parsing_patterns(self):
        """Setup comprehensive regex patterns for parsing solution formats with intelligent prioritization"""
        # High-priority patterns (explicit final answer formats)
        self.high_priority_patterns = [
            # Boxed formats (LaTeX math)
            (re.compile(r'\\boxed\{(\d+)\}', re.IGNORECASE), "boxed_latex", 0.98),
            (re.compile(r'boxed\{(\d+)\}', re.IGNORECASE), "boxed_simple", 0.98),
            
            # Answer tags (as explicitly requested)
            (re.compile(r'<answer>\s*(\d+)\s*</answer>', re.IGNORECASE | re.DOTALL), "answer_tags", 0.99),
            (re.compile(r'\[answer\]\s*(\d+)\s*\[/answer\]', re.IGNORECASE), "bracket_answer_tags", 0.99),
            
            # Final answer declarations
            (re.compile(r'final\s+(?:answer|solution)\s*:?\s*(\d+)', re.IGNORECASE), "final_answer", 0.96),
            (re.compile(r'(?:the\s+)?(?:final\s+)?(?:answer\s+is|solution\s+is)\s*:?\s*(\d+)', re.IGNORECASE), "answer_is", 0.95),
        ]
        
        # Medium-priority patterns (contextual answer formats)
        self.medium_priority_patterns = [
            # "Therefore" and conclusion patterns
            (re.compile(r'therefore\s*,?\s*(?:x\s*=\s*|the\s+answer\s+is\s*)?\s*(\d+)', re.IGNORECASE), "therefore_conclusion", 0.92),
            (re.compile(r'(?:thus|hence|so)\s*,?\s*(?:x\s*=\s*)?\s*(\d+)(?=\s*$|\s*\.|\s*is\s+the)', re.IGNORECASE), "conclusion_keywords", 0.90),
            
            # "The value of x is" patterns
            (re.compile(r'(?:the\s+)?(?:value\s+of\s+)?x\s+is\s*(\d+)', re.IGNORECASE), "x_is_value", 0.88),
            (re.compile(r'(?:we\s+get|we\s+have|we\s+find)\s+x\s*=\s*(\d+)', re.IGNORECASE), "we_get_x", 0.87),
            
            # Solution keyword patterns
            (re.compile(r'(?:solution|answer|result)\s*:?\s*(\d+)(?=\s*$|\s*\.|\s*is\s+the)', re.IGNORECASE), "solution_keyword", 0.85),
        ]
        
        # Equation-based patterns (may extract intermediate values)
        self.equation_patterns = [
            # X equals patterns - but with context awareness
            (re.compile(r'(?:therefore\s*,?\s*)?x\s*=\s*(\d+)(?=\s*$|\s*\.|\s*is\s+the)', re.IGNORECASE), "x_equals_final", 0.83),
            (re.compile(r'x\s*=\s*(\d+)\s*(?:is\s+the|satisfies)', re.IGNORECASE), "x_equals_verified", 0.82),
            
            # Parametric solution extraction (e.g., "= 231n + 203" where 203 is the answer)
            (re.compile(r'=\s*\d+[a-z]\s*\+\s*(\d+)\s*$', re.MULTILINE), "parametric_solution", 0.75),
            (re.compile(r'x\s*=\s*\d+[a-z]\s*\+\s*(\d+)', re.IGNORECASE), "parametric_x_solution", 0.75),
        ]
        
        # Low-priority fallback patterns
        self.fallback_patterns = [
            # Numbers at end of response
            (re.compile(r'(\d+)\s*$', re.MULTILINE), "number_at_end", 0.40),
            
            # Last line number
            (re.compile(r'(\d+)(?:\s*\.?\s*)?\s*$'), "last_line_number", 0.35),
            
            # Any equation with x (lowest priority)
            (re.compile(r'x\s*=\s*(\d+)', re.IGNORECASE), "x_equals_any", 0.30),
        ]
        
        # Combine all patterns with proper prioritization
        self.solution_patterns = (self.high_priority_patterns + 
                                self.medium_priority_patterns + 
                                self.equation_patterns + 
                                self.fallback_patterns)
    
    def generate_problem(self, constraint_type: str, num_equations: int) -> Dict[str, Any]:
        """Generate a modular system problem"""
        # Generate system based on constraint type
        if constraint_type == "basic":
            equations, constraints = ModularSystemGenerator.generate_basic_system(num_equations)
        elif constraint_type == "prime":
            equations, constraints = ModularSystemGenerator.generate_prime_constraint_system(num_equations)
        elif constraint_type == "range":
            equations, constraints = ModularSystemGenerator.generate_range_constraint_system(num_equations)
        elif constraint_type == "divisible":
            equations, constraints = ModularSystemGenerator.generate_divisible_constraint_system(num_equations)
        elif constraint_type == "mixed":
            equations, constraints = ModularSystemGenerator.generate_mixed_constraint_system(num_equations)
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        # Find reference solution
        reference_solution = ModularSystemSolver.solve_with_constraints(equations, constraints)
        
        return {
            'num_equations': num_equations,
            'constraint_type': constraint_type,
            'equations': equations,
            'constraints': constraints,
            'reference_solution': reference_solution
        }
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create comprehensive prompt for modular systems problem"""
        equations = problem['equations']
        constraints = problem['constraints']
        constraint_type = problem.get('constraint_type', 'basic')
        
        # Build equations string
        equations_str = '\n'.join([f"- {eq}" for eq in equations])
        
        # Build constraints string
        constraints_str = ""
        if constraints:
            constraints_str = "\n\nADDITIONAL CONSTRAINTS:\n"
            constraints_str += '\n'.join([f"- {constraint}" for constraint in constraints])
        
        # Add strategic hints based on constraint type
        strategy_hint = ""
        if constraint_type == "prime":
            strategy_hint = "\n\nSTRATEGY HINT: After finding the basic solution using the Chinese Remainder Theorem, check for primality and search through equivalent solutions."
        elif constraint_type == "mixed":
            strategy_hint = "\n\nSTRATEGY HINT: This problem combines multiple constraint types. Solve the modular system first, then systematically check all constraints."
        elif len(equations) >= 4:
            strategy_hint = "\n\nSTRATEGY HINT: For systems with many equations, verify that your solution satisfies ALL modular equations before finalizing."
        
        # Create comprehensive prompt
        prompt = f"""ADVANCED MODULAR SYSTEMS PROBLEM:

Your task: Solve the following system of modular arithmetic equations and find the value of x that satisfies all conditions.

SYSTEM OF EQUATIONS:
{equations_str}{constraints_str}

MATHEMATICAL BACKGROUND:
- Modular arithmetic: x ≡ a (mod m) means x leaves remainder a when divided by m
- Use the Chinese Remainder Theorem for systems with coprime moduli
- For non-coprime moduli, use the general solution method
- Additional constraints must be satisfied by the final answer

SOLUTION APPROACH:
1. Identify if the moduli are pairwise coprime
2. Apply the appropriate solution method (CRT or general method)
3. Find the basic solution to the modular system
4. Check if the solution satisfies all additional constraints
5. If not, search through equivalent solutions (solution + k×LCM) until constraints are met

IMPORTANT REQUIREMENTS:
- Your final answer must satisfy EVERY modular equation
- Your final answer must satisfy ALL additional constraints
- Show your work for verification
- Double-check your arithmetic{strategy_hint}
- CRITICAL: Clearly indicate your final answer - don't leave it ambiguous!

VERIFICATION CHECKLIST:
1. Substitute your answer back into each modular equation
2. Verify all additional constraints are satisfied
3. Ensure your answer is the smallest positive solution (if no range constraint)

ANSWER FORMAT REQUIREMENTS:
⚠️  EXTREMELY IMPORTANT: Your response must end with your final numerical answer in one of these formats:

Option 1 (Preferred): 
<answer>
[Your integer answer here]
</answer>

Option 2 (Alternative):
Therefore, x = [Your integer answer here]

Option 3 (LaTeX format):
\\boxed{{[Your integer answer here]}}

DO NOT end with intermediate calculations or parametric expressions like "x = 123k + 456". 
Your final line must contain the actual numerical solution that satisfies all equations.

EXAMPLE VERIFICATION:
If x = 123, check: 123 ≡ ? (mod each modulus), and verify all constraints.

Find the value of x that satisfies this complete system."""
        
        return prompt
    
    def detect_repetitive_loops(self, response: str) -> bool:
        """Detect if response contains repetitive calculation loops"""
        lines = response.split('\n')
        if len(lines) < 10:
            return False
        
        # Check for repeated patterns in the last part of response
        last_50_lines = lines[-50:] if len(lines) > 50 else lines
        
        # Look for mathematical expressions that repeat
        math_expressions = []
        for line in last_50_lines:
            # Look for patterns like "123 ≡ 456" or "123 = 456"
            if re.search(r'\d+\s*[≡=]\s*\d+', line):
                math_expressions.append(line.strip())
        
        # If we have more than 5 identical mathematical expressions, it's likely a loop
        if len(math_expressions) > 5:
            from collections import Counter
            expr_counts = Counter(math_expressions)
            max_count = max(expr_counts.values()) if expr_counts else 0
            return max_count > 3
        
        return False
    
    def extract_problem_context_numbers(self, problem: Dict[str, Any]) -> Set[int]:
        """Extract numbers that appear in the problem statement to avoid extracting them as answers"""
        problem_numbers = set()
        
        # Add moduli and remainders
        for eq in problem['equations']:
            problem_numbers.add(eq.modulus)
            problem_numbers.add(eq.remainder)
        
        # Add constraint parameters
        for constraint in problem['constraints']:
            if 'divisor' in constraint.parameters:
                problem_numbers.add(constraint.parameters['divisor'])
            if 'min' in constraint.parameters:
                problem_numbers.add(constraint.parameters['min'])
            if 'max' in constraint.parameters:
                problem_numbers.add(constraint.parameters['max'])
        
        return problem_numbers
    
    def calculate_position_weight(self, match_pos: int, total_length: int) -> float:
        """Calculate position-based weight (favor answers near the end)"""
        if total_length == 0:
            return 1.0
        
        position_ratio = match_pos / total_length
        # Give higher weight to positions in the last 30% of the response
        if position_ratio > 0.7:
            return 1.2
        elif position_ratio > 0.5:
            return 1.0
        else:
            return 0.8
    
    def parse_response(self, response: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with ultra-robust multi-stage parsing and intelligent prioritization"""
        parsing_attempts = []
        candidates = []
        
        # Stage 1: Check for repetitive loops
        if self.detect_repetitive_loops(response):
            return {
                'parsed_solution': None,
                'parsing_method': 'loop_detected',
                'parsing_confidence': 0.0,
                'parsing_attempts': [("loop_detection", "Repetitive calculation loop detected", 0.0)],
                'validation_info': {'error': 'Model stuck in repetitive calculation loop'},
                'is_correct': False,
                'raw_response': response[:500] + '... [TRUNCATED - REPETITIVE LOOP DETECTED]'
            }
        
        # Stage 2: Get problem context numbers to avoid false positives
        problem_numbers = self.extract_problem_context_numbers(problem)
        
        # Stage 3: Multi-tier parsing with intelligent extraction
        full_response = response.strip()
        total_length = len(full_response)
        
        # First, try to extract from explicit answer sections
        answer_sections = []
        if '<answer>' in response and '</answer>' in response:
            start = response.find('<answer>') + len('<answer>')
            end = response.find('</answer>', start)
            if start < end:
                answer_sections.append((response[start:end].strip(), "answer_tags", 1.0))
        
        # Also check for code blocks that might contain final calculations
        code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', response, re.DOTALL)
        for block in code_blocks:
            if re.search(r'\d+', block) and 'final' in block.lower():
                answer_sections.append((block.strip(), "code_block", 0.3))
        
        # If no explicit sections found, use full response
        if not answer_sections:
            answer_sections = [(full_response, "full_response", 0.0)]
        
        # Stage 4: Apply parsing patterns to each section
        for section_text, section_type, section_weight in answer_sections:
            for pattern, method_name, base_confidence in self.solution_patterns:
                try:
                    for match in pattern.finditer(section_text):
                        solution_str = match.group(1)
                        solution = int(solution_str)
                        
                        # Skip if it's a problem context number (likely false positive)
                        if solution in problem_numbers:
                            parsing_attempts.append((method_name, f"Skipped {solution} (problem context)", 0.0))
                            continue
                        
                        # Calculate position-based weight
                        match_pos = match.start() if section_type == "full_response" else total_length * 0.8
                        position_weight = self.calculate_position_weight(match_pos, total_length)
                        
                        # Calculate final confidence
                        final_confidence = base_confidence * position_weight * (1.0 + section_weight)
                        
                        # Bonus for certain contextual clues
                        context_before = section_text[:match.start()].lower()
                        context_after = section_text[match.end():match.end()+50].lower()
                        
                        if any(phrase in context_before[-100:] for phrase in ['therefore', 'final', 'answer', 'solution']):
                            final_confidence *= 1.2
                        
                        if any(phrase in context_after for phrase in ['is the answer', 'is the solution', 'satisfies']):
                            final_confidence *= 1.1
                        
                        candidates.append({
                            'solution': solution,
                            'method': method_name,
                            'confidence': final_confidence,
                            'position': match_pos,
                            'section': section_type,
                            'context': f"{context_before[-20:]}[{solution}]{context_after[:20]}"
                        })
                        
                        parsing_attempts.append((method_name, f"Found: {solution} (conf: {final_confidence:.2f})", final_confidence))
                        
                except (ValueError, IndexError, AttributeError) as e:
                    parsing_attempts.append((method_name, f"Error: {str(e)}", 0.0))
        
        # Stage 5: Select best candidate
        if not candidates:
            parsing_attempts.append(("no_candidates", "No valid candidates found", 0.0))
            best_solution = None
            best_confidence = 0.0
            best_method = "none"
        else:
            # Sort by confidence, then by position (later is better)
            candidates.sort(key=lambda x: (x['confidence'], x['position']), reverse=True)
            best_candidate = candidates[0]
            
            best_solution = best_candidate['solution']
            best_confidence = best_candidate['confidence']
            best_method = best_candidate['method']
            
            # Log all top candidates for debugging
            for i, candidate in enumerate(candidates[:3]):
                parsing_attempts.append((f"candidate_{i+1}", 
                                        f"Sol: {candidate['solution']}, Conf: {candidate['confidence']:.2f}, Method: {candidate['method']}", 
                                        candidate['confidence']))
        
        # Stage 6: Validation
        validation_info = {}
        if best_solution is not None:
            # Check if solution satisfies modular equations
            equations_satisfied = []
            for eq in problem['equations']:
                satisfied = (best_solution % eq.modulus) == eq.remainder
                equations_satisfied.append(satisfied)
            
            # Check constraints
            constraints_satisfied = ModularSystemSolver.satisfies_constraints(
                best_solution, problem['constraints']
            )
            
            all_equations_correct = all(equations_satisfied)
            is_correct = all_equations_correct and constraints_satisfied
            
            validation_info = {
                'equations_satisfied': equations_satisfied,
                'all_equations_correct': all_equations_correct,
                'constraints_satisfied': constraints_satisfied,
                'is_complete_solution': is_correct,
                'solution_value': best_solution,
                'total_candidates': len(candidates)
            }
        else:
            is_correct = False
            validation_info = {
                'equations_satisfied': [],
                'all_equations_correct': False,
                'constraints_satisfied': False,
                'is_complete_solution': False,
                'solution_value': None,
                'total_candidates': len(candidates),
                'error': "Could not parse any numerical solution from response"
            }
        
        return {
            'parsed_solution': best_solution,
            'parsing_method': best_method,
            'parsing_confidence': best_confidence,
            'parsing_attempts': parsing_attempts,
            'validation_info': validation_info,
            'is_correct': is_correct,
            'raw_response': response
        }
    
    def validate_solution(self, solution: int, problem: Dict[str, Any]) -> bool:
        """Validate that solution satisfies all conditions"""
        # Check modular equations
        for eq in problem['equations']:
            if (solution % eq.modulus) != eq.remainder:
                return False
        
        # Check additional constraints
        if not ModularSystemSolver.satisfies_constraints(solution, problem['constraints']):
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Modular Systems Solver Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Inference engine')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for VLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization for VLLM')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of datapoints per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--num_equations', type=str, default=','.join(map(str, NUM_EQUATIONS)), help='Number of equations (comma-separated)')
    parser.add_argument('--constraint_types', type=str, default=','.join(CONSTRAINT_TYPES), help='Constraint types (comma-separated)')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='modular_systems_solver_results', help='Output directory')

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
    num_equations_list = [int(x.strip()) for x in args.num_equations.split(',')]
    constraint_types = [x.strip() for x in args.constraint_types.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("🔢 Starting Modular Systems Solver Evaluation")
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
        logging.info(f"🎯 Initializing Modular Systems Solver task")
        task = ModularSystemsTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            num_equations_list=num_equations_list,
            constraint_types=constraint_types,
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
        logging.info(f"🚀 Starting evaluation with equation counts: {num_equations_list}")
        all_metrics = task.run_evaluation(list_sizes=num_equations_list)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(args.output_dir, 'modular_systems_solver', num_equations_list)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, num_equations_list, args.output_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, num_equations in enumerate(num_equations_list):
                # Find metrics for this equation count from all_metrics
                eq_metrics = [m for m in all_metrics if m.get('num_equations') == num_equations or
                            (i < len(all_metrics) and all_metrics[i] is m)]

                if not eq_metrics:
                    # Create a default metric for this equation count
                    eq_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                 'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"modular_systems_solver_{num_equations}",
                    'test_case': i,
                    'complexity': num_equations,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in eq_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in eq_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in eq_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in eq_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in eq_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, num_equations_list, args.output_dir)
        
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
                if hasattr(model_handler, 'cleanup'):
                    model_handler.cleanup()
                    logging.info("🧹 Model handler cleanup completed")
            except Exception as cleanup_error:
                logging.warning(f"⚠️  Cleanup warning: {cleanup_error}")

if __name__ == "__main__":
    main()
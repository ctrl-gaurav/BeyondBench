"""
Enhanced Boolean Satisfiability (SAT) Problem Solving Task

This task evaluates LLM's ability to solve complex Boolean SAT problems through advanced logical reasoning.
SAT involves finding truth assignments to Boolean variables that satisfy a given Boolean formula in CNF.

ENHANCEMENTS:
- Complex SAT variants: horn, xor, random-k-sat, graph-based, mixed-clause-length formulas
- Larger problem sizes: 4-12 variables, up to 20+ clauses
- Higher clause/variable ratios: forcing harder constraint satisfaction
- Strategic SAT types: unit propagation challenges, backdoor variables, phase transitions
- Enhanced parsing: 15+ parsing patterns for diverse response formats
- Robust validation: Detailed clause satisfaction analysis

Algorithm: Generate satisfiable CNF formulas with known solutions using advanced SAT generation techniques.
Reasoning: Requires systematic logical reasoning, constraint propagation, conflict analysis, and strategic search.

Example: Complex mixed formula with unit clauses, binary clauses, and ternary clauses requiring careful analysis.

CLI USAGE:
python boolean_sat_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --datapoints 20 --folds 1 --num_variables 4,6,8,10 --sat_types "horn,xor,random_k,graph_based,mixed" --clause_ratios 2.5,3.0,3.5,4.0 --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42
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
NUM_VARIABLES = [4, 6, 8, 10]  # Enhanced range for more challenging problems
SAT_TYPES = ["horn", "xor", "random_k", "graph_based", "mixed"]  # Different SAT problem types
CLAUSE_RATIOS = [2.5, 3.0, 3.5, 4.0]  # Clause-to-variable ratios for difficulty scaling
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
import ast
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import itertools

from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

class SATSolver:
    """Enhanced Boolean SAT solver and generator with multiple complexity types"""
    
    @staticmethod
    def generate_satisfiable_formula(num_vars: int, sat_type: str = "random_k", clause_ratio: float = 3.0) -> Tuple[List[List[int]], Dict[int, bool]]:
        """Generate a satisfiable CNF formula with known solution using different SAT types"""
        num_clauses = int(num_vars * clause_ratio)
        
        if sat_type == "horn":
            return SATSolver.generate_horn_formula(num_vars, num_clauses)
        elif sat_type == "xor":
            return SATSolver.generate_xor_formula(num_vars, num_clauses)
        elif sat_type == "graph_based":
            return SATSolver.generate_graph_based_formula(num_vars, num_clauses)
        elif sat_type == "mixed":
            return SATSolver.generate_mixed_formula(num_vars, num_clauses)
        else:  # random_k
            return SATSolver.generate_random_k_formula(num_vars, num_clauses)
    
    @staticmethod
    def generate_horn_formula(num_vars: int, num_clauses: int) -> Tuple[List[List[int]], Dict[int, bool]]:
        """Generate Horn SAT formula (at most one positive literal per clause)"""
        # Generate random truth assignment
        assignment = {i+1: random.choice([True, False]) for i in range(num_vars)}
        
        clauses = []
        for _ in range(num_clauses):
            clause_size = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
            clause_vars = random.sample(range(1, num_vars + 1), min(clause_size, num_vars))
            
            clause = []
            # Horn: at most one positive literal
            has_positive = False
            for var in clause_vars:
                if not has_positive and random.random() < 0.4:  # Add positive literal
                    clause.append(var)
                    has_positive = True
                else:  # Add negative literal
                    clause.append(-var)
            
            # Ensure clause is satisfied
            if not SATSolver.is_clause_satisfied(clause, assignment):
                # Fix by making first literal satisfy the assignment
                var = abs(clause[0])
                clause[0] = var if assignment[var] else -var
            
            clauses.append(clause)
        
        return clauses, assignment
    
    @staticmethod
    def generate_xor_formula(num_vars: int, num_clauses: int) -> Tuple[List[List[int]], Dict[int, bool]]:
        """Generate XOR-SAT like formula (exactly one literal true per clause)"""
        assignment = {i+1: random.choice([True, False]) for i in range(num_vars)}
        
        clauses = []
        for _ in range(num_clauses):
            clause_size = random.choices([2, 3, 4], weights=[0.4, 0.4, 0.2])[0]
            clause_vars = random.sample(range(1, num_vars + 1), min(clause_size, num_vars))
            
            # Create clause where exactly one literal should be true
            clause = []
            true_count = 0
            for var in clause_vars:
                if random.random() < 0.5:
                    clause.append(var)
                    if assignment[var]:
                        true_count += 1
                else:
                    clause.append(-var)
                    if not assignment[var]:
                        true_count += 1
            
            # Adjust to ensure exactly one is true (for satisfiability)
            if true_count == 0:
                # Make first literal true
                var = abs(clause[0])
                clause[0] = var if assignment[var] else -var
            elif true_count > 1:
                # Make all but first literal false
                for i in range(1, len(clause)):
                    var = abs(clause[i])
                    clause[i] = var if not assignment[var] else -var
            
            clauses.append(clause)
        
        return clauses, assignment
    
    @staticmethod
    def generate_graph_based_formula(num_vars: int, num_clauses: int) -> Tuple[List[List[int]], Dict[int, bool]]:
        """Generate graph-based SAT (graph coloring, clique cover, etc.)"""
        assignment = {i+1: random.choice([True, False]) for i in range(num_vars)}
        
        clauses = []
        
        # Add graph-like constraints (conflicts between related variables)
        for _ in range(num_clauses):
            if random.random() < 0.6:
                # Binary conflict clause: at most one of two variables can be true
                var1, var2 = random.sample(range(1, num_vars + 1), 2)
                # ¬x1 ∨ ¬x2 (at most one true)
                clause = [-var1, -var2]
                # Ensure satisfiability
                if assignment[var1] and assignment[var2]:
                    # Both would be true, so flip one
                    assignment[var2] = False
            else:
                # Forcing clause: at least one must be true
                clause_vars = random.sample(range(1, num_vars + 1), random.randint(2, 3))
                clause = clause_vars  # All positive
                # Ensure at least one is true
                if not any(assignment[var] for var in clause_vars):
                    assignment[clause_vars[0]] = True
            
            clauses.append(clause)
        
        return clauses, assignment
    
    @staticmethod
    def generate_mixed_formula(num_vars: int, num_clauses: int) -> Tuple[List[List[int]], Dict[int, bool]]:
        """Generate mixed formula with various clause types"""
        assignment = {i+1: random.choice([True, False]) for i in range(num_vars)}
        
        clauses = []
        clause_types = ["unit", "binary", "ternary", "long"]
        
        for _ in range(num_clauses):
            clause_type = random.choices(clause_types, weights=[0.2, 0.4, 0.3, 0.1])[0]
            
            if clause_type == "unit":
                # Unit clause (single literal)
                var = random.randint(1, num_vars)
                clause = [var if assignment[var] else -var]
            
            elif clause_type == "binary":
                # Binary clause
                vars = random.sample(range(1, num_vars + 1), 2)
                clause = []
                for var in vars:
                    clause.append(var if random.random() < 0.5 else -var)
                # Ensure satisfiability
                if not SATSolver.is_clause_satisfied(clause, assignment):
                    var = abs(clause[0])
                    clause[0] = var if assignment[var] else -var
            
            elif clause_type == "ternary":
                # Ternary clause
                vars = random.sample(range(1, num_vars + 1), 3)
                clause = []
                for var in vars:
                    clause.append(var if random.random() < 0.5 else -var)
                # Ensure satisfiability
                if not SATSolver.is_clause_satisfied(clause, assignment):
                    var = abs(clause[0])
                    clause[0] = var if assignment[var] else -var
            
            else:  # long
                # Long clause (4+ literals)
                clause_size = random.randint(4, min(num_vars, 6))
                vars = random.sample(range(1, num_vars + 1), clause_size)
                clause = []
                for var in vars:
                    clause.append(var if random.random() < 0.5 else -var)
                # Ensure satisfiability
                if not SATSolver.is_clause_satisfied(clause, assignment):
                    var = abs(clause[0])
                    clause[0] = var if assignment[var] else -var
            
            clauses.append(clause)
        
        return clauses, assignment
    
    @staticmethod
    def generate_random_k_formula(num_vars: int, num_clauses: int, k: int = 3) -> Tuple[List[List[int]], Dict[int, bool]]:
        """Generate random k-SAT formula"""
        assignment = {i+1: random.choice([True, False]) for i in range(num_vars)}
        
        clauses = []
        for _ in range(num_clauses):
            clause_size = min(k, num_vars)
            clause_vars = random.sample(range(1, num_vars + 1), clause_size)
            clause = []
            
            # Make sure at least one literal in clause is true under assignment
            satisfied = False
            for var in clause_vars:
                if random.random() < 0.5:  # Positive literal
                    clause.append(var)
                    if assignment[var]:
                        satisfied = True
                else:  # Negative literal
                    clause.append(-var)
                    if not assignment[var]:
                        satisfied = True
            
            # If clause not satisfied, fix it
            if not satisfied and clause:
                var = abs(clause[0])
                if assignment[var]:
                    clause[0] = var
                else:
                    clause[0] = -var
            
            if clause:
                clauses.append(clause)
        
        return clauses, assignment
    
    @staticmethod
    def is_clause_satisfied(clause: List[int], assignment: Dict[int, bool]) -> bool:
        """Check if a clause is satisfied by the assignment"""
        for literal in clause:
            var = abs(literal)
            if var in assignment:
                if (literal > 0 and assignment[var]) or (literal < 0 and not assignment[var]):
                    return True
        return False
    
    @staticmethod
    def evaluate_formula(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
        """Check if assignment satisfies the formula"""
        for clause in clauses:
            clause_satisfied = False
            for literal in clause:
                var = abs(literal)
                if var in assignment:
                    if (literal > 0 and assignment[var]) or (literal < 0 and not assignment[var]):
                        clause_satisfied = True
                        break
            if not clause_satisfied:
                return False
        return True

class BooleanSATTask(BaseTask):
    """Enhanced Boolean SAT evaluation task with robust parsing and multiple complexity types"""
    
    def __init__(self, model_handler, output_dir, num_variables_list, sat_types_list, clause_ratios_list,
                 num_folds, num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        self._task_name = "boolean_sat"
        super().__init__(model_handler, output_dir, min(num_variables_list), max(num_variables_list), num_folds, num_samples, 
                        store_details, temperature, top_p, max_tokens, seed)
        # Store enhanced parameters
        self.num_variables_list = num_variables_list
        self.sat_types_list = sat_types_list
        self.clause_ratios_list = clause_ratios_list
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Enhanced parsing patterns with confidence scoring
        self.setup_parsing_patterns()
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, list_size=None, **kwargs):
        """Generate evaluation data for specific number of variables with proper distribution"""
        test_cases = []
        
        if list_size is not None:
            # Generate for specific number of variables
            num_vars = list_size
            
            # Distribute samples across SAT types and clause ratios
            total_combinations = len(self.sat_types_list) * len(self.clause_ratios_list)
            base_samples_per_combo = self.num_samples // total_combinations
            extra_samples = self.num_samples % total_combinations
            
            combo_count = 0
            for sat_type in self.sat_types_list:
                for clause_ratio in self.clause_ratios_list:
                    # Calculate samples for this combination
                    samples_for_combo = base_samples_per_combo + (1 if combo_count < extra_samples else 0)
                    
                    for _ in range(samples_for_combo):
                        problem = self.generate_problem(num_vars, sat_type, clause_ratio)
                        # Add metadata for proper naming
                        problem['sat_type'] = sat_type
                        problem['clause_ratio'] = clause_ratio
                        test_cases.append(problem)
                    
                    combo_count += 1
        else:
            # Fallback: generate for all variable counts
            total_combinations = len(self.num_variables_list) * len(self.sat_types_list) * len(self.clause_ratios_list)
            base_samples_per_combo = self.num_samples // total_combinations
            extra_samples = self.num_samples % total_combinations
            
            combo_count = 0
            for num_vars in self.num_variables_list:
                for sat_type in self.sat_types_list:
                    for clause_ratio in self.clause_ratios_list:
                        samples_for_combo = base_samples_per_combo + (1 if combo_count < extra_samples else 0)
                        
                        for _ in range(samples_for_combo):
                            problem = self.generate_problem(num_vars, sat_type, clause_ratio)
                            problem['sat_type'] = sat_type
                            problem['clause_ratio'] = clause_ratio
                            test_cases.append(problem)
                        
                        combo_count += 1
        
        return test_cases
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response"""
        result = self.parse_response(response, data_point)
        return {
            'accuracy': 1 if result['is_correct'] else 0,
            'instruction_followed': 1 if result.get('parsing_confidence', 0) > 0 else 0,
            'predicted_answer': result.get('parsed_assignment', {}),
            'ground_truth': data_point.get('reference_solution', {}),
            'parsing_info': {
                'method': result.get('parsing_method', 'unknown'),
                'confidence': result.get('parsing_confidence', 0.0),
                'attempts': result.get('parsing_attempts', [])
            },
            'validation_info': result.get('validation_info', {})
        }
    
    def setup_parsing_patterns(self):
        """Setup regex patterns for parsing different assignment formats with enhanced robustness"""
        self.assignment_patterns = [
            # Pattern 1: Standard dictionary format {1: True, 2: False, 3: True}
            (re.compile(r'\{\s*([^}]+)\s*\}', re.IGNORECASE | re.DOTALL), "dict_format", 0.9),
            
            # Pattern 2: Variable assignment pairs "x1 = True, x2 = False, x3 = True"
            (re.compile(r'x(\d+)\s*=\s*(True|False|true|false|1|0)', re.IGNORECASE), "variable_equals", 0.8),
            
            # Pattern 3: Assignment with colon "x1: True, x2: False"
            (re.compile(r'x(\d+)\s*:\s*(True|False|true|false|1|0)', re.IGNORECASE), "variable_colon", 0.8),
            
            # Pattern 4: List format "variables: [1, 2, 3], values: [True, False, True]"
            (re.compile(r'variables?\s*:\s*\[([^\]]+)\]\s*,?\s*values?\s*:\s*\[([^\]]+)\]', re.IGNORECASE), "parallel_lists", 0.7),
            
            # Pattern 5: Simple assignment "1=True, 2=False, 3=True"
            (re.compile(r'(\d+)\s*=\s*(True|False|true|false|1|0)', re.IGNORECASE), "simple_assignment", 0.8),
            
            # Pattern 6: Numbered variables "variable 1 is True, variable 2 is False"
            (re.compile(r'variable\s+(\d+)\s+is\s+(True|False|true|false)', re.IGNORECASE), "variable_is", 0.7),
            
            # Pattern 7: Binary sequence "1 0 1" or "True False True"
            (re.compile(r'([01TF](?:\s+[01TF])*)', re.IGNORECASE), "sequence_format", 0.5),
            
            # Pattern 8: Assignment arrows "1 -> True, 2 -> False"
            (re.compile(r'(\d+)\s*->\s*(True|False|true|false|1|0)', re.IGNORECASE), "arrow_assignment", 0.7),
            
            # Pattern 9: Brackets with assignments [x1=True, x2=False]
            (re.compile(r'\[([^\]]+)\]', re.IGNORECASE), "bracket_assignments", 0.6),
            
            # Pattern 10: Solution keyword "Solution: x1=True, x2=False"
            (re.compile(r'solution\s*:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL), "solution_keyword", 0.8),
            
            # Pattern 11: Assignment with "is" "x1 is True, x2 is False"
            (re.compile(r'x(\d+)\s+is\s+(True|False|true|false|1|0)', re.IGNORECASE), "variable_is_extended", 0.75),
            
            # Pattern 12: Let statements "Let x1 = True, x2 = False"
            (re.compile(r'let\s+x(\d+)\s*=\s*(True|False|true|false|1|0)', re.IGNORECASE), "let_assignment", 0.72),
            
            # Pattern 13: Set statements "Set x1 to True, x2 to False"
            (re.compile(r'set\s+x(\d+)\s+to\s+(True|False|true|false)', re.IGNORECASE), "set_assignment", 0.73),
            
            # Pattern 14: Assign statements "Assign True to x1, False to x2"
            (re.compile(r'assign\s+(True|False|true|false)\s+to\s+x(\d+)', re.IGNORECASE), "assign_to_variable", 0.74),
            
            # Pattern 15: Variable gets "x1 gets True, x2 gets False"
            (re.compile(r'x(\d+)\s+gets\s+(True|False|true|false|1|0)', re.IGNORECASE), "variable_gets", 0.71),
            
            # NEW: Pattern 16: Code block format ```plaintext\n{1: True, 2: False}\n```
            (re.compile(r'```(?:plaintext|text)?\s*\n?\s*\{([^}]+)\}\s*\n?```', re.IGNORECASE | re.DOTALL), "code_block_dict", 0.95),
            
            # NEW: Pattern 17: Code block without language specifier ```{1: True, 2: False}```
            (re.compile(r'```\s*\{([^}]+)\}\s*```', re.IGNORECASE | re.DOTALL), "simple_code_block", 0.93),
            
            # NEW: Pattern 18: Final Answer format "Final Answer: {1: True, 2: False}"
            (re.compile(r'final\s+answer\s*:?\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL), "final_answer", 0.92),
            
            # NEW: Pattern 19: Answer format "Answer: {1: True, 2: False}"
            (re.compile(r'answer\s*:?\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL), "answer_format", 0.91),
            
            # NEW: Pattern 20: Multiple assignment formats in one line "x1=True, x2=False, ..."
            (re.compile(r'x(\d+)\s*=\s*(True|False|true|false|1|0)(?:\s*,\s*x(\d+)\s*=\s*(True|False|true|false|1|0))*', re.IGNORECASE), "multi_assignment", 0.85),
            
            # NEW: Pattern 21: Dictionary with x prefix {x1: True, x2: False}
            (re.compile(r'\{\s*([x\d:TrueFalse,\s]+)\s*\}', re.IGNORECASE | re.DOTALL), "x_dict_format", 0.88),
            
            # NEW: Pattern 22: Assignment format "The solution is: x1=True, x2=False"
            (re.compile(r'(?:the\s+)?solution\s+is\s*:?\s*(.+)', re.IGNORECASE | re.DOTALL), "solution_is", 0.86)
        ]
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for prompt"""
        return len(prompt) // 4
    
    def generate_problem(self, num_vars: int, sat_type: str, clause_ratio: float) -> Dict[str, Any]:
        """Generate an enhanced SAT problem"""
        clauses, reference_solution = SATSolver.generate_satisfiable_formula(num_vars, sat_type, clause_ratio)
        num_clauses = len(clauses)
        
        return {
            'num_variables': num_vars,
            'num_clauses': num_clauses,
            'sat_type': sat_type,
            'clause_ratio': clause_ratio,
            'clauses': clauses,
            'reference_solution': reference_solution,
            'answer': reference_solution  # For BaseTask compatibility
        }
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create enhanced prompt for complex SAT problems"""
        clauses = problem['clauses']
        num_vars = problem['num_variables']
        sat_type = problem.get('sat_type', 'unknown')
        clause_ratio = problem.get('clause_ratio', 0.0)
        num_clauses = len(clauses)
        
        # Format clauses for display
        clauses_str = []
        for i, clause in enumerate(clauses):
            clause_parts = []
            for literal in clause:
                if literal > 0:
                    clause_parts.append(f"x{literal}")
                else:
                    clause_parts.append(f"¬x{abs(literal)}")
            clause_display = f"({' ∨ '.join(clause_parts)})"
            clauses_str.append(clause_display)
        
        formula = " ∧ ".join(clauses_str)
        
        # Add strategic hints based on SAT type
        strategy_hint = ""
        if sat_type == "horn":
            strategy_hint = "\n\nSTRATEGY HINT: This is a Horn SAT problem. Each clause has at most one positive literal. Consider unit propagation - if a clause has only one literal, that literal must be true."
        elif sat_type == "xor":
            strategy_hint = "\n\nSTRATEGY HINT: This formula has XOR-like constraints. Look for clauses where exactly one literal should be true."
        elif sat_type == "graph_based":
            strategy_hint = "\n\nSTRATEGY HINT: This SAT formula represents graph-based constraints (conflicts and requirements between variables)."
        elif sat_type == "mixed":
            strategy_hint = "\n\nSTRATEGY HINT: This formula has mixed clause lengths. Pay attention to unit clauses (single literals) which directly determine variable values."
        elif clause_ratio >= 4.0:
            strategy_hint = "\n\nSTRATEGY HINT: This is a highly constrained problem with many clauses. Systematic analysis is required to avoid conflicts."
        
        # Add complexity analysis
        complexity_analysis = ""
        if num_clauses >= 15:
            complexity_analysis = f"""\n\nCOMPLEXITY ANALYSIS:
- {num_vars} variables, {num_clauses} clauses (ratio: {clause_ratio:.1f})
- This is a complex SAT instance requiring careful constraint analysis
- Look for unit clauses that directly determine variable values
- Check for pure literals (variables that appear only positive or only negative)
- Consider clause dependencies and conflicts"""
        
        # Add systematic solving approach
        solving_approach = f"""\n\nSYSTEMATIC APPROACH:
1. IDENTIFY UNIT CLAUSES: Look for clauses with only one literal - these variables must have specific values
2. APPLY UNIT PROPAGATION: If x{random.randint(1, num_vars)} must be True/False, simplify other clauses accordingly
3. CHECK FOR CONFLICTS: Ensure no clause becomes unsatisfiable
4. ASSIGN REMAINING VARIABLES: Choose values that satisfy the maximum number of remaining clauses
5. VERIFY SOLUTION: Check that ALL {num_clauses} clauses are satisfied by your assignment"""
        
        prompt = f"""CHALLENGING BOOLEAN SATISFIABILITY (SAT) PROBLEM:

Find a truth assignment for variables x1, x2, ..., x{num_vars} that satisfies this complex formula:

{formula}

LOGICAL OPERATORS:
- ∨ means OR (disjunction)
- ∧ means AND (conjunction)
- ¬ means NOT (negation){strategy_hint}{complexity_analysis}{solving_approach}

SOLUTION FORMAT REQUIREMENTS:
Your answer must assign True/False to ALL {num_vars} variables.

REQUIRED FORMAT (STRICTLY FOLLOW THIS):
<answer>
{{1: True, 2: False, 3: True, ..., {num_vars}: True}}
</answer>

IMPORTANT FORMATTING NOTES:
- Use EXACTLY the format shown above inside <answer> tags
- Use numbers (1, 2, 3...) NOT variable names (x1, x2, x3...)
- Use True/False (not true/false, 1/0, T/F)
- Include ALL {num_vars} variables in your answer
- Do NOT use code blocks, markdown, or other formatting

ALTERNATIVE FORMATS (if <answer> tags fail):
- Final Answer: {{1: True, 2: False, 3: True, ..., {num_vars}: True}}
- Answer: {{1: True, 2: False, 3: True, ..., {num_vars}: True}}
- x1=True, x2=False, x3=True, ..., x{num_vars}=True

VERIFICATION CHECKLIST:
1. Did you assign a value to ALL {num_vars} variables (x1 through x{num_vars})?
2. Does your assignment satisfy ALL {num_clauses} clauses in the formula?
3. Is your answer in the EXACT format specified above?

Provide your complete variable assignment that satisfies the entire formula.
REMEMBER: Use the <answer> tags with the exact dictionary format shown above!"""
        
        return prompt
    
    def parse_response(self, response: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with enhanced robustness and detailed diagnostics"""
        num_vars = problem['num_variables']
        parsing_attempts = []
        best_assignment = {}
        best_confidence = 0.0
        best_method = "none"
        
        # Pre-process response to clean up common formatting issues
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
                    assignment = self.parse_with_pattern(text_variant, pattern, method_name, num_vars)
                    if assignment and len(assignment) == num_vars:
                        confidence = base_confidence * (len(assignment) / num_vars)
                        # Bonus for complete assignments
                        confidence *= 1.1
                        parsing_attempts.append((method_name, f"Parsed {len(assignment)} variables", confidence))
                        
                        if confidence > best_confidence:
                            best_assignment = assignment
                            best_confidence = confidence
                            best_method = method_name
                    elif assignment:
                        partial_confidence = base_confidence * (len(assignment) / num_vars) * 0.5
                        parsing_attempts.append((method_name, f"Partial parse: {len(assignment)}/{num_vars} variables", partial_confidence))
                        
                        # Even partial assignments might be useful if nothing better is found
                        if partial_confidence > best_confidence and len(assignment) >= num_vars // 2:
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
            is_valid = SATSolver.evaluate_formula(problem['clauses'], best_assignment)
            
            # Check completeness
            missing_vars = [v for v in range(1, num_vars + 1) if v not in best_assignment]
            if missing_vars:
                is_valid = False
                validation_msg = f"Missing variables: {missing_vars}"
            else:
                validation_msg = "Valid assignment" if is_valid else "Assignment does not satisfy all clauses"
                
                # Check which clauses are satisfied
                satisfied_clauses = []
                unsatisfied_clauses = []
                for i, clause in enumerate(problem['clauses']):
                    clause_satisfied = any(
                        (lit > 0 and best_assignment.get(lit, False)) or
                        (lit < 0 and not best_assignment.get(-lit, True))
                        for lit in clause
                    )
                    if clause_satisfied:
                        satisfied_clauses.append(i + 1)
                    else:
                        unsatisfied_clauses.append(i + 1)
            
            validation_info = {
                'is_valid_assignment': is_valid,
                'missing_variables': missing_vars,
                'satisfied_clauses': satisfied_clauses,
                'unsatisfied_clauses': unsatisfied_clauses,
                'validation_message': validation_msg
            }
        else:
            is_valid = False
            validation_info = {
                'is_valid_assignment': False,
                'missing_variables': list(range(1, num_vars + 1)),
                'satisfied_clauses': [],
                'unsatisfied_clauses': list(range(1, len(problem['clauses']) + 1)),
                'validation_message': "Could not parse any assignment from response"
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
    
    def parse_with_pattern(self, text: str, pattern: re.Pattern, method_name: str, num_vars: int) -> Dict[int, bool]:
        """Parse assignment using a specific pattern"""
        assignment = {}
        
        if method_name == "dict_format":
            # Standard dictionary format
            matches = pattern.findall(text)
            if matches:
                dict_content = matches[0]
                return self.parse_dict_content(dict_content)
        
        elif method_name == "variable_equals":
            # Variable assignment pairs "x1 = True"
            matches = pattern.findall(text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "variable_colon":
            # Assignment with colon "x1: True"
            matches = pattern.findall(text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "parallel_lists":
            # List format "variables: [1, 2, 3], values: [True, False, True]"
            matches = pattern.findall(text)
            if matches:
                vars_str, vals_str = matches[0]
                variables = [int(v.strip()) for v in vars_str.split(',') if v.strip().isdigit()]
                values = [self.parse_boolean_value(v.strip()) for v in vals_str.split(',')]
                
                if len(variables) == len(values):
                    for var, val in zip(variables, values):
                        if 1 <= var <= num_vars:
                            assignment[var] = val
        
        elif method_name == "simple_assignment":
            # Simple assignment "1=True"
            matches = pattern.findall(text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "variable_is":
            # "variable 1 is True"
            matches = pattern.findall(text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "sequence_format":
            # Binary sequence "1 0 1" or "True False True"
            matches = pattern.findall(text)
            if matches:
                sequence = matches[0].split()
                if len(sequence) == num_vars:
                    for i, val_str in enumerate(sequence, 1):
                        assignment[i] = self.parse_boolean_value(val_str)
        
        elif method_name == "arrow_assignment":
            # Assignment arrows "1 -> True"
            matches = pattern.findall(text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "bracket_assignments":
            # Brackets with assignments
            matches = pattern.findall(text)
            if matches:
                content = matches[0]
                pairs = re.findall(r'x?(\d+)\s*=\s*(True|False|true|false|1|0)', content, re.IGNORECASE)
                for var_str, val_str in pairs:
                    var_num = int(var_str)
                    if 1 <= var_num <= num_vars:
                        assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "solution_keyword":
            # Solution keyword
            matches = pattern.findall(text)
            if matches:
                solution_text = matches[0].strip()
                # Recursively parse the solution text
                pairs = re.findall(r'x?(\d+)\s*[=:]\s*(True|False|true|false|1|0)', solution_text, re.IGNORECASE)
                for var_str, val_str in pairs:
                    var_num = int(var_str)
                    if 1 <= var_num <= num_vars:
                        assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name in ["variable_is_extended", "let_assignment", "variable_gets"]:
            # Extended variable assignment patterns
            matches = pattern.findall(text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "set_assignment":
            # Set statements "Set x1 to True"
            matches = pattern.findall(text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "assign_to_variable":
            # Assign statements "Assign True to x1"
            matches = pattern.findall(text)
            for val_str, var_str in matches:  # Note: order is swapped in this pattern
                var_num = int(var_str)
                if 1 <= var_num <= num_vars:
                    assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name in ["code_block_dict", "simple_code_block", "final_answer", "answer_format"]:
            # Code block or answer formats with dictionary content
            matches = pattern.findall(text)
            if matches:
                dict_content = matches[0]
                assignment = self.parse_dict_content(dict_content)
        
        elif method_name == "multi_assignment":
            # Multiple assignments in one line
            # This pattern captures multiple x#=value pairs
            full_matches = pattern.findall(text)
            if full_matches:
                # Extract all variable-value pairs from the match
                var_val_pairs = re.findall(r'x(\d+)\s*=\s*(True|False|true|false|1|0)', text, re.IGNORECASE)
                for var_str, val_str in var_val_pairs:
                    var_num = int(var_str)
                    if 1 <= var_num <= num_vars:
                        assignment[var_num] = self.parse_boolean_value(val_str)
        
        elif method_name == "x_dict_format":
            # Dictionary format with x prefixes {x1: True, x2: False}
            matches = pattern.findall(text)
            if matches:
                dict_content = matches[0]
                # Parse x-prefixed variables
                pairs = re.findall(r'x(\d+)\s*:\s*(True|False|true|false|1|0)', dict_content, re.IGNORECASE)
                for var_str, val_str in pairs:
                    var_num = int(var_str)
                    if 1 <= var_num <= num_vars:
                        assignment[var_num] = self.parse_boolean_value(val_str)
        
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
                    # Try variable assignment pairs
                    pairs = re.findall(r'x?(\d+)\s*[=:]\s*(True|False|true|false|1|0)', solution_text, re.IGNORECASE)
                    for var_str, val_str in pairs:
                        var_num = int(var_str)
                        if 1 <= var_num <= num_vars:
                            assignment[var_num] = self.parse_boolean_value(val_str)
        
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
        cleaned = re.sub(r'```\s*plaintext\s*\n', '```plaintext\n', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*text\s*\n', '```text\n', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def parse_dict_content(self, content: str) -> Dict[int, bool]:
        """Parse dictionary-like content"""
        assignment = {}
        
        # Try ast.literal_eval first
        try:
            if content.startswith('{') and content.endswith('}'):
                raw_dict = ast.literal_eval(content)
            else:
                raw_dict = ast.literal_eval('{' + content + '}')
                
            if isinstance(raw_dict, dict):
                for k, v in raw_dict.items():
                    var_num = int(k) if isinstance(k, (int, str)) else k
                    assignment[var_num] = self.parse_boolean_value(v)
        except:
            # Manual parsing as fallback
            pairs = re.findall(r'(\d+)\s*[=:]\s*([^,}\]\)]+)', content)
            for var_str, val_str in pairs:
                var_num = int(var_str)
                assignment[var_num] = self.parse_boolean_value(val_str.strip())
        
        return assignment
    
    def parse_boolean_value(self, value) -> bool:
        """Parse various boolean representations"""
        if isinstance(value, bool):
            return value
        
        str_val = str(value).lower().strip('"\' \t\n')
        if str_val in ['true', '1', 'yes', 't']:
            return True
        elif str_val in ['false', '0', 'no', 'f']:
            return False
        else:
            # Default to True for unrecognized values
            return True
    
    
    

def main():
    """Main evaluation function with proper logging and reporting"""
    parser = argparse.ArgumentParser(description="Boolean SAT Problem Solving Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Engine to use')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of data points per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--num_variables', type=str, default=','.join(map(str, NUM_VARIABLES)), help='Number of variables')
    parser.add_argument('--sat_types', type=str, default=','.join(SAT_TYPES), help='SAT problem types (comma-separated)')
    parser.add_argument('--clause_ratios', type=str, default=','.join(map(str, CLAUSE_RATIOS)), help='Clause-to-variable ratios (comma-separated)')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='boolean_sat_results', help='Output directory')

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
    num_variables_list = [int(x.strip()) for x in args.num_variables.split(',')]
    sat_types_list = [x.strip() for x in args.sat_types.split(',')]
    clause_ratios_list = [float(x.strip()) for x in args.clause_ratios.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("🔍 Starting Boolean SAT Problem Solving Evaluation")
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
        logging.info(f"🎯 Initializing Enhanced Boolean SAT task")
        task = BooleanSATTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            num_variables_list=num_variables_list,
            sat_types_list=sat_types_list,
            clause_ratios_list=clause_ratios_list,
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
        logging.info(f"🚀 Starting evaluation with variables: {num_variables_list}, SAT types: {sat_types_list}, clause ratios: {clause_ratios_list}")
        all_metrics = task.run_evaluation(list_sizes=num_variables_list)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(task.task_dir, 'boolean_sat', num_variables_list)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, num_variables_list, task.task_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, num_variables in enumerate(num_variables_list):
                # Find metrics for this variable count from all_metrics
                var_metrics = [m for m in all_metrics if m.get('num_variables') == num_variables or
                             (i < len(all_metrics) and all_metrics[i] is m)]

                if not var_metrics:
                    # Create a default metric for this variable count
                    var_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                  'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"boolean_sat_{num_variables}",
                    'test_case': i,
                    'complexity': num_variables,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in var_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in var_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in var_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in var_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in var_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, num_variables_list, task.task_dir)
        
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
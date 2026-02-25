"""
Enhanced Tower of Hanoi Problem Solving Task - More Robust Implementation

This enhanced version includes:
- Multiple parsing patterns for robustness
- Better move format handling  
- More comprehensive validation
- Enhanced error recovery
- Detailed diagnostic information

Based on the comprehensive implementation patterns from the main TOH directory.
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
NUM_DISKS = [3, 4, 5]
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
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.report_generator import generate_final_report
from ...utils.logging_utils import setup_logging

@dataclass
class Move:
    """Represents a move in Tower of Hanoi"""
    disk: int
    from_peg: str
    to_peg: str
    
    def __str__(self):
        return f"Move disk {self.disk} from {self.from_peg} to {self.to_peg}"

class ParseResult(Enum):
    """Parse result types"""
    SUCCESS = "success"
    PARTIAL = "partial"  
    FAILED = "failed"
    INVALID_FORMAT = "invalid_format"
    EMPTY_RESPONSE = "empty_response"

@dataclass
class ParsedMove:
    """Parsed move with confidence scoring"""
    disk: Optional[int]
    source: Optional[str]
    target: Optional[str]
    confidence: float
    original_text: str
    line_number: int
    
    def to_move(self) -> Optional[Move]:
        if all([self.disk, self.source, self.target]):
            return Move(self.disk, self.source, self.target)
        return None
    
    def is_valid(self) -> bool:
        return all([self.disk is not None, self.source, self.target])

@dataclass
class ParsingResult:
    """Complete parsing result with diagnostics"""
    moves: List[Move]
    parsed_moves: List[ParsedMove]
    result_type: ParseResult
    confidence: float
    diagnostics: Dict[str, Any]
    raw_response: str

class TowerHanoiResponseParser:
    """Enhanced parser with multiple regex patterns"""
    
    def __init__(self):
        # Multiple parsing patterns for robustness
        self.move_patterns = [
            # Standard: "Move disk 1 from peg A to peg C"
            re.compile(r'move\s+(?:disk\s+)?(\d+)\s+from\s+(?:peg\s+)?([abc123])\s+to\s+(?:peg\s+)?([abc123])', re.IGNORECASE),
            
            # Simplified: "Move 1 from A to C"  
            re.compile(r'move\s+(\d+)\s+from\s+([abc123])\s+to\s+([abc123])', re.IGNORECASE),
            
            # Arrow format: "1: A -> C" or "disk 1: A -> C"
            re.compile(r'(?:disk\s+)?(\d+)\s*:\s*([abc123])\s*->\s*([abc123])', re.IGNORECASE),
            
            # Step format: "Step 1: Move disk 1 from A to C"
            re.compile(r'step\s+\d+\s*:\s*move\s+(?:disk\s+)?(\d+)\s+from\s+([abc123])\s+to\s+([abc123])', re.IGNORECASE),
            
            # Numbered: "1. Move disk 1 from A to C"
            re.compile(r'\d+\.\s*move\s+(?:disk\s+)?(\d+)\s+from\s+(?:peg\s+)?([abc123])\s+to\s+(?:peg\s+)?([abc123])', re.IGNORECASE),
            
            # Transfer: "Transfer disk 1 from A to C"
            re.compile(r'transfer\s+(?:disk\s+)?(\d+)\s+from\s+(?:peg\s+)?([abc123])\s+to\s+(?:peg\s+)?([abc123])', re.IGNORECASE),
            
            # Parentheses: "disk 1 (A to C)"
            re.compile(r'disk\s+(\d+)\s*\(\s*([abc123])\s+to\s+([abc123])\s*\)', re.IGNORECASE),
            
            # Simple arrow: "A -> C" (assumes smallest available disk)
            re.compile(r'([abc123])\s*->\s*([abc123])', re.IGNORECASE),
        ]
        
        # Peg mappings
        self.peg_mappings = {
            'a': 'A', '1': 'A', 'left': 'A', 'first': 'A',
            'b': 'B', '2': 'B', 'middle': 'B', 'center': 'B', 'second': 'B', 
            'c': 'C', '3': 'C', 'right': 'C', 'third': 'C'
        }
    
    def normalize_peg(self, peg_str: str) -> Optional[str]:
        """Normalize peg name to A, B, or C"""
        if not peg_str:
            return None
        return self.peg_mappings.get(peg_str.lower(), peg_str.upper())
    
    def parse_response(self, response: str, max_disks: int = 10) -> ParsingResult:
        """Parse LLM response with multiple strategies"""
        if not response or not response.strip():
            return ParsingResult(
                moves=[], parsed_moves=[], result_type=ParseResult.EMPTY_RESPONSE,
                confidence=0.0, diagnostics={'error': 'Empty response'}, raw_response=response
            )
        
        # Try to extract from <answer> tags first
        if '<answer>' in response and '</answer>' in response:
            answer_text = response.split('<answer>')[1].split('</answer>')[0].strip()
        else:
            answer_text = response.strip()
        
        parsed_moves = []
        lines = answer_text.split('\n')
        
        # Try each line with all patterns
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            best_match = None
            best_confidence = 0.0
            
            # Try all patterns
            for pattern_idx, pattern in enumerate(self.move_patterns):
                match = pattern.search(line)
                if match:
                    confidence = self._calculate_confidence(pattern_idx, match, line)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        
                        if len(match.groups()) == 3:
                            disk, source, target = match.groups()
                            try:
                                disk_num = int(disk)
                                norm_source = self.normalize_peg(source)
                                norm_target = self.normalize_peg(target)
                                
                                if 1 <= disk_num <= max_disks and norm_source and norm_target:
                                    best_match = ParsedMove(
                                        disk=disk_num,
                                        source=norm_source,
                                        target=norm_target,
                                        confidence=confidence,
                                        original_text=line,
                                        line_number=line_num
                                    )
                            except ValueError:
                                continue
                        elif len(match.groups()) == 2:  # Simple arrow format
                            source, target = match.groups()
                            norm_source = self.normalize_peg(source)
                            norm_target = self.normalize_peg(target)
                            
                            if norm_source and norm_target:
                                best_match = ParsedMove(
                                    disk=None,  # Will be inferred
                                    source=norm_source,
                                    target=norm_target,
                                    confidence=confidence * 0.7,  # Lower confidence for inferred moves
                                    original_text=line,
                                    line_number=line_num
                                )
            
            if best_match:
                parsed_moves.append(best_match)
        
        # Convert to moves
        valid_moves = []
        for parsed_move in parsed_moves:
            move = parsed_move.to_move()
            if move:
                valid_moves.append(move)
        
        # Calculate overall results
        if not parsed_moves:
            result_type = ParseResult.FAILED
            confidence = 0.0
        elif len(valid_moves) == len(parsed_moves):
            result_type = ParseResult.SUCCESS
            confidence = np.mean([pm.confidence for pm in parsed_moves])
        else:
            result_type = ParseResult.PARTIAL
            confidence = np.mean([pm.confidence for pm in parsed_moves]) * 0.8
        
        diagnostics = {
            'total_lines': len(lines),
            'parsed_lines': len(parsed_moves),
            'valid_moves': len(valid_moves),
            'avg_confidence': confidence,
            'patterns_used': list(set([pm.line_number for pm in parsed_moves]))
        }
        
        return ParsingResult(
            moves=valid_moves,
            parsed_moves=parsed_moves,
            result_type=result_type,
            confidence=confidence,
            diagnostics=diagnostics,
            raw_response=response
        )
    
    def _calculate_confidence(self, pattern_idx: int, match, line: str) -> float:
        """Calculate confidence score based on pattern and context"""
        base_confidence = [0.95, 0.90, 0.85, 0.88, 0.92, 0.87, 0.83, 0.70][pattern_idx]
        
        # Adjust based on line context
        line_lower = line.lower()
        if any(word in line_lower for word in ['step', 'move', 'disk']):
            base_confidence += 0.05
        if any(word in line_lower for word in ['solution', 'answer']):
            base_confidence += 0.03
            
        return min(1.0, base_confidence)

class TowerState:
    """Enhanced Tower of Hanoi state management"""
    
    def __init__(self, pegs: Dict[str, List[int]]):
        self.pegs = {k: list(v) for k, v in pegs.items()}
    
    def copy(self) -> 'TowerState':
        return TowerState(self.pegs)
    
    def is_valid_move(self, from_peg: str, to_peg: str) -> bool:
        """Validate move according to Tower of Hanoi rules"""
        if not self.pegs[from_peg]:
            return False
        if not self.pegs[to_peg]:
            return True
        return self.pegs[from_peg][-1] < self.pegs[to_peg][-1]
    
    def make_move(self, from_peg: str, to_peg: str) -> bool:
        """Execute move if valid"""
        if not self.is_valid_move(from_peg, to_peg):
            return False
        disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk)
        return True
    
    def is_solved(self, target_peg: str, num_disks: int) -> bool:
        """Check if puzzle is solved"""
        return (len(self.pegs[target_peg]) == num_disks and 
                self.pegs[target_peg] == list(range(num_disks, 0, -1)))
    
    def get_top_disk(self, peg: str) -> Optional[int]:
        """Get top disk on peg"""
        return self.pegs[peg][-1] if self.pegs[peg] else None
    
    def __str__(self) -> str:
        return f"A={self.pegs['A']}, B={self.pegs['B']}, C={self.pegs['C']}"

class TowerHanoiSolver:
    """Enhanced Tower of Hanoi solver with validation"""
    
    @staticmethod
    def solve(num_disks: int, source: str = 'A', target: str = 'C', auxiliary: str = 'B') -> List[Move]:
        """Generate optimal solution"""
        moves = []
        TowerHanoiSolver._hanoi_recursive(num_disks, source, target, auxiliary, moves)
        return moves
    
    @staticmethod
    def _hanoi_recursive(n: int, src: str, tgt: str, aux: str, moves: List[Move]):
        """Recursive helper"""
        if n == 1:
            moves.append(Move(1, src, tgt))
        else:
            TowerHanoiSolver._hanoi_recursive(n-1, src, aux, tgt, moves)
            moves.append(Move(n, src, tgt))
            TowerHanoiSolver._hanoi_recursive(n-1, aux, tgt, src, moves)
    
    @staticmethod
    def validate_solution(moves: List[Move], num_disks: int) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced validation with detailed diagnostics"""
        if not moves:
            return False, "No moves provided", {}
        
        # Initialize state
        state = TowerState({
            'A': list(range(num_disks, 0, -1)),
            'B': [],
            'C': []
        })
        
        diagnostics = {
            'total_moves': len(moves),
            'optimal_moves': (2**num_disks) - 1,
            'move_history': [],
            'error_move': None,
            'final_state': None
        }
        
        # Execute moves
        for i, move in enumerate(moves):
            if not state.is_valid_move(move.from_peg, move.to_peg):
                diagnostics['error_move'] = i + 1
                diagnostics['final_state'] = str(state)
                return False, f"Invalid move {i+1}: {move}. State: {state}", diagnostics
            
            if not state.make_move(move.from_peg, move.to_peg):
                diagnostics['error_move'] = i + 1
                diagnostics['final_state'] = str(state)
                return False, f"Failed to execute move {i+1}: {move}", diagnostics
            
            diagnostics['move_history'].append(str(state))
        
        # Check if solved
        diagnostics['final_state'] = str(state)
        if state.is_solved('C', num_disks):
            efficiency = diagnostics['optimal_moves'] / len(moves) if moves else 0
            return True, f"Solved in {len(moves)} moves (optimal: {diagnostics['optimal_moves']}, efficiency: {efficiency:.2f})", diagnostics
        else:
            return False, f"Not solved. Final state: {state}", diagnostics

class RobustTowerHanoiTask(BaseTask):
    """Enhanced Tower of Hanoi task with robust parsing and validation"""
    
    def __init__(self, model_handler, output_dir, num_disks_list, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        self._task_name = "tower_hanoi_results"
        super().__init__(model_handler, output_dir, 1, 10, num_folds, num_samples, 
                        store_details, temperature, top_p, max_tokens, seed)
        self.num_disks_list = num_disks_list
        self.parser = TowerHanoiResponseParser()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    @property
    def task_name(self):
        """Return task name"""
        return self._task_name
    
    def generate_data(self, **kwargs):
        """Generate evaluation data"""
        num_disks = kwargs.get('list_size', random.choice(self.num_disks_list))
        return [self.generate_problem(num_disks) for _ in range(self.num_samples)]
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response"""
        result = self.parse_response(response, data_point)
        return {
            'accuracy': 1 if result['is_correct'] else 0,
            'instruction_followed': 1 if result.get('parsing_confidence', 0) > 0 else 0,
            'predicted_answer': result.get('parsed_moves', []),
            'ground_truth': data_point.get('solution_moves', []),
            'parsing_info': {
                'method': result.get('parsing_method', 'unknown'),
                'confidence': result.get('parsing_confidence', 0.0),
                'attempts': result.get('parsing_attempts', [])
            },
            'validation_info': result.get('validation_info', {})
        }
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for prompt"""
        return len(prompt) // 4
    
    def generate_problem(self, num_disks: int) -> Dict[str, Any]:
        """Generate Tower of Hanoi problem"""
        initial_state = TowerState({
            'A': list(range(num_disks, 0, -1)),
            'B': [],
            'C': []
        })
        
        solution_moves = TowerHanoiSolver.solve(num_disks)
        
        return {
            'num_disks': num_disks,
            'initial_state': initial_state.pegs,
            'solution_moves': [(m.disk, m.from_peg, m.to_peg) for m in solution_moves],
            'num_optimal_moves': len(solution_moves)
        }
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create enhanced prompt"""
        initial = problem['initial_state']
        num_disks = problem['num_disks']
        
        prompt = f"""Solve this Tower of Hanoi puzzle with {num_disks} disks.

RULES:
1. Only one disk can be moved at a time
2. A larger disk cannot be placed on top of a smaller disk
3. Only the topmost disk on any peg can be moved

INITIAL STATE:
Peg A: {initial['A']} (disk 1 is smallest, disk {num_disks} is largest)
Peg B: {initial['B']}
Peg C: {initial['C']}

GOAL: Move all disks from Peg A to Peg C.

Provide the complete sequence of moves. You can use any of these formats:
- "Move disk X from peg Y to peg Z"
- "Move X from Y to Z"
- "X: Y -> Z"
- "Transfer disk X from Y to Z"

<answer>
Move disk 1 from peg A to peg C
Move disk 2 from peg A to peg B
...
</answer>"""
        
        return prompt
    
    def parse_response(self, response: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced response parsing with detailed diagnostics"""
        try:
            parsing_result = self.parser.parse_response(response, problem['num_disks'])
            
            if parsing_result.moves:
                is_correct, validation_msg, diagnostics = TowerHanoiSolver.validate_solution(
                    parsing_result.moves, problem['num_disks']
                )
            else:
                is_correct = False
                validation_msg = "No valid moves parsed"
                diagnostics = {}
            
            return {
                'parsed_moves': [(m.disk, m.from_peg, m.to_peg) for m in parsing_result.moves],
                'num_moves': len(parsing_result.moves),
                'is_correct': is_correct,
                'validation_info': validation_msg,
                'parsing_confidence': parsing_result.confidence,
                'parsing_diagnostics': parsing_result.diagnostics,
                'validation_diagnostics': diagnostics,
                'raw_response': response
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return {
                'parsed_moves': [],
                'num_moves': 0,
                'is_correct': False,
                'validation_info': f"Parse error: {str(e)}",
                'parsing_confidence': 0.0,
                'raw_response': response
            }
    
    def generate_test_cases(self, fold: int) -> List[Dict[str, Any]]:
        """Generate test cases with token filtering"""
        test_cases = []
        
        for num_disks in self.num_disks_list:
            for _ in range(self.num_samples // len(self.num_disks_list)):
                problem = self.generate_problem(num_disks)
                prompt = self.create_prompt(problem)
                if self.estimate_tokens(prompt) < self.max_tokens // 2:
                    test_cases.append(problem)
        
        return test_cases
    
    def evaluate_fold(self, fold: int) -> Dict[str, Any]:
        """Evaluate with enhanced metrics"""
        self.logger.info(f"Starting evaluation fold {fold + 1}/{self.num_folds}")
        
        test_cases = self.generate_test_cases(fold)
        results = []
        correct_count = 0
        total_tokens = 0
        confidence_scores = []
        
        for i, problem in enumerate(test_cases):
            self.logger.info(f"Processing test case {i + 1}/{len(test_cases)} - {problem['num_disks']} disks")
            
            prompt = self.create_prompt(problem)
            
            try:
                response, tokens_used = self.model_handler.generate_response(
                    prompt, self.temperature, self.top_p, self.max_tokens
                )
                total_tokens += tokens_used
                
                evaluation = self.parse_response(response, problem)
                
                if evaluation['is_correct']:
                    correct_count += 1
                
                confidence_scores.append(evaluation.get('parsing_confidence', 0.0))
                
                result = {
                    'test_case_id': i + 1,
                    'fold': fold + 1,
                    'problem': problem,
                    'prompt': prompt,
                    'response': response,
                    'evaluation': evaluation,
                    'tokens_used': tokens_used
                }
                
                results.append(result)
                
                if self.store_details:
                    status = '✓' if evaluation['is_correct'] else '✗'
                    confidence = evaluation.get('parsing_confidence', 0.0)
                    self.logger.info(f"Result: {status} {problem['num_disks']} disks "
                                   f"({evaluation['num_moves']} moves, conf: {confidence:.2f}, {tokens_used} tokens)")
                
            except Exception as e:
                self.logger.error(f"Error processing test case {i + 1}: {e}")
        
        accuracy = correct_count / len(results) if results else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            'fold': fold + 1,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(results),
            'avg_tokens_per_problem': total_tokens / len(results) if results else 0,
            'total_tokens': total_tokens,
            'avg_parsing_confidence': avg_confidence,
            'results': results if self.store_details else []
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation"""
        fold_results = [self.evaluate_fold(fold) for fold in range(self.num_folds)]
        
        all_accuracies = [result['accuracy'] for result in fold_results]
        all_tokens = [result['avg_tokens_per_problem'] for result in fold_results]
        all_confidences = [result['avg_parsing_confidence'] for result in fold_results]
        
        return {
            'task_name': self.task_name,
            'model_id': getattr(self.model_handler, 'model_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_disks_list': self.num_disks_list,
                'num_folds': self.num_folds,
                'num_samples': self.num_samples,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'max_tokens': self.max_tokens,
                'seed': self.seed
            },
            'overall_accuracy': np.mean(all_accuracies),
            'accuracy_std': np.std(all_accuracies),
            'avg_tokens_per_problem': np.mean(all_tokens),
            'avg_parsing_confidence': np.mean(all_confidences),
            'fold_results': fold_results
        }


class TowerHanoiTask(BaseTask):
    """Tower of Hanoi task that properly inherits from BaseTask for consistent evaluation"""
    
    @property
    def task_name(self):
        return "tower_hanoi"
    
    def generate_data(self, list_size=3):
        """Generate Tower of Hanoi problem data
        
        Args:
            list_size: Number of disks (3, 4, 5, etc.)
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        data = []
        for _ in range(self.num_samples):
            # Generate Tower of Hanoi problem
            num_disks = list_size
            initial_state = {
                'A': list(range(num_disks, 0, -1)),  # [3, 2, 1] for 3 disks
                'B': [],
                'C': []
            }
            
            # Calculate optimal solution
            optimal_moves = self._solve_hanoi(num_disks, 'A', 'C', 'B')
            
            data_point = {
                'num_disks': num_disks,
                'initial_state': initial_state,
                'target_peg': 'C',
                'optimal_moves': optimal_moves,
                'optimal_move_count': len(optimal_moves),
                'answer': optimal_moves  # For evaluation
            }
            data.append(data_point)
        
        return data
    
    def _solve_hanoi(self, n, source, target, auxiliary):
        """Solve Tower of Hanoi recursively"""
        if n == 1:
            return [f"Move disk 1 from {source} to {target}"]
        
        moves = []
        # Move n-1 disks from source to auxiliary
        moves.extend(self._solve_hanoi(n-1, source, auxiliary, target))
        # Move the largest disk from source to target
        moves.append(f"Move disk {n} from {source} to {target}")
        # Move n-1 disks from auxiliary to target
        moves.extend(self._solve_hanoi(n-1, auxiliary, target, source))
        
        return moves
    
    def create_prompt(self, data_point):
        """Create Tower of Hanoi prompt"""
        num_disks = data_point['num_disks']
        initial_state = data_point['initial_state']
        
        prompt = f"""Tower of Hanoi Problem:

You have {num_disks} disks of different sizes on peg A, arranged from largest (bottom) to smallest (top).
Your goal is to move all disks to peg C, following these rules:
1. You can only move one disk at a time
2. You can only move the top disk from a peg
3. A larger disk cannot be placed on top of a smaller disk

Initial state:
Peg A: {initial_state['A']} (bottom to top)
Peg B: {initial_state['B']}
Peg C: {initial_state['C']}

Target: Move all disks to peg C

Please provide the sequence of moves needed to solve this puzzle. Format each move as:
"Move disk X from Y to Z"

Solution:"""

        return prompt
    
    def evaluate_response(self, response, data_point):
        """Evaluate the Tower of Hanoi solution"""
        try:
            # Use the robust parser to extract moves
            parser = TowerHanoiResponseParser()
            parsing_result = parser.parse_response(response)
            
            if parsing_result.result_type == ParseResult.SUCCESS:
                predicted_moves = parsing_result.moves
                optimal_moves = data_point['optimal_moves']
                
                # Check if solution is correct
                is_correct = self._validate_solution(
                    predicted_moves, 
                    data_point['num_disks'],
                    data_point['initial_state']
                )
                
                # Check instruction following (proper format)
                instruction_followed = parsing_result.confidence > 0.7
                
                return {
                    'accuracy': 1 if is_correct else 0,
                    'instruction_followed': 1 if instruction_followed else 0,
                    'predicted_answer': [str(move) for move in predicted_moves],
                    'ground_truth': optimal_moves,
                    'parsing_confidence': parsing_result.confidence,
                    'parsing_diagnostics': parsing_result.diagnostics,
                    'move_count': len(predicted_moves),
                    'optimal_move_count': len(optimal_moves),
                    'efficiency_ratio': len(optimal_moves) / max(len(predicted_moves), 1)
                }
            else:
                # Parsing failed
                return {
                    'accuracy': 0,
                    'instruction_followed': 0,
                    'predicted_answer': None,
                    'ground_truth': data_point['optimal_moves'],
                    'parsing_confidence': parsing_result.confidence,
                    'parsing_diagnostics': parsing_result.diagnostics,
                    'parsing_error': f"Failed to parse: {parsing_result.result_type.value}"
                }
                
        except Exception as e:
            return {
                'accuracy': 0,
                'instruction_followed': 0,
                'predicted_answer': None,
                'ground_truth': data_point['optimal_moves'],
                'evaluation_error': str(e)
            }
    
    def _validate_solution(self, moves, num_disks, initial_state):
        """Validate if the sequence of moves solves the puzzle correctly"""
        try:
            # Initialize state
            state = {
                'A': initial_state['A'].copy(),
                'B': initial_state['B'].copy(), 
                'C': initial_state['C'].copy()
            }
            
            # Apply each move
            for move in moves:
                if not self._apply_move(state, move):
                    return False
            
            # Check if all disks are on peg C in correct order
            expected_final = list(range(num_disks, 0, -1))
            return (state['C'] == expected_final and 
                    len(state['A']) == 0 and 
                    len(state['B']) == 0)
            
        except Exception:
            return False
    
    def _apply_move(self, state, move):
        """Apply a single move and check if it's valid"""
        try:
            # Extract disk number and pegs from move
            if hasattr(move, 'disk') and hasattr(move, 'from_peg') and hasattr(move, 'to_peg'):
                disk = move.disk
                from_peg = move.from_peg
                to_peg = move.to_peg
            else:
                return False
            
            # Check if from_peg has disks
            if not state[from_peg]:
                return False
            
            # Check if the top disk matches
            if state[from_peg][-1] != disk:
                return False
            
            # Check if move is valid (larger disk not on smaller)
            if state[to_peg] and state[to_peg][-1] < disk:
                return False
            
            # Apply the move
            moved_disk = state[from_peg].pop()
            state[to_peg].append(moved_disk)
            
            return True
            
        except Exception:
            return False

def main():
    """Main evaluation function with proper logging and reporting"""
    parser = argparse.ArgumentParser(description="Tower of Hanoi Reasoning Task Evaluation")
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model identifier')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'], help='Engine to use')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE, help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION, help='GPU memory utilization')
    parser.add_argument('--trust_remote_code', type=bool, default=TRUST_REMOTE_CODE, help='Trust remote code')
    
    # Task configuration  
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS, help='Number of data points per fold')
    parser.add_argument('--folds', type=int, default=FOLDS, help='Number of evaluation folds')
    parser.add_argument('--num_disks', type=str, default=','.join(map(str, NUM_DISKS)), help='Comma-separated list of disk numbers')
    parser.add_argument('--store_details', type=bool, default=STORE_DETAILS, help='Store detailed results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=TOP_P, help='Top-p sampling')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='tower_hanoi_results', help='Output directory')

    # API configuration
    parser.add_argument('--api_provider', type=str, choices=['openai', 'gemini'], help='API provider for cloud models')
    parser.add_argument('--api_key', type=str, help='API key for cloud models')

    # Reasoning configuration
    parser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'],
                       help='OpenAI reasoning effort level (for GPT-5 models)')
    parser.add_argument('--thinking_budget', type=int,
                       help='Gemini thinking budget (0 to disable, -1 for dynamic, or specific number)')

    args = parser.parse_args()
    
    # Parse disk numbers
    num_disks_list = [int(x.strip()) for x in args.num_disks.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("🗼 Starting Tower of Hanoi Reasoning Task Evaluation")
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
        logging.info(f"🎯 Initializing Tower of Hanoi task")
        task = TowerHanoiTask(
            model_handler=model_handler,
            output_dir=args.output_dir,
            min_val=min(num_disks_list),
            max_val=max(num_disks_list), 
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
        logging.info(f"🚀 Starting evaluation with disk numbers: {num_disks_list}")
        all_metrics = task.run_evaluation(list_sizes=num_disks_list)
        
        # Generate final report with individual test case breakdown
        logging.info("📊 Generating final report...")

        # Import the reconstruction function and use it directly
        from reporting import reconstruct_individual_metrics

        # Try to reconstruct individual metrics from detailed results
        reconstructed_metrics = reconstruct_individual_metrics(args.output_dir, 'tower_hanoi', num_disks_list)

        if reconstructed_metrics:
            # Use reconstructed metrics to show individual test cases
            logging.info(f"📈 Found {len(reconstructed_metrics)} individual test case metrics")
            final_report = generate_final_report(reconstructed_metrics, num_disks_list, args.output_dir)
        else:
            # Create individual metrics manually from all_metrics if reconstruction fails
            logging.info("📊 Creating individual metrics manually...")
            manual_metrics = []
            for i, disk_count in enumerate(num_disks_list):
                # Find metrics for this disk count from all_metrics
                disk_metrics = [m for m in all_metrics if m.get('disk_count') == disk_count or
                               (i < len(all_metrics) and all_metrics[i] is m)]

                if not disk_metrics:
                    # Create a default metric for this disk count
                    disk_metrics = [{'accuracy': 0, 'instruction_followed_pct': 0, 'avg_response_length': 0,
                                   'avg_word_count': 0, 'avg_output_tokens': 0}]

                metric = {
                    'task': f"tower_hanoi_{disk_count}",
                    'test_case': i,
                    'complexity': disk_count,
                    'accuracy': np.mean([m.get('accuracy', 0) for m in disk_metrics]),
                    'instruction_followed_pct': np.mean([m.get('instruction_followed_pct', 0) for m in disk_metrics]),
                    'avg_response_length': np.mean([m.get('avg_response_length', 0) for m in disk_metrics]),
                    'avg_word_count': np.mean([m.get('avg_word_count', 0) for m in disk_metrics]),
                    'avg_output_tokens': np.mean([m.get('avg_output_tokens', 0) for m in disk_metrics])
                }
                manual_metrics.append(metric)

            final_report = generate_final_report(manual_metrics, num_disks_list, args.output_dir)
        
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
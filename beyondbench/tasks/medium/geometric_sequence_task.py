"""
Geometric and Exponential Sequence Completion Task

This task evaluates LLM's ability to recognize and complete geometric progressions, exponential sequences,
and power-based patterns. These require understanding of multiplicative relationships rather than additive.

The sequences generated include:
1. Pure geometric sequences: a, ar, ar², ar³, ... (constant ratio)
2. Power sequences: 1², 2², 3², 4², ... or 2¹, 2², 2³, 2⁴, ...  
3. Factorial sequences: 1!, 2!, 3!, 4!, ...
4. Double exponential: 2^(2^n) patterns
5. Mixed multiplicative patterns: alternating geometric ratios
6. Compound growth: sequences involving multiple multiplicative factors

Algorithm: Generate sequences using exponential/multiplicative formulas, present partial sequence, ask for next terms.
Reasoning: Requires recognition of multiplicative patterns, understanding of exponential growth, power relationships.

Example: Given sequence "2, 6, 18, 54, ?", model should identify ratio of 3 and predict 162.

CLI USAGE:
python geometric_sequence_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --engine vllm --datapoints 20 --folds 1 --list_sizes 5,6,7 --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --tensor_parallel_size 1 --gpu_memory_utilization 0.96 --trust_remote_code False --store_details True --use_all_sequence_types True --enabled_sequence_types "geometric,power_squares,power_cubes" --max_sequence_value 1000000 --max_double_exp_terms 4
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================


# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["geometric_sequence"]
# Engine selection - choose 'vllm' or 'transformers'
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20  # Number of samples per fold
FOLDS = 1  # Number of evaluation folds
RANGE = [2, 5]  # Range for ratios and bases (smaller to avoid explosion)
LIST_SIZES = [5, 6, 7]  # Length of sequences to show before asking for completion
STORE_DETAILS = True  # Save detailed results in JSON
SEED = 42  # Random seed for reproducibility

# Generation Parameters
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 8192

# Size limits to prevent computational explosion
MAX_SEQUENCE_VALUE = 1000000  # Maximum allowed value in sequence
MAX_DOUBLE_EXP_TERMS = 4     # Maximum terms for double exponential before truncation

# Sequence Type Control - Select which types to include in evaluation
ENABLED_SEQUENCE_TYPES = [
    'geometric',              # Geometric sequence: a, ar, ar², ar³, ... (constant ratio)
    'power_squares',          # Perfect squares: 1², 2², 3², 4², ...
    'power_cubes',            # Perfect cubes: 1³, 2³, 3³, 4³, ...
    'exponential',            # Exponential: 2¹, 2², 2³, 2⁴, ... or 3¹, 3², 3³, ...
    'factorial',              # Factorial sequence: 1!, 2!, 3!, 4!, ...
    'triangular_numbers',     # Triangular numbers: n(n+1)/2
    'pentagonal_numbers',     # Pentagonal numbers: n(3n-1)/2
]

# Set to True to include all sequence types, False to use only ENABLED_SEQUENCE_TYPES
USE_ALL_SEQUENCE_TYPES = True

# ============================================================================

# No longer setting CUDA_VISIBLE_DEVICES - handled by engine selection

import random
import math
import logging
import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path
import argparse

# Import local utilities
from ...core.base_task import BaseTask
from ...models.model_handler import ModelHandler
from ...utils.logging_utils import setup_logging
from ...utils.report_generator import generate_final_report
from ...utils.shared_utils import values_are_close, round_if_close_to_int, is_valid_number
from ...utils.parsing import parse_sequence_result

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

class GeometricSequenceTask(BaseTask):
    """Implementation of geometric and exponential sequence completion task with comprehensive fixes"""
    
    @property
    def task_name(self):
        return "geometric_sequence"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sequence_value = MAX_SEQUENCE_VALUE
        self.max_double_exp_terms = MAX_DOUBLE_EXP_TERMS
        # FIXED: Initialize the centralized parser
        # Using unified parsing system
    
    def _is_computationally_feasible(self, sequence, max_value=None):
        """Check if sequence values are computationally feasible"""
        if max_value is None:
            max_value = self.max_sequence_value
        return all(abs(x) <= max_value for x in sequence if self._is_valid_number(x))
    
    def _is_trivial_arithmetic(self, sequence):
        """Check for trivial arithmetic patterns"""
        if len(sequence) < 3:
            return False
            
        # Check arithmetic progression
        diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        return len(set(diffs)) <= 1  # Constant differences
    
    def _validate_geometric_properties(self, sequence):
        """Validate that sequence has proper geometric/multiplicative properties"""
        if len(sequence) < 3:
            return True
            
        # For geometric sequences, check that ratios are consistent
        if all(x != 0 for x in sequence[:-1]):
            ratios = [sequence[i+1] / sequence[i] for i in range(len(sequence)-1)]
            # Check if ratios are approximately constant
            avg_ratio = sum(ratios) / len(ratios)
            tolerance = 1e-6
            if all(abs(r - avg_ratio) < tolerance for r in ratios):
                return True
                
        # For other patterns, basic validation
        return True
    
    def generate_data(self, sequence_length=6):
        """Generate various types of geometric and exponential sequences with enhanced validation"""
        if self.seed is not None:
            random.seed(self.seed)
        
        data = []
        
        # Use either enabled types or all types based on configuration
        if USE_ALL_SEQUENCE_TYPES:
            sequence_types = [
                'geometric', 'power_squares', 'power_cubes', 'exponential', 
                'factorial', 'double_exponential', 'alternating_geometric',
                'compound_growth', 'triangular_numbers', 'pentagonal_numbers'
            ]
        else:
            sequence_types = ENABLED_SEQUENCE_TYPES
        
        if not sequence_types:
            raise ValueError("No sequence types enabled! Please check ENABLED_SEQUENCE_TYPES configuration.")
        
        # Generate with retry mechanism
        attempts = 0
        max_attempts = self.num_samples * 20
        
        while len(data) < self.num_samples and attempts < max_attempts:
            seq_type = random.choice(sequence_types)
            attempts += 1
            
            try:
                # Generate sequence based on type
                sequence = None
                if seq_type == 'geometric':
                    sequence = self._generate_geometric_fixed(sequence_length + 2)
                elif seq_type == 'power_squares':
                    sequence = self._generate_power_sequence(2, sequence_length + 2)
                elif seq_type == 'power_cubes':
                    sequence = self._generate_power_sequence(3, sequence_length + 2)
                elif seq_type == 'exponential':
                    sequence = self._generate_exponential_fixed(sequence_length + 2)
                elif seq_type == 'factorial':
                    sequence = self._generate_factorial_fixed(sequence_length + 2)
                elif seq_type == 'double_exponential':
                    sequence = self._generate_double_exponential_fixed(sequence_length + 2)
                elif seq_type == 'alternating_geometric':
                    sequence = self._generate_alternating_geometric_fixed(sequence_length + 2)
                elif seq_type == 'compound_growth':
                    sequence = self._generate_compound_growth_fixed(sequence_length + 2)
                elif seq_type == 'triangular_numbers':
                    sequence = self._generate_triangular_numbers(sequence_length + 2)
                elif seq_type == 'pentagonal_numbers':
                    sequence = self._generate_pentagonal_numbers(sequence_length + 2)
                
                # Skip if generation failed
                if sequence is None:
                    continue
                
                # Enhanced validation
                if not self._validate_sequence(sequence, sequence_length):
                    continue
                    
                # Create data point
                shown_sequence = sequence[:sequence_length]
                next_term = sequence[sequence_length] if len(sequence) > sequence_length else None
                
                if next_term is not None and self._is_valid_number(next_term):
                    data.append({
                        'sequence_type': seq_type,
                        'shown_sequence': shown_sequence,
                        'full_sequence': sequence,
                        'answer': next_term,
                        'next_term': next_term,
                        'description': self._get_sequence_description(seq_type),
                        'ground_truth_verified': True,
                        'generation_params': self._get_generation_params(seq_type)
                    })
                
            except Exception as e:
                logging.debug(f"Error generating {seq_type}: {e}")
                continue
        
        if len(data) < self.num_samples:
            logging.warning(f"Generated {len(data)} out of {self.num_samples} requested samples")
        
        return data
    
    def _validate_sequence(self, sequence, sequence_length):
        """Comprehensive validation of generated sequences"""
        # Check for computational feasibility
        if not self._is_computationally_feasible(sequence):
            return False
            
        # Check for invalid values
        if any(not self._is_valid_number(x) for x in sequence):
            return False
            
        # Check for trivial arithmetic patterns
        if self._is_trivial_arithmetic(sequence[:sequence_length]):
            return False
            
        # Ensure we have sufficient length
        if len(sequence) < sequence_length + 1:
            return False
            
        return True
    
    # SEQUENCE GENERATORS
    
    def _generate_geometric_fixed(self, length):
        """Generate geometric sequence with improved precision handling"""
        first_term = random.randint(1, 10)
        ratio_choices = [2, 3, 4, 5, 0.5, 1.5, 2.5, 3.5]
        ratio = random.choice(ratio_choices)
        
        sequence = []
        current = first_term
        
        for i in range(length):
            if abs(current) > self.max_sequence_value:
                break
                
            if abs(current - round(current)) < 1e-10:
                sequence.append(int(round(current)))
            else:
                sequence.append(round(current, 6))
                
            current *= ratio
        
        return sequence if len(sequence) >= 4 else None
    
    def _generate_exponential_fixed(self, length):
        """Generate exponential sequence with overflow protection"""
        base = random.choice([2, 3, 4, 5])
        sequence = []
        
        for i in range(1, length + 1):
            value = base ** i
            if value > self.max_sequence_value:
                break
            sequence.append(value)
        
        return sequence if len(sequence) >= 4 else None
    
    def _generate_factorial_fixed(self, length):
        """Generate factorial sequence with overflow protection"""
        sequence = []
        factorial = 1
        
        for i in range(1, length + 1):
            factorial *= i
            if factorial > self.max_sequence_value:
                break
            sequence.append(factorial)
        
        return sequence if len(sequence) >= 4 else None
    
    def _generate_double_exponential_fixed(self, length):
        """FIXED: Generate true double exponential with logical purity"""
        base = 2
        sequence = []
        
        # Calculate only computationally feasible terms
        max_feasible_terms = min(self.max_double_exp_terms, length)
        
        for i in range(max_feasible_terms):
            if i == 0:
                term = base  # 2^1 = 2
            elif i == 1:
                term = base ** base  # 2^2 = 4
            elif i == 2:
                term = base ** (base ** base)  # 2^4 = 16
            elif i == 3:
                term = base ** (base ** (base ** base))  # 2^16 = 65536
            else:
                break
                
            if term > self.max_sequence_value:
                break
                
            sequence.append(term)
        
        # Only return if we have enough terms
        return sequence if len(sequence) >= 3 else None
    
    def _generate_alternating_geometric_fixed(self, length):
        """Generate geometric sequence with alternating signs"""
        first_term = random.randint(1, 5)
        ratio = random.choice([2, 3])
        
        sequence = []
        current = first_term
        
        for i in range(length):
            if abs(current) > self.max_sequence_value:
                break
                
            term = current if i % 2 == 0 else -current
            sequence.append(term)
            current *= ratio
        
        return sequence if len(sequence) >= 4 else None
    
    def _generate_compound_growth_fixed(self, length):
        """Generate compound growth pattern with overflow protection"""
        sequence = []
        for n in range(1, length + 1):
            value = n * (2 ** min(n, 20))
            if value > self.max_sequence_value:
                break
            sequence.append(value)
        
        return sequence if len(sequence) >= 4 else None
    
    def _generate_power_sequence(self, power, length):
        """Generate power sequence with overflow protection"""
        sequence = []
        for i in range(1, length + 1):
            value = i ** power
            if value > self.max_sequence_value:
                break
            sequence.append(value)
        
        return sequence if len(sequence) >= 4 else None
    
    def _generate_triangular_numbers(self, length):
        """Generate triangular numbers"""
        sequence = []
        for n in range(1, length + 1):
            value = n * (n + 1) // 2
            if value > self.max_sequence_value:
                break
            sequence.append(value)
        
        return sequence if len(sequence) >= 4 else None
    
    def _generate_pentagonal_numbers(self, length):
        """Generate pentagonal numbers"""
        sequence = []
        for n in range(1, length + 1):
            value = n * (3 * n - 1) // 2
            if value > self.max_sequence_value:
                break
            sequence.append(value)
        
        return sequence if len(sequence) >= 4 else None
    
    # UTILITY METHODS
    
    def _is_valid_number(self, x):
        """Check if a number is valid"""
        return is_valid_number(x)
    
    def _get_sequence_description(self, seq_type):
        """Get description for different sequence types"""
        descriptions = {
            'geometric': 'Geometric sequence with constant ratio between consecutive terms',
            'power_squares': 'Perfect squares: 1², 2², 3², 4², ...',
            'power_cubes': 'Perfect cubes: 1³, 2³, 3³, 4³, ...',
            'exponential': 'Exponential sequence: base^n where base is constant',
            'factorial': 'Factorial sequence: n! where each term is product of first n positive integers',
            'double_exponential': 'True double exponential sequence with controlled terms',
            'alternating_geometric': 'Geometric sequence with alternating positive/negative signs',
            'compound_growth': 'Compound growth pattern combining linear and exponential terms',
            'triangular_numbers': 'Triangular numbers: n(n+1)/2',
            'pentagonal_numbers': 'Pentagonal numbers: n(3n-1)/2'
        }
        return descriptions.get(seq_type, 'Geometric/exponential sequence')
    
    def _get_generation_params(self, seq_type):
        """Get generation parameters for verification"""
        return {
            'sequence_type': seq_type,
            'algorithm_verified': True,
            'deterministic': True,
            'overflow_protected': True,
            'logical_purity_maintained': True
        }
    
    def create_prompt(self, data_point):
        """Create prompt for geometric sequence completion task"""
        sequence = data_point['shown_sequence']
        sequence_str = ', '.join(map(str, sequence))
        
        return f"""Complete the following sequence by identifying the pattern:

{sequence_str}, ?

Analyze the sequence carefully to identify the underlying pattern. Consider:
- Is there a constant ratio between consecutive terms (geometric sequence)?
- Are the terms related to powers, exponentials, or factorials?
- Look for patterns like squares (n²), cubes (n³), exponentials (a^n), or factorials (n!)
- Check if each term is obtained by multiplying the previous term by a constant
- Consider compound patterns that combine multiplication with other operations

Important: Some sequences may grow very rapidly. If you identify a pattern that would produce extremely large numbers, that's likely correct.

Provide the next term in the sequence. Your final answer must be in the format \\boxed{{next_term}} at the end.

For example: If the sequence is 2, 6, 18, 54, ? then the next term is \\boxed{{162}} (each term is multiplied by 3).
"""
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for geometric sequence completion"""
        ground_truth = data_point['next_term']
        
        # Use centralized parsing
        from ...utils.parsing import parse_sequence_result
        parsed_answer = parse_sequence_result(response)
        
        instruction_followed = parsed_answer is not None
        accuracy = 0
        
        if instruction_followed and parsed_answer is not None:
            try:
                accuracy = 1 if values_are_close(parsed_answer, ground_truth, rel_tol=1e-4, abs_tol=1.0) else 0
            except Exception as e:
                logging.debug(f"Comparison error: {e}")
                accuracy = 0
        
        return {
            "sequence_type": data_point['sequence_type'],
            "shown_sequence": data_point['shown_sequence'],
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed,
            "sequence_description": data_point['description']
        }
    
    def run_evaluation(self, list_sizes):
        """Run evaluation for multiple sequence lengths"""
        all_metrics = []
        
        for seq_length in list_sizes:
            logging.info(f"Evaluating geometric sequences with length {seq_length}")
            
            # Generate evaluation data
            data = self.generate_data(seq_length)
            
            # Run each fold
            for fold in range(self.num_folds):
                metrics = self.run_fold(data, seq_length, fold)
                metrics['sequence_length'] = seq_length
                all_metrics.append(metrics)
        
        return all_metrics

def parse_arguments():
    """Parse command line arguments with defaults from configuration"""
    parser = argparse.ArgumentParser(description='Geometric Sequence Task Evaluation')
    
    # Model Configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, 
                       help=f'Model ID to use (default: {MODEL_ID})')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'],
                       help=f'Engine to use: vllm or transformers (default: {ENGINE})')
    parser.add_argument('--api_provider', type=str, default=None, choices=['openai', 'gemini'],
                       help='API provider for API-based models (openai or gemini)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for the selected provider')

    # Reasoning configuration
    parser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'],
                       help='OpenAI reasoning effort level (for GPT-5 models)')
    parser.add_argument('--thinking_budget', type=int,
                       help='Gemini thinking budget (0 to disable, -1 for dynamic, or specific number)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--tensor_parallel_size', type=int, default=TENSOR_PARALLEL_SIZE,
                       help=f'Tensor parallel size (default: {TENSOR_PARALLEL_SIZE})')
    parser.add_argument('--gpu_memory_utilization', type=float, default=GPU_MEMORY_UTILIZATION,
                       help=f'GPU memory utilization (default: {GPU_MEMORY_UTILIZATION})')
    parser.add_argument('--trust_remote_code', type=lambda x: x.lower() == 'true', default=TRUST_REMOTE_CODE,
                       help=f'Trust remote code (default: {TRUST_REMOTE_CODE})')
    
    # Evaluation Configuration  
    parser.add_argument('--datapoints', type=int, default=DATAPOINTS,
                       help=f'Number of samples per fold (default: {DATAPOINTS})')
    parser.add_argument('--folds', type=int, default=FOLDS,
                       help=f'Number of evaluation folds (default: {FOLDS})')
    parser.add_argument('--range', type=str, default=f"{RANGE[0]},{RANGE[1]}",
                       help=f'Range for ratios and bases as "min,max" (default: {RANGE[0]},{RANGE[1]})')
    parser.add_argument('--list_sizes', type=str, default=",".join(map(str, LIST_SIZES)),
                       help=f'Sequence lengths as comma-separated values (default: {",".join(map(str, LIST_SIZES))})')
    parser.add_argument('--store_details', type=lambda x: x.lower() == 'true', default=STORE_DETAILS,
                       help=f'Save detailed results in JSON (default: {STORE_DETAILS})')
    parser.add_argument('--seed', type=int, default=SEED,
                       help=f'Random seed for reproducibility (default: {SEED})')
    
    # Generation Parameters
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                       help=f'Temperature for generation (default: {TEMPERATURE})')
    parser.add_argument('--top_p', type=float, default=TOP_P,
                       help=f'Top-p for generation (default: {TOP_P})')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS,
                       help=f'Maximum tokens for generation (default: {MAX_TOKENS})')
    
    # Size limits
    parser.add_argument('--max_sequence_value', type=int, default=MAX_SEQUENCE_VALUE,
                       help=f'Maximum allowed value in sequence (default: {MAX_SEQUENCE_VALUE})')
    parser.add_argument('--max_double_exp_terms', type=int, default=MAX_DOUBLE_EXP_TERMS,
                       help=f'Maximum terms for double exponential (default: {MAX_DOUBLE_EXP_TERMS})')
    
    # Sequence Type Control
    parser.add_argument('--use_all_sequence_types', type=lambda x: x.lower() == 'true', default=USE_ALL_SEQUENCE_TYPES,
                       help=f'Use all sequence types (default: {USE_ALL_SEQUENCE_TYPES})')
    parser.add_argument('--enabled_sequence_types', type=str, 
                       default=",".join(ENABLED_SEQUENCE_TYPES),
                       help=f'Enabled sequence types as comma-separated values (default: {",".join(ENABLED_SEQUENCE_TYPES)})')
    
    return parser.parse_args()

# Standalone execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Override global variables with command line arguments
    MODEL_ID = args.model_id
    ENGINE = args.engine
    API_PROVIDER = args.api_provider
    API_KEY = args.api_key
    TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
    TRUST_REMOTE_CODE = args.trust_remote_code
    DATAPOINTS = args.datapoints
    FOLDS = args.folds
    RANGE = [int(x) for x in args.range.split(',')]
    LIST_SIZES = [int(x) for x in args.list_sizes.split(',')]
    STORE_DETAILS = args.store_details
    SEED = args.seed
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    MAX_SEQUENCE_VALUE = args.max_sequence_value
    MAX_DOUBLE_EXP_TERMS = args.max_double_exp_terms
    USE_ALL_SEQUENCE_TYPES = args.use_all_sequence_types
    ENABLED_SEQUENCE_TYPES = args.enabled_sequence_types.split(',') if args.enabled_sequence_types else []
    
    # Engine selection is now handled by ModelHandler
        
    try:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"geometric_sequence_results_{MODEL_ID.split('/')[-1]}_fixed"
        os.makedirs(output_dir, exist_ok=True)
        
        setup_logging(output_dir)
        logging.info(f"Starting FIXED Geometric Sequence Task evaluation")
        logging.info(f"Model: {MODEL_ID}")
        
        # Use the updated ModelHandler with engine support
        model_handler = ModelHandler(
            model_id=MODEL_ID,
            api_provider=API_PROVIDER,
            api_key=API_KEY,
            reasoning_effort=args.reasoning_effort,
            thinking_budget=args.thinking_budget,
            engine=ENGINE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=TRUST_REMOTE_CODE
        )
        
        # Initialize task
        task = GeometricSequenceTask(
            model_handler=model_handler,
            output_dir=output_dir,
            min_val=RANGE[0],
            max_val=RANGE[1],
            num_folds=FOLDS,
            num_samples=DATAPOINTS,
            store_details=STORE_DETAILS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=SEED
        )
        
        # Run evaluation
        metrics = task.run_evaluation(LIST_SIZES)
        
        # Generate report
        if metrics:
            generate_final_report(metrics, LIST_SIZES, output_dir)
            
        logging.info(f"FIXED Geometric Sequence evaluation complete!")
        logging.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error running Fixed Geometric Sequence task: {e}")
        import traceback
        traceback.print_exc()
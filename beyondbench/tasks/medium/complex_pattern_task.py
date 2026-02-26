"""
Complex Pattern Recognition Task

This task evaluates LLM's ability to recognize and complete sequences with complex, multi-layered patterns
that require deeper analysis and mathematical reasoning. These patterns often combine multiple mathematical
concepts or have nested structures that are not immediately obvious.

The sequences generated include:
1. Nested arithmetic/geometric patterns: combining different progression types
2. Polynomial sequences: following quadratic, cubic, or higher-order patterns
3. Interleaved sequences: alternating between different mathematical rules
4. Conditional patterns: rules that change based on position or value
5. Recursive with multiple dependencies: F(n) depends on multiple previous terms
6. Pattern transformations: sequences that follow one rule then switch to another
7. Multi-dimensional patterns: considering position, value, and derived properties
8. Modular arithmetic patterns: complex patterns involving remainders and cycles
9. Combinatorial sequences: involving combinations, permutations, or graph theory
10. Chaotic but deterministic: sequences with complex but discoverable rules

Algorithm: Generate sequences with layered mathematical relationships, present partial sequence,
require deep analysis to identify the underlying complex pattern.

Reasoning: Requires multi-step analysis, pattern recognition across multiple layers, mathematical insight.

Example: "1, 2, 2, 4, 3, 6, 4, 8, ?" -> pattern is two interleaved sequences: (1,2,3,4,...) and (2,4,6,8,...)

CLI USAGE:
python complex_pattern_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --engine "vllm" --datapoints 20 --folds 1 --list_sizes 8,10,12 --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --tensor_parallel_size 1 --gpu_memory_utilization 0.96 --trust_remote_code False --store_details True --use_all_sequence_types True --enabled_sequence_types "nested_arithmetic_geometric,polynomial_quadratic,polynomial_cubic"
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["complex_pattern"]
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20  # Number of samples per fold
FOLDS = 1  # Number of evaluation folds
RANGE = [1, 10]  # Range for initial values and coefficients
LIST_SIZES = [8, 10, 12]  # Longer sequences needed for complex patterns
STORE_DETAILS = True  # Save detailed results in JSON
SEED = 42  # Random seed for reproducibility

# Generation Parameters
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 8192

# Sequence Type Control - Select which types to include in evaluation
ENABLED_SEQUENCE_TYPES = [
    'nested_arithmetic_geometric',    # Nested pattern combining arithmetic and geometric progressions
    'polynomial_quadratic',           # Quadratic polynomial sequence following an² + bn + c
    'polynomial_cubic',               # Cubic polynomial sequence following an³ + bn² + cn + d
    'interleaved_sequences',          # Two different sequences interleaved alternately
    'conditional_pattern',            # Pattern that changes based on position or value conditions
    'multi_recursive',                # Recursive sequence depending on three previous terms
    'pattern_transformation',         # Pattern that transforms at a certain breakpoint
    'modular_arithmetic',             # Complex pattern involving modular arithmetic
]

# Set to True to include all sequence types, False to use only ENABLED_SEQUENCE_TYPES
USE_ALL_SEQUENCE_TYPES = True

# ============================================================================

import os

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

class ComplexPatternTask(BaseTask):
    """Implementation of complex pattern recognition task"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the centralized parser
        # Using unified parsing system
    
    @property
    def task_name(self):
        return "complex_pattern"
    
    def generate_data(self, sequence_length=10):
        """
        Generate various types of complex pattern sequences
        
        Args:
            sequence_length: Length of sequence to generate (will show partial)
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        data = []
        
        # Use either enabled types or all types based on configuration
        if USE_ALL_SEQUENCE_TYPES:
            sequence_types = [
                'nested_arithmetic_geometric', 'polynomial_quadratic', 'polynomial_cubic',
                'interleaved_sequences', 'conditional_pattern', 'multi_recursive',
                'pattern_transformation', 'modular_arithmetic', 'alternating_operations',
                'layered_differences', 'position_dependent', 'spiral_pattern'
            ]
        else:
            sequence_types = ENABLED_SEQUENCE_TYPES
        
        if not sequence_types:
            raise ValueError("No sequence types enabled! Please check ENABLED_SEQUENCE_TYPES configuration.")
        
        for _ in range(self.num_samples):
            seq_type = random.choice(sequence_types)
            
            try:
                if seq_type == 'nested_arithmetic_geometric':
                    # Arithmetic sequence where each term is then modified geometrically
                    sequence, next_terms = self._generate_nested_arith_geom(sequence_length + 2)
                    
                elif seq_type == 'polynomial_quadratic':
                    # Quadratic sequence: an² + bn + c
                    sequence, next_terms = self._generate_polynomial_quadratic(sequence_length + 2)
                    
                elif seq_type == 'polynomial_cubic':
                    # Cubic sequence: an³ + bn² + cn + d
                    sequence, next_terms = self._generate_polynomial_cubic(sequence_length + 2)
                    
                elif seq_type == 'interleaved_sequences':
                    # Two sequences interleaved: A₁, B₁, A₂, B₂, A₃, B₃, ...
                    sequence, next_terms = self._generate_interleaved_sequences(sequence_length + 2)
                    
                elif seq_type == 'conditional_pattern':
                    # Pattern changes based on position or value conditions
                    sequence, next_terms = self._generate_conditional_pattern(sequence_length + 2)
                    
                elif seq_type == 'multi_recursive':
                    # F(n) = aF(n-1) + bF(n-2) + cF(n-3)
                    sequence, next_terms = self._generate_multi_recursive(sequence_length + 2)
                    
                elif seq_type == 'pattern_transformation':
                    # Pattern changes at a certain point
                    sequence, next_terms = self._generate_pattern_transformation(sequence_length + 2)
                    
                elif seq_type == 'modular_arithmetic':
                    # Complex modular patterns
                    sequence, next_terms = self._generate_modular_arithmetic(sequence_length + 2)
                    
                elif seq_type == 'alternating_operations':
                    # Alternating between different operations
                    sequence, next_terms = self._generate_alternating_operations(sequence_length + 2)
                    
                elif seq_type == 'layered_differences':
                    # Pattern in differences of differences
                    sequence, next_terms = self._generate_layered_differences(sequence_length + 2)
                    
                elif seq_type == 'position_dependent':
                    # Each position follows a different rule
                    sequence, next_terms = self._generate_position_dependent(sequence_length + 2)
                    
                elif seq_type == 'spiral_pattern':
                    # Pattern based on spiral or matrix-like thinking
                    sequence, next_terms = self._generate_spiral_pattern(sequence_length + 2)
                
                # Skip if sequence is too large or contains errors
                if any(abs(x) > 10000 for x in sequence):
                    continue
                    
                # Create data point - reconstruct full sequence from returned parts
                full_sequence = sequence + next_terms
                shown_sequence = sequence
                
                data.append({
                    'sequence_type': seq_type,
                    'shown_sequence': shown_sequence,
                    'full_sequence': full_sequence,
                    'answer': next_terms,
                    'next_term': next_terms[0] if next_terms else None,
                    'description': self._get_sequence_description(seq_type)
                })
                
            except Exception as e:
                logging.debug(f"Error generating {seq_type}: {e}")
                continue
        
        return data
    
    def _generate_nested_arith_geom(self, length):
        """Generate nested arithmetic-geometric pattern"""
        # Start with arithmetic, then multiply by geometric factor
        arith_start = random.randint(1, 5)
        arith_diff = random.randint(1, 3)
        geom_mult = random.choice([2, 3])
        
        sequence = []
        for i in range(length):
            arith_term = arith_start + i * arith_diff
            geom_factor = geom_mult ** (i % 3)  # Cycle through powers
            sequence.append(arith_term * geom_factor)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_polynomial_quadratic(self, length):
        """Generate quadratic sequence: an² + bn + c"""
        a = random.randint(1, 3)
        b = random.randint(-2, 3)
        c = random.randint(0, 5)
        
        sequence = []
        for n in range(1, length + 1):
            term = a * n * n + b * n + c
            sequence.append(term)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_polynomial_cubic(self, length):
        """Generate cubic sequence: an³ + bn² + cn + d"""
        # Use smaller coefficients to avoid explosion
        a = random.choice([1, -1])
        b = random.randint(-1, 2)
        c = random.randint(-2, 3)
        d = random.randint(0, 5)
        
        sequence = []
        for n in range(1, length + 1):
            term = a * n * n * n + b * n * n + c * n + d
            sequence.append(term)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_interleaved_sequences(self, length):
        """Generate two interleaved sequences"""
        # Sequence A: arithmetic progression
        a_start = random.randint(1, 3)
        a_diff = random.randint(1, 2)
        
        # Sequence B: geometric progression  
        b_start = random.randint(2, 4)
        b_ratio = random.choice([2, 3])
        
        sequence = []
        for i in range(length):
            if i % 2 == 0:  # Even positions: sequence A
                a_term = a_start + (i // 2) * a_diff
                sequence.append(a_term)
            else:  # Odd positions: sequence B
                b_term = b_start * (b_ratio ** (i // 2))
                sequence.append(b_term)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_conditional_pattern(self, length):
        """Generate pattern that changes based on conditions"""
        sequence = []
        base = random.randint(2, 5)
        
        for i in range(length):
            if i < 3:
                # First few terms: simple arithmetic
                sequence.append(base + i)
            elif i % 2 == 0:
                # Even positions after 3: multiply previous by 2
                sequence.append(sequence[-1] * 2)
            else:
                # Odd positions after 3: add difference of last two
                sequence.append(sequence[-1] + (sequence[-1] - sequence[-2]))
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_multi_recursive(self, length):
        """Generate F(n) = aF(n-1) + bF(n-2) + cF(n-3)"""
        # Start with three initial values
        start_vals = [random.randint(1, 3) for _ in range(3)]
        
        # Use small coefficients
        a, b, c = random.choice([1, -1]), random.choice([1, -1]), random.choice([1, 0])
        
        sequence = start_vals.copy()
        for i in range(3, length):
            next_val = a * sequence[i-1] + b * sequence[i-2] + c * sequence[i-3]
            sequence.append(next_val)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_pattern_transformation(self, length):
        """Generate pattern that changes at a breakpoint"""
        breakpoint = length // 2
        sequence = []
        
        # First half: arithmetic progression
        start = random.randint(1, 3)
        diff = random.randint(1, 2)
        
        for i in range(breakpoint):
            sequence.append(start + i * diff)
        
        # Second half: each term is double previous plus constant
        constant = random.randint(1, 3)
        for i in range(breakpoint, length):
            sequence.append(sequence[-1] * 2 + constant)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_modular_arithmetic(self, length):
        """Generate complex modular arithmetic pattern"""
        mod = random.choice([5, 7, 11])
        base = random.randint(2, 4)
        
        sequence = []
        for i in range(length):
            # Pattern: (base^i + i²) mod mod
            term = (pow(base, i+1) + (i+1) * (i+1)) % mod
            sequence.append(term + 1)  # Shift to avoid zeros
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_alternating_operations(self, length):
        """Generate sequence with alternating operations"""
        sequence = [random.randint(2, 5)]
        operations = ['add', 'multiply', 'square']
        
        for i in range(1, length):
            prev = sequence[-1]
            op = operations[i % len(operations)]
            
            if op == 'add':
                sequence.append(prev + i)
            elif op == 'multiply':
                sequence.append(prev * 2)
            elif op == 'square':
                sequence.append(prev + i * i)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_layered_differences(self, length):
        """Generate sequence where pattern is in differences of differences"""
        # Start with sequence where second differences are constant
        a, b, c = random.randint(1, 3), random.randint(1, 4), random.randint(0, 5)
        
        sequence = []
        for i in range(length):
            # Quadratic gives constant second differences
            term = a * i * i + b * i + c
            sequence.append(term)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_position_dependent(self, length):
        """Generate sequence where each position follows different rule"""
        sequence = []
        
        for i in range(length):
            pos = i + 1
            if pos % 3 == 1:  # Positions 1, 4, 7, ...
                sequence.append(pos * 2)
            elif pos % 3 == 2:  # Positions 2, 5, 8, ...
                sequence.append(pos * pos)
            else:  # Positions 3, 6, 9, ...
                sequence.append(pos + 10)
        
        return sequence[:-2], sequence[-2:]
    
    def _generate_spiral_pattern(self, length):
        """Generate pattern based on spiral-like matrix filling"""
        # Imagine filling a matrix in spiral order, extract sequence
        sequence = []
        for i in range(length):
            # Simple spiral-like pattern
            layer = int(math.sqrt(i))
            pos_in_layer = i - layer * layer
            sequence.append(layer * 4 + pos_in_layer + 1)
        
        return sequence[:-2], sequence[-2:]
    
    def _get_sequence_description(self, seq_type):
        """Get description for different sequence types"""
        descriptions = {
            'nested_arithmetic_geometric': 'Nested pattern combining arithmetic and geometric progressions',
            'polynomial_quadratic': 'Quadratic polynomial sequence following an² + bn + c',
            'polynomial_cubic': 'Cubic polynomial sequence following an³ + bn² + cn + d',
            'interleaved_sequences': 'Two different sequences interleaved alternately',
            'conditional_pattern': 'Pattern that changes based on position or value conditions',
            'multi_recursive': 'Recursive sequence depending on three previous terms',
            'pattern_transformation': 'Pattern that transforms at a certain breakpoint',
            'modular_arithmetic': 'Complex pattern involving modular arithmetic',
            'alternating_operations': 'Sequence with alternating mathematical operations',
            'layered_differences': 'Pattern visible in second or higher-order differences',
            'position_dependent': 'Each position follows a different mathematical rule',
            'spiral_pattern': 'Pattern based on spiral or matrix-like arrangement'
        }
        return descriptions.get(seq_type, 'Complex mathematical pattern')
    
    def _values_are_close(self, val1, val2, rel_tol=1e-4, abs_tol=1.0):
        """Check if two values are close enough"""
        return values_are_close(val1, val2, rel_tol, abs_tol)
    
    def create_prompt(self, data_point):
        """Create prompt for complex pattern completion task"""
        sequence = data_point['shown_sequence']
        sequence_str = ', '.join(map(str, sequence))
        
        return f"""Complete the following complex sequence by identifying the underlying pattern:

{sequence_str}, ?

This sequence follows a complex mathematical pattern that may involve:
- Multiple layers of relationships (nested patterns)
- Polynomial sequences (quadratic, cubic, or higher order)
- Interleaved sequences (alternating between different rules)
- Conditional patterns that change based on position or value
- Recursive relationships involving multiple previous terms
- Pattern transformations that change at certain points
- Modular arithmetic or position-dependent rules

Analyze the sequence carefully, looking for:
1. Differences between consecutive terms
2. Ratios between consecutive terms  
3. Patterns in even/odd positions
4. Polynomial relationships (quadratic, cubic)
5. Changes in the pattern at certain positions

Provide the next term in the sequence. Your final answer must be in the format \\boxed{{next_term}} at the end.

For example: If the sequence is 1, 4, 9, 16, 25, ? then the next term is \\boxed{{36}} (perfect squares: n²).
"""
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for complex pattern completion"""
        ground_truth = data_point['next_term']
        
        # Use centralized parsing
        from ...utils.parsing import parse_sequence_result
        parsed_answer = parse_sequence_result(response)

        instruction_followed = parsed_answer is not None
        accuracy = 0

        if instruction_followed and parsed_answer is not None:
            try:
                # Use robust tolerance-based comparison for all numeric values
                accuracy = 1 if self._values_are_close(parsed_answer, ground_truth) else 0
            except Exception as e:
                logging.debug(f"Comparison error: {e}")
                accuracy = 0

        result = {
            "sequence_type": data_point['sequence_type'],
            "shown_sequence": data_point['shown_sequence'],
            "ground_truth": ground_truth,
            "predicted_answer": parsed_answer,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed,
            "sequence_description": data_point['description']
        }
        
        return result
    
    def run_evaluation(self, list_sizes):
        """Run evaluation for multiple sequence lengths"""
        all_metrics = []
        
        for seq_length in list_sizes:
            logging.info(f"\n{'='*50}")
            logging.info(f"Evaluating complex patterns with length {seq_length}")
            logging.info(f"{'='*50}")
            
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
    parser = argparse.ArgumentParser(description='Complex Pattern Task Evaluation')
    
    # Model Configuration
    parser.add_argument('--model_id', type=str, default=MODEL_ID, 
                       help=f'Model ID to use (default: {MODEL_ID})')
    parser.add_argument('--engine', type=str, default=ENGINE, choices=['vllm', 'transformers'],
                       help=f'Engine to use (default: {ENGINE})')
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
                       help=f'Range for initial values and coefficients as "min,max" (default: {RANGE[0]},{RANGE[1]})')
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
    USE_ALL_SEQUENCE_TYPES = args.use_all_sequence_types
    ENABLED_SEQUENCE_TYPES = args.enabled_sequence_types.split(',') if args.enabled_sequence_types else []
    
        
    try:
        # Create model handler for standalone execution
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
        
        # Setup
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"complex_pattern_results_{MODEL_ID.split('/')[-1]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        setup_logging(output_dir)
        logging.info(f"🚀 Starting Complex Pattern Task evaluation")
        logging.info(f"📋 Model: {MODEL_ID}")
        logging.info(f"🔧 Using centralized SequenceAnswerParser for consistent parsing")
        
        
        # Initialize task
        task = ComplexPatternTask(
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
            
        logging.info(f"✅ Complex Pattern evaluation complete!")
        logging.info(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"❌ Error running Complex Pattern task: {e}")
        import traceback
        traceback.print_exc()
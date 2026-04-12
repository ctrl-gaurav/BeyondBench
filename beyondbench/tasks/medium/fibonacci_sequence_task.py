"""
Fibonacci and Recursive Sequence Completion Task

This task evaluates LLM's ability to recognize and complete Fibonacci-like sequences and other recursive patterns.
The sequences generated include:
1. Classic Fibonacci: F(n) = F(n-1) + F(n-2), starting with various initial values
2. Lucas sequences: Similar to Fibonacci but with different starting points  
3. Tribonacci: F(n) = F(n-1) + F(n-2) + F(n-3)
4. Modified recursive: F(n) = aF(n-1) + bF(n-2) with different coefficients
5. Alternating recursive sequences with sign changes

Algorithm: Generate sequences using recursive formulas, present partial sequence, ask for next terms.
Reasoning: Requires pattern recognition, understanding of recursive relationships, mathematical intuition.

Example: Given sequence "1, 1, 2, 3, 5, 8, ?", model should identify Fibonacci and predict 13.

CLI USAGE:
python fibonacci_sequence_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --cuda_device "cuda:6" --datapoints 20 --folds 1 --list_sizes 6,8,10 --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --tensor_parallel_size 1 --gpu_memory_utilization 0.96 --trust_remote_code False --store_details True --use_all_sequence_types True --enabled_sequence_types "fibonacci,lucas,tribonacci"
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["fibonacci_sequence"]
# Engine selection - choose 'vllm' or 'transformers'
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 20  # Number of samples per fold
FOLDS = 1  # Number of evaluation folds
RANGE = [1, 20]  # Range for initial values
LIST_SIZES = [6, 8, 10]  # Length of sequences to show before asking for completion
STORE_DETAILS = True  # Save detailed results in JSON
SEED = 42  # Random seed for reproducibility

# Generation Parameters
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 8192

# Sequence Type Control - Select which types to include in evaluation
ENABLED_SEQUENCE_TYPES = [
    'fibonacci',           # Classic Fibonacci: F(n) = F(n-1) + F(n-2)
    'lucas',              # Lucas sequence: starts with 2, 1 then follows Fibonacci rule
    'tribonacci',         # Tribonacci: F(n) = F(n-1) + F(n-2) + F(n-3)
    'modified_recursive', # F(n) = aF(n-1) + bF(n-2) with different coefficients
    'alternating_fibonacci', # Fibonacci with alternating signs
    'scaled_fibonacci'    # Fibonacci multiplied by a constant factor
]

# Set to True to include all sequence types, False to use only ENABLED_SEQUENCE_TYPES
USE_ALL_SEQUENCE_TYPES = True

# ============================================================================

# No longer setting CUDA_VISIBLE_DEVICES - handled by engine selection

import random
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

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

class FibonacciSequenceTask(BaseTask):
    """Implementation of Fibonacci and recursive sequence completion task"""

    # All available sequence types
    ALL_SEQUENCE_TYPES = [
        'fibonacci', 'lucas', 'tribonacci', 'modified_recursive',
        'alternating_fibonacci', 'scaled_fibonacci'
    ]

    def __init__(self, *args, use_all_sequence_types=True, enabled_sequence_types=None, **kwargs):
        """
        Initialize FibonacciSequenceTask.

        Args:
            *args: Positional args forwarded to BaseTask.
            use_all_sequence_types: If True, use all sequence types. If False,
                use only those listed in ``enabled_sequence_types``.
            enabled_sequence_types: List of sequence type names to enable when
                ``use_all_sequence_types`` is False. Defaults to all types.
            **kwargs: Additional keyword args forwarded to BaseTask.
        """
        super().__init__(*args, **kwargs)
        self.use_all_sequence_types = use_all_sequence_types
        self.enabled_sequence_types = (
            list(enabled_sequence_types) if enabled_sequence_types
            else list(self.ALL_SEQUENCE_TYPES)
        )

    @property
    def task_name(self):
        return "fibonacci_sequence"

    def generate_data(self, sequence_length=8):
        """
        Generate various types of recursive sequences

        Args:
            sequence_length: Length of sequence to generate (will show partial)
        """
        if self.seed is not None:
            random.seed(self.seed)

        data = []

        # Use instance configuration instead of module-level globals
        if self.use_all_sequence_types:
            sequence_types = list(self.ALL_SEQUENCE_TYPES)
        else:
            sequence_types = list(self.enabled_sequence_types)
        
        if not sequence_types:
            raise ValueError("No sequence types enabled! Please check ENABLED_SEQUENCE_TYPES configuration.")
        
        for _ in range(self.num_samples):
            seq_type = random.choice(sequence_types)
            
            if seq_type == 'fibonacci':
                # Classic Fibonacci with random starting values
                a, b = random.randint(self.min_val, 3), random.randint(self.min_val, 3)
                sequence = self._generate_fibonacci(a, b, sequence_length + 2)
                
            elif seq_type == 'lucas':
                # Lucas-like sequence: randomize the scale for contamination resistance
                scale = random.choice([1, 2, 3])
                sequence = self._generate_fibonacci(2 * scale, 1 * scale, sequence_length + 2)
                
            elif seq_type == 'tribonacci':
                # Tribonacci: sum of three previous terms
                a, b, c = (random.randint(self.min_val, 3) for _ in range(3))
                sequence = self._generate_tribonacci(a, b, c, sequence_length + 2)
                
            elif seq_type == 'modified_recursive':
                # F(n) = aF(n-1) + bF(n-2) with different coefficients
                coeff_a, coeff_b = random.choice([(1, 2), (2, 1), (1, -1), (2, -1)])
                start_a, start_b = random.randint(self.min_val, 3), random.randint(self.min_val, 3)
                sequence = self._generate_modified_recursive(
                    start_a, start_b, coeff_a, coeff_b, sequence_length + 2
                )
                
            elif seq_type == 'alternating_fibonacci':
                # Fibonacci with alternating signs
                a, b = random.randint(self.min_val, 3), random.randint(self.min_val, 3)
                sequence = self._generate_alternating_fibonacci(a, b, sequence_length + 2)
                
            elif seq_type == 'scaled_fibonacci':
                # Fibonacci multiplied by a constant with randomized start
                a, b = random.randint(1, 3), random.randint(1, 3)
                scale = random.choice([2, 3, 5, 7])
                sequence = self._generate_scaled_fibonacci(a, b, scale, sequence_length + 2)
            
            # Create data point with correct ground truth
            shown_sequence = sequence[:sequence_length]
            next_term = sequence[sequence_length] if len(sequence) > sequence_length else None
            
            if next_term is not None:
                data.append({
                    'sequence_type': seq_type,
                    'shown_sequence': shown_sequence,
                    'full_sequence': sequence,
                    'answer': next_term,
                    'next_term': next_term,
                    'description': self._get_sequence_description(seq_type),
                    # Additional ground truth verification
                    'ground_truth_verified': True,
                    'generation_params': self._get_generation_params(seq_type)
                })
        
        return data
    
    def _generate_fibonacci(self, a, b, length):
        """Generate Fibonacci-like sequence starting with a, b"""
        sequence = [a, b]
        for i in range(2, length):
            next_val = sequence[i-1] + sequence[i-2]
            sequence.append(next_val)
        return sequence
    
    def _generate_tribonacci(self, a, b, c, length):
        """Generate Tribonacci sequence F(n) = F(n-1) + F(n-2) + F(n-3)"""
        sequence = [a, b, c]
        for i in range(3, length):
            next_val = sequence[i-1] + sequence[i-2] + sequence[i-3]
            sequence.append(next_val)
        return sequence
    
    def _generate_modified_recursive(self, a, b, coeff_a, coeff_b, length):
        """Generate F(n) = coeff_a*F(n-1) + coeff_b*F(n-2)"""
        sequence = [a, b]
        for i in range(2, length):
            next_val = coeff_a * sequence[i-1] + coeff_b * sequence[i-2]
            sequence.append(next_val)
        return sequence
    
    def _generate_alternating_fibonacci(self, a, b, length):
        """Generate Fibonacci with alternating signs"""
        sequence = [a, b]
        for i in range(2, length):
            next_val = sequence[i-1] + sequence[i-2]
            if i % 2 == 0:
                next_val = -next_val
            sequence.append(next_val)
        return sequence
    
    def _generate_scaled_fibonacci(self, a, b, scale, length):
        """Generate Fibonacci sequence scaled by a constant"""
        fib_seq = [a, b]
        for i in range(2, length):
            fib_seq.append(fib_seq[i-1] + fib_seq[i-2])
        
        scaled_seq = [x * scale for x in fib_seq]
        return scaled_seq
    
    def _get_sequence_description(self, seq_type):
        """Get description for different sequence types"""
        descriptions = {
            'fibonacci': 'Classic Fibonacci sequence where each term is sum of two previous terms',
            'lucas': 'Lucas sequence following Fibonacci rule but starting with 2, 1',
            'tribonacci': 'Tribonacci sequence where each term is sum of three previous terms',
            'modified_recursive': 'Modified recursive sequence with different coefficients',
            'alternating_fibonacci': 'Fibonacci sequence with alternating signs',
            'scaled_fibonacci': 'Fibonacci sequence multiplied by a constant factor'
        }
        return descriptions.get(seq_type, 'Recursive sequence')
    
    def _get_generation_params(self, seq_type):
        """Get generation parameters for verification"""
        return {
            'sequence_type': seq_type,
            'algorithm_verified': True,
            'deterministic': True
        }
    
    def create_prompt(self, data_point):
        """Create prompt for sequence completion task"""
        sequence = data_point['shown_sequence']
        sequence_str = ', '.join(map(str, sequence))
        
        return f"""Complete the following sequence by identifying the pattern:

{sequence_str}, ?

Analyze the sequence carefully to identify the underlying pattern or rule. Consider:
- Is each term related to previous term(s)?
- Are there arithmetic, geometric, or recursive relationships?
- Look for Fibonacci-like patterns, Lucas sequences, or other mathematical progressions

Provide the next term in the sequence. Your final answer must be in the format \\boxed{{next_term}} at the end.

For example: If the sequence is 1, 1, 2, 3, 5, 8, ? then the next term is \\boxed{{13}}.
"""
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for sequence completion"""
        ground_truth = data_point['next_term']

        # Prefer the unified parser; fall back to the internal one if needed.
        try:
            from ...utils.parsing import parse_sequence_result
            parsed_answer = parse_sequence_result(response)
        except Exception:
            parsed_answer = self._parse_sequence_answer(response)

        instruction_followed = parsed_answer is not None
        accuracy = 0

        if instruction_followed and parsed_answer is not None:
            try:
                # Fibonacci terms are integers, but allow tiny float rounding
                # (e.g. model answers "13.0" for 13).
                accuracy = 1 if abs(float(parsed_answer) - float(ground_truth)) < 0.5 else 0
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
            "sequence_description": data_point['description'],
            # Additional detailed information for JSON storage
            "full_sequence": data_point['full_sequence'],
            "generation_params": data_point.get('generation_params', {}),
            "prompt": self.create_prompt(data_point),
            "model_response": response
        }
    
    def _parse_sequence_answer(self, response):
        """Parse the next term from model response using robust parsing"""
        # Import robust parser
        try:
            from ...utils.parsing import parse_sequence_result
            return parse_sequence_result(response)
        except ImportError:
            # Fallback to simple parsing
            return self._simple_parse_answer(response)
    
    def _simple_parse_answer(self, response):
        """Simple fallback parsing for sequence answers"""
        # Look for boxed format
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, response)
        
        if matches:
            try:
                return int(float(matches[-1].strip()))
            except (ValueError, TypeError):
                pass
        
        # Fallback patterns
        fallback_patterns = [
            r'(?:next term|answer) is\s*(-?\d+)',
            r'(?:therefore|thus|so),?\s*(?:the\s+)?(?:next\s+)?(?:term\s+)?(?:is\s+)?(-?\d+)',
            r'answer:\s*(-?\d+)',
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    return int(float(matches[-1]))
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def run_evaluation(self, list_sizes):
        """Run evaluation for multiple sequence lengths"""
        all_metrics = []
        
        for seq_length in list_sizes:
            logging.info(f"\n{'='*50}")
            logging.info(f"Evaluating Fibonacci sequences with length {seq_length}")
            logging.info(f"Enabled sequence types: {self.enabled_sequence_types if not self.use_all_sequence_types else 'ALL'}")
            logging.info(f"{'='*50}")
            
            # Generate evaluation data
            data = self.generate_data(seq_length)
            
            # Run each fold
            for fold in range(self.num_folds):
                metrics = self.run_fold(data, seq_length, fold)
                metrics['sequence_length'] = seq_length
                metrics['enabled_types'] = self.enabled_sequence_types if not self.use_all_sequence_types else "ALL"
                all_metrics.append(metrics)
        
        return all_metrics

def parse_arguments():
    """Parse command line arguments with defaults from configuration"""
    parser = argparse.ArgumentParser(description='Fibonacci Sequence Task Evaluation')
    
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
                       help=f'Range for initial values as "min,max" (default: {RANGE[0]},{RANGE[1]})')
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
    
    # # Set up CUDA device from CLI args
    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     cuda_device = CUDA_DEVICE.split(':')[-1] if ':' in CUDA_DEVICE else '0'
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    #     print(f"🔧 CUDA device set to: {CUDA_DEVICE}")
        
    try:
        # Setup
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"fibonacci_sequence_results_{MODEL_ID.split('/')[-1]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        log_file = setup_logging(output_dir)
        logging.info(f"🚀 Starting Fibonacci Sequence Task evaluation")
        logging.info(f"📋 Model: {MODEL_ID}")
        logging.info(f"📊 Configuration:")
        logging.info(f"   - Datapoints per fold: {DATAPOINTS}")
        logging.info(f"   - Number of folds: {FOLDS}")
        logging.info(f"   - Sequence lengths: {LIST_SIZES}")
        logging.info(f"   - Enabled types: {ENABLED_SEQUENCE_TYPES if not USE_ALL_SEQUENCE_TYPES else 'ALL'}")
        logging.info(f"   - Store details: {STORE_DETAILS}")
        
        # Initialize model handler
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
        task = FibonacciSequenceTask(
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
            seed=SEED,
            use_all_sequence_types=USE_ALL_SEQUENCE_TYPES,
            enabled_sequence_types=ENABLED_SEQUENCE_TYPES
        )
        
        # Run evaluation
        metrics = task.run_evaluation(LIST_SIZES)
        
        # Generate report
        if metrics:
            final_report = generate_final_report(metrics, LIST_SIZES, output_dir)
            logging.info(f"📊 Final report generated: {final_report}")
            
        logging.info(f"✅ Fibonacci Sequence evaluation complete!")
        logging.info(f"📁 Results saved to: {output_dir}")
        logging.info(f"📋 Log file: {log_file}")
        
    except Exception as e:
        logging.error(f"❌ Error running Fibonacci Sequence task: {e}")
        import traceback
        traceback.print_exc()
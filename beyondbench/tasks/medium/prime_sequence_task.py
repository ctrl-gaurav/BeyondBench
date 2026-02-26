"""
Prime and Number Theory Sequence Completion Task

This task evaluates LLM's ability to recognize and complete sequences involving prime numbers,
number theoretic functions, and advanced mathematical patterns from number theory.

The sequences generated include:
1. Prime number sequences: 2, 3, 5, 7, 11, 13, ...
2. Twin primes: pairs of primes differing by 2
3. Prime gaps: differences between consecutive primes
4. Mersenne numbers: 2^p - 1 where p is prime
5. Perfect numbers and their patterns
6. Euler's totient function φ(n) sequences
7. Sum of divisors σ(n) sequences
8. Catalan numbers with number theory properties
9. Fibonacci modulo prime sequences
10. Prime counting function approximations

Algorithm: Generate sequences using number theory algorithms, present partial sequence, ask for next terms.
Reasoning: Requires deep understanding of prime properties, number theory, mathematical functions.

Example: Given sequence "2, 3, 5, 7, 11, ?", model should identify primes and predict 13.

CLI USAGE:
python prime_sequence_task.py --model_id "Qwen/Qwen2.5-3B-Instruct" --engine "vllm" --datapoints 20 --folds 1 --list_sizes 5,6,7 --temperature 0.1 --top_p 0.9 --max_tokens 8192 --seed 42 --tensor_parallel_size 1 --gpu_memory_utilization 0.96 --trust_remote_code False --store_details True --use_all_sequence_types True --enabled_sequence_types "primes,twin_primes,prime_gaps"
"""

# ============================================================================
# CONFIGURATION PARAMETERS (Customize as needed)
# ============================================================================

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TASKS = ["prime_sequence"]
ENGINE = "vllm"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.96
TRUST_REMOTE_CODE = False

# Evaluation Configuration
DATAPOINTS = 5  # Number of samples per fold
FOLDS = 1  # Number of evaluation folds
RANGE = [2, 100]  # Range for prime generation
LIST_SIZES = [5, 6, 7]  # Length of sequences to show before asking for completion
STORE_DETAILS = True  # Save detailed results in JSON
SEED = 42  # Random seed for reproducibility

# Generation Parameters
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 8192

# Sequence Type Control - Select which types to include in evaluation
ENABLED_SEQUENCE_TYPES = [
    'primes',                 # Prime number sequences
    'twin_primes',           # Twin prime pairs (p, p+2)
    'prime_gaps',            # Differences between consecutive primes
    'mersenne_numbers',      # Mersenne numbers: 2^p - 1 for prime p
    'euler_totient',         # Euler's totient function φ(n)
    'sum_of_divisors',       # Sum of divisors σ(n) sequences
    'catalan_numbers',       # Catalan numbers with number theory properties
    'sophie_germain_primes'  # Sophie Germain primes: p where 2p+1 is also prime
]

# Set to True to include all sequence types, False to use only ENABLED_SEQUENCE_TYPES
USE_ALL_SEQUENCE_TYPES = True

# ============================================================================

import os

import random
import math
import logging
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

class PrimeSequenceTask(BaseTask):
    """Implementation of prime and number theory sequence completion task"""
    
    @property
    def task_name(self):
        return "prime_sequence"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compute primes for efficiency - start with reasonable initial set
        self.primes = self._sieve_of_eratosthenes(1000)  # Generate primes up to 1000
        # Initialize centralized parser
        # Using unified parsing system
    
    def _sieve_of_eratosthenes(self, limit):
        """Generate all prime numbers up to limit using Sieve of Eratosthenes"""
        if limit < 2:
            return []
            
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def _ensure_primes_count(self, n):
        """Ensure we have at least n primes - dynamic generation"""
        if len(self.primes) >= n:
            return
        
        # Estimate upper bound for nth prime using approximation
        if n < 6:
            limit = 15
        else:
            # Using prime number theorem approximation: n-th prime ≈ n * (ln(n) + ln(ln(n)))
            limit = int(n * (math.log(n) + math.log(math.log(n)) + 2))
        
        self.primes = self._sieve_of_eratosthenes(limit)
        
        # If still not enough, keep doubling until we have enough primes
        while len(self.primes) < n:
            limit *= 2
            if limit > 1000000:  # Safety check to prevent excessive memory usage
                logging.warning(f"Prime generation limit reached. Only have {len(self.primes)} primes, needed {n}")
                break
            self.primes = self._sieve_of_eratosthenes(limit)
    
    def _is_prime(self, n):
        """Check if a number is prime"""
        if n < 2:
            return False
        if n in self.primes:
            return True
        # For larger numbers not in our precomputed list
        if n > max(self.primes) if self.primes else n > 1000:
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        return False
    
    def _euler_totient(self, n):
        """Compute Euler's totient function φ(n)"""
        if n <= 0:
            return 0
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    
    def _sum_of_divisors(self, n):
        """Compute sum of divisors σ(n)"""
        if n <= 0:
            return 0
        divisors_sum = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divisors_sum += i
                if i != n // i:
                    divisors_sum += n // i
        return divisors_sum
    
    def _catalan_number(self, n):
        """Compute nth Catalan number"""
        if n < 0:
            return 0
        if n <= 1:
            return 1
        catalan = [0] * (n + 1)
        catalan[0], catalan[1] = 1, 1
        
        for i in range(2, n + 1):
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i - 1 - j]
        return catalan[n]
    
    def generate_data(self, sequence_length=6):
        """
        Generate various types of prime and number theory sequences
        
        Args:
            sequence_length: Length of sequence to generate (will show partial)
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        data = []
        
        # Use either enabled types or all types based on configuration
        if USE_ALL_SEQUENCE_TYPES:
            sequence_types = [
                'primes', 'twin_primes', 'prime_gaps', 'mersenne_numbers',
                'euler_totient', 'sum_of_divisors', 'catalan_numbers',
                'fibonacci_mod_prime', 'prime_powers', 'semiprime_factors', 'sophie_germain_primes'
            ]
        else:
            sequence_types = ENABLED_SEQUENCE_TYPES
        
        if not sequence_types:
            raise ValueError("No sequence types enabled! Please check ENABLED_SEQUENCE_TYPES configuration.")
        
        # Generate with retry mechanism to ensure we get enough valid samples
        attempts = 0
        max_attempts = self.num_samples * 10  # Allow up to 10x attempts
        
        while len(data) < self.num_samples and attempts < max_attempts:
            seq_type = random.choice(sequence_types)
            attempts += 1
            
            # Use more specific exception handling instead of catching everything
            try:
                sequence = None
                next_terms = None
                
                if seq_type == 'primes':
                    # Prime number sequence
                    sequence, next_terms = self._generate_primes(sequence_length + 2)
                    
                elif seq_type == 'twin_primes':
                    # Twin prime pairs (p, p+2)
                    sequence, next_terms = self._generate_twin_primes(sequence_length + 2)
                    
                elif seq_type == 'prime_gaps':
                    # Differences between consecutive primes
                    sequence, next_terms = self._generate_prime_gaps(sequence_length + 2)
                    
                elif seq_type == 'mersenne_numbers':
                    # Mersenne numbers: 2^p - 1 for prime p
                    sequence, next_terms = self._generate_mersenne_numbers(min(6, sequence_length + 2))
                    
                elif seq_type == 'euler_totient':
                    # Euler's totient function φ(n)
                    sequence, next_terms = self._generate_euler_totient_sequence(sequence_length + 2)
                    
                elif seq_type == 'sum_of_divisors':
                    # Sum of divisors σ(n)
                    sequence, next_terms = self._generate_sum_of_divisors_sequence(sequence_length + 2)
                    
                elif seq_type == 'catalan_numbers':
                    # Catalan numbers
                    sequence, next_terms = self._generate_catalan_sequence(min(8, sequence_length + 2))
                    
                elif seq_type == 'fibonacci_mod_prime':
                    # Fibonacci sequence modulo a prime
                    prime = random.choice([3, 5, 7, 11])
                    sequence, next_terms = self._generate_fibonacci_mod_prime(prime, sequence_length + 2)
                    
                elif seq_type == 'prime_powers':
                    # Powers of primes: 2^n, 3^n, 5^n, etc.
                    prime = random.choice([2, 3, 5])
                    sequence, next_terms = self._generate_prime_powers(prime, sequence_length + 2)
                    
                elif seq_type == 'semiprime_factors':
                    # Products of exactly two primes
                    sequence, next_terms = self._generate_semiprimes(sequence_length + 2)
                    
                elif seq_type == 'sophie_germain_primes':
                    # Sophie Germain primes: p where 2p+1 is also prime
                    sequence, next_terms = self._generate_sophie_germain_primes(sequence_length + 2)
                
                # Validate sequence was generated successfully
                if sequence is None or next_terms is None or len(sequence) == 0 or len(next_terms) == 0:
                    logging.debug(f"Failed to generate valid sequence for type {seq_type}")
                    continue
                
                # Skip if sequence is too large or contains errors
                if any(abs(x) > 10000 for x in sequence + next_terms):
                    logging.debug(f"Sequence {seq_type} contains values that are too large")
                    continue
                
                # Ensure we have enough elements for the requested sequence length
                if len(sequence) < sequence_length:
                    logging.debug(f"Generated sequence too short for type {seq_type}: {len(sequence)} < {sequence_length}")
                    continue
                    
                # Create data point - reconstruct full sequence from returned parts  
                full_sequence = sequence + next_terms
                shown_sequence = sequence[:sequence_length]  # Only show requested length
                next_term = next_terms[0] if next_terms else None
                
                if next_term is not None:
                    data.append({
                        'sequence_type': seq_type,
                        'shown_sequence': shown_sequence,
                        'full_sequence': full_sequence,
                        'answer': next_terms,
                        'next_term': next_term,
                        'description': self._get_sequence_description(seq_type),
                        'ground_truth_verified': True,
                        'generation_params': {'sequence_type': seq_type}
                    })
                
            except (ValueError, ArithmeticError, IndexError) as e:
                logging.debug(f"Specific error generating {seq_type}: {e}")
                continue
            except Exception as e:
                logging.warning(f"Unexpected error generating {seq_type}: {e}")
                continue
        
        # Log if we couldn't generate enough samples
        if len(data) < self.num_samples:
            logging.warning(f"Could only generate {len(data)} out of {self.num_samples} requested samples after {attempts} attempts")
        
        return data
    
    def _generate_primes(self, length):
        """Generate sequence of prime numbers"""
        self._ensure_primes_count(length)
        if len(self.primes) < length:
            raise ValueError(f"Could not generate enough primes: need {length}, have {len(self.primes)}")
        
        sequence = self.primes[:length]
        return sequence[:-2], sequence[-2:]
    
    def _generate_twin_primes(self, length):
        """Generate twin prime sequence (primes p where p+2 is also prime)"""
        # Ensure we have enough primes to find twin primes
        self._ensure_primes_count(length * 3)  # Conservative estimate
        
        twin_primes = []
        for p in self.primes:
            if p + 2 in self.primes:
                twin_primes.append(p)
                if len(twin_primes) >= length:
                    break
        
        if len(twin_primes) < length:
            raise ValueError(f"Could not find enough twin primes: need {length}, found {len(twin_primes)}")
            
        return twin_primes[:-2], twin_primes[-2:]
    
    def _generate_prime_gaps(self, length):
        """Generate sequence of gaps between consecutive primes"""
        self._ensure_primes_count(length + 1)  # Need one more prime than gaps
        
        if len(self.primes) < length + 1:
            raise ValueError(f"Not enough primes to generate {length} gaps")
        
        gaps = []
        for i in range(min(len(self.primes) - 1, length)):
            gaps.append(self.primes[i + 1] - self.primes[i])
            
        if len(gaps) < length:
            raise ValueError(f"Could not generate enough prime gaps: need {length}, have {len(gaps)}")
            
        return gaps[:-2], gaps[-2:]
    
    def _generate_mersenne_numbers(self, length):
        """Generate Mersenne numbers: 2^p - 1 for prime p"""
        self._ensure_primes_count(length)
        
        mersenne = []
        for p in self.primes:
            if p <= 10:  # Keep numbers reasonable
                mersenne_num = 2**p - 1
                mersenne.append(mersenne_num)
                if len(mersenne) >= length:
                    break
        
        if len(mersenne) < length:
            raise ValueError(f"Could not generate enough Mersenne numbers: need {length}, have {len(mersenne)}")
            
        return mersenne[:-2], mersenne[-2:]
    
    def _generate_euler_totient_sequence(self, length):
        """Generate sequence of Euler's totient function φ(n)"""
        sequence = []
        n = 1
        while len(sequence) < length:
            phi_n = self._euler_totient(n)
            sequence.append(phi_n)
            n += 1
        return sequence[:-2], sequence[-2:]
    
    def _generate_sum_of_divisors_sequence(self, length):
        """Generate sequence of sum of divisors σ(n)"""
        sequence = []
        n = 1
        while len(sequence) < length:
            sigma_n = self._sum_of_divisors(n)
            sequence.append(sigma_n)
            n += 1
        return sequence[:-2], sequence[-2:]
    
    def _generate_catalan_sequence(self, length):
        """Generate Catalan number sequence"""
        sequence = []
        for n in range(length):
            cat_n = self._catalan_number(n)
            sequence.append(cat_n)
        return sequence[:-2], sequence[-2:]
    
    def _generate_fibonacci_mod_prime(self, prime, length):
        """Generate Fibonacci sequence modulo prime"""
        if length < 2:
            raise ValueError("Fibonacci sequence needs at least 2 terms")
            
        sequence = [1, 1]
        for i in range(2, length):
            next_val = (sequence[i-1] + sequence[i-2]) % prime
            sequence.append(next_val)
        return sequence[:-2], sequence[-2:]
    
    def _generate_prime_powers(self, prime, length):
        """Generate powers of a prime: p^1, p^2, p^3, ..."""
        sequence = []
        for i in range(1, length + 1):
            power = prime ** i
            if power > 1000:  # Keep numbers reasonable
                break
            sequence.append(power)
            
        if len(sequence) < length:
            raise ValueError(f"Prime powers got too large: need {length}, have {len(sequence)}")
            
        return sequence[:-2], sequence[-2:]
    
    def _generate_semiprimes(self, length):
        """Generate semiprimes (products of exactly two primes)"""
        semiprimes = []
        n = 4  # First semiprime
        while len(semiprimes) < length and n <= 200:  # Prevent infinite loop
            # Check if n is a semiprime
            factors = []
            temp = n
            for p in self.primes:
                if p * p > temp:
                    break
                while temp % p == 0:
                    factors.append(p)
                    temp //= p
            if temp > 1:
                factors.append(temp)
            
            if len(factors) == 2:
                semiprimes.append(n)
            n += 1
        
        if len(semiprimes) < length:
            raise ValueError(f"Could not find enough semiprimes: need {length}, found {len(semiprimes)}")
            
        return semiprimes[:-2], semiprimes[-2:]
    
    def _generate_sophie_germain_primes(self, length):
        """Generate Sophie Germain primes: p where 2p+1 is also prime"""
        # Ensure we have enough primes
        self._ensure_primes_count(length * 3)  # Conservative estimate
        
        sophie_primes = []
        for p in self.primes:
            if 2 * p + 1 in self.primes or self._is_prime(2 * p + 1):
                sophie_primes.append(p)
                if len(sophie_primes) >= length:
                    break
        
        if len(sophie_primes) < length:
            raise ValueError(f"Could not find enough Sophie Germain primes: need {length}, found {len(sophie_primes)}")
            
        return sophie_primes[:-2], sophie_primes[-2:]
    
    def _get_sequence_description(self, seq_type):
        """Get description for different sequence types"""
        descriptions = {
            'primes': 'Prime numbers: integers greater than 1 with no divisors other than 1 and themselves',
            'twin_primes': 'Twin primes: pairs of primes that differ by 2',
            'prime_gaps': 'Prime gaps: differences between consecutive prime numbers',
            'mersenne_numbers': 'Mersenne numbers: numbers of form 2^p - 1 where p is prime',
            'euler_totient': 'Euler totient function φ(n): count of integers ≤ n that are coprime to n',
            'sum_of_divisors': 'Sum of divisors σ(n): sum of all positive divisors of n',
            'catalan_numbers': 'Catalan numbers: sequence appearing in combinatorics',
            'fibonacci_mod_prime': 'Fibonacci sequence reduced modulo a prime number',
            'prime_powers': 'Powers of a prime number: p^1, p^2, p^3, ...',
            'semiprime_factors': 'Semiprimes: numbers that are products of exactly two primes',
            'sophie_germain_primes': 'Sophie Germain primes: p where 2p+1 is also prime'
        }
        return descriptions.get(seq_type, 'Number theory sequence')
    
    def _values_are_close(self, val1, val2, rel_tol=1e-4, abs_tol=1.0):
        """Check if two values are close enough"""
        return values_are_close(val1, val2, rel_tol, abs_tol)
    
    def create_prompt(self, data_point):
        """Create prompt for prime sequence completion task"""
        sequence = data_point['shown_sequence']
        sequence_str = ', '.join(map(str, sequence))
        
        return f"""Complete the following number theory sequence by identifying the pattern:

{sequence_str}, ?

This sequence involves concepts from number theory. Analyze carefully and consider:
- Are these prime numbers or related to primes?
- Look for patterns involving divisibility, factors, or number theoretic functions
- Consider sequences like: prime numbers, twin primes, Fibonacci numbers, perfect numbers
- Check if numbers follow arithmetic properties, modular arithmetic, or special functions
- Think about famous sequences in mathematics and number theory

Provide the next term in the sequence. Your final answer must be in the format \\boxed{{next_term}} at the end.

For example: If the sequence is 2, 3, 5, 7, 11, ? then the next term is \\boxed{{13}} (consecutive prime numbers).
"""
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for prime sequence completion"""
        ground_truth = data_point['next_term']
        
        # Use centralized parser instead of local parsing method
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
            "model_response": response,
            "parsing_debug": []
        }
    
    def run_evaluation(self, list_sizes):
        """Run evaluation for multiple sequence lengths"""
        all_metrics = []
        
        for seq_length in list_sizes:
            logging.info(f"\n{'='*50}")
            logging.info(f"Evaluating prime sequences with length {seq_length}")
            logging.info(f"Enabled sequence types: {ENABLED_SEQUENCE_TYPES if not USE_ALL_SEQUENCE_TYPES else 'ALL'}")
            logging.info(f"{'='*50}")
            
            # Generate evaluation data
            data = self.generate_data(seq_length)
            
            # Run each fold
            for fold in range(self.num_folds):
                metrics = self.run_fold(data, seq_length, fold)
                metrics['sequence_length'] = seq_length
                metrics['enabled_types'] = ENABLED_SEQUENCE_TYPES if not USE_ALL_SEQUENCE_TYPES else "ALL"
                all_metrics.append(metrics)
        
        return all_metrics

def parse_arguments():
    """Parse command line arguments with defaults from configuration"""
    parser = argparse.ArgumentParser(description='Prime Sequence Task Evaluation')
    
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
                       help=f'Range for prime generation as "min,max" (default: {RANGE[0]},{RANGE[1]})')
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
            output_dir = f"prime_sequence_results_{MODEL_ID.split('/')[-1]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        setup_logging(output_dir)
        logging.info(f"🚀 Starting Prime Sequence Task evaluation")
        logging.info(f"📋 Model: {MODEL_ID}")
        
        
        # Initialize task
        task = PrimeSequenceTask(
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
            
        logging.info(f"✅ Prime Sequence evaluation complete!")
        logging.info(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"❌ Error running Prime Sequence task: {e}")
        import traceback
        traceback.print_exc()
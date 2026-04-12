# Updated base_task.py with proper token counting

from abc import ABC, abstractmethod
import logging
import os
import json
import numpy as np
import time
import traceback
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple


# ------------------------------------------------------------------ #
# Lazy GPU utilization sampler for progress bars.
# We use a module-level cached GPUUtilizationSampler so repeated
# set_postfix() calls during a fold don't pay the nvidia-smi cost more
# than once per ~2 seconds. Lives here (rather than result_aggregator)
# so single-GPU evaluation paths can share it.
# ------------------------------------------------------------------ #
_GPU_SAMPLER = None
_GPU_LAST_SAMPLE_TIME: float = 0.0
_GPU_LAST_SAMPLE: Dict[int, Dict[str, int]] = {}
_GPU_SAMPLE_INTERVAL = 2.0  # seconds — throttle nvidia-smi calls


def _progress_gpu_postfix() -> str:
    """Return a short `GPU0 73%/18.1GB` string for tqdm.set_postfix.

    Throttled: calls nvidia-smi at most once every ``_GPU_SAMPLE_INTERVAL``
    seconds. Returns an empty string on CPU boxes or when nvidia-smi is
    unavailable so callers can unconditionally concatenate it.
    """
    global _GPU_SAMPLER, _GPU_LAST_SAMPLE_TIME, _GPU_LAST_SAMPLE

    if os.environ.get("BEYONDBENCH_DISABLE_GPU_POSTFIX") == "1":
        return ""

    now = time.time()
    if now - _GPU_LAST_SAMPLE_TIME < _GPU_SAMPLE_INTERVAL and _GPU_LAST_SAMPLE:
        snap = _GPU_LAST_SAMPLE
    else:
        if _GPU_SAMPLER is None:
            try:
                from .result_aggregator import GPUUtilizationSampler
                _GPU_SAMPLER = GPUUtilizationSampler()
            except Exception:
                _GPU_SAMPLER = False  # type: ignore[assignment]
        if _GPU_SAMPLER is False or _GPU_SAMPLER is None:
            return ""
        if not getattr(_GPU_SAMPLER, "available", False):
            return ""
        try:
            snap = _GPU_SAMPLER.sample_once()
        except Exception:
            return ""
        _GPU_LAST_SAMPLE = snap
        _GPU_LAST_SAMPLE_TIME = now

    if not snap:
        return ""
    # Only show the active/visible GPUs (CUDA_VISIBLE_DEVICES) to keep the
    # postfix short during multi-GPU runs.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed: Optional[set] = None
    if visible:
        try:
            allowed = {int(x) for x in visible.split(",") if x.strip().isdigit()}
        except Exception:
            allowed = None

    parts = []
    for gid in sorted(snap):
        if allowed is not None and gid not in allowed:
            continue
        info = snap[gid]
        mem_used_gib = info.get("mem_used_mib", 0) / 1024.0
        parts.append(f"G{gid}:{info.get('util', 0)}%/{mem_used_gib:.1f}GB")
    return " ".join(parts)

# Add these imports for proper token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

class BaseTask(ABC):
    """Abstract base class for all evaluation tasks with enhanced error handling and retry mechanisms"""
    
    def __init__(self, model_handler, output_dir, min_val, max_val, num_folds,
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None, max_retries=3, **kwargs):
        """
        Initialize base task with common parameters

        Args:
            model_handler: Model handler for inference (supports both vLLM and Transformers)
            output_dir: Directory to save results
            min_val: Minimum value for number generation
            max_val: Maximum value for number generation
            num_folds: Number of evaluation folds
            num_samples: Number of samples to generate per test case
            store_details: Whether to store detailed results
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            max_retries: Maximum retries for failed operations
            **kwargs: Additional parameters for flexibility
        """
        self.model_handler = model_handler
        self.output_dir = output_dir
        self.min_val = min_val
        self.max_val = max_val
        self.num_folds = num_folds
        self.num_samples = num_samples
        self.store_details = store_details
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.prompt_style = kwargs.get('prompt_style', None)
        self.contamination_resistance = kwargs.get('contamination_resistance', 'off')

        # Retry configuration
        self.max_retries = max_retries  # Maximum number of retries per sample
        self.retry_delay = 1.0  # Initial delay between retries (seconds)
        self.max_consecutive_failures = 10  # Maximum consecutive failures before stopping
        
        # Initialize token counter
        self._token_counting_warned = False  # Only warn once about fallback
        self._init_token_counter()
        
        # Create task-specific directory
        self.task_dir = os.path.join(output_dir, self.task_name)
        os.makedirs(self.task_dir, exist_ok=True)
        
        # Log model information
        model_info = self.model_handler.get_model_info()
        logging.info(f"🤖 Model Configuration for {self.task_name}:")
        for key, value in model_info.items():
            logging.info(f"   - {key}: {value}")
        
        # Log token counting method
        logging.info(f"🔢 Token counting method: {self.token_counting_method}")
    
    def _init_token_counter(self):
        """Initialize the appropriate token counter based on available libraries and model"""
        self.token_counter = None
        self.token_counting_method = "word_estimate"
        
        try:
            # Get model info to determine the best tokenizer
            model_info = self.model_handler.get_model_info()
            model_name = model_info.get('model_name', '').lower()
            
            # Try to initialize tiktoken for OpenAI models or as a general fallback
            if TIKTOKEN_AVAILABLE:
                try:
                    # Map common model names to tiktoken encodings
                    if any(name in model_name for name in ['gpt-4', 'gpt-3.5', 'text-davinci', 'text-curie']):
                        if 'gpt-4' in model_name:
                            self.token_counter = tiktoken.encoding_for_model("gpt-4")
                        elif 'gpt-3.5' in model_name:
                            self.token_counter = tiktoken.encoding_for_model("gpt-3.5-turbo")
                        else:
                            self.token_counter = tiktoken.encoding_for_model("text-davinci-003")
                        self.token_counting_method = f"tiktoken_{model_name}"
                    else:
                        # Use cl100k_base as a general-purpose encoder for most modern models
                        self.token_counter = tiktoken.get_encoding("cl100k_base")
                        self.token_counting_method = "tiktoken_cl100k_base"
                    
                    logging.info(f"✅ Initialized tiktoken encoder: {self.token_counting_method}")
                    return
                    
                except Exception as e:
                    logging.warning(f"⚠️  Failed to initialize tiktoken: {e}")
            
            # Try to use the model's own tokenizer if available
            if hasattr(self.model_handler, 'tokenizer') and self.model_handler.tokenizer is not None:
                try:
                    self.token_counter = self.model_handler.tokenizer
                    self.token_counting_method = "model_tokenizer"
                    logging.info("✅ Using model's own tokenizer for token counting")
                    return
                except Exception as e:
                    logging.warning(f"⚠️  Failed to use model tokenizer: {e}")
            
            # Try to load tokenizer from transformers library
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Try to load tokenizer based on model name
                    if model_name:
                        # Common model name mappings
                        tokenizer_name = model_name
                        if 'llama' in model_name:
                            tokenizer_name = "meta-llama/Llama-2-7b-hf"  # Default Llama tokenizer
                        elif 'mistral' in model_name:
                            tokenizer_name = "mistralai/Mistral-7B-v0.1"
                        elif 'falcon' in model_name:
                            tokenizer_name = "tiiuae/falcon-7b"
                        
                        self.token_counter = AutoTokenizer.from_pretrained(tokenizer_name)
                        self.token_counting_method = f"transformers_{tokenizer_name}"
                        logging.info(f"✅ Loaded transformers tokenizer: {tokenizer_name}")
                        return
                        
                except Exception as e:
                    logging.warning(f"⚠️  Failed to load transformers tokenizer: {e}")
            
            # Fallback to word-based estimation
            logging.warning("⚠️  Using word-based token estimation. Install tiktoken for accurate counting.")
            
        except Exception as e:
            logging.error(f"❌ Error initializing token counter: {e}")
            logging.warning("⚠️  Falling back to word-based estimation")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the best available method
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            if self.token_counter is not None:
                if self.token_counting_method.startswith("tiktoken"):
                    # Using tiktoken
                    return len(self.token_counter.encode(text))
                
                elif self.token_counting_method == "model_tokenizer":
                    # Using model's tokenizer
                    if hasattr(self.token_counter, 'encode'):
                        tokens = self.token_counter.encode(text)
                        return len(tokens) if isinstance(tokens, list) else tokens.size(0)
                    elif hasattr(self.token_counter, 'tokenize'):
                        return len(self.token_counter.tokenize(text))
                
                elif self.token_counting_method.startswith("transformers"):
                    # Using transformers tokenizer
                    tokens = self.token_counter.encode(text, add_special_tokens=False)
                    return len(tokens)
            
            # Fallback to improved word-based estimation
            if not self._token_counting_warned:
                logging.warning("⚠️  Token counter unavailable, using word-based estimation for this evaluation")
                self._token_counting_warned = True
            return self._estimate_tokens_from_words(text)

        except Exception as e:
            if not self._token_counting_warned:
                logging.warning(f"⚠️  Token counting failed: {e}, using word estimation for this evaluation")
                self._token_counting_warned = True
            return self._estimate_tokens_from_words(text)
    
    def _estimate_tokens_from_words(self, text: str) -> int:
        """
        Improved word-based token estimation
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        
        # Split by whitespace and punctuation for better estimation
        import re
        
        # Count words and punctuation separately
        words = len(re.findall(r'\b\w+\b', text))
        punctuation = len(re.findall(r'[^\w\s]', text))
        
        # Improved estimation based on empirical observations:
        # - Most words are 1-2 tokens
        # - Punctuation is usually 1 token
        # - Account for subword tokenization
        estimated_tokens = int(words * 1.3 + punctuation * 0.8)
        
        return max(1, estimated_tokens)  # Ensure at least 1 token for non-empty text
    
    @property
    @abstractmethod
    def task_name(self):
        """Return task name, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def generate_data(self, **kwargs):
        """Generate evaluation data, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def create_prompt(self, data_point):
        """Create prompt for the task, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def evaluate_response(self, response, data_point):
        """Evaluate model response, to be implemented by subclasses"""
        pass

    def get_prompt(self, data_point) -> str:
        """Return the prompt for *data_point*, respecting ``self.prompt_style``.

        * When ``prompt_style`` is None (default) → delegates to the task's own
          ``create_prompt`` (fully backwards-compatible).
        * When ``prompt_style`` is set → looks up the template from
          ``PromptLibrary`` and renders it with ``data=str(data_point)``.
          Falls back to ``create_prompt`` if the task or style is not registered.
        """
        if self.prompt_style is None:
            prompt = self.create_prompt(data_point)
        else:
            try:
                from ..prompts import get_prompt_library
                lib = get_prompt_library()
                tmpl = lib.get(self.task_name, self.prompt_style)
                prompt = tmpl.render(data=str(data_point))
            except Exception:
                # Graceful fallback: use original create_prompt so evaluation never breaks
                prompt = self.create_prompt(data_point)

        # Apply noise injection if contamination resistance is enabled
        if self.contamination_resistance and self.contamination_resistance != 'off':
            try:
                from .noise_injector import NoiseInjector
                injector = NoiseInjector(level=self.contamination_resistance, seed=self.seed)
                prompt = injector.inject(prompt, data_point)
            except Exception as _ni_err:
                logging.warning(f"Noise injection failed ({_ni_err}); using original prompt")

        return prompt

    def save_detailed_results(self, results, test_case_id, fold):
        """Save detailed results for each test case and fold"""
        if not self.store_details:
            return
            
        case_dir = os.path.join(self.task_dir, f"test_case_{test_case_id}")
        os.makedirs(case_dir, exist_ok=True)
        
        filename = os.path.join(case_dir, f"detailed_results_fold_{fold}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)  # default=str handles non-serializable objects
            logging.info(f"💾 Saved detailed results for test case {test_case_id} fold {fold} to {filename}")
        except Exception as e:
            logging.error(f"❌ Failed to save detailed results: {e}")
    
    def process_fold_metrics(self, fold_results):
        """Calculate metrics for a single fold"""
        if not fold_results:
            logging.warning("⚠️  No results to process for fold metrics")
            return {
                'total': 0,
                'correct': 0,
                'instruction_followed': 0,
                'accuracy': 0.0,
                'instruction_followed_pct': 0.0,
                'avg_response_length': 0.0,
                'avg_word_count': 0.0,
                'avg_output_tokens': 0.0,
                'generation_failures': 0,
                'parsing_failures': 0
            }
        
        # Count different types of results
        successful_results = [r for r in fold_results if r.get('generation_success', True)]
        generation_failures = len(fold_results) - len(successful_results)
        parsing_failures = sum(1 for r in successful_results if not r.get('parsing_success', True))
        
        # Calculate metrics only on successful results
        if successful_results:
            metrics = {
                'total': len(fold_results),
                'successful_generations': len(successful_results),
                'correct': sum(item.get('accuracy', 0) for item in successful_results),
                'instruction_followed': sum(item.get('instruction_followed', 0) for item in successful_results),
                'response_lengths': [item.get('string_len', 0) for item in successful_results],
                'word_counts': [item.get('words', 0) for item in successful_results],
                'output_tokens': [item.get('tokens', 0) for item in successful_results],
                'generation_failures': generation_failures,
                'parsing_failures': parsing_failures
            }
            
            # Calculate percentages and averages
            metrics['accuracy'] = round(metrics['correct'] / len(successful_results), 4) if successful_results else 0.0
            metrics['instruction_followed_pct'] = round(metrics['instruction_followed'] / len(successful_results), 4) if successful_results else 0.0
            metrics['avg_response_length'] = round(np.mean(metrics['response_lengths']), 2) if metrics['response_lengths'] else 0.0
            metrics['avg_word_count'] = round(np.mean(metrics['word_counts']), 2) if metrics['word_counts'] else 0.0
            metrics['avg_output_tokens'] = round(np.mean(metrics['output_tokens']), 2) if metrics['output_tokens'] else 0.0
            
            # Calculate success rates
            metrics['generation_success_rate'] = round(len(successful_results) / len(fold_results), 4)
            metrics['parsing_success_rate'] = round((len(successful_results) - parsing_failures) / len(successful_results), 4) if successful_results else 0.0
            
        else:
            # All generations failed
            metrics = {
                'total': len(fold_results),
                'successful_generations': 0,
                'correct': 0,
                'instruction_followed': 0,
                'accuracy': 0.0,
                'instruction_followed_pct': 0.0,
                'avg_response_length': 0.0,
                'avg_word_count': 0.0,
                'avg_output_tokens': 0.0,
                'generation_failures': generation_failures,
                'parsing_failures': 0,
                'generation_success_rate': 0.0,
                'parsing_success_rate': 0.0
            }
        
        return metrics
    
    def generate_response_with_retry(self, prompt: str, data_point: Dict) -> Tuple[str, int, bool, Dict]:
        """
        Generate response with retry mechanism for failed generations
        
        Args:
            prompt: The input prompt
            data_point: The data point being processed
            
        Returns:
            Tuple of (response, tokens, success, metadata)
        """
        last_error = None
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Determine backend and generate accordingly
                if self.model_handler.backend == 'vllm':
                    response, tokens = self._generate_vllm_response(prompt)
                elif self.model_handler.backend == 'transformers':
                    response, tokens = self._generate_transformers_response(prompt)
                elif self.model_handler.backend in ['openai', 'gemini', 'anthropic']:
                    response, tokens = self._generate_api_response(prompt)
                else:
                    raise RuntimeError(f"Unknown backend: {self.model_handler.backend}")
                
                # If we get here, generation was successful
                metadata = {
                    'attempt': attempt + 1,
                    'backend': self.model_handler.backend,
                    'retry_count': retry_count,
                    'token_counting_method': self.token_counting_method
                }
                
                if attempt > 0:
                    logging.info(f"✅ Generation successful on attempt {attempt + 1}")
                
                return response, tokens, True, metadata
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                error_msg = str(e)
                logging.warning(f"⚠️  Generation attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < self.max_retries:
                    # Wait before retrying with exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    logging.info(f"🔄 Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
                    # For some errors, try regenerating the data point
                    if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                        logging.info("🔄 CUDA/Memory error detected, regenerating data point...")
                        try:
                            # Attempt to regenerate data point (implementation specific)
                            new_data_point = self.regenerate_data_point(data_point)
                            if new_data_point != data_point:
                                logging.info("✨ Generated new data point for retry")
                                data_point.update(new_data_point)
                                prompt = self.get_prompt(data_point)
                        except Exception as regen_error:
                            logging.warning(f"⚠️  Failed to regenerate data point: {regen_error}")
        
        # All retries failed
        logging.error(f"❌ All {self.max_retries + 1} generation attempts failed. Last error: {last_error}")
        
        metadata = {
            'attempt': self.max_retries + 1,
            'backend': self.model_handler.backend,
            'retry_count': retry_count,
            'final_error': str(last_error),
            'token_counting_method': self.token_counting_method
        }
        
        return "", 0, False, metadata
    
    def _generate_vllm_response(self, prompt: str) -> Tuple[str, int]:
        """Generate response using vLLM backend with accurate token counting.

        Chat template is now applied centrally in ModelHandler._generate_local(),
        so we pass the raw user prompt here.
        """
        # Use the model_handler's generate method (handler applies chat template)
        responses = self.model_handler.generate(
            prompts=[prompt],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        
        response = responses[0] if responses else ""
        
        # Count tokens accurately
        tokens = self.count_tokens(response)
        
        return response, tokens
    
    def _generate_transformers_response(self, prompt: str) -> Tuple[str, int]:
        """Generate response using Transformers backend with accurate token counting"""
        # Use the model_handler's generate method
        responses = self.model_handler.generate(
            prompts=[prompt],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        
        response = responses[0] if responses else ""
        
        # Count tokens accurately
        tokens = self.count_tokens(response)
        
        return response, tokens

    def _generate_api_response(self, prompt: str) -> Tuple[str, int]:
        """Generate response using API backends (OpenAI, Gemini) with accurate token counting"""
        # Use the model_handler's generate method for API calls
        responses = self.model_handler.generate(
            prompts=[prompt],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )

        response = responses[0] if responses else ""

        # Count tokens accurately - API backends handle this internally
        tokens = self.count_tokens(response)

        return response, tokens

    def regenerate_data_point(self, original_data_point: Dict) -> Dict:
        """
        Regenerate a data point when the original fails
        Default implementation - subclasses can override for specific logic
        
        Args:
            original_data_point: The original data point that failed
            
        Returns:
            New data point dictionary
        """
        try:
            # Generate a new data point using the same parameters
            new_data = self.generate_data()
            if isinstance(new_data, list) and len(new_data) > 0:
                return new_data[0]  # Return first item if it's a list
            elif isinstance(new_data, dict):
                return new_data
            else:
                logging.warning("⚠️  Regenerated data is not in expected format")
                return original_data_point
        except Exception as e:
            logging.warning(f"⚠️  Failed to regenerate data point: {e}")
            return original_data_point
    
    def _run_fold_batch(self, data, test_case_id, fold) -> List[Dict]:
        """
        Run fold using batch processing for vLLM backend.

        This method:
        1. Generates ALL prompts first
        2. Runs batch inference on all prompts at once
        3. Evaluates all responses

        This is MUCH more efficient for vLLM as it utilizes GPU parallelism.
        """
        fold_results = []

        logging.info(f"🚀 Using BATCH processing mode for {len(data)} samples")

        # Step 1: Generate all prompts
        logging.info("📝 Step 1: Generating all prompts...")
        prompts = []
        data_points = []

        for i, data_point in enumerate(tqdm(data, desc="Preparing prompts")):
            try:
                prompt = self.get_prompt(data_point)
                # Chat template is now applied centrally in ModelHandler._generate_local()
                prompts.append(prompt)
                data_points.append((i, data_point, prompt))

            except Exception as e:
                logging.warning(f"⚠️  Failed to create prompt for sample {i}: {e}")
                prompts.append("")
                data_points.append((i, data_point, ""))

        logging.info(f"✅ Prepared {len(prompts)} prompts for batch inference")

        # Step 2: Run batch inference
        logging.info("🤖 Step 2: Running batch inference with vLLM...")
        try:
            responses = self.model_handler.generate(
                prompts=prompts,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            logging.info(f"✅ Batch inference complete: {len(responses)} responses generated")

        except Exception as e:
            logging.error(f"❌ Batch inference failed: {e}")
            logging.info("🔄 Retrying failed prompts sequentially...")
            responses = []
            retry_failures = 0
            for idx, (i, data_point, original_prompt) in enumerate(data_points):
                try:
                    retry_resp = self.model_handler.generate(
                        prompts=[prompts[idx]],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens
                    )
                    responses.append(retry_resp[0] if retry_resp else "")
                except Exception as retry_e:
                    logging.warning(f"⚠️  Sequential retry failed for sample {i}: {retry_e}")
                    responses.append("")
                    retry_failures += 1
            if retry_failures > 0:
                logging.warning(f"⚠️  {retry_failures}/{len(data_points)} samples failed even on sequential retry")

        # Retry individual empty responses from batch that may have partially failed
        partial_failure_count = sum(1 for r in responses if not r)
        if partial_failure_count > 0 and partial_failure_count < len(responses):
            logging.info(f"🔄 Retrying {partial_failure_count} empty responses sequentially...")
            retried = 0
            for idx, resp in enumerate(responses):
                if not resp and prompts[idx]:
                    try:
                        retry_resp = self.model_handler.generate(
                            prompts=[prompts[idx]],
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_tokens=self.max_tokens
                        )
                        if retry_resp and retry_resp[0]:
                            responses[idx] = retry_resp[0]
                            retried += 1
                    except Exception as retry_e:
                        logging.debug(f"Sequential retry failed for sample {idx}: {retry_e}")
            if retried > 0:
                logging.info(f"✅ Recovered {retried}/{partial_failure_count} failed samples via sequential retry")

        # Step 3: Evaluate all responses
        logging.info("📊 Step 3: Evaluating all responses...")
        progress_bar = tqdm(zip(data_points, responses), total=len(responses), desc="Evaluating responses")

        for (i, data_point, original_prompt), response in progress_bar:
            try:
                # Count tokens
                tokens = self.count_tokens(response)

                # Initialize result
                eval_result = {
                    "prompt": original_prompt,
                    "model_response": response,
                    "string_len": len(response),
                    "words": len(response.split()) if response else 0,
                    "tokens": tokens,
                    "generation_success": bool(response),
                    "generation_metadata": {
                        "backend": "vllm",
                        "batch_mode": True,
                        "token_counting_method": self.token_counting_method
                    }
                }

                if response:
                    # Evaluate response
                    try:
                        response_eval = self.evaluate_response(response, data_point)
                        eval_result.update(response_eval)
                        eval_result["parsing_success"] = True

                    except Exception as eval_error:
                        logging.warning(f"⚠️  Evaluation failed for sample {i}: {eval_error}")
                        ground_truth = self._get_ground_truth(data_point)
                        eval_result.update({
                            "accuracy": 0,
                            "instruction_followed": 0,
                            "parsing_success": False,
                            "parsing_error": str(eval_error),
                            "predicted_answer": None,
                            "ground_truth": ground_truth
                        })
                else:
                    # Empty response
                    ground_truth = self._get_ground_truth(data_point)
                    eval_result.update({
                        "accuracy": 0,
                        "instruction_followed": 0,
                        "parsing_success": False,
                        "parsing_error": "Empty response",
                        "predicted_answer": None,
                        "ground_truth": ground_truth
                    })

                # Add fingerprint
                try:
                    from .fingerprint import fingerprint_instance
                    eval_result['fingerprint'] = fingerprint_instance(data_point, self.task_name, self.seed)
                except Exception:
                    pass

                fold_results.append(eval_result)

                # Update progress bar
                if len(fold_results) > 0:
                    current_accuracy = sum(r.get('accuracy', 0) for r in fold_results) / len(fold_results)
                    _postfix = {'acc': f"{current_accuracy:.3f}"}
                    _gpu_s = _progress_gpu_postfix()
                    if _gpu_s:
                        _postfix['gpu'] = _gpu_s
                    progress_bar.set_postfix(_postfix)

            except Exception as e:
                logging.error(f"❌ Error processing sample {i}: {e}")
                ground_truth = self._get_ground_truth(data_point)
                fold_results.append({
                    "prompt": original_prompt,
                    "model_response": response if response else "",
                    "string_len": 0,
                    "words": 0,
                    "tokens": 0,
                    "generation_success": False,
                    "parsing_success": False,
                    "accuracy": 0,
                    "instruction_followed": 0,
                    "error": str(e),
                    "predicted_answer": None,
                    "ground_truth": ground_truth
                })

        progress_bar.close()
        return fold_results

    def _run_fold_sequential(self, data, test_case_id, fold) -> List[Dict]:
        """
        Run fold using sequential processing (one sample at a time).

        Used for:
        - API backends (OpenAI, Gemini, Anthropic)
        - Transformers backend
        - Cases where batch processing is not available
        """
        fold_results = []
        consecutive_failures = 0

        logging.info(f"📝 Using SEQUENTIAL processing mode for {len(data)} samples")

        progress_bar = tqdm(data, desc=f"Test case {test_case_id} - Fold {fold}")

        for i, data_point in enumerate(progress_bar):
            try:
                # Create prompt
                prompt = self.get_prompt(data_point)

                # Generate response with retry mechanism
                response, tokens, generation_success, generation_metadata = self.generate_response_with_retry(
                    prompt, data_point
                )

                # Initialize result with generation info
                eval_result = {
                    "prompt": prompt,
                    "model_response": response,
                    "string_len": len(response),
                    "words": len(response.split()),
                    "tokens": tokens,
                    "generation_success": generation_success,
                    "generation_metadata": generation_metadata
                }

                if generation_success:
                    # Reset consecutive failures counter
                    consecutive_failures = 0

                    # Evaluate response
                    try:
                        response_eval = self.evaluate_response(response, data_point)
                        eval_result.update(response_eval)
                        eval_result["parsing_success"] = True

                    except Exception as eval_error:
                        logging.warning(f"⚠️  Response evaluation failed for sample {i}: {eval_error}")
                        ground_truth = self._get_ground_truth(data_point)
                        eval_result.update({
                            "accuracy": 0,
                            "instruction_followed": 0,
                            "parsing_success": False,
                            "parsing_error": str(eval_error),
                            "predicted_answer": None,
                            "ground_truth": ground_truth
                        })

                else:
                    # Generation failed completely
                    consecutive_failures += 1
                    ground_truth = self._get_ground_truth(data_point)
                    eval_result.update({
                        "accuracy": 0,
                        "instruction_followed": 0,
                        "parsing_success": False,
                        "parsing_error": "Generation failed",
                        "predicted_answer": None,
                        "ground_truth": ground_truth
                    })

                    # Check if we should stop due to too many consecutive failures
                    if consecutive_failures >= self.max_consecutive_failures:
                        logging.error(f"❌ Stopping evaluation due to {consecutive_failures} consecutive failures")
                        break

                # Add fingerprint
                try:
                    from .fingerprint import fingerprint_instance
                    eval_result['fingerprint'] = fingerprint_instance(data_point, self.task_name, self.seed)
                except Exception:
                    pass

                fold_results.append(eval_result)

                # Update progress bar with current stats
                if len(fold_results) > 0:
                    current_accuracy = sum(r.get('accuracy', 0) for r in fold_results) / len(fold_results)
                    _postfix = {
                        'acc': f"{current_accuracy:.3f}",
                        'fails': consecutive_failures,
                    }
                    _gpu_s = _progress_gpu_postfix()
                    if _gpu_s:
                        _postfix['gpu'] = _gpu_s
                    progress_bar.set_postfix(_postfix)

            except KeyboardInterrupt:
                logging.info("⏹️  Evaluation interrupted by user")
                break

            except Exception as e:
                logging.error(f"❌ Unexpected error processing sample {i}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")

                ground_truth = self._get_ground_truth(data_point)
                fold_results.append({
                    "prompt": "Error occurred before prompt creation",
                    "model_response": "",
                    "string_len": 0,
                    "words": 0,
                    "tokens": 0,
                    "generation_success": False,
                    "parsing_success": False,
                    "accuracy": 0,
                    "instruction_followed": 0,
                    "error": str(e),
                    "predicted_answer": None,
                    "ground_truth": ground_truth
                })

                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    logging.error(f"❌ Stopping evaluation due to {consecutive_failures} consecutive failures")
                    break

        progress_bar.close()
        return fold_results

    def _get_ground_truth(self, data_point) -> Any:
        """Extract ground truth from a data point, handling different formats."""
        if isinstance(data_point, dict):
            # Try common ground truth keys used across different task types
            for key in ['answer', 'ground_truth', 'expected_answer', 'solution',
                        'correct_answer', 'target', 'label', 'sum', 'expected_relation',
                        'optimal_moves', 'next_term',
                        # Medium/Hard task ground truth keys
                        'expected_sequence', 'expected_grid', 'expected_moves',
                        'expected_path', 'valid_assignment',
                        # Additional keys used by various tasks
                        'sorted_list', 'maximum', 'minimum', 'mean', 'median',
                        'mode', 'count', 'product', 'difference', 'quotient',
                        'result', 'expected_output']:
                if key in data_point:
                    return data_point[key]
            return "unknown"
        elif isinstance(data_point, (list, tuple)):
            # For list-based tasks (e.g., sorting, sum), the data_point IS the input
            # Ground truth depends on the task, so return the raw data
            return str(data_point)
        else:
            return str(data_point) if data_point is not None else "unknown"

    def run_fold(self, data, test_case_id, fold):
        """Run a single evaluation fold with batch processing support for vLLM"""
        fold_results = []

        # Check if we can use batch processing (vLLM backend)
        use_batch = self.model_handler.backend == 'vllm'

        if use_batch:
            # BATCH PROCESSING MODE: Generate all prompts first, then run batch inference
            fold_results = self._run_fold_batch(data, test_case_id, fold)
        else:
            # SEQUENTIAL MODE: Process one at a time (for API backends or transformers)
            fold_results = self._run_fold_sequential(data, test_case_id, fold)
        
        # Save detailed results if requested
        self.save_detailed_results(fold_results, test_case_id, fold)
        
        # Calculate fold metrics
        metrics = self.process_fold_metrics(fold_results)
        metrics.update({
            'test_case_id': test_case_id,
            'fold': fold,
            'task': self.task_name,
            'total_samples_attempted': len(data),
            'total_samples_completed': len(fold_results),
            'token_counting_method': self.token_counting_method
        })
        
        # Log fold summary
        logging.info(f"📊 Fold {fold} Summary:")
        logging.info(f"   - Samples: {metrics['total_samples_completed']}/{metrics['total_samples_attempted']}")
        logging.info(f"   - Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"   - Generation Success Rate: {metrics.get('generation_success_rate', 0):.4f}")
        logging.info(f"   - Parsing Success Rate: {metrics.get('parsing_success_rate', 0):.4f}")
        logging.info(f"   - Avg Tokens per Response: {metrics.get('avg_output_tokens', 0):.1f}")
        
        return metrics
    
    def run_evaluation(self, list_sizes: Optional[List[int]] = None) -> List[Dict]:
        """
        Run complete evaluation across all test cases and folds
        
        Args:
            list_sizes: List of sizes to test (for applicable tasks)
            
        Returns:
            List of metrics dictionaries
        """
        all_metrics = []
        
        try:
            # Determine test cases based on task type
            if list_sizes is not None:
                test_cases = list_sizes
                logging.info(f"🎯 Running {self.task_name} evaluation with list sizes: {test_cases}")
            else:
                test_cases = [None]  # Single test case for non-list tasks
                logging.info(f"🎯 Running {self.task_name} evaluation")
            
            for test_case_id, test_case in enumerate(test_cases):
                logging.info(f"\n{'='*50}")
                logging.info(f"📋 Test Case {test_case_id + 1}: {test_case if test_case is not None else 'Default'}")
                logging.info(f"{'='*50}")
                
                # Generate data for this test case
                try:
                    if test_case is not None:
                        data = self.generate_data(list_size=test_case)
                    else:
                        data = self.generate_data()
                    
                    logging.info(f"✅ Generated {len(data)} samples for test case")
                    
                except Exception as e:
                    logging.error(f"❌ Failed to generate data for test case {test_case}: {e}")
                    continue
                
                # Run all folds for this test case
                for fold in range(self.num_folds):
                    try:
                        fold_metrics = self.run_fold(data, test_case_id, fold)
                        all_metrics.append(fold_metrics)
                        
                    except Exception as e:
                        logging.error(f"❌ Failed to complete fold {fold} for test case {test_case}: {e}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        continue
            
            # Log overall summary
            if all_metrics:
                avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
                avg_gen_success = np.mean([m.get('generation_success_rate', 0) for m in all_metrics])
                avg_tokens = np.mean([m.get('avg_output_tokens', 0) for m in all_metrics])
                logging.info(f"\n🎉 {self.task_name} Evaluation Complete!")
                logging.info(f"📊 Overall Results:")
                logging.info(f"   - Average Accuracy: {avg_accuracy:.4f}")
                logging.info(f"   - Average Generation Success Rate: {avg_gen_success:.4f}")
                logging.info(f"   - Average Tokens per Response: {avg_tokens:.1f}")
                logging.info(f"   - Token Counting Method: {self.token_counting_method}")
                logging.info(f"   - Total Metrics Collected: {len(all_metrics)}")
            else:
                logging.warning(f"⚠️  No metrics collected for {self.task_name}")
                
        except Exception as e:
            logging.error(f"❌ Critical error in {self.task_name} evaluation: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
        
        return all_metrics
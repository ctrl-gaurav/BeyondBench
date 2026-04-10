# model_handler.py
"""
Model handler for BeyondBench evaluation framework.

Supports multiple backends:
- OpenAI GPT models (gpt-4o, gpt-5, etc.)
- Google Gemini models (gemini-2.5-pro, etc.)
- Anthropic Claude models (claude-sonnet-4-20250514, etc.)
- Local models via vLLM or Transformers
"""

import logging
import time
from typing import Optional, List, Dict, Any

# Model loading and inference handler with API support
class ModelHandler:
    """
    Enhanced model handler with multi-backend API support.

    Supported backends:
    - OpenAI GPT models (gpt-4o, gpt-4o-mini, gpt-5, gpt-5-mini, gpt-5-nano)
    - Google Gemini models (gemini-2.5-pro, gemini-2.5-flash)
    - Anthropic Claude models (claude-sonnet-4-20250514, claude-opus-4-20250514)
    - Local models via vLLM or Transformers
    """

    def __init__(self, model_id: str, api_provider: str = None, api_key: str = None,
                 reasoning_effort: str = None, thinking_budget: int = None,
                 backend: str = None, **kwargs):
        """
        Initialize model handler for API-based or local inference.

        Args:
            model_id: Model identifier (e.g., 'gpt-4o', 'gemini-pro', 'meta-llama/Llama-3-8B-Instruct')
            api_provider: API provider ('openai', 'gemini', 'anthropic'), None for local models
            api_key: API key for the provider
            reasoning_effort: OpenAI reasoning effort ('minimal', 'low', 'medium', 'high')
            thinking_budget: Gemini thinking budget (integer, 0 to disable, -1 for dynamic)
            backend: For local models: 'vllm' (default, fast) or 'transformers' (HuggingFace)
            **kwargs: Additional parameters (cuda_device, tensor_parallel_size, gpu_memory_utilization, trust_remote_code)
        """
        self.model_id = model_id
        self.api_provider = api_provider
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = thinking_budget
        self.client = None
        self.skip_chat_template = kwargs.get('skip_chat_template', False)
        self._chat_template_warned = False

        # Determine backend: api_provider takes precedence, then explicit backend, then kwargs engine, then default vllm
        if api_provider:
            self.backend = api_provider
        elif backend:
            self.backend = backend
        else:
            self.backend = kwargs.get('engine', 'vllm')

        # Statistics tracking
        self.stats = {
            'api_calls': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'start_time': time.time(),
            'prompts_processed': []
        }

        # Initialize token counter
        self.token_counter = None
        self._init_token_counter()

        if api_provider and api_key:
            # Setup API client
            self._setup_api_client()
        elif api_provider:
            # API provider specified but no key
            raise RuntimeError(
                f"API provider '{api_provider}' specified but no API key provided.\n"
                "Please provide both --api_provider and --api_key parameters."
            )
        else:
            # Local model setup (VLLM or transformers)
            self._setup_local_model(**kwargs)

    def _init_token_counter(self):
        """Initialize tiktoken for accurate token counting"""
        try:
            import tiktoken
            # Use appropriate encoding based on model
            if 'gpt-4' in self.model_id.lower():
                self.token_counter = tiktoken.encoding_for_model("gpt-4")
            elif 'gpt-3.5' in self.model_id.lower():
                self.token_counter = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base for most modern models
                self.token_counter = tiktoken.get_encoding("cl100k_base")
            logging.info("✅ Initialized tiktoken for accurate token counting")
        except ImportError:
            logging.warning("⚠️ tiktoken not available. Install with: pip install tiktoken")
            self.token_counter = None
        except Exception as e:
            logging.warning(f"⚠️ Failed to initialize tiktoken: {e}")
            self.token_counter = None

    def _setup_api_client(self):
        """Setup the appropriate API client"""
        if self.api_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logging.info(f"✅ Successfully initialized OpenAI client for model '{self.model_id}'")
            except ImportError:
                raise RuntimeError("OpenAI package not installed. Run: pip install openai")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        elif self.api_provider == "gemini":
            try:
                # Use the new genai package structure
                from google import genai
                from google.genai import types
                self.client = genai.Client()
                self.genai_types = types
                logging.info(f"✅ Successfully initialized Gemini client for model '{self.model_id}'")
            except ImportError:
                try:
                    # Fallback to old google-generativeai package
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self.client = genai
                    self.genai_types = None
                    logging.info(f"✅ Successfully initialized Gemini client (legacy) for model '{self.model_id}'")
                except ImportError:
                    raise RuntimeError("Google GenAI package not installed. Run: pip install google-genai")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini client: {e}")

        elif self.api_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logging.info(f"✅ Successfully initialized Anthropic client for model '{self.model_id}'")
            except ImportError:
                raise RuntimeError("Anthropic package not installed. Run: pip install anthropic")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Anthropic client: {e}")

        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")

    def _setup_local_model(self, **kwargs):
        """Setup local model using VLLM or transformers.

        By default, tries vLLM first for optimal performance. If vLLM is not
        available or fails to initialize, automatically falls back to transformers.
        """
        # Use self.backend which was already set in __init__
        engine = self.backend

        if engine == 'vllm':
            try:
                from vllm import LLM, SamplingParams
                from transformers import AutoTokenizer
                # VLLM setup with original parameters
                self.model = LLM(
                    model=self.model_id,
                    tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
                    gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.96),
                    trust_remote_code=kwargs.get('trust_remote_code', False)
                )
                self.sampling_params_class = SamplingParams
                # Setup tokenizer for VLLM
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=kwargs.get('trust_remote_code', False)
                )
                logging.info(f"✅ Successfully initialized VLLM model '{self.model_id}'")
                return
            except ImportError:
                logging.warning("⚠️ vLLM not installed. Falling back to transformers backend. "
                              "For better performance, install vLLM: pip install vllm")
                engine = 'transformers'
                self.backend = 'transformers'
            except Exception as e:
                logging.warning(f"⚠️ vLLM initialization failed: {e}. Falling back to transformers backend.")
                engine = 'transformers'
                self.backend = 'transformers'

        if engine == 'transformers':
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                # Transformers setup
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=kwargs.get('trust_remote_code', False)
                )
                logging.info(f"✅ Successfully initialized Transformers model '{self.model_id}'")
            except ImportError:
                raise RuntimeError("Transformers not installed. Run: pip install transformers torch")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Transformers model: {e}")
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def generate(self, prompts: List[str], max_tokens: int = 32768, temperature: float = 0.1,
                 top_p: float = 0.9, **kwargs) -> List[str]:
        """
        Generate responses for given prompts using API or local models.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens to generate (default: 32768, falls back to 8192 on error)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            List of generated responses
        """
        # Route to appropriate generation method
        try:
            if self.api_provider:
                return self._generate_api(prompts, max_tokens, temperature, top_p, **kwargs)
            else:
                return self._generate_local(prompts, max_tokens, temperature, top_p, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            # If the error is related to max_tokens being too large, retry with smaller value
            if max_tokens > 8192 and any(keyword in error_msg for keyword in [
                'max_tokens', 'max tokens', 'context length', 'maximum context',
                'token limit', 'sequence length', 'too long', 'exceeds',
                'max_model_len', 'max_new_tokens', 'maximum generation'
            ]):
                fallback_tokens = 8192
                logging.warning(
                    f"⚠️ max_tokens={max_tokens} caused an error. "
                    f"Falling back to max_tokens={fallback_tokens}. Error: {e}"
                )
                if self.api_provider:
                    return self._generate_api(prompts, fallback_tokens, temperature, top_p, **kwargs)
                else:
                    return self._generate_local(prompts, fallback_tokens, temperature, top_p, **kwargs)
            raise

    def _generate_api(self, prompts: List[str], max_tokens: int = 1024, temperature: float = 0.1,
                      top_p: float = 0.9, **kwargs) -> List[str]:
        """Generate responses using API providers.

        Supports streaming for OpenAI, Gemini, and Anthropic backends.
        Streaming is used by default to allow partial token tracking.
        """
        responses = []
        use_streaming = kwargs.get('stream', True)

        for prompt in prompts:
            generated_text = None
            max_rate_limit_retries = 3

            for attempt in range(max_rate_limit_retries + 1):
                try:
                    # Count input tokens
                    input_tokens = self._count_tokens(prompt)

                    if self.api_provider == "openai":
                        api_params = {
                            "model": self.model_id,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_completion_tokens": max_tokens,
                        }

                        if self.reasoning_effort:
                            api_params['reasoning'] = {"effort": self.reasoning_effort}

                        api_params['temperature'] = temperature
                        api_params['top_p'] = top_p

                        if use_streaming:
                            api_params['stream'] = True
                            stream = self.client.chat.completions.create(**api_params)
                            chunks = []
                            for chunk in stream:
                                if chunk.choices and chunk.choices[0].delta.content:
                                    chunks.append(chunk.choices[0].delta.content)
                            generated_text = "".join(chunks)
                            output_tokens = self._count_tokens(generated_text)
                            logging.info(f"🤖 OpenAI API call (streamed): model={self.model_id}, estimated_output_tokens={output_tokens}")
                        else:
                            response = self.client.chat.completions.create(**api_params)
                            generated_text = response.choices[0].message.content
                            if hasattr(response, 'usage') and response.usage:
                                input_tokens = response.usage.prompt_tokens
                                output_tokens = response.usage.completion_tokens
                                logging.info(f"🤖 OpenAI API call: model={self.model_id}, prompt_tokens={input_tokens}, completion_tokens={output_tokens}")
                            else:
                                output_tokens = self._count_tokens(generated_text)

                    elif self.api_provider == "gemini":
                        if self.genai_types:
                            config = self.genai_types.GenerateContentConfig(
                                max_output_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p
                            )

                            if self.thinking_budget is not None:
                                config.thinking_config = self.genai_types.ThinkingConfig(
                                    thinking_budget=self.thinking_budget
                                )

                            if use_streaming:
                                stream = self.client.models.generate_content_stream(
                                    model=self.model_id,
                                    contents=prompt,
                                    config=config
                                )
                                chunks = []
                                for chunk in stream:
                                    if hasattr(chunk, 'text') and chunk.text:
                                        chunks.append(chunk.text)
                                generated_text = "".join(chunks)
                            else:
                                response = self.client.models.generate_content(
                                    model=self.model_id,
                                    contents=prompt,
                                    config=config
                                )
                                generated_text = response.text
                        else:
                            model = self.client.GenerativeModel(self.model_id)
                            generation_config = self.client.types.GenerationConfig(
                                max_output_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p
                            )
                            response = model.generate_content(prompt, generation_config=generation_config)
                            generated_text = response.text

                        output_tokens = self._count_tokens(generated_text)
                        thinking_budget_msg = f", thinking_budget={self.thinking_budget}" if self.thinking_budget is not None else ""
                        logging.info(f"🧠 Gemini API call: model={self.model_id}, estimated_input_tokens={input_tokens}, estimated_output_tokens={output_tokens}{thinking_budget_msg}")

                    elif self.api_provider == "anthropic":
                        if use_streaming:
                            with self.client.messages.stream(
                                model=self.model_id,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                messages=[{"role": "user", "content": prompt}]
                            ) as stream:
                                generated_text = stream.get_final_text()
                                msg = stream.get_final_message()
                                if hasattr(msg, 'usage'):
                                    input_tokens = msg.usage.input_tokens
                                    output_tokens = msg.usage.output_tokens
                                else:
                                    output_tokens = self._count_tokens(generated_text)
                            logging.info(f"🎭 Anthropic API call (streamed): model={self.model_id}, input_tokens={input_tokens}, output_tokens={output_tokens}")
                        else:
                            response = self.client.messages.create(
                                model=self.model_id,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            generated_text = ""
                            for block in response.content:
                                if hasattr(block, 'text'):
                                    generated_text += block.text
                            if hasattr(response, 'usage'):
                                input_tokens = response.usage.input_tokens
                                output_tokens = response.usage.output_tokens
                            else:
                                output_tokens = self._count_tokens(generated_text)
                            logging.info(f"🎭 Anthropic API call: model={self.model_id}, input_tokens={input_tokens}, output_tokens={output_tokens}")

                    # Update statistics
                    self.stats['api_calls'] += 1
                    self.stats['total_input_tokens'] += input_tokens
                    self.stats['total_output_tokens'] += output_tokens
                    self.stats['prompts_processed'].append({
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'timestamp': time.time()
                    })

                    responses.append(generated_text)
                    break

                except Exception as e:
                    error_code = getattr(e, 'status_code', None)
                    if error_code == 429 and attempt < max_rate_limit_retries:
                        backoff = min(2 ** attempt * 0.5, 30)
                        logging.warning(f"Rate limited (429), retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_rate_limit_retries})")
                        time.sleep(backoff)
                        continue
                    logging.error(f"API call failed for prompt: {e}")
                    responses.append("")
                    break

        return responses

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to a raw user prompt.

        Chat templating is centralized here so both vLLM and transformers
        branches produce identically formatted inputs. Task-side code should
        pass plain user prompts — this method wraps them in the model's chat
        format (e.g. ``<|im_start|>user\n{prompt}<|im_end|>``).

        If the tokenizer has no chat template or ``skip_chat_template`` is set,
        the raw prompt is returned unchanged.
        """
        if getattr(self, 'skip_chat_template', False):
            return prompt

        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is None:
            return prompt

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            return formatted
        except Exception as e:
            if not getattr(self, '_chat_template_warned', False):
                logging.warning(
                    f"⚠️  Chat template not available for {self.model_id}, using raw prompt: {e}"
                )
                self._chat_template_warned = True
            return prompt

    def _generate_local(self, prompts: List[str], max_tokens: int = 1024, temperature: float = 0.1,
                        top_p: float = 0.9, **kwargs) -> List[str]:
        """Generate responses using local models (VLLM or transformers).

        Chat templates are applied here centrally for BOTH backends.
        Task code should pass raw user prompts — do NOT pre-template.
        """
        responses = []

        # Apply chat template to all prompts centrally
        formatted_prompts = [self._apply_chat_template(p) for p in prompts]

        if self.backend == 'vllm':
            # VLLM generation - supports efficient batch processing
            try:
                sampling_params = self.sampling_params_class(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                # Log batch processing info
                batch_size = len(formatted_prompts)
                if batch_size > 1:
                    logging.info(f"🚀 vLLM batch processing: {batch_size} prompts at once")

                # Generate for all prompts at once with VLLM (batched inference)
                # vLLM automatically handles batching for optimal throughput
                start_time = time.time()
                outputs = self.model.generate(formatted_prompts, sampling_params)
                generation_time = time.time() - start_time

                total_input_tokens = 0
                total_output_tokens = 0

                for output in outputs:
                    generated_text = output.outputs[0].text

                    # Count tokens for statistics
                    input_tokens = self._count_tokens(output.prompt)
                    output_tokens = self._count_tokens(generated_text)

                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    # Update statistics
                    self.stats['api_calls'] += 1
                    self.stats['total_input_tokens'] += input_tokens
                    self.stats['total_output_tokens'] += output_tokens
                    self.stats['prompts_processed'].append({
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'timestamp': time.time()
                    })

                    responses.append(generated_text)

                # Log batch completion
                if batch_size > 1:
                    tokens_per_sec = (total_input_tokens + total_output_tokens) / max(generation_time, 0.001)
                    logging.info(f"✅ vLLM batch complete: {batch_size} prompts in {generation_time:.2f}s ({tokens_per_sec:.1f} tok/s)")

            except Exception as e:
                logging.error(f"VLLM generation failed: {e}")
                responses = [""] * len(prompts)

        elif self.backend == 'transformers':
            # Transformers generation
            import torch
            for prompt in formatted_prompts:
                try:
                    # Tokenize input
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    input_tokens = len(inputs['input_ids'][0])

                    # Generate - use do_sample=False when temperature is 0
                    do_sample = temperature > 0
                    gen_kwargs = {
                        **inputs,
                        'max_new_tokens': max_tokens,
                        'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
                    }
                    if do_sample:
                        gen_kwargs['temperature'] = temperature
                        gen_kwargs['top_p'] = top_p
                        gen_kwargs['do_sample'] = True
                    else:
                        gen_kwargs['do_sample'] = False

                    with torch.no_grad():
                        outputs = self.model.generate(**gen_kwargs)

                    # Decode response
                    generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_tokens = len(generated_tokens)

                    # Update statistics
                    self.stats['api_calls'] += 1
                    self.stats['total_input_tokens'] += input_tokens
                    self.stats['total_output_tokens'] += output_tokens
                    self.stats['prompts_processed'].append({
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'timestamp': time.time()
                    })

                    responses.append(generated_text)

                except Exception as e:
                    logging.error(f"Transformers generation failed for prompt: {e}")
                    responses.append("")

        return responses

    def generate_batch(self, prompts: List[str], max_tokens: int = 1024,
                      temperature: float = 0.1, top_p: float = 0.9, **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts.
        For API-based models, this is the same as generate() but included for compatibility.
        """
        return self.generate(prompts, max_tokens, temperature, top_p, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_id,
            "backend": self.backend,
            "api_provider": self.api_provider,
            "model_type": "api_based" if self.api_provider else "local"
        }

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken for accuracy.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if not text:
            return 0
        if self.token_counter:
            try:
                return len(self.token_counter.encode(text))
            except Exception as e:
                logging.warning(f"Token counting failed: {e}")
        return int(len(text.split()) * 1.3)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (public method for compatibility).

        Args:
            text: Input text

        Returns:
            Token count
        """
        return self._count_tokens(text)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about API usage.

        Returns:
            Dictionary containing detailed statistics
        """
        end_time = time.time()
        total_time = end_time - self.stats['start_time']

        stats = {
            'total_time_seconds': total_time,
            'total_time_formatted': self._format_duration(total_time),
            'api_calls': self.stats['api_calls'],
            'total_input_tokens': self.stats['total_input_tokens'],
            'total_output_tokens': self.stats['total_output_tokens'],
            'total_tokens': self.stats['total_input_tokens'] + self.stats['total_output_tokens'],
            'average_input_tokens': self.stats['total_input_tokens'] / max(1, self.stats['api_calls']),
            'average_output_tokens': self.stats['total_output_tokens'] / max(1, self.stats['api_calls']),
            'tokens_per_second': (self.stats['total_input_tokens'] + self.stats['total_output_tokens']) / max(1, total_time),
            'api_calls_per_second': self.stats['api_calls'] / max(1, total_time),
            'model_id': self.model_id,
            'api_provider': self.api_provider
        }

        return stats

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    def print_statistics_report(self):
        """Print a comprehensive statistics report"""
        stats = self.get_statistics()

        print("\n" + "="*80)
        print("📊 API USAGE STATISTICS REPORT")
        print("="*80)
        print(f"🤖 Model: {stats['model_id']}")
        print(f"🌐 Provider: {stats['api_provider']}")
        print(f"⏱️  Total Time: {stats['total_time_formatted']}")
        print(f"📞 API Calls: {stats['api_calls']:,}")
        print("-"*80)
        print("📥 INPUT TOKENS:")
        print(f"   Total: {stats['total_input_tokens']:,}")
        print(f"   Average per call: {stats['average_input_tokens']:.1f}")
        print("-"*80)
        print("📤 OUTPUT TOKENS:")
        print(f"   Total: {stats['total_output_tokens']:,}")
        print(f"   Average per call: {stats['average_output_tokens']:.1f}")
        print("-"*80)
        print("📊 COMBINED METRICS:")
        print(f"   Total tokens: {stats['total_tokens']:,}")
        print(f"   Tokens per second: {stats['tokens_per_second']:.1f}")
        print(f"   API calls per second: {stats['api_calls_per_second']:.2f}")
        print("="*80)

        # Estimated costs (2025 pricing)
        input_cost, output_cost = self._estimate_cost(stats)
        if input_cost > 0 or output_cost > 0:
            total_cost = input_cost + output_cost
            provider_label = stats['api_provider'] or 'local'
            print(f"💰 ESTIMATED COST ({provider_label}):")
            print(f"   Input tokens: ${input_cost:.4f}")
            print(f"   Output tokens: ${output_cost:.4f}")
            print(f"   Total: ${total_cost:.4f}")
            print("   (Prices as of 2025, check provider pricing for current rates)")
            print("="*80)

    # 2025 pricing per 1M tokens (input, output)
    _PRICING = {
        # OpenAI
        'gpt-4o': (2.50, 10.00),
        'gpt-4o-mini': (0.15, 0.60),
        'gpt-5': (10.00, 30.00),
        'gpt-5-mini': (1.00, 4.00),
        # Gemini
        'gemini-2.5-pro': (1.25, 10.00),
        'gemini-2.5-flash': (0.15, 0.60),
        # Anthropic
        'claude-sonnet-4-20250514': (3.00, 15.00),
        'claude-opus-4-20250514': (15.00, 75.00),
        'claude-haiku-4-5-20251001': (0.80, 4.00),
    }

    def _estimate_cost(self, stats: dict) -> tuple:
        """Estimate cost based on token counts and 2025 model pricing.

        Returns:
            (input_cost, output_cost) tuple in USD
        """
        model_id = stats.get('model_id', '')
        input_tokens = stats.get('total_input_tokens', 0)
        output_tokens = stats.get('total_output_tokens', 0)

        # Find matching pricing entry
        for model_prefix, (inp_per_m, out_per_m) in self._PRICING.items():
            if model_prefix in model_id:
                input_cost = input_tokens * inp_per_m / 1_000_000
                output_cost = output_tokens * out_per_m / 1_000_000
                return input_cost, output_cost

        return 0.0, 0.0

    def estimate_cost_before_evaluation(self, num_prompts: int, avg_prompt_tokens: int = 200,
                                        avg_response_tokens: int = 500) -> float:
        """Estimate cost before evaluation starts.

        Args:
            num_prompts: Number of prompts to evaluate
            avg_prompt_tokens: Average tokens per prompt
            avg_response_tokens: Average tokens per response

        Returns:
            Estimated total cost in USD
        """
        total_input = num_prompts * avg_prompt_tokens
        total_output = num_prompts * avg_response_tokens
        stats = {
            'model_id': self.model_id,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
        }
        input_cost, output_cost = self._estimate_cost(stats)
        total = input_cost + output_cost
        if total > 0:
            logging.info(f"💰 Estimated cost for {num_prompts} prompts: ${total:.4f}")
        return total

    def __del__(self):
        """Cleanup method (no resources to clean for API clients)"""
        pass
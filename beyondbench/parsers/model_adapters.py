"""
Model-specific output normalization adapters.

Each adapter strips model-specific artifacts before parsing to improve
extraction accuracy. Auto-detection selects the right adapter given a model_id.
"""

import re
from typing import Optional


class BaseAdapter:
    """Identity adapter — no normalization."""

    NAME = "base"

    def normalize(self, text: str) -> str:
        return text if text else ""

    def strip_special_tokens(self, text: str) -> str:
        return text


class QwenAdapter(BaseAdapter):
    """
    Qwen family: strips <|im_end|>, <|im_start|>, <|endoftext|> tokens.
    Handles Chinese/English mixed output — preserves the numeric answer.
    """

    NAME = "qwen"

    _TOKENS = re.compile(
        r'<\|(?:im_end|im_start|endoftext|user|assistant|system)\|>'
        r'(?:assistant|user|system)?',
        re.IGNORECASE,
    )

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = self._TOKENS.sub('', text)
        # Clean any remaining angle-bracket tokens
        text = re.sub(r'<\|[^|]+\|>', '', text)
        return text.strip()


class LlamaAdapter(BaseAdapter):
    """
    Llama family: strips [INST]...[/INST] artifacts, verbose CoT preambles.
    """

    NAME = "llama"

    _INST_PATTERN = re.compile(r'\[/?INST\]', re.IGNORECASE)
    _PREAMBLE_PATTERN = re.compile(
        r'^(?:Let me (?:think|solve|work)|Sure!|Of course!|I\'ll|To solve this)[^\n]*\n',
        re.IGNORECASE | re.MULTILINE,
    )

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = self._INST_PATTERN.sub('', text)
        # Remove common verbose CoT openers (first line only)
        text = self._PREAMBLE_PATTERN.sub('', text, count=3)
        return text.strip()


class PhiAdapter(BaseAdapter):
    """
    Phi family: handles terse single-number outputs, <|end|> tokens.
    """

    NAME = "phi"

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<\|end\|>', '', text)
        text = re.sub(r'<\|[^|]+\|>', '', text)
        return text.strip()


class MistralAdapter(BaseAdapter):
    """
    Mistral family: strips [INST]...[/INST] artifacts.
    """

    NAME = "mistral"

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\[/?INST\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[/?SYS\]', '', text, flags=re.IGNORECASE)
        return text.strip()


class GemmaAdapter(BaseAdapter):
    """
    Gemma family: strips <end_of_turn> and <start_of_turn> tokens.
    """

    NAME = "gemma"

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<(?:end|start)_of_turn>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:model|user)\b\s*\n', '', text)
        return text.strip()


class APIAdapter(BaseAdapter):
    """
    API response normalization (OpenAI, Gemini, Anthropic).
    Handles any extra whitespace, role labels, or system markers.
    """

    NAME = "api"

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Strip leading/trailing whitespace and role labels
        text = re.sub(r'^(?:assistant|user|system):\s*', '', text, flags=re.IGNORECASE)
        return text.strip()


# ============================================================================
# Auto-detection
# ============================================================================

_ADAPTER_REGISTRY = [
    # (substring patterns, adapter class) — checked in order
    (['qwen', 'qwen2'], QwenAdapter),
    (['llama', 'llama-2', 'llama-3', 'codellama', 'meta-llama'], LlamaAdapter),
    (['phi', 'phi-2', 'phi-3', 'phi-3.5', 'microsoft/phi'], PhiAdapter),
    (['mistral', 'mixtral', 'mistralai'], MistralAdapter),
    (['gemma', 'gemma-2', 'google/gemma'], GemmaAdapter),
    (['gpt-', 'gpt4', 'text-davinci', 'claude-', 'gemini-'], APIAdapter),
]


def get_adapter(model_id: Optional[str]) -> BaseAdapter:
    """
    Given a model_id string, return the appropriate adapter.

    Falls back to BaseAdapter if no match is found.

    Args:
        model_id: HuggingFace model path or API model name.

    Returns:
        An instantiated adapter.
    """
    if not model_id:
        return BaseAdapter()

    lowered = model_id.lower()
    for patterns, adapter_cls in _ADAPTER_REGISTRY:
        if any(p in lowered for p in patterns):
            return adapter_cls()

    return BaseAdapter()

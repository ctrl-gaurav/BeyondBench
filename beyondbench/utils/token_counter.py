"""Token counting utilities with multi-backend support."""

import logging
from typing import Optional, Dict, Any


class TokenCounter:
    """Unified token counting with multiple backend support."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tiktoken_encoder = None
        self._init_tiktoken()

    def _init_tiktoken(self):
        """Initialize tiktoken for accurate token counting."""
        try:
            import tiktoken
            self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            self.logger.debug("✅ Initialized tiktoken encoder")
        except ImportError:
            self.logger.warning("⚠️ tiktoken not available, using fallback estimation")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize tiktoken: {e}")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text
            model: Model name for model-specific counting

        Returns:
            Token count
        """
        if not text:
            return 0

        # Try tiktoken first
        if self.tiktoken_encoder:
            try:
                return len(self.tiktoken_encoder.encode(text))
            except Exception as e:
                self.logger.warning(f"tiktoken encoding failed: {e}")

        # Fallback to estimation
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word-based heuristics."""
        # Rough estimation: 1 token ≈ 0.75 words for English
        return max(1, int(len(text.split()) * 1.3))
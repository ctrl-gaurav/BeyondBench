"""
BoxedStrategy: Extract answers from LaTeX boxed formats.

Handles \\boxed{X}, $$\\boxed{X}$$, nested, malformed, and all variants.
Takes the LAST match (most likely the final answer).
"""

import re
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ParseResult:
    """Result from a single parsing strategy."""
    value: Optional[str]
    confidence: float       # 0.0-1.0
    strategy_used: str
    raw_match: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.value is not None


# Ordered from most-specific to most-general
_BOXED_PATTERNS = [
    # Nested LaTeX text commands inside boxed
    (r'\\boxed\{\\text\{([^}]+)\}\}',          0.98),
    (r'\\boxed\{\\textbf\{([^}]+)\}\}',         0.98),
    (r'\\boxed\{\\mathrm\{([^}]+)\}\}',         0.98),
    (r'\\boxed\{\\mathbf\{([^}]+)\}\}',         0.98),
    # Double-dollar wrapped
    (r'\$\$\s*\\boxed\{([^{}]+)\}\s*\$\$',      0.97),
    # Single-dollar wrapped
    (r'\$\\boxed\{([^{}]+)\}\$',                 0.97),
    # LaTeX math environment with boxed
    (r'\\[\(\[]\\boxed\{([^{}]+)\}\\[\)\]]',    0.97),
    # Standard LaTeX boxed (handles nested braces)
    (r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', 0.96),
    # Markdown-style boxed
    (r'\[boxed\{([^{}]+)\}\]',                   0.90),
    # Alternative boxed formats
    (r'\\box\{([^{}]+)\}',                       0.88),
    (r'\\fbox\{([^{}]+)\}',                      0.88),
    # Answer/solution tags
    (r'<answer>\s*([^<]+?)\s*</answer>',         0.92),
    (r'<solution>\s*([^<]+?)\s*</solution>',     0.88),
    (r'\[ANSWER\]\s*([^\[]+?)\s*\[/ANSWER\]',   0.92),
    # Bold final answer
    (r'(?:final answer|the answer)(?:\s+is)?[:\s]+\*\*([^*]+)\*\*', 0.85),
    # Escaped braces (some model outputs)
    (r'\\boxed\s*\{\s*([^{}]+?)\s*\}',          0.95),
]

_LATEX_CLEANUP = [
    (r'\\text\{([^}]*)\}',    r'\1'),
    (r'\\textbf\{([^}]*)\}',  r'\1'),
    (r'\\mathrm\{([^}]*)\}',  r'\1'),
    (r'\\mathbf\{([^}]*)\}',  r'\1'),
    (r'\\displaystyle\s*',     ''),
    # Remove remaining LaTeX commands but preserve minus signs
    (r'\\(?!-)[a-zA-Z]+\s*',  ''),
]


def _clean_latex(text: str) -> str:
    for pattern, repl in _LATEX_CLEANUP:
        text = re.sub(pattern, repl, text)
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class BoxedStrategy:
    """Extract answers from LaTeX boxed and answer-tag formats."""

    NAME = "boxed"

    def extract(self, text: str) -> ParseResult:
        """
        Args:
            text: The model response text (should be cleaned of \\n already).
        Returns:
            ParseResult with best found match, or failure result.
        """
        all_matches = []  # (raw_match, confidence)

        for pattern, conf in _BOXED_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                all_matches.append((m, conf))

        if all_matches:
            # Take the last match with highest confidence among final-few
            raw = all_matches[-1][0]
            conf = all_matches[-1][1]
            cleaned = _clean_latex(raw.strip())
            if cleaned:
                return ParseResult(
                    value=cleaned,
                    confidence=conf,
                    strategy_used=self.NAME,
                    raw_match=raw,
                )

        # Malformed boxed: \boxed{42 (missing closing brace)
        malformed = re.findall(r'\\boxed\{([^{}]+)$', text, re.MULTILINE)
        if malformed:
            raw = malformed[-1].strip()
            cleaned = _clean_latex(raw)
            if cleaned:
                return ParseResult(
                    value=cleaned,
                    confidence=0.70,
                    strategy_used=self.NAME + "_malformed",
                    raw_match=raw,
                )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

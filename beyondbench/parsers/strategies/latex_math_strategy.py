"""
LatexMathStrategy: Extract answers from LaTeX math expressions.

Handles $...$, $$...$$, \\(...\\), \\[...\\], equation environments.
"""

import re
from typing import Optional
from .boxed_strategy import ParseResult


_LATEX_PATTERNS = [
    (r'\$\$([^$]+)\$\$',                                          0.75),
    (r'\$([^$\n]+)\$',                                            0.72),
    (r'\\[\(\[]([^\\]+)\\[\)\]]',                                  0.72),
    (r'\\begin\{equation\*?\}([\s\S]+?)\\end\{equation\*?\}',     0.72),
    (r'\\begin\{align\*?\}([\s\S]+?)\\end\{align\*?\}',           0.70),
    (r'\\begin\{math\}([\s\S]+?)\\end\{math\}',                   0.70),
]

_CLEANUP = [
    (r'\\text\{([^}]*)\}',   r'\1'),
    (r'\\textbf\{([^}]*)\}', r'\1'),
    (r'\\mathrm\{([^}]*)\}', r'\1'),
    (r'\\[a-zA-Z]+\s*',      ''),
]


def _clean(text: str) -> str:
    for pat, repl in _CLEANUP:
        text = re.sub(pat, repl, text)
    text = text.replace('{', '').replace('}', '').strip()
    return text


class LatexMathStrategy:
    """Extract answers from LaTeX math expressions."""

    NAME = "latex_math"

    def extract(self, text: str) -> ParseResult:
        all_matches = []

        for pattern, conf in _LATEX_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches:
                cleaned = _clean(m.strip())
                if cleaned:
                    all_matches.append((cleaned, conf, m))

        if all_matches:
            # Take the last match (most likely final answer)
            value, conf, raw = all_matches[-1]
            return ParseResult(
                value=value,
                confidence=conf,
                strategy_used=self.NAME,
                raw_match=raw,
            )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

"""
CodeBlockStrategy: Extract answers from code blocks.

Handles ```python...``` blocks, print() statements, return values, assignments.
"""

import re
from typing import Optional
from .boxed_strategy import ParseResult


class CodeBlockStrategy:
    """Extract answers from code blocks in model responses."""

    NAME = "code_block"

    _BLOCK_PATTERNS = [
        r'```(?:python|py|plaintext|text|output|result)?\s*\n([\s\S]+?)\n```',
        r'```([\s\S]+?)```',
        r'`([^`\n]{1,200})`',
    ]

    _INNER_PATTERNS = [
        # print("answer") or print(42)
        (r'print\s*\(\s*["\']?([^"\')\n]+?)["\']?\s*\)', 0.82),
        # return value
        (r'\breturn\s+([^\s;#\n]+)', 0.80),
        # answer = X or result = X
        (r'(?:answer|result|output|solution)\s*=\s*([^\s;#\n]+)', 0.78),
        # Standalone number/expression on last line
        (r'^([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$', 0.72),
    ]

    def extract(self, text: str) -> ParseResult:
        best: Optional[ParseResult] = None

        for block_pattern in self._BLOCK_PATTERNS:
            blocks = re.findall(block_pattern, text, re.IGNORECASE)
            if not blocks:
                continue

            # Try the last block first (most likely the final answer)
            for block in reversed(blocks):
                block = block.strip()
                if not block:
                    continue

                for inner_pattern, conf in self._INNER_PATTERNS:
                    matches = re.findall(inner_pattern, block, re.MULTILINE)
                    if matches:
                        raw = matches[-1].strip()
                        if raw and len(raw) < 200:
                            result = ParseResult(
                                value=raw,
                                confidence=conf,
                                strategy_used=self.NAME,
                                raw_match=raw,
                            )
                            if best is None or conf > best.confidence:
                                best = result
                            break  # found something in this block

                # If nothing matched the inner patterns, check if the whole block
                # is just a number (terse code block)
                if best is None:
                    lines = [l.strip() for l in block.splitlines() if l.strip()]
                    if lines:
                        last_line = lines[-1]
                        m = re.match(r'^([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', last_line)
                        if m:
                            best = ParseResult(
                                value=m.group(1).replace(',', ''),
                                confidence=0.68,
                                strategy_used=self.NAME + "_last_line",
                                raw_match=last_line,
                            )

        return best or ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

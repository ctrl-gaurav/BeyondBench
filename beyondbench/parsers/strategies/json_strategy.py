"""
JsonStrategy: Extract structured answers from JSON/dict formatted responses.
"""

import re
import json
from typing import Optional
from .boxed_strategy import ParseResult

_ANSWER_KEYS = {"answer", "result", "value", "solution", "output", "next_term", "next"}


class JsonStrategy:
    """Extract answers from JSON or Python dict formatted responses."""

    NAME = "json"

    def extract(self, text: str) -> ParseResult:
        # Try to find JSON objects in the response
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)

        for raw in reversed(matches):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Try Python-style dict (single quotes, trailing commas)
                try:
                    cleaned = raw.replace("'", '"')
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    obj = json.loads(cleaned)
                except (json.JSONDecodeError, Exception):
                    continue

            if not isinstance(obj, dict):
                continue

            # Look for answer-like keys
            for key in _ANSWER_KEYS:
                if key in obj:
                    val = str(obj[key]).strip()
                    if val:
                        return ParseResult(
                            value=val,
                            confidence=0.82,
                            strategy_used=self.NAME,
                            raw_match=raw,
                        )

            # If single-key dict, use the value
            if len(obj) == 1:
                val = str(list(obj.values())[0]).strip()
                if val:
                    return ParseResult(
                        value=val,
                        confidence=0.72,
                        strategy_used=self.NAME,
                        raw_match=raw,
                    )

        # Try parsing outer JSON arrays too
        arr_pattern = r'\[[\d\s,\.\-]+\]'
        arr_matches = re.findall(arr_pattern, text)
        if arr_matches:
            raw = arr_matches[-1]
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    return ParseResult(
                        value=str(parsed),
                        confidence=0.70,
                        strategy_used=self.NAME + "_array",
                        raw_match=raw,
                    )
            except json.JSONDecodeError:
                pass

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

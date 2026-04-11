"""Parser for interleave_lists task answers."""
from typing import Optional, List, Tuple
from .list_parsing_helpers import parse_list_answer


def parse_interleave_lists_answer(response: str) -> Tuple[Optional[List[float]], bool]:
    """Extract interleaved list from LLM response."""
    return parse_list_answer(response)

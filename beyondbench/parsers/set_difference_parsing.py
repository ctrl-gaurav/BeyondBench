"""Parser for set_difference task answers."""
from typing import Optional, List, Tuple
from .list_parsing_helpers import parse_list_answer


def parse_set_difference_answer(response: str) -> Tuple[Optional[List[float]], bool]:
    """Extract set difference list from LLM response."""
    return parse_list_answer(response)

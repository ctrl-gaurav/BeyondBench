"""Parser for moving_average task answers."""
from typing import Optional, List, Tuple
from .list_parsing_helpers import parse_list_answer


def parse_moving_average_answer(response: str) -> Tuple[Optional[List[float]], bool]:
    """Extract moving average list from LLM response."""
    return parse_list_answer(response)

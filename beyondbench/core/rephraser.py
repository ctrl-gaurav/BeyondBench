"""Problem rephrasing for contamination resistance."""
import random
from typing import Dict, List, Optional


class ProblemRephraser:
    """Generates varied phrasings for mathematical operations to resist memorization."""

    # Mapping from operation type to list of equivalent phrasings
    PHRASINGS: Dict[str, List[str]] = {
        "sum": [
            "Add the following numbers",
            "Calculate the total of these numbers",
            "What is the sum of",
            "Find the aggregate of",
            "Compute the total when you add",
        ],
        "sort_ascending": [
            "Sort this list in ascending order",
            "Arrange these numbers from smallest to largest",
            "Order these values in increasing sequence",
            "Rearrange from least to greatest",
        ],
        "find_maximum": [
            "Find the maximum value",
            "What is the largest number",
            "Identify the greatest element",
            "Determine the highest value",
        ],
        "find_minimum": [
            "Find the minimum value",
            "What is the smallest number",
            "Identify the least element",
            "Determine the lowest value",
        ],
        "count": [
            "Count the number of",
            "How many",
            "Determine the count of",
            "Find how many",
        ],
        "multiply": [
            "Multiply the following numbers",
            "Calculate the product of",
            "What is the result when you multiply",
            "Find the product of",
        ],
        "subtract": [
            "Subtract the second number from the first",
            "Calculate the difference between",
            "What is the result of subtracting",
            "Find the difference of",
        ],
        "mean": [
            "Calculate the mean",
            "Find the average",
            "Compute the arithmetic mean",
            "What is the average value",
        ],
        "median": [
            "Find the median",
            "Determine the middle value",
            "What is the median of",
            "Calculate the median value",
        ],
        "compare": [
            "Compare these two numbers",
            "Determine the relationship between",
            "Which number is larger",
            "Evaluate how these numbers relate",
        ],
        "sequence_completion": [
            "What comes next in this sequence",
            "Predict the next term",
            "Continue this pattern",
            "Find the next number in the series",
        ],
        "graph_problem": [
            "Given the following graph",
            "Consider this graph structure",
            "In the graph described below",
            "For the following network",
        ],
        "optimization": [
            "Find the optimal solution",
            "Determine the best possible outcome",
            "Maximize the result",
            "What is the optimal value",
        ],
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def rephrase(self, operation: str) -> str:
        """Get a random phrasing for the given operation type."""
        phrasings = self.PHRASINGS.get(operation, [operation])
        return self.rng.choice(phrasings)

    def get_all_phrasings(self, operation: str) -> List[str]:
        """Get all available phrasings for an operation."""
        return self.PHRASINGS.get(operation, [operation])

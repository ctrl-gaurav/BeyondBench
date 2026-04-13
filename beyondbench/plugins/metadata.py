"""
Task metadata system for BeyondBench plugin framework.

Tasks can declare metadata via a class attribute ``metadata`` or a
``get_metadata()`` classmethod.  The plugin system reads this to surface
richer information in list-tasks --detailed output.
"""

from dataclasses import dataclass, field
from typing import Optional


# Valid values for difficulty and category
VALID_DIFFICULTIES = ("easy", "medium", "hard")
VALID_CATEGORIES = (
    "arithmetic",
    "comparison",
    "sequence",
    "optimization",
    "constraint",
    "graph",
    "string",
    "statistical",
    "set_operation",
    "pattern",
    "general",
)


@dataclass
class TaskMetadata:
    """
    Metadata descriptor for a BeyondBench task.

    Fields
    ------
    name : str
        Canonical task name (must match BaseTask.task_name).
    description : str
        Human-readable description of what the task tests.
    difficulty : str
        Difficulty tier: "easy", "medium", or "hard".
    category : str
        Conceptual category from the taxonomy:
        arithmetic, comparison, sequence, optimization, constraint,
        graph, string, statistical, set_operation, pattern, general.
    author : str
        Author / maintainer name.
    version : str
        Semantic version string (e.g. "1.0.0").
    """

    name: str
    description: str = ""
    difficulty: str = "easy"
    category: str = "general"
    author: str = ""
    version: str = "1.0.0"

    # Extra free-form tags for future use
    tags: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{self.difficulty}'. "
                f"Must be one of: {', '.join(VALID_DIFFICULTIES)}"
            )
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'. "
                f"Must be one of: {', '.join(VALID_CATEGORIES)}"
            )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "category": self.category,
            "author": self.author,
            "version": self.version,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _infer_category_from_name(task_name: str) -> str:
    """Heuristically derive a category from a task name."""
    name_lower = task_name.lower()
    mapping = {
        "sort": "comparison",
        "compar": "comparison",
        "maximum": "comparison",
        "minimum": "comparison",
        "range": "comparison",
        "sum": "arithmetic",
        "multiply": "arithmetic",
        "multiplication": "arithmetic",
        "division": "arithmetic",
        "subtract": "arithmetic",
        "mean": "statistical",
        "median": "statistical",
        "mode": "statistical",
        "variance": "statistical",
        "standard_deviation": "statistical",
        "average": "statistical",
        "sequence": "sequence",
        "fibonacci": "sequence",
        "prime": "sequence",
        "geometric": "sequence",
        "arithmetic_prog": "sequence",
        "harmonic": "sequence",
        "collatz": "sequence",
        "pattern": "pattern",
        "graph": "graph",
        "path": "graph",
        "topological": "graph",
        "traveling": "optimization",
        "knapsack": "optimization",
        "coin_change": "optimization",
        "edit_distance": "string",
        "regex": "string",
        "palindrom": "string",
        "set_": "set_operation",
        "intersection": "set_operation",
        "difference": "set_operation",
        "n_queens": "constraint",
        "sudoku": "constraint",
        "sat": "constraint",
        "cryptarith": "constraint",
        "constraint": "constraint",
        "boolean_sat": "constraint",
        "hanoi": "optimization",
        "matrix": "arithmetic",
        "combinatorics": "arithmetic",
        "gcd": "arithmetic",
        "lcm": "arithmetic",
        "modular": "arithmetic",
        "base_conv": "arithmetic",
        "polynomial": "sequence",
        "logical": "constraint",
        "minimax": "optimization",
        "interval": "optimization",
        "count": "arithmetic",
        "odd": "arithmetic",
        "even": "arithmetic",
        "abs": "arithmetic",
        "dot_product": "arithmetic",
        "weighted": "arithmetic",
        "running": "statistical",
        "cumulative": "arithmetic",
        "moving": "statistical",
        "parity": "arithmetic",
        "reverse": "sequence",
        "rotate": "sequence",
        "interleave": "sequence",
        "element_freq": "statistical",
        "frequency": "statistical",
        "second_max": "comparison",
        "second_min": "comparison",
        "index_of": "comparison",
        "local_max": "comparison",
        "alternating": "sequence",
        "multiples": "arithmetic",
        "digit": "arithmetic",
        "perfect_square": "arithmetic",
        "longest": "sequence",
        "increasing": "sequence",
        "common_subsequence": "sequence",
    }
    for keyword, cat in mapping.items():
        if keyword in name_lower:
            return cat
    return "general"


def get_task_metadata(task_class, suite: str = "easy") -> "TaskMetadata":
    """
    Return a :class:`TaskMetadata` for *task_class*.

    Looks for metadata in this order:
    1. ``task_class.metadata`` class attribute (must be a ``TaskMetadata`` instance)
    2. ``task_class.get_metadata()`` classmethod
    3. Auto-generated defaults from the task name and suite.

    Parameters
    ----------
    task_class:
        A subclass of ``BaseTask``.
    suite : str
        The suite the task belongs to — used as the default difficulty when
        auto-generating metadata.
    """
    # 1. Class attribute
    meta = getattr(task_class, "metadata", None)
    if isinstance(meta, TaskMetadata):
        return meta

    # 2. get_metadata() classmethod
    get_meta = getattr(task_class, "get_metadata", None)
    if callable(get_meta):
        try:
            result = get_meta()
            if isinstance(result, TaskMetadata):
                return result
        except Exception:
            pass

    # 3. Auto-generate
    task_name: str = ""
    try:
        # task_name is an abstractproperty — access it on the class safely
        task_name_prop = getattr(task_class, "task_name", None)
        if isinstance(task_name_prop, property):
            # Instantiate would fail without args; derive from class name
            task_name = task_class.__name__.lower().replace("task", "").strip("_")
        elif isinstance(task_name_prop, str):
            task_name = task_name_prop
        else:
            task_name = task_class.__name__.lower().replace("task", "").strip("_")
    except Exception:
        task_name = task_class.__name__.lower()

    difficulty = suite if suite in VALID_DIFFICULTIES else "easy"
    category = _infer_category_from_name(task_name)
    description = task_name.replace("_", " ").title() + f" task ({suite} suite)"

    return TaskMetadata(
        name=task_name,
        description=description,
        difficulty=difficulty,
        category=category,
        author="",
        version="1.0.0",
    )

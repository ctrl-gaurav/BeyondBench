"""
Scaffolding generator for BeyondBench custom task plugins.

Called by ``beyondbench create-task <name>`` to produce a ready-to-develop
plugin project directory.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Name utilities
# ---------------------------------------------------------------------------

def _snake_to_pascal(name: str) -> str:
    """Convert ``my_task_name`` → ``MyTaskName``."""
    return "".join(word.capitalize() for word in name.split("_"))


def _validate_task_name(name: str) -> str:
    """
    Validate and normalise a plugin task name.

    Rules:
    - Must be a valid Python identifier.
    - Must start with a letter.
    - Only letters, digits, underscores allowed.

    Returns the normalised name (lowercase + underscores).
    """
    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    if not re.match(r"^[a-z][a-z0-9_]*$", normalised):
        raise ValueError(
            f"Invalid task name '{name}'. "
            "Must start with a letter and contain only letters, digits, "
            "and underscores."
        )
    return normalised


# ---------------------------------------------------------------------------
# File templates
# ---------------------------------------------------------------------------

_TASK_PY_TEMPLATE = '''\
"""
{task_name_title} task implementation.

TODO: Implement generate_data, create_prompt, and evaluate_response.
"""

import random
import logging
from beyondbench.core.base_task import BaseTask
from beyondbench.plugins import TaskMetadata
from .parser import parse_{task_name}


class {class_name}(BaseTask):
    """
    {description}

    Difficulty : {suite}
    Author     : {author}
    """

    # ------------------------------------------------------------------ #
    # Optional: declare rich metadata for list-tasks --detailed           #
    # ------------------------------------------------------------------ #
    metadata = TaskMetadata(
        name="{task_name}",
        description="{description}",
        difficulty="{suite}",
        category="general",  # TODO: pick the right category
        author="{author}",
        version="0.1.0",
    )

    @property
    def task_name(self) -> str:
        return "{task_name}"

    # ------------------------------------------------------------------ #
    # Data generation                                                      #
    # ------------------------------------------------------------------ #

    def generate_data(self, **kwargs):
        """
        Generate evaluation data points.

        Must return a list of data points.  Each data point is passed as
        the ``data_point`` argument to ``create_prompt`` and
        ``evaluate_response``.

        TODO: Replace this stub with your actual data generation logic.
        """
        rng = random.Random(self.seed)
        data = []
        for _ in range(self.num_samples):
            # TODO: generate a real sample
            sample = rng.randint(self.min_val, self.max_val)
            data.append(sample)
        return data

    # ------------------------------------------------------------------ #
    # Prompt creation                                                      #
    # ------------------------------------------------------------------ #

    def create_prompt(self, data_point) -> str:
        """
        Create the prompt that will be sent to the model.

        Args:
            data_point: A single element returned by generate_data().

        Returns:
            A string prompt.

        TODO: Replace with your actual prompt template.
        """
        return (
            f"TODO: write your prompt here.\\n"
            f"Input: {{data_point}}\\n\\n"
            f"Provide your answer inside \\\\boxed{{{{your answer here}}}}."
        )

    # ------------------------------------------------------------------ #
    # Response evaluation                                                  #
    # ------------------------------------------------------------------ #

    def evaluate_response(self, response: str, data_point) -> dict:
        """
        Evaluate the model\'s response against the ground truth.

        Args:
            response: Raw string output from the model.
            data_point: The original data point used to create the prompt.

        Returns:
            A dict with at minimum these keys:
              - "accuracy"  (float 0.0-1.0)
              - "instruction_followed"  (bool)

        TODO: Replace with your actual evaluation logic.
        """
        ground_truth = self._compute_ground_truth(data_point)
        parsed = parse_{task_name}(response)

        instruction_followed = parsed is not None
        accuracy = 0.0

        if instruction_followed:
            try:
                accuracy = 1.0 if parsed == ground_truth else 0.0
            except Exception as exc:
                logging.debug("Comparison error: %s", exc)

        return {{
            "input": data_point,
            "ground_truth": ground_truth,
            "parsed_answer": parsed,
            "instruction_followed": instruction_followed,
            "accuracy": accuracy,
        }}

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_ground_truth(self, data_point):
        """
        TODO: Compute the correct answer for data_point.
        """
        raise NotImplementedError("Implement _compute_ground_truth")
'''

_PARSER_PY_TEMPLATE = '''\
"""
Response parser for the {task_name_title} task.

TODO: Implement parse_{task_name} to extract the answer from the model\'s
      raw response string.
"""

import re
from typing import Optional


def parse_{task_name}(response: str) -> Optional[object]:
    """
    Parse the model\'s response and return the extracted answer.

    Looks for a \\\\boxed{{...}} pattern first, then falls back to a
    heuristic numeric extraction.

    Args:
        response: Raw model output string.

    Returns:
        The parsed answer, or None if extraction fails.

    TODO: Customise this for your specific answer format.
    """
    if not isinstance(response, str):
        return None

    # 1. Try \\boxed{{...}}
    boxed_match = re.search(r"\\\\boxed\\{{([^}}]*)\\}}", response)
    if boxed_match:
        raw = boxed_match.group(1).strip()
        return _cast_answer(raw)

    # 2. Fallback: last number in the response
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", response)
    if numbers:
        return _cast_answer(numbers[-1])

    return None


def _cast_answer(raw: str) -> Optional[object]:
    """Try to cast *raw* to int, then float, then return as string."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw if raw else None
'''

_INIT_PY_TEMPLATE = '''\
"""
{task_name_title} — BeyondBench plugin task.

Install with:

    pip install -e .

Then run:

    beyondbench list-tasks  # should show {task_name}
"""

from .task import {class_name}

__all__ = ["{class_name}"]
'''

_TEST_PY_TEMPLATE = '''\
"""
Tests for the {task_name_title} plugin task.

Run with:
    pytest test_task.py -v
"""

import pytest
from unittest.mock import MagicMock

from task import {class_name}


@pytest.fixture()
def task():
    """Return a {class_name} instance with a mock model handler."""
    mock_handler = MagicMock()
    mock_handler.get_model_info.return_value = {{"model_name": "mock-model"}}
    mock_handler.generate.return_value = ["\\\\boxed{{42}}"]
    return {class_name}(
        model_handler=mock_handler,
        output_dir="/tmp/test_{task_name}",
        min_val=0,
        max_val=100,
        num_folds=1,
        num_samples=5,
        store_details=False,
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        seed=42,
    )


def test_task_name(task):
    assert task.task_name == "{task_name}"


def test_generate_data_returns_list(task):
    data = task.generate_data()
    assert isinstance(data, list)
    assert len(data) == task.num_samples


def test_create_prompt_returns_string(task):
    data = task.generate_data()
    prompt = task.create_prompt(data[0])
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_evaluate_response_structure(task):
    data = task.generate_data()
    result = task.evaluate_response("\\\\boxed{{42}}", data[0])
    assert "accuracy" in result
    assert "instruction_followed" in result
    assert isinstance(result["accuracy"], (int, float))


def test_metadata():
    from beyondbench.plugins import TaskMetadata
    meta = {class_name}.metadata
    assert isinstance(meta, TaskMetadata)
    assert meta.name == "{task_name}"
'''

_PYPROJECT_TOML_TEMPLATE = '''\
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{task_name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"
authors = [
    {{name = "{author}"}},
]
dependencies = [
    "beyondbench",
]

[project.entry-points."beyondbench.tasks"]
{task_name} = "{task_name}:{class_name}"

[tool.setuptools.packages.find]
where = ["."]
include = ["{task_name}*"]
'''

_README_MD_TEMPLATE = '''\
# {task_name_title} — BeyondBench Plugin

{description}

## Installation

```bash
pip install -e .
```

## Usage

After installation, the task is automatically discovered by BeyondBench:

```bash
# Verify it appears in the task list
beyondbench list-tasks

# Run evaluation (requires a model)
beyondbench evaluate --model-id <model-id> --tasks {task_name}
```

## Development

Edit `task.py` to implement:

- `generate_data(**kwargs)` — produce evaluation data points
- `create_prompt(data_point)` — build the model prompt
- `evaluate_response(response, data_point)` — score the answer
- `_compute_ground_truth(data_point)` — compute expected answer

Edit `parser.py` to customise answer extraction.

Run the tests:

```bash
pytest test_task.py -v
```

## Metadata

| Field       | Value          |
|-------------|----------------|
| Suite       | {suite}        |
| Author      | {author}       |
| Version     | 0.1.0          |
'''


# ---------------------------------------------------------------------------
# Scaffold creation
# ---------------------------------------------------------------------------

def create_task_scaffold(
    task_name: str,
    output_dir: str = ".",
    suite: str = "easy",
    author: str = "",
    description: str = "",
) -> Path:
    """
    Create a scaffolded plugin project for a new BeyondBench task.

    Parameters
    ----------
    task_name : str
        Snake-case task name (e.g. ``my_custom_task``).
    output_dir : str
        Parent directory where the project folder will be created.
    suite : str
        Default suite: ``easy``, ``medium``, or ``hard``.
    author : str
        Author name for package metadata.
    description : str
        Short description of the task.

    Returns
    -------
    Path
        The path to the newly created project directory.
    """
    task_name = _validate_task_name(task_name)
    class_name = _snake_to_pascal(task_name) + "Task"
    task_name_title = task_name.replace("_", " ").title()

    if not description:
        description = f"{task_name_title} evaluation task for BeyondBench."
    if not author:
        author = "Unknown"
    if suite not in ("easy", "medium", "hard"):
        suite = "easy"

    base_dir = Path(output_dir) / task_name
    if base_dir.exists():
        raise FileExistsError(
            f"Directory '{base_dir}' already exists. "
            "Remove it or choose a different --output-dir."
        )

    base_dir.mkdir(parents=True)

    ctx = {
        "task_name": task_name,
        "class_name": class_name,
        "task_name_title": task_name_title,
        "suite": suite,
        "author": author,
        "description": description,
    }

    def _write(filename: str, template: str) -> None:
        content = template.format(**ctx)
        (base_dir / filename).write_text(content, encoding="utf-8")

    _write("task.py", _TASK_PY_TEMPLATE)
    _write("parser.py", _PARSER_PY_TEMPLATE)
    _write("__init__.py", _INIT_PY_TEMPLATE)
    _write("test_task.py", _TEST_PY_TEMPLATE)
    _write("pyproject.toml", _PYPROJECT_TOML_TEMPLATE)
    _write("README.md", _README_MD_TEMPLATE)

    return base_dir

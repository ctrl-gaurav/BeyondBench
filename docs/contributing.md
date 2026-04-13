# Contributing to BeyondBench

Thank you for your interest in contributing to BeyondBench! This guide covers how to add new tasks, parsers, model backends, and how to follow the code style and PR workflow.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Style and Quality](#code-style-and-quality)
3. [Testing](#testing)
4. [Adding a New Task](#adding-a-new-task)
5. [Adding a New Parser](#adding-a-new-parser)
6. [Adding a New Model Backend](#adding-a-new-model-backend)
7. [Pull Request Workflow](#pull-request-workflow)

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/ctrl-gaurav/BeyondBench.git
cd BeyondBench
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install in Editable Mode with Dev Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

Pre-commit runs `ruff` on every commit automatically.

### 5. Verify the Setup

```bash
beyondbench --version
pytest tests/ -v --timeout=30 -k "unit" -x
```

---

## Code Style and Quality

BeyondBench uses:
- **ruff** for linting and formatting (line length 100)
- **mypy** for optional type checking

### Running Quality Checks

```bash
# Lint
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Format check
ruff format --check .

# Apply formatting
ruff format .

# Type checking (optional, non-blocking for contributions)
mypy beyondbench/ --ignore-missing-imports
```

### Style Rules

- Maximum line length: **100 characters**
- Use `Optional[X]` over `X | None` for Python 3.10 compatibility
- All public classes and methods should have docstrings
- Follow existing conventions in the module you're editing

---

## Testing

### Running the Test Suite

```bash
# All tests (slow — requires GPU)
pytest tests/ -v

# Fast unit tests only (no GPU required)
pytest tests/ -v -m "unit" --timeout=30

# With coverage
pytest tests/ -v --cov=beyondbench --cov-report=term-missing

# Specific test file
pytest tests/unit/test_evaluation_engine.py -v
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Fast, no GPU required |
| `integration` | Requires real CUDA GPU |
| `e2e` | Full pipeline, real GPU |
| `api` | Requires API keys |
| `slow` | Long-running tests |
| `gpu` | Any test needing GPU |

### Writing Tests for a New Task

Place your test in `tests/unit/tasks/test_<task_name>_task.py`:

```python
import pytest
from unittest.mock import MagicMock

@pytest.mark.unit
class TestMyCustomTask:

    def setup_method(self):
        self.mock_handler = MagicMock()
        self.mock_handler.generate.return_value = [r"\boxed{42}"]
        self.mock_handler.get_model_info.return_value = {"model_name": "test", "backend": "vllm"}

    def test_task_name(self):
        from beyondbench.tasks.easy.my_custom_task import MyCustomTask
        task = MyCustomTask(
            model_handler=self.mock_handler,
            output_dir="/tmp/test",
            min_val=-10,
            max_val=10,
            num_folds=1,
            num_samples=5,
            store_details=False,
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )
        assert task.task_name == "my_custom_task"

    def test_generate_data(self):
        # ... test data generation
        pass

    def test_evaluate_response_correct(self):
        # ... test correct answer detection
        pass

    def test_evaluate_response_incorrect(self):
        # ... test incorrect answer detection
        pass
```

---

## Adding a New Task

### Step 1: Choose a Suite

| Suite | Criteria |
|-------|----------|
| `easy` | Basic arithmetic, list operations, single-step computation |
| `medium` | Sequences, pattern recognition, multi-step math |
| `hard` | Optimization, constraint satisfaction, dynamic programming |

### Step 2: Create the Task File

Create `beyondbench/tasks/<suite>/<task_name>_task.py`:

```python
"""
My Custom Task

One-line summary of what this task evaluates.

Data point format: { ... }
Output: type and description
"""

import random
import re
from ...core.base_task import BaseTask


class MyCustomTask(BaseTask):
    """Evaluates model ability to ... (one-sentence description)."""

    @property
    def task_name(self) -> str:
        return "my_custom_task"

    def generate_data(self, list_size: int = 10, **kwargs):
        """Generate evaluation data points.

        Args:
            list_size: Number of elements per sample (for list-based tasks)

        Returns:
            List of data points, one per sample
        """
        if self.seed is not None:
            random.seed(self.seed)

        return [
            [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            for _ in range(self.num_samples)
        ]

    def create_prompt(self, data_point) -> str:
        """Create the evaluation prompt for a single data point."""
        return (
            f"Given the list: {data_point}\n\n"
            "Compute the answer.\n\n"
            r"Put your final answer in \boxed{answer}."
        )

    def evaluate_response(self, response: str, data_point) -> bool:
        """Check if the model response is correct.

        Args:
            response: Raw string from the model
            data_point: The data point used to generate the prompt

        Returns:
            True if the response is correct, False otherwise
        """
        expected = sum(data_point)  # replace with actual computation
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if not match:
            return False
        try:
            return int(match.group(1).strip()) == expected
        except (ValueError, AttributeError):
            return False
```

### Step 3: Register the Task

Open `beyondbench/core/task_registry.py` and add your task name to the appropriate suite list inside `_discover_and_register_tasks`:

```python
easy_tasks = [
    # ... existing tasks ...
    'my_custom_task',   # <-- add here
]
```

### Step 4: Export from `__init__.py`

Open `beyondbench/tasks/<suite>/__init__.py` and add the import:

```python
from .my_custom_task import MyCustomTask
```

### Step 5: Write Tests

Create `tests/unit/tasks/test_my_custom_task.py` (see [Testing](#testing) section).

### Step 6: Verify

```bash
beyondbench list-tasks --suite easy   # should show my_custom_task
beyondbench evaluate --model-id gpt-4o --api-provider openai \
  --tasks my_custom_task --datapoints 5
```

---

## Adding a New Parser

Parsers live in `beyondbench/parsers/`. The unified parser (`core.py`) uses a strategy pattern.

### Step 1: Create a Parser Module

Create `beyondbench/parsers/<task_name>_parsing.py`:

```python
"""Parser for my_custom_task responses."""

import re
from typing import Optional


def parse_my_custom_task(response: str) -> Optional[int]:
    """Extract the answer from a model response.

    Tries \boxed{} notation first, then falls back to pattern matching.

    Args:
        response: Raw model response string

    Returns:
        Parsed integer answer, or None if parsing fails
    """
    # Try \boxed{} first (standard format)
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        try:
            return int(match.group(1).strip())
        except ValueError:
            pass

    # Fallback: look for a bare number at the end of the response
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^-?\d+$', line):
            return int(line)

    return None
```

### Step 2: Register in Unified Parser

Open `beyondbench/parsers/core.py` and register your parser in the strategy dispatch table:

```python
from .my_custom_task_parsing import parse_my_custom_task

TASK_PARSERS = {
    # ... existing entries ...
    "my_custom_task": parse_my_custom_task,
}
```

### Step 3: Use the Parser in Your Task

In `evaluate_response`, call your parser instead of inline regex:

```python
from ...parsers.my_custom_task_parsing import parse_my_custom_task

def evaluate_response(self, response, data_point):
    parsed = parse_my_custom_task(response)
    if parsed is None:
        return False
    return parsed == sum(data_point)
```

---

## Adding a New Model Backend

Model backends are implemented in `beyondbench/models/model_handler.py` within the `ModelHandler` class.

### Step 1: Add Setup Logic

In `_setup_api_client`, add a new branch for your provider:

```python
elif self.api_provider == "my_provider":
    try:
        import my_provider_sdk
        self.client = my_provider_sdk.Client(api_key=self.api_key)
        logging.info(f"Initialized my_provider client for '{self.model_id}'")
    except ImportError:
        raise RuntimeError("my_provider_sdk not installed. Run: pip install my-provider-sdk")
```

### Step 2: Add Generation Logic

In `_generate_api`, add generation for your provider:

```python
elif self.api_provider == "my_provider":
    response = self.client.generate(
        model=self.model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    generated_text = response.text
    self.stats["api_calls"] += 1
    self.stats["total_output_tokens"] += len(generated_text.split())
```

### Step 3: Add CLI Support

In `beyondbench/cli/main.py`, extend the `--api-provider` choice:

```python
@click.option('--api-provider',
              type=click.Choice(['openai', 'gemini', 'anthropic', 'my_provider']),
              ...)
```

### Step 4: Add Optional Dependency to `pyproject.toml`

```toml
[project.optional-dependencies]
my_provider = ["my-provider-sdk>=1.0.0"]
```

---

## Pull Request Workflow

1. **Fork** the repository on GitHub and clone your fork
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-new-task
   ```
3. **Make changes** following the code style guidelines
4. **Add tests** for any new functionality
5. **Run quality checks**:
   ```bash
   ruff check . && ruff format --check .
   pytest tests/ -v -m "unit" --timeout=30
   ```
6. **Commit** with a descriptive message:
   ```bash
   git commit -m "feat: add my_custom_task to easy suite"
   ```
7. **Push** and open a Pull Request against `main`:
   - Title: short, imperative mood ("Add X", "Fix Y", "Refactor Z")
   - Description: what you added, why, and how to test it
   - Link any related issues

### PR Checklist

- [ ] Task file created in the correct suite directory
- [ ] Task registered in `task_registry.py`
- [ ] Exported from `__init__.py`
- [ ] Tests added and passing (`pytest -m unit`)
- [ ] `ruff check` and `ruff format --check` pass
- [ ] Docstrings on new public classes and methods
- [ ] `beyondbench list-tasks` shows the new task
- [ ] Manual smoke test with `--datapoints 5` passes

---

## Getting Help

- Open an issue on [GitHub](https://github.com/ctrl-gaurav/BeyondBench/issues)
- Check existing issues for similar questions
- For research inquiries, contact the authors listed in `pyproject.toml`

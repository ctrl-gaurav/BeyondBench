# Contributing to BeyondBench

Thank you for your interest in contributing to BeyondBench!

## Development Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/BeyondBench.git
   cd BeyondBench
   ```

2. Create a virtual environment and install in development mode:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

3. (Optional) Install pre-commit hooks:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running Tests

```bash
pytest tests/ -v
```

To run with coverage:

```bash
pytest tests/ -v --cov=beyondbench
```

## Linting and Formatting

BeyondBench uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```bash
# Check for lint issues
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Check formatting
ruff format --check .

# Apply formatting
ruff format .
```

## Code Style

- Ruff handles all formatting and linting automatically.
- Maximum line length is 100 characters.
- Follow existing code conventions in the project.

## Adding a New Evaluation Task

1. Create a new Python file in the appropriate suite directory under `beyondbench/`.
2. Implement your task class following existing task patterns.
3. Register the task in `beyondbench/task_registry.py`.
4. Add tests for the new task in `tests/`.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and add tests where appropriate.
3. Ensure all tests pass (`pytest tests/ -v`).
4. Ensure linting passes (`ruff check .` and `ruff format --check .`).
5. Open a pull request against `main` with a clear description of your changes.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.

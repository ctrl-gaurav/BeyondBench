# Changelog

All notable changes to BeyondBench (beyondbench) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-03-06

### Added
- **FastAPI REST API server** (`beyondbench serve`) with endpoints for tasks, evaluation, jobs, and results
- **`beyondbench init`** command for interactive config file creation
- **`beyondbench info <task>`** command for viewing task details
- **`beyondbench results list/show/compare`** commands for results viewer
- **Config file validation** with JSON schema
- **Example configs**: `default.yaml`, `openai_example.yaml`, `full_evaluation.yaml`
- **Comprehensive test suite** (239 tests) covering data generation and evaluation for all task suites
- **Real data generation + evaluation tests** for all task suites
- **Ruff linting configuration** for code quality
- **PEP 561 `py.typed` marker** for type checking support
- **pytest-xdist** parallel test support
- **GitHub Actions CI/CD**: `test.yml` (Python 3.10-3.13 matrix), `lint.yml`, `publish.yml` (PyPI OIDC)
- **Pre-commit configuration** (ruff + hooks)
- **`CONTRIBUTING.md`** and **`SECURITY.md`** documentation
- **Issue templates** (bug report, feature request) and **PR template**
- **Dependabot configuration** for automated dependency updates
- **`__main__.py`** for `python -m beyondbench` support
- **`configs/default.yaml`** for default evaluation configuration
- **Exponential backoff** for API rate limiting
- **Improved error messages** with Rich formatting

### Changed
- **License** changed from MIT to Apache-2.0
- **Simplified ModelHandler** GPT-5 logic
- **Single version source of truth** via `importlib.metadata`
- **Wizard** now wires to actual evaluation pipeline
- **Minimum Python version** raised from 3.8 to 3.10

### Fixed
- **Token extraction** for all API backends
- **EvaluationEngine seed handling** for reproducible evaluations
- **Fixed license** metadata to Apache-2.0

### Removed
- **Redundant `setup.py`** (replaced by `pyproject.toml`)
- **Dead code**: `simple_cli.py`, `PlaceholderTask`

---

## [0.0.2] - 2026-02-25

### Critical Bug Fixes & Stability

This release is a significant stability improvement over v0.0.1, resolving 41 bugs identified across all task suites. The framework is now substantially more reliable for production evaluation runs.

### Fixed

- **Easy Suite (29 tasks)**: Fixed evaluation logic, parsing, and result formatting across all easy tasks — zero errors on 0.5B–14B models
- **Medium Suite (5 tasks)**: Resolved sequence generation and answer extraction issues
- **Hard Suite (10 tasks)**: Fixed Tower of Hanoi (N=3–7), cryptarithmetic, matrix chain, logic grid puzzles, and more
- **vLLM backend**: Corrected batch processing pipeline and token handling
- **Parsing utilities**: Improved robustness of `\boxed{}`, code block, and structured output extraction
- **CLI**: Fixed `--model-id`, `--datapoints`, `--folds` option handling and max-tokens fallback (32768 → 8192)
- **EvaluationEngine**: Fixed `inspect.signature`-based parameter dispatch for hard task constructors
- **ModelHandler**: Improved multi-backend stability (vLLM, Transformers, OpenAI, Gemini, Anthropic)

---

## [0.0.1] - 2026-02-25

### First Public Release

BeyondBench v0.0.1 marks the first public release of our contamination-resistant evaluation framework for Large Language Models. This release accompanies our paper accepted at ICLR 2026.

### Highlights

- **44 Algorithmic Tasks** spanning easy, medium, and hard difficulty levels
- **117 Task Variations** for fine-grained capability evaluation
- **101+ Models Evaluated** across major model families
- **Problem Space >10^15** unique instances per task for contamination resistance

---

### Added

#### Core Framework
- **EvaluationEngine**: Comprehensive evaluation orchestration with progress tracking
- **TaskRegistry**: Centralized task management with suite-based organization
- **ModelHandler**: Unified interface for multiple inference backends
- **BaseTask**: Extensible base class for custom task development

#### Task Suites

**Easy Suite (29 Tasks)**
- Arithmetic: `sum`, `multiplication`, `subtraction`, `division`, `absolute_difference`
- Statistics: `mean`, `median`, `mode`
- Counting: `odd_count`, `even_count`, `count_negative`, `count_unique`, `count_greater_than_previous`, `count_palindromic`, `count_perfect_squares`, `count_multiples`, `local_maxima_count`
- Extrema: `find_maximum`, `find_minimum`, `second_maximum`, `range`, `index_of_maximum`, `max_adjacent_difference`, `sum_of_max_indices`
- Comparison: `comparison`
- Sequences: `sorting`, `longest_increasing_subsequence`, `alternating_sum`, `sum_of_digits`

**Medium Suite (5 Tasks, 49 Variations)**
- `fibonacci_sequence`: Standard, Tribonacci, Lucas, modified recursive (6 variations)
- `algebraic_sequence`: Polynomial, arithmetic, quadratic patterns (10 variations)
- `geometric_sequence`: Exponential, compound growth, factorial (10 variations)
- `prime_sequence`: Prime gaps, twin primes, Sophie Germain (11 variations)
- `complex_pattern`: Interleaved, conditional, multi-rule (12 variations)

**Hard Suite (10 Tasks, 68 Variations)**
- `tower_of_hanoi`: Classic recursive puzzle (6 variations)
- `n_queens`: NP-complete board placement (4 variations)
- `graph_coloring`: Chromatic number computation (10 variations)
- `boolean_sat`: Satisfiability solving (5 variations)
- `sudoku`: Constraint satisfaction puzzles (8 variations)
- `cryptarithmetic`: Digit-letter puzzles (12 variations)
- `matrix_chain`: Dynamic programming optimization (5 variations)
- `modular_systems`: Chinese Remainder Theorem (5 variations)
- `constraint_optimization`: Operations research problems (5 variations)
- `logic_grid_puzzles`: Deductive reasoning (8 variations)

#### Model Backends
- **vLLM**: High-performance batch inference with tensor parallelism
- **Transformers**: HuggingFace integration for CPU/GPU inference
- **OpenAI API**: GPT-4o, GPT-4o-mini, GPT-5, GPT-5-mini support with reasoning effort
- **Gemini API**: Gemini 2.5 Pro/Flash with thinking budget configuration
- **Anthropic API**: Claude Sonnet 4, Claude Opus 4 support

#### CLI Features
- **Interactive Wizard**: Beautiful Rich-based UI with spinners and progress bars
- **Command-line Interface**: Full-featured `beyondbench` command
  - `evaluate`: Run model evaluations
  - `list-tasks`: Display available tasks and variations
  - `run-config`: Execute from YAML configuration files
  - `wizard`: Launch interactive setup wizard
- **Unified Bash Script**: `scripts/run_evaluation.sh` for all backends

#### Parsing System
- **Robust Answer Extraction**:
  - LaTeX `\boxed{answer}` parsing
  - Code block output extraction
  - Natural language answer detection
  - JSON/structured output parsing
  - Multi-format fallback chains
- **Type-Specific Parsers**:
  - Number parsing with scientific notation
  - List/array parsing with multiple formats
  - Boolean parsing with various representations
  - Grid/matrix parsing for puzzle solutions

#### Documentation
- Professional README with badges and comprehensive guides
- SVG logo and banner assets
- Installation guides for all backends
- API documentation with examples
- Configuration file templates
- Interactive React leaderboard website

#### Developer Tools
- `setup_beyondbench.sh`: Interactive setup script
- `pyproject.toml`: Modern Python packaging configuration
- Pre-commit hooks configuration
- Comprehensive `.gitignore`

---

### Technical Specifications

#### Requirements
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+

#### Optional Dependencies
```bash
pip install beyondbench[all-apis]  # All API clients
pip install beyondbench[vllm]      # vLLM support
pip install beyondbench[full]      # Everything
```

#### Supported Platforms
- Linux (recommended)
- macOS
- Windows (limited vLLM support)

---

### Performance Notes

- vLLM batch processing provides 3-10x speedup over sequential inference
- Recommended: Use vLLM for local models, APIs for cloud models
- Memory requirements vary by model size (8GB+ VRAM for 7B models)

---

## [0.1.0] - 2024-12-19 (Internal)

### Initial Development Release

- Core task framework implementation
- Basic evaluation engine
- Initial set of easy tasks
- Preliminary documentation

---

## Links

- **Repository**: [github.com/ctrl-gaurav/BeyondBench](https://github.com/ctrl-gaurav/BeyondBench)
- **Documentation**: [github.com/ctrl-gaurav/BeyondBench#readme](https://github.com/ctrl-gaurav/BeyondBench#readme)
- **Issues**: [github.com/ctrl-gaurav/BeyondBench/issues](https://github.com/ctrl-gaurav/BeyondBench/issues)
- **PyPI**: [pypi.org/project/beyondbench/](https://pypi.org/project/beyondbench/)

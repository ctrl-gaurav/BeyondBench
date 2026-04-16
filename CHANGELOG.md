# Changelog

All notable changes to BeyondBench (beyondbench) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-15

A major release delivering 79 reasoning tasks, multi-GPU parallel evaluation, a plugin system, Gradio dashboard, comprehensive documentation, and production-grade infrastructure.

### Added

#### Parser & Evaluation Core
- **Universal Parser** (`parsers/core.py`): unified parsing engine with strategy pipeline and confidence scoring, replacing 14 individual parser files
- **10 Pluggable Parse Strategies**: boxed, explicit_statement, code_block, latex_math, json, list, grid, comparison, sequence, fallback — each with configurable priority
- **Per-Model Parsing Adapters** (`parsers/model_adapters.py`): auto-detection and normalization for Qwen, Llama, Phi, Mistral, Gemma, and API model families
- **Common Parser Library** (`parsers/common.py`): shared extraction functions for boxed formats (nested braces, double-dollar LaTeX, malformed), explicit statements (multilingual, inverted patterns), and number cleaning (commas, fractions, scientific notation)
- **Parser Legacy Fallback**: `--parser=unified|legacy` CLI flag with automatic legacy fallback when unified parser returns low confidence

#### Multi-GPU Parallel Evaluation Engine
- **GPU-Aware Task Distribution** (`core/gpu_scheduler.py`): GPU discovery via nvidia-smi, VRAM estimation, occupancy detection, `CUDA_VISIBLE_DEVICES` awareness
- **Parallel Evaluation Engine** (`core/parallel_engine.py`): task_parallel, data_parallel, and model_parallel strategies using multiprocessing with spawn context
- **Result Aggregation** (`core/result_aggregator.py`): merge multi-GPU results, handle partial failures, data_parallel shard merging, unified final_results.json format
- **Real-Time Progress Tracking**: Rich Live table with per-GPU status (task, progress, accuracy, elapsed, utilization, memory)
- **CLI Flags**: `--parallel`, `--gpus auto|0,1,2`, `--strategy task_parallel|data_parallel`

#### Model Behavior Profiling
- **Model Profiler** (`core/model_profiler.py`): calibration-based output pattern analysis with caching to `~/.beyondbench/profiles/`
- **14 Pre-Built Profiles**: Qwen (4 sizes), Llama (2), Phi, Mistral, Gemma (2), GPT-4o (2), Gemini, Claude
- **Adaptive Parsing**: profile-driven strategy reordering in UnifiedParser for higher parse success rates
- **Model Comparison Workflow** (`core/model_comparison.py`): simultaneous multi-model evaluation with `beyondbench compare` CLI command

#### Live Observability Dashboard
- **Gradio Dashboard** (`dashboard/`): 5-tab interface — Overview (progress, accuracy gauges), Per-Task Results (sortable table), GPU Monitor (memory, utilization, temperature timeline), Token Analytics (input/output charts, cost), Live Log (streaming, filtering, search, download)
- **Data Bridge** (`dashboard/data_bridge.py`): event bridge between evaluation engine and dashboard with gr.Timer polling every 2s
- **CLI Integration**: `beyondbench dashboard` command and `--dashboard` flag during evaluation
- **Compare Mode**: `--compare` for side-by-side result visualization

#### Token & Cost Analytics
- **Token Counter** (`utils/token_counter.py`): model_tokenizer → tiktoken → estimate fallback chain, TokenUsage/TokenAnalytics dataclasses, per-request logging, tokens/sec throughput
- **Cost Tracker** (`utils/cost_tracker.py`): 2025 pricing tables for OpenAI (gpt-4o/4o-mini/o1/o3/o4/gpt-5), Anthropic (claude-3-5/sonnet-4/opus-4), Gemini (2.5 pro/flash/flash-lite + 1.5), per-task cost breakdown, pre-evaluation estimates, custom pricing support
- **Request Logger** (`utils/request_logger.py`): JSONL append-only writer with context manager, load/replay helpers, `--log-requests` flag

#### Result Visualization & Reporting
- **Report Generator** (`utils/report_generator.py`): HTML (embedded Plotly charts + dark CSS), Markdown (GitHub-flavored tables), LaTeX (booktabs), PDF via WeasyPrint (optional, gracefully skipped)
- **Visualizer** (`utils/visualizer.py`): 8 chart types — accuracy_by_task, accuracy_by_suite, accuracy_by_difficulty (heatmap), token_distribution, latency_distribution, model_comparison, scaling_curve, radar_chart — with Plotly → Matplotlib fallback
- **CLI Command**: `beyondbench report --format html|markdown|pdf|latex|all --charts-dir --compare`

#### Configuration System Hardening
- **JSON Schema Validation**: full config schema with enum/range/type constraints, required fields, additionalProperties:false
- **Config Presets**: quick_test, full_evaluation, paper_quality, debug — ready-to-use YAML configurations
- **Environment Variable Support**: 14 `BEYONDBENCH_*` env vars with type casting and deep-merge, `.env` file auto-loading via python-dotenv
- **Model Validation**: HuggingFace existence probe, GPU free-memory precheck, backend-dependency check, API-key validation, VRAM estimation
- **Config Loader**: `ConfigLoader.load()` with `ConfigProxy` hybrid (attribute + dict) access, root-only shortcut aliases

#### CLI Polish & Developer Experience
- **Rich Console Output**: tables, panels, banners throughout CLI using Rich library
- **Interactive Wizard Upgrade**: step-by-step model → suite → params → confirm → run with Rich Prompt/Panel/Live/Progress
- **New Commands**: `info`, `doctor`, `benchmark`, `export`, `profile-model`, `install-completion`
- **Global Flags**: `--verbose/-v`, `--quiet/-q`, `--json` with mutual-exclusion check and CLIContext propagation
- **GPU Utilization Display**: `GPUUtilizationSampler` background thread, Rich live table columns, tqdm postfix (`G0:73%/18.1GB`)
- **Shell Completion**: `beyondbench install-completion {bash,zsh,fish}`

#### FastAPI Server v2 — Production Gateway
- **WebSocket Endpoints**: `/ws/jobs/{job_id}` per-job + `/ws/progress` all-jobs stream with keepalive pings and log replay
- **Authentication & Rate Limiting**: Bearer token via `BEYONDBENCH_API_KEY`, sliding-window rate limiter via `BEYONDBENCH_RATE_LIMIT_RPM`
- **Batch Evaluation API**: `POST /evaluate/batch` (max 20 configs), `GET /jobs` with status filter, `DELETE /jobs/{job_id}` cancel/remove
- **Result Comparison API**: `POST /compare` + `POST /evaluate/compare` — load from filesystem or in-memory completed jobs
- **Prometheus Metrics**: `GET /metrics` — beyondbench_up, uptime_seconds, gpu_count, jobs_total by status
- **Full OpenAPI Documentation**: Swagger UI at `/docs`, ReDoc at `/redoc`, all models have `json_schema_extra` examples

#### New Tasks
- **15 New Easy Tasks** (44 total): weighted_sum, running_average, parity_check, cumulative_sum, reverse_list, rotate_list, interleave_lists, set_intersection, set_difference, moving_average, element_frequency, second_minimum, variance, standard_deviation, dot_product
- **10 New Medium Tasks** (15 total): arithmetic_progression, harmonic_sequence, collatz_sequence, polynomial_evaluation, matrix_operations, number_base_conversion, logical_operations, pattern_completion, gcd_lcm, combinatorics
- **10 New Hard Tasks** (20 total): shortest_path (Dijkstra), knapsack (0/1 DP), traveling_salesman (brute-force), longest_common_subsequence (DP), minimax_game (game tree), regex_matching (fullmatch), topological_sort (Kahn's), interval_scheduling (greedy), coin_change (DP), edit_distance (Levenshtein)

#### Prompt Engineering & Optimization
- **Prompt Template System** (`prompts/`): PromptTemplate, PromptLibrary with 79 tasks × 4 styles = 316 templates
- **4 Prompt Styles**: concise, detailed, few_shot, cot — selectable via `--prompt-style`
- **Dynamic Few-Shot Generation**: FewShotGenerator produces fresh examples at runtime with seed variation for contamination resistance
- **Model-Family Hints**: Qwen/Llama/Phi/Mistral/Gemma-specific `\boxed{}` instructions appended when model_id is known

#### Contamination Resistance Hardening
- **Instance Fingerprinting** (`core/fingerprint.py`): SHA-256 per instance (task + data + seed), within-dataset uniqueness verification
- **Problem Rephrasing** (`core/rephraser.py`): 14 operation types × 3–5 phrasings each, seed-controlled random selection
- **Noise Injection** (`core/noise_injector.py`): low (polite prefix), medium (context sentences), high (distractor numbers) — `--contamination-resistance low|medium|high`
- **Audit Script**: `audit_contamination.py` verified all 79 tasks produce distinct datasets across seeds

#### Evaluation Reproducibility & Seed Management
- **Seed Propagation**: `seed_manager.py` propagates to random, numpy, torch, torch.cuda with deterministic flags
- **Evaluation Fingerprint** (`eval_fingerprint.py`): SHA-256 of model_id + config + seed + tasks + version, stored in final_results.json
- **Result Versioning**: environment dict with python, platform, beyondbench, torch, transformers, vllm, numpy versions, seed, timestamp (UTC ISO), git_hash, gpu_count, gpus (name + memory)

#### Gradio Example Apps
- **Live Eval Dashboard** (`examples/gradio_live_eval.py`): model/suite/GPU selection, real-time results, log stream, CSV/JSON download — port 7860
- **Model Arena** (`examples/gradio_model_arena.py`): side-by-side comparison, parallel response fetching, human voting (A/B/tie), leaderboard — port 7861
- **Task Explorer** (`examples/gradio_task_explorer.py`): browse 79 tasks by suite, search/filter, sample instances with ground truth, try custom responses — port 7862
- **Parser Debugger** (`examples/gradio_parser_debugger.py`): paste model response, see per-strategy parse results with confidence — port 7863
- **GPU Monitor** (`examples/gradio_gpu_monitor.py`): real-time GPU stats with rolling history and alert thresholds — port 7864

#### Regression Testing & Baseline Management
- **Baseline Manager** (`eval/baseline.py`): save/load/compare/check_regression with JSON storage in `tests/baselines/`
- **CLI Commands**: `beyondbench baseline save|compare|list`, `--fail-on-regression` exits with code 1 on threshold breach (>10% accuracy drop)

#### CI/CD Pipeline & Quality Gates
- **GitHub Actions**: Python 3.10–3.14 matrix CI, unit tests with coverage (`--cov-fail-under=45`), GPU integration tests (weekly), TestPyPI + PyPI publish pipeline
- **Pre-Commit Hooks**: ruff lint + format, isort, trailing-whitespace, check-yaml/json, large file check, mypy, pytest on commit

#### Documentation
- **Getting Started Guide** (`docs/getting_started.md`): installation, first evaluation walkthrough, viewing results
- **User Guide** (`docs/user_guide.md`): all CLI commands, config files, model backends, multi-GPU, prompt styles, contamination resistance
- **Task Reference** (`docs/task_reference.md`): all 79+ tasks with descriptions, I/O format, difficulty, organized by category
- **API Reference** (`docs/api_reference.md`): EvaluationEngine, TaskRegistry, ModelHandler, BaseTask, ParallelEvaluationEngine, ResultAggregator
- **Contributing Guide** (`docs/contributing.md`): dev setup, code style, testing, adding tasks/parsers/backends, PR workflow
- **Jupyter Notebooks** (`docs/examples/`): 4 notebooks — basic evaluation, custom tasks, model comparison, result analysis

#### Plugin System & Custom Task SDK
- **Plugin Manager** (`plugins/`): entry_points-based discovery (`beyondbench.tasks` group) + local directory discovery, plugin validation
- **Task Scaffolding**: `beyondbench create-task <name>` generates complete plugin project (task.py, parser.py, test_task.py, pyproject.toml with entry-point)
- **Task Metadata System**: TaskMetadata dataclass with name/description/difficulty/category/author/version/tags, 11 category taxonomy, `beyondbench list-tasks --detailed`

#### Performance Optimization & Caching
- **Response Cache** (`core/cache.py`): disk-based LRU, SHA-256 key, 10GB default, thread-safe, `beyondbench cache stats|clear`, `--no-cache` flag
- **vLLM Batch Auto-Tuning**: `_estimate_vllm_batch_size()` auto-tunes max_num_seqs (64–512) from GPU free VRAM, `enable_prefix_caching=True`
- **Quantization** (`--quantization 4bit|8bit|gptq|awq`): bitsandbytes, GPTQConfig, AwqConfig with ImportError guards
- **torch.compile** (`--torch-compile`): faster transformers inference via compilation
- **Model Warm-Up**: 3 dummy prompts (max_tokens=32), `--no-warmup` to skip, latency logged separately
- **GPU Memory Cleanup**: `ModelHandler.cleanup()` deletes model/tokenizer, gc.collect + torch.cuda.empty_cache, wired into parallel engine worker finally blocks

#### Website & Release Polish
- **React Website Update**: task explorer page showing all 79 tasks with search/filter/expand, results showcase with interactive Recharts (bar, radar, scatter), documentation page with syntax-highlighted code blocks
- **PyPI Package**: version 0.2.0, all classifiers and optional dependencies documented, README_PYPI.md description
- **Comprehensive CHANGELOG**: organized by category with migration guide

### Changed
- **Task Count**: 44 tasks → 79 tasks (44 easy + 15 medium + 20 hard)
- **Variation Count**: 117 → 138 total task variations
- **Default vLLM Settings**: prefix caching enabled by default, auto-tuned max_num_seqs based on available VRAM
- **Test Suite**: expanded from 239 to 1800+ tests (unit + integration + e2e) across all components
- **Parser Architecture**: monolithic per-task parsers replaced with pluggable strategy pipeline + confidence scoring
- **Version**: 0.1.0 → 0.2.0 (pyproject.toml, `__init__.py` fallback)
- **Contact Email**: updated to gks@vt.edu across all files

### Fixed
- **Parser Robustness**: universal parser handles nested `\boxed{}`, multilingual answer patterns, scientific notation, fractions, comma-separated thousands, inverted answer patterns — 435/435 format variation tests pass
- **BaseTask Error Handling**: `_run_fold_batch()` now retries individual failed prompts, `_get_ground_truth()` handles all task types including medium/hard-specific keys
- **Token Counting**: fallback chain (model_tokenizer → tiktoken → estimate) with clear logging and accuracy tracking
- **ModelHandler**: chat templates for transformers backend, streaming support for API backends, updated 2025 pricing
- **CLI Auto-Detection**: auto-detect API provider from model_id, auto-detect API keys from environment
- **GPU Memory**: cleanup verified — GPUs return to baseline VRAM after parallel evaluation completes
- **vLLM Batch Processing**: prompt length sorting for optimal throughput, dtype → torch_dtype fallback bug fixed
- **Contamination Resistance**: fixed NQueensTask (randomized reference solution), TowerHanoiTask (randomized peg assignment), GraphColoringTask (seed-dependent generation), LogicGridPuzzlesTask (shuffled entity assignments), FibonacciSequenceTask (randomized starting values)
- **Parallel Engine**: queue deadlock fix with `_slim_result`, occupancy detection prevents vLLM conflicts on shared machines, `CUDA_VISIBLE_DEVICES` awareness
- **Gradio 6.0 Compatibility**: theme/css moved from Blocks() to launch() with version-adaptive code

### Removed
- **Dead Contact Email**: removed `contact@beyondbench.org` (replaced with `gks@vt.edu`)

### Migration from v0.1.0
- All existing CLI commands (`evaluate`, `list-tasks`, `run-config`, `wizard`, `serve`, `init`, `info`, `results`) remain fully compatible
- New `--parallel` flag for multi-GPU evaluation (off by default — single-GPU behavior unchanged)
- Response cache enabled by default for deterministic runs (temperature=0 or seeded); use `--no-cache` to disable
- New optional dependency groups: `pip install beyondbench[dashboard]` for Gradio, `pip install beyondbench[viz]` for charts, `pip install beyondbench[full]` for everything
- `--parser unified` is the new default; `--parser legacy` falls back to v0.1.0 parser behavior
- New `BEYONDBENCH_*` environment variables available for config overrides (see `docs/user_guide.md`)
- Python 3.10+ required (unchanged from v0.1.0)

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

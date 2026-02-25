# BeyondBench Usage Guide

Complete reference for using beyondbench (BeyondBench) evaluation framework.

## Table of Contents

1. [Installation](#installation)
2. [CLI Commands](#cli-commands)
3. [Backend Configuration](#backend-configuration)
4. [Python API](#python-api)
5. [Task Suites](#task-suites)
6. [Advanced Options](#advanced-options)
7. [Examples](#examples)

---

## Installation

### Basic Installation

```bash
pip install beyondbench
```

### With Specific Backends

```bash
# OpenAI support
pip install beyondbench[openai]

# Google Gemini support
pip install beyondbench[gemini]

# Anthropic Claude support
pip install beyondbench[anthropic]

# All API clients
pip install beyondbench[all-apis]

# vLLM support (requires CUDA)
pip install beyondbench[vllm]

# Transformers (HuggingFace) support
pip install beyondbench[transformers]

# Full installation (everything)
pip install beyondbench[full]
```

### From Source

```bash
git clone https://github.com/ctrl-gaurav/BeyondBench.git
cd BeyondBench
pip install -e .
```

---

## CLI Commands

### Interactive Wizard

Launch the interactive setup wizard:

```bash
beyondbench
```

The wizard guides you through:
1. Selecting backend (API/Local)
2. Choosing model
3. Configuring API keys
4. Selecting task suite
5. Setting evaluation parameters

### Evaluate Command

Run model evaluation:

```bash
beyondbench evaluate [OPTIONS]
```

#### Required Options

| Option | Description |
|--------|-------------|
| `--model-id MODEL` | Model identifier (e.g., `gpt-4o`, `meta-llama/Llama-3-8B-Instruct`) |

#### Backend Selection

| Option | Description |
|--------|-------------|
| `--backend BACKEND` | Local backend: `vllm` (default, fast) or `transformers` |
| `--api-provider PROVIDER` | API backend: `openai`, `gemini`, or `anthropic` |
| `--api-key KEY` | API key (or set via environment variable) |

#### Task Selection

| Option | Description |
|--------|-------------|
| `--suite SUITE` | Task suite: `easy`, `medium`, `hard`, or `all` (default: `all`) |
| `--tasks TASKS` | Specific tasks (comma-separated): `--tasks sum,sorting,median` |

#### Evaluation Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--datapoints N` | 50 | Number of data points per task |
| `--temperature T` | 0.1 | Sampling temperature |
| `--top-p P` | 0.9 | Top-p (nucleus) sampling |
| `--max-tokens N` | 32768 | Maximum tokens to generate |
| `--seed SEED` | 42 | Random seed for reproducibility |
| `--num-folds N` | 1 | Number of evaluation folds |

#### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir DIR` | `./results` | Output directory for results |
| `--store-details` | False | Store detailed per-example results |

#### Model-Specific Options

| Option | Description |
|--------|-------------|
| `--reasoning-effort EFFORT` | OpenAI GPT-5 reasoning effort: `minimal`, `low`, `medium`, `high` |
| `--thinking-budget N` | Gemini thinking budget: integer, 0 to disable, -1 for dynamic |

#### Hardware Options (Local Models)

| Option | Default | Description |
|--------|---------|-------------|
| `--tensor-parallel-size N` | 1 | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization F` | 0.96 | GPU memory utilization (0.0-1.0) |
| `--trust-remote-code` | False | Trust remote code from HuggingFace |

### List Tasks Command

```bash
beyondbench list-tasks [--suite SUITE]
```

Options:
- `--suite`: Filter by suite (`easy`, `medium`, `hard`, `all`)

---

## Backend Configuration

### OpenAI

```bash
# Set API key via environment
export OPENAI_API_KEY="sk-..."

# Or pass directly
beyondbench evaluate \
    --model-id gpt-4o \
    --api-provider openai \
    --api-key "sk-..." \
    --suite easy
```

Supported models:
- `gpt-4o` - GPT-4 Optimized
- `gpt-4o-mini` - Smaller GPT-4 variant
- `gpt-5` - Latest GPT-5 (with reasoning)
- `gpt-5-mini` - Smaller GPT-5
- `gpt-5-nano` - Smallest GPT-5

For GPT-5 models with reasoning:
```bash
beyondbench evaluate \
    --model-id gpt-5 \
    --api-provider openai \
    --reasoning-effort high \
    --suite hard
```

### Google Gemini

```bash
export GEMINI_API_KEY="..."

beyondbench evaluate \
    --model-id gemini-2.5-pro \
    --api-provider gemini \
    --suite medium
```

With thinking configuration:
```bash
beyondbench evaluate \
    --model-id gemini-2.5-pro \
    --api-provider gemini \
    --thinking-budget 16384 \
    --suite hard
```

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

beyondbench evaluate \
    --model-id claude-sonnet-4-20250514 \
    --api-provider anthropic \
    --suite all
```

Supported models:
- `claude-sonnet-4-20250514` - Claude Sonnet 4
- `claude-opus-4-20250514` - Claude Opus 4

### vLLM (Local, Recommended)

vLLM provides fast batch inference with GPU parallelism:

```bash
beyondbench evaluate \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --backend vllm \
    --suite all
```

With multi-GPU:
```bash
beyondbench evaluate \
    --model-id meta-llama/Llama-3.3-70B-Instruct \
    --backend vllm \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --suite all
```

### Transformers (Local, HuggingFace)

For CPU or single-GPU inference:

```bash
beyondbench evaluate \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --backend transformers \
    --suite easy
```

---

## Python API

### Basic Usage

```python
from beyondbench import EvaluationEngine, ModelHandler, TaskRegistry

# Initialize model handler
model = ModelHandler(
    model_id="gpt-4o",
    api_provider="openai",
    api_key="your-api-key"
)

# Run evaluation
engine = EvaluationEngine(
    model_handler=model,
    output_dir="./results",
    datapoints=100,
    temperature=0.1,
    max_tokens=32768
)

results = engine.run_evaluation(suite="easy")
print(f"Average Accuracy: {results['summary']['avg_accuracy']:.2%}")
```

### With Local Model (vLLM)

```python
from beyondbench import ModelHandler, EvaluationEngine

# vLLM backend (fast, batched inference)
model = ModelHandler(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    backend="vllm",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.96
)

engine = EvaluationEngine(
    model_handler=model,
    output_dir="./results"
)

results = engine.run_evaluation(suite="all")
```

### With Local Model (Transformers)

```python
from beyondbench import ModelHandler, EvaluationEngine

# Transformers backend
model = ModelHandler(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    backend="transformers",
    trust_remote_code=True
)

engine = EvaluationEngine(
    model_handler=model,
    output_dir="./results"
)

results = engine.run_evaluation(suite="easy")
```

### Run Specific Tasks

```python
# Get specific tasks
registry = TaskRegistry()
tasks = registry.get_tasks_for_suite("easy")

# Run only selected tasks
results = engine.run_evaluation(
    tasks=["sum", "sorting", "median"]
)
```

### Access Results

```python
# Overall metrics
print(f"Total Tasks: {results['summary']['total_tasks']}")
print(f"Average Accuracy: {results['summary']['avg_accuracy']:.2%}")
print(f"Total Tokens: {results['summary']['total_tokens']}")

# Per-task metrics
for task_name, metrics in results['tasks'].items():
    print(f"{task_name}: {metrics['accuracy']:.2%}")
```

---

## Task Suites

### Easy Suite (29 Tasks)

Fundamental operations with clear numerical answers:

```bash
beyondbench evaluate --model-id gpt-4o --api-provider openai --suite easy
```

Tasks:
- **Arithmetic**: sum, multiplication, subtraction, division, absolute_difference
- **Statistics**: mean, median, mode, variance, std_dev
- **Counting**: odd_count, even_count, count_negative, count_positive, count_unique, count_prime
- **Extrema**: find_maximum, find_minimum, second_maximum, second_minimum, range
- **Ordering**: sorting, index_of_maximum, index_of_minimum
- **Difference**: max_adjacent_difference
- **Comparison**: comparison

### Medium Suite (5 Tasks, 49 Variations)

Sequence pattern recognition:

```bash
beyondbench evaluate --model-id gpt-4o --api-provider openai --suite medium
```

Tasks:
- **fibonacci_sequence** (6 variations): Tribonacci, Lucas, Modified recursive
- **algebraic_sequence** (10 variations): Polynomial, arithmetic, quadratic
- **geometric_sequence** (10 variations): Exponential, compound, factorial
- **prime_sequence** (11 variations): Prime gaps, twin primes, Sophie Germain
- **complex_pattern** (12 variations): Interleaved, conditional, multi-rule

### Hard Suite (10 Tasks, 68 Variations)

NP-complete and constraint satisfaction problems:

```bash
beyondbench evaluate --model-id gpt-4o --api-provider openai --suite hard
```

Tasks:
- **tower_of_hanoi** (6 variations): Classic, bidirectional, cyclic
- **n_queens** (4 variations): Standard, modified constraints
- **graph_coloring** (10 variations): Various graph types
- **boolean_sat** (5 variations): 2-SAT, 3-SAT, Horn clauses
- **sudoku** (8 variations): Standard, diagonal, irregular
- **cryptarithmetic** (12 variations): Various equation types
- **matrix_chain** (5 variations): Multiplication ordering
- **modular_systems** (5 variations): Chinese remainder theorem
- **constraint_optimization** (5 variations): Knapsack, scheduling
- **logic_grid_puzzles** (8 variations): Einstein puzzles, zebra

---

## Advanced Options

### Reproducibility

```bash
beyondbench evaluate \
    --model-id gpt-4o \
    --api-provider openai \
    --suite all \
    --seed 42 \
    --temperature 0.0
```

### Detailed Results

Store per-example results for analysis:

```bash
beyondbench evaluate \
    --model-id gpt-4o \
    --api-provider openai \
    --suite easy \
    --store-details \
    --output-dir ./detailed_results
```

### Multi-Fold Evaluation

Run multiple evaluation folds:

```bash
beyondbench evaluate \
    --model-id gpt-4o \
    --api-provider openai \
    --suite easy \
    --num-folds 3
```

### Custom Data Points

```bash
beyondbench evaluate \
    --model-id gpt-4o \
    --api-provider openai \
    --suite hard \
    --datapoints 200
```

---

## Examples

### Complete Evaluation of GPT-4o

```bash
export OPENAI_API_KEY="sk-..."

beyondbench evaluate \
    --model-id gpt-4o \
    --api-provider openai \
    --suite all \
    --datapoints 100 \
    --temperature 0.1 \
    --max-tokens 32768 \
    --seed 42 \
    --store-details \
    --output-dir ./results/gpt-4o
```

### Evaluate Local Model with vLLM

```bash
beyondbench evaluate \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --suite all \
    --datapoints 50 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --output-dir ./results/qwen-7b
```

### Quick Test on Easy Tasks

```bash
beyondbench evaluate \
    --model-id gpt-4o-mini \
    --api-provider openai \
    --suite easy \
    --datapoints 10 \
    --tasks sum,sorting,median
```

### Compare Multiple Models

```bash
# Script to compare models
for model in "gpt-4o" "gpt-4o-mini" "claude-sonnet-4-20250514"; do
    beyondbench evaluate \
        --model-id $model \
        --api-provider ${model%%/*} \
        --suite easy \
        --datapoints 50 \
        --output-dir ./results/$model
done
```

### Gemini with Thinking Budget

```bash
export GEMINI_API_KEY="..."

beyondbench evaluate \
    --model-id gemini-2.5-pro \
    --api-provider gemini \
    --thinking-budget 16384 \
    --suite hard \
    --datapoints 50 \
    --output-dir ./results/gemini-thinking
```

### GPT-5 with Reasoning

```bash
export OPENAI_API_KEY="sk-..."

beyondbench evaluate \
    --model-id gpt-5 \
    --api-provider openai \
    --reasoning-effort high \
    --suite hard \
    --datapoints 50 \
    --output-dir ./results/gpt5-high-reasoning
```

---

## Output Format

Results are saved in JSON format:

```json
{
  "summary": {
    "model_id": "gpt-4o",
    "suite": "easy",
    "total_tasks": 29,
    "avg_accuracy": 0.923,
    "total_tokens": 150432,
    "evaluation_time": "12m 34s"
  },
  "tasks": {
    "sum": {
      "accuracy": 0.98,
      "instruction_followed": 0.99,
      "avg_tokens": 45.2
    },
    "sorting": {
      "accuracy": 0.95,
      "instruction_followed": 0.97,
      "avg_tokens": 123.8
    }
  },
  "metadata": {
    "timestamp": "2025-02-05T10:30:00Z",
    "seed": 42,
    "datapoints": 100
  }
}
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce GPU memory utilization:
```bash
--gpu-memory-utilization 0.7
```

Or use smaller batch sizes by reducing datapoints:
```bash
--datapoints 20
```

### API Rate Limits

The framework automatically handles rate limiting with exponential backoff. For heavy usage, consider:
- Using lower `--datapoints`
- Running tasks sequentially
- Using multiple API keys

### Model Not Found

For HuggingFace models, ensure the model ID is correct:
```bash
# Correct format
--model-id Qwen/Qwen2.5-3B-Instruct

# Not correct
--model-id qwen2.5-3b
```

### Trust Remote Code

Some models require trusting remote code:
```bash
--trust-remote-code
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `CUDA_VISIBLE_DEVICES` | GPU selection for local models |
| `HF_TOKEN` | HuggingFace token for gated models |

---

## Support

- GitHub Issues: https://github.com/ctrl-gaurav/BeyondBench/issues
- Documentation: https://github.com/ctrl-gaurav/BeyondBench/wiki


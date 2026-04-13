# BeyondBench User Guide

Complete reference for all CLI commands, configuration options, model backends, and advanced usage.

## Table of Contents

1. [CLI Commands](#cli-commands)
2. [Configuration Files](#configuration-files)
3. [Model Backends](#model-backends)
4. [Understanding Results](#understanding-results)
5. [Multi-GPU Parallel Evaluation](#multi-gpu-parallel-evaluation)
6. [Prompt Styles](#prompt-styles)
7. [Contamination Resistance](#contamination-resistance)
8. [Advanced Options](#advanced-options)

---

## CLI Commands

BeyondBench installs a single entry-point: `beyondbench`. Run `beyondbench --help` at any time.

### Global Flags

These apply to all subcommands:

| Flag | Description |
|------|-------------|
| `--version` | Print version and exit |
| `--verbose`, `-v` | Enable DEBUG log level |
| `--quiet`, `-q` | Suppress all non-error output |
| `--json` | Emit machine-readable JSON (supported commands only) |

---

### `beyondbench evaluate`

The primary command. Evaluates one or more models on BeyondBench tasks.

```bash
beyondbench evaluate --model-id MODEL_ID [OPTIONS]
```

#### Required

| Option | Description |
|--------|-------------|
| `--model-id MODEL` | HuggingFace model path or API model name |

#### Task Selection

| Option | Default | Description |
|--------|---------|-------------|
| `--suite SUITE` | `all` | Task suite: `easy`, `medium`, `hard`, or `all` |
| `--tasks TASK,...` | (all in suite) | Comma-separated specific task names |

#### Backend Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--backend BACKEND` | auto | Force backend: `vllm`, `transformers`, `openai`, `gemini` |
| `--api-provider PROVIDER` | None | API provider: `openai`, `gemini`, `anthropic` |
| `--api-key KEY` | None | API key (or set via env var) |

#### Hardware Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--cuda-device DEVICE` | `cuda:0` | CUDA device for local models |
| `--tensor-parallel-size N` | `1` | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization RATIO` | `0.96` | GPU memory fraction to use |
| `--trust-remote-code` | False | Allow remote code from HuggingFace |
| `--no-chat-template` | False | Skip chat template (raw prompt mode) |

#### Generation Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--temperature FLOAT` | `0.7` | Sampling temperature (0.0 = greedy) |
| `--top-p FLOAT` | `0.9` | Nucleus sampling parameter |
| `--max-tokens INT` | `32768` | Maximum tokens to generate |
| `--seed INT` | None | Random seed for reproducibility |

#### API-Specific Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--reasoning-effort LEVEL` | `medium` | OpenAI reasoning effort: `minimal`, `low`, `medium`, `high` |
| `--thinking-budget INT` | `1024` | Gemini thinking budget (0 = disabled, -1 = dynamic) |

#### Evaluation Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--datapoints INT` | `100` | Samples per task configuration |
| `--folds INT` | `1` | Cross-validation folds |
| `--list-sizes SIZES` | None | Comma-separated list sizes for scalable tasks (e.g., `"8,16,32,64"`) |
| `--range-min INT` | `-100` | Minimum value for number generation |
| `--range-max INT` | `100` | Maximum value for number generation |

#### Output Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir PATH` | `./beyondbench_results` | Directory for results |
| `--store-details` | False | Save per-example results |
| `--log-level LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

#### Performance Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size INT` | `1` | Batch size for local inference |
| `--max-retries INT` | `3` | Retries for failed operations |
| `--timeout INT` | `300` | Per-operation timeout in seconds |

#### Parser Mode

| Option | Default | Description |
|--------|---------|-------------|
| `--parser MODE` | `unified` | Parser mode: `unified` or `legacy` |

#### Multi-GPU Options

| Option | Default | Description |
|--------|---------|-------------|
| `--parallel` | False | Enable multi-GPU parallel evaluation |
| `--gpus SPEC` | `auto` | GPU IDs: `"auto"`, `"0,1,2,3"`, or `"0-7"` |
| `--strategy STRATEGY` | `auto` | Parallel strategy (see [Multi-GPU section](#multi-gpu-parallel-evaluation)) |

#### Dashboard

| Option | Default | Description |
|--------|---------|-------------|
| `--dashboard` | False | Launch Gradio observability dashboard |
| `--dashboard-port INT` | `7860` | Dashboard port |

#### Prompt Engineering

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt-style STYLE` | None | Prompt variant: `concise`, `detailed`, `few_shot`, `cot` |

#### Contamination Resistance

| Option | Default | Description |
|--------|---------|-------------|
| `--contamination-resistance LEVEL` | `off` | Noise level: `off`, `low`, `medium`, `high` |

#### Pre-flight Validation

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-validation` | False | Skip model pre-flight checks |
| `--validation-warn-only` | False | Treat validation issues as warnings |

---

### `beyondbench list-tasks`

Print available tasks organized by suite.

```bash
beyondbench list-tasks
beyondbench list-tasks --suite easy
beyondbench list-tasks --suite hard --json
```

---

### `beyondbench run-config`

Run evaluation from a YAML config file.

```bash
beyondbench run-config path/to/config.yaml
```

Config values can be overridden with environment variables:

```bash
BEYONDBENCH_MODEL_ID=gpt-4o beyondbench run-config openai_example.yaml
```

---

### `beyondbench wizard`

Interactive setup wizard that guides you through configuration.

```bash
beyondbench wizard
```

The wizard prompts for backend, model, API keys, suite selection, and output directory.

---

## Configuration Files

Config files use YAML and support four top-level sections: `model`, `evaluation`, `output`, and `performance`.

### Minimal Config

```yaml
model:
  model_id: "meta-llama/Llama-3.2-3B-Instruct"
  backend: "vllm"

evaluation:
  suite: "easy"
  datapoints: 100
  folds: 1

output:
  output_dir: "./beyondbench_results"
```

### Full Config Reference

```yaml
model:
  model_id: "meta-llama/Llama-3.2-3B-Instruct"
  backend: "vllm"                  # vllm | transformers | openai | gemini
  api_provider: null               # openai | gemini | anthropic
  cuda_device: "cuda:0"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.96
  trust_remote_code: false

evaluation:
  suite: "easy"                    # easy | medium | hard | all
  datapoints: 100
  folds: 1
  list_sizes: [8, 16, 32]
  temperature: 0.7
  top_p: 0.9
  max_tokens: 32768
  range_min: -100
  range_max: 100
  seed: 42
  reasoning_effort: "medium"       # OpenAI only
  thinking_budget: 1024            # Gemini only

output:
  output_dir: "./beyondbench_results"
  store_details: false
  log_level: "INFO"

performance:
  batch_size: 1
  max_retries: 3
  timeout: 300
```

### Bundled Config Presets

The package ships several ready-to-use configs in `beyondbench/configs/`:

| File | Description |
|------|-------------|
| `default.yaml` | Local Llama-3.2-3B, easy suite, 100 datapoints |
| `quick_test.yaml` | Qwen 1.5B, easy suite, 5 datapoints (fast smoke test) |
| `full_evaluation.yaml` | All suites, 100 datapoints, stores details |
| `paper_quality.yaml` | All suites, 500 datapoints, 5 folds, seed=42 |
| `openai_example.yaml` | GPT-4o via OpenAI API, easy suite |

---

## Model Backends

### vLLM (Local, Recommended)

vLLM is the fastest local inference option. Requires CUDA and `pip install beyondbench[vllm]`.

```bash
beyondbench evaluate \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --backend vllm \
  --cuda-device cuda:0 \
  --gpu-memory-utilization 0.90 \
  --suite easy
```

For large models requiring multiple GPUs:

```bash
beyondbench evaluate \
  --model-id meta-llama/Llama-3.1-70B-Instruct \
  --backend vllm \
  --tensor-parallel-size 4 \
  --suite easy
```

### Transformers (Local, HuggingFace)

Use the HuggingFace `transformers` library when vLLM is not available:

```bash
beyondbench evaluate \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --backend transformers \
  --cuda-device cuda:0 \
  --suite easy
```

### OpenAI API

Requires `pip install beyondbench[openai]` and an API key.

```bash
export OPENAI_API_KEY="sk-..."

beyondbench evaluate \
  --model-id gpt-4o \
  --api-provider openai \
  --suite all \
  --datapoints 100
```

For o-series reasoning models:

```bash
beyondbench evaluate \
  --model-id gpt-5 \
  --api-provider openai \
  --reasoning-effort high \
  --suite hard
```

### Google Gemini API

Requires `pip install beyondbench[gemini]`.

```bash
export GEMINI_API_KEY="..."

beyondbench evaluate \
  --model-id gemini-2.5-pro \
  --api-provider gemini \
  --thinking-budget 2048 \
  --suite medium
```

### Anthropic Claude API

Requires `pip install beyondbench[anthropic]`.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

beyondbench evaluate \
  --model-id claude-sonnet-4-20250514 \
  --api-provider anthropic \
  --suite hard \
  --datapoints 50
```

---

## Understanding Results

### Output File Structure

```
./beyondbench_results/
  final_results.json        # Aggregated results across all tasks
  evaluation.log            # Full evaluation log
  task_results/
    sum/
      results.json          # Per-task detailed results
    mean/
      results.json
    ...
```

### final_results.json Schema

```json
{
  "model_info": {
    "model_name": "gpt-4o",
    "backend": "openai",
    "api_provider": "openai",
    "model_type": "api_based"
  },
  "overall_stats": {
    "total_tasks": 44,
    "completed_tasks": 44,
    "failed_tasks": 0,
    "total_evaluations": 4400,
    "successful_evaluations": 4312
  },
  "task_results": {
    "sum": {
      "average_accuracy": 0.975,
      "fold_results": [...],
      "total_samples": 100,
      "successful_samples": 97
    },
    ...
  },
  "suite_summaries": {
    "easy": {"average_accuracy": 0.92, "task_count": 44},
    "medium": {"average_accuracy": 0.78, "task_count": 15},
    "hard": {"average_accuracy": 0.45, "task_count": 20}
  }
}
```

### Accuracy Metrics

- **average_accuracy**: Mean correctness across all evaluated samples
- **fold_results**: Per-fold breakdown when `--folds > 1`
- Easy tasks: Exact-match comparison with tolerance for floating-point
- Hard tasks: Exact solution verification or feasibility check

### Reading Results in Python

```python
import json

with open("./beyondbench_results/final_results.json") as f:
    results = json.load(f)

# Print accuracy per task
for task, data in results["task_results"].items():
    print(f"{task:40s} {data['average_accuracy']:.1%}")

# Suite summaries
for suite, summary in results.get("suite_summaries", {}).items():
    print(f"{suite}: avg={summary['average_accuracy']:.1%} ({summary['task_count']} tasks)")
```

---

## Multi-GPU Parallel Evaluation

For large evaluations, use `--parallel` to distribute work across multiple GPUs.

### Basic Usage

```bash
# Auto-detect free GPUs and run in parallel
beyondbench evaluate \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --backend vllm \
  --suite all \
  --parallel \
  --gpus auto
```

### Specifying GPUs

```bash
# Use specific GPUs by index
beyondbench evaluate --model-id ... --parallel --gpus "0,1,2,3"

# Use a range
beyondbench evaluate --model-id ... --parallel --gpus "0-7"
```

### Parallel Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `task_parallel` | Many tasks, 1+ GPUs | Each GPU runs different tasks |
| `data_parallel` | Few tasks, many datapoints | Same task split across GPUs |
| `model_parallel` | Model comparison | Each GPU runs a different model |
| `auto` | General use | Automatically chosen based on task/GPU ratio |

```bash
beyondbench evaluate \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --suite all \
  --parallel \
  --gpus "0,1,2,3" \
  --strategy task_parallel
```

### Memory Considerations

In parallel mode, each GPU loads its own model copy. The default `--gpu-memory-utilization` is automatically capped at a safe level (0.45) for parallel workers. Adjust if needed:

```bash
beyondbench evaluate ... --parallel --gpus "0,1" --gpu-memory-utilization 0.45
```

---

## Prompt Styles

BeyondBench supports four prompt style variants:

| Style | Description |
|-------|-------------|
| (default) | Each task's built-in prompt (recommended) |
| `concise` | Minimal, terse prompt |
| `detailed` | Expanded explanation with step-by-step instructions |
| `few_shot` | Dynamic few-shot examples prepended (contamination-safe) |
| `cot` | Chain-of-thought prompt encouraging reasoning steps |

```bash
beyondbench evaluate \
  --model-id gpt-4o \
  --api-provider openai \
  --suite hard \
  --prompt-style cot
```

Few-shot examples are generated dynamically using `FewShotGenerator` to avoid contamination.

---

## Contamination Resistance

Contamination resistance injects noise into prompts to prevent models from matching against memorized benchmark text:

| Level | Effect |
|-------|--------|
| `off` | No changes (default) |
| `low` | Minor formatting changes |
| `medium` | Additional context added |
| `high` | Distractors + heavy rephrasing |

```bash
beyondbench evaluate \
  --model-id gpt-4o \
  --api-provider openai \
  --suite easy \
  --contamination-resistance high
```

Higher levels stress-test genuine reasoning ability. Use `high` when benchmarking models that may have been trained on BeyondBench data.

---

## Advanced Options

### Custom List Sizes

Many easy/medium tasks operate on lists. Control tested sizes:

```bash
beyondbench evaluate \
  --model-id ... \
  --suite easy \
  --list-sizes "4,8,16,32,64,128"
```

Larger list sizes are harder and stress working memory.

### Number Range Configuration

Control the range of generated numbers for numeric tasks:

```bash
beyondbench evaluate \
  --model-id ... \
  --suite easy \
  --range-min -1000 \
  --range-max 1000
```

### Cross-Validation Folds

Run multiple evaluation folds for statistical robustness:

```bash
beyondbench evaluate \
  --model-id ... \
  --suite easy \
  --folds 5 \
  --seed 42
```

Results per fold are stored in `fold_results` within each task's output.

### Storing Detailed Results

```bash
beyondbench evaluate \
  --model-id ... \
  --suite easy \
  --store-details
```

With `--store-details`, each task directory gets individual example files showing the prompt, model response, expected answer, and whether it was correct.

### Dashboard Monitoring

Launch a live Gradio dashboard during evaluation:

```bash
pip install beyondbench[dashboard]

beyondbench evaluate \
  --model-id ... \
  --suite all \
  --dashboard \
  --dashboard-port 7860
```

Navigate to `http://localhost:7860` to watch evaluation progress in real time.

### Paper-Quality Evaluation

For publication-grade results:

```bash
beyondbench run-config beyondbench/configs/paper_quality.yaml
```

Or equivalently:

```bash
beyondbench evaluate \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --backend vllm \
  --suite all \
  --datapoints 500 \
  --folds 5 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --store-details \
  --output-dir ./paper_results
```

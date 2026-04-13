# Getting Started with BeyondBench

BeyondBench is a contamination-resistant evaluation benchmark for large language models (LLMs). It dynamically generates mathematical and reasoning tasks so models cannot rely on memorized answers, giving you an accurate picture of genuine reasoning capability.

## Installation

### Standard Installation

```bash
pip install beyondbench
```

### With Backend Support

Install optional extras based on which model backends you need:

```bash
# OpenAI API support (GPT-4o, GPT-5, etc.)
pip install beyondbench[openai]

# Google Gemini API support
pip install beyondbench[gemini]

# Anthropic Claude API support
pip install beyondbench[anthropic]

# All API clients at once
pip install beyondbench[all-apis]

# vLLM local inference (requires CUDA)
pip install beyondbench[vllm]

# Full installation (all backends + dev + visualization)
pip install beyondbench[full]
```

### Editable Install (Development)

```bash
git clone https://github.com/ctrl-gaurav/BeyondBench.git
cd BeyondBench
pip install -e .
```

### Verify Installation

```bash
beyondbench --version
beyondbench list-tasks --suite easy
```

---

## Quick Start: Evaluating a Local Model

The fastest way to get started is with a small local model via vLLM:

```bash
beyondbench evaluate \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --backend vllm \
  --suite easy \
  --datapoints 20 \
  --output-dir ./my_results
```

This runs 20 evaluation samples on each easy suite task using vLLM on `cuda:0`.

### Using a Config File

For repeated runs, use a YAML config file instead of CLI flags:

```bash
beyondbench run-config beyondbench/configs/quick_test.yaml
```

A minimal config file looks like:

```yaml
model:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"
  backend: "vllm"

evaluation:
  suite: "easy"
  datapoints: 20
  folds: 1

output:
  output_dir: "./my_results"
```

---

## Quick Start: Evaluating an API Model

### OpenAI

```bash
export OPENAI_API_KEY="your-key-here"

beyondbench evaluate \
  --model-id gpt-4o \
  --api-provider openai \
  --suite easy \
  --datapoints 50 \
  --output-dir ./openai_results
```

### Google Gemini

```bash
export GEMINI_API_KEY="your-key-here"

beyondbench evaluate \
  --model-id gemini-2.5-pro \
  --api-provider gemini \
  --suite medium \
  --datapoints 30
```

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-key-here"

beyondbench evaluate \
  --model-id claude-sonnet-4-20250514 \
  --api-provider anthropic \
  --suite hard \
  --datapoints 25
```

---

## Understanding Results

After evaluation completes, results are saved to the output directory. The file layout is:

```
./beyondbench_results/
  final_results.json          # Top-level summary
  evaluation.log              # Detailed log file
  task_results/
    sum/                      # Per-task directory
      results.json            # Task-level results
    mean/
      results.json
    ...
```

### Reading final_results.json

```python
import json

with open("./beyondbench_results/final_results.json") as f:
    results = json.load(f)

# Top-level accuracy per task
for task_name, task_data in results["task_results"].items():
    acc = task_data.get("average_accuracy", 0)
    print(f"{task_name}: {acc:.1%}")

# Overall stats
stats = results.get("overall_stats", {})
print(f"Total tasks: {stats.get('total_tasks')}")
print(f"Completed:   {stats.get('completed_tasks')}")
```

### Sample Output

```
sum:                  97.5%
mean:                 94.0%
sorting:              88.5%
fibonacci_sequence:   72.0%
knapsack:             45.0%
sudoku_solving:       31.0%
```

Easy tasks typically achieve 80–99% accuracy for strong models; hard tasks may be 20–60%.

---

## Next Steps

- **Full CLI reference**: [user_guide.md](user_guide.md)
- **All 84 tasks explained**: [task_reference.md](task_reference.md)
- **Python API**: [api_reference.md](api_reference.md)
- **Add custom tasks**: [contributing.md](contributing.md)
- **Jupyter notebooks**: [examples/](examples/)

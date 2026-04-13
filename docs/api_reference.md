# BeyondBench Python API Reference

BeyondBench can be used as a Python library for programmatic evaluation, result analysis, and custom task development.

## Table of Contents

1. [Quick Start](#quick-start)
2. [ModelHandler](#modelhandler)
3. [EvaluationEngine](#evaluationengine)
4. [TaskRegistry](#taskregistry)
5. [BaseTask](#basetask)
6. [ParallelEvaluationEngine](#parallelevaluationengine)
7. [ResultAggregator](#resultaggregator)
8. [GPUUtilizationSampler](#gpuutilizationsampler)

---

## Quick Start

```python
from beyondbench.models.model_handler import ModelHandler
from beyondbench.core.evaluation_engine import EvaluationEngine

# Initialize a model handler (OpenAI example)
model = ModelHandler(
    model_id="gpt-4o",
    api_provider="openai",
    api_key="sk-...",
)

# Initialize and run the evaluation engine
engine = EvaluationEngine(
    model_handler=model,
    output_dir="./my_results",
    store_details=True,
)

results = engine.run_evaluation(
    suite="easy",
    datapoints=50,
    folds=1,
)

# Print per-task accuracy
for task_name, task_data in results["task_results"].items():
    if isinstance(task_data, list):
        acc = sum(m.get("accuracy", 0) for m in task_data) / len(task_data)
    else:
        acc = task_data.get("summary", {}).get("avg_accuracy", 0)
    print(f"{task_name}: {acc:.1%}")
```

---

## ModelHandler

```python
from beyondbench.models.model_handler import ModelHandler
```

Handles all model inference, supporting API-based and local models.

### Constructor

```python
ModelHandler(
    model_id: str,
    api_provider: str = None,
    api_key: str = None,
    reasoning_effort: str = None,
    thinking_budget: int = None,
    backend: str = None,
    **kwargs
)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | Model identifier: HuggingFace path or API model name |
| `api_provider` | `str` | `"openai"`, `"gemini"`, or `"anthropic"` (None for local) |
| `api_key` | `str` | API key for the provider |
| `reasoning_effort` | `str` | OpenAI reasoning effort: `"minimal"`, `"low"`, `"medium"`, `"high"` |
| `thinking_budget` | `int` | Gemini thinking budget (0 = off, -1 = dynamic) |
| `backend` | `str` | For local models: `"vllm"` (default) or `"transformers"` |
| `**kwargs` | | `cuda_device`, `tensor_parallel_size`, `gpu_memory_utilization`, `trust_remote_code` |

**Examples**:

```python
# OpenAI GPT
handler = ModelHandler(
    model_id="gpt-4o",
    api_provider="openai",
    api_key="sk-...",
)

# Anthropic Claude
handler = ModelHandler(
    model_id="claude-sonnet-4-20250514",
    api_provider="anthropic",
    api_key="sk-ant-...",
)

# Google Gemini
handler = ModelHandler(
    model_id="gemini-2.5-pro",
    api_provider="gemini",
    api_key="...",
    thinking_budget=2048,
)

# Local model via vLLM
handler = ModelHandler(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    backend="vllm",
    cuda_device="cuda:0",
    gpu_memory_utilization=0.90,
)

# Local model via HuggingFace Transformers
handler = ModelHandler(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    backend="transformers",
    cuda_device="cuda:0",
)
```

### Methods

#### `generate(prompts, max_tokens, temperature, top_p, **kwargs) -> List[str]`

Generate responses for a list of prompts.

```python
prompts = ["What is 2 + 2?", "What is the capital of France?"]
responses = handler.generate(
    prompts=prompts,
    max_tokens=512,
    temperature=0.0,
    top_p=1.0,
)
# responses: ["4", "Paris"]
```

**Parameters**:
- `prompts`: List of prompt strings
- `max_tokens`: Maximum tokens to generate (default: 32768, auto-falls-back to 8192 on error)
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter

**Returns**: `List[str]` — one response per prompt

#### `generate_batch(prompts, max_tokens, temperature, top_p, **kwargs) -> List[str]`

Alias for `generate()`, included for API compatibility.

#### `get_model_info() -> Dict[str, Any]`

Returns metadata about the loaded model.

```python
info = handler.get_model_info()
# {
#   "model_name": "gpt-4o",
#   "backend": "openai",
#   "api_provider": "openai",
#   "model_type": "api_based"
# }
```

#### `count_tokens(text) -> int`

Count tokens in a text string using tiktoken.

```python
n = handler.count_tokens("Hello, world!")
# 4
```

#### `get_statistics() -> Dict[str, Any]`

Returns API usage statistics (calls, tokens, timing).

```python
stats = handler.get_statistics()
print(f"API calls: {stats['api_calls']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Duration: {stats['total_time_formatted']}")
```

#### `print_statistics_report()`

Print a formatted statistics report to stdout.

---

## EvaluationEngine

```python
from beyondbench.core.evaluation_engine import EvaluationEngine
```

Orchestrates evaluation across one or more tasks.

### Constructor

```python
EvaluationEngine(
    model_handler,
    output_dir: str = "./beyondbench_results",
    store_details: bool = False,
    max_retries: int = 3,
    **kwargs
)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_handler` | `ModelHandler` | Initialized model handler |
| `output_dir` | `str` | Directory to store results |
| `store_details` | `bool` | Save per-example results |
| `max_retries` | `int` | Retries per failed sample |

### Methods

#### `run_evaluation(suite, tasks, datapoints, folds, **eval_params) -> Dict[str, Any]`

Run evaluation and return results dict.

```python
results = engine.run_evaluation(
    suite="easy",           # "easy" | "medium" | "hard" | "all"
    tasks=None,             # None = all tasks in suite; or ["sum", "mean"]
    datapoints=100,
    folds=1,
    temperature=0.7,
    top_p=0.9,
    max_tokens=32768,
    seed=42,
    range_min=-100,
    range_max=100,
    list_sizes=[8, 16, 32],
    prompt_style=None,      # None | "concise" | "detailed" | "few_shot" | "cot"
    contamination_resistance="off",  # "off" | "low" | "medium" | "high"
)
```

**Returns**: Dict with keys:
- `"summary"`: Aggregate metrics (avg_accuracy, total_tasks, etc.)
- `"task_results"`: Per-task results keyed by task name
- `"overall_stats"`: Counts of completed/failed tasks
- `"model_info"`: From `model_handler.get_model_info()`
- `"evaluation_config"`: Suite, tasks list, output dir

#### `get_available_tasks(suite) -> Dict[str, List[str]]`

List available tasks organized by suite.

```python
registry_view = engine.get_available_tasks("all")
# {"easy": ["sum", "mean", ...], "medium": [...], "hard": [...]}
```

#### `validate_configuration() -> bool`

Validate that the engine configuration is correct before running.

---

## TaskRegistry

```python
from beyondbench.core.task_registry import TaskRegistry
```

Central registry for task discovery and retrieval.

### Constructor

```python
registry = TaskRegistry()
```

Automatically discovers and registers all available tasks on initialization.

### Methods

#### `get_task_class(task_name) -> Optional[Type[BaseTask]]`

Get the task class for a given task name.

```python
TaskClass = registry.get_task_class("knapsack")
# <class 'beyondbench.tasks.hard.knapsack_task.KnapsackTask'>
```

#### `get_tasks_for_suite(suite) -> List[str]`

Get all task names for a suite.

```python
easy_tasks = registry.get_tasks_for_suite("easy")
# ["sum", "mean", "sorting", ...]

all_tasks = registry.get_tasks_for_suite("all")
```

#### `get_available_tasks(suite) -> Dict[str, List[str]]`

Get tasks organized by suite.

```python
all_tasks = registry.get_available_tasks("all")
# {"easy": [...], "medium": [...], "hard": [...]}
```

#### `register_task(task_name, task_class, suite)`

Register a custom task class.

```python
from beyondbench.core.task_registry import TaskRegistry
from beyondbench.core.base_task import BaseTask

class MyCustomTask(BaseTask):
    @property
    def task_name(self):
        return "my_custom_task"
    # ... implement other abstract methods

registry = TaskRegistry()
registry.register_task("my_custom_task", MyCustomTask, "easy")
```

#### `validate_task_name(task_name) -> bool`

Check whether a task name is registered.

```python
assert registry.validate_task_name("sum") is True
assert registry.validate_task_name("nonexistent") is False
```

---

## BaseTask

```python
from beyondbench.core.base_task import BaseTask
```

Abstract base class that all tasks extend. Use this to implement custom tasks.

### Constructor

```python
BaseTask(
    model_handler,
    output_dir: str,
    min_val: int,
    max_val: int,
    num_folds: int,
    num_samples: int,
    store_details: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int = None,
    max_retries: int = 3,
    **kwargs
)
```

### Abstract Properties and Methods

Subclasses must implement:

#### `task_name` (property)

Return a unique string identifier for the task.

```python
@property
def task_name(self):
    return "my_task"
```

#### `generate_data(**kwargs) -> List[Any]`

Generate evaluation data points.

```python
def generate_data(self, list_size=10, **kwargs):
    import random
    return [
        [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
        for _ in range(self.num_samples)
    ]
```

#### `create_prompt(data_point) -> str`

Convert a data point to a prompt string.

```python
def create_prompt(self, data_point):
    return (
        f"Add the numbers: {data_point}\n\n"
        "Put your final answer in \\boxed{{answer}}."
    )
```

#### `evaluate_response(response, data_point) -> bool`

Check if the model's response is correct for this data point.

```python
def evaluate_response(self, response, data_point):
    import re
    expected = sum(data_point)
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if not match:
        return False
    try:
        return int(match.group(1)) == expected
    except ValueError:
        return False
```

### Public Methods

#### `get_prompt(data_point) -> str`

Returns the prompt for a data point, respecting `prompt_style` and contamination resistance settings. Prefer this over calling `create_prompt` directly.

#### `count_tokens(text) -> int`

Count tokens in a string using the best available tokenizer.

#### `save_detailed_results(results, test_case_id, fold)`

Save detailed results for a test case (only active when `store_details=True`).

---

## ParallelEvaluationEngine

```python
from beyondbench.core.parallel_engine import ParallelEvaluationEngine
```

Extends `EvaluationEngine` with multi-GPU support via Python multiprocessing.

### Constructor

```python
ParallelEvaluationEngine(
    model_handler,
    output_dir: str = "./beyondbench_results",
    store_details: bool = False,
    max_retries: int = 3,
    **kwargs
)
```

### Methods

#### `run_parallel_evaluation(suite, tasks, gpu_ids, strategy, model_id, model_kwargs, **eval_config) -> Dict[str, Any]`

Run evaluation across multiple GPUs.

```python
from beyondbench.core.parallel_engine import ParallelEvaluationEngine
from beyondbench.core.gpu_scheduler import GPUScheduler
from beyondbench.models.model_handler import ModelHandler

# Lightweight handler (no GPU weights loaded in parent process)
handler = ModelHandler(model_id="meta-llama/Llama-3.2-3B-Instruct")

engine = ParallelEvaluationEngine(
    model_handler=handler,
    output_dir="./parallel_results",
)

scheduler = GPUScheduler()
gpu_ids = scheduler.parse_gpu_spec("0,1,2,3", "meta-llama/Llama-3.2-3B-Instruct")

results = engine.run_parallel_evaluation(
    suite="all",
    tasks=None,
    gpu_ids=gpu_ids,
    strategy="task_parallel",   # "task_parallel" | "data_parallel" | "auto"
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    model_kwargs={"backend": "vllm", "gpu_memory_utilization": 0.45},
    datapoints=100,
    folds=1,
    temperature=0.7,
)
```

**Notes**:
- Each GPU worker spawns in a separate process with its own CUDA context
- `gpu_memory_utilization` should be lowered for parallel mode (e.g., 0.45) to avoid OOM
- Results are merged by `ResultAggregator` and match the format of single-GPU evaluation

---

## ResultAggregator

```python
from beyondbench.core.result_aggregator import ResultAggregator
```

Merges results from multiple GPU workers into a single unified result dict.

### Static Method: `aggregate`

```python
ResultAggregator.aggregate(
    worker_results: List[Dict[str, Any]],
    strategy: str,
    overall_start_time: float,
    model_info: Dict[str, Any],
    eval_config: Dict[str, Any],
) -> Dict[str, Any]
```

**Parameters**:
- `worker_results`: List of result dicts from each GPU worker (keys: `task_results`, `overall_stats`, `gpu_id`)
- `strategy`: `"task_parallel"`, `"data_parallel"`, `"model_parallel"`, or `"auto"`
- `overall_start_time`: `time.time()` at evaluation start
- `model_info`: From `ModelHandler.get_model_info()`
- `eval_config`: Evaluation config dict for metadata

**Returns**: Unified result dict in the same format as `EvaluationEngine._aggregate_results()`.

**Note**: This is typically used internally by `ParallelEvaluationEngine`. Direct use is only needed for custom parallel workflows.

---

## GPUUtilizationSampler

```python
from beyondbench.core.result_aggregator import GPUUtilizationSampler
```

Background sampler for per-GPU utilization and memory usage via `nvidia-smi`.

### Constructor

```python
sampler = GPUUtilizationSampler(interval_seconds=2.0)
```

### Methods

```python
sampler.start()       # Start background sampling thread
sampler.stop()        # Stop sampling thread
snap = sampler.sample()       # Latest snapshot (threadsafe)
snap = sampler.sample_once()  # One-shot synchronous sample
```

**Snapshot format**:
```python
{
    0: {"util": 73, "mem_used_mib": 18542, "mem_total_mib": 24576},
    1: {"util": 45, "mem_used_mib": 9216, "mem_total_mib": 24576},
}
```

Degrades gracefully to an empty dict if `nvidia-smi` is unavailable (CPU boxes, CI).

---

## Complete Programmatic Example

```python
"""
Full programmatic evaluation example.
Evaluates a local model on the easy suite and prints results.
"""
import json
from beyondbench.models.model_handler import ModelHandler
from beyondbench.core.evaluation_engine import EvaluationEngine

# 1. Create model handler
handler = ModelHandler(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    backend="vllm",
    cuda_device="cuda:0",
    gpu_memory_utilization=0.90,
)

# 2. Verify model info
print(handler.get_model_info())

# 3. Create evaluation engine
engine = EvaluationEngine(
    model_handler=handler,
    output_dir="./programmatic_results",
    store_details=True,
    max_retries=3,
)

# 4. Run specific tasks
results = engine.run_evaluation(
    suite="easy",
    tasks=["sum", "mean", "sorting", "find_maximum"],
    datapoints=50,
    folds=1,
    temperature=0.0,
    top_p=1.0,
    max_tokens=4096,
    seed=42,
    list_sizes=[8, 16],
)

# 5. Inspect results
summary = results.get("summary", {})
print(f"Average accuracy: {summary.get('avg_accuracy', 0):.1%}")
print(f"Completed tasks:  {summary.get('completed_tasks', 0)}/{summary.get('total_tasks', 0)}")

# 6. Per-task breakdown
for task_name, task_data in results["task_results"].items():
    if isinstance(task_data, list) and task_data:
        acc = sum(m.get("accuracy", 0) for m in task_data) / len(task_data)
    elif isinstance(task_data, dict):
        acc = task_data.get("summary", {}).get("avg_accuracy", 0)
    else:
        acc = 0.0
    print(f"  {task_name:30s}: {acc:.1%}")

# 7. Save results (engine already saves, but you can re-save anywhere)
with open("custom_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

---

## Custom Task Example

```python
"""
Implementing a custom task that tests multiplication tables.
"""
import random
import re
from beyondbench.core.base_task import BaseTask
from beyondbench.core.task_registry import TaskRegistry


class MultiplicationTableTask(BaseTask):
    """Test multiplication table knowledge."""

    @property
    def task_name(self):
        return "multiplication_table"

    def generate_data(self, **kwargs):
        """Generate (a, b) pairs where a, b are in [1, 12]."""
        if self.seed is not None:
            random.seed(self.seed)
        return [
            (random.randint(1, 12), random.randint(1, 12))
            for _ in range(self.num_samples)
        ]

    def create_prompt(self, data_point):
        a, b = data_point
        return (
            f"What is {a} times {b}?\n\n"
            r"Put your answer in \boxed{answer}."
        )

    def evaluate_response(self, response, data_point):
        a, b = data_point
        expected = a * b
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if not match:
            return False
        try:
            return int(match.group(1).strip()) == expected
        except (ValueError, AttributeError):
            return False


# Register and use the custom task
from beyondbench.models.model_handler import ModelHandler
from beyondbench.core.evaluation_engine import EvaluationEngine

handler = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="sk-...")
registry = TaskRegistry()
registry.register_task("multiplication_table", MultiplicationTableTask, "easy")

engine = EvaluationEngine(handler, output_dir="./custom_results")
results = engine.run_evaluation(
    suite="easy",
    tasks=["multiplication_table"],
    datapoints=30,
)
```

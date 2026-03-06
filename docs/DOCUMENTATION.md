# 📚 beyondbench Complete Documentation

## 🚀 Command Reference

### Core Commands

#### `beyondbench evaluate`
Primary evaluation command with comprehensive options.

```bash
beyondbench evaluate [OPTIONS]
```

**Required Parameters:**
- `--model-id TEXT`: Model identifier (HuggingFace path or API model name)

**Backend Configuration:**
- `--backend [vllm|transformers|openai|gemini]`: Inference backend (default: vllm for local models, with automatic fallback to transformers). For API models, you can also use `--api-provider` instead.
- `--api-provider [openai|gemini|anthropic]`: API provider for cloud models (preferred way to specify API backends)
- `--api-key TEXT`: API key (or set environment variables)

**Hardware Configuration:**
- `--cuda-device TEXT`: CUDA device (default: cuda:0)
- `--tensor-parallel-size INTEGER`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu-memory-utilization FLOAT`: GPU memory utilization ratio (default: 0.96)
- `--trust-remote-code`: Allow remote code execution

**Generation Parameters:**
- `--temperature FLOAT`: Sampling temperature (default: 0.7)
- `--top-p FLOAT`: Nucleus sampling parameter (default: 0.9)
- `--max-tokens INTEGER`: Maximum tokens to generate (default: 32768, falls back to 8192 on error)
- `--seed INTEGER`: Random seed for reproducibility

**API-Specific Parameters:**
- `--reasoning-effort [minimal|low|medium|high]`: OpenAI GPT-5 reasoning effort (default: medium)
- `--thinking-budget INTEGER`: Gemini thinking budget (default: 1024, 0 to disable, -1 for dynamic)

**Evaluation Parameters:**
- `--tasks TEXT`: Task selection, comma-separated (e.g., --tasks sorting,comparison)
- `--suite [easy|medium|hard|all]`: Task suite to run (default: all)
- `--datapoints INTEGER`: Number of datapoints per task (default: 100)
- `--folds INTEGER`: Number of cross-validation folds (default: 1)
- `--list-sizes TEXT`: Comma-separated list sizes (e.g., "8,16,32,64")
- `--range-min INTEGER`: Minimum value for number generation (default: -100)
- `--range-max INTEGER`: Maximum value for number generation (default: 100)

**Output Configuration:**
- `--output-dir TEXT`: Output directory for results (default: ./beyondbench_results)
- `--store-details`: Store detailed per-example results
- `--log-level [DEBUG|INFO|WARNING|ERROR]`: Logging level (default: INFO)

**Performance Options:**
- `--batch-size INTEGER`: Batch size for local model inference (default: 1)
- `--max-retries INTEGER`: Maximum retries for failed operations (default: 3)
- `--timeout INTEGER`: Timeout for individual operations in seconds (default: 300)

#### `beyondbench` (Interactive Wizard)
Running `beyondbench` without arguments launches an interactive setup wizard that guides you through configuration step by step.

```bash
beyondbench
```

#### `beyondbench list-tasks`
List available tasks in each suite.

```bash
beyondbench list-tasks [OPTIONS]
```

**Options:**
- `--suite [easy|medium|hard|all]`: Task suite to list (default: all)
- `--format [table|json|yaml]`: Output format (default: table)

#### `beyondbench run-config`
Run evaluation from configuration file.

```bash
beyondbench run-config CONFIG_FILE
```

#### `beyondbench serve`
Start the BeyondBench API server (requires `pip install beyondbench[serve]`).

```bash
beyondbench serve [OPTIONS]
```

**Options:**
- `--host TEXT`: Host to bind to (default: 0.0.0.0)
- `--port INTEGER`: Port to listen on (default: 8000)
- `--reload`: Enable auto-reload for development

**API Endpoints:**
- `GET /health` - Health check
- `GET /tasks` - List tasks (filterable by suite)
- `GET /tasks/{task_name}` - Task details
- `POST /evaluate` - Start evaluation job
- `GET /jobs/{job_id}` - Check job status
- `GET /results` - List past results
- `GET /results/{result_id}` - Get result details
- `GET /docs` - Interactive API documentation (Swagger UI)

#### `beyondbench init`
Create a configuration file interactively.

```bash
beyondbench init [--output beyondbench.yaml]
```

#### `beyondbench info`
Show detailed information about a specific task.

```bash
beyondbench info TASK_NAME
```

#### `beyondbench results`
View and compare past evaluation results.

```bash
# List results
beyondbench results list [--output-dir ./beyondbench_results]

# Show detailed results
beyondbench results show PATH_TO_RESULTS_JSON

# Compare two results
beyondbench results compare PATH1 PATH2
```

## 🔧 Environment Variables

Set these environment variables for seamless API usage:

```bash
# OpenAI Configuration
export OPENAI_API_KEY="your-openai-api-key"

# Gemini Configuration
export GEMINI_API_KEY="your-gemini-api-key"
# or
export GOOGLE_API_KEY="your-google-api-key"

# Anthropic Configuration
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For gated HuggingFace models (e.g., Llama, Mistral)
export HF_TOKEN="your-huggingface-token"

# CUDA Configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

## 📋 Configuration Files

### YAML Configuration

Create `eval_config.yaml`:

```yaml
model:
  model_id: "gpt-4o"
  api_provider: "openai"

evaluation:
  suite: "easy"
  tasks:
    - "sorting"
    - "comparison"
  datapoints: 50
  folds: 3
  temperature: 0.1
  max_tokens: 1024
  seed: 42

output:
  output_dir: "./results"
  store_details: true
  log_level: "INFO"
```

### JSON Configuration

Create `eval_config.json`:

```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "backend": "vllm",
  "suite": "medium",
  "datapoints": 100,
  "list_sizes": "8,16,32",
  "temperature": 0.7,
  "tensor_parallel_size": 1,
  "output_dir": "./llama_results",
  "store_details": false
}
```

## 🎯 Task-Specific Usage

### Easy Suite Tasks

**Scalable Tasks** (support `--list-sizes`):
```bash
# Test with different complexities
beyondbench evaluate --model-id gpt-4o --tasks sorting,sum,find_maximum --list-sizes "8,16,32,64"
```

**Fixed Tasks** (single test case):
```bash
# Simple comparison tasks
beyondbench evaluate --model-id gpt-4o --tasks comparison,division,absolute_difference
```

### Medium Suite Tasks

**Sequence Types:**
```bash
# All sequence types
beyondbench evaluate --model-id gpt-4o --suite medium --datapoints 20

# Specific sequence families
beyondbench evaluate --model-id gpt-4o --tasks fibonacci_sequence,prime_sequence
```

### Hard Suite Tasks

**Complexity Levels:**
```bash
# Start with easier problems
beyondbench evaluate --model-id gpt-4o --tasks tower_hanoi --datapoints 10

# Full hard suite (computational intensive)
beyondbench evaluate --model-id gpt-4o --suite hard --datapoints 5 --timeout 600
```

## 🚀 Backend-Specific Configuration

### vLLM Backend

**Single GPU:**
```bash
beyondbench evaluate \\
  --model-id meta-llama/Llama-3.2-7B-Instruct \\
  --backend vllm \\
  --gpu-memory-utilization 0.9
```

**Multi-GPU (Tensor Parallelism):**
```bash
beyondbench evaluate \\
  --model-id meta-llama/Llama-3.2-70B-Instruct \\
  --backend vllm \\
  --tensor-parallel-size 4 \\
  --gpu-memory-utilization 0.95
```

### Transformers Backend

**Automatic Device Mapping:**
```bash
beyondbench evaluate \\
  --model-id microsoft/Phi-3.5-mini-instruct \\
  --backend transformers \\
  --trust-remote-code
```

### OpenAI API

**Supported Models:** gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-5, gpt-5-mini, gpt-5-nano, o1, o1-mini, o3, o3-mini, o4-mini

**GPT-4o with Standard Parameters:**
```bash
beyondbench evaluate \\
  --model-id gpt-4o \\
  --api-provider openai \\
  --temperature 0.1 \\
  --top-p 0.95
```

**GPT-5 with Reasoning Effort:**
```bash
beyondbench evaluate \\
  --model-id gpt-5 \\
  --api-provider openai \\
  --reasoning-effort high
```

### Gemini API

**Supported Models:** gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro, gemini-1.5-flash

**With Thinking Budget:**
```bash
beyondbench evaluate \\
  --model-id gemini-2.5-pro \\
  --api-provider gemini \\
  --thinking-budget 2048
```

### Anthropic API

**Supported Models:** claude-sonnet-4-20250514, claude-opus-4-20250514, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229, claude-3-haiku-20240307

**Claude Models:**
```bash
beyondbench evaluate \\
  --model-id claude-sonnet-4-20250514 \\
  --api-provider anthropic \\
  --suite all
```

## 📊 Output Analysis

### Results Structure

```
beyondbench_results/
├── final_results.json              # Main results file
├── evaluation_summary.json         # Summary metrics
├── model_statistics.json           # Model usage statistics
├── beyondbench_YYYYMMDD_HHMMSS.log # Detailed logs
└── task_results/                   # Per-task results (if --store-details)
    ├── sorting/
    │   ├── test_case_0/
    │   │   └── detailed_results_fold_0.json
    │   └── ...
    ├── comparison/
    └── ...
```

### Metrics Interpretation

**Accuracy**: Task-specific correctness (0.0 to 1.0)
- Easy tasks: Typically >0.90 for strong models
- Medium tasks: 0.60-0.85 range for mathematical sequences
- Hard tasks: 0.30-0.70 range for complex reasoning

**Efficiency**: Accuracy per token used
- Higher values indicate concise, correct reasoning
- Useful for comparing verbose vs. concise models

**Success Rate**: Percentage of successfully parsed responses
- Should be >95% for production use
- Lower rates indicate parsing issues

**Instruction Following**: Whether the model followed the requested output format (e.g., used `\boxed{}`)
- Tracked per sample in detailed results
- High instruction following with low accuracy indicates the model understands the format but not the task

## 🔧 Troubleshooting

### Common Issues

**1. API Rate Limiting**
```bash
# Reduce concurrency and add delays
beyondbench evaluate --model-id gpt-4o --datapoints 10 --max-retries 5
```

**2. GPU Memory Issues**
```bash
# Reduce memory utilization
beyondbench evaluate --model-id large-model --gpu-memory-utilization 0.8

# Use smaller tensor parallel size
beyondbench evaluate --model-id large-model --tensor-parallel-size 1
```

**3. Parsing Failures**
```bash
# Enable debug logging for parsing issues
beyondbench evaluate --model-id model --log-level DEBUG --store-details
```

**4. Timeout Issues**
```bash
# Increase timeout for complex tasks
beyondbench evaluate --model-id model --suite hard --timeout 600
```

### Debug Mode

```bash
# Enable comprehensive debugging
beyondbench evaluate \\
  --model-id gpt-4o \\
  --tasks sorting \\
  --datapoints 1 \\
  --log-level DEBUG \\
  --store-details
```

## 🧪 Testing and Validation

### Quick Functionality Test

```bash
# Test basic functionality
beyondbench evaluate --model-id gpt-4o-mini --tasks sorting --datapoints 1

# Validate all backends
beyondbench list-tasks --suite all --format json

# Test configuration file
echo '{"model_id": "gpt-4o", "tasks": ["sorting"], "datapoints": 1}' > test_config.json
beyondbench run-config test_config.json
```

### Performance Benchmarking

```bash
# Benchmark local model
time beyondbench evaluate \\
  --model-id meta-llama/Llama-3.2-3B-Instruct \\
  --backend vllm \\
  --tasks sorting,comparison \\
  --datapoints 50

# Compare backends
for backend in vllm transformers; do
  echo "Testing $backend..."
  beyondbench evaluate --model-id same-model --backend $backend --output-dir results_$backend
done
```

## 💡 Best Practices

### Production Usage

1. **Use Environment Variables**: Set API keys via environment variables
2. **Enable Logging**: Use INFO or DEBUG level for production monitoring
3. **Store Details**: Enable for important evaluations to debug issues
4. **Multiple Folds**: Use 3-5 folds for statistical reliability
5. **Appropriate Timeouts**: Set based on task complexity

### Performance Optimization

1. **Local Models**: Prefer vLLM for better throughput
2. **GPU Memory**: Use 0.9-0.95 utilization for optimal performance
3. **Batch Processing**: Increase batch size for non-interactive tasks
4. **Tensor Parallelism**: Use for models >30B parameters

### Cost Management

1. **Start Small**: Test with fewer datapoints first
2. **Use Mini Models**: Test workflows with cheaper models
3. **Monitor Usage**: Check statistics reports for cost tracking
4. **Efficient Prompting**: Use appropriate max_tokens limits

## 🔗 Integration Examples

### Python Integration

```python
from beyondbench import ModelHandler, EvaluationEngine

# Direct programmatic usage
handler = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="your-api-key")
engine = EvaluationEngine(model_handler=handler, output_dir="./results")
results = engine.run_evaluation(suite="easy", datapoints=50)
```

### Batch Processing

```bash
#!/bin/bash
# evaluate_multiple_models.sh

models=("gpt-4o" "gpt-4o-mini" "gemini-2.5-pro")
suites=("easy" "medium")

for model in "${models[@]}"; do
  for suite in "${suites[@]}"; do
    echo "Evaluating $model on $suite suite..."
    beyondbench evaluate \\
      --model-id "$model" \\
      --suite "$suite" \\
      --output-dir "results_${model}_${suite}" \\
      --datapoints 50
  done
done
```

### CI/CD Integration

```yaml
# .github/workflows/evaluation.yml
name: Model Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install beyondbench
      run: pip install beyondbench
    - name: Run evaluation
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        beyondbench evaluate --model-id gpt-4o-mini --tasks sorting --datapoints 5
```

---

For more examples and advanced usage patterns, see our [GitHub repository](https://github.com/beyondbench/beyondbench) and [API documentation](https://docs.beyondbench.org).
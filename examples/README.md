# BeyondBench Interactive Demos

Five Gradio apps that showcase BeyondBench capabilities. Each runs standalone — no GPU required (apps fall back to simulation mode when vLLM is unavailable).

## Requirements

```bash
pip install gradio
pip install -e .  # or: pip install beyondbench
```

## Apps

### 1. Live Evaluation Dashboard (`gradio_live_eval.py`)

Select a model, suite, GPU, and datapoints — click "Start" and watch results populate in real time. Download results as CSV or JSON when complete.

```bash
python examples/gradio_live_eval.py  # http://localhost:7860
```

- Real-time results table with accuracy, parse rate, and timing per task
- Live log stream with auto-scroll
- Start/stop controls, GPU selector, output directory config
- CSV and JSON export

### 2. Model Arena (`gradio_model_arena.py`)

Side-by-side model comparison. Pick two models, generate the same task instance for both, compare responses, and vote for the better one.

```bash
python examples/gradio_model_arena.py  # http://localhost:7861
```

- Parallel response generation from two models
- Ground truth display for objective comparison
- Human voting (A wins / B wins / Tie)
- In-memory leaderboard with win rates

### 3. Task Explorer (`gradio_task_explorer.py`)

Browse all 79 tasks across Easy, Medium, and Hard suites. Click any task to see a sample instance with ground truth, regenerate with new seeds, or paste custom responses to test the parser.

```bash
python examples/gradio_task_explorer.py  # http://localhost:7862
```

- Filterable task catalog with descriptions
- Sample instance generation with configurable seed
- Custom response evaluation through the unified parser

### 4. Parser Debugger (`gradio_parser_debugger.py`)

Paste any model response and see exactly how each parsing strategy handles it — extracted value, confidence score, raw match, and which strategy wins.

```bash
python examples/gradio_parser_debugger.py  # http://localhost:7863
```

- Per-strategy breakdown: boxed, explicit_statement, code_block, latex_math, json, list, grid, comparison, sequence, fallback
- Auto-debug on text change
- Example responses for common tasks
- Strategy reference table

### 5. GPU Monitor (`gradio_gpu_monitor.py`)

Real-time GPU utilization, memory, temperature, and power for all available GPUs with rolling history charts and configurable alert thresholds.

```bash
python examples/gradio_gpu_monitor.py  # http://localhost:7864
```

- Live stats table updated every 2 seconds
- Memory usage and utilization timeline plots
- Configurable alert thresholds for memory, utilization, and temperature
- Graceful fallback when no GPUs are detected

## Port Summary

| App | Port |
|-----|------|
| Live Evaluation | 7860 |
| Model Arena | 7861 |
| Task Explorer | 7862 |
| Parser Debugger | 7863 |
| GPU Monitor | 7864 |

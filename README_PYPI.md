# BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2509.24210-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2509.24210)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue?style=for-the-badge)](https://iclr.cc/)
[![PyPI](https://img.shields.io/pypi/v/beyondbench.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/beyondbench/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/beyondbench/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](https://github.com/ctrl-gaurav/BeyondBench/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/ctrl-gaurav/BeyondBench?style=for-the-badge&logo=github)](https://github.com/ctrl-gaurav/BeyondBench/stargazers)

**101+ Models Evaluated | 44 Reasoning Tasks | 117 Variations | >10^15 Unique Instances**

[Explore Leaderboard](https://ctrl-gaurav.github.io/BeyondBench/) | [Read Paper](https://arxiv.org/abs/2509.24210) | [GitHub](https://github.com/ctrl-gaurav/BeyondBench) | [Documentation](https://github.com/ctrl-gaurav/BeyondBench/blob/main/docs/DOCUMENTATION.md)

---

## What is BeyondBench?

BeyondBench introduces a **revolutionary approach** to evaluating reasoning capabilities in language models without relying on traditional static benchmarks. Our system **dynamically generates** novel problems across **44 distinct reasoning tasks** with **117 variations**, ensuring that models cannot memorize solutions and must demonstrate **true reasoning abilities**.

### Key Features

- **Dynamic Problem Generation** — Problem space >10^15 unique instances, zero risk of data contamination
- **Three Difficulty Levels** — Easy (29 tasks), Medium (5 tasks, 49 variations), Hard (10 tasks, 68 variations)
- **Multi-Backend Support** — OpenAI, Gemini, Anthropic APIs + vLLM and HuggingFace Transformers
- **Contamination-Resistant** — No static benchmark memorization, novel problems every run
- **Comprehensive Metrics** — Accuracy, instruction-following compliance, token efficiency
- **101+ Models Evaluated** — Open-source and proprietary, regularly updated

---

## Installation

### From PyPI

```bash
pip install beyondbench
```

### With Optional Dependencies

```bash
# All API clients (OpenAI, Gemini, Anthropic)
pip install beyondbench[all-apis]

# vLLM support (requires CUDA)
pip install beyondbench[vllm]

# Everything (all APIs + vLLM + dev tools + visualization)
pip install beyondbench[full]
```

### From Source

```bash
git clone https://github.com/ctrl-gaurav/BeyondBench.git
cd BeyondBench
pip install -e .
```

---

## Quick Start

### Interactive Wizard

```bash
beyondbench
```

### Command Line

```bash
# Evaluate GPT-4o on the easy suite
beyondbench evaluate --model-id gpt-4o --api-provider openai --suite easy

# Evaluate a local model with vLLM
beyondbench evaluate --model-id meta-llama/Llama-3.2-3B-Instruct --backend vllm --suite all

# Evaluate Claude on hard tasks
beyondbench evaluate --model-id claude-sonnet-4-20250514 --api-provider anthropic --suite hard

# List all available tasks
beyondbench list-tasks
```

### Python API

```python
from beyondbench import EvaluationEngine, ModelHandler, TaskRegistry

# Initialize model handler
model = ModelHandler(
    model_id="gpt-4o",
    api_provider="openai",
    api_key="your-api-key"
)

# Run evaluation
engine = EvaluationEngine(model_handler=model, output_dir="./results")
results = engine.run_evaluation(suite="easy", datapoints=100)

# Print results
print(f"Average Accuracy: {results['summary']['avg_accuracy']:.2%}")
```

---

### New in v0.1.0

- **API Server**: `beyondbench serve` - FastAPI REST API
- **Config Files**: `beyondbench init` and `beyondbench run-config`
- **Results Viewer**: `beyondbench results list/show/compare`
- **Task Info**: `beyondbench info <task>`
- **CI/CD**: GitHub Actions for testing, linting, PyPI publishing
- **239 Tests**: Comprehensive test coverage

---

## Supported Backends

| Backend | Models | Features |
|---------|--------|----------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-5, GPT-5-mini | Reasoning effort control |
| **Gemini** | Gemini 2.5 Pro, Gemini 2.5 Flash | Thinking budget configuration |
| **Anthropic** | Claude Sonnet 4, Claude Opus 4 | Latest Claude models |
| **vLLM** | Any HuggingFace model | Batch processing, tensor parallelism |
| **Transformers** | Any HuggingFace model | CPU/GPU inference |

---

## Task Suites

### Easy Suite (29 Tasks)
Arithmetic (sum, multiplication, subtraction, division, absolute\_difference), Statistics (mean, median, mode), Counting (odd\_count, even\_count, count\_negative, count\_unique, and more), Extrema (find\_maximum, find\_minimum, second\_maximum, range, and more), Sequences (sorting, longest\_increasing\_subsequence, alternating\_sum, sum\_of\_digits), Comparison

### Medium Suite (5 Tasks, 49 Variations)
Fibonacci Sequence (6 variations), Algebraic Sequence (10), Geometric Sequence (10), Prime Sequence (11), Complex Pattern (12)

### Hard Suite (10 Tasks, 68 Variations)
Tower of Hanoi, N-Queens, Graph Coloring, Boolean SAT, Sudoku, Cryptarithmetic, Matrix Chain, Modular Systems, Constraint Optimization, Logic Grid Puzzles

---

## Leaderboard (Top 5)

| Rank | Model | Overall | Instruction Following |
|------|-------|---------|-----------------------|
| 1 | **GPT-5\*** | **83.56%** | 96.15% |
| 2 | **GPT-5-Nano\*** | **82.04%** | 93.58% |
| 3 | **GPT-5-Mini\*** | **81.67%** | 94.23% |
| 4 | **o3\*** | **80.36%** | 94.96% |
| 5 | **o4-Mini\*** | **79.04%** | 95.30% |

*\*Models use reasoning/thinking tokens. Full results for 101+ models on the [leaderboard](https://ctrl-gaurav.github.io/BeyondBench/).*

---

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Citation

If you use BeyondBench in your research, please cite our paper (accepted at **ICLR 2026**):

```bibtex
@misc{srivastava2025beyondbenchbenchmarkfreeevaluationreasoning,
      title={BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models},
      author={Gaurav Srivastava and Aafiya Hussain and Zhenyu Bi and Swastik Roy and Priya Pitre and Meng Lu and Morteza Ziyadi and Xuan Wang},
      year={2025},
      eprint={2509.24210},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.24210},
}
```

---

## Links

- **Leaderboard**: [ctrl-gaurav.github.io/BeyondBench](https://ctrl-gaurav.github.io/BeyondBench/)
- **GitHub**: [github.com/ctrl-gaurav/BeyondBench](https://github.com/ctrl-gaurav/BeyondBench)
- **Paper**: [arXiv:2509.24210](https://arxiv.org/abs/2509.24210)
- **Documentation**: [Full Docs](https://github.com/ctrl-gaurav/BeyondBench/blob/main/docs/DOCUMENTATION.md) | [Usage Guide](https://github.com/ctrl-gaurav/BeyondBench/blob/main/docs/USAGE.md)
- **Issues**: [GitHub Issues](https://github.com/ctrl-gaurav/BeyondBench/issues)
- **Email**: [gks@vt.edu](mailto:gks@vt.edu), [xuanw@vt.edu](mailto:xuanw@vt.edu)

---

**Made with care by the BeyondBench Team** | Virginia Tech, Department of Computer Science | Amazon AGI

*Advancing the frontier of AI reasoning evaluation, one benchmark at a time.*

**License**: Apache-2.0

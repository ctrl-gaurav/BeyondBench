"""
beyondbench: BeyondBench Evaluation Package
=====================================

A contamination-resistant evaluation framework for language model reasoning
capabilities across easy, medium, and hard computational tasks.

Key Features:
- 79 reasoning tasks (44 easy + 15 medium + 20 hard)
- 138 task variations across difficulty levels
- Multi-backend support (vLLM, Transformers, OpenAI, Gemini, Anthropic)
- Multi-GPU parallel evaluation engine
- Response caching and performance optimization
- Plugin system and custom task SDK
- Gradio dashboard for live monitoring
- Robust parsing and error handling
- Comprehensive metrics and reporting
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError
try:
    __version__ = _pkg_version("beyondbench")
except PackageNotFoundError:
    __version__ = "0.2.0"  # fallback for development

__author__ = "BeyondBench Team"
__email__ = "gks@vt.edu"
__license__ = "Apache-2.0"

from .core.evaluation_engine import EvaluationEngine
from .core.task_registry import TaskRegistry
from .models.model_handler import ModelHandler
from .cli.main import main as cli_main

__all__ = [
    "EvaluationEngine",
    "TaskRegistry",
    "ModelHandler",
    "cli_main"
]
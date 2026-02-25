"""
beyondbench: BeyondBench Evaluation Package
=====================================

A comprehensive evaluation framework for language model reasoning capabilities
across easy, medium, and hard computational tasks.

Key Features:
- 29 easy suite tasks with scalable complexity
- 5 medium suite tasks with 49 variations
- 10 hard suite tasks with 68 variations
- Multi-backend support (vLLM, Transformers, OpenAI, Gemini)
- Robust parsing and error handling
- Comprehensive metrics and reporting
"""

__version__ = "0.0.1"
__author__ = "BeyondBench Team"
__email__ = "contact@beyondbench.org"
__license__ = "MIT"

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
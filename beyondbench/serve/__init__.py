"""
BeyondBench FastAPI serve module.

Provides a REST API for running evaluations, listing tasks, and viewing results.
"""

from .app import create_app

__all__ = ["create_app"]

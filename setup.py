"""
Setup configuration for beyondbench package
"""

from setuptools import setup, find_packages
import os

def read_file(filename):
    """Read file contents."""
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

def get_requirements():
    """Get package requirements."""
    requirements = [
        # Core dependencies
        "torch>=2.5.0",
        "transformers>=4.47.0",
        "numpy>=2.0.0",
        "pandas>=2.2.0",
        "tqdm>=4.67.0",

        # Tokenization and utilities
        "tiktoken>=0.12.0",
        "sentencepiece>=0.2.0",

        # CLI and UI
        "click>=8.1.0",
        "rich>=14.0.0",
        "colorama>=0.4.6",

        # Data handling and analysis
        "scipy>=1.14.0",
        "tabulate>=0.9.0",

        # Configuration and logging
        "pyyaml>=6.0.0",
        "jsonschema>=4.23.0",

        # Performance
        "psutil>=6.0.0",
    ]

    return requirements

def get_extra_requirements():
    """Get optional dependencies."""
    return {
        "openai": ["openai>=2.0.0"],
        "gemini": ["google-genai>=1.0.0"],
        "anthropic": ["anthropic>=0.40.0"],
        "vllm": ["vllm>=0.7.0"],
        "all-apis": [
            "openai>=2.0.0",
            "google-genai>=1.0.0",
            "anthropic>=0.40.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.9.0",
            "seaborn>=0.13.0",
            "tabulate>=0.9.0",
        ],
        "full": [
            "openai>=2.0.0",
            "google-genai>=1.0.0",
            "anthropic>=0.40.0",
            "vllm>=0.7.0",
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=3.0.0",
            "matplotlib>=3.9.0",
            "seaborn>=0.13.0",
            "tabulate>=0.9.0",
        ],
    }

setup(
    name="beyondbench",
    version="0.0.1",
    description="BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="Gaurav Srivastava",
    author_email="gks@vt.edu",
    url="https://github.com/ctrl-gaurav/BeyondBench",
    license="MIT",

    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    package_data={
        "beyondbench": [
            "data/*.json",
            "configs/*.yaml",
            "templates/*.txt",
        ]
    },

    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require=get_extra_requirements(),

    entry_points={
        "console_scripts": [
            "beyondbench=beyondbench.cli.main:cli",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
    ],

    keywords=[
        "language-models",
        "evaluation",
        "benchmark",
        "reasoning",
        "nlp",
        "ai",
        "machine-learning",
        "llm-evaluation",
        "computational-reasoning",
        "contamination-resistant",
    ],

    project_urls={
        "Homepage": "https://github.com/ctrl-gaurav/BeyondBench",
        "Documentation": "https://github.com/ctrl-gaurav/BeyondBench#readme",
        "Repository": "https://github.com/ctrl-gaurav/BeyondBench",
        "Bug Tracker": "https://github.com/ctrl-gaurav/BeyondBench/issues",
        "Changelog": "https://github.com/ctrl-gaurav/BeyondBench/blob/main/CHANGELOG.md",
        "Paper": "https://arxiv.org/abs/2509.24210",
    },
)

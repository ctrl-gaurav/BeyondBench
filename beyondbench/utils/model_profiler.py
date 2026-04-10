"""
Model Behavior Profiler for BeyondBench.

Runs a calibration set of easy tasks against a loaded model and analyzes
output patterns: boxed format usage, explicit statement usage, code block
usage, chain-of-thought rate, average response length, and language mix.

Profiles are cached to ~/.beyondbench/profiles/<sanitized_model_id>.json
so subsequent evaluations can reuse them without re-running calibration.

Usage (standalone):
    python -m beyondbench.utils.model_profiler \
        --model-id Qwen/Qwen2.5-1.5B-Instruct \
        --output ~/.beyondbench/profiles/

Usage (programmatic):
    from beyondbench.utils.model_profiler import ModelProfiler
    profiler = ModelProfiler(model_handler)
    profile = profiler.profile()
    profiler.save(profile)
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Calibration prompts — 20 simple, unambiguous easy-suite questions
# ============================================================================

_CALIBRATION_PROMPTS: List[Dict[str, str]] = [
    {"prompt": "What is the sum of [3, 7, 2, 8, 5]? Give only the final answer.", "task": "sum"},
    {"prompt": "What is the sum of [10, 20, 30, 40]? Give only the final answer.", "task": "sum"},
    {"prompt": "What is 15 - 7? Give only the final answer.", "task": "subtraction"},
    {"prompt": "What is 8 × 6? Give only the final answer.", "task": "multiplication"},
    {"prompt": "What is 20 / 4? Give only the final answer.", "task": "division"},
    {"prompt": "What is the maximum of [4, 9, 2, 7, 1]? Give only the final answer.", "task": "max"},
    {"prompt": "What is the minimum of [4, 9, 2, 7, 1]? Give only the final answer.", "task": "min"},
    {"prompt": "How many even numbers are in [2, 5, 8, 3, 10, 7]? Give only the final answer.", "task": "even_count"},
    {"prompt": "How many odd numbers are in [1, 4, 7, 2, 9]? Give only the final answer.", "task": "odd_count"},
    {"prompt": "Sort [5, 2, 8, 1, 9, 3] in ascending order. Give only the sorted list.", "task": "sorting"},
    {"prompt": "What is the mean of [10, 20, 30]? Give only the final answer.", "task": "mean"},
    {"prompt": "What is the median of [3, 1, 4, 1, 5, 9, 2, 6]? Give only the final answer.", "task": "median"},
    {"prompt": "What is 3 × 12? Give only the final answer.", "task": "multiplication"},
    {"prompt": "What is the sum of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]? Give only the final answer.", "task": "sum"},
    {"prompt": "What is the absolute difference between 15 and 9? Give only the final answer.", "task": "abs_diff"},
    {"prompt": "What is 100 / 5? Give only the final answer.", "task": "division"},
    {"prompt": "How many negative numbers are in [-3, 5, -1, 7, -2, 4]? Give only the final answer.", "task": "count_neg"},
    {"prompt": "What is the second maximum of [3, 7, 1, 9, 5]? Give only the final answer.", "task": "second_max"},
    {"prompt": "What is 7 × 8? Give only the final answer.", "task": "multiplication"},
    {"prompt": "What is the range of [5, 2, 8, 1, 9, 3]? (range = max - min) Give only the final answer.", "task": "range"},
]

# ============================================================================
# ModelProfile dataclass
# ============================================================================

@dataclass
class ModelProfile:
    """
    Characterizes a model's output behavior for parsing optimization.

    All rate fields are proportions [0.0, 1.0] over the calibration set.
    """
    model_id: str
    # Format statistics
    boxed_rate: float = 0.0          # fraction using \\boxed{...}
    explicit_rate: float = 0.0       # fraction using "The answer is ..."
    code_block_rate: float = 0.0     # fraction wrapping answer in ```...```
    cot_rate: float = 0.0            # fraction with chain-of-thought preamble
    # Length statistics
    avg_response_length: float = 0.0 # average character length
    avg_token_estimate: float = 0.0  # estimated tokens (~4 chars/token)
    # Language mix
    non_latin_rate: float = 0.0      # fraction of responses with non-Latin chars
    # Recommended parser strategy order (derived from above stats)
    recommended_strategies: List[str] = field(default_factory=list)
    # Known quirks (free-text notes for debugging)
    quirks: List[str] = field(default_factory=list)
    # Metadata
    num_calibration_samples: int = 0
    calibration_timestamp: float = 0.0
    profile_version: str = "1.0"


# ============================================================================
# Detection helpers
# ============================================================================

_BOXED_RE = re.compile(r'\\boxed\s*\{', re.IGNORECASE)
_EXPLICIT_RE = re.compile(
    r'\b(?:the answer is|the result is|therefore[,:]|so[,:]|hence[,:]|thus[,:]|'
    r'answer[:\s]+|final answer[:\s]+)',
    re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r'```', re.IGNORECASE)
_COT_RE = re.compile(
    r'^(?:let me|to solve|first,|step 1|i\'ll|we need to|we can|we start|'
    r'we have|note that|observe that|consider|let\'s)',
    re.IGNORECASE | re.MULTILINE,
)


def _has_non_latin(text: str) -> bool:
    """Return True if text contains substantial non-Latin Unicode characters."""
    count = 0
    for ch in text:
        if ord(ch) > 0x024F and not unicodedata.category(ch).startswith('P'):
            count += 1
            if count >= 3:
                return True
    return False


def _derive_strategies(profile: ModelProfile) -> List[str]:
    """
    Derive recommended parser strategy order from a ModelProfile.

    Higher-frequency formats come first so the parser finds answers faster.
    """
    scored: List[tuple] = [
        ("boxed",              profile.boxed_rate),
        ("explicit_statement", profile.explicit_rate),
        ("code_block",         profile.code_block_rate),
        ("latex_math",         0.3),   # always reasonable, medium priority
        ("fallback",           0.1),   # always last
    ]
    # Sort descending by score, keeping fallback last
    non_fallback = [(s, sc) for s, sc in scored if s != "fallback"]
    non_fallback.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in non_fallback] + ["fallback"]


# ============================================================================
# ModelProfiler
# ============================================================================

class ModelProfiler:
    """
    Run calibration prompts against a model and produce a ModelProfile.

    The profiler is backend-agnostic: it accepts any object that has a
    ``generate(prompt: str) -> str`` method.  The ModelHandler class from
    beyondbench.models.model_handler satisfies this interface.
    """

    def __init__(self, model_handler, model_id: Optional[str] = None):
        """
        Args:
            model_handler: Object with .generate(prompt) -> str  or
                           ModelHandler with .generate_response(prompt, ...) -> str.
            model_id: Override the model identifier string.  If None and the
                      handler has a model_id attribute, that is used.
        """
        self.handler = model_handler
        self.model_id = (
            model_id
            or getattr(model_handler, 'model_id', None)
            or "unknown_model"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def profile(
        self,
        prompts: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
    ) -> ModelProfile:
        """
        Run calibration and return a ModelProfile.

        Args:
            prompts: List of {"prompt": ..., "task": ...} dicts.
                     Defaults to the 20 built-in calibration prompts.
            max_tokens: Token budget per calibration prompt.

        Returns:
            A populated ModelProfile.
        """
        if prompts is None:
            prompts = _CALIBRATION_PROMPTS

        responses: List[str] = []
        for item in prompts:
            try:
                resp = self._generate(item["prompt"], max_tokens=max_tokens)
                responses.append(resp)
            except Exception as exc:
                logger.warning("Calibration prompt failed: %s", exc)
                responses.append("")

        return self._analyze(responses)

    def save(self, profile: ModelProfile, profile_dir: Optional[str] = None) -> Path:
        """
        Persist profile to disk as JSON.

        Args:
            profile: The ModelProfile to save.
            profile_dir: Directory path.  Defaults to ~/.beyondbench/profiles/.

        Returns:
            Path to the saved JSON file.
        """
        if profile_dir is None:
            profile_dir = Path.home() / ".beyondbench" / "profiles"
        else:
            profile_dir = Path(profile_dir)

        profile_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _sanitize_model_id(profile.model_id)
        out_path = profile_dir / f"{safe_name}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2, ensure_ascii=False)

        logger.info("Profile saved to %s", out_path)
        return out_path

    @staticmethod
    def load(model_id: str, profile_dir: Optional[str] = None) -> Optional[ModelProfile]:
        """
        Load a cached profile from disk.

        Args:
            model_id: Model identifier string.
            profile_dir: Directory to search.  Defaults to ~/.beyondbench/profiles/.

        Returns:
            ModelProfile or None if not found.
        """
        if profile_dir is None:
            profile_dir = Path.home() / ".beyondbench" / "profiles"
        else:
            profile_dir = Path(profile_dir)

        safe_name = _sanitize_model_id(model_id)
        path = profile_dir / f"{safe_name}.json"

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ModelProfile(**data)
        except Exception as exc:
            logger.warning("Failed to load profile %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Call the underlying model handler and return the text response."""
        handler = self.handler

        # ModelHandler.generate(prompts: List[str], ...) -> List[str]
        # Also supports simple .generate(prompt) -> str or
        # .generate_response(prompt, ...) -> str for mock/custom handlers.
        if hasattr(handler, 'generate_response'):
            return handler.generate_response(prompt, max_new_tokens=max_tokens) or ""

        if hasattr(handler, 'generate'):
            result = handler.generate([prompt], max_tokens=max_tokens)
            if isinstance(result, list):
                return result[0] if result else ""
            return result or ""

        raise AttributeError(
            f"Model handler {type(handler).__name__} has no generate or generate_response method."
        )

    def _analyze(self, responses: List[str]) -> ModelProfile:
        """Analyze a list of raw responses and build a ModelProfile."""
        n = len(responses)
        if n == 0:
            return ModelProfile(model_id=self.model_id)

        boxed_count = 0
        explicit_count = 0
        code_block_count = 0
        cot_count = 0
        non_latin_count = 0
        total_length = 0

        for resp in responses:
            if not resp:
                continue
            total_length += len(resp)
            if _BOXED_RE.search(resp):
                boxed_count += 1
            if _EXPLICIT_RE.search(resp):
                explicit_count += 1
            if _CODE_BLOCK_RE.search(resp):
                code_block_count += 1
            if _COT_RE.search(resp):
                cot_count += 1
            if _has_non_latin(resp):
                non_latin_count += 1

        avg_len = total_length / n

        profile = ModelProfile(
            model_id=self.model_id,
            boxed_rate=boxed_count / n,
            explicit_rate=explicit_count / n,
            code_block_rate=code_block_count / n,
            cot_rate=cot_count / n,
            avg_response_length=avg_len,
            avg_token_estimate=avg_len / 4.0,
            non_latin_rate=non_latin_count / n,
            num_calibration_samples=n,
            calibration_timestamp=time.time(),
            profile_version="1.0",
        )

        profile.recommended_strategies = _derive_strategies(profile)
        profile.quirks = _detect_quirks(profile, self.model_id)
        return profile


# ============================================================================
# Quirk detection
# ============================================================================

def _detect_quirks(profile: ModelProfile, model_id: str) -> List[str]:
    quirks: List[str] = []
    mid = model_id.lower()

    if profile.non_latin_rate > 0.2:
        quirks.append("Outputs contain non-Latin characters — enable multilingual number extraction.")

    if profile.cot_rate > 0.7:
        quirks.append("High chain-of-thought rate — consider stripping CoT preamble before parsing.")

    if profile.avg_response_length > 800:
        quirks.append("Very long responses (>800 chars avg) — fallback strategy may match noise; boxed preferred.")

    if profile.avg_response_length < 20:
        quirks.append("Very short responses (<20 chars avg) — model outputs terse single-token answers.")

    if "qwen" in mid:
        quirks.append("Qwen family: may mix Chinese characters with numeric answers.")

    if "llama" in mid or "meta-llama" in mid:
        quirks.append("Llama family: verbose CoT common; [INST] artifacts possible in older checkpoints.")

    if "phi" in mid:
        quirks.append("Phi family: terse outputs; single-number answers common.")

    if "mistral" in mid or "mixtral" in mid:
        quirks.append("Mistral family: [INST]/[/INST] artifacts possible.")

    return quirks


# ============================================================================
# Utilities
# ============================================================================

def _sanitize_model_id(model_id: str) -> str:
    """Convert a model_id (possibly a HuggingFace path) to a safe filename."""
    return re.sub(r'[^\w\-.]', '_', model_id.lower()).strip('_')


# ============================================================================
# CLI entry-point
# ============================================================================

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile a model's output behavior for BeyondBench adaptive parsing."
    )
    parser.add_argument("--model-id", required=True, help="HuggingFace model path or API model name")
    parser.add_argument("--output", default=None, help="Output directory for profiles (default: ~/.beyondbench/profiles/)")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"], help="Local backend")
    parser.add_argument("--gpu", default="0", help="CUDA device index")
    parser.add_argument("--force", action="store_true", help="Re-profile even if cached profile exists")
    args = parser.parse_args()

    # Check cache first
    if not args.force:
        cached = ModelProfiler.load(args.model_id, args.output)
        if cached is not None:
            print(f"[profiler] Loaded cached profile for {args.model_id}")
            _print_profile(cached)
            return

    print(f"[profiler] Profiling {args.model_id} on GPU {args.gpu} ...")

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    from beyondbench.models.model_handler import ModelHandler
    handler = ModelHandler(
        model_id=args.model_id,
        backend=args.backend,
        cuda_device=f"cuda:{args.gpu}",
    )

    profiler = ModelProfiler(handler, model_id=args.model_id)
    profile = profiler.profile()
    profiler.save(profile, args.output)
    _print_profile(profile)


def _print_profile(profile: ModelProfile) -> None:
    print(f"\n=== Model Profile: {profile.model_id} ===")
    print(f"  Boxed rate:          {profile.boxed_rate:.1%}")
    print(f"  Explicit stmt rate:  {profile.explicit_rate:.1%}")
    print(f"  Code block rate:     {profile.code_block_rate:.1%}")
    print(f"  CoT rate:            {profile.cot_rate:.1%}")
    print(f"  Non-Latin rate:      {profile.non_latin_rate:.1%}")
    print(f"  Avg response length: {profile.avg_response_length:.0f} chars")
    print(f"  Recommended order:   {profile.recommended_strategies}")
    if profile.quirks:
        print("  Quirks:")
        for q in profile.quirks:
            print(f"    - {q}")
    print()


if __name__ == "__main__":
    _cli()

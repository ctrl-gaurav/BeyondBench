"""
Pre-evaluation model validation (Phase 9.2.1).

Runs a series of cheap checks BEFORE an evaluation kicks off so users get
fast, actionable feedback instead of mysterious runtime failures:

  1. Model identifier — HuggingFace existence check via HEAD request
     (skipped for API models where the provider handles resolution).
  2. Backend dependencies — is `vllm` importable for backend="vllm",
     `transformers` importable for backend="transformers", the API SDK
     importable + API key set for api_provider="openai"/"gemini"/"anthropic".
  3. GPU memory — rough size-based check that the target CUDA device has
     enough free memory for the model. Only applies to local backends.
  4. Optimal settings — rule-based temperature / max_tokens suggestions
     keyed off model family + size, plus any known quirks from the
     built-in model profiles.

Every check produces a :class:`ValidationIssue` with severity
``"error"`` (fail-fast), ``"warning"`` (log + continue), or ``"info"``
(suggestion only). The validator never raises on its own — callers
decide whether to enforce errors via :meth:`ModelValidator.enforce`.

Example
-------
>>> from beyondbench.configs.model_validator import ModelValidator
>>> v = ModelValidator()
>>> result = v.validate({"model": {"model_id": "Qwen/Qwen2.5-3B-Instruct",
...                                 "backend": "vllm",
...                                 "cuda_device": "cuda:0"}})
>>> result.ok       # True if no "error" severity issues
True
>>> for issue in result.issues:
...     print(issue.severity, issue.check, issue.message)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single finding from a validation check."""
    check: str
    severity: str       # "error" | "warning" | "info"
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Aggregated result of running all validation checks."""
    issues: List[ValidationIssue] = field(default_factory=list)
    suggested_settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when there are no ``error``-severity issues."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "info"]

    def add(
        self,
        check: str,
        severity: str,
        message: str,
        suggestion: Optional[str] = None,
    ) -> None:
        self.issues.append(ValidationIssue(check, severity, message, suggestion))


# ---------------------------------------------------------------------------
# Size heuristics
# ---------------------------------------------------------------------------

# Very rough VRAM footprint (GiB) for a model loaded in fp16/bf16 at the
# given parameter count. Real vLLM usage depends on KV-cache sizing, context
# length, and quantization — this is a ballpark to catch obvious misuse
# ("trying to run 70B on a 24GB card") rather than a precise estimator.
# Formula: params * 2 bytes (fp16) + ~20% overhead for activations/KV.
_PARAM_TO_VRAM_GIB = 2.4  # GiB per billion params

_SIZE_REGEX = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)\s*[bB](?![a-zA-Z])")


def _extract_model_size_billions(model_id: str) -> Optional[float]:
    """Extract parameter count in billions from a model_id.

    Matches patterns like "Llama-3.2-3B", "qwen2.5-1.5b", "mixtral-8x7B"
    (returns 56 for 8x7 since 8*7=56, the effective param count).
    Returns None when no size can be inferred.
    """
    # Handle "NxMb" mixture-of-experts notation first
    moe_match = re.search(r"(\d+)x(\d+(?:\.\d+)?)\s*[bB]", model_id)
    if moe_match:
        n, m = moe_match.groups()
        try:
            return float(n) * float(m)
        except ValueError:
            pass

    m = _SIZE_REGEX.search(model_id)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def estimate_vram_gib(model_id: str) -> Optional[float]:
    """Rough upper-bound VRAM estimate (GiB) for loading ``model_id`` in bf16.

    Returns None when the model size can't be inferred from the identifier.
    """
    size_b = _extract_model_size_billions(model_id)
    if size_b is None:
        return None
    return round(size_b * _PARAM_TO_VRAM_GIB, 2)


# ---------------------------------------------------------------------------
# GPU probe
# ---------------------------------------------------------------------------

def _parse_cuda_device(cuda_device: Optional[str]) -> Optional[int]:
    """Parse a ``"cuda:N"`` string to an integer index."""
    if not cuda_device:
        return None
    m = re.match(r"cuda:(\d+)", str(cuda_device).strip())
    return int(m.group(1)) if m else None


def _query_gpu_free_memory_mib() -> Dict[int, Tuple[int, int]]:
    """Return ``{gpu_index: (total_mib, free_mib)}`` via ``nvidia-smi``.

    Empty dict on failure; the validator treats that as "skip GPU check".
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {}

    result: Dict[int, Tuple[int, int]] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[0])
            total_mib = int(parts[1])
            free_mib = int(parts[2])
        except ValueError:
            continue
        result[idx] = (total_mib, free_mib)
    return result


# ---------------------------------------------------------------------------
# Optimal-settings heuristics
# ---------------------------------------------------------------------------

# Family-level defaults. Keys are substrings checked against the lower-cased
# model_id; first match wins. Keeps this file self-contained rather than
# coupling to the UnifiedParser model profiles (which focus on parsing, not
# sampling hyperparameters).
_FAMILY_HINTS: List[Tuple[str, Dict[str, Any]]] = [
    ("gpt-5", {"temperature": 0.0, "max_tokens": 8192,
               "note": "GPT-5 reasoning models ignore temperature — use reasoning_effort."}),
    ("gpt-4", {"temperature": 0.0, "max_tokens": 4096,
               "note": "OpenAI deterministic evaluation: temperature=0, moderate max_tokens."}),
    ("o1",    {"temperature": 1.0, "max_tokens": 8192,
               "note": "o-series reasoning models: leave temperature at 1.0 (required)."}),
    ("o3",    {"temperature": 1.0, "max_tokens": 8192,
               "note": "o-series reasoning models: leave temperature at 1.0 (required)."}),
    ("o4",    {"temperature": 1.0, "max_tokens": 8192,
               "note": "o-series reasoning models: leave temperature at 1.0 (required)."}),
    ("gemini-2.5", {"temperature": 0.1, "max_tokens": 8192,
                    "note": "Gemini 2.5 Flash/Pro: low temperature works best for math."}),
    ("gemini",     {"temperature": 0.1, "max_tokens": 4096}),
    ("claude",     {"temperature": 0.0, "max_tokens": 8192}),
    ("qwen2.5-0.5b", {"temperature": 0.1, "max_tokens": 2048}),
    ("qwen2.5-1.5b", {"temperature": 0.1, "max_tokens": 4096}),
    ("qwen2.5-3b",   {"temperature": 0.1, "max_tokens": 4096}),
    ("qwen2.5-7b",   {"temperature": 0.1, "max_tokens": 8192}),
    ("qwen",         {"temperature": 0.1, "max_tokens": 4096}),
    ("llama-3.2-1b", {"temperature": 0.1, "max_tokens": 2048}),
    ("llama-3.2-3b", {"temperature": 0.1, "max_tokens": 4096}),
    ("llama",        {"temperature": 0.1, "max_tokens": 4096}),
    ("mistral",      {"temperature": 0.1, "max_tokens": 4096}),
    ("phi-3",        {"temperature": 0.1, "max_tokens": 2048}),
    ("gemma",        {"temperature": 0.1, "max_tokens": 2048}),
]


def suggest_optimal_settings(model_id: str) -> Dict[str, Any]:
    """Return rule-based suggested sampling settings for ``model_id``.

    Always includes at least ``temperature`` and ``max_tokens``. Falls back
    to a conservative default when no family match is found.
    """
    model_lc = model_id.lower()
    for key, hints in _FAMILY_HINTS:
        if key in model_lc:
            return dict(hints)
    return {
        "temperature": 0.1,
        "max_tokens": 4096,
        "note": "No family profile matched — using conservative defaults.",
    }


# ---------------------------------------------------------------------------
# Model validator
# ---------------------------------------------------------------------------

API_BACKENDS = {"openai", "gemini", "anthropic"}
LOCAL_BACKENDS = {"vllm", "transformers"}


class ModelValidator:
    """Runs pre-evaluation validation checks against a config dict.

    The validator is deliberately offline-friendly: HuggingFace hub checks
    require network and are skipped (with a ``warning``) when unreachable.
    GPU checks are skipped when ``nvidia-smi`` is unavailable.
    """

    def __init__(
        self,
        *,
        check_hf_existence: bool = True,
        check_gpu_memory: bool = True,
        check_dependencies: bool = True,
        suggest_settings: bool = True,
        network_timeout: float = 5.0,
    ):
        self.check_hf_existence = check_hf_existence
        self.check_gpu_memory = check_gpu_memory
        self.check_dependencies = check_dependencies
        self.suggest_settings = suggest_settings
        self.network_timeout = network_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Run all enabled checks against ``config`` and return a result.

        ``config`` is the nested dict produced by
        :func:`beyondbench.configs.load_config`. Missing sub-sections are
        tolerated — required fields surface as ``error`` issues instead of
        raising.
        """
        result = ValidationResult()
        model_cfg = config.get("model") or {}
        model_id = model_cfg.get("model_id")

        if not model_id:
            result.add(
                "model.model_id",
                "error",
                "model.model_id is required but missing/empty.",
                suggestion="Set model.model_id in your config or pass --model-id.",
            )
            return result  # can't run any other check without an id

        backend = (model_cfg.get("backend") or "").lower()
        api_provider = (model_cfg.get("api_provider") or "").lower()

        # Prefer explicit api_provider over backend for routing decisions
        effective_backend = api_provider or backend or self._guess_backend(model_id)

        # 1. Dependency check
        if self.check_dependencies:
            self._check_dependencies(effective_backend, result)

        # 2. HF existence (local only)
        if self.check_hf_existence and effective_backend in LOCAL_BACKENDS:
            self._check_hf_model_exists(model_id, result)

        # 3. GPU memory (local only)
        if self.check_gpu_memory and effective_backend in LOCAL_BACKENDS:
            cuda_device = model_cfg.get("cuda_device", "cuda:0")
            gpu_mem_util = float(model_cfg.get("gpu_memory_utilization", 0.96) or 0.96)
            self._check_gpu_memory(model_id, cuda_device, gpu_mem_util, result)

        # 4. API key check for API backends
        if effective_backend in API_BACKENDS:
            self._check_api_key(effective_backend, model_cfg, result)

        # 5. Optimal settings suggestion
        if self.suggest_settings:
            result.suggested_settings = suggest_optimal_settings(model_id)
            eval_cfg = config.get("evaluation") or {}
            self._compare_suggestions(eval_cfg, result.suggested_settings, result)

        return result

    def enforce(self, result: ValidationResult) -> None:
        """Raise ``ValueError`` if ``result`` contains any error-severity issues.

        The raised error contains a bullet list of every error and is safe to
        surface directly to the user.
        """
        if result.ok:
            return
        bullets = "\n".join(f"  - [{i.check}] {i.message}" for i in result.errors)
        raise ValueError(
            f"Model validation failed with {len(result.errors)} error(s):\n{bullets}"
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _guess_backend(self, model_id: str) -> str:
        """Guess a backend from an API-style model_id; defaults to ``vllm``."""
        mid = model_id.lower()
        if mid.startswith("gpt-") or "o1" in mid or "o3" in mid or "o4" in mid:
            return "openai"
        if mid.startswith("gemini"):
            return "gemini"
        if mid.startswith("claude"):
            return "anthropic"
        return "vllm"

    def _check_dependencies(self, backend: str, result: ValidationResult) -> None:
        """Verify the Python package(s) for the chosen backend are importable."""
        required = {
            "vllm":         [("vllm",          "pip install 'beyondbench[vllm]'")],
            "transformers": [("transformers",  "pip install transformers")],
            "openai":       [("openai",        "pip install 'beyondbench[openai]'")],
            "gemini":       [("google.genai",  "pip install 'beyondbench[gemini]'")],
            "anthropic":    [("anthropic",     "pip install 'beyondbench[anthropic]'")],
        }.get(backend, [])

        for mod_name, install_hint in required:
            try:
                __import__(mod_name)
            except ImportError:
                result.add(
                    f"dependency.{backend}",
                    "error",
                    f"Backend '{backend}' requires `{mod_name}` but it is not installed.",
                    suggestion=install_hint,
                )

    def _check_hf_model_exists(self, model_id: str, result: ValidationResult) -> None:
        """HEAD ``https://huggingface.co/<model_id>`` to verify the repo exists.

        Skipped with a ``warning`` severity note when we can't reach HF (no
        network in CI, firewall, etc.). A 404 response becomes an ``error``
        — that's a typo the user should fix before wasting GPU time.
        """
        try:
            from urllib.request import Request, urlopen
            from urllib.error import HTTPError, URLError
        except ImportError:
            return

        # Detect local paths and treat them as already-validated on-disk models.
        # HuggingFace IDs are "user/repo" (single slash, no leading . / ~); true
        # local paths either start with /, ./, or ~, or contain multiple path
        # separators (e.g. "/home/me/models/qwen").
        is_local_path = (
            model_id.startswith((os.sep, "./", "../", "~"))
            or model_id.count(os.sep) > 1
        )
        if is_local_path:
            if os.path.isdir(os.path.expanduser(model_id)):
                return
            result.add(
                "model.model_id",
                "error",
                f"Local path does not exist: {model_id}",
                suggestion="Check the path or pass a HuggingFace model id.",
            )
            return

        url = f"https://huggingface.co/{model_id}"
        req = Request(url, method="HEAD", headers={"User-Agent": "beyondbench/validator"})
        try:
            with urlopen(req, timeout=self.network_timeout) as resp:
                code = resp.getcode()
                if 200 <= code < 400:
                    return
                result.add(
                    "huggingface.exists",
                    "warning",
                    f"Unexpected HTTP status {code} for {url}",
                )
        except HTTPError as e:
            if e.code == 404:
                result.add(
                    "huggingface.exists",
                    "error",
                    f"Model '{model_id}' was not found on HuggingFace (HTTP 404).",
                    suggestion="Double-check spelling, or for gated models run `huggingface-cli login`.",
                )
            elif e.code in (401, 403):
                result.add(
                    "huggingface.exists",
                    "warning",
                    f"Access denied for '{model_id}' (HTTP {e.code}) — likely a gated model.",
                    suggestion="Run `huggingface-cli login` and accept the model license.",
                )
            else:
                result.add(
                    "huggingface.exists",
                    "warning",
                    f"HTTP {e.code} checking {url}: {e.reason}",
                )
        except URLError as e:
            # Network unreachable — downgrade to info so offline CI stays green
            result.add(
                "huggingface.exists",
                "info",
                f"Skipping HF existence check: network unreachable ({e.reason}).",
            )
        except Exception as e:  # pragma: no cover — defensive
            result.add(
                "huggingface.exists",
                "info",
                f"Skipping HF existence check: {e}",
            )

    def _check_gpu_memory(
        self,
        model_id: str,
        cuda_device: str,
        gpu_mem_util: float,
        result: ValidationResult,
    ) -> None:
        """Compare estimated model VRAM footprint against free GPU memory."""
        estimated = estimate_vram_gib(model_id)
        if estimated is None:
            return  # silently skip when we can't estimate

        gpu_info = _query_gpu_free_memory_mib()
        if not gpu_info:
            result.add(
                "gpu.memory",
                "info",
                "Skipping GPU memory check: nvidia-smi unavailable.",
            )
            return

        idx = _parse_cuda_device(cuda_device)
        if idx is None or idx not in gpu_info:
            result.add(
                "gpu.memory",
                "warning",
                f"Requested device '{cuda_device}' not found by nvidia-smi "
                f"(visible indices: {sorted(gpu_info)}).",
                suggestion="Set cuda_device to a valid 'cuda:N' for a visible GPU.",
            )
            return

        total_mib, free_mib = gpu_info[idx]
        free_gib = free_mib / 1024
        available_gib = free_gib * gpu_mem_util

        if estimated > available_gib:
            result.add(
                "gpu.memory",
                "error",
                f"Model '{model_id}' needs ~{estimated:.1f} GiB of VRAM but "
                f"GPU {idx} has {free_gib:.1f} GiB free "
                f"({available_gib:.1f} GiB usable at gpu_memory_utilization={gpu_mem_util:.2f}).",
                suggestion=(
                    "Use a smaller model, lower gpu_memory_utilization, enable "
                    "tensor_parallel_size>1 across multiple GPUs, or switch to "
                    "backend=transformers with 4/8-bit quantization."
                ),
            )
        elif estimated > 0.85 * available_gib:
            result.add(
                "gpu.memory",
                "warning",
                f"Model '{model_id}' (~{estimated:.1f} GiB) will use >85% of "
                f"available VRAM on GPU {idx} ({available_gib:.1f} GiB usable). "
                "KV-cache may be tight.",
            )

    def _check_api_key(
        self,
        backend: str,
        model_cfg: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Ensure an API key is set in the config or environment."""
        if model_cfg.get("api_key"):
            return

        env_vars = {
            "openai":    ["OPENAI_API_KEY"],
            "gemini":    ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
        }.get(backend, [])

        if any(os.environ.get(v) for v in env_vars):
            return

        result.add(
            f"api_key.{backend}",
            "error",
            f"No API key found for '{backend}'.",
            suggestion=(
                f"Set {' or '.join(env_vars)} in the environment, "
                f"or pass --api-key on the CLI."
            ),
        )

    def _compare_suggestions(
        self,
        eval_cfg: Dict[str, Any],
        suggestions: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Emit info-level notes when user settings diverge from suggestions."""
        if not suggestions:
            return

        note = suggestions.get("note")
        if note:
            result.add("suggestions.note", "info", note)

        user_temp = eval_cfg.get("temperature")
        sugg_temp = suggestions.get("temperature")
        if (
            user_temp is not None
            and sugg_temp is not None
            and abs(float(user_temp) - float(sugg_temp)) > 0.25
        ):
            result.add(
                "suggestions.temperature",
                "info",
                f"temperature={user_temp} is far from recommended {sugg_temp} "
                "for this model family.",
                suggestion=f"Consider evaluation.temperature={sugg_temp}.",
            )

        user_max = eval_cfg.get("max_tokens")
        sugg_max = suggestions.get("max_tokens")
        if (
            user_max is not None
            and sugg_max is not None
            and user_max < sugg_max // 4
        ):
            result.add(
                "suggestions.max_tokens",
                "info",
                f"max_tokens={user_max} is well below the recommended {sugg_max} "
                "for this model family; responses may get truncated.",
                suggestion=f"Consider evaluation.max_tokens={sugg_max}.",
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def validate_model_config(
    config: Dict[str, Any],
    *,
    check_hf_existence: bool = True,
    check_gpu_memory: bool = True,
    check_dependencies: bool = True,
    suggest_settings: bool = True,
) -> ValidationResult:
    """One-shot helper mirroring :meth:`ModelValidator.validate`."""
    return ModelValidator(
        check_hf_existence=check_hf_existence,
        check_gpu_memory=check_gpu_memory,
        check_dependencies=check_dependencies,
        suggest_settings=suggest_settings,
    ).validate(config)


__all__ = [
    "ModelValidator",
    "ValidationIssue",
    "ValidationResult",
    "validate_model_config",
    "estimate_vram_gib",
    "suggest_optimal_settings",
]

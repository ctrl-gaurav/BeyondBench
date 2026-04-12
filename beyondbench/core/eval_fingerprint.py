"""Evaluation fingerprint for reproducibility verification."""

import hashlib
import json
import platform
import sys
from typing import Any, Dict, List, Optional


def compute_eval_fingerprint(
    model_id: str,
    config: Dict[str, Any],
    seed: Optional[int],
    task_list: List[str],
    beyondbench_version: str,
) -> str:
    """Compute a unique fingerprint for an evaluation configuration.

    The fingerprint is a hash of all parameters that affect evaluation results.
    Two evaluations with the same fingerprint should produce identical results
    (assuming the model is deterministic).

    Args:
        model_id: Model identifier string
        config: Evaluation configuration dict (temperature, top_p, max_tokens, etc.)
        seed: Random seed
        task_list: Ordered list of task names being evaluated
        beyondbench_version: Package version string

    Returns:
        32-char hex fingerprint string
    """
    # Only include config keys that affect results
    relevant_keys = [
        'temperature', 'top_p', 'max_tokens', 'datapoints', 'folds',
        'range_min', 'range_max', 'list_sizes', 'prompt_style',
        'contamination_resistance',
    ]
    filtered_config = {k: config.get(k) for k in relevant_keys if config.get(k) is not None}

    payload = {
        "model_id": model_id,
        "config": filtered_config,
        "seed": seed,
        "tasks": sorted(task_list),
        "version": beyondbench_version,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def get_environment_info() -> Dict[str, str]:
    """Collect environment info for result versioning.

    Returns dict with:
    - beyondbench_version
    - python_version
    - platform
    - torch_version (if available)
    - transformers_version (if available)
    - vllm_version (if available)
    - numpy_version
    """
    from importlib.metadata import version as pkg_version, PackageNotFoundError

    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # Git hash
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["git_hash"] = result.stdout.strip()
    except Exception:
        pass

    # Hardware info (GPU)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            info["gpu_count"] = str(gpu_count)
            gpus = []
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory
                gpus.append(f"{name} ({mem / 1024**3:.1f}GB)")
            info["gpus"] = gpus
    except Exception:
        pass

    # beyondbench
    try:
        info["beyondbench_version"] = pkg_version("beyondbench")
    except PackageNotFoundError:
        try:
            from beyondbench import __version__
            info["beyondbench_version"] = __version__
        except ImportError:
            info["beyondbench_version"] = "unknown"

    # Optional packages
    for pkg_name in ["torch", "transformers", "vllm", "numpy"]:
        try:
            info[f"{pkg_name}_version"] = pkg_version(pkg_name)
        except PackageNotFoundError:
            info[f"{pkg_name}_version"] = "not installed"

    return info


def check_version_compatibility(
    result_env: Dict[str, str],
    current_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Check if two environments are compatible for result comparison.

    Args:
        result_env: Environment info from saved results
        current_env: Current environment info (computed if None)

    Returns:
        List of warning messages (empty = compatible)
    """
    if current_env is None:
        current_env = get_environment_info()

    warnings = []

    # Check beyondbench version
    v1 = result_env.get("beyondbench_version", "unknown")
    v2 = current_env.get("beyondbench_version", "unknown")
    if v1 != v2 and v1 != "unknown" and v2 != "unknown":
        warnings.append(f"BeyondBench version mismatch: results={v1}, current={v2}")

    # Check Python version (major.minor)
    py1 = ".".join(result_env.get("python_version", "0.0").split(".")[:2])
    py2 = ".".join(current_env.get("python_version", "0.0").split(".")[:2])
    if py1 != py2:
        warnings.append(f"Python version mismatch: results={py1}, current={py2}")

    # Check torch version
    t1 = result_env.get("torch_version", "")
    t2 = current_env.get("torch_version", "")
    if t1 != t2 and t1 and t2 and t1 != "not installed" and t2 != "not installed":
        warnings.append(f"PyTorch version mismatch: results={t1}, current={t2}")

    return warnings

"""Instance fingerprinting for contamination resistance."""
import hashlib
import json
from typing import Any


def fingerprint_instance(data_point: Any, task_name: str, seed: int = None) -> str:
    """Generate a unique SHA-256 fingerprint for a generated data point.

    Args:
        data_point: The generated evaluation data (list, dict, etc.)
        task_name: Name of the task that generated it
        seed: The seed used to generate it

    Returns:
        Hex digest string (first 16 chars for brevity)
    """
    payload = {
        "task": task_name,
        "data": _serialize(data_point),
        "seed": seed,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _serialize(obj):
    """Recursively serialize an object for hashing."""
    # Handle numpy arrays before generic checks
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass

    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, set):
        return sorted([_serialize(item) for item in obj], key=str)
    # Fallback: stringify
    return str(obj)

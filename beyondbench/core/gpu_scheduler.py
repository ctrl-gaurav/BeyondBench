"""
GPU scheduler for BeyondBench parallel evaluation.

Discovers available GPUs, tracks memory, and allocates GPUs to workers.
Uses nvidia-smi for real-time memory queries.
"""

import os
import subprocess
import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GPUScheduler")

# VRAM requirements at fp16 (conservative estimates for vLLM with gpu_memory_utilization=0.45)
# Format: (param_billion_threshold, required_gb)
_MODEL_VRAM_TABLE = [
    (0.6,  3),   # 0.5B  model → 3 GB
    (1.8,  5),   # 1.5B  model → 5 GB
    (3.5,  8),   # 3B    model → 8 GB
    (8.0,  16),  # 7B    model → 16 GB
    (14.0, 28),  # 13B   model → 28 GB
    (35.0, 40),  # 30B   model → 40 GB
]
_DEFAULT_VRAM_GB = 20  # fallback when size cannot be estimated


def _query_gpu_memory() -> Dict[int, Tuple[int, int]]:
    """Query GPU memory via nvidia-smi.

    Returns:
        {gpu_id: (total_mb, free_mb)}
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
        result: Dict[int, Tuple[int, int]] = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                idx, total, free = int(parts[0]), int(parts[1]), int(parts[2])
                result[idx] = (total, free)
        return result
    except Exception as exc:
        logger.warning(f"nvidia-smi query failed: {exc}")
        return {}


def estimate_model_vram(model_id: str) -> int:
    """Estimate VRAM requirement in GB from model_id.

    Parses parameter count from names like:
      Qwen2.5-1.5B-Instruct → 1.5B → 5 GB
      meta-llama/Llama-3.2-3B → 3B → 8 GB
    """
    import re
    # Match patterns like 0.5B, 1.5B, 3B, 7B, 13B, 30B, 70B
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_id)
    if match:
        params = float(match.group(1))
        for threshold, vram in _MODEL_VRAM_TABLE:
            if params <= threshold:
                return vram
        return _DEFAULT_VRAM_GB
    return _DEFAULT_VRAM_GB


class GPUScheduler:
    """Thread-safe GPU allocator.

    Tracks which GPUs are allocated and their current free memory.
    Memory is re-queried from nvidia-smi on each call to get_free_gpus()
    so stale allocations from other processes are visible.

    Respects CUDA_VISIBLE_DEVICES: if set, only those GPUs are considered.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # gpu_id → model_id that currently owns it (None = free)
        self._allocated: Dict[int, Optional[str]] = {}
        # Respect CUDA_VISIBLE_DEVICES if set
        self._visible_gpus: Optional[List[int]] = None
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and cvd.strip():
            try:
                self._visible_gpus = [int(x.strip()) for x in cvd.split(",") if x.strip()]
                logger.info(f"Respecting CUDA_VISIBLE_DEVICES={cvd} → GPUs {self._visible_gpus}")
            except ValueError:
                logger.warning(f"Could not parse CUDA_VISIBLE_DEVICES='{cvd}', using all GPUs")
        # Initialise with discovered GPUs (filtered by CUDA_VISIBLE_DEVICES)
        mem = _query_gpu_memory()
        for gpu_id in mem:
            if self._visible_gpus is not None and gpu_id not in self._visible_gpus:
                continue
            self._allocated[gpu_id] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_gpus(self) -> List[int]:
        """Return all discovered GPU IDs (sorted)."""
        return sorted(self._allocated.keys())

    def get_free_gpus(self, min_memory_gb: int = 8, max_used_mb: int = 1024) -> List[int]:
        """Return GPU IDs that are truly free for vLLM.

        A GPU is considered free only if:
        1. Not already allocated by us
        2. Has enough total VRAM for the model
        3. Is not occupied by another process (used memory < max_used_mb)

        vLLM requires near-exclusive GPU access. A GPU with 15GB "free" but
        29GB used by another process will fail or conflict with vLLM.

        Args:
            min_memory_gb: Minimum free VRAM required (GB).
            max_used_mb:   Maximum used VRAM (MB) to consider a GPU unoccupied.
                           Default 1024MB allows for driver/CUDA context overhead.

        Returns:
            Sorted list of eligible GPU IDs.
        """
        mem = _query_gpu_memory()
        min_free_mb = min_memory_gb * 1024

        with self._lock:
            free = []
            for gpu_id, owner in self._allocated.items():
                if owner is not None:
                    continue
                total_mb, free_mb = mem.get(gpu_id, (0, 0))
                used_mb = total_mb - free_mb
                if used_mb > max_used_mb:
                    logger.debug(
                        f"GPU {gpu_id} skipped: {used_mb}MB used by other processes "
                        f"(threshold {max_used_mb}MB)"
                    )
                    continue
                if free_mb >= min_free_mb:
                    free.append(gpu_id)
            return sorted(free)

    def allocate_gpu(self, model_id: str, required_memory_gb: Optional[int] = None,
                     max_used_mb: int = 1024) -> Optional[int]:
        """Allocate the GPU with the most free memory that meets the requirement.

        Only considers GPUs that are truly unoccupied (used < max_used_mb).

        Args:
            model_id: Model being assigned (used for logging).
            required_memory_gb: VRAM needed. Estimated from model_id if None.
            max_used_mb: Maximum used VRAM to consider a GPU unoccupied.

        Returns:
            GPU ID, or None if no suitable GPU is available.
        """
        if required_memory_gb is None:
            required_memory_gb = estimate_model_vram(model_id)

        min_free_mb = required_memory_gb * 1024
        mem = _query_gpu_memory()

        with self._lock:
            best_gpu: Optional[int] = None
            best_free = -1
            for gpu_id, owner in self._allocated.items():
                if owner is not None:
                    continue
                total_mb, free_mb = mem.get(gpu_id, (0, 0))
                used_mb = total_mb - free_mb
                if used_mb > max_used_mb:
                    continue
                if free_mb >= min_free_mb and free_mb > best_free:
                    best_gpu = gpu_id
                    best_free = free_mb

            if best_gpu is not None:
                self._allocated[best_gpu] = model_id
                logger.info(
                    f"Allocated GPU {best_gpu} to '{model_id}' "
                    f"(free={best_free // 1024}GB, required≥{required_memory_gb}GB)"
                )
            else:
                logger.warning(
                    f"No unoccupied GPU with ≥{required_memory_gb}GB free found for '{model_id}'. "
                    f"GPU status: { {g: f'used={mem.get(g,(0,0))[0]-mem.get(g,(0,0))[1]}MB free={mem.get(g,(0,0))[1]//1024}GB' for g in self._allocated} }"
                )
            return best_gpu

    def release_gpu(self, gpu_id: int) -> None:
        """Mark a GPU as available again."""
        with self._lock:
            if gpu_id in self._allocated:
                owner = self._allocated[gpu_id]
                self._allocated[gpu_id] = None
                logger.info(f"Released GPU {gpu_id} (was held by '{owner}')")

    def gpu_memory_info(self) -> Dict[int, Dict[str, int]]:
        """Return current memory info for all GPUs."""
        mem = _query_gpu_memory()
        return {
            gpu_id: {
                "total_gb": mem[gpu_id][0] // 1024 if gpu_id in mem else 0,
                "free_gb":  mem[gpu_id][1] // 1024 if gpu_id in mem else 0,
                "used_gb":  (mem[gpu_id][0] - mem[gpu_id][1]) // 1024 if gpu_id in mem else 0,
            }
            for gpu_id in sorted(self._allocated)
        }

    def parse_gpu_spec(self, gpus_spec: str, model_id: str = "") -> List[int]:
        """Parse --gpus CLI argument.

        Args:
            gpus_spec: "auto", "0,1,2,3", or "0-3" style range.
            model_id:  Used for VRAM estimation when gpus_spec=="auto".

        Returns:
            List of GPU IDs to use.
        """
        if gpus_spec.strip().lower() == "auto":
            required_gb = estimate_model_vram(model_id)
            free = self.get_free_gpus(min_memory_gb=required_gb)
            if not free:
                # Fall back to minimum bar
                free = self.get_free_gpus(min_memory_gb=4)
            logger.info(f"Auto-detected free GPUs: {free}")
            return free

        gpu_ids: List[int] = []
        for part in gpus_spec.split(","):
            part = part.strip()
            if "-" in part and not part.startswith("-"):
                start, end = part.split("-", 1)
                gpu_ids.extend(range(int(start), int(end) + 1))
            else:
                gpu_ids.append(int(part))

        # Validate that requested GPUs actually exist
        all_known = set(self._allocated.keys())
        valid = sorted(g for g in set(gpu_ids) if g in all_known)
        invalid = sorted(set(gpu_ids) - all_known)
        if invalid:
            logger.warning(f"GPUs {invalid} do not exist (available: {sorted(all_known)}), skipping them")
        return valid

"""
Response cache for BeyondBench evaluation framework.

Disk-based LRU cache for model responses to avoid redundant inference.
Cache keys are derived from model_id, prompt, temperature, and seed.
"""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default cache directory
_DEFAULT_CACHE_DIR = Path.home() / ".beyondbench" / "cache"
# Default max cache size: 10 GB
_DEFAULT_MAX_SIZE_BYTES = 10 * 1024 * 1024 * 1024


class ResponseCache:
    """
    Thread-safe, disk-based LRU response cache for model outputs.

    Cache entries are stored as individual JSON files named by their SHA-256
    hash. An index file tracks metadata (model_id, size, access times) and is
    used for LRU eviction when the cache exceeds max_size_bytes.

    Only deterministic responses should be cached:
      - temperature == 0  (greedy, always reproducible)
      - temperature > 0 with an explicit seed  (seeded sampling)

    Usage:
        cache = ResponseCache()
        cache.put("my-model", "What is 2+2?", 0.0, None, "4")
        response = cache.get("my-model", "What is 2+2?", 0.0, None)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ):
        """
        Initialize the response cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.beyondbench/cache/
            max_size_bytes: Maximum total cache size in bytes (default: 10 GB).
        """
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._max_size_bytes = max_size_bytes
        self._lock = threading.Lock()

        # In-memory hit/miss counters
        self._hits = 0
        self._misses = 0

        # Ensure cache directory exists
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directory {self._cache_dir}: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
    ) -> Optional[str]:
        """
        Look up a cached response.

        Args:
            model_id: Model identifier string.
            prompt: The raw prompt text.
            temperature: Sampling temperature used for generation.
            seed: Random seed (None if not set).

        Returns:
            Cached response string, or None on a cache miss.
        """
        key = self._make_key(model_id, prompt, temperature, seed)
        entry_path = self._entry_path(key)

        with self._lock:
            if not entry_path.exists():
                self._misses += 1
                return None

            try:
                with open(entry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                response = data.get("response")
                if response is None:
                    self._misses += 1
                    return None

                # Update last_access in index
                self._touch_index(key)
                self._hits += 1
                logger.debug(f"Cache HIT  key={key[:12]}...")
                return response

            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning(f"Cache entry read error for key {key[:12]}: {e}")
                # Remove corrupted entry
                self._remove_entry(key)
                self._misses += 1
                return None

    def put(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
        response: str,
    ) -> None:
        """
        Store a response in the cache.

        Args:
            model_id: Model identifier string.
            prompt: The raw prompt text.
            temperature: Sampling temperature used for generation.
            seed: Random seed (None if not set).
            response: The generated response to cache.
        """
        key = self._make_key(model_id, prompt, temperature, seed)
        entry_path = self._entry_path(key)

        entry_data = {
            "model_id": model_id,
            "temperature": temperature,
            "seed": seed,
            "response": response,
            "created_at": time.time(),
        }
        serialized = json.dumps(entry_data, ensure_ascii=False)
        size_bytes = len(serialized.encode("utf-8"))

        with self._lock:
            # Write cache entry file
            try:
                with open(entry_path, "w", encoding="utf-8") as f:
                    f.write(serialized)
            except OSError as e:
                logger.warning(f"Could not write cache entry {key[:12]}: {e}")
                return

            # Update index
            now = time.time()
            index = self._load_index()
            index[key] = {
                "model_id": model_id,
                "size_bytes": size_bytes,
                "last_access": now,
                "created_at": now,
            }
            self._save_index(index)

            logger.debug(f"Cache STORE key={key[:12]}... size={size_bytes}B")

            # Evict if over limit
            self._evict_if_needed(index)

    def clear(self) -> None:
        """Remove all cached entries and reset hit/miss counters."""
        with self._lock:
            try:
                # Remove all .json files in the cache directory (entries)
                for p in self._cache_dir.glob("*.json"):
                    if p.name != "index.json":
                        try:
                            p.unlink()
                        except OSError:
                            pass
                # Reset index
                self._save_index({})
            except Exception as e:
                logger.warning(f"Cache clear error: {e}")

            self._hits = 0
            self._misses = 0
        logger.info("Response cache cleared.")

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with keys: hits, misses, total_entries, total_size_bytes,
            hit_rate, cache_dir.
        """
        with self._lock:
            index = self._load_index()
            total_entries = len(index)
            total_size = sum(v.get("size_bytes", 0) for v in index.values())
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "max_size_bytes": self._max_size_bytes,
            "cache_dir": str(self._cache_dir),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        model_id: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
    ) -> str:
        """Compute a deterministic SHA-256 cache key."""
        seed_str = str(seed) if seed is not None else "None"
        raw = f"{model_id}\x00{prompt}\x00{temperature}\x00{seed_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _entry_path(self, key: str) -> Path:
        """Return the file path for a cache entry key."""
        return self._cache_dir / f"{key}.json"

    @property
    def _index_path(self) -> Path:
        return self._cache_dir / "index.json"

    def _load_index(self) -> Dict[str, Any]:
        """Load the cache index from disk. Returns empty dict on any error."""
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except (json.JSONDecodeError, OSError):
            # Corrupted index — start fresh
            logger.warning("Cache index is corrupted; resetting.")
            return {}

    def _save_index(self, index: Dict[str, Any]) -> None:
        """Persist the cache index to disk."""
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"Could not save cache index: {e}")

    def _touch_index(self, key: str) -> None:
        """Update last_access for a key in the index (call while holding lock)."""
        index = self._load_index()
        if key in index:
            index[key]["last_access"] = time.time()
            self._save_index(index)

    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry file and its index entry (call while holding lock)."""
        entry_path = self._entry_path(key)
        try:
            entry_path.unlink(missing_ok=True)
        except OSError:
            pass
        index = self._load_index()
        index.pop(key, None)
        self._save_index(index)

    def _evict_if_needed(self, index: Dict[str, Any]) -> None:
        """
        Evict least-recently-used entries until total size is within limit.
        Called while holding the lock; index is the current (already-updated) index.
        """
        total_size = sum(v.get("size_bytes", 0) for v in index.values())
        if total_size <= self._max_size_bytes:
            return

        # Sort by last_access ascending (LRU first)
        sorted_keys = sorted(
            index.keys(),
            key=lambda k: index[k].get("last_access", 0),
        )

        for key in sorted_keys:
            if total_size <= self._max_size_bytes:
                break
            size = index[key].get("size_bytes", 0)
            self._remove_entry(key)
            index.pop(key, None)
            total_size -= size
            logger.debug(f"Cache LRU evict key={key[:12]}... freed={size}B")


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_global_cache: Optional[ResponseCache] = None
_global_cache_lock = threading.Lock()


def get_global_cache() -> ResponseCache:
    """Return the module-level shared ResponseCache instance (lazy init)."""
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = ResponseCache()
    return _global_cache

"""
Per-request JSONL logger for model evaluations.

Logs every prompt/response pair with token counts, latency, and metadata.
The JSONL file can be replayed for parser development without re-running inference.

Usage::

    logger = RequestLogger(output_dir="./results", enabled=True)

    # Record a request
    entry = logger.log(
        task_name="sum",
        fold=0,
        prompt="What is 3 + 5?",
        response="8",
        input_tokens=12,
        output_tokens=2,
        latency_ms=142.3,
        correct=True,
        metadata={"list_size": 8},
    )

    logger.close()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RequestEntry:
    """A single logged request/response pair."""
    timestamp: float
    task_name: str
    fold: int
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    correct: Optional[bool]
    expected: Any
    parsed_answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure JSON-serialisable values
        d["timestamp"] = self.timestamp
        return d


class RequestLogger:
    """
    Append-only JSONL request/response logger.

    Each line is a JSON object (RequestEntry.to_dict()).
    The file is flushed after every write so it is safe to read mid-run.

    Parameters
    ----------
    output_dir : str | Path
        Directory where ``requests.jsonl`` will be written.
    enabled : bool
        When False, all log() calls are no-ops.
    filename : str
        Override the default filename.
    """

    def __init__(
        self,
        output_dir: str | Path,
        enabled: bool = True,
        filename: str = "requests.jsonl",
    ):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled
        self._file = None
        self._path: Optional[Path] = None
        self._count = 0

        if enabled:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            self._path = out / filename
            try:
                self._file = open(self._path, "a", encoding="utf-8")
                self.logger.info(f"Request logger writing to: {self._path}")
            except OSError as e:
                self.logger.error(f"Could not open request log file: {e}")
                self.enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        task_name: str,
        prompt: str,
        response: str,
        *,
        fold: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        correct: Optional[bool] = None,
        expected: Any = None,
        parsed_answer: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RequestEntry]:
        """
        Log a request/response pair.

        Returns the RequestEntry (whether or not logging is enabled).
        """
        entry = RequestEntry(
            timestamp=time.time(),
            task_name=task_name,
            fold=fold,
            prompt=prompt,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            correct=correct,
            expected=expected,
            parsed_answer=parsed_answer,
            metadata=metadata or {},
        )

        if self.enabled and self._file is not None:
            try:
                line = json.dumps(entry.to_dict(), ensure_ascii=False, default=str)
                self._file.write(line + "\n")
                self._file.flush()
                self._count += 1
            except Exception as e:
                self.logger.warning(f"Failed to write request log entry: {e}")

        return entry

    def close(self):
        """Flush and close the underlying file."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._count > 0:
            self.logger.info(f"Request logger closed — {self._count} entries written to {self._path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Replay / analysis helpers
    # ------------------------------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def entry_count(self) -> int:
        return self._count

    @staticmethod
    def load(path: str | Path) -> List[Dict[str, Any]]:
        """
        Load all entries from a JSONL log file.

        Returns a list of dicts (one per logged request).
        """
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    @staticmethod
    def replay_prompts(path: str | Path) -> List[str]:
        """Return just the prompt strings from a JSONL log (for parser replay)."""
        return [e["prompt"] for e in RequestLogger.load(path)]

    @staticmethod
    def replay_responses(path: str | Path) -> List[str]:
        """Return just the response strings from a JSONL log."""
        return [e["response"] for e in RequestLogger.load(path)]

    @staticmethod
    def summarize(path: str | Path) -> Dict[str, Any]:
        """
        Return a quick summary of a JSONL log file.

        Keys: total_requests, tasks, avg_input_tokens, avg_output_tokens,
              avg_latency_ms, correct_rate.
        """
        entries = RequestLogger.load(path)
        if not entries:
            return {"total_requests": 0}

        tasks = list({e.get("task_name", "unknown") for e in entries})
        inputs = [e.get("input_tokens", 0) for e in entries]
        outputs = [e.get("output_tokens", 0) for e in entries]
        latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms", 0) > 0]
        correct_vals = [e["correct"] for e in entries if e.get("correct") is not None]

        return {
            "total_requests": len(entries),
            "tasks": tasks,
            "avg_input_tokens": round(sum(inputs) / len(inputs), 1) if inputs else 0,
            "avg_output_tokens": round(sum(outputs) / len(outputs), 1) if outputs else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "correct_rate": round(sum(correct_vals) / len(correct_vals), 4) if correct_vals else None,
        }

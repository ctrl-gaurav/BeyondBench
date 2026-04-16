"""
FastAPI application for BeyondBench evaluation server — v2 Production Gateway.

Features:
- REST API: task listing, single + batch evaluation, job management, result retrieval
- WebSocket: ws://localhost:8000/ws/jobs/{job_id} — live progress / logs stream
- Authentication: Bearer token via BEYONDBENCH_API_KEY env var
- Rate limiting: configurable requests-per-minute per API key
- Comparison API: POST /compare for side-by-side evaluation results
- Full OpenAPI documentation with examples for every endpoint
"""

import asyncio
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    CompareRequest,
    CompareResponse,
    DeleteJobResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    JobListResponse,
    JobResultResponse,
    JobStatusResponse,
    ResultDetailResponse,
    ResultListResponse,
    ResultSummary,
    TaskInfo,
    TaskInfoResponse,
    TaskListResponse,
)

# --------------------------------------------------------------------------- #
# In-memory job store
# --------------------------------------------------------------------------- #
# Each entry: {
#   "status": str,           # running | completed | failed | cancelled
#   "result": dict | None,
#   "error": str | None,
#   "progress": dict | None, # {task, completed, total, percent}
#   "logs": list[str],       # accumulated log lines
#   "created_at": datetime,
#   "updated_at": datetime,
#   "cancel_flag": threading.Event,
# }
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# WebSocket subscriber sets per job_id: set of asyncio.Queue
_ws_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
_ws_lock = threading.Lock()

# Global progress subscribers (ws/progress) — receive events from ALL jobs
_GLOBAL_PROGRESS_QUEUES: Set[asyncio.Queue] = set()

# --------------------------------------------------------------------------- #
# Rate-limiting state: api_key → list of request timestamps
# --------------------------------------------------------------------------- #
_rate_state: Dict[str, List[float]] = defaultdict(list)
_rate_lock = threading.Lock()

_RATE_LIMIT_RPM = int(os.environ.get("BEYONDBENCH_RATE_LIMIT_RPM", "60"))

# --------------------------------------------------------------------------- #
# Server startup time & GPU detection
# --------------------------------------------------------------------------- #
_START_TIME = time.time()


def _detect_gpu_count() -> int:
    """Detect GPU count via nvidia-smi; returns 0 on failure."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
    except Exception:
        pass
    return 0

# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
_security = HTTPBearer(auto_error=False)
_API_KEY: Optional[str] = os.environ.get("BEYONDBENCH_API_KEY")


def _check_auth(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    """Validate Bearer token.  Returns the key (for rate-limit keying)."""
    if _API_KEY is None:
        # Auth disabled — no key configured
        return "__no_auth__"
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Use: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _check_rate_limit(api_key: str) -> None:
    """Sliding-window rate limiter: max _RATE_LIMIT_RPM requests per minute."""
    now = time.time()
    window = 60.0
    with _rate_lock:
        timestamps = _rate_state[api_key]
        # Prune old timestamps
        _rate_state[api_key] = [t for t in timestamps if now - t < window]
        if len(_rate_state[api_key]) >= _RATE_LIMIT_RPM:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {_RATE_LIMIT_RPM} requests/minute.",
                headers={"Retry-After": "60"},
            )
        _rate_state[api_key].append(now)


async def _auth_and_rate(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> str:
    key = _check_auth(credentials)
    _check_rate_limit(key)
    return key


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_job_entry() -> Dict[str, Any]:
    now = datetime.utcnow()
    return {
        "status": "running",
        "result": None,
        "error": None,
        "progress": None,
        "logs": [],
        "created_at": now,
        "updated_at": now,
        "cancel_flag": threading.Event(),
    }


def _job_to_response(job_id: str, job: Dict[str, Any]) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
    )


def _push_ws_event(job_id: str, event: Dict[str, Any]) -> None:
    """Thread-safe: push an event dict into every subscriber queue for job_id."""
    with _ws_lock:
        queues = list(_ws_subscribers.get(job_id, []))
    for q in queues:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass
    # Also push to global /ws/progress subscribers (with job_id tag)
    global_event = {"job_id": job_id, **event}
    for gq in list(_GLOBAL_PROGRESS_QUEUES):
        try:
            gq.put_nowait(global_event)
        except asyncio.QueueFull:
            pass


def _run_evaluation_thread(job_id: str, req: EvaluateRequest) -> None:
    """Background thread: run an evaluation and update the job store."""
    job = _jobs[job_id]
    cancel = job["cancel_flag"]

    def _progress_cb(task_name: str, completed: int, total: int) -> None:
        if cancel.is_set():
            raise InterruptedError("Job cancelled by user.")
        progress = {
            "task": task_name,
            "completed": completed,
            "total": total,
            "percent": round(completed / total, 4) if total else 0.0,
        }
        with _jobs_lock:
            job["progress"] = progress
            job["updated_at"] = datetime.utcnow()
        _push_ws_event(
            job_id,
            {"type": "progress", "data": progress, "timestamp": datetime.utcnow().isoformat()},
        )

    def _log_cb(line: str) -> None:
        with _jobs_lock:
            job["logs"].append(line)
            job["updated_at"] = datetime.utcnow()
        _push_ws_event(
            job_id,
            {"type": "log", "data": {"line": line}, "timestamp": datetime.utcnow().isoformat()},
        )

    try:
        if cancel.is_set():
            with _jobs_lock:
                job["status"] = "cancelled"
                job["updated_at"] = datetime.utcnow()
            _push_ws_event(job_id, {"type": "cancelled", "timestamp": datetime.utcnow().isoformat()})
            return

        from ..models.model_handler import ModelHandler
        from ..core.evaluation_engine import EvaluationEngine

        model_handler = ModelHandler(
            model_id=req.model_id,
            backend=req.backend,
            api_provider=req.api_provider,
            api_key=req.api_key,
            cuda_device=req.cuda_device,
            tensor_parallel_size=req.tensor_parallel_size,
            gpu_memory_utilization=req.gpu_memory_utilization,
            trust_remote_code=req.trust_remote_code,
            reasoning_effort=req.reasoning_effort,
            thinking_budget=req.thinking_budget,
        )

        engine = EvaluationEngine(
            model_handler=model_handler,
            output_dir=req.output_dir,
            store_details=req.store_details,
            max_retries=req.max_retries,
        )

        results = engine.run_evaluation(
            suite=req.suite,
            tasks=req.tasks or [],
            datapoints=req.datapoints,
            folds=req.folds,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            list_sizes=req.list_sizes,
            range_min=req.range_min,
            range_max=req.range_max,
            seed=req.seed,
        )

        with _jobs_lock:
            job["status"] = "completed"
            job["result"] = results
            job["updated_at"] = datetime.utcnow()
        _push_ws_event(
            job_id,
            {
                "type": "completed",
                "data": {"result": results},
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    except InterruptedError:
        with _jobs_lock:
            job["status"] = "cancelled"
            job["updated_at"] = datetime.utcnow()
        _push_ws_event(job_id, {"type": "cancelled", "timestamp": datetime.utcnow().isoformat()})
    except Exception as exc:
        with _jobs_lock:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["updated_at"] = datetime.utcnow()
        _push_ws_event(
            job_id,
            {
                "type": "error",
                "data": {"error": str(exc)},
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    finally:
        # Signal EOF so WS consumers can exit
        _push_ws_event(job_id, {"type": "done", "timestamp": datetime.utcnow().isoformat()})


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #

def create_app() -> FastAPI:
    """Create and configure the BeyondBench FastAPI application (v2)."""

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        app_version = _pkg_version("beyondbench")
    except PackageNotFoundError:
        app_version = "0.2.0"

    _auth_note = (
        "\n\n**Authentication:** Set `BEYONDBENCH_API_KEY` environment variable to enable "
        "Bearer token auth. Include `Authorization: Bearer <key>` header on all requests."
    )
    _rate_note = (
        f"\n\n**Rate Limiting:** {_RATE_LIMIT_RPM} requests/minute per API key "
        "(configurable via `BEYONDBENCH_RATE_LIMIT_RPM` env var)."
    )

    app = FastAPI(
        title="BeyondBench API",
        description=(
            "Production-grade REST API for the BeyondBench LLM evaluation framework.\n\n"
            "Evaluate LLM reasoning capabilities across easy, medium, and hard tasks "
            "on local (vLLM / transformers) or cloud (OpenAI / Gemini / Anthropic) models.\n\n"
            "**Features:**\n"
            "- Submit single or batch evaluation jobs\n"
            "- Monitor live progress via WebSocket\n"
            "- Cancel running jobs\n"
            "- Compare results across runs\n"
            + _auth_note
            + _rate_note
        ),
        version=app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized — invalid or missing API key"},
            429: {"model": ErrorResponse, "description": "Too Many Requests — rate limit exceeded"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
        },
        openapi_tags=[
            {"name": "System", "description": "Health and server metadata"},
            {"name": "Tasks", "description": "List and inspect available evaluation tasks"},
            {"name": "Evaluation", "description": "Submit and manage evaluation jobs"},
            {"name": "Results", "description": "Retrieve past evaluation results"},
            {"name": "Comparison", "description": "Compare evaluation results across models/runs"},
            {"name": "WebSocket", "description": "Live job streaming via WebSocket"},
        ],
    )

    # ------------------------------------------------------------------ #
    # GET /health
    # ------------------------------------------------------------------ #
    _gpu_count = _detect_gpu_count()

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Server health check",
        description="Returns server status, version, GPU count, and uptime. No authentication required.",
    )
    async def health():
        """Health check — no auth required."""
        return HealthResponse(
            status="ok",
            version=app_version,
            gpu_count=_gpu_count,
            uptime=round(time.time() - _START_TIME, 2),
        )

    # ------------------------------------------------------------------ #
    # GET /metrics  (Prometheus-compatible)
    # ------------------------------------------------------------------ #
    @app.get(
        "/metrics",
        tags=["System"],
        summary="Prometheus-compatible metrics",
        description=(
            "Returns server metrics in Prometheus text exposition format.\n\n"
            "Metrics include: uptime, GPU count, job counts by status, "
            "and rate limit configuration."
        ),
    )
    async def metrics():
        """Prometheus-compatible metrics endpoint — no auth required."""
        from fastapi.responses import PlainTextResponse

        uptime = round(time.time() - _START_TIME, 2)
        with _jobs_lock:
            total_jobs = len(_jobs)
            running = sum(1 for j in _jobs.values() if j["status"] == "running")
            completed = sum(1 for j in _jobs.values() if j["status"] == "completed")
            failed = sum(1 for j in _jobs.values() if j["status"] == "failed")
            cancelled = sum(1 for j in _jobs.values() if j["status"] == "cancelled")

        lines = [
            "# HELP beyondbench_up Server is up (1 = up).",
            "# TYPE beyondbench_up gauge",
            "beyondbench_up 1",
            "# HELP beyondbench_uptime_seconds Server uptime in seconds.",
            "# TYPE beyondbench_uptime_seconds gauge",
            f"beyondbench_uptime_seconds {uptime}",
            "# HELP beyondbench_gpu_count Number of available GPUs.",
            "# TYPE beyondbench_gpu_count gauge",
            f"beyondbench_gpu_count {_gpu_count}",
            "# HELP beyondbench_jobs_total Total number of jobs by status.",
            "# TYPE beyondbench_jobs_total gauge",
            f'beyondbench_jobs_total{{status="running"}} {running}',
            f'beyondbench_jobs_total{{status="completed"}} {completed}',
            f'beyondbench_jobs_total{{status="failed"}} {failed}',
            f'beyondbench_jobs_total{{status="cancelled"}} {cancelled}',
            "# HELP beyondbench_jobs_count Total number of jobs in memory.",
            "# TYPE beyondbench_jobs_count gauge",
            f"beyondbench_jobs_count {total_jobs}",
            "# HELP beyondbench_rate_limit_rpm Configured rate limit (requests per minute).",
            "# TYPE beyondbench_rate_limit_rpm gauge",
            f"beyondbench_rate_limit_rpm {_RATE_LIMIT_RPM}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")

    # ------------------------------------------------------------------ #
    # GET /tasks
    # ------------------------------------------------------------------ #
    @app.get(
        "/tasks",
        response_model=TaskListResponse,
        tags=["Tasks"],
        summary="List available evaluation tasks",
        description=(
            "Returns all available tasks grouped by suite (easy / medium / hard). "
            "Filter by `suite` query parameter."
        ),
    )
    async def list_tasks(
        suite: str = Query(
            "all",
            pattern="^(easy|medium|hard|all)$",
            description="Task suite filter: easy, medium, hard, or all",
            examples={"all": {"value": "all"}, "easy": {"value": "easy"}},
        ),
        _key: str = Depends(_auth_and_rate),
    ):
        """List available evaluation tasks, optionally filtered by suite."""
        from ..core.task_registry import TaskRegistry

        registry = TaskRegistry()
        tasks = registry.get_available_tasks(suite)
        total = sum(len(v) for v in tasks.values())
        return TaskListResponse(tasks=tasks, total=total)

    # ------------------------------------------------------------------ #
    # GET /tasks/{task_name}
    # ------------------------------------------------------------------ #
    @app.get(
        "/tasks/{task_name}",
        response_model=TaskInfoResponse,
        responses={404: {"model": ErrorResponse, "description": "Task not found"}},
        tags=["Tasks"],
        summary="Get task details",
        description="Returns detailed metadata for a named evaluation task.",
    )
    async def get_task(
        task_name: str,
        _key: str = Depends(_auth_and_rate),
    ):
        """Get detailed information about a specific task."""
        from ..core.task_registry import TaskRegistry

        registry = TaskRegistry()
        info = registry.get_task_info(task_name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
        return TaskInfoResponse(task=TaskInfo(**info))

    # ------------------------------------------------------------------ #
    # POST /evaluate
    # ------------------------------------------------------------------ #
    @app.post(
        "/evaluate",
        response_model=EvaluateResponse,
        status_code=202,
        tags=["Evaluation"],
        summary="Submit a single evaluation job",
        description=(
            "Starts a background evaluation job. Returns a `job_id` for polling "
            "(`GET /jobs/{job_id}`) or live streaming (`ws://host/ws/jobs/{job_id}`).\n\n"
            "**Minimal example:**\n"
            "```json\n"
            '{"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "backend": "vllm", '
            '"suite": "easy", "datapoints": 50}\n'
            "```"
        ),
    )
    async def start_evaluation(
        request: EvaluateRequest,
        _key: str = Depends(_auth_and_rate),
    ):
        """Start an evaluation job in the background."""
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = _make_job_entry()

        thread = threading.Thread(
            target=_run_evaluation_thread, args=(job_id, request), daemon=True
        )
        thread.start()

        return EvaluateResponse(
            job_id=job_id,
            status="started",
            message=f"Evaluation job {job_id} started in background.",
        )

    # ------------------------------------------------------------------ #
    # POST /evaluate/batch
    # ------------------------------------------------------------------ #
    @app.post(
        "/evaluate/batch",
        response_model=BatchEvaluateResponse,
        status_code=202,
        tags=["Evaluation"],
        summary="Submit a batch of evaluation jobs",
        description=(
            "Submit multiple evaluation configurations at once (max 20). "
            "Each config is started as an independent background job. "
            "Returns a list of job IDs."
        ),
    )
    async def start_batch_evaluation(
        request: BatchEvaluateRequest,
        _key: str = Depends(_auth_and_rate),
    ):
        """Start multiple evaluation jobs in the background."""
        job_ids: List[str] = []
        for req in request.jobs:
            job_id = str(uuid.uuid4())
            with _jobs_lock:
                _jobs[job_id] = _make_job_entry()
            thread = threading.Thread(
                target=_run_evaluation_thread, args=(job_id, req), daemon=True
            )
            thread.start()
            job_ids.append(job_id)

        return BatchEvaluateResponse(
            job_ids=job_ids,
            total=len(job_ids),
            message=f"{len(job_ids)} evaluation job(s) started.",
        )

    # ------------------------------------------------------------------ #
    # GET /jobs
    # ------------------------------------------------------------------ #
    @app.get(
        "/jobs",
        response_model=JobListResponse,
        tags=["Evaluation"],
        summary="List all evaluation jobs",
        description=(
            "Returns all jobs in the in-memory store. "
            "Filter by `status` (running, completed, failed, cancelled)."
        ),
    )
    async def list_jobs(
        status_filter: Optional[str] = Query(
            None,
            alias="status",
            pattern="^(running|completed|failed|cancelled)$",
            description="Filter by job status",
        ),
        _key: str = Depends(_auth_and_rate),
    ):
        """List all background evaluation jobs."""
        with _jobs_lock:
            items = list(_jobs.items())

        responses = []
        for job_id, job in items:
            if status_filter and job["status"] != status_filter:
                continue
            responses.append(_job_to_response(job_id, job))

        return JobListResponse(jobs=responses, total=len(responses))

    # ------------------------------------------------------------------ #
    # GET /jobs/{job_id}
    # ------------------------------------------------------------------ #
    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        responses={404: {"model": ErrorResponse, "description": "Job not found"}},
        tags=["Evaluation"],
        summary="Get job status",
        description=(
            "Poll the status of a background evaluation job. "
            "For live streaming, use the WebSocket endpoint instead."
        ),
    )
    async def get_job_status(
        job_id: str,
        _key: str = Depends(_auth_and_rate),
    ):
        """Check the status of a background evaluation job."""
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return _job_to_response(job_id, job)

    # ------------------------------------------------------------------ #
    # GET /jobs/{job_id}/results
    # ------------------------------------------------------------------ #
    @app.get(
        "/jobs/{job_id}/results",
        response_model=JobResultResponse,
        responses={
            404: {"model": ErrorResponse, "description": "Job not found"},
            409: {"model": ErrorResponse, "description": "Job not yet completed"},
        },
        tags=["Evaluation"],
        summary="Get completed job results",
        description=(
            "Returns the evaluation results for a completed job. "
            "Returns 409 if the job is still running or has failed."
        ),
    )
    async def get_job_results(
        job_id: str,
        _key: str = Depends(_auth_and_rate),
    ):
        """Get the results of a completed evaluation job."""
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        if job["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Job '{job_id}' is not completed (status: {job['status']})",
            )
        return JobResultResponse(
            job_id=job_id,
            status=job["status"],
            result=job.get("result"),
        )

    # ------------------------------------------------------------------ #
    # DELETE /jobs/{job_id}
    # ------------------------------------------------------------------ #
    @app.delete(
        "/jobs/{job_id}",
        response_model=DeleteJobResponse,
        responses={
            404: {"model": ErrorResponse, "description": "Job not found"},
            409: {"model": ErrorResponse, "description": "Job already finished"},
        },
        tags=["Evaluation"],
        summary="Cancel / delete a job",
        description=(
            "Signals a running job to cancel. The background thread checks the cancel flag "
            "before each task and raises `InterruptedError` to stop cleanly. "
            "Already-finished jobs are removed from the store."
        ),
    )
    async def delete_job(
        job_id: str,
        _key: str = Depends(_auth_and_rate),
    ):
        """Cancel a running job or remove a finished one."""
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
            current_status = job["status"]
            if current_status == "running":
                job["cancel_flag"].set()
                job["status"] = "cancelled"
                job["updated_at"] = datetime.utcnow()
                msg = f"Job {job_id} cancel signal sent."
            else:
                del _jobs[job_id]
                msg = f"Job {job_id} ({current_status}) removed."

        return DeleteJobResponse(job_id=job_id, message=msg)

    # ------------------------------------------------------------------ #
    # POST /compare
    # ------------------------------------------------------------------ #
    @app.post(
        "/compare",
        response_model=CompareResponse,
        responses={
            404: {"model": ErrorResponse, "description": "One or more result IDs not found"},
        },
        tags=["Comparison"],
        summary="Compare evaluation results",
        description=(
            "Load final_results.json files for two or more result IDs and produce a "
            "side-by-side comparison. `result_ids` may be directory names under "
            "`output_dir` OR completed `job_id`s (the engine writes results under "
            "`output_dir/<job_id>/`)."
        ),
    )
    async def compare_results(
        request: CompareRequest,
        _key: str = Depends(_auth_and_rate),
    ):
        """Compare multiple evaluation results."""
        results_dir = Path(request.output_dir)
        comparison: Dict[str, Any] = {}
        all_metrics: set = set()

        for rid in request.result_ids:
            # Check completed in-memory job first
            with _jobs_lock:
                job = _jobs.get(rid)
            if job is not None and job["status"] == "completed" and job["result"]:
                data = job["result"]
            else:
                # Fall back to file system
                candidate = results_dir / rid / "final_results.json"
                if not candidate.exists():
                    # Search recursively
                    found = None
                    for f in results_dir.rglob("final_results.json"):
                        if f.parent.name == rid:
                            found = f
                            break
                    if found is None:
                        raise HTTPException(
                            status_code=404, detail=f"Result '{rid}' not found"
                        )
                    candidate = found
                try:
                    with open(candidate, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error reading result file for '{rid}': {exc}",
                    )

            summary = data.get("summary", {})
            task_results = data.get("task_results", {})
            entry: Dict[str, Any] = {}

            if request.metrics:
                for m in request.metrics:
                    if m in summary:
                        entry[m] = summary[m]
                    elif m in task_results:
                        entry[m] = task_results[m]
                    all_metrics.update(request.metrics)
            else:
                entry.update(summary)
                # Flatten per-task accuracy if available
                for task_name, t_data in task_results.items():
                    if isinstance(t_data, dict) and "accuracy" in t_data:
                        key = f"task_{task_name}_accuracy"
                        entry[key] = t_data["accuracy"]
                        all_metrics.add(key)
                all_metrics.update(summary.keys())

            comparison[rid] = entry

        return CompareResponse(
            comparison=comparison,
            result_ids=request.result_ids,
            metrics_compared=sorted(all_metrics),
        )

    # ------------------------------------------------------------------ #
    # POST /evaluate/compare  (alias for /compare)
    # ------------------------------------------------------------------ #
    @app.post(
        "/evaluate/compare",
        response_model=CompareResponse,
        responses={
            404: {"model": ErrorResponse, "description": "One or more result IDs not found"},
        },
        tags=["Comparison"],
        summary="Compare evaluation results (alias)",
        description="Alias for `POST /compare`. See that endpoint for full documentation.",
    )
    async def compare_results_alias(
        request: CompareRequest,
        _key: str = Depends(_auth_and_rate),
    ):
        """Compare multiple evaluation results (alias route)."""
        return await compare_results(request, _key)

    # ------------------------------------------------------------------ #
    # GET /results
    # ------------------------------------------------------------------ #
    @app.get(
        "/results",
        response_model=ResultListResponse,
        tags=["Results"],
        summary="List past evaluation results",
        description="Scans `output_dir` for `final_results.json` files and returns summaries.",
    )
    async def list_results(
        output_dir: str = Query(
            "./beyondbench_results", description="Directory to scan for results"
        ),
        _key: str = Depends(_auth_and_rate),
    ):
        """List past evaluation results from output directory."""
        results_dir = Path(output_dir)
        summaries: list[ResultSummary] = []

        if results_dir.exists():
            for results_file in sorted(results_dir.rglob("final_results.json")):
                result_id = str(results_file.parent.name) or results_file.stem
                summary_data = None
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    summary_data = data.get("summary")
                except Exception:
                    pass
                summaries.append(
                    ResultSummary(
                        result_id=result_id,
                        path=str(results_file),
                        summary=summary_data,
                    )
                )

        return ResultListResponse(results=summaries, total=len(summaries))

    # ------------------------------------------------------------------ #
    # GET /results/{result_id}
    # ------------------------------------------------------------------ #
    @app.get(
        "/results/{result_id}",
        response_model=ResultDetailResponse,
        responses={404: {"model": ErrorResponse, "description": "Result not found"}},
        tags=["Results"],
        summary="Get detailed evaluation result",
        description="Returns the full `final_results.json` payload for a specific result ID.",
    )
    async def get_result(
        result_id: str,
        output_dir: str = Query(
            "./beyondbench_results", description="Directory to scan for results"
        ),
        _key: str = Depends(_auth_and_rate),
    ):
        """Get detailed data for a specific evaluation result."""
        results_dir = Path(output_dir)
        candidate = results_dir / result_id / "final_results.json"
        if not candidate.exists():
            for f in results_dir.rglob("final_results.json"):
                if f.parent.name == result_id:
                    candidate = f
                    break
            else:
                raise HTTPException(
                    status_code=404, detail=f"Result '{result_id}' not found"
                )

        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Error reading result file: {exc}"
            )

        return ResultDetailResponse(result_id=result_id, data=data)

    # ------------------------------------------------------------------ #
    # WebSocket: ws://host/ws/jobs/{job_id}
    # ------------------------------------------------------------------ #
    @app.websocket(
        "/ws/jobs/{job_id}",
    )
    async def ws_job_stream(websocket: WebSocket, job_id: str):
        """
        **WebSocket — live job streaming.**

        Connect to `ws://localhost:8000/ws/jobs/{job_id}` after submitting a job.

        Each message is a JSON object:

        | `type`      | `data` fields                                      |
        |-------------|---------------------------------------------------|
        | `progress`  | `task`, `completed`, `total`, `percent`           |
        | `log`       | `line`                                             |
        | `completed` | `result` (full evaluation result dict)            |
        | `error`     | `error` (error message string)                    |
        | `cancelled` | —                                                 |
        | `done`      | — (final sentinel; connection closes after this)  |

        **Authentication:** Pass the API key as a query parameter:
        `ws://localhost:8000/ws/jobs/{job_id}?api_key=<key>`
        """
        # WebSocket auth: query param or skip if no key configured
        if _API_KEY is not None:
            provided = websocket.query_params.get("api_key")
            if provided != _API_KEY:
                await websocket.close(code=4001, reason="Unauthorized")
                return

        with _jobs_lock:
            job = _jobs.get(job_id)

        if job is None:
            await websocket.close(code=4004, reason=f"Job '{job_id}' not found")
            return

        await websocket.accept()

        # If job already finished, send a synthetic completed/error/cancelled event + done
        with _jobs_lock:
            current_status = job["status"]
            current_result = job.get("result")
            current_error = job.get("error")
            existing_logs = list(job.get("logs", []))

        if current_status != "running":
            # Replay logs
            for line in existing_logs:
                await websocket.send_json(
                    {"type": "log", "data": {"line": line}, "timestamp": datetime.utcnow().isoformat()}
                )
            if current_status == "completed":
                await websocket.send_json(
                    {
                        "type": "completed",
                        "data": {"result": current_result},
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            elif current_status == "failed":
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"error": current_error},
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            elif current_status == "cancelled":
                await websocket.send_json(
                    {"type": "cancelled", "timestamp": datetime.utcnow().isoformat()}
                )
            await websocket.send_json({"type": "done", "timestamp": datetime.utcnow().isoformat()})
            await websocket.close()
            return

        # Job still running — subscribe to events
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        with _ws_lock:
            _ws_subscribers[job_id].add(queue)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    try:
                        await websocket.send_json(
                            {"type": "ping", "timestamp": datetime.utcnow().isoformat()}
                        )
                    except Exception:
                        break
                    continue

                try:
                    await websocket.send_json(event)
                except Exception:
                    break

                if event.get("type") == "done":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            with _ws_lock:
                _ws_subscribers[job_id].discard(queue)
            try:
                await websocket.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # WebSocket: ws://host/ws/progress  (all-jobs progress stream)
    # ------------------------------------------------------------------ #
    @app.websocket("/ws/progress")
    async def ws_progress_stream(websocket: WebSocket):
        """
        **WebSocket — live progress stream for all running jobs.**

        Connect to `ws://localhost:8000/ws/progress` to receive progress events
        from every running job.  Each message is a JSON object with a `job_id` field
        identifying which job the event belongs to.

        **Authentication:** Pass the API key as a query parameter:
        `ws://localhost:8000/ws/progress?api_key=<key>`

        The connection stays open and sends keepalive pings every 30 s.
        """
        # Auth
        if _API_KEY is not None:
            provided = websocket.query_params.get("api_key")
            if provided != _API_KEY:
                await websocket.close(code=4001, reason="Unauthorized")
                return

        await websocket.accept()

        # Subscribe a single queue to ALL current and future jobs
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        _GLOBAL_PROGRESS_QUEUES.add(queue)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json(
                            {"type": "ping", "timestamp": datetime.utcnow().isoformat()}
                        )
                    except Exception:
                        break
                    continue

                try:
                    await websocket.send_json(event)
                except Exception:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            _GLOBAL_PROGRESS_QUEUES.discard(queue)
            try:
                await websocket.close()
            except Exception:
                pass

    return app

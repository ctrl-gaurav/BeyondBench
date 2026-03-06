"""
FastAPI application for BeyondBench evaluation server.

Provides REST API endpoints for task listing, evaluation, and result retrieval.
"""

import json
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from .models import (
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    JobStatusResponse,
    ResultDetailResponse,
    ResultListResponse,
    ResultSummary,
    TaskInfo,
    TaskInfoResponse,
    TaskListResponse,
)

# In-memory job store (for background evaluation tracking)
_jobs: dict = {}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        app_version = _pkg_version("beyondbench")
    except PackageNotFoundError:
        app_version = "0.1.0"

    app = FastAPI(
        title="BeyondBench API",
        description=(
            "REST API for the BeyondBench evaluation framework. "
            "Evaluate LLM reasoning capabilities across easy, medium, and hard tasks."
        ),
        version=app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        responses={
            500: {"model": ErrorResponse},
        },
    )

    # ------------------------------------------------------------------ #
    # GET /health
    # ------------------------------------------------------------------ #
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """Health check endpoint."""
        return HealthResponse(status="ok", version=app_version)

    # ------------------------------------------------------------------ #
    # GET /tasks
    # ------------------------------------------------------------------ #
    @app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
    async def list_tasks(
        suite: str = Query("all", regex="^(easy|medium|hard|all)$", description="Task suite filter"),
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
        responses={404: {"model": ErrorResponse}},
        tags=["Tasks"],
    )
    async def get_task(task_name: str):
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
    )
    async def start_evaluation(request: EvaluateRequest):
        """Start an evaluation job in the background. Returns a job ID for tracking."""
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {"status": "running", "result": None, "error": None}

        def _run_evaluation(jid: str, req: EvaluateRequest):
            try:
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

                _jobs[jid]["status"] = "completed"
                _jobs[jid]["result"] = results
            except Exception as exc:
                _jobs[jid]["status"] = "failed"
                _jobs[jid]["error"] = str(exc)

        thread = threading.Thread(target=_run_evaluation, args=(job_id, request), daemon=True)
        thread.start()

        return EvaluateResponse(
            job_id=job_id,
            status="started",
            message=f"Evaluation job {job_id} started in background.",
        )

    # ------------------------------------------------------------------ #
    # GET /jobs/{job_id}
    # ------------------------------------------------------------------ #
    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        responses={404: {"model": ErrorResponse}},
        tags=["Evaluation"],
    )
    async def get_job_status(job_id: str):
        """Check the status of a background evaluation job."""
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        job = _jobs[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            result=job.get("result"),
            error=job.get("error"),
        )

    # ------------------------------------------------------------------ #
    # GET /results
    # ------------------------------------------------------------------ #
    @app.get("/results", response_model=ResultListResponse, tags=["Results"])
    async def list_results(
        output_dir: str = Query("./beyondbench_results", description="Directory to scan for results"),
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
        responses={404: {"model": ErrorResponse}},
        tags=["Results"],
    )
    async def get_result(
        result_id: str,
        output_dir: str = Query("./beyondbench_results", description="Directory to scan for results"),
    ):
        """Get detailed data for a specific evaluation result."""
        results_dir = Path(output_dir)
        # Try exact sub-directory match first
        candidate = results_dir / result_id / "final_results.json"
        if not candidate.exists():
            # Fallback: search for any matching path
            for f in results_dir.rglob("final_results.json"):
                if f.parent.name == result_id:
                    candidate = f
                    break
            else:
                raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found")

        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error reading result file: {exc}")

        return ResultDetailResponse(result_id=result_id, data=data)

    return app

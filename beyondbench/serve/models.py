"""
Pydantic models for BeyondBench API request/response schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Request Models ---


class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""

    model_id: str = Field(..., description="Model identifier (HuggingFace path or API model name)")
    backend: Optional[str] = Field(
        None, description="Backend to use: vllm, transformers, openai, gemini"
    )
    api_provider: Optional[str] = Field(
        None, description="API provider: openai, gemini, anthropic"
    )
    api_key: Optional[str] = Field(None, description="API key for cloud models")
    suite: str = Field("all", description="Task suite: easy, medium, hard, all")
    tasks: Optional[List[str]] = Field(None, description="Specific tasks to evaluate")
    datapoints: int = Field(100, description="Number of datapoints per task")
    folds: int = Field(1, description="Number of cross-validation folds")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    max_tokens: int = Field(32768, description="Maximum tokens to generate")
    list_sizes: Optional[List[int]] = Field(None, description="List sizes for scalable tasks")
    range_min: int = Field(-100, description="Minimum value for number generation")
    range_max: int = Field(100, description="Maximum value for number generation")
    output_dir: str = Field("./beyondbench_results", description="Output directory for results")
    store_details: bool = Field(False, description="Store detailed per-example results")
    reasoning_effort: str = Field("medium", description="OpenAI reasoning effort level")
    thinking_budget: int = Field(1024, description="Gemini thinking budget")
    cuda_device: str = Field("cuda:0", description="CUDA device for local models")
    tensor_parallel_size: int = Field(1, description="Tensor parallelism GPUs")
    gpu_memory_utilization: float = Field(0.96, description="GPU memory utilization ratio")
    trust_remote_code: bool = Field(False, description="Allow remote code execution")
    max_retries: int = Field(3, description="Maximum retries for failed operations")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "backend": "vllm",
                    "suite": "easy",
                    "datapoints": 50,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "cuda_device": "cuda:0",
                    "gpu_memory_utilization": 0.45,
                }
            ]
        }
    }


class BatchEvaluateRequest(BaseModel):
    """Request body for POST /evaluate/batch — submit multiple evaluation configs."""

    jobs: List[EvaluateRequest] = Field(
        ..., description="List of evaluation configurations to run", min_length=1, max_length=20
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "jobs": [
                        {
                            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                            "backend": "vllm",
                            "suite": "easy",
                            "datapoints": 10,
                            "cuda_device": "cuda:0",
                            "gpu_memory_utilization": 0.45,
                        },
                        {
                            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                            "backend": "vllm",
                            "suite": "easy",
                            "datapoints": 10,
                            "cuda_device": "cuda:1",
                            "gpu_memory_utilization": 0.45,
                        },
                    ]
                }
            ]
        }
    }


class CompareRequest(BaseModel):
    """Request body for POST /compare — compare multiple evaluation results."""

    result_ids: List[str] = Field(
        ..., description="List of result IDs or job IDs to compare", min_length=2
    )
    output_dir: str = Field(
        "./beyondbench_results", description="Directory to scan for results"
    )
    metrics: Optional[List[str]] = Field(
        None, description="Specific metrics to include (default: all)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "result_ids": ["run_20240101_qwen05b", "run_20240101_qwen15b"],
                    "output_dir": "./beyondbench_results",
                    "metrics": ["overall_accuracy", "task_accuracy"],
                }
            ]
        }
    }


# --- Response Models ---


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str
    gpu_count: int = Field(0, description="Number of available GPUs")
    uptime: float = Field(0.0, description="Server uptime in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "ok", "version": "0.2.0", "gpu_count": 8, "uptime": 3600.5}
            ]
        }
    }


class TaskInfo(BaseModel):
    """Information about a single task."""

    name: str
    suite: str
    type: str
    available: bool
    description: str


class TaskListResponse(BaseModel):
    """Response for GET /tasks."""

    tasks: Dict[str, List[str]]
    total: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tasks": {
                        "easy": ["sum", "sorting", "maximum"],
                        "medium": ["matrix_transpose"],
                        "hard": ["graph_bfs"],
                    },
                    "total": 6,
                }
            ]
        }
    }


class TaskInfoResponse(BaseModel):
    """Response for GET /tasks/{task_name}."""

    task: TaskInfo

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task": {
                        "name": "sum",
                        "suite": "easy",
                        "type": "numeric",
                        "available": True,
                        "description": "Sum a list of integers",
                    }
                }
            ]
        }
    }


class EvaluateResponse(BaseModel):
    """Response for POST /evaluate."""

    job_id: str
    status: str = "started"
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "started",
                    "message": "Evaluation job 550e8400-e29b-41d4-a716-446655440000 started in background.",
                }
            ]
        }
    }


class BatchEvaluateResponse(BaseModel):
    """Response for POST /evaluate/batch."""

    job_ids: List[str]
    total: int
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_ids": [
                        "550e8400-e29b-41d4-a716-446655440000",
                        "550e8400-e29b-41d4-a716-446655440001",
                    ],
                    "total": 2,
                    "message": "2 evaluation jobs started.",
                }
            ]
        }
    }


class JobStatusResponse(BaseModel):
    """Response for checking job status."""

    job_id: str
    status: str = Field(..., description="Job status: running, completed, failed, cancelled")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    progress: Optional[Dict[str, Any]] = Field(
        None, description="Live progress data: {task, completed, total, percent}"
    )
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "running",
                    "created_at": "2024-01-01T12:00:00",
                    "updated_at": "2024-01-01T12:01:30",
                    "progress": {"task": "sum", "completed": 50, "total": 100, "percent": 0.5},
                    "result": None,
                    "error": None,
                }
            ]
        }
    }


class JobListResponse(BaseModel):
    """Response for GET /jobs."""

    jobs: List[JobStatusResponse]
    total: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "jobs": [
                        {
                            "job_id": "550e8400-e29b-41d4-a716-446655440000",
                            "status": "completed",
                        }
                    ],
                    "total": 1,
                }
            ]
        }
    }


class DeleteJobResponse(BaseModel):
    """Response for DELETE /jobs/{job_id}."""

    job_id: str
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "message": "Job 550e8400-e29b-41d4-a716-446655440000 cancelled.",
                }
            ]
        }
    }


class CompareResponse(BaseModel):
    """Response for POST /compare."""

    comparison: Dict[str, Any]
    result_ids: List[str]
    metrics_compared: List[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "result_ids": ["run_qwen05b", "run_qwen15b"],
                    "metrics_compared": ["overall_accuracy"],
                    "comparison": {
                        "run_qwen05b": {"overall_accuracy": 0.72},
                        "run_qwen15b": {"overall_accuracy": 0.81},
                    },
                }
            ]
        }
    }


class ResultSummary(BaseModel):
    """Summary of a single evaluation result."""

    result_id: str
    path: str
    summary: Optional[Dict[str, Any]] = None


class ResultListResponse(BaseModel):
    """Response for GET /results."""

    results: List[ResultSummary]
    total: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "result_id": "qwen05b_easy_20240101",
                            "path": "./beyondbench_results/qwen05b_easy_20240101/final_results.json",
                            "summary": {"overall_accuracy": 0.72},
                        }
                    ],
                    "total": 1,
                }
            ]
        }
    }


class ResultDetailResponse(BaseModel):
    """Response for GET /results/{result_id}."""

    result_id: str
    data: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "result_id": "qwen05b_easy_20240101",
                    "data": {
                        "summary": {"overall_accuracy": 0.72},
                        "task_results": {},
                    },
                }
            ]
        }
    }


class JobResultResponse(BaseModel):
    """Response for GET /jobs/{job_id}/results — completed job results only."""

    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed",
                    "result": {"summary": {"overall_accuracy": 0.72}},
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str

    model_config = {
        "json_schema_extra": {"examples": [{"detail": "Job '123' not found"}]}
    }

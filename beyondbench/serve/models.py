"""
Pydantic models for BeyondBench API request/response schemas.
"""

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


# --- Response Models ---


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str


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


class TaskInfoResponse(BaseModel):
    """Response for GET /tasks/{task_name}."""

    task: TaskInfo


class EvaluateResponse(BaseModel):
    """Response for POST /evaluate."""

    job_id: str
    status: str = "started"
    message: str


class JobStatusResponse(BaseModel):
    """Response for checking job status."""

    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResultSummary(BaseModel):
    """Summary of a single evaluation result."""

    result_id: str
    path: str
    summary: Optional[Dict[str, Any]] = None


class ResultListResponse(BaseModel):
    """Response for GET /results."""

    results: List[ResultSummary]
    total: int


class ResultDetailResponse(BaseModel):
    """Response for GET /results/{result_id}."""

    result_id: str
    data: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str

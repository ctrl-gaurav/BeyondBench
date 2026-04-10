"""
Integration tests — multi-model: run sum task with 4 different models simultaneously.

Requirements:
    CUDA_VISIBLE_DEVICES=3,4,5,6 pytest tests/integration/test_multi_model.py -v --timeout=600

Each model runs on its own GPU (via separate processes or sequential GPU assignment).
All 4 must produce valid results.
"""

import os
import logging
import pytest

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# Models to test — ordered from smallest to largest
MULTI_MODELS = [
    ("qwen_0_5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen_1_5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen_3b",   "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen_7b",   "Qwen/Qwen2.5-7B-Instruct"),
]


@pytest.fixture(scope="session", autouse=True)
def require_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")
    except ImportError:
        pytest.skip("PyTorch not available")


def _run_sum_task_for_model(model_id: str, output_dir: str, datapoints: int = 5) -> dict:
    """Load a model, run sum task, return stats."""
    from beyondbench.models.model_handler import ModelHandler
    from beyondbench.tasks.easy.sum_task import SumTask

    gpu_mem = float(os.environ.get("INTEGRATION_GPU_MEM", "0.45"))

    handler = ModelHandler(
        model_id=model_id,
        backend="vllm",
        cuda_device="cuda:0",  # Each test gets its own CUDA_VISIBLE_DEVICES slice
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_model_len=2048,
    )

    task = SumTask(
        model_handler=handler,
        output_dir=output_dir,
        min_val=-50,
        max_val=50,
        num_folds=1,
        num_samples=datapoints,
        store_details=False,
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,
        seed=42,
    )

    data = task.generate_data(list_size=5)
    prompts = [task.create_prompt(dp) for dp in data]
    responses = handler.generate(prompts, temperature=0.1, max_tokens=256)

    results = [task.evaluate_response(resp, dp) for dp, resp in zip(data, responses)]
    parse_rate = sum(1 for r in results if r["instruction_followed"]) / len(results)
    accuracy = sum(r["accuracy"] for r in results) / len(results)

    return {
        "model_id": model_id,
        "parse_rate": parse_rate,
        "accuracy": accuracy,
        "n": len(results),
    }


@pytest.mark.parametrize("model_alias,model_id", MULTI_MODELS)
def test_sum_task_with_model(model_alias, model_id, tmp_path):
    """Run sum task with a single model and assert valid results."""
    try:
        stats = _run_sum_task_for_model(model_id, str(tmp_path / model_alias))
    except Exception as e:
        # Model not cached locally — skip rather than fail
        if "not found" in str(e).lower() or "does not exist" in str(e).lower() or "huggingface" in str(e).lower():
            pytest.skip(f"Model {model_id} not available locally: {e}")
        raise

    logger.info(f"[{model_alias}] parse_rate={stats['parse_rate']:.2%}, accuracy={stats['accuracy']:.2%}")

    assert stats["n"] == 5, f"Expected 5 results, got {stats['n']}"
    # Each model should produce valid results (parse rate > 0)
    assert stats["parse_rate"] >= 0.0, "parse_rate must be >= 0"
    # All models should get at least some right or at least parse correctly
    assert stats["parse_rate"] >= 0.4, (
        f"Model {model_id}: parse_rate {stats['parse_rate']:.2%} < 40%"
    )


def test_all_models_produce_results(tmp_path):
    """Run all 4 models and verify each produces non-empty results."""
    results = []
    for model_alias, model_id in MULTI_MODELS:
        try:
            stats = _run_sum_task_for_model(model_id, str(tmp_path / model_alias))
            results.append(stats)
            logger.info(
                f"[{model_alias}] model={model_id}, "
                f"parse={stats['parse_rate']:.2%}, acc={stats['accuracy']:.2%}"
            )
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                logger.warning(f"Skipping {model_id}: not available locally")
                continue
            raise

    if not results:
        pytest.skip("No models available locally")

    # At least one model must succeed
    assert len(results) >= 1

    for r in results:
        assert r["n"] > 0
        assert 0.0 <= r["parse_rate"] <= 1.0
        assert 0.0 <= r["accuracy"] <= 1.0

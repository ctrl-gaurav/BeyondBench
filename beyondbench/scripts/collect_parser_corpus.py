"""
Collect real model response samples for the parser corpus.

Runs BeyondBench evaluations and saves {response, expected_answer, task_name, model_id}
for each sample. Used to build the ground-truth corpus for parser testing.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m beyondbench.scripts.collect_parser_corpus \
        --model-id Qwen/Qwen2.5-1.5B-Instruct \
        --tasks sum,sorting,comparison \
        --datapoints 20 \
        --backend vllm \
        --output tests/parser_corpus/
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Collect parser corpus samples")
    p.add_argument("--model-id", required=True, help="HuggingFace model ID or API model name")
    p.add_argument("--tasks", required=True, help="Comma-separated task names")
    p.add_argument("--datapoints", type=int, default=20, help="Datapoints per task")
    p.add_argument("--backend", default="vllm", choices=["vllm", "transformers", "openai", "gemini"])
    p.add_argument("--output", default="tests/parser_corpus", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def get_model_family(model_id: str) -> str:
    """Infer model family from model_id."""
    low = model_id.lower()
    for family in ["qwen", "llama", "phi", "mistral", "gemma", "gpt", "gemini", "claude"]:
        if family in low:
            return family
    return "unknown"


def collect_samples(args):
    """
    Run evaluation for each task and collect (response, expected_answer) pairs.
    Saves as JSON samples in the corpus directory.
    """
    from beyondbench.models.model_handler import ModelHandler
    from beyondbench.core.task_registry import TaskRegistry
    from beyondbench.parsers.core import UnifiedParser
    from beyondbench.parsers.task_configs import get_task_config

    output_dir = Path(args.output) / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = [t.strip() for t in args.tasks.split(",")]
    model_family = get_model_family(args.model_id)

    logger.info("Initializing model: %s (backend=%s)", args.model_id, args.backend)
    model_handler = ModelHandler(
        model_id=args.model_id,
        backend=args.backend,
    )

    registry = TaskRegistry()
    all_samples = []

    for task_name in task_names:
        logger.info("Collecting samples for task: %s", task_name)
        task_dir = output_dir / task_name / model_family
        task_dir.mkdir(parents=True, exist_ok=True)

        task_class = registry.get_task(task_name)
        if task_class is None:
            logger.warning("Task not found: %s — skipping", task_name)
            continue

        task_instance = task_class(
            model_handler=model_handler,
            num_samples=args.datapoints,
            seed=args.seed,
        )

        config = get_task_config(task_name)
        parser = UnifiedParser(config)

        # Generate data and get model responses
        try:
            data = task_instance.generate_data(8)
        except TypeError:
            try:
                data = task_instance.generate_data()
            except Exception as e:
                logger.error("generate_data failed for %s: %s", task_name, e)
                continue

        data = data[:args.datapoints]

        for idx, data_point in enumerate(data):
            prompt = task_instance.create_prompt(data_point)
            try:
                response = model_handler.generate(prompt)
            except Exception as e:
                logger.warning("Generation failed for %s sample %d: %s", task_name, idx, e)
                continue

            # Ground truth
            try:
                eval_result = task_instance.evaluate_response(response, data_point)
                expected_answer = eval_result.get("accuracy", None)
                # Get the actual ground truth value from eval_result
                for key in ["sum", "result", "answer", "expected", "ground_truth"]:
                    if key in eval_result:
                        expected_answer = eval_result[key]
                        break
            except Exception:
                expected_answer = None

            # Parse with unified parser
            unified_result = parser.parse(response, model_id=args.model_id)

            sample = {
                "task_name": task_name,
                "model_id": args.model_id,
                "model_family": model_family,
                "response": response,
                "expected_answer": expected_answer,
                "expected_type": config.expected_type,
                "unified_parse": unified_result.value,
                "unified_confidence": unified_result.confidence,
                "unified_strategy": unified_result.strategy_used,
            }

            sample_path = task_dir / f"sample_{idx:04d}.json"
            with open(sample_path, "w") as f:
                json.dump(sample, f, indent=2, default=str)

            all_samples.append(sample)
            logger.info("  [%s/%d] strategy=%s conf=%.2f",
                        idx + 1, len(data), unified_result.strategy_used, unified_result.confidence)

    # Save corpus stats
    stats = {
        "total_samples": len(all_samples),
        "model_id": args.model_id,
        "model_family": model_family,
        "tasks": task_names,
        "by_task": {
            task: sum(1 for s in all_samples if s["task_name"] == task)
            for task in task_names
        },
        "by_strategy": {},
    }
    for s in all_samples:
        strat = s.get("unified_strategy", "none")
        stats["by_strategy"][strat] = stats["by_strategy"].get(strat, 0) + 1

    stats_path = Path(args.output) / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Collected %d samples → %s", len(all_samples), args.output)
    return all_samples


if __name__ == "__main__":
    args = parse_args()
    collect_samples(args)

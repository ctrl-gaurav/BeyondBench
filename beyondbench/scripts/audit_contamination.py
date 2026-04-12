"""Audit all tasks for contamination resistance.

For each task:
1. Generate data with seed=42
2. Generate data with seed=43
3. Verify the two datasets differ (seeds work)
4. Compute fingerprints and check uniqueness within each dataset
5. Report results

Usage:
    python -m beyondbench.scripts.audit_contamination
    # or
    python beyondbench/scripts/audit_contamination.py
"""
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _make_stub_handler():
    """Create a minimal stub model handler for data-generation-only audit."""

    class _StubHandler:
        backend = "stub"

        def get_model_info(self):
            return {"model_name": "stub", "backend": "stub"}

        def generate(self, prompts, **kwargs):
            return [""] * len(prompts)

        def get_statistics(self):
            return {}

    return _StubHandler()


def _generate_task_data(task_class, seed: int, output_dir: str = "/tmp/bb_audit") -> Optional[List[Any]]:
    """Instantiate task and generate data.  Returns None on failure."""
    import inspect

    stub = _make_stub_handler()
    init_sig = inspect.signature(task_class.__init__)
    init_params = set(init_sig.parameters.keys()) - {"self"}
    accepts_kwargs = "kwargs" in init_params

    common_kwargs: Dict[str, Any] = {
        "model_handler": stub,
        "output_dir": output_dir,
        "num_folds": 1,
        "num_samples": 5,
        "store_details": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "seed": seed,
        "prompt_style": None,
        "contamination_resistance": "off",
    }

    # Add range params if needed
    if "min_val" in init_params or accepts_kwargs:
        common_kwargs["min_val"] = 1
        common_kwargs["max_val"] = 100

    # Hard task defaults
    hard_defaults = {
        "board_sizes": [4],
        "num_disks_list": [3],
        "num_variables_list": [4],
        "sat_types_list": ["random_k"],
        "clause_ratios_list": [3.0],
        "graph_types": ["random"],
        "num_vertices_list": [5],
        "grid_sizes": [3],
        "difficulty_levels": ["easy"],
        "word_lengths": [3],
        "puzzle_types": ["addition"],
        "num_equations_list": [2],
        "constraint_types": ["standard"],
        "matrix_counts_list": [3],
        "dimension_patterns": ["uniform"],
        "num_projects_list": [3],
        "technique_levels": ["basic"],
    }
    for param, default in hard_defaults.items():
        if param in init_params:
            common_kwargs[param] = default

    if accepts_kwargs:
        filtered = common_kwargs
    else:
        filtered = {k: v for k, v in common_kwargs.items() if k in init_params}

    try:
        task = task_class(**filtered)
    except TypeError as e:
        return None

    try:
        data = task.generate_data()
    except Exception:
        try:
            data = task.generate_data(list_size=8)
        except Exception:
            return None

    if not isinstance(data, list):
        return None
    return data


def audit_all_tasks(output_dir: str = "/tmp/bb_audit", verbose: bool = False) -> Dict[str, Any]:
    """Run contamination resistance audit on all tasks.

    Returns a summary dict with per-task results and aggregate statistics.
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    from beyondbench.core.task_registry import TaskRegistry
    from beyondbench.core.fingerprint import fingerprint_instance

    registry = TaskRegistry()
    all_task_names = []
    for suite in ("easy", "medium", "hard"):
        all_task_names.extend(registry.get_tasks_for_suite(suite))

    results: Dict[str, Any] = {}
    summary = {
        "total": len(all_task_names),
        "passed": 0,
        "failed_seed_variance": 0,
        "failed_generation": 0,
        "failed_fingerprint_collision": 0,
    }

    for task_name in all_task_names:
        task_class = registry.get_task_class(task_name)
        if task_class is None:
            results[task_name] = {"status": "SKIP", "reason": "class not found"}
            continue

        # Generate with two different seeds
        data_42 = _generate_task_data(task_class, seed=42, output_dir=output_dir)
        data_43 = _generate_task_data(task_class, seed=43, output_dir=output_dir)

        if data_42 is None or data_43 is None:
            results[task_name] = {"status": "FAIL", "reason": "generation failed"}
            summary["failed_generation"] += 1
            if verbose:
                print(f"  FAIL  {task_name}: generation failed")
            continue

        # Check seed variance: at least one data point should differ
        def _normalize(d):
            return json.dumps(d, sort_keys=True, default=str)

        pairs_differ = any(
            _normalize(a) != _normalize(b)
            for a, b in zip(data_42[:3], data_43[:3])
        )

        if not pairs_differ:
            results[task_name] = {
                "status": "WARN",
                "reason": "seed variance: data identical for seed=42 and seed=43",
                "seed42_sample": str(data_42[0])[:100] if data_42 else "",
            }
            summary["failed_seed_variance"] += 1
            if verbose:
                print(f"  WARN  {task_name}: seed variance issue")
            continue

        # Check fingerprint uniqueness within seed=42 dataset
        fingerprints = [fingerprint_instance(dp, task_name, seed=42) for dp in data_42]
        unique_fps = set(fingerprints)
        collision = len(fingerprints) != len(unique_fps)

        if collision:
            results[task_name] = {
                "status": "WARN",
                "reason": "fingerprint collision: duplicate data points detected",
                "fingerprints": fingerprints,
            }
            summary["failed_fingerprint_collision"] += 1
            if verbose:
                print(f"  WARN  {task_name}: fingerprint collision")
            continue

        results[task_name] = {
            "status": "PASS",
            "n_seed42": len(data_42),
            "n_seed43": len(data_43),
            "fingerprints_seed42": fingerprints,
            "seed_variance": True,
        }
        summary["passed"] += 1
        if verbose:
            print(f"  PASS  {task_name}")

    summary["pass_rate"] = summary["passed"] / max(1, summary["total"])
    return {"summary": summary, "tasks": results}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audit BeyondBench tasks for contamination resistance")
    parser.add_argument("--output-dir", default="/tmp/bb_audit", help="Scratch output directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Print JSON result")
    args = parser.parse_args()

    print("Running contamination resistance audit...")
    result = audit_all_tasks(output_dir=args.output_dir, verbose=args.verbose)
    summary = result["summary"]

    if args.json_out:
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print("Contamination Resistance Audit Results")
        print(f"{'='*60}")
        print(f"Total tasks:             {summary['total']}")
        print(f"Passed:                  {summary['passed']}")
        print(f"Failed (generation):     {summary['failed_generation']}")
        print(f"Failed (seed variance):  {summary['failed_seed_variance']}")
        print(f"Failed (fp collision):   {summary['failed_fingerprint_collision']}")
        print(f"Pass rate:               {summary['pass_rate']:.1%}")
        print(f"{'='*60}")

        failed = {n: r for n, r in result["tasks"].items() if r["status"] not in ("PASS", "SKIP")}
        if failed:
            print("\nTasks requiring attention:")
            for name, info in failed.items():
                print(f"  [{info['status']}] {name}: {info.get('reason', '')}")

    sys.exit(0 if summary["failed_generation"] == 0 else 1)


if __name__ == "__main__":
    main()

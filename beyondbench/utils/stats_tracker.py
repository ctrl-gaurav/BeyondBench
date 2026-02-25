"""Statistics tracking utilities."""

import time
from typing import Dict, Any, List
from collections import defaultdict


class StatsTracker:
    """Track comprehensive statistics for evaluations."""

    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(int))
        self.start_time = time.time()
        self.detailed_results = []

    def update_stats(self, result: Dict[str, Any]):
        """Update statistics with a single result."""
        task_name = result.get("task_name", "unknown")
        status = result.get("status", "unknown")

        self.stats[task_name]["total"] += 1
        self.stats[task_name][status] += 1

        if status == "success" and "evaluation" in result:
            eval_result = result["evaluation"]
            if "accuracy" in eval_result:
                self.stats[task_name]["total_accuracy"] += eval_result["accuracy"]

        self.detailed_results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary."""
        summary = {}
        total_time = time.time() - self.start_time

        for task_name, task_stats in self.stats.items():
            total = task_stats["total"]
            success = task_stats.get("success", 0)

            summary[task_name] = {
                "total_evaluations": total,
                "successful_evaluations": success,
                "success_rate": success / max(1, total),
                "average_accuracy": task_stats.get("total_accuracy", 0) / max(1, success)
            }

        summary["overall"] = {
            "total_time": total_time,
            "total_tasks": len(self.stats),
            "total_evaluations": sum(s["total"] for s in self.stats.values())
        }

        return summary
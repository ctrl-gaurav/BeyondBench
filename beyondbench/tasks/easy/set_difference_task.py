"""
Set Difference Task - elements in list1 not in list2 (sorted).
"""

import random
from ...core.base_task import BaseTask
from ...parsers.set_difference_parsing import parse_set_difference_answer


class SetDifferenceTask(BaseTask):
    """Find elements in list1 that are not in list2 (sorted)."""

    @property
    def task_name(self):
        return "set_difference"

    def generate_data(self, list_size=6):
        if self.seed is not None:
            random.seed(self.seed)
        data = []
        for _ in range(self.num_samples):
            full_pool = list(range(self.min_val, self.max_val + 1))
            list1 = random.sample(full_pool, min(list_size, len(full_pool)))
            list2 = random.sample(full_pool, min(list_size, len(full_pool)))
            answer = sorted(set(list1) - set(list2))
            data.append({"list1": list1, "list2": list2, "answer": answer})
        return data

    def create_prompt(self, data_point):
        list1 = data_point["list1"]
        list2 = data_point["list2"]
        return (
            f"Find the set difference of lists {list1} and {list2}. "
            f"The set difference contains elements that are in the first list but NOT in the second list. "
            f"Treat each list as a set (ignore duplicates). Return the result as a sorted list. "
            f"If the result is empty, return \\boxed{{[]}}.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point["answer"]
        parsed, instruction_followed = parse_set_difference_answer(response)
        accuracy = 0
        if parsed is not None:
            parsed_sorted = sorted(int(round(x)) for x in parsed)
            if parsed_sorted == ground_truth:
                accuracy = 1
        return {
            "list1": data_point["list1"],
            "list2": data_point["list2"],
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

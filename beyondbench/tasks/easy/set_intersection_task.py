"""
Set Intersection Task - elements in both lists (sorted).
"""

import random
from ...core.base_task import BaseTask
from ...parsers.set_intersection_parsing import parse_set_intersection_answer


class SetIntersectionTask(BaseTask):
    """Find the sorted intersection of two lists."""

    @property
    def task_name(self):
        return "set_intersection"

    def generate_data(self, list_size=6):
        if self.seed is not None:
            random.seed(self.seed)
        data = []
        for _ in range(self.num_samples):
            # Use overlapping ranges to guarantee non-empty intersection
            mid = (self.min_val + self.max_val) // 2
            # list1 drawn from full range, list2 drawn from overlapping range
            full_pool = list(range(self.min_val, self.max_val + 1))
            overlap_pool = list(range(mid - list_size, mid + list_size + 1))
            list1 = random.sample(full_pool, min(list_size, len(full_pool)))
            list2 = random.sample(overlap_pool, min(list_size, len(overlap_pool)))
            answer = sorted(set(list1) & set(list2))
            data.append({"list1": list1, "list2": list2, "answer": answer})
        return data

    def create_prompt(self, data_point):
        list1 = data_point["list1"]
        list2 = data_point["list2"]
        return (
            f"Find the intersection of the two lists {list1} and {list2}. "
            f"The intersection contains elements that appear in both lists (treating each list as a set). "
            f"Return the result as a sorted list. If the intersection is empty, return \\boxed{{[]}}.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point["answer"]
        parsed, instruction_followed = parse_set_intersection_answer(response)
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

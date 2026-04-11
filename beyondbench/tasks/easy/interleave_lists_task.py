"""
Interleave Lists Task - interleave two equal-length lists.
"""

import random
from ...core.base_task import BaseTask
from ...parsers.interleave_lists_parsing import parse_interleave_lists_answer


class InterleaveListsTask(BaseTask):
    """Interleave two equal-length lists element by element."""

    @property
    def task_name(self):
        return "interleave_lists"

    def generate_data(self, list_size=5):
        if self.seed is not None:
            random.seed(self.seed)
        data = []
        for _ in range(self.num_samples):
            pool = list(range(self.min_val, self.max_val + 1))
            if 2 * list_size > len(pool):
                list1 = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
                list2 = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            else:
                sampled = random.sample(pool, 2 * list_size)
                list1 = sampled[:list_size]
                list2 = sampled[list_size:]
            answer = []
            for a, b in zip(list1, list2):
                answer.append(a)
                answer.append(b)
            data.append({"list1": list1, "list2": list2, "answer": answer})
        return data

    def create_prompt(self, data_point):
        list1 = data_point["list1"]
        list2 = data_point["list2"]
        return (
            f"Interleave the two lists {list1} and {list2}. "
            f"Take elements alternately: first from list1, then from list2, then list1, etc. "
            f"The result should have {len(list1) + len(list2)} elements.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point["answer"]
        parsed, instruction_followed = parse_interleave_lists_answer(response)
        accuracy = 0
        if parsed is not None and len(parsed) == len(ground_truth):
            if all(int(round(a)) == b for a, b in zip(parsed, ground_truth)):
                accuracy = 1
        return {
            "list1": data_point["list1"],
            "list2": data_point["list2"],
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

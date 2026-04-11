"""
Dot Product Task - dot product of two equal-length lists.
"""

import random
from ...core.base_task import BaseTask
from ...utils.parsing import parse_sum


class DotProductTask(BaseTask):
    """Compute dot product of two equal-length lists."""

    @property
    def task_name(self):
        return "dot_product"

    def generate_data(self, list_size=5):
        if self.seed is not None:
            random.seed(self.seed)
        data = []
        for _ in range(self.num_samples):
            full_pool = list(range(self.min_val, self.max_val + 1))
            if 2 * list_size > len(full_pool):
                list1 = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
                list2 = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            else:
                sampled = random.sample(full_pool, 2 * list_size)
                list1 = sampled[:list_size]
                list2 = sampled[list_size:]
            answer = sum(a * b for a, b in zip(list1, list2))
            data.append({"list1": list1, "list2": list2, "answer": answer})
        return data

    def create_prompt(self, data_point):
        list1 = data_point["list1"]
        list2 = data_point["list2"]
        return (
            f"Calculate the dot product of the two lists {list1} and {list2}. "
            f"The dot product is the sum of element-wise products: sum(a[i] * b[i]) for all i.\n\n"
            f"Your final answer must be in the format \\boxed{{answer}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point["answer"]
        parsed = parse_sum(response)
        instruction_followed = parsed is not None
        accuracy = 0
        if instruction_followed:
            try:
                accuracy = 1 if int(round(float(parsed))) == ground_truth else 0
            except (TypeError, ValueError):
                pass
        return {
            "list1": data_point["list1"],
            "list2": data_point["list2"],
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

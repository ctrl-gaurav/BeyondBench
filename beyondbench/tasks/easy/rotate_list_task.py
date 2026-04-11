"""
Rotate List Task - rotate list right by k positions.
"""

import random
from ...core.base_task import BaseTask
from ...parsers.rotate_list_parsing import parse_rotate_list_answer


class RotateListTask(BaseTask):
    """Rotate a list right by k positions."""

    @property
    def task_name(self):
        return "rotate_list"

    def generate_data(self, list_size=8):
        if self.seed is not None:
            random.seed(self.seed)
        data = []
        for _ in range(self.num_samples):
            pool = range(self.min_val, self.max_val + 1)
            if list_size > len(pool):
                lst = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            else:
                lst = random.sample(pool, list_size)
            k = random.randint(1, list_size - 1)
            answer = lst[-k:] + lst[:-k]
            data.append({"list": lst, "k": k, "answer": answer})
        return data

    def create_prompt(self, data_point):
        lst = data_point["list"]
        k = data_point["k"]
        return (
            f"Rotate the list {lst} to the right by {k} position(s). "
            f"In a right rotation by k, the last k elements move to the front.\n\n"
            f"Your final answer must be in the format \\boxed{{[v1, v2, ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point["answer"]
        parsed, instruction_followed = parse_rotate_list_answer(response)
        accuracy = 0
        if parsed is not None and len(parsed) == len(ground_truth):
            if all(int(round(a)) == b for a, b in zip(parsed, ground_truth)):
                accuracy = 1
        return {
            "input_list": data_point["list"],
            "k": data_point["k"],
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

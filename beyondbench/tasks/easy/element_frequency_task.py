"""
Element Frequency Task - frequency of each unique element as [[val, count], ...] sorted by val.
"""

import random
from collections import Counter
from ...core.base_task import BaseTask
from ...parsers.element_frequency_parsing import parse_element_frequency_answer


class ElementFrequencyTask(BaseTask):
    """Count frequency of each unique element; output sorted [[val, count], ...]."""

    @property
    def task_name(self):
        return "element_frequency"

    def generate_data(self, list_size=8):
        if self.seed is not None:
            random.seed(self.seed)
        data = []
        for _ in range(self.num_samples):
            # Allow duplicates by using randint
            lst = [random.randint(self.min_val, self.max_val) for _ in range(list_size)]
            freq = Counter(lst)
            answer = sorted([[k, v] for k, v in freq.items()])
            data.append({"list": lst, "answer": answer})
        return data

    def create_prompt(self, data_point):
        lst = data_point["list"]
        return (
            f"Count the frequency of each unique element in the list {lst}. "
            f"Return the result as a list of [value, count] pairs sorted by value in ascending order. "
            f"For example, if 3 appears 2 times and 7 appears 1 time, the answer would be [[3, 2], [7, 1]].\n\n"
            f"Your final answer must be in the format \\boxed{{[[v1, c1], [v2, c2], ...]}} at the end of your response."
        )

    def evaluate_response(self, response, data_point):
        ground_truth = data_point["answer"]
        parsed, instruction_followed = parse_element_frequency_answer(response)
        accuracy = 0
        if parsed is not None:
            try:
                parsed_sorted = sorted([[int(round(p[0])), int(round(p[1]))] for p in parsed])
                if parsed_sorted == ground_truth:
                    accuracy = 1
            except (TypeError, IndexError, ValueError):
                pass
        return {
            "input_list": data_point["list"],
            "ground_truth": ground_truth,
            "predicted_answer": parsed,
            "accuracy": accuracy,
            "instruction_followed": int(instruction_followed),
        }

"""Noise injection for contamination resistance."""
import random
from typing import Any, Optional


class NoiseInjector:
    """Injects configurable noise into problem statements to prevent memorization.

    Levels:
    - low: minor formatting changes (spacing, number format)
    - medium: add irrelevant context/numbers, rephrase
    - high: add distractor information, change units/context, heavy rephrasing
    """

    LEVELS = ("low", "medium", "high")

    def __init__(self, level: str = "low", seed: Optional[int] = None):
        if level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}, got {level!r}")
        self.level = level
        self.rng = random.Random(seed)

    def inject(self, prompt: str, data_point: Any = None) -> str:
        """Apply noise to a prompt based on the configured level."""
        if self.level == "low":
            return self._inject_low(prompt)
        elif self.level == "medium":
            return self._inject_medium(prompt, data_point)
        else:
            return self._inject_high(prompt, data_point)

    def _inject_low(self, prompt: str) -> str:
        """Low noise: minor formatting variations."""
        # Randomly add a polite prefix sometimes
        prefixes = ["", "Please ", "Kindly ", ""]
        prefix = self.rng.choice(prefixes)
        if prefix and not prompt.lower().startswith(prefix.lower()):
            # Only add to first sentence — lower-case the first char to match
            prompt = prefix + prompt[0].lower() + prompt[1:]
        return prompt

    def _inject_medium(self, prompt: str, data_point: Any = None) -> str:
        """Medium noise: add context, rephrase slightly."""
        prompt = self._inject_low(prompt)
        # Add a random context sentence before the prompt
        contexts = [
            "This is a mathematical problem.",
            "Consider the following carefully.",
            "Pay close attention to the numbers.",
            "Think through this step by step.",
            "Here is a problem for you to solve.",
        ]
        context = self.rng.choice(contexts)
        prompt = context + "\n\n" + prompt
        return prompt

    def _inject_high(self, prompt: str, data_point: Any = None) -> str:
        """High noise: heavy context, distractors."""
        prompt = self._inject_medium(prompt, data_point)
        # Add distractor numbers — inserted BEFORE the \\boxed instruction so the
        # answer format is always the last meaningful line.
        distractors = [
            (
                f"\n\n(Note: The room temperature is {self.rng.randint(60, 85)}\u00b0F,"
                " which is irrelevant to this problem.)"
            ),
            (
                f"\n\n(Hint: Ignore any numbers not part of the main problem."
                f" For example, {self.rng.randint(100, 999)} is not relevant.)"
            ),
            (
                f"\n\n(Background: This problem was generated at timestamp"
                f" {self.rng.randint(1000000, 9999999)}, which has no bearing on the answer.)"
            ),
        ]
        distractor = self.rng.choice(distractors)
        # Insert before the \\boxed answer instruction to preserve parser compatibility
        if "\\boxed" in prompt:
            parts = prompt.rsplit("\\boxed", 1)
            prompt = parts[0] + distractor + "\n\n\\boxed" + parts[1]
        else:
            prompt += distractor
        return prompt

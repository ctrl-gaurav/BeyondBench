"""PromptTemplate: a named, style-tagged prompt with variable substitution."""

from __future__ import annotations

from typing import Any, Dict, Optional


VALID_STYLES = ("concise", "detailed", "few_shot", "cot")

# Model-family specific instruction suffixes.  Keyed by a substring
# that appears in the model_id (lower-cased).  Applied as a trailing
# hint *after* the main prompt body when the caller supplies a model_id.
MODEL_FAMILY_HINTS: Dict[str, str] = {
    "qwen": "Format your final answer exactly as \\boxed{answer} with no extra text inside the braces.",
    "llama": "Important: put ONLY the final answer inside \\boxed{}, with no explanation inside the braces.",
    "phi": "Respond with your final answer in \\boxed{answer}. Do not include units or explanation inside the box.",
    "mistral": "End your response with \\boxed{answer} containing only the numerical or required answer.",
    "gemma": "Your answer MUST end with \\boxed{answer}. Place only the answer value inside the braces.",
}


class PromptTemplate:
    """A named prompt template that supports Python-style {variable} substitution.

    Parameters
    ----------
    name:
        Human-readable name for the template (e.g. ``"sum_concise"``).
    template_str:
        Template string using ``{variable}`` placeholders.  Literal braces
        must be doubled (``{{`` / ``}}``).
    style:
        One of ``"concise"``, ``"detailed"``, ``"few_shot"``, or ``"cot"``.
    description:
        Optional human-readable description of this variant.
    """

    def __init__(
        self,
        name: str,
        template_str: str,
        style: str,
        description: str = "",
    ) -> None:
        if style not in VALID_STYLES:
            raise ValueError(
                f"style must be one of {VALID_STYLES}, got {style!r}"
            )
        self.name = name
        self.template_str = template_str
        self.style = style
        self.description = description

    # ------------------------------------------------------------------
    def render(self, *, model_id: Optional[str] = None, **kwargs: Any) -> str:
        """Fill in template variables and return the rendered prompt.

        Any keyword argument is available as a ``{variable}`` in the
        template.  Unknown variables are left as-is (no KeyError) so that
        partial rendering is safe.

        Parameters
        ----------
        model_id:
            Optional model identifier.  When provided, a model-family-specific
            hint is appended to improve instruction following for that family.
        """
        try:
            rendered = self.template_str.format(**kwargs)
        except KeyError:
            # Leave unfilled placeholders rather than crashing.
            import string
            formatter = string.Formatter()
            rendered = ""
            for literal_text, field_name, format_spec, conversion in formatter.parse(
                self.template_str
            ):
                rendered += literal_text
                if field_name is not None:
                    key = field_name.split(".")[0].split("[")[0]
                    if key in kwargs:
                        rendered += format(kwargs[key])
                    else:
                        # Re-insert the placeholder unchanged
                        fmt = f"{{{field_name}"
                        if conversion:
                            fmt += f"!{conversion}"
                        if format_spec:
                            fmt += f":{format_spec}"
                        fmt += "}"
                        rendered += fmt

        # Append model-family hint if applicable
        if model_id:
            model_lower = model_id.lower()
            for family_key, hint in MODEL_FAMILY_HINTS.items():
                if family_key in model_lower:
                    rendered = rendered.rstrip() + "\n\n" + hint
                    break

        return rendered

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, style={self.style!r})"

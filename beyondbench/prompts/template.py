"""PromptTemplate: a named, style-tagged prompt with variable substitution."""

from __future__ import annotations

from typing import Any


VALID_STYLES = ("concise", "detailed", "few_shot", "cot")


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
    def render(self, **kwargs: Any) -> str:
        """Fill in template variables and return the rendered prompt.

        Any keyword argument is available as a ``{variable}`` in the
        template.  Unknown variables are left as-is (no KeyError) so that
        partial rendering is safe.
        """
        try:
            return self.template_str.format(**kwargs)
        except KeyError as exc:
            # Leave unfilled placeholders rather than crashing.
            import string
            formatter = string.Formatter()
            result = ""
            for literal_text, field_name, format_spec, conversion in formatter.parse(
                self.template_str
            ):
                result += literal_text
                if field_name is not None:
                    key = field_name.split(".")[0].split("[")[0]
                    if key in kwargs:
                        result += format(kwargs[key])
                    else:
                        # Re-insert the placeholder unchanged
                        fmt = f"{{{field_name}"
                        if conversion:
                            fmt += f"!{conversion}"
                        if format_spec:
                            fmt += f":{format_spec}"
                        fmt += "}"
                        result += fmt
            return result

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, style={self.style!r})"

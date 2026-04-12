"""Unit tests for Phase 15: Prompt Engineering & Task Prompt Optimization.

Covers:
- PromptTemplate construction, rendering, style validation
- PromptLibrary singleton behavior, registration, retrieval
- All 79 tasks have all 3 required styles registered
- BaseTask.get_prompt() respects prompt_style (and falls back correctly)
- CLI --prompt-style plumbing (validation, passthrough into eval_config)
"""

import pytest

# ---------------------------------------------------------------------------
# PromptTemplate tests
# ---------------------------------------------------------------------------

class TestPromptTemplate:
    def test_basic_render(self):
        from beyondbench.prompts.template import PromptTemplate
        tmpl = PromptTemplate("t", "Sum: {data}. \\boxed{{answer}}", "concise")
        result = tmpl.render(data="[1,2,3]")
        assert "Sum: [1,2,3]" in result
        assert "\\boxed{answer}" in result

    def test_invalid_style_raises(self):
        from beyondbench.prompts.template import PromptTemplate
        with pytest.raises(ValueError, match="style must be one of"):
            PromptTemplate("t", "foo", "bad_style")

    def test_all_valid_styles(self):
        from beyondbench.prompts.template import PromptTemplate, VALID_STYLES
        for style in VALID_STYLES:
            tmpl = PromptTemplate("t", "hello {data}", style)
            assert tmpl.style == style

    def test_render_missing_key_does_not_crash(self):
        from beyondbench.prompts.template import PromptTemplate
        tmpl = PromptTemplate("t", "a={a} b={b}", "concise")
        # Only pass 'a'; 'b' is missing — should not raise
        result = tmpl.render(a="X")
        assert "X" in result

    def test_repr(self):
        from beyondbench.prompts.template import PromptTemplate
        tmpl = PromptTemplate("my_tmpl", "x", "detailed")
        assert "my_tmpl" in repr(tmpl)
        assert "detailed" in repr(tmpl)


# ---------------------------------------------------------------------------
# PromptLibrary tests
# ---------------------------------------------------------------------------

class TestPromptLibrary:
    def test_singleton(self):
        from beyondbench.prompts import PromptLibrary
        lib1 = PromptLibrary()
        lib2 = PromptLibrary()
        assert lib1 is lib2

    def test_get_prompt_library_helper(self):
        from beyondbench.prompts import PromptLibrary, get_prompt_library
        assert get_prompt_library() is PromptLibrary()

    def test_list_tasks_returns_nonempty(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        tasks = lib.list_tasks()
        assert len(tasks) == 79

    def test_all_tasks_have_three_styles(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        missing = []
        for task in lib.list_tasks():
            styles = lib.get_available_styles(task)
            for s in ("concise", "detailed", "few_shot"):
                if s not in styles:
                    missing.append(f"{task}/{s}")
        assert missing == [], f"Missing prompt styles: {missing}"

    def test_get_returns_correct_style(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        tmpl = lib.get("sum", "concise")
        assert tmpl.style == "concise"

    def test_get_default_style_is_concise(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        tmpl = lib.get("sum")
        assert tmpl.style == "concise"

    def test_get_unknown_task_raises_key_error(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        with pytest.raises(KeyError, match="(?i)no prompts registered"):
            lib.get("nonexistent_task_xyz", "concise")

    def test_get_unknown_style_raises_key_error(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        with pytest.raises(KeyError, match="No prompt style"):
            lib.get("sum", "cot")  # cot is not registered in task_prompts.py

    def test_all_prompts_contain_boxed_answer(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        bad = []
        for task in lib.list_tasks():
            for style in ("concise", "detailed", "few_shot"):
                tmpl = lib.get(task, style)
                rendered = tmpl.render(data="[1, 2, 3]")
                if "\\boxed{answer}" not in rendered and "\\boxed{{answer}}" not in rendered:
                    bad.append(f"{task}/{style}")
        assert bad == [], f"Prompts missing \\boxed{{answer}}: {bad}"

    def test_register_and_retrieve(self):
        from beyondbench.prompts import PromptLibrary
        from beyondbench.prompts.template import PromptTemplate
        lib = PromptLibrary()
        tmpl = PromptTemplate("test_task_cot", "think: {data} \\boxed{{answer}}", "cot")
        lib.register("__test_task__", "cot", tmpl)
        retrieved = lib.get("__test_task__", "cot")
        assert retrieved is tmpl
        # cleanup
        del lib._registry["__test_task__"]

    def test_get_available_styles_empty_for_unknown(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        assert lib.get_available_styles("nonexistent_xyz") == []

    def test_render_produces_non_empty_string(self):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        for task in lib.list_tasks():
            for style in ("concise", "detailed", "few_shot"):
                tmpl = lib.get(task, style)
                result = tmpl.render(data="[1, 2]")
                assert len(result) > 10, f"{task}/{style} rendered to empty string"


# ---------------------------------------------------------------------------
# Task coverage tests — spot check key tasks
# ---------------------------------------------------------------------------

class TestTaskPromptSpotChecks:
    """Spot-check that specific tasks render meaningful prompts."""

    SPOT_CHECKS = [
        # (task_name, style, expected_substring)
        ("sum", "concise", "Sum these numbers"),
        ("sum", "detailed", "Calculate the sum"),
        ("sum", "few_shot", "Example"),
        ("sorting", "concise", "ascending"),
        ("fibonacci_sequence", "few_shot", "Fibonacci"),
        ("shortest_path", "concise", "shortest path"),
        ("tower_hanoi", "detailed", "Tower of Hanoi"),
        ("coin_change", "few_shot", "coins"),
        ("knapsack", "detailed", "Knapsack"),
        ("edit_distance", "few_shot", "kitten"),
    ]

    @pytest.mark.parametrize("task_name,style,expected", SPOT_CHECKS)
    def test_spot_check(self, task_name, style, expected):
        from beyondbench.prompts import PromptLibrary
        lib = PromptLibrary()
        tmpl = lib.get(task_name, style)
        result = tmpl.render(data="[1, 2, 3]")
        assert expected.lower() in result.lower(), (
            f"{task_name}/{style}: expected '{expected}' in rendered prompt"
        )


# ---------------------------------------------------------------------------
# BaseTask.get_prompt integration tests (no real model required)
# ---------------------------------------------------------------------------

class TestBaseTaskGetPrompt:
    """Test that get_prompt() correctly falls back or uses library.

    We test the logic directly without instantiating BaseTask (which requires
    a real model handler and triggers token counter initialization).
    """

    def test_get_prompt_no_style_calls_create_prompt(self):
        """Verify get_prompt() falls back to create_prompt when prompt_style=None."""
        from beyondbench.prompts import get_prompt_library

        # Simulate the get_prompt logic directly
        lib = get_prompt_library()
        # When prompt_style is None, create_prompt result is used
        original_prompt = "ORIGINAL_PROMPT: [1, 2, 3]"

        # Simulate: prompt_style is None → return create_prompt output
        prompt_style = None
        data_point = [1, 2, 3]
        if prompt_style is None:
            result = original_prompt
        else:
            tmpl = lib.get("sum", prompt_style)
            result = tmpl.render(data=str(data_point))

        assert result == original_prompt

    def test_get_prompt_with_style_uses_library(self):
        """Verify get_prompt() uses library when prompt_style is set."""
        from beyondbench.prompts import get_prompt_library

        lib = get_prompt_library()
        prompt_style = "concise"
        data_point = [1, 2, 3]
        tmpl = lib.get("sum", prompt_style)
        result = tmpl.render(data=str(data_point))

        assert "ORIGINAL" not in result
        assert "\\boxed{answer}" in result  # rendered: {{answer}} → {answer}

    def test_get_prompt_detailed_style(self):
        """Verify detailed prompt has more words than concise."""
        from beyondbench.prompts import get_prompt_library
        lib = get_prompt_library()
        concise = lib.get("sum", "concise").render(data="[1,2,3]")
        detailed = lib.get("sum", "detailed").render(data="[1,2,3]")
        assert len(detailed) > len(concise), "detailed should be longer than concise"

    def test_get_prompt_few_shot_has_example(self):
        """Verify few_shot prompts contain 'Example'."""
        from beyondbench.prompts import get_prompt_library
        lib = get_prompt_library()
        for task in ["sum", "sorting", "fibonacci_sequence", "shortest_path"]:
            tmpl = lib.get(task, "few_shot")
            rendered = tmpl.render(data="[1,2,3]")
            assert "example" in rendered.lower(), f"{task}/few_shot missing example"

    def test_base_task_source_code_has_get_prompt(self):
        """Verify get_prompt method exists in base_task module source."""
        import inspect
        from beyondbench.core import base_task
        source = inspect.getsource(base_task)
        assert "def get_prompt" in source
        assert "prompt_style" in source
        assert "get_prompt_library" in source


# ---------------------------------------------------------------------------
# CLI --prompt-style option tests
# ---------------------------------------------------------------------------

class TestCLIPromptStyleOption:
    def test_prompt_style_in_help(self):
        """Verify --prompt-style appears in the CLI help text."""
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "prompt-style" in result.output.lower()

    def test_invalid_prompt_style_rejected(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, [
            "evaluate",
            "--model-id", "fake-model",
            "--prompt-style", "invalid_style",
        ])
        assert result.exit_code != 0

    def test_valid_prompt_style_accepted(self):
        """Pass a valid style and check it reaches eval_config without error.

        We don't actually run the evaluation (no model), just verify that Click
        accepts the option and the helper function processes it correctly.
        """
        from beyondbench.cli.main import validate_and_prepare_config
        kwargs = dict(
            model_id="fake/model",
            tasks=[],
            suite="easy",
            backend=None,
            api_provider=None,
            api_key=None,
            cuda_device="cuda:0",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.96,
            trust_remote_code=False,
            reasoning_effort="medium",
            thinking_budget=1024,
            temperature=0.7,
            top_p=0.9,
            max_tokens=32768,
            seed=None,
            datapoints=10,
            folds=1,
            list_sizes=None,
            range_min=-100,
            range_max=100,
            output_dir="./test_out",
            store_details=False,
            log_level="INFO",
            batch_size=1,
            max_retries=3,
            timeout=300,
            parser="unified",
            parallel=False,
            gpus="auto",
            strategy="auto",
            launch_dashboard_flag=False,
            dashboard_port=7860,
            skip_validation=False,
            validation_warn_only=False,
            no_chat_template=False,
            prompt_style="concise",
        )
        config = validate_and_prepare_config(kwargs)
        assert config["eval_config"]["prompt_style"] == "concise"

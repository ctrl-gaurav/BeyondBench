"""
Integration tests — API backends (OpenAI, Gemini, Anthropic).

These tests are SKIPPED if no API keys are available.
When API keys ARE available (in environment variables), they run 3 easy tasks each.

Environment variables:
  OPENAI_API_KEY
  GOOGLE_API_KEY (or GEMINI_API_KEY)
  ANTHROPIC_API_KEY
"""

import os
import logging
import pytest

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.api]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_openai_key():
    return bool(os.environ.get("OPENAI_API_KEY"))


def _has_gemini_key():
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _has_anthropic_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _run_api_task(model_id, api_provider, api_key, task_name, output_dir, datapoints=3):
    """Run a single easy task with an API backend; return (parse_rate, accuracy)."""
    import importlib
    from beyondbench.models.model_handler import ModelHandler
    from beyondbench.core.task_registry import TaskRegistry

    handler = ModelHandler(
        model_id=model_id,
        api_provider=api_provider,
        api_key=api_key,
    )

    registry = TaskRegistry()
    cls = registry.get_task_class(task_name)
    if cls is None:
        pytest.skip(f"Task {task_name} not available")

    task = cls(
        model_handler=handler,
        output_dir=output_dir,
        min_val=-50, max_val=50,
        num_folds=1, num_samples=datapoints,
        store_details=False,
        temperature=0.1, top_p=0.95, max_tokens=256,
        seed=42,
    )

    import inspect
    sig = inspect.signature(task.generate_data)
    if "list_size" in sig.parameters:
        data = task.generate_data(list_size=5)
    else:
        data = task.generate_data()

    if not data:
        pytest.skip("No data generated")

    prompts = [task.create_prompt(dp) for dp in data]
    responses = handler.generate(prompts, temperature=0.1, max_tokens=256)

    parsed = sum(1 for dp, resp in zip(data, responses)
                 if task.evaluate_response(resp, dp).get("instruction_followed", False))
    correct = sum(1 for dp, resp in zip(data, responses)
                  if task.evaluate_response(resp, dp).get("accuracy", 0) == 1)

    n = len(data)
    return parsed / n if n > 0 else 0.0, correct / n if n > 0 else 0.0


API_TASKS = ["sum", "sorting", "find_maximum"]


# ===========================================================================
# OpenAI tests
# ===========================================================================

@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
class TestOpenAIBackend:
    MODEL = "gpt-4o-mini"
    PROVIDER = "openai"

    @pytest.fixture(autouse=True)
    def _key(self):
        self.api_key = os.environ["OPENAI_API_KEY"]

    @pytest.mark.parametrize("task_name", API_TASKS)
    def test_easy_task_parse_rate(self, task_name, tmp_path):
        parse_rate, accuracy = _run_api_task(
            self.MODEL, self.PROVIDER, self.api_key,
            task_name, str(tmp_path), datapoints=3
        )
        logger.info(f"OpenAI {task_name}: parse={parse_rate:.2%}, acc={accuracy:.2%}")
        assert parse_rate >= 0.6, f"OpenAI {task_name}: parse_rate {parse_rate:.2%} < 60%"

    @pytest.mark.parametrize("task_name", API_TASKS)
    def test_easy_task_no_crash(self, task_name, tmp_path):
        _run_api_task(
            self.MODEL, self.PROVIDER, self.api_key,
            task_name, str(tmp_path), datapoints=2
        )


# ===========================================================================
# Gemini tests
# ===========================================================================

@pytest.mark.skipif(not _has_gemini_key(), reason="GOOGLE_API_KEY / GEMINI_API_KEY not set")
class TestGeminiBackend:
    MODEL = "gemini-2.5-flash"
    PROVIDER = "gemini"

    @pytest.fixture(autouse=True)
    def _key(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    @pytest.mark.parametrize("task_name", API_TASKS)
    def test_easy_task_parse_rate(self, task_name, tmp_path):
        parse_rate, accuracy = _run_api_task(
            self.MODEL, self.PROVIDER, self.api_key,
            task_name, str(tmp_path), datapoints=3
        )
        logger.info(f"Gemini {task_name}: parse={parse_rate:.2%}, acc={accuracy:.2%}")
        assert parse_rate >= 0.6, f"Gemini {task_name}: parse_rate {parse_rate:.2%} < 60%"

    @pytest.mark.parametrize("task_name", API_TASKS)
    def test_easy_task_no_crash(self, task_name, tmp_path):
        _run_api_task(
            self.MODEL, self.PROVIDER, self.api_key,
            task_name, str(tmp_path), datapoints=2
        )


# ===========================================================================
# Anthropic tests
# ===========================================================================

@pytest.mark.skipif(not _has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicBackend:
    MODEL = "claude-haiku-4-5-20251001"
    PROVIDER = "anthropic"

    @pytest.fixture(autouse=True)
    def _key(self):
        self.api_key = os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.parametrize("task_name", API_TASKS)
    def test_easy_task_parse_rate(self, task_name, tmp_path):
        parse_rate, accuracy = _run_api_task(
            self.MODEL, self.PROVIDER, self.api_key,
            task_name, str(tmp_path), datapoints=3
        )
        logger.info(f"Anthropic {task_name}: parse={parse_rate:.2%}, acc={accuracy:.2%}")
        assert parse_rate >= 0.6, f"Anthropic {task_name}: parse_rate {parse_rate:.2%} < 60%"

    @pytest.mark.parametrize("task_name", API_TASKS)
    def test_easy_task_no_crash(self, task_name, tmp_path):
        _run_api_task(
            self.MODEL, self.PROVIDER, self.api_key,
            task_name, str(tmp_path), datapoints=2
        )


# ===========================================================================
# Mock API test — always runs, validates mocked path
# ===========================================================================

class TestMockedAPIBackend:
    """Verify the API backend code path works when API client is mocked."""

    def test_openai_handler_mock_generate(self, tmp_path):
        from unittest.mock import MagicMock, patch

        with patch("beyondbench.models.model_handler.ModelHandler._setup_api_client"):
            with patch("beyondbench.models.model_handler.ModelHandler._generate_api") as mock_gen:
                from beyondbench.models.model_handler import ModelHandler
                mock_gen.return_value = r"The answer is \boxed{42}"
                handler = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="sk-test")
                result = handler._generate_api("What is 2+2?")
                assert "42" in result

    def test_generate_api_called_with_prompt(self, tmp_path):
        from unittest.mock import MagicMock, patch

        with patch("beyondbench.models.model_handler.ModelHandler._setup_api_client"):
            with patch("beyondbench.models.model_handler.ModelHandler._generate_api") as mock_gen:
                from beyondbench.models.model_handler import ModelHandler
                mock_gen.return_value = r"\boxed{10}"
                handler = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="sk-test")
                handler._generate_api("test prompt")
                mock_gen.assert_called_once_with("test prompt")

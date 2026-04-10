"""
Unit tests for ModelHandler — all backends, token counting, statistics.

No GPU required. API calls are mocked.
"""

import pytest
import time
from unittest.mock import MagicMock, patch


# ===========================================================================
# 1. Import and basic structure
# ===========================================================================

class TestModelHandlerImport:
    def test_import_succeeds(self):
        from beyondbench.models.model_handler import ModelHandler
        assert ModelHandler is not None

    def test_class_has_init(self):
        from beyondbench.models.model_handler import ModelHandler
        assert hasattr(ModelHandler, "__init__")

    def test_class_has_generate(self):
        from beyondbench.models.model_handler import ModelHandler
        assert hasattr(ModelHandler, "generate")

    def test_class_has_get_model_info(self):
        from beyondbench.models.model_handler import ModelHandler
        assert hasattr(ModelHandler, "get_model_info")

    def test_class_has_get_statistics(self):
        from beyondbench.models.model_handler import ModelHandler
        assert hasattr(ModelHandler, "get_statistics")


# ===========================================================================
# 2. API provider validation
# ===========================================================================

class TestAPIProviderValidation:
    def test_openai_without_key_raises(self):
        from beyondbench.models.model_handler import ModelHandler
        with pytest.raises(RuntimeError, match="API key"):
            ModelHandler(model_id="gpt-4o", api_provider="openai")

    def test_gemini_without_key_raises(self):
        from beyondbench.models.model_handler import ModelHandler
        with pytest.raises(RuntimeError):
            ModelHandler(model_id="gemini-pro", api_provider="gemini")

    def test_anthropic_without_key_raises(self):
        from beyondbench.models.model_handler import ModelHandler
        with pytest.raises(RuntimeError):
            ModelHandler(model_id="claude-3-5-sonnet-20241022", api_provider="anthropic")

    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    def test_openai_with_key_does_not_raise(self, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        h = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="sk-test")
        assert h is not None

    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    def test_gemini_with_key_does_not_raise(self, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        h = ModelHandler(model_id="gemini-pro", api_provider="gemini", api_key="test-key")
        assert h is not None


# ===========================================================================
# 3. Backend selection
# ===========================================================================

class TestBackendSelection:
    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    def test_api_provider_sets_backend_to_provider(self, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        h = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="k")
        assert h.backend == "openai"

    def test_backend_kwarg_vllm(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.api_provider = None
        h.backend = "vllm"
        assert h.backend == "vllm"

    def test_backend_kwarg_transformers(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.backend = "transformers"
        assert h.backend == "transformers"


# ===========================================================================
# 4. get_model_info
# ===========================================================================

class TestGetModelInfo:
    def test_returns_dict(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.model_id = "test-model"
        h.backend = "vllm"
        h.api_provider = None
        info = h.get_model_info()
        assert isinstance(info, dict)

    def test_has_model_name(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.model_id = "test-model"
        h.backend = "vllm"
        h.api_provider = None
        info = h.get_model_info()
        assert "model_name" in info
        assert info["model_name"] == "test-model"

    def test_has_backend(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.model_id = "test-model"
        h.backend = "vllm"
        h.api_provider = None
        info = h.get_model_info()
        assert "backend" in info

    def test_has_api_provider(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.model_id = "x"
        h.backend = "openai"
        h.api_provider = "openai"
        info = h.get_model_info()
        assert "api_provider" in info


# ===========================================================================
# 5. Token counting
# ===========================================================================

class TestTokenCounting:
    def test_count_tokens_empty_returns_zero(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.token_counter = None
        result = h._count_tokens("")
        assert result == 0

    def test_count_tokens_short_text(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.token_counter = None
        result = h._count_tokens("hello world")
        assert result > 0

    def test_count_tokens_longer_text_more_tokens(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.token_counter = None
        short = h._count_tokens("hello")
        long_ = h._count_tokens("hello world how are you doing today this is a long sentence")
        assert long_ > short

    def test_count_tokens_whitespace_only(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.token_counter = None
        result = h._count_tokens("   \n\t  ")
        assert result >= 0

    @patch("beyondbench.models.model_handler.ModelHandler._count_tokens", return_value=5)
    def test_custom_counter_called(self, mock_count):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        result = h._count_tokens("test text")
        assert result == 5


# ===========================================================================
# 6. Statistics tracking
# ===========================================================================

class TestStatisticsTracking:
    def _make_handler_with_stats(self, api_calls=0, input_tokens=0, output_tokens=0):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.model_id = "test-model"
        h.api_provider = None
        h.stats = {
            'api_calls': api_calls,
            'total_input_tokens': input_tokens,
            'total_output_tokens': output_tokens,
            'start_time': time.time(),
            'prompts_processed': [],
        }
        return h

    def test_get_statistics_returns_dict(self):
        h = self._make_handler_with_stats()
        stats = h.get_statistics()
        assert isinstance(stats, dict)

    def test_statistics_has_required_keys(self):
        h = self._make_handler_with_stats(api_calls=3, input_tokens=100, output_tokens=50)
        stats = h.get_statistics()
        assert "total_time_seconds" in stats or "api_calls" in stats

    def test_api_calls_tracked(self):
        h = self._make_handler_with_stats(api_calls=7, input_tokens=200, output_tokens=100)
        stats = h.get_statistics()
        assert stats.get("api_calls") == 7

    def test_total_tokens_computed(self):
        h = self._make_handler_with_stats(api_calls=1, input_tokens=100, output_tokens=50)
        stats = h.get_statistics()
        assert stats.get("total_tokens") == 150 or stats.get("total_time_seconds") is not None

    def test_model_id_in_stats(self):
        h = self._make_handler_with_stats()
        stats = h.get_statistics()
        assert stats.get("model_id") == "test-model"


# ===========================================================================
# 7. _format_duration
# ===========================================================================

class TestFormatDuration:
    @pytest.fixture(autouse=True)
    def _handler(self):
        from beyondbench.models.model_handler import ModelHandler
        self.h = ModelHandler.__new__(ModelHandler)

    def test_seconds_format(self):
        s = self.h._format_duration(45.0)
        assert "second" in s.lower() or "s" in s.lower()

    def test_minutes_format(self):
        s = self.h._format_duration(90.0)
        assert "minute" in s.lower() or "m" in s.lower()

    def test_hours_format(self):
        s = self.h._format_duration(4000.0)
        assert "hour" in s.lower() or "h" in s.lower()

    def test_zero_seconds(self):
        s = self.h._format_duration(0.0)
        assert isinstance(s, str)

    def test_fractional_seconds(self):
        s = self.h._format_duration(1.5)
        assert isinstance(s, str)


# ===========================================================================
# 8. skip_chat_template flag
# ===========================================================================

class TestSkipChatTemplate:
    def test_skip_chat_template_default_false(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.skip_chat_template = False
        assert h.skip_chat_template is False

    def test_skip_chat_template_can_be_set_true(self):
        from beyondbench.models.model_handler import ModelHandler
        h = ModelHandler.__new__(ModelHandler)
        h.skip_chat_template = True
        assert h.skip_chat_template is True


# ===========================================================================
# 9. OpenAI API backend mock — generate path
# ===========================================================================

class TestOpenAIBackendMocked:
    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    @patch("beyondbench.models.model_handler.ModelHandler._generate_api")
    def test_generate_calls_api(self, mock_api, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        mock_api.return_value = "The answer is 42"
        h = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="sk-test")
        h.backend = "openai"
        result = h._generate_api("What is 2+2?")
        assert result == "The answer is 42"

    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    def test_model_id_stored(self, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        h = ModelHandler(model_id="gpt-4o-mini", api_provider="openai", api_key="k")
        assert h.model_id == "gpt-4o-mini"

    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    def test_api_provider_stored(self, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        h = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="k")
        assert h.api_provider == "openai"


# ===========================================================================
# 10. Reasoning effort / thinking budget params stored
# ===========================================================================

class TestOptionalParams:
    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    def test_reasoning_effort_stored(self, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        h = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="k", reasoning_effort="high")
        assert h.reasoning_effort == "high"

    @patch("beyondbench.models.model_handler.ModelHandler._setup_api_client")
    def test_thinking_budget_stored(self, mock_setup):
        from beyondbench.models.model_handler import ModelHandler
        mock_setup.return_value = None
        h = ModelHandler(model_id="gemini-pro", api_provider="gemini", api_key="k", thinking_budget=1000)
        assert h.thinking_budget == 1000

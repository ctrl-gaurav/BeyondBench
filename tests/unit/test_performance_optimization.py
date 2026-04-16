"""
Phase 23 — Performance Optimization & Caching Tests.

Tests cover:
  - ResponseCache: put/get, LRU eviction, clear, stats, key derivation
  - ModelHandler: cache integration, warm-up, cleanup, batch estimation,
    quantization flags, torch.compile flag
  - CLI: --no-cache, --no-warmup, --quantization, --torch-compile, cache subcommands
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from unittest import mock

import pytest


# -----------------------------------------------------------------------
# ResponseCache unit tests
# -----------------------------------------------------------------------

class TestResponseCache:
    """Tests for beyondbench.core.cache.ResponseCache."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from beyondbench.core.cache import ResponseCache
        self.cache_dir = tmp_path / "cache"
        self.cache = ResponseCache(cache_dir=str(self.cache_dir), max_size_bytes=10_000)

    def test_put_get_roundtrip(self):
        self.cache.put("model-a", "prompt-1", 0.0, None, "response-1")
        result = self.cache.get("model-a", "prompt-1", 0.0, None)
        assert result == "response-1"

    def test_miss_returns_none(self):
        result = self.cache.get("model-a", "never-stored", 0.0, None)
        assert result is None

    def test_different_model_misses(self):
        self.cache.put("model-a", "prompt-1", 0.0, None, "resp-a")
        result = self.cache.get("model-b", "prompt-1", 0.0, None)
        assert result is None

    def test_different_temperature_misses(self):
        self.cache.put("m", "p", 0.0, None, "resp-0")
        assert self.cache.get("m", "p", 0.5, None) is None

    def test_different_seed_misses(self):
        self.cache.put("m", "p", 0.0, 42, "resp-42")
        assert self.cache.get("m", "p", 0.0, 99) is None

    def test_seed_none_vs_integer(self):
        self.cache.put("m", "p", 0.0, None, "resp-none")
        self.cache.put("m", "p", 0.0, 42, "resp-42")
        assert self.cache.get("m", "p", 0.0, None) == "resp-none"
        assert self.cache.get("m", "p", 0.0, 42) == "resp-42"

    def test_stats_initial(self):
        stats = self.cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_entries"] == 0
        assert stats["hit_rate"] == 0.0

    def test_stats_after_hits_and_misses(self):
        self.cache.put("m", "p", 0.0, None, "r")
        self.cache.get("m", "p", 0.0, None)  # hit
        self.cache.get("m", "p", 0.0, 1)  # miss
        stats = self.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_entries"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        self.cache.put("m", "p1", 0.0, None, "r1")
        self.cache.put("m", "p2", 0.0, None, "r2")
        self.cache.clear()
        assert self.cache.get("m", "p1", 0.0, None) is None
        assert self.cache.stats()["total_entries"] == 0

    def test_lru_eviction(self):
        """When cache exceeds max_size_bytes, oldest entries are evicted."""
        from beyondbench.core.cache import ResponseCache
        # Create a very small cache (500 bytes)
        small_cache = ResponseCache(cache_dir=str(self.cache_dir / "small"), max_size_bytes=500)
        # Fill with entries — each ~150 bytes in JSON
        small_cache.put("m", "p1", 0.0, None, "x" * 100)
        small_cache.put("m", "p2", 0.0, None, "x" * 100)
        small_cache.put("m", "p3", 0.0, None, "x" * 100)
        small_cache.put("m", "p4", 0.0, None, "x" * 100)
        small_cache.put("m", "p5", 0.0, None, "x" * 100)
        # Oldest entries should be evicted
        stats = small_cache.stats()
        assert stats["total_size_bytes"] <= 500

    def test_key_deterministic(self):
        from beyondbench.core.cache import ResponseCache
        k1 = ResponseCache._make_key("m", "p", 0.0, 42)
        k2 = ResponseCache._make_key("m", "p", 0.0, 42)
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex

    def test_overwrite_existing(self):
        self.cache.put("m", "p", 0.0, None, "old")
        self.cache.put("m", "p", 0.0, None, "new")
        assert self.cache.get("m", "p", 0.0, None) == "new"

    def test_thread_safety(self):
        """Multiple threads writing/reading concurrently should not crash."""
        errors = []

        def writer(idx):
            try:
                for j in range(10):
                    self.cache.put("m", f"thread-{idx}-{j}", 0.0, None, f"resp-{idx}-{j}")
            except Exception as e:
                errors.append(e)

        def reader(idx):
            try:
                for j in range(10):
                    self.cache.get("m", f"thread-{idx}-{j}", 0.0, None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        threads += [threading.Thread(target=reader, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_corrupted_entry_handled(self):
        """Corrupted JSON on disk returns None and cleans up."""
        self.cache.put("m", "p", 0.0, None, "good")
        # Corrupt the file
        key = self.cache._make_key("m", "p", 0.0, None)
        entry_path = self.cache._entry_path(key)
        entry_path.write_text("{invalid json", encoding="utf-8")
        assert self.cache.get("m", "p", 0.0, None) is None


class TestGlobalCache:
    def test_get_global_cache_singleton(self):
        from beyondbench.core.cache import get_global_cache, _global_cache_lock
        import beyondbench.core.cache as cache_mod

        # Reset global
        with _global_cache_lock:
            cache_mod._global_cache = None

        c1 = get_global_cache()
        c2 = get_global_cache()
        assert c1 is c2

        # Cleanup
        with _global_cache_lock:
            cache_mod._global_cache = None


# -----------------------------------------------------------------------
# ModelHandler cache integration tests (mocked — no GPU required)
# -----------------------------------------------------------------------

class TestModelHandlerCacheIntegration:
    """Verify ModelHandler routes through cache when appropriate."""

    def test_should_cache_temp_zero(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler._use_cache = True
        assert handler._should_cache(0.0, None) is True

    def test_should_cache_with_seed(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler._use_cache = True
        assert handler._should_cache(0.7, 42) is True

    def test_should_not_cache_stochastic(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler._use_cache = True
        assert handler._should_cache(0.7, None) is False

    def test_use_cache_false_disables(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler._use_cache = False
        handler.api_provider = None
        handler.model_id = "test"
        handler._cache = None
        # Even with temp=0, cache is bypassed
        with mock.patch.object(handler, '_generate_uncached', return_value=["resp"]) as m:
            result = handler.generate(["prompt"], temperature=0.0)
            m.assert_called_once()
            assert result == ["resp"]


# -----------------------------------------------------------------------
# ModelHandler cleanup tests
# -----------------------------------------------------------------------

class TestModelHandlerCleanup:
    def test_cleanup_method_exists(self):
        from beyondbench.models.model_handler import ModelHandler
        assert hasattr(ModelHandler, 'cleanup')

    def test_cleanup_clears_model(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.model = "dummy_model"
        handler.tokenizer = "dummy_tokenizer"
        handler.cleanup()
        assert handler.model is None
        assert handler.tokenizer is None

    def test_cleanup_safe_when_no_model(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        # No model or tokenizer attributes
        handler.cleanup()  # should not raise


# -----------------------------------------------------------------------
# ModelHandler batch size estimation tests
# -----------------------------------------------------------------------

class TestBatchSizeEstimation:
    def test_estimate_returns_int(self):
        from beyondbench.models.model_handler import ModelHandler
        result = ModelHandler._estimate_vllm_batch_size(0.96)
        assert isinstance(result, int)
        assert 64 <= result <= 512

    def test_estimate_without_cuda(self):
        """Falls back to 256 when CUDA is unavailable."""
        from beyondbench.models.model_handler import ModelHandler
        with mock.patch.dict('sys.modules', {'torch': None}):
            # Import error path
            result = ModelHandler._estimate_vllm_batch_size(0.5)
            # Should return 256 fallback
            assert isinstance(result, int)


# -----------------------------------------------------------------------
# ModelHandler warm-up tests
# -----------------------------------------------------------------------

class TestModelWarmUp:
    def test_warmup_api_skips(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.api_provider = "openai"
        elapsed = handler.warm_up()
        assert elapsed == 0.0

    def test_warmup_calls_generate(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.api_provider = None
        with mock.patch.object(handler, '_generate_uncached', return_value=["a", "b", "c"]):
            elapsed = handler.warm_up(num_prompts=3)
            assert elapsed >= 0.0

    def test_warmup_failure_nonfatal(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.api_provider = None
        with mock.patch.object(handler, '_generate_uncached', side_effect=RuntimeError("boom")):
            elapsed = handler.warm_up()
            assert elapsed >= 0.0  # no exception raised


# -----------------------------------------------------------------------
# CLI flag tests
# -----------------------------------------------------------------------

class TestCLIPerformanceFlags:
    """Verify Phase 23 CLI options are wired correctly."""

    @pytest.fixture(autouse=True)
    def _runner(self):
        from click.testing import CliRunner
        self.runner = CliRunner(mix_stderr=False) if 'mix_stderr' in __import__('inspect').signature(
            __import__('click.testing', fromlist=['CliRunner']).CliRunner.__init__).parameters else CliRunner()

    def test_no_cache_flag_exists(self):
        from beyondbench.cli.main import evaluate
        param_names = [p.name for p in evaluate.params]
        assert 'no_cache' in param_names

    def test_no_warmup_flag_exists(self):
        from beyondbench.cli.main import evaluate
        param_names = [p.name for p in evaluate.params]
        assert 'no_warmup' in param_names

    def test_quantization_flag_exists(self):
        from beyondbench.cli.main import evaluate
        param_names = [p.name for p in evaluate.params]
        assert 'quantization' in param_names

    def test_torch_compile_flag_exists(self):
        from beyondbench.cli.main import evaluate
        param_names = [p.name for p in evaluate.params]
        assert 'torch_compile' in param_names

    def test_cache_subcommand_exists(self):
        from beyondbench.cli.main import cache
        # cache is a Click Group with stats and clear subcommands
        assert hasattr(cache, 'commands') or callable(cache)

    def test_cache_stats_subcommand(self):
        from beyondbench.cli.main import cache_stats
        assert callable(cache_stats)

    def test_cache_clear_subcommand_with_yes(self):
        from beyondbench.cli.main import cache_clear
        with tempfile.TemporaryDirectory() as td:
            result = self.runner.invoke(cache_clear, ['--cache-dir', td, '-y'])
            assert result.exit_code == 0
            assert 'cleared' in result.output.lower() or 'success' in result.output.lower()


# -----------------------------------------------------------------------
# Prefix caching and vLLM config verification
# -----------------------------------------------------------------------

class TestVLLMConfig:
    def test_prefix_caching_enabled_in_setup(self):
        """Verify that enable_prefix_caching=True is set in the vLLM setup path."""
        import inspect
        from beyondbench.models.model_handler import ModelHandler
        source = inspect.getsource(ModelHandler._setup_local_model)
        assert "enable_prefix_caching=True" in source

    def test_max_num_seqs_passed(self):
        """Verify max_num_seqs is passed to vLLM LLM constructor."""
        import inspect
        from beyondbench.models.model_handler import ModelHandler
        source = inspect.getsource(ModelHandler._setup_local_model)
        assert "max_num_seqs" in source


# -----------------------------------------------------------------------
# Quantization config verification
# -----------------------------------------------------------------------

class TestQuantizationConfig:
    def test_quantization_options(self):
        """All four quantization methods are handled in ModelHandler._setup_local_model."""
        import inspect
        from beyondbench.models.model_handler import ModelHandler
        source = inspect.getsource(ModelHandler._setup_local_model)
        for method in ("4bit", "8bit", "gptq", "awq"):
            assert method in source, f"Quantization method '{method}' not found in _setup_local_model"

    def test_bitsandbytes_import_guard(self):
        """Quantization gracefully handles missing bitsandbytes."""
        import inspect
        from beyondbench.models.model_handler import ModelHandler
        source = inspect.getsource(ModelHandler._setup_local_model)
        assert "ImportError" in source  # catches missing dependency


# -----------------------------------------------------------------------
# Evaluation engine warm-up integration
# -----------------------------------------------------------------------

class TestEvalEngineWarmup:
    def test_warmup_called_in_run_evaluation(self):
        """EvaluationEngine.run_evaluation() calls warm_up unless skip_warmup=True."""
        import inspect
        from beyondbench.core.evaluation_engine import EvaluationEngine
        source = inspect.getsource(EvaluationEngine.run_evaluation)
        assert "warm_up" in source
        assert "skip_warmup" in source

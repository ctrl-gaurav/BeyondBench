"""
Unit tests for Phase 9 configuration system hardening.

Covers:
- load_config / validate_config / load_and_validate_config
- Environment variable overrides (BEYONDBENCH_*)
- .env file auto-loading via python-dotenv (when available)
- ConfigLoader class façade with ConfigProxy attribute + dict access
- ModelValidator: HF existence (mocked), GPU memory, backend deps,
  API key checks, optimal-settings suggestions, VRAM estimation
- Preset discovery and loading
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from beyondbench.configs import (
    ConfigLoader,
    ModelValidator,
    ValidationIssue,
    ValidationResult,
    get_env_overrides,
    list_presets,
    load_and_validate_config,
    load_config,
    validate_config,
)
from beyondbench.configs.config_loader import ConfigProxy
from beyondbench.configs.model_validator import (
    _extract_model_size_billions,
    estimate_vram_gib,
    suggest_optimal_settings,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """A valid minimal config dict."""
    return {
        "model": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "backend": "vllm",
            "cuda_device": "cuda:0",
        },
        "evaluation": {
            "suite": "easy",
            "datapoints": 5,
            "folds": 1,
            "temperature": 0.1,
            "max_tokens": 4096,
        },
    }


@pytest.fixture
def config_yaml(tmp_path: Path, minimal_config: Dict[str, Any]) -> Path:
    """Write minimal_config to a YAML file and return its path."""
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(minimal_config))
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear every BEYONDBENCH_* env var for the duration of a test.

    Autouse because some tests (the .env discovery ones) call ``load_dotenv``
    which directly mutates ``os.environ``; without a blanket clear, those
    leaked values pollute sibling test classes and cause flaky assertions.
    """
    for k in list(os.environ):
        if k.startswith("BEYONDBENCH_"):
            monkeypatch.delenv(k, raising=False)
    # Also clear provider API keys so api-key checks are deterministic
    for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    # Reset the _DOTENV_LOADED memo between tests so each run gets a fresh probe
    import beyondbench.configs as cfg_mod
    cfg_mod._DOTENV_LOADED = False
    yield


# =============================================================================
# load_config / validate_config
# =============================================================================

class TestLoadConfig:
    """Functional config-loading API."""

    def test_load_minimal_yaml(self, config_yaml):
        cfg = load_config(config_yaml)
        assert cfg["model"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg["evaluation"]["datapoints"] == 5

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_load_json_file(self, tmp_path, minimal_config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(minimal_config))
        cfg = load_config(path)
        assert cfg["model"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_load_unsupported_extension(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("[model]\nmodel_id = 'x'\n")
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(path)

    def test_load_and_validate_valid(self, config_yaml):
        cfg = load_and_validate_config(config_yaml)
        assert cfg["model"]["backend"] == "vllm"

    def test_load_and_validate_invalid_raises(self, tmp_path):
        import yaml
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"model": {"model_id": "x", "backend": "invalid"}}))
        with pytest.raises(Exception):  # jsonschema.ValidationError
            load_and_validate_config(path)


class TestValidateConfig:
    """JSON-schema validation."""

    def test_valid_config_passes(self, minimal_config):
        assert validate_config(minimal_config) is True

    def test_missing_model_raises(self):
        with pytest.raises(Exception) as ei:
            validate_config({})
        assert "model" in str(ei.value).lower()

    def test_bad_backend_raises(self):
        with pytest.raises(Exception) as ei:
            validate_config({"model": {"model_id": "x", "backend": "not-a-backend"}})
        assert "not-a-backend" in str(ei.value) or "enum" in str(ei.value).lower()

    def test_temperature_out_of_range(self):
        with pytest.raises(Exception):
            validate_config({
                "model": {"model_id": "x"},
                "evaluation": {"temperature": 5.0},
            })

    def test_negative_datapoints(self):
        with pytest.raises(Exception):
            validate_config({
                "model": {"model_id": "x"},
                "evaluation": {"datapoints": 0},
            })

    def test_empty_model_id_rejected_by_downstream_validator(self):
        """Empty model_id passes jsonschema (string type) but ModelValidator catches it."""
        # jsonschema schema only requires `model_id` key, not non-empty
        validate_config({"model": {"model_id": ""}})
        # But the pre-flight validator should reject it
        v = ModelValidator(check_hf_existence=False, check_gpu_memory=False)
        r = v.validate({"model": {"model_id": ""}})
        assert not r.ok
        assert any("model_id" in i.message for i in r.errors)


# =============================================================================
# Environment variable overrides
# =============================================================================

class TestEnvVarOverrides:
    """BEYONDBENCH_* env vars take priority over file values."""

    def test_no_env_returns_empty(self, clean_env):
        assert get_env_overrides() == {}

    def test_single_env_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("BEYONDBENCH_MODEL_ID", "custom/model")
        ov = get_env_overrides()
        assert ov == {"model": {"model_id": "custom/model"}}

    def test_type_casting(self, clean_env, monkeypatch):
        monkeypatch.setenv("BEYONDBENCH_DATAPOINTS", "42")
        monkeypatch.setenv("BEYONDBENCH_TEMPERATURE", "0.3")
        monkeypatch.setenv("BEYONDBENCH_GPU_MEMORY_UTIL", "0.85")
        ov = get_env_overrides()
        assert ov["evaluation"]["datapoints"] == 42
        assert ov["evaluation"]["temperature"] == pytest.approx(0.3)
        assert ov["model"]["gpu_memory_utilization"] == pytest.approx(0.85)

    def test_env_overrides_file(self, clean_env, monkeypatch, config_yaml):
        monkeypatch.setenv("BEYONDBENCH_MODEL_ID", "env/override-model")
        monkeypatch.setenv("BEYONDBENCH_DATAPOINTS", "99")
        cfg = load_config(config_yaml)
        assert cfg["model"]["model_id"] == "env/override-model"
        assert cfg["evaluation"]["datapoints"] == 99
        # Unrelated file values are preserved
        assert cfg["evaluation"]["temperature"] == pytest.approx(0.1)

    def test_invalid_type_cast_ignored(self, clean_env, monkeypatch):
        monkeypatch.setenv("BEYONDBENCH_DATAPOINTS", "not-a-number")
        ov = get_env_overrides()
        assert "evaluation" not in ov or "datapoints" not in ov.get("evaluation", {})


# =============================================================================
# .env file loading
# =============================================================================

pytest.importorskip("dotenv", reason="python-dotenv not installed")


class TestDotEnvLoading:
    """Automatic .env file discovery + load."""

    def test_dotenv_in_cwd_is_loaded(self, clean_env, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "BEYONDBENCH_MODEL_ID=dotenv/model\n"
            "BEYONDBENCH_DATAPOINTS=17\n"
        )
        monkeypatch.chdir(tmp_path)
        ov = get_env_overrides()
        assert ov["model"]["model_id"] == "dotenv/model"
        assert ov["evaluation"]["datapoints"] == 17

    def test_real_env_wins_over_dotenv(self, clean_env, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("BEYONDBENCH_MODEL_ID=dotenv/model\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BEYONDBENCH_MODEL_ID", "real-env/model")
        ov = get_env_overrides()
        assert ov["model"]["model_id"] == "real-env/model"

    def test_beyondbench_dotenv_env_var_path(self, clean_env, tmp_path, monkeypatch):
        env_file = tmp_path / "custom.env"
        env_file.write_text("BEYONDBENCH_SUITE=hard\n")
        monkeypatch.setenv("BEYONDBENCH_DOTENV", str(env_file))
        monkeypatch.chdir(tmp_path / ".." if tmp_path.parent else tmp_path)
        ov = get_env_overrides()
        assert ov.get("evaluation", {}).get("suite") == "hard"


# =============================================================================
# ConfigLoader class façade
# =============================================================================

class TestConfigLoaderClass:
    """Class-style API with attribute + dict access."""

    def test_load_returns_proxy(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        assert isinstance(cfg, ConfigProxy)

    def test_shortcut_properties(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        assert cfg.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.backend == "vllm"
        assert cfg.datapoints == 5
        assert cfg.suite == "easy"
        assert cfg.temperature == pytest.approx(0.1)
        assert cfg.max_tokens == 4096

    def test_nested_attribute_access(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        assert cfg.model.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.model.backend == "vllm"
        assert cfg.evaluation.datapoints == 5
        assert cfg.evaluation.temperature == pytest.approx(0.1)

    def test_dict_access_still_works(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        assert cfg["model"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg["evaluation"]["datapoints"] == 5

    def test_as_dict_returns_raw(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        d = cfg.as_dict()
        assert isinstance(d, dict)
        assert d["model"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_to_dict_returns_deep_copy(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        d = cfg.to_dict()
        d["model"]["model_id"] = "mutated"
        # Original unchanged
        assert cfg.model_id == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_missing_attribute_raises(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        with pytest.raises(AttributeError):
            _ = cfg.does_not_exist

    def test_nested_missing_attribute_raises(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        with pytest.raises(AttributeError):
            _ = cfg.model.ghost_field

    def test_validate_accepts_proxy_and_dict(self, config_yaml, minimal_config):
        cfg = ConfigLoader.load(config_yaml)
        assert ConfigLoader.validate(cfg) is True
        assert ConfigLoader.validate(minimal_config) is True

    def test_list_presets_returns_builtins(self):
        presets = ConfigLoader.list_presets()
        assert "quick_test" in presets
        assert "full_evaluation" in presets
        assert "paper_quality" in presets
        assert "debug" in presets

    def test_load_preset_by_name(self):
        cfg = ConfigLoader.load_preset("quick_test")
        assert cfg.suite == "easy"
        assert cfg.datapoints == 5

    def test_load_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            ConfigLoader.load_preset("does-not-exist")

    def test_proxy_from_non_dict_raises(self):
        with pytest.raises(TypeError):
            ConfigProxy("not a dict")  # type: ignore[arg-type]

    def test_contains_len_iter(self, config_yaml):
        cfg = ConfigLoader.load(config_yaml)
        assert "model" in cfg
        assert len(cfg) >= 2
        assert sorted(list(cfg)) == sorted(cfg.keys())


# =============================================================================
# Preset presence
# =============================================================================

class TestPresets:
    """The four documented presets exist and validate."""

    REQUIRED = ["quick_test", "full_evaluation", "paper_quality", "debug"]

    def test_all_required_presets_discoverable(self):
        presets = list_presets()
        for name in self.REQUIRED:
            assert name in presets, f"Missing preset: {name}"
            assert presets[name].exists()

    @pytest.mark.parametrize("name", REQUIRED)
    def test_preset_loads_and_validates(self, name):
        cfg = ConfigLoader.load_preset(name)
        assert cfg.model_id, f"{name} has no model_id"
        assert cfg.datapoints and cfg.datapoints > 0

    def test_quick_test_values(self):
        cfg = ConfigLoader.load_preset("quick_test")
        assert cfg.suite == "easy"
        assert cfg.datapoints == 5

    def test_paper_quality_values(self):
        cfg = ConfigLoader.load_preset("paper_quality")
        assert cfg.suite == "all"
        assert cfg.datapoints == 500
        assert cfg.evaluation.folds == 5
        assert cfg.evaluation.seed == 42

    def test_debug_values(self):
        cfg = ConfigLoader.load_preset("debug")
        assert cfg.datapoints == 2
        assert cfg.log_level == "DEBUG"


# =============================================================================
# ModelValidator
# =============================================================================

class TestVramEstimation:
    """Parameter-count → VRAM heuristics."""

    @pytest.mark.parametrize("mid,expected_b", [
        ("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
        ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
        ("Qwen/Qwen2.5-3B-Instruct", 3.0),
        ("Qwen/Qwen2.5-7B-Instruct", 7.0),
        ("meta-llama/Llama-3.2-3B-Instruct", 3.0),
        ("meta-llama/Llama-3.3-70B-Instruct", 70.0),
        ("mistralai/Mistral-7B-Instruct-v0.3", 7.0),
    ])
    def test_size_extraction(self, mid, expected_b):
        assert _extract_model_size_billions(mid) == pytest.approx(expected_b)

    def test_moe_notation(self):
        # 8x7B = effective 56B
        assert _extract_model_size_billions("mistralai/Mixtral-8x7B-Instruct-v0.1") == pytest.approx(56.0)

    def test_unknown_returns_none(self):
        assert _extract_model_size_billions("unknown-model-id") is None

    def test_estimate_vram_scales_with_size(self):
        small = estimate_vram_gib("Qwen/Qwen2.5-0.5B-Instruct")
        large = estimate_vram_gib("meta-llama/Llama-3.3-70B-Instruct")
        assert small is not None and large is not None
        assert large > small * 100  # 70B >> 0.5B

    def test_estimate_vram_unknown_returns_none(self):
        assert estimate_vram_gib("some/opaque-id") is None


class TestOptimalSettings:
    """Family-based sampling-parameter suggestions."""

    def test_qwen_family(self):
        s = suggest_optimal_settings("Qwen/Qwen2.5-3B-Instruct")
        assert "temperature" in s
        assert "max_tokens" in s

    def test_openai_reasoning_models(self):
        for mid in ["o1-mini", "o3-mini", "o4-preview"]:
            s = suggest_optimal_settings(mid)
            assert s["temperature"] == pytest.approx(1.0), f"{mid} should get T=1.0"

    def test_gemini(self):
        s = suggest_optimal_settings("gemini-2.5-flash")
        assert s["temperature"] == pytest.approx(0.1)

    def test_unknown_falls_back_to_defaults(self):
        s = suggest_optimal_settings("unknown/model-id-xyz")
        assert s["temperature"] == pytest.approx(0.1)
        assert s["max_tokens"] == 4096


class TestModelValidator:
    """End-to-end validator behavior with mocked externals."""

    def _validator_no_network_no_gpu(self):
        return ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=False,
            check_dependencies=False,  # deps vary by test env
            suggest_settings=False,
        )

    def test_missing_model_id_is_error(self):
        r = self._validator_no_network_no_gpu().validate({"model": {}})
        assert not r.ok
        assert any("model_id" in i.check for i in r.errors)

    def test_valid_minimal_is_ok(self, minimal_config):
        r = self._validator_no_network_no_gpu().validate(minimal_config)
        assert r.ok, [i.message for i in r.errors]

    def test_enforce_raises_on_errors(self):
        v = self._validator_no_network_no_gpu()
        r = v.validate({"model": {}})
        with pytest.raises(ValueError, match="validation failed"):
            v.enforce(r)

    def test_enforce_no_raise_when_ok(self, minimal_config):
        v = self._validator_no_network_no_gpu()
        r = v.validate(minimal_config)
        v.enforce(r)  # must not raise

    def test_result_severity_helpers(self):
        r = ValidationResult()
        r.add("x", "error", "e")
        r.add("y", "warning", "w")
        r.add("z", "info", "i")
        assert len(r.errors) == 1
        assert len(r.warnings) == 1
        assert len(r.infos) == 1
        assert not r.ok

    def test_api_backend_missing_key_is_error(self, clean_env):
        v = ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=False,
            check_dependencies=False,
            suggest_settings=False,
        )
        r = v.validate({
            "model": {"model_id": "gpt-4o-mini", "backend": "openai"},
        })
        assert not r.ok
        assert any("api_key" in i.check for i in r.errors)

    def test_api_backend_with_env_key_ok(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        v = ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=False,
            check_dependencies=False,
            suggest_settings=False,
        )
        r = v.validate({
            "model": {"model_id": "gpt-4o-mini", "backend": "openai"},
        })
        assert r.ok, [i.message for i in r.errors]

    def test_api_backend_with_inline_key_ok(self, clean_env):
        v = ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=False,
            check_dependencies=False,
            suggest_settings=False,
        )
        r = v.validate({
            "model": {
                "model_id": "gpt-4o-mini",
                "backend": "openai",
                "api_key": "sk-inline-fake",
            },
        })
        assert r.ok

    def test_hf_404_surfaces_as_error(self):
        """A HEAD that returns 404 should become an error issue."""
        from urllib.error import HTTPError

        def fake_urlopen(req, timeout=None):
            raise HTTPError(req.full_url, 404, "Not Found", {}, None)

        v = ModelValidator(
            check_hf_existence=True,
            check_gpu_memory=False,
            check_dependencies=False,
            suggest_settings=False,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            r = v.validate({
                "model": {"model_id": "nonexistent/fake-model", "backend": "vllm"},
            })
        assert not r.ok
        assert any("huggingface" in i.check for i in r.errors)

    def test_hf_network_unreachable_is_info(self):
        """If the network is unreachable, skip the check with an info note."""
        from urllib.error import URLError

        def fake_urlopen(req, timeout=None):
            raise URLError("offline")

        v = ModelValidator(
            check_hf_existence=True,
            check_gpu_memory=False,
            check_dependencies=False,
            suggest_settings=False,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            r = v.validate({
                "model": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "backend": "vllm"},
            })
        # Info issues don't flip ok to False
        assert r.ok
        assert any(
            "huggingface" in i.check and i.severity == "info"
            for i in r.issues
        )

    def test_gpu_memory_too_small_errors(self):
        """70B model on a simulated 24GB GPU should produce an error."""
        v = ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=True,
            check_dependencies=False,
            suggest_settings=False,
        )
        # total=24576 MiB, free=24000 MiB → ~23.4 GiB < 168 GiB needed
        with patch(
            "beyondbench.configs.model_validator._query_gpu_free_memory_mib",
            return_value={0: (24576, 24000)},
        ):
            r = v.validate({
                "model": {
                    "model_id": "meta-llama/Llama-3.3-70B-Instruct",
                    "backend": "vllm",
                    "cuda_device": "cuda:0",
                    "gpu_memory_utilization": 0.96,
                },
            })
        assert not r.ok
        assert any("gpu.memory" in i.check for i in r.errors)

    def test_gpu_memory_sufficient_no_error(self):
        """3B model on a simulated 24GB GPU should pass."""
        v = ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=True,
            check_dependencies=False,
            suggest_settings=False,
        )
        with patch(
            "beyondbench.configs.model_validator._query_gpu_free_memory_mib",
            return_value={0: (24576, 24000)},
        ):
            r = v.validate({
                "model": {
                    "model_id": "Qwen/Qwen2.5-3B-Instruct",
                    "backend": "vllm",
                    "cuda_device": "cuda:0",
                    "gpu_memory_utilization": 0.96,
                },
            })
        # No error severity (may still have warning if close to cap)
        assert r.ok

    def test_gpu_device_not_found_is_warning(self):
        """Requesting cuda:5 when only cuda:0 exists should warn."""
        v = ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=True,
            check_dependencies=False,
            suggest_settings=False,
        )
        with patch(
            "beyondbench.configs.model_validator._query_gpu_free_memory_mib",
            return_value={0: (24576, 24000)},
        ):
            r = v.validate({
                "model": {
                    "model_id": "Qwen/Qwen2.5-3B-Instruct",
                    "backend": "vllm",
                    "cuda_device": "cuda:5",
                    "gpu_memory_utilization": 0.96,
                },
            })
        assert r.ok  # warning, not error
        assert any(
            i.severity == "warning" and "gpu.memory" in i.check for i in r.issues
        )

    def test_suggestion_divergence_info(self):
        """Large temperature/max_tokens gap produces info-level suggestion."""
        v = ModelValidator(
            check_hf_existence=False,
            check_gpu_memory=False,
            check_dependencies=False,
            suggest_settings=True,
        )
        r = v.validate({
            "model": {"model_id": "gpt-4o", "backend": "openai", "api_key": "sk-fake"},
            "evaluation": {"temperature": 1.5, "max_tokens": 4096},
        })
        assert r.suggested_settings  # populated
        assert any(
            i.severity == "info" and "temperature" in i.check for i in r.issues
        )

    def test_validation_issue_dataclass(self):
        issue = ValidationIssue("x", "error", "msg", "hint")
        assert issue.check == "x"
        assert issue.severity == "error"
        assert issue.message == "msg"
        assert issue.suggestion == "hint"


# =============================================================================
# Integration: CLI --skip-validation flag
# =============================================================================

class TestCLIValidationFlags:
    """The evaluate command exposes --skip-validation and --validation-warn-only."""

    def test_evaluate_help_has_flags(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--skip-validation" in result.output
        assert "--validation-warn-only" in result.output

    def test_run_config_help_has_skip_flag(self):
        from click.testing import CliRunner
        from beyondbench.cli.main import main
        runner = CliRunner()
        result = runner.invoke(main, ["run-config", "--help"])
        assert result.exit_code == 0
        assert "--skip-validation" in result.output

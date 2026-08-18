"""OpenRouter helpers: key/model resolution, schemas, usage mapping, images."""
import base64
import types

import pytest

import gemini_worker


@pytest.fixture(autouse=True)
def _clean_ai_env(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


class TestResolveApiKey:
    def test_openrouter_wins_without_touching_gemini(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-old")
        assert gemini_worker.resolve_provider() == "openrouter"
        assert gemini_worker.resolve_api_key() == "sk-or-1"
        assert gemini_worker.gemini_key() == "AIza-old"

    def test_gemini_key_never_returned_for_openrouter(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "openrouter")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-studio")
        assert gemini_worker.resolve_api_key() is None
        assert gemini_worker.openrouter_key() is None

    def test_openrouter_key_never_returned_for_gemini(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secret")
        assert gemini_worker.resolve_api_key() is None
        assert gemini_worker.gemini_key() is None

    def test_aiza_in_openrouter_env_is_rejected(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "AIza-leaked")
        assert gemini_worker.openrouter_key() is None

    def test_sk_or_in_gemini_env_is_rejected(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "sk-or-v1-leaked")
        assert gemini_worker.gemini_key() is None

    def test_gemini_only_selects_gemini_provider(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-studio")
        assert gemini_worker.resolve_provider() == "gemini"
        assert gemini_worker.resolve_api_key() == "AIza-studio"

    def test_missing_both_is_none(self):
        assert gemini_worker.resolve_api_key() is None


class TestKeyIsolationClients:
    def test_openrouter_client_rejects_ai_studio_key(self):
        with pytest.raises(RuntimeError, match="AIza"):
            gemini_worker.openrouter_client("AIza-studio")

    def test_openrouter_client_does_not_read_gemini_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-studio")
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            gemini_worker.openrouter_client()

    def test_gemini_client_rejects_openrouter_key(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        with pytest.raises(RuntimeError, match="OpenRouter"):
            gemini_worker.gemini_client("sk-or-v1-secret")

    def test_gemini_client_refuses_when_provider_is_openrouter(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "openrouter")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-studio")
        with pytest.raises(RuntimeError, match="openrouter"):
            gemini_worker.gemini_client()


class TestApplyAiKeysToEnv:
    def test_openrouter_job_keeps_compose_key_and_drops_gemini(self):
        env = {
            "OPENROUTER_API_KEY": "sk-or-compose",
            "GEMINI_API_KEY": "AIza-studio",
            "MANAGED_GEMINI_API_KEY": "AIza-managed",
        }
        gemini_worker.apply_ai_keys_to_env(env, "openrouter")
        assert env["OPENROUTER_API_KEY"] == "sk-or-compose"
        assert "GEMINI_API_KEY" not in env
        assert env["MANAGED_GEMINI_API_KEY"] == "AIza-managed"
        assert env["AI_PROVIDER"] == "openrouter"

    def test_managed_gemini_is_never_copied_to_openrouter(self):
        env = {"MANAGED_GEMINI_API_KEY": "AIza-managed"}
        gemini_worker.apply_ai_keys_to_env(env, "openrouter")
        assert env.get("OPENROUTER_API_KEY") in (None, "")
        assert env.get("GEMINI_API_KEY") in (None, "")

    def test_gemini_job_drops_openrouter_key(self):
        env = {
            "OPENROUTER_API_KEY": "sk-or-compose",
            "GEMINI_API_KEY": "AIza-studio",
        }
        gemini_worker.apply_ai_keys_to_env(env, "gemini")
        assert env["GEMINI_API_KEY"] == "AIza-studio"
        assert "OPENROUTER_API_KEY" not in env

    def test_byok_aiza_rejected_for_openrouter(self):
        env = {"OPENROUTER_API_KEY": "sk-or-compose"}
        with pytest.raises(RuntimeError, match="AIza"):
            gemini_worker.apply_ai_keys_to_env(env, "openrouter", byok_key="AIza-studio")

    def test_byok_openrouter_rejected_for_gemini(self):
        env = {"GEMINI_API_KEY": "AIza-studio"}
        with pytest.raises(RuntimeError, match="OpenRouter"):
            gemini_worker.apply_ai_keys_to_env(env, "gemini", byok_key="sk-or-v1-secret")


class TestResolveModel:
    def test_default_is_openrouter_gemini_flash(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        assert gemini_worker.resolve_model() == "google/gemini-2.5-flash"

    def test_gemini_provider_uses_native_default(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        assert gemini_worker.resolve_model() == "gemini-3.1-flash-lite"

    def test_openrouter_model_wins(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
        assert gemini_worker.resolve_model() == "anthropic/claude-sonnet-4"

    def test_gemini_model_is_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
        assert gemini_worker.resolve_model() == "google/gemini-2.5-flash"

    def test_bare_gemini_id_is_prefixed(self):
        assert gemini_worker.map_model_id("gemini-2.5-flash") == "google/gemini-2.5-flash"
        assert gemini_worker.map_model_id("gemini-3.1-flash-lite") == "google/gemini-3.1-flash-lite"

    def test_already_namespaced_id_is_left_alone(self):
        assert gemini_worker.map_model_id("google/gemini-2.5-flash") == "google/gemini-2.5-flash"
        assert gemini_worker.map_model_id("openai/gpt-4o") == "openai/gpt-4o"

    def test_explicit_arg_wins_and_is_mapped(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "ignored")
        assert gemini_worker.resolve_model("gemini-2.0-flash") == "google/gemini-2.0-flash"

    def test_gemini_ui_path_ignores_process_openrouter_default(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
        monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        assert gemini_worker.resolve_provider() == "openrouter"
        assert gemini_worker.resolve_model(provider="gemini") == "gemini-3.1-flash-lite"
        assert "/" not in gemini_worker.resolve_model(provider="gemini")
        assert gemini_worker.resolve_model(
            explicit="google/gemini-2.5-flash", provider="gemini"
        ) == "gemini-2.5-flash"


class TestRequestProvider:
    def test_explicit_header_wins_even_on_billing(self):
        assert gemini_worker.resolve_request_provider(
            "openrouter", billing=True) == "openrouter"
        assert gemini_worker.resolve_request_provider(
            "gemini", billing=True) == "gemini"

    def test_billing_defaults_to_gemini_not_compose_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
        assert gemini_worker.resolve_provider() == "openrouter"
        assert gemini_worker.resolve_request_provider(None, billing=True) == "gemini"

    def test_self_host_without_header_still_infers_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
        assert gemini_worker.resolve_request_provider(
            None, billing=False) == "openrouter"


class TestResponseFormat:
    def test_structured_schema_uses_pydantic_json_schema(self):
        fmt = gemini_worker._response_format_for(gemini_worker.ScoreResponse, "structured-schema")
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "ScoreResponse"
        schema = fmt["json_schema"]["schema"]
        assert "windows" in schema.get("properties", {})

    def test_detail_schema_keeps_required_fields(self):
        fmt = gemini_worker._response_format_for(gemini_worker.DetailResponse, "structured-schema")
        props = fmt["json_schema"]["schema"]["properties"]
        assert "shorts" in props

    def test_text_recovery_is_json_object(self):
        assert gemini_worker._response_format_for(
            gemini_worker.ScoreResponse, "json-text-recovery") == {"type": "json_object"}


class TestTextRepairFallback:
    def test_fenced_json_is_repaired(self):
        raw = '```json\n{"windows": [{"id": "w1", "start": 0, "end": 10, "score": 80, "reason": "hook"}]}\n```'
        parsed = gemini_worker._parse_structured_payload(raw, gemini_worker.ScoreResponse)
        assert parsed["windows"][0]["id"] == "w1"
        assert parsed["windows"][0]["score"] == 80

    def test_empty_body_raises_the_retryable_message(self):
        with pytest.raises(ValueError, match="empty response body"):
            gemini_worker._parse_structured_payload("", gemini_worker.ScoreResponse)


class TestUsageMapping:
    def test_openai_usage_does_not_double_count_reasoning(self):
        # OpenRouter: reasoning_tokens already sit inside completion_tokens.
        # Bill 200, not 200+50.
        from clip_selection import lookup_model_prices
        usage = types.SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=200,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=50),
        )
        response = types.SimpleNamespace(usage=usage, usage_metadata=None)
        cost = gemini_worker._calculate_cost_analysis(response, "google/gemini-2.5-flash")
        inp, out = lookup_model_prices("google/gemini-2.5-flash")
        assert cost["input_tokens"] == 1000
        assert cost["output_tokens"] == 200
        assert cost["thinking_tokens"] == 50
        assert cost["model"] == "google/gemini-2.5-flash"
        assert cost["price_estimated"] is False
        assert abs(cost["output_cost"] - (200 / 1_000_000) * out) < 1e-12
        expected = (1000 / 1_000_000) * inp + (200 / 1_000_000) * out
        assert abs(cost["total_cost"] - expected) < 1e-12
        assert abs(cost["output_cost"] - (250 / 1_000_000) * out) > 1e-12

    def test_legacy_gemini_thoughts_are_added(self):
        from clip_selection import lookup_model_prices
        usage = types.SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=2,
        )
        response = types.SimpleNamespace(usage=None, usage_metadata=usage)
        cost = gemini_worker._calculate_cost_analysis(response, "gemini-2.5-flash")
        _inp, out = lookup_model_prices("gemini-2.5-flash")
        assert cost["input_tokens"] == 10
        assert cost["output_tokens"] == 5
        assert cost["thinking_tokens"] == 2
        assert abs(cost["output_cost"] - (7 / 1_000_000) * out) < 1e-12

    def test_unknown_model_does_not_invent_a_rate(self):
        usage = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=200)
        response = types.SimpleNamespace(usage=usage, usage_metadata=None)
        cost = gemini_worker._calculate_cost_analysis(response, "anthropic/claude-sonnet-4")
        assert cost["price_estimated"] is True
        assert cost["input_tokens"] == 1000
        assert cost["output_tokens"] == 200
        assert cost["total_cost"] == 0.0
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0


class TestTemperature:
    def test_detail_stays_creative(self):
        assert gemini_worker._temperature_for_strategy("structured-schema", "detail") == 0.9
        assert gemini_worker._temperature_for_strategy("structured-schema", "score") == 0.2


class TestJpegDataUris:
    def test_frames_become_image_url_data_uris(self):
        jpeg = b"\xff\xd8\xfffakejpeg"
        parts = gemini_worker.jpeg_image_url_parts([jpeg])
        assert parts[0]["type"] == "image_url"
        url = parts[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == jpeg


class TestFileApiSkip:
    def test_image_generation_is_gemini_only(self):
        assert "OpenRouter" in gemini_worker.GEMINI_IMAGE_ONLY

    def test_visual_clips_does_not_raise_file_api_unavailable(self):
        main = pytest.importorskip("main")
        assert main.get_visual_clips("missing.mp4", 30) is None

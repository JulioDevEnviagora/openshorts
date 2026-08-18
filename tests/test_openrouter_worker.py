"""OpenRouter helpers: key/model resolution, schemas, usage mapping, images."""
import base64
import types

import pytest

import gemini_worker


class TestResolveApiKey:
    def test_openrouter_wins(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-old")
        assert gemini_worker.resolve_api_key() == "sk-or-1"

    def test_falls_back_to_gemini_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-old")
        assert gemini_worker.resolve_api_key() == "AIza-old"

    def test_missing_both_is_none(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert gemini_worker.resolve_api_key() is None


class TestResolveModel:
    def test_default_is_openrouter_gemini_flash(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        assert gemini_worker.resolve_model() == "google/gemini-2.5-flash"

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
    def test_openai_usage_maps_to_cost_analysis(self):
        usage = types.SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=200,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=50),
        )
        response = types.SimpleNamespace(usage=usage, usage_metadata=None)
        cost = gemini_worker._calculate_cost_analysis(response, "google/gemini-2.5-flash")
        assert cost["input_tokens"] == 1000
        assert cost["output_tokens"] == 200
        assert cost["thinking_tokens"] == 50
        assert cost["model"] == "google/gemini-2.5-flash"
        assert cost["total_cost"] > 0
        assert cost["price_estimated"] is False

    def test_legacy_gemini_usage_metadata_still_works(self):
        usage = types.SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=2,
        )
        response = types.SimpleNamespace(usage=None, usage_metadata=usage)
        cost = gemini_worker._calculate_cost_analysis(response, "gemini-2.5-flash")
        assert cost["input_tokens"] == 10
        assert cost["output_tokens"] == 5
        assert cost["thinking_tokens"] == 2


class TestJpegDataUris:
    def test_frames_become_image_url_data_uris(self):
        jpeg = b"\xff\xd8\xfffakejpeg"
        parts = gemini_worker.jpeg_image_url_parts([jpeg])
        assert parts[0]["type"] == "image_url"
        url = parts[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == jpeg


class TestFileApiSkip:
    def test_message_names_file_api_and_openrouter(self):
        assert "File API" in gemini_worker.FILE_API_UNAVAILABLE
        assert "OpenRouter" in gemini_worker.FILE_API_UNAVAILABLE

    def test_visual_clips_skips_without_calling_file_api(self):
        main = pytest.importorskip("main")
        assert main.get_visual_clips("missing.mp4", 30) is None

"""Thumbnail path is OpenRouter-only. No live API. No Gemini required."""
import asyncio
import base64
import inspect
from io import BytesIO

import httpx
import pytest
from PIL import Image

import app as app_module
import gemini_worker
import thumbnail


@pytest.fixture(autouse=True)
def _clean_ai_env(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_IMAGE_MODEL", raising=False)


def _tiny_jpeg_b64():
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_thumbnail_module_does_not_import_google_genai():
    source = inspect.getsource(thumbnail)
    assert "from google" not in source
    assert "import google" not in source
    assert "files.upload" not in source


def test_analyze_titles_uses_transcript_not_file_api(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        thumbnail, "_openrouter_client", lambda key: captured.setdefault("key", key) or object())
    monkeypatch.setattr(thumbnail, "_text_model", lambda: "google/gemini-2.5-flash")

    def _complete(client, model, prompt, schema, **kwargs):
        captured["prompt"] = prompt
        captured["schema"] = schema
        captured["contents"] = kwargs.get("contents")
        return {
            "titles": ["Hook one"],
            "transcript_summary": "A podcast.",
            "language": "en",
            "recommended": [],
        }, None

    monkeypatch.setattr(gemini_worker, "complete_json", _complete)
    result = thumbnail.analyze_video_for_titles(
        "sk-or-v1-test",
        "unused.mp4",
        transcript={"text": "hello world", "language": "en", "segments": [{"end": 12}]},
    )
    assert result["titles"] == ["Hook one"]
    assert result["video_duration"] == 12
    assert captured["key"] == "sk-or-v1-test"
    assert "hello world" in captured["prompt"]
    assert captured["contents"] is None
    assert captured["schema"] is thumbnail.TitlesResponse


def test_generate_thumbnail_saves_openrouter_bytes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    called = {}

    def _fake_images(api_key, prompt, **kwargs):
        called["key"] = api_key
        called["prompt"] = prompt
        called["kwargs"] = kwargs
        return {"data": [{"b64_json": _tiny_jpeg_b64(), "media_type": "image/jpeg"}]}

    monkeypatch.setattr(gemini_worker, "generate_openrouter_images", _fake_images)
    paths = thumbnail.generate_thumbnail("sk-or-v1-test", "Viral title", "sess1", count=1)
    assert paths == ["/thumbnails/sess1/thumb_1.jpg"]
    assert called["key"] == "sk-or-v1-test"
    assert called["kwargs"]["aspect_ratio"] == "16:9"
    assert (tmp_path / "output" / "thumbnails" / "sess1" / "thumb_1.jpg").is_file()


def test_generate_thumbnail_rejects_ai_studio_key():
    with pytest.raises(RuntimeError, match="AIza"):
        thumbnail.generate_thumbnail("AIza-studio", "title", "sess")


def test_thumbnail_generate_endpoint_does_not_400_without_gemini(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-compose")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def _fake_generate(api_key, title, session_id, *args, **kwargs):
        assert api_key == "sk-or-compose"
        assert title == "My clip"
        return ["/thumbnails/s1/thumb_1.jpg"]

    monkeypatch.setattr(app_module, "generate_thumbnail", _fake_generate)

    async def _run():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post(
                "/api/thumbnail/generate",
                data={"session_id": "s1", "title": "My clip", "count": "1"},
            )

    resp = asyncio.run(_run())
    assert resp.status_code == 200, resp.text
    assert resp.json()["thumbnails"] == ["/thumbnails/s1/thumb_1.jpg"]


def test_thumbnail_generate_endpoint_400s_without_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-studio")

    async def _run():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post(
                "/api/thumbnail/generate",
                data={"session_id": "s1", "title": "My clip", "count": "1"},
                headers={"X-Gemini-Key": "AIza-studio", "X-AI-Provider": "gemini"},
            )

    resp = asyncio.run(_run())
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "OpenRouter" in detail
    assert "Gemini is optional" in detail

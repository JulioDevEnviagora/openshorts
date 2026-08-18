"""VideoEditor Gemini UI path must not inherit process-level OpenRouter."""
import pytest

import gemini_worker

editor_mod = pytest.importorskip("editor")


@pytest.fixture(autouse=True)
def _clean_ai_env(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_EDITOR", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)


def test_editor_defaults_to_gemini_when_compose_has_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setattr(gemini_worker, "make_client", lambda *a, **k: object())
    editor = editor_mod.VideoEditor("AIza-studio")
    assert editor.provider == "gemini"
    assert editor.model_name == "gemini-3.1-flash-lite"
    assert not editor.model_name.startswith("google/")


def test_editor_explicit_openrouter_keeps_openrouter_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    monkeypatch.setattr(gemini_worker, "make_client", lambda *a, **k: object())
    editor = editor_mod.VideoEditor("sk-or-1", provider="openrouter")
    assert editor.provider == "openrouter"
    assert editor.model_name == "google/gemini-2.5-flash"
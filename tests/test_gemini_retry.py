"""Retry behaviour of the OpenRouter scoring/detail stage.

Prod (22-jul-2026) lost 3 jobs to Gemini answering 200 with an empty body.
That raises while parsing, not while calling, so it used to escape the retry
loop and kill the job on the first blip. The OpenRouter path keeps the same
retry contract.
"""
import json
import types
import pytest

# main pulls in cv2/torch/mediapipe at import time; the minimal CI env lacks
# them, so skip there. Runs fully in the container/local where deps exist.
main = pytest.importorskip("main")


class _FakeMessage:
    def __init__(self, content="", refusal=None):
        self.content = content
        self.refusal = refusal


class _FakeChoice:
    def __init__(self, content="", finish_reason="stop", refusal=None):
        self.message = _FakeMessage(content, refusal)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, parsed=None, content=None, finish_reason="stop"):
        if content is None:
            content = "" if parsed is None else (
                parsed if isinstance(parsed, str) else json.dumps(parsed)
            )
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = None
        self.parsed = parsed


class _FakeCompletions:
    """Returns an empty body for the first `blips` calls, then a good one."""

    def __init__(self, blips, payload=None):
        self.blips = blips
        self.calls = 0
        self.payload = payload if payload is not None else {"windows": [{"id": "w0", "score": 90}]}

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.blips:
            return _FakeResponse(parsed=None, content="")
        return _FakeResponse(parsed=self.payload)


def _client(completions):
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)


def test_recovers_from_a_single_empty_body(monkeypatch):
    models = _FakeCompletions(blips=1)
    parsed, _cost = main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 2
    assert parsed["windows"][0]["score"] == 90


def test_recovers_from_two_consecutive_blips():
    models = _FakeCompletions(blips=2)
    parsed, _cost = main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 3
    assert parsed["windows"]


def test_gives_up_after_three_attempts():
    models = _FakeCompletions(blips=99)
    with pytest.raises(Exception) as exc:
        main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 3
    assert "empty response body" in str(exc.value)


def test_non_transient_errors_are_not_retried():
    class _Boom:
        calls = 0

        def create(self, **kwargs):
            _Boom.calls += 1
            raise ValueError("400 INVALID_ARGUMENT: bad request")

    with pytest.raises(ValueError):
        main._run_gemini_stage(_client(_Boom()), "m", "prompt", object)
    assert _Boom.calls == 1


def test_succeeds_without_retrying_when_the_first_call_is_fine():
    models = _FakeCompletions(blips=0)
    main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 1


class _BlockedResponse:
    """OpenRouter/OpenAI shape: content_filter finish_reason, empty content."""
    usage = None
    parsed = None
    choices = [_FakeChoice(content="", finish_reason="content_filter")]


def test_policy_block_fails_fast_without_retrying():
    class _Models:
        calls = 0

        def create(self, **kwargs):
            _Models.calls += 1
            return _BlockedResponse()

    import gemini_worker
    with pytest.raises(gemini_worker.GeminiBlockedError) as exc:
        main._run_gemini_stage(_client(_Models()), "m", "prompt", object)
    assert _Models.calls == 1
    assert "CONTENT_FILTER" in str(exc.value)


def test_blocked_finish_reason_also_raises():
    import gemini_worker

    class _Cand:
        class _FR:
            name = "SAFETY"
        finish_reason = _FR()
        content = None

    class _Resp:
        prompt_feedback = None
        candidates = [_Cand()]
        choices = []

    with pytest.raises(gemini_worker.GeminiBlockedError):
        gemini_worker.raise_if_blocked(_Resp())


def test_clean_response_is_not_flagged_as_blocked():
    import gemini_worker

    class _Resp:
        prompt_feedback = None
        candidates = []
        choices = []

    gemini_worker.raise_if_blocked(_Resp())  # must not raise

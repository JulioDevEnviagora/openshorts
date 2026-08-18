import argparse
import base64
import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from clip_selection import (clip_count_targets, clip_duration_bounds,
                            lookup_model_prices)

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
PROVIDERS = ("openrouter", "gemini")

# Native Imagen / flash-image — OpenRouter cannot do this.
GEMINI_IMAGE_ONLY = (
    "Thumbnail image generation uses Gemini's native image model and is not "
    "available via OpenRouter. Switch the provider to Gemini."
)


def looks_like_ai_studio_key(key: Optional[str]) -> bool:
    return (key or "").strip().startswith("AIza")


def looks_like_openrouter_key(key: Optional[str]) -> bool:
    raw = (key or "").strip()
    return raw.startswith("sk-or-") or raw.startswith("sk-orv-")


def openrouter_key() -> Optional[str]:
    """Only ``OPENROUTER_API_KEY``. Never an AI Studio ``AIza…`` token."""
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None
    if key and looks_like_ai_studio_key(key):
        return None
    return key


def gemini_key() -> Optional[str]:
    """Only ``GEMINI_API_KEY``. Never an OpenRouter ``sk-or-`` token."""
    key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
    if key and looks_like_openrouter_key(key):
        return None
    return key


def assert_openrouter_safe_key(key: Optional[str]) -> str:
    raw = (key or "").strip()
    if not raw:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY before calling OpenRouter. "
            "A Google AI Studio key (GEMINI_API_KEY / AIza…) is never sent to openrouter.ai."
        )
    if looks_like_ai_studio_key(raw):
        raise RuntimeError(
            "Gemini AI Studio keys (AIza…) cannot be sent to OpenRouter. "
            "Use OPENROUTER_API_KEY or switch AI_PROVIDER=gemini."
        )
    return raw


def assert_google_safe_key(key: Optional[str]) -> str:
    raw = (key or "").strip()
    if not raw:
        raise RuntimeError(
            "Set GEMINI_API_KEY before using the native Gemini client. "
            "An OpenRouter key is never sent to generativelanguage.googleapis.com."
        )
    if looks_like_openrouter_key(raw):
        raise RuntimeError(
            "OpenRouter keys (sk-or-…) cannot be sent to Google Gemini. "
            "Set a Google AI Studio key (AIza…) or switch AI_PROVIDER=openrouter."
        )
    return raw


def resolve_provider() -> str:
    """``AI_PROVIDER`` if set, else infer from which *usable* key is present.

    OpenRouter when ``OPENROUTER_API_KEY`` is a real OpenRouter key; Gemini
    when ``GEMINI_API_KEY`` is a usable Google key. Keys are never crossed.
    """
    raw = (os.getenv("AI_PROVIDER") or "").strip().lower()
    if raw in PROVIDERS:
        return raw
    if openrouter_key():
        return "openrouter"
    if gemini_key():
        return "gemini"
    return "openrouter"


def resolve_api_key() -> Optional[str]:
    """Key for the active provider only. Never crosses Gemini ↔ OpenRouter."""
    if resolve_provider() == "gemini":
        return gemini_key()
    return openrouter_key()


def apply_ai_keys_to_env(
    env: dict,
    provider: Optional[str] = None,
    byok_key: Optional[str] = None,
) -> dict:
    """Write isolated keys into a job env. Never copies Gemini↔OpenRouter.

    OpenRouter jobs keep ``OPENROUTER_API_KEY`` (host compose or BYOK) and
    drop ``GEMINI_API_KEY`` so a worker cannot leak an AI Studio token.
    Gemini jobs keep ``GEMINI_API_KEY`` and drop ``OPENROUTER_API_KEY``.
    ``MANAGED_GEMINI_API_KEY`` is never copied into ``OPENROUTER_API_KEY``.
    """
    chosen = (provider or "").strip().lower()
    if chosen not in PROVIDERS:
        chosen = resolve_provider()
    env["AI_PROVIDER"] = chosen
    byok = (byok_key or "").strip() or None

    if chosen == "openrouter":
        env.pop("GEMINI_API_KEY", None)
        if byok:
            env["OPENROUTER_API_KEY"] = assert_openrouter_safe_key(byok)
        elif looks_like_ai_studio_key(env.get("OPENROUTER_API_KEY") or ""):
            env.pop("OPENROUTER_API_KEY", None)
    else:
        env.pop("OPENROUTER_API_KEY", None)
        if byok:
            env["GEMINI_API_KEY"] = assert_google_safe_key(byok)
        elif looks_like_openrouter_key(env.get("GEMINI_API_KEY") or ""):
            env.pop("GEMINI_API_KEY", None)
    return env


def map_model_id(name: str) -> str:
    """Map a bare ``gemini-*`` id to OpenRouter's ``google/<name>`` form."""
    model = (name or "").strip()
    if model.startswith("gemini-") and "/" not in model:
        return f"google/{model}"
    return model


def resolve_request_provider(explicit: Optional[str] = None, *, billing: bool = False) -> str:
    """Provider for one request. An explicit header/setting wins.

    Cloud/billing defaults to Gemini (managed key) unless the request
    explicitly asks for OpenRouter. Self-host without an explicit choice
    still infers from env keys (OpenRouter if only ``OPENROUTER_API_KEY``).
    """
    chosen = (explicit or "").strip().lower()
    if chosen in PROVIDERS:
        return chosen
    if billing:
        return "gemini"
    return resolve_provider()


def resolve_model(explicit: Optional[str] = None, provider: Optional[str] = None) -> str:
    """Model id for ``provider`` (or the process default).

    Gemini-native callers must pass ``provider='gemini'`` so a compose
    ``OPENROUTER_API_KEY`` cannot turn the id into ``google/gemini-2.5-flash``.
    """
    chosen = (provider or "").strip().lower() or resolve_provider()
    if chosen == "gemini":
        raw = (explicit or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
        if raw.startswith("google/"):
            raw = raw.split("/", 1)[1]
        return raw or DEFAULT_GEMINI_MODEL
    raw = (explicit or os.getenv("OPENROUTER_MODEL") or os.getenv("GEMINI_MODEL")
           or DEFAULT_OPENROUTER_MODEL)
    return map_model_id(str(raw).strip() or DEFAULT_OPENROUTER_MODEL)


def openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    key = assert_openrouter_safe_key(
        api_key if api_key is not None else openrouter_key())
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
        default_headers={
            "HTTP-Referer": os.getenv(
                "OPENROUTER_HTTP_REFERER",
                "https://github.com/JulioDevEnviagora/openshorts",
            ),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "OpenShorts"),
        },
    )


def gemini_client(api_key: Optional[str] = None):
    """Native Google AI Studio client. Refuses OpenRouter keys."""
    if resolve_provider() != "gemini":
        raise RuntimeError(
            "Refusing to construct google.genai while AI_PROVIDER=openrouter "
            "(would send the key to generativelanguage.googleapis.com).")
    key = assert_google_safe_key(
        api_key if api_key is not None else gemini_key())
    from google import genai
    return genai.Client(api_key=key)


def make_client(api_key: Optional[str] = None, provider: Optional[str] = None):
    """Build the client for ``provider`` (or the process default).

    An explicit provider wins over ``AI_PROVIDER`` so in-process request
    handlers can honour ``X-AI-Provider`` without mutating the server env.
    Keys are still isolated: Gemini keys never go to OpenRouter and vice versa.
    """
    chosen = (provider or "").strip().lower() or resolve_provider()
    if chosen == "gemini":
        from google import genai
        key = assert_google_safe_key(
            api_key if api_key is not None else gemini_key())
        return genai.Client(api_key=key)
    return openrouter_client(api_key)


def jpeg_image_url_parts(frames: List[bytes]) -> list:
    """OpenAI-style image_url parts from JPEG bytes (data URIs)."""
    parts = []
    for data in frames:
        b64 = base64.b64encode(data).decode("ascii")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return parts


# --- Structured output schemas (passed as response_schema so the API
# --- guarantees the format instead of us repairing free-form JSON). ---

class ScoredWindowModel(BaseModel):
    id: str
    start: float
    end: float
    score: int
    reason: str


class ScoreResponse(BaseModel):
    windows: List[ScoredWindowModel]


class DetailClipModel(BaseModel):
    start: float
    end: float
    source_window_id: str
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str


class DetailResponse(BaseModel):
    shorts: List[DetailClipModel]


# Visual (no-transcript) clip selection. Gemini can upload the file; OpenRouter
# only sees sampled stills (timestamps are approximate). Same output shape as
# DetailClipModel minus the transcript-only source_window_id.
class VisualClipModel(BaseModel):
    start: float
    end: float
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str


class VisualResponse(BaseModel):
    shorts: List[VisualClipModel]


VISUAL_PROMPT_TEMPLATE = """
You are a senior short-form video editor. This video has NO speech/audio — judge
it purely by what you SEE. Watch the whole thing and pick the {min_clips}–{max_clips} MOST engaging
visual moments for TikTok / Reels / Shorts (action, reveals, transformations,
striking or funny shots, satisfying payoffs, dramatic movement).

TIME CONTRACT — STRICT:
- Timestamps in ABSOLUTE SECONDS from the start (usable with ffmpeg -ss/-to).
- Only numbers with up to 3 decimals (e.g. 0, 12.5, 47.250).
- 0 <= start < end <= {video_duration}.
- Each clip {min_secs:g} to {max_secs:g} seconds long. If the whole video is
  shorter than {min_secs:g}s, return one clip spanning the full video.
- Cut on visual scene changes, never mid-motion.

For each clip write catchy copy in {language} (a scroll-stopping hook, a TikTok
and an Instagram description, and a YouTube title ≤100 chars). Order clips best
to worst by how likely they are to stop a viewer scrolling.
"""


class LayoutChoice(BaseModel):
    layout: str
    confidence: float
    why: str


# Scored 94/92/96% over the 48-clip corpus against hand-checked labels, with
# 0-1 false positives out of the 28 clips that must not be touched. Do not
# reword casually: the wins come from the explicit "none is usually right"
# instruction and from naming the exact decorations (corner bugs, score
# counters, subtitles) that four earlier attempts kept mistaking for content.
LAYOUT_CHOICE_PROMPT = """
These frames are sampled at regular intervals from a single landscape video.
You are choosing how to re-frame that video into a vertical 9:16 clip.

Pick ONE layout:

- "none": crop to the speaker and fill the frame. This is the RIGHT answer for
  ordinary talking heads, interviews shot in close-up, b-roll, sport, action,
  music, and any footage whose meaning survives a centre crop. Corner logos,
  score bugs, subscriber counters, lower-thirds and burned-in subtitles do NOT
  change this: they are decoration, and losing them costs nothing.
- "screencast": keep the screen. ONLY when the video is built around a screen
  recording, slides, a spreadsheet, a chart or a map that the viewer must read
  to follow it. If you cannot read words or numbers off the screen that matter
  to the point being made, it is not this.
  (A "camera_inset" option was added here and removed on 31-jul-2026. Whether a
  webcam is composited into a corner of that screen is not something the model
  can see: on the five clips that have one it answered "screencast" every time,
  in both runs, while overall accuracy fell from 92% to 83-85%. camera_inset.py
  finds the same five geometrically with no false positives, so that question is
  answered downstream instead of being asked here.)
- "split": stack two people. ONLY when two people are visible IN THE SAME SHOT
  at the same time in most frames, talking to each other. Frames that alternate
  between one-person close-ups are NOT this, however many people appear.

"none" is by far the most common correct answer. Choose anything else only if
you would defend it to an editor. If you are unsure, answer "none".

confidence is 0..1. why is at most 12 words.
"""


class WideContentRangeModel(BaseModel):
    start: float
    end: float
    what: str
    width_fraction: float


class WideContentResponse(BaseModel):
    ranges: List[WideContentRangeModel]


WIDE_CONTENT_PROMPT_TEMPLATE = """
You are preparing a landscape video to be re-framed to a vertical 9:16 crop.
The crop keeps a tall centre strip and THROWS AWAY the left and right sides.

List every time range where on-screen content would be cut by that, and for each
one report HOW MUCH OF THE FRAME WIDTH the content spans.

width_fraction is the single most important field. Measure the content's own
horizontal extent, from its left edge to its right edge, as a fraction of the
full frame width:
- a spreadsheet, slide, screen recording or map filling the picture: 0.9 - 1.0
- a chart or diagram beside a speaker: 0.4 - 0.7
- a lower-third or headline strip across the bottom: 0.6 - 0.9
- a logo, channel bug, score counter or subscriber count in a corner: 0.1 - 0.2
- subtitles centred at the bottom: 0.3 - 0.5

Report what you actually see. Do NOT inflate the number to make a range seem
worth reporting, and do NOT leave out corner graphics — report them with their
true small width_fraction. A range reported honestly at 0.15 is useful; the same
range reported at 0.9 makes the video worse.

COUNT a range when the frame shows:
- a screen recording, slide, spreadsheet, chart, graph or map
- headlines, labels, statistics or comparison tables burned into the picture
- a side-by-side or split-screen layout
- any diagram or product shot where the edges carry the meaning

DO NOT count an ordinary talking head, even against a busy background, and do
not count b-roll, landscapes, crowds or action footage with no graphics.

TIME CONTRACT — STRICT:
- ABSOLUTE SECONDS from the start, numbers only, up to 3 decimals.
- 0 <= start < end <= {video_duration}.
- Merge ranges that are less than 1 second apart.
- Return an EMPTY list if the video never shows such content. An empty list is
  the correct, expected answer for most talking-head and b-roll videos — do not
  invent ranges to seem useful.

For "what", name the content in three words or fewer (e.g. "stock chart",
"spreadsheet", "corner ticker").
"""


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _log(message: str) -> None:
    stream = sys.stdout
    text = str(message)
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text + "\n")
    stream.flush()

SCORE_PROMPT_TEMPLATE = """
You are a senior short-form video strategist.
Select the MOST viral candidate windows from this batch.

Rules:
- Return only valid JSON.
- Choose up to 3 windows from this batch.
- `score` must be an integer from 0 to 100.
- THE 2-SECOND TEST is the main criterion: would the first 2 seconds of this
  moment force a cold viewer (no context) to keep watching? Windows that only
  work with prior context score low.
- Prefer windows with strong hooks, conflict, surprise, outrage, emotion,
  novelty, big numbers, or a clear payoff.
- Ignore weak filler, housekeeping, outros, rambling transitions, and
  low-signal padding unless there is an obvious hook or payoff.

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
WINDOWS_JSON:
{windows_json}

Return only:
{{
  "windows": [
    {{
      "id": "<window id>",
      "start": <number>,
      "end": <number>,
      "score": <integer 0-100>,
      "reason": "<very short reason>"
    }}
  ]
}}
"""

DETAIL_PROMPT_TEMPLATE = """
You are a senior short-form video editor and viral copywriter.
Choose the BEST short clips from these shortlisted candidate windows.

CLIP RULES:
- Return only valid JSON.
- Each clip must be {min_secs:g} to {max_secs:g} seconds long, in absolute seconds from the start of the source video.
- Stay within the candidate window boundaries.
- THE 2-SECOND RULE: the clip MUST open on its strongest moment. If the first
  2 seconds would not stop a cold viewer from scrolling, move the start or skip the clip.
- Start slightly before the hook and end slightly after the payoff when possible.
- Do not cut in the middle of a word or phrase.
- No generic intros/outros unless they are the hook.
- STANDS ALONE: the clip must make sense to someone who has seen nothing else.
  If it opens on a pronoun, a "that", a "so anyway", or an answer whose question
  was asked earlier, move the start back to where the idea begins or skip it.
  A brilliant moment that needs the previous five minutes is not a clip.
  Fix this by moving the START earlier, never by cutting the ending short: a
  clip that loses its payoff to gain context has traded down.
- HOW MANY: return {min_clips} to {max_clips} clips. Work through EVERY candidate
  window — they were already scored as the best moments in the video, so a window
  that yields nothing should be the exception, not the norm. Two or three clips
  from one window are fine when they are genuinely different moments. The rules
  above let you skip a weak clip; they are not a licence to return one clip and
  stop. Only fall short of {min_clips} when the material truly does not hold
  them, and never pad with a clip you would not publish yourself.
- DIVERSITY: never return two clips that make the same point, tell the same
  story, or land the same joke — even across different windows. Pick the
  stronger one and drop the other. Two clips on the same broad topic are fine
  as long as each lands its own moment.

HOOK PLAYBOOK — pick the strongest fitting pattern for `viral_hook_text` (max 10 words):
- Open question: "Why does everyone get this wrong?"
- Hot take / controversy: "Stop doing this. Seriously."
- Number / fact shock: "97% of people miss this."
- Story loop: "This one email almost ruined me."
- POV / pattern interrupt: "POV: you finally understand it."
(These are English PATTERNS — always write the actual hook in TRANSCRIPT_LANGUAGE.)

COPY RULES — ALL text fields (descriptions, title, hook) MUST be written in TRANSCRIPT_LANGUAGE ({language}):
- Descriptions (TikTok + Instagram): 1-2 punchy sentences that tease the payoff
  without spoiling it, then 3-5 topically relevant hashtags. No generic hashtag spam.
- `video_title_for_youtube_short`: max 100 chars, curiosity-driven, no fake claims.
- `predicted_score`: honest 0-100 estimate of viral potential.

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
CANDIDATE_WINDOWS_JSON:
{windows_json}

Return only:
{{
  "shorts": [
    {{
      "start": <number>,
      "end": <number>,
      "source_window_id": "<window id>",
      "predicted_score": <integer 0-100>,
      "video_description_for_tiktok": "<description + hashtags>",
      "video_description_for_instagram": "<description + hashtags>",
      "video_title_for_youtube_short": "<title max 100 chars>",
      "viral_hook_text": "<short overlay max 10 words>"
    }}
  ]
}}
"""


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _escape_invalid_unicode_escapes(text: str) -> str:
    chars = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "u":
            hex_digits = text[i + 2:i + 6]
            if len(hex_digits) < 4 or any(ch not in "0123456789abcdefABCDEF" for ch in hex_digits):
                chars.append("\\\\u")
                i += 2
                continue
        chars.append(text[i])
        i += 1
    return "".join(chars)


def _parse_json_response_text(text: str) -> dict:
    if not text:
        raise ValueError("Gemini returned an empty response body.")
    candidate = _extract_json_candidate(text).replace("\x00", "").strip()
    if not candidate:
        raise ValueError("Gemini response did not contain a JSON object.")
    parse_attempts = [candidate]
    sanitized_candidate = _escape_invalid_unicode_escapes(candidate)
    if sanitized_candidate != candidate:
        parse_attempts.append(sanitized_candidate)
    last_error: Optional[Exception] = None
    for parse_candidate in parse_attempts:
        try:
            return json.loads(parse_candidate)
        except json.JSONDecodeError as e:
            last_error = e
    raise ValueError(f"Failed to parse Gemini JSON response: {last_error}")


class GeminiBlockedError(ValueError):
    """The API refused the request for content-policy reasons.

    Deterministic: the same payload is rejected every time (verified in prod,
    23-jul-2026 — a stand-up video came back PROHIBITED_CONTENT in ~300ms on
    every attempt), and BLOCK_NONE safety settings do NOT lift it. Retrying is
    pointless, so callers must fail fast with a message that tells the user the
    video's content is the problem, not the service."""


_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST",
                           "SPII", "IMAGE_SAFETY", "RECITATION"}


def raise_if_blocked(response):
    """Raise GeminiBlockedError when the API refused to answer on policy grounds.

    Accepts OpenRouter/OpenAI chat completions and leftover google.genai
    responses (File API callers that still go through this helper).
    """
    # OpenAI / OpenRouter
    for c in getattr(response, "choices", None) or []:
        fr = getattr(c, "finish_reason", None)
        name = (getattr(fr, "name", None) or str(fr or "")).upper()
        if name in _BLOCKED_FINISH_REASONS or name == "CONTENT_FILTER":
            raise GeminiBlockedError(
                f"OpenRouter blocked its answer for this video ({name}). The AI "
                "provider's usage policies reject this material, so it can't be analyzed.")
        msg = getattr(c, "message", None)
        refusal = getattr(msg, "refusal", None) if msg is not None else None
        if refusal:
            raise GeminiBlockedError(
                f"OpenRouter blocked its answer for this video (refusal). The AI "
                "provider's usage policies reject this material, so it can't be analyzed.")

    pf = getattr(response, "prompt_feedback", None)
    reason = getattr(pf, "block_reason", None)
    if reason:
        name = getattr(reason, "name", None) or str(reason)
        raise GeminiBlockedError(
            f"Gemini blocked this video's content ({name}). The AI provider's "
            "usage policies reject this material, so it can't be analyzed.")
    for c in (getattr(response, "candidates", None) or []):
        fr = getattr(c, "finish_reason", None)
        name = (getattr(fr, "name", None) or str(fr or "")).upper()
        if name in _BLOCKED_FINISH_REASONS:
            raise GeminiBlockedError(
                f"Gemini blocked its answer for this video ({name}). The AI "
                "provider's usage policies reject this material, so it can't be analyzed.")


def _get_response_text(response) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        parts = []
        for choice in choices:
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if content:
                parts.append(content)
        if parts:
            return "\n".join(parts).strip()

    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()


def _usage_token_counts(usage) -> tuple:
    """Map OpenAI/OpenRouter usage (or leftover Gemini usage_metadata) to tokens.

    Returns ``(prompt, output, thinking, billed_output)``.

    OpenRouter ``reasoning_tokens`` already sit inside ``completion_tokens`` —
    they are reported as ``thinking`` but must not be added again.
    Gemini-legacy ``thoughts_token_count`` is *not* inside
    ``candidates_token_count`` and is the only extra that gets billed.
    """
    if usage is None:
        return 0, 0, 0, 0
    prompt_tokens = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "prompt_token_count", None)
        or 0
    )
    output_tokens = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "candidates_token_count", None)
        or 0
    )
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        getattr(details, "reasoning_tokens", None) if details is not None else None
    ) or 0
    gemini_thoughts = getattr(usage, "thoughts_token_count", None) or 0
    thinking_tokens = int(gemini_thoughts or reasoning_tokens or 0)
    billed_output = int(output_tokens or 0) + int(gemini_thoughts or 0)
    return (
        int(prompt_tokens or 0),
        int(output_tokens or 0),
        thinking_tokens,
        billed_output,
    )


def _calculate_cost_analysis(response, model_name: str) -> Optional[dict]:
    usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    if not usage:
        return None
    prompt_tokens, output_tokens, thinking_tokens, billed_output = _usage_token_counts(usage)
    prices = lookup_model_prices(model_name)
    if prices is None:
        # Unknown OpenRouter/Gemini id: report tokens, do not invent a rate.
        return {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "model": model_name,
            "price_estimated": True,
        }
    input_price_per_million, output_price_per_million = prices
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (billed_output / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model_name,
        "price_estimated": False,
    }


def _response_format_for(schema: Optional[type], strategy: str) -> Optional[dict]:
    """OpenAI-style response_format. Structured schema is the primary path."""
    if strategy == "json-text-recovery":
        return {"type": "json_object"}
    if strategy == "strict-json":
        return {"type": "json_object"}
    if schema is None or not hasattr(schema, "model_json_schema"):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": getattr(schema, "__name__", "response") or "response",
            "schema": schema.model_json_schema(),
        },
    }


def _temperature_for_strategy(strategy: str, mode: str) -> float:
    creative = mode == "detail"
    if strategy == "strict-json":
        return 0.7 if creative else 0.1
    if strategy == "json-text-recovery":
        return 0.2 if creative else 0.0
    return 0.9 if creative else 0.2


def _parse_structured_payload(raw_text: str, schema: Optional[type] = None) -> dict:
    """Prefer schema validation; fall back to the existing text-repair path."""
    if schema is not None and raw_text:
        try:
            return schema.model_validate_json(raw_text).model_dump()
        except Exception:
            pass
    data = _parse_json_response_text(raw_text)
    if schema is not None:
        try:
            return schema.model_validate(data).model_dump()
        except Exception:
            pass
    return data


def _is_openrouter_client(client) -> bool:
    chat = getattr(client, "chat", None)
    return chat is not None and getattr(chat, "completions", None) is not None


def complete_gemini_json(client, model_name, prompt, schema, strategy="structured-schema",
                         mode="score", contents=None):
    """Native google.genai generate_content with the original schema path."""
    from google.genai import types as genai_types

    kwargs = {
        "response_mime_type": "application/json",
        "candidate_count": 1,
        "temperature": _temperature_for_strategy(strategy, mode),
    }
    if strategy == "structured-schema" and schema is not None:
        kwargs["response_schema"] = schema
    config = genai_types.GenerateContentConfig(**kwargs)
    response = client.models.generate_content(
        model=model_name,
        contents=contents if contents is not None else prompt,
        config=config,
    )
    raise_if_blocked(response)
    raw_text = _get_response_text(response)
    parsed_obj = getattr(response, "parsed", None)
    if parsed_obj is not None:
        parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        if isinstance(parsed, dict):
            return parsed, response
    parsed = _parse_structured_payload(raw_text, schema)
    return parsed, response


def complete_openrouter_json(client, model_name, prompt, schema, strategy="structured-schema",
                             mode="score", messages=None):
    """OpenRouter chat.completions.create with response_format."""
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": _temperature_for_strategy(strategy, mode),
    }
    response_format = _response_format_for(schema, strategy)
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    raise_if_blocked(response)
    raw_text = _get_response_text(response)
    parsed_obj = getattr(response, "parsed", None)
    if parsed_obj is not None:
        parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        if isinstance(parsed, dict):
            return parsed, response
    parsed = _parse_structured_payload(raw_text, schema)
    return parsed, response


def complete_json(client, model_name, prompt, schema, strategy="structured-schema",
                  mode="score", messages=None, contents=None):
    """One structured JSON call on whichever client ``make_client`` built.

    OpenRouter uses ``messages`` (image_url data URIs). Native Gemini uses
    ``contents`` (File API objects or Part bytes). Returns ``(parsed_dict, response)``.
    """
    if _is_openrouter_client(client):
        return complete_openrouter_json(
            client, model_name, prompt, schema, strategy=strategy, mode=mode,
            messages=messages)
    return complete_gemini_json(
        client, model_name, prompt, schema, strategy=strategy, mode=mode,
        contents=contents)


def complete_json_with_frames(client, model_name, prompt, schema, frames,
                              strategy="structured-schema", mode="score"):
    """Send JPEG bytes as chat image_url data URIs (OpenRouter) or Parts (Gemini)."""
    if _is_openrouter_client(client):
        content = jpeg_image_url_parts(frames)
        content.append({"type": "text", "text": prompt})
        return complete_openrouter_json(
            client, model_name, prompt, schema, strategy=strategy, mode=mode,
            messages=[{"role": "user", "content": content}])
    from google.genai import types as genai_types
    parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]
    return complete_gemini_json(
        client, model_name, prompt, schema, strategy=strategy, mode=mode,
        contents=parts + [prompt])


def main() -> int:
    _configure_stdio()

    parser = argparse.ArgumentParser(description="Run a single clip scoring/detailing request.")
    parser.add_argument("--mode", choices=["score", "detail"], required=True)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--strategy", default="structured-schema")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    api_key = resolve_api_key()
    if not api_key:
        raise SystemExit("Missing API key for the active AI provider.")

    with open(args.input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = resolve_model(args.model)
    client = make_client(api_key)
    provider = resolve_provider()
    language = str(payload.get("language") or "unknown")

    template = SCORE_PROMPT_TEMPLATE if args.mode == "score" else DETAIL_PROMPT_TEMPLATE
    schema = ScoreResponse if args.mode == "score" else DetailResponse
    fmt = {
        "video_duration": payload["video_duration"],
        "language": language,
        "windows_json": json.dumps(payload["windows"], ensure_ascii=False),
    }
    if args.mode != "score":
        # Score mode receives every window, not a shortlist, so a count target
        # derived from it would be meaningless — and the score template has no
        # placeholder for one anyway.
        fmt["min_clips"], fmt["max_clips"] = clip_count_targets(len(payload.get("windows") or []))
        fmt["min_secs"], fmt["max_secs"] = clip_duration_bounds()
    prompt = template.format(**fmt)

    _log(f"🤖 Worker request: provider={provider} mode={args.mode} strategy={args.strategy} model={model_name} items={len(payload.get('windows', []))}")
    parsed, response = complete_json(
        client, model_name, prompt, schema, strategy=args.strategy, mode=args.mode)
    raw_text = _get_response_text(response)
    result = {
        "mode": args.mode,
        "payload": parsed,
        "cost_analysis": _calculate_cost_analysis(response, model_name),
        "raw_text": raw_text,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log(f"✅ Worker success: provider={provider} mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

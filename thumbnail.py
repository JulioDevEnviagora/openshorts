import os
from typing import List

from PIL import Image
from pydantic import BaseModel

# Text/analysis model for titles and descriptions (OpenRouter chat).
TEXT_MODEL = (
    os.environ.get("OPENROUTER_MODEL_THUMBNAIL")
    or os.environ.get("OPENROUTER_MODEL")
    or os.environ.get("GEMINI_MODEL_THUMBNAIL")
    or os.environ.get("GEMINI_MODEL")
    or None
)


class TitleRecommendation(BaseModel):
    index: int
    reason: str


class TitlesResponse(BaseModel):
    titles: List[str]
    transcript_summary: str = ""
    language: str = ""
    recommended: List[TitleRecommendation] = []


class RefinedTitlesResponse(BaseModel):
    titles: List[str]


class DescriptionResponse(BaseModel):
    description: str


def _text_model():
    import gemini_worker
    return gemini_worker.resolve_model(explicit=TEXT_MODEL, provider="openrouter")


def _openrouter_client(api_key):
    import gemini_worker
    gemini_worker.assert_openrouter_safe_key(api_key)
    return gemini_worker.openrouter_client(api_key)


def analyze_video_for_titles(api_key, video_path, transcript=None):
    """
    Transcribes a video and uses OpenRouter to suggest viral YouTube titles.
    If transcript is provided, skips Whisper transcription.
    No File API upload — the model sees the transcript, not the video file.
    Returns: { "titles": [...], "transcript_summary": "...", "language": "...", "segments": [...], "video_duration": ... }
    """
    import gemini_worker

    if transcript is None:
        from main import transcribe_video
        print("🎬 [Thumbnail] Transcribing video...")
        transcript = transcribe_video(video_path)
    else:
        print("🎬 [Thumbnail] Using pre-computed transcript (Whisper already done)...")

    prompt = f"""You are a YouTube title expert who creates viral, click-worthy titles.

Analyze this video from its transcript, then suggest 10 YouTube titles that would maximize CTR (click-through rate).

TRANSCRIPT:
{transcript['text']}

RULES:
- Titles must be under 70 characters
- Use power words, curiosity gaps, and emotional triggers
- Mix styles: how-to, listicle, story-driven, controversial, question-based
- Make them specific to the actual content, not generic
- Include numbers where appropriate
- Consider the language of the video (detected: {transcript['language']})
- Titles should be in the SAME LANGUAGE as the video transcript

Also provide a brief summary of the video content (2-3 sentences).

After generating all 10 titles, pick the TOP 2 you most recommend and explain concisely WHY (CTR potential, emotional hook, uniqueness, etc.). Reference them by their 0-based index in the titles array.

OUTPUT JSON:
{{
    "titles": ["title1", "title2", ...],
    "transcript_summary": "Brief summary of the video content...",
    "language": "{transcript['language']}",
    "recommended": [
        {{"index": 0, "reason": "Why this title is best..."}},
        {{"index": 3, "reason": "Why this title is second best..."}}
    ]
}}"""

    print("🤖 [Thumbnail] Asking OpenRouter for title suggestions...")
    model_name = _text_model()
    client = _openrouter_client(api_key)
    parsed, _response = gemini_worker.complete_json(
        client, model_name, prompt, TitlesResponse)

    segments = transcript.get("segments", [])
    video_duration = segments[-1]["end"] if segments else 0

    result = parsed if isinstance(parsed, dict) else {}
    titles = result.get("titles") or []
    if not titles:
        print("❌ [Thumbnail] Title response had no titles — using transcript fallback.")
        return {
            "titles": ["Could not generate titles - please try again"],
            "transcript_summary": transcript["text"][:500],
            "language": transcript["language"],
            "segments": segments,
            "video_duration": video_duration,
        }
    result["transcript_summary"] = result.get("transcript_summary", "")
    result["language"] = result.get("language", transcript["language"])
    result["segments"] = segments
    result["video_duration"] = video_duration
    return result


def refine_titles(api_key, context, user_message, conversation_history=None):
    """
    Takes video context + user feedback and returns refined title suggestions.
    """
    import gemini_worker
    client = _openrouter_client(api_key)
    model_name = _text_model()

    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            history_text += f"\n{role.upper()}: {msg['content']}"

    prompt = f"""You are a YouTube title expert. Based on the video context and the user's feedback, suggest 8 new refined YouTube titles.

VIDEO CONTEXT:
{context}

CONVERSATION HISTORY:{history_text}

USER'S NEW REQUEST:
{user_message}

RULES:
- Titles must be under 70 characters
- Incorporate the user's feedback/direction
- Keep titles viral and click-worthy
- If the user asks for a specific style, follow it
- Titles should be in the same language as the original content

OUTPUT JSON:
{{
    "titles": ["title1", "title2", ...]
}}"""

    parsed, _response = gemini_worker.complete_json(
        client, model_name, prompt, RefinedTitlesResponse)
    if isinstance(parsed, dict) and parsed.get("titles"):
        return parsed
    print("❌ [Thumbnail] Failed to parse refined titles")
    return {"titles": ["Could not refine titles - please try again"]}


def _save_b64_image(b64_json: str, filepath: str) -> None:
    import base64
    from io import BytesIO

    raw = base64.b64decode(b64_json)
    image = Image.open(BytesIO(raw))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(filepath, format="JPEG", quality=90)


def generate_thumbnail(api_key, title, session_id, face_image_path=None, bg_image_path=None, extra_prompt="", count=3, video_context=""):
    """
    Generates YouTube thumbnails via the OpenRouter Image API.
    No google.genai / Imagen. Returns list of saved image paths (relative URLs).
    """
    import gemini_worker
    gemini_worker.assert_openrouter_safe_key(api_key)

    output_dir = os.path.join("output", "thumbnails", session_id)
    os.makedirs(output_dir, exist_ok=True)

    context_block = ""
    if video_context:
        context_block = f"""
VIDEO CONTEXT (use this to understand the video and design a relevant thumbnail):
{video_context}
"""

    extra_block = ""
    if extra_prompt:
        extra_block = f"""
⚠️ MANDATORY USER INSTRUCTIONS (MUST follow these exactly — they override any default behavior):
{extra_prompt}
"""

    text_prompt = f"""Generate a professional, eye-catching YouTube thumbnail image.

VIDEO TITLE (for reference — do NOT put the full title on the thumbnail): "{title}"
{context_block}
TEXT ON THE THUMBNAIL:
- Based on the title AND the video context, create a SHORT visual hook: 1 to 5 words maximum
- It should capture the core emotion, surprise, or promise of the video
- The thumbnail text should COMPLEMENT the YouTube title (which appears below), not repeat it
- Examples: "$10K EN 30 DÍAS", "ESTO FUNCIONA", "NO LO SABÍAS", "GRATIS 🔥"
- Use ALL CAPS for maximum impact, split into 2-3 lines
{extra_block}
DESIGN REQUIREMENTS:
- The text MUST be large, bold, and high-contrast (readable at small sizes)
- Use vibrant, eye-catching colors that match the video's mood
- Professional YouTube thumbnail aesthetic
- Clean composition — text and face/subject as clear focal points
- NO clutter, NO small text, NO watermarks
- Aspect ratio 16:9"""

    if face_image_path and os.path.exists(face_image_path):
        text_prompt += "\n- Include the provided face/person prominently with an exaggerated expression (surprise, excitement, shock)"

    if bg_image_path and os.path.exists(bg_image_path):
        text_prompt += "\n- Use the provided background image as the base/backdrop"

    references = []
    if face_image_path and os.path.exists(face_image_path):
        references.append(gemini_worker.image_file_reference(face_image_path))
    if bg_image_path and os.path.exists(bg_image_path):
        references.append(gemini_worker.image_file_reference(bg_image_path))

    thumbnails = []
    last_error = None
    image_model = gemini_worker.resolve_image_model()
    for i in range(count):
        print(f"🎨 [Thumbnail] Generating thumbnail {i + 1}/{count} via OpenRouter ({image_model})...")
        try:
            payload = gemini_worker.generate_openrouter_images(
                api_key,
                text_prompt,
                n=1,
                aspect_ratio="16:9",
                quality="high",
                output_format="jpeg",
                references=references or None,
            )
            saved = False
            for item in payload.get("data") or []:
                b64 = (item or {}).get("b64_json")
                if not b64:
                    continue
                filename = f"thumb_{i + 1}.jpg"
                filepath = os.path.join(output_dir, filename)
                _save_b64_image(b64, filepath)
                thumbnails.append(f"/thumbnails/{session_id}/{filename}")
                print(f"✅ [Thumbnail] Saved: {filepath}")
                saved = True
                break
            if not saved:
                last_error = "OpenRouter returned no image bytes"
                print(f"❌ [Thumbnail] Generation {i + 1} failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"❌ [Thumbnail] Generation {i + 1} failed: {e}")

    if not thumbnails and last_error:
        raise RuntimeError(f"All thumbnail generations failed. Last error: {last_error}")

    return thumbnails


def generate_youtube_description(api_key, title, transcript_segments, language, video_duration):
    """
    Uses OpenRouter to generate a YouTube description with chapter markers.
    Returns: { "description": "full description text with chapters" }
    """
    import gemini_worker
    client = _openrouter_client(api_key)
    model_name = _text_model()

    formatted_segments = []
    for seg in transcript_segments:
        start = seg.get("start", 0)
        mins = int(start // 60)
        secs = int(start % 60)
        timestamp = f"{mins}:{secs:02d}"
        formatted_segments.append(f"[{timestamp}] {seg.get('text', '').strip()}")

    segments_text = "\n".join(formatted_segments)

    dur_mins = int(video_duration // 60)
    dur_secs = int(video_duration % 60)
    duration_str = f"{dur_mins}:{dur_secs:02d}"

    prompt = f"""You are a YouTube SEO expert. Generate a complete YouTube video description for the following video.

VIDEO TITLE: "{title}"
VIDEO LANGUAGE: {language}
VIDEO DURATION: {duration_str}

TRANSCRIPT WITH TIMESTAMPS:
{segments_text}

REQUIREMENTS:
1. Write the description in the SAME LANGUAGE as the video ({language})
2. Start with a compelling 2-3 sentence summary/hook
3. Add relevant CTAs (subscribe, like, comment)
4. Generate YouTube CHAPTERS based on the transcript timestamps:
   - First chapter MUST start at 0:00
   - Minimum 3 chapters, each at least 10 seconds apart
   - Chapter titles should be concise and descriptive
   - Format: 0:00 Chapter Title
   - Place chapters in their own section with a blank line before and after
5. Add 5-10 relevant hashtags at the end
6. Keep the total description under 5000 characters

OUTPUT JSON: {{"description": "the ready-to-paste YouTube description, no markdown code blocks"}}"""

    print("🤖 [Thumbnail] Generating YouTube description with chapters...")
    parsed, _response = gemini_worker.complete_json(
        client, model_name, prompt, DescriptionResponse)
    description = ""
    if isinstance(parsed, dict):
        description = str(parsed.get("description") or "").strip()
    if description.startswith("```"):
        lines = description.split("\n")
        description = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return {"description": description}

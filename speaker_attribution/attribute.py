"""Stage 3: resolve anonymous speaker labels to named people/roles via Claude.

Takes the diarized transcript + editorial metadata (shotlist / script / byline)
and returns a validated mapping. The shotlist makes this deterministic matching
rather than open-ended inference, so we prompt for structured JSON and validate.
"""
from __future__ import annotations

import json
import os

import anthropic

from .models import AttributionResult, ReviewStatus, Segment, SpeakerMapping

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an assistant inside a newsroom video pipeline at a wire agency.
You map anonymous diarization labels (SPEAKER_00, SPEAKER_01, ...) to real people
using the editorial shotlist/script for the same video.

Rules:
- Use ONLY the provided shotlist/script and transcript. Never invent names.
- The correspondent's narration/questions map to the reporter named in the byline.
- Shotlists list soundbites in broadcast order; use segment order and quoted text
  overlap to match soundbites to speaker labels.
- If a speaker cannot be matched with confidence, keep name = "Unidentified"
  and set confidence = "low". A wrong attribution is far worse than an
  unidentified one.
- Respond with ONLY a JSON object, no markdown fences, no prose, shaped as:
{
  "mappings": [
    {
      "label": "SPEAKER_00",
      "name": "Mike Huckabee",
      "role": "U.S. Ambassador to Israel",
      "confidence": "high",
      "evidence": "one-sentence justification citing shotlist line + matching quote"
    }
  ],
  "warnings": ["optional list of issues, e.g. speaker count mismatch"]
}"""


def _transcript_for_prompt(segments: list[Segment], max_chars: int = 30_000) -> str:
    lines = [
        f"[{s.start:8.2f}-{s.end:8.2f}] {s.speaker}: {s.text}" for s in segments
    ]
    text = "\n".join(lines)
    return text[:max_chars]


def attribute_speakers(
    video_id: str,
    segments: list[Segment],
    shotlist: str,
    byline: str = "",
    client: anthropic.Anthropic | None = None,
) -> AttributionResult:
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_prompt = (
        f"VIDEO ID: {video_id}\n\n"
        f"BYLINE / CREDITS:\n{byline or '(none provided)'}\n\n"
        f"SHOTLIST / SCRIPT:\n{shotlist}\n\n"
        f"DIARIZED TRANSCRIPT:\n{_transcript_for_prompt(segments)}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return AttributionResult(
            video_id=video_id,
            segments=segments,
            mappings={},
            status=ReviewStatus.FAILED,
            warnings=[f"LLM returned non-JSON output: {exc}"],
        )

    mappings = {
        m["label"]: SpeakerMapping(
            label=m["label"],
            name=m.get("name", "Unidentified"),
            role=m.get("role", ""),
            confidence=m.get("confidence", "low"),
            evidence=m.get("evidence", ""),
        )
        for m in data.get("mappings", [])
    }
    warnings = list(data.get("warnings", []))

    status = _validate(segments, mappings, warnings)
    return AttributionResult(
        video_id=video_id,
        segments=segments,
        mappings=mappings,
        status=status,
        warnings=warnings,
    )


def _validate(
    segments: list[Segment],
    mappings: dict[str, SpeakerMapping],
    warnings: list[str],
) -> ReviewStatus:
    """Cheap programmatic checks; anything suspicious -> human review."""
    labels_in_audio = {s.speaker for s in segments}

    unmapped = labels_in_audio - set(mappings)
    if unmapped:
        warnings.append(f"Unmapped speaker labels: {sorted(unmapped)}")

    low_conf = [m.label for m in mappings.values() if m.confidence != "high"]
    if low_conf:
        warnings.append(f"Low/medium confidence mappings: {low_conf}")

    unidentified = [m.label for m in mappings.values() if m.name == "Unidentified"]
    if unidentified:
        warnings.append(f"Unidentified speakers: {unidentified}")

    return ReviewStatus.NEEDS_REVIEW if warnings else ReviewStatus.AUTO_OK

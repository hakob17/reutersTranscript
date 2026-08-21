"""Stage 4: write outputs — WebVTT with speaker voice tags + archive JSON."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import AttributionResult


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_vtt(result: AttributionResult, path: Path) -> None:
    """WebVTT with <v Speaker> voice tags — JW Player renders these as captions."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(result.segments, 1):
        who = result.display_name(seg.speaker)
        lines.append(str(i))
        lines.append(f"{_ts(seg.start)} --> {_ts(seg.end)}")
        lines.append(f"<v {who}>{seg.text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(result: AttributionResult, path: Path) -> None:
    """Structured record for CMS/archive: searchable soundbites per person."""
    payload = {
        "video_id": result.video_id,
        "status": result.status.value,
        "warnings": result.warnings,
        "speakers": {label: asdict(m) for label, m in result.mappings.items()},
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker_label": seg.speaker,
                "speaker": result.display_name(seg.speaker),
                "language": seg.language,
                "text": seg.text,
            }
            for seg in result.segments
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_labeled_txt(result: AttributionResult, path: Path) -> None:
    """Human-readable transcript for the review UI / quick editorial check."""
    lines = []
    prev = None
    for seg in result.segments:
        who = result.display_name(seg.speaker)
        if who != prev:
            lines.append(f"\n{who}:")
            prev = who
        lines.append(f"  {seg.text}")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

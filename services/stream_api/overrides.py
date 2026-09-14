"""Editor name corrections for a video's speakers.

Stored per video next to its cached event log and applied to events as they
are streamed, so a correction shows on every load — cached replays and live
runs alike — without reprocessing. A human correction outranks every
automatic source.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LABEL_RE = re.compile(r"^SPEAKER_\d{1,3}$")
MAX_NAME = 80


def path_for(cache_dir: Path, video_id: str) -> Path:
    return cache_dir / f"{video_id}.names.json"


def load(cache_dir: Path, video_id: str) -> dict[str, str]:
    p = path_for(cache_dir, video_id)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items()
                if LABEL_RE.match(k) and isinstance(v, str)}
    except Exception:
        return {}


def clear(cache_dir: Path, video_id: str) -> None:
    """Drop every correction for a video (reprocessing starts clean)."""
    path_for(cache_dir, video_id).unlink(missing_ok=True)


def clean_name(name: str) -> str:
    """Collapse whitespace and strip control characters; '' clears."""
    name = "".join(ch for ch in (name or "") if ch.isprintable())
    return re.sub(r"\s+", " ", name).strip()[:MAX_NAME]


def save(cache_dir: Path, video_id: str, label: str, name: str) -> dict[str, str]:
    if not LABEL_RE.match(label or ""):
        raise ValueError("label must look like SPEAKER_00")
    current = load(cache_dir, video_id)
    name = clean_name(name)
    if name:
        current[label] = name
    else:
        current.pop(label, None)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path_for(cache_dir, video_id).write_text(
        json.dumps(current, ensure_ascii=False), encoding="utf-8")
    return current


def apply(event: dict, overrides: dict[str, str]) -> dict:
    """Return the event with editor names applied (the original is untouched,
    so the cached event log keeps the pipeline's own output)."""
    if not overrides:
        return event
    if event.get("type") == "names":
        mapping = {k: dict(v) for k, v in event.get("mapping", {}).items()}
        for label, name in overrides.items():
            mapping[label] = {"name": name, "role": "", "confidence": "human"}
        return {**event, "mapping": mapping}
    if event.get("type") == "face_tracks":
        tracks = []
        for tr in event.get("tracks", []):
            label = tr.get("speaker_label")
            if label in overrides:
                tr = {k: v for k, v in tr.items()
                      if k not in ("gallery", "name_tentative")}
                tr["name"] = overrides[label]
            tracks.append(tr)
        return {**event, "tracks": tracks}
    return event

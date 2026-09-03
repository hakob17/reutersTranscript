"""Caption-first fast path: use the broadcaster's own captions instead of ASR.

Reuters-style HLS master playlists declare a WebVTT subtitle rendition
(#EXT-X-MEDIA:TYPE=SUBTITLES,URI="..."). When present, the caption cues give
us text + precise timings for free, so the pipeline only needs diarization
(who speaks when) + attribution (names). That drops per-video time from
minutes to well under one, removes the ASR transcription-error class
entirely, and needs no large ASR model.

No third-party dependencies — parsing is regex-based on purpose so this
module can ship inside the attribution Lambda untouched.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

from .models import Segment

_TIMESTAMP = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


def _http_get(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def discover_caption_url(master_url: str) -> str | None:
    """Return the subtitle rendition URI declared in an HLS master playlist."""
    try:
        playlist = _http_get(master_url)
    except Exception:
        return None
    if not playlist.lstrip().startswith("#EXTM3U"):
        return None
    for line in playlist.splitlines():
        if line.startswith("#EXT-X-MEDIA:") and "TYPE=SUBTITLES" in line:
            m = re.search(r'URI="([^"]+)"', line)
            if m:
                return urllib.parse.urljoin(master_url, m.group(1))
    return None


def _ts_to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(text: str) -> list[dict]:
    """WebVTT/SRT -> [{"start", "end", "text"}], tags and settings stripped."""
    cues: list[dict] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        m = _TIMESTAMP.search(block)
        if not m:
            continue
        start = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        lines = block[m.end():].strip().splitlines()
        cue_text = _TAG.sub("", " ".join(line.strip() for line in lines)).strip()
        if cue_text:
            cues.append({"start": start, "end": end, "text": cue_text})
    cues.sort(key=lambda c: c["start"])
    return cues


def fetch_caption_cues(url: str) -> list[dict]:
    """Fetch cues from a VTT file, or from an HLS subtitle playlist of VTTs."""
    body = _http_get(url)
    if not body.lstrip().startswith("#EXTM3U"):
        return parse_vtt(body)

    cues: list[dict] = []
    seen: set[tuple[float, str]] = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for cue in parse_vtt(_http_get(urllib.parse.urljoin(url, line))):
            key = (cue["start"], cue["text"])
            if key not in seen:  # segmented VTTs repeat cues at boundaries
                seen.add(key)
                cues.append(cue)
    cues.sort(key=lambda c: c["start"])
    return cues


def assign_cue_speakers(
    cues: list[dict],
    turns: list[tuple[float, float, str]],
    language: str | None = "en",
) -> list[Segment]:
    """Label each caption cue with the diarized speaker overlapping it most.

    turns: [(start, end, speaker_label), ...] from diarization. Cues with no
    overlap fall back to the turn whose midpoint is nearest (captions can
    lead/lag speech slightly).
    """
    segments: list[Segment] = []
    for cue in cues:
        best_label, best_overlap = None, 0.0
        for start, end, label in turns:
            overlap = min(end, cue["end"]) - max(start, cue["start"])
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
        if best_label is None and turns:
            mid = (cue["start"] + cue["end"]) / 2
            best_label = min(
                turns, key=lambda t: abs((t[0] + t[1]) / 2 - mid))[2]
        segments.append(Segment(
            start=cue["start"],
            end=cue["end"],
            speaker=best_label or "SPEAKER_UNKNOWN",
            text=cue["text"],
            language=language,
        ))
    return segments

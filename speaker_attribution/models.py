"""Data models shared across pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReviewStatus(str, Enum):
    AUTO_OK = "auto_ok"          # attribution confident, mapping validated
    NEEDS_REVIEW = "needs_review"  # mismatch between diarization and shotlist
    FAILED = "failed"


@dataclass
class Segment:
    """One diarized transcript segment."""
    start: float          # seconds
    end: float            # seconds
    speaker: str          # anonymous label, e.g. "SPEAKER_00"
    text: str
    language: str | None = None


@dataclass
class SpeakerMapping:
    """Resolved identity for one anonymous speaker label."""
    label: str            # "SPEAKER_00"
    name: str             # "Mike Huckabee"
    role: str             # "U.S. Ambassador to Israel"
    confidence: str       # "high" | "medium" | "low"
    evidence: str         # short justification from the LLM


@dataclass
class AttributionResult:
    video_id: str
    segments: list[Segment]
    mappings: dict[str, SpeakerMapping]
    status: ReviewStatus
    warnings: list[str] = field(default_factory=list)

    def display_name(self, label: str) -> str:
        m = self.mappings.get(label)
        if m is None:
            return label
        return f"{m.name}, {m.role}" if m.role else m.name

"""Offline tests: validation logic and output writers (no GPU, no API calls)."""
from pathlib import Path

from speaker_attribution.attribute import _validate
from speaker_attribution.models import (
    AttributionResult, ReviewStatus, Segment, SpeakerMapping,
)
from speaker_attribution.outputs import write_json, write_labeled_txt, write_vtt

SEGMENTS = [
    Segment(0.0, 4.2, "SPEAKER_01", "The U.S. ambassador to Israel spoke to Reuters in Jerusalem.", "en"),
    Segment(4.5, 12.8, "SPEAKER_00", "I think the message is bigger than just even the Trump administration.", "en"),
    Segment(13.0, 19.9, "SPEAKER_00", "You don't take something that doesn't belong to you.", "en"),
    Segment(21.0, 27.5, "SPEAKER_02", "I am in contact with the embassy but the settlers are still near my property.", "en"),
]

MAPPINGS = {
    "SPEAKER_00": SpeakerMapping("SPEAKER_00", "Mike Huckabee", "U.S. Ambassador to Israel", "high", "matches soundbite 2"),
    "SPEAKER_01": SpeakerMapping("SPEAKER_01", "Alexander Cornwell", "Reuters correspondent", "high", "byline narration"),
    "SPEAKER_02": SpeakerMapping("SPEAKER_02", "Loui Ridi", "Palestinian American resident of Qusra", "high", "matches soundbite 6"),
}


def _result(mappings=MAPPINGS, warnings=None):
    warnings = warnings if warnings is not None else []
    status = _validate(SEGMENTS, mappings, warnings)
    return AttributionResult("786096", SEGMENTS, mappings, status, warnings)


def test_validate_all_high_confidence_is_auto_ok():
    assert _result().status is ReviewStatus.AUTO_OK


def test_validate_unmapped_label_needs_review():
    partial = {k: v for k, v in MAPPINGS.items() if k != "SPEAKER_02"}
    r = _result(mappings=partial)
    assert r.status is ReviewStatus.NEEDS_REVIEW
    assert any("Unmapped" in w for w in r.warnings)


def test_validate_low_confidence_needs_review():
    m = dict(MAPPINGS)
    m["SPEAKER_02"] = SpeakerMapping("SPEAKER_02", "Unidentified", "", "low", "no match")
    r = _result(mappings=m)
    assert r.status is ReviewStatus.NEEDS_REVIEW


def test_vtt_output(tmp_path: Path):
    out = tmp_path / "x.vtt"
    write_vtt(_result(), out)
    text = out.read_text()
    assert text.startswith("WEBVTT")
    assert "<v Mike Huckabee, U.S. Ambassador to Israel>" in text
    assert "00:00:04.500 --> 00:00:12.800" in text


def test_json_output(tmp_path: Path):
    import json
    out = tmp_path / "x.json"
    write_json(_result(), out)
    data = json.loads(out.read_text())
    assert data["video_id"] == "786096"
    assert data["segments"][1]["speaker"] == "Mike Huckabee, U.S. Ambassador to Israel"
    assert data["status"] == "auto_ok"


def test_labeled_txt_groups_consecutive_speaker(tmp_path: Path):
    out = tmp_path / "x.txt"
    write_labeled_txt(_result(), out)
    text = out.read_text()
    # Huckabee has two consecutive segments -> header should appear once
    assert text.count("Mike Huckabee, U.S. Ambassador to Israel:") == 1

"""Offline tests for editor speaker-name corrections."""
import pytest

from services.stream_api import overrides


def test_save_load_and_clear(tmp_path):
    overrides.save(tmp_path, "vid1", "SPEAKER_00", "  Dylan   O'Brien ")
    assert overrides.load(tmp_path, "vid1") == {"SPEAKER_00": "Dylan O'Brien"}
    overrides.save(tmp_path, "vid1", "SPEAKER_00", "")
    assert overrides.load(tmp_path, "vid1") == {}


def test_save_rejects_bad_label(tmp_path):
    with pytest.raises(ValueError):
        overrides.save(tmp_path, "vid1", "../etc", "x")


def test_apply_names_event_human_wins_and_original_untouched():
    ev = {"type": "names", "mapping": {
        "SPEAKER_00": {"name": "Unidentified", "role": "Unidentified", "confidence": "low"},
        "SPEAKER_01": {"name": "Pam Bondi", "role": "AG", "confidence": "high"}}}
    out = overrides.apply(ev, {"SPEAKER_00": "Sian Heder"})
    assert out["mapping"]["SPEAKER_00"] == {"name": "Sian Heder", "role": "",
                                           "confidence": "human"}
    assert out["mapping"]["SPEAKER_01"]["name"] == "Pam Bondi"
    assert ev["mapping"]["SPEAKER_00"]["name"] == "Unidentified"


def test_apply_face_tracks_names_linked_faces():
    ev = {"type": "face_tracks", "tracks": [
        {"id": 1, "speaker_label": "SPEAKER_00", "name": "Wrong Match",
         "gallery": 0.45, "name_tentative": True},
        {"id": 2, "speaker_label": None, "name": None}]}
    out = overrides.apply(ev, {"SPEAKER_00": "Sian Heder"})
    t1 = out["tracks"][0]
    assert t1["name"] == "Sian Heder" and "gallery" not in t1 and "name_tentative" not in t1
    assert out["tracks"][1]["name"] is None


def test_clear_removes_all_corrections(tmp_path):
    overrides.save(tmp_path, "vid1", "SPEAKER_01", "Some woman speaking")
    overrides.clear(tmp_path, "vid1")
    assert overrides.load(tmp_path, "vid1") == {}
    overrides.clear(tmp_path, "vid1")          # clearing twice is harmless

"""Offline tests for live-mode label continuity."""
from speaker_attribution.live import remap_labels


def test_remap_preserves_identity_across_shuffle():
    # cycle 1 established stable labels; cycle 2 shuffled them
    prev = [(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01")]
    new = [(0, 10, "SPEAKER_01"), (10, 20, "SPEAKER_00"),
           (20, 30, "SPEAKER_01")]   # same voice as 0-10s, new speech
    out = remap_labels(prev, new)
    assert out == [(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01"),
                   (20, 30, "SPEAKER_00")]


def test_remap_new_speaker_gets_fresh_stable_id():
    prev = [(0, 10, "SPEAKER_00")]
    new = [(0, 10, "SPEAKER_00"), (12, 20, "SPEAKER_01")]
    out = remap_labels(prev, new)
    assert (12, 20, "SPEAKER_01") in out          # fresh id, numbering continues
    assert (0, 10, "SPEAKER_00") in out


def test_remap_first_cycle_passthrough():
    new = [(0, 5, "SPEAKER_00")]
    assert remap_labels([], new) == new


def test_remap_weak_overlap_not_stolen():
    # a new label overlapping only a sliver of a stable speaker is NOT them
    prev = [(0, 30, "SPEAKER_00")]
    new = [(28, 40, "SPEAKER_05")]
    out = remap_labels(prev, new)
    assert out[0][2] != "SPEAKER_00"
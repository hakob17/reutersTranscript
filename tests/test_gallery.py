"""Offline tests for the face gallery's enroll/match logic."""
from speaker_attribution.gallery import MATCH_THRESHOLD, enroll, match


def _vec(seed, n=8):
    return [((seed * (i + 3)) % 7) - 3.0 for i in range(n)]


def test_enroll_and_match_same_identity():
    g = {}
    enroll(g, "Danielle Bensky", _vec(5), "epstein_778738 @39s")
    hit = match(g, _vec(5))
    assert hit and hit[0] == "Danielle Bensky" and hit[1] >= MATCH_THRESHOLD


def test_match_below_threshold_refuses():
    g = {}
    enroll(g, "A", [1.0, 0.0, 0.0, 0.0], "src")
    assert match(g, [0.0, 1.0, 0.0, 0.0]) is None      # orthogonal


def test_match_margin_refuses_ambiguity():
    g = {}
    enroll(g, "A", [1.0, 0.05, 0.0], "src")
    enroll(g, "B", [1.0, -0.05, 0.0], "src")
    assert match(g, [1.0, 0.0, 0.0]) is None           # both ~equally close


def test_enroll_caps_embeddings():
    g = {}
    for i in range(15):
        enroll(g, "A", _vec(i + 1), f"src{i}")
    assert len(g["A"]["embs"]) == 10 and len(g["A"]["sources"]) == 10


def test_enroll_idempotent_per_source():
    g = {}
    assert enroll(g, "A", _vec(1), "wikidata:Q1") is True
    assert enroll(g, "A", _vec(1), "wikidata:Q1") is False   # re-seed: no-op
    assert len(g["A"]["embs"]) == 1


def test_confidence_tiers():
    from speaker_attribution.gallery import HIGH_CONFIDENCE, is_confident
    assert is_confident(0.82) and is_confident(HIGH_CONFIDENCE)
    assert not is_confident(0.494)            # correct-but-weak -> hedged

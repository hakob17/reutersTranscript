"""Offline tests for Wikidata candidate selection (no network)."""
from speaker_attribution.wikidata import candidate_people, normalize_name


def test_candidate_people_keeps_person_like_entries():
    notes = ("disability, TIFF, Toronto International Film Festival, Being Heumann, "
             "activist, American, Sian Heder, Judy Heumann, Dylan O’Brien")
    got = candidate_people(notes)
    assert "Sian Heder" in got and "Judy Heumann" in got and "Dylan O’Brien" in got
    # single words and lowercase topics are never looked up
    for topic in ("disability", "TIFF", "activist", "American"):
        assert topic not in got


def test_candidate_people_allows_name_particles_and_rejects_digits():
    got = candidate_people("Ursula von der Leyen, G20 Summit, COP 29, Mohamed bin Zayed")
    assert "Ursula von der Leyen" in got and "Mohamed bin Zayed" in got
    assert "G20 Summit" not in got and "COP 29" not in got


def test_normalize_name_ignores_case_accents_apostrophes():
    assert normalize_name("Dylan O’Brien") == normalize_name("dylan o'brien")
    assert normalize_name("António  Guterres") == normalize_name("Antonio Guterres")
    assert normalize_name("Being Heumann") != normalize_name("Judy Heumann")

"""Offline test for the measured-cost accumulator."""
from types import SimpleNamespace

from speaker_attribution import costs


def test_record_and_summary():
    costs.reset()
    costs.record("attribution", "claude-sonnet-4-6",
                 SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000))
    costs.record("scenes", "claude-haiku-4-5",
                 SimpleNamespace(input_tokens=500_000, output_tokens=0))
    s = costs.summary()
    assert s["stages"]["attribution"] == 4.5     # $3 in + $1.5 out
    assert s["stages"]["scenes"] == 0.5
    assert s["total"] == 5.0
    costs.reset()
    assert costs.summary()["total"] == 0

"""Offline tests for the caption-first path (no network, no GPU)."""
from speaker_attribution.captions import assign_cue_speakers, parse_vtt

SAMPLE_VTT = """WEBVTT
X-TIMESTAMP-MAP=MPEGTS:186000,LOCAL:00:00:00.000

﻿1
00:00:02.952 --> 00:00:05.755
I’m at the Reuters Future of Insurance event in Chicago,

2
00:00:05.755 --> 00:00:09.542
and I'm now joined by <b>Juan García</b>,
who's the co-founder and CEO of Tuio,

3
00:00:10.960 --> 00:00:11.695
Happy to be here.

NOTE this block has no timestamp and must be ignored

4
00:01:02.000 --> 01:00:03.500
An hour-long cue with an hours field.
"""


def test_parse_vtt_cues():
    cues = parse_vtt(SAMPLE_VTT)
    assert len(cues) == 4
    assert cues[0]["start"] == 2.952
    assert cues[0]["text"].startswith("I’m at the Reuters")
    # multi-line cue joined, inline tags stripped
    assert "Juan García," in cues[1]["text"]
    assert "<b>" not in cues[1]["text"]
    # hours field parsed
    assert cues[3]["end"] == 3600 + 3.5


def test_assign_cue_speakers_overlap_and_fallback():
    cues = [
        {"start": 0.0, "end": 4.0, "text": "hello from the host"},
        {"start": 4.0, "end": 8.0, "text": "guest reply"},
        {"start": 20.0, "end": 21.0, "text": "cue in a diarization gap"},
    ]
    turns = [(0.0, 4.5, "SPEAKER_00"), (4.5, 9.0, "SPEAKER_01"),
             (22.0, 30.0, "SPEAKER_01")]
    segs = assign_cue_speakers(cues, turns)
    assert [s.speaker for s in segs] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    # cue text and timing survive untouched
    assert segs[0].text == "hello from the host"
    assert segs[2].start == 20.0

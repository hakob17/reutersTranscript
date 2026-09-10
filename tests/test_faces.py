"""Offline tests for the face tracker and conservative name binding."""
from speaker_attribution.faces import (IouTracker, bind_names_to_tracks,
                                       bind_tracks_to_turns)


def _det(x, y=0.3, w=0.2, h=0.3):
    return {"x": x, "y": y, "w": w, "h": h}


def test_tracker_associates_and_splits():
    tr = IouTracker()
    # one face drifting right for 2s, then a cut: face elsewhere
    for i in range(9):                    # t = 0.0 .. 2.0
        tr.update(i * 0.25, [_det(0.1 + i * 0.01)])
    for i in range(9, 18):                # after "cut", far away -> new track
        tr.update(i * 0.25, [_det(0.7)])
    tracks = tr.finish()
    assert len(tracks) == 2
    assert tracks[0]["t0"] == 0.0
    assert tracks[1]["boxes"][0]["x"] == 0.7


def test_tracker_drops_blips():
    tr = IouTracker()
    tr.update(0.0, [_det(0.1)])
    tr.update(0.25, [_det(0.1)])          # only 0.25s long -> dropped
    for i in range(10):
        tr.update(3 + i * 0.25, [_det(0.5)])
    tracks = tr.finish()
    assert len(tracks) == 1 and tracks[0]["boxes"][0]["x"] == 0.5


def _track(tid, t0, t1, x, name=None):
    boxes = [{"t": t, "x": x, "y": 0.2, "w": 0.25, "h": 0.35}
             for t in [t0 + i * 0.25 for i in range(int((t1 - t0) / 0.25) + 1)]]
    return {"id": tid, "name": name, "t0": t0, "t1": t1, "boxes": boxes}


def test_binding_single_face_binds():
    tracks = [_track(1, 0.0, 5.0, 0.4)]
    warnings = bind_names_to_tracks(tracks, [{"name": "Haley Robson", "time": 2.0}])
    assert tracks[0]["name"] == "Haley Robson"
    assert not warnings


def test_binding_two_faces_refuses():
    tracks = [_track(1, 0.0, 5.0, 0.2), _track(2, 0.0, 5.0, 0.7)]
    warnings = bind_names_to_tracks(tracks, [{"name": "Michael Loney", "time": 2.0}])
    assert tracks[0]["name"] is None and tracks[1]["name"] is None
    assert warnings and "ambiguous" in warnings[0]


def test_turn_binding_solo_visibility():
    # track 1 alone on screen during SPEAKER_00's turns; track 2 during 01's
    tracks = [_track(1, 0.0, 10.0, 0.4), _track(2, 12.0, 20.0, 0.5)]
    turns = [(1.0, 8.0, "SPEAKER_00"), (13.0, 19.0, "SPEAKER_01")]
    bind_tracks_to_turns(tracks, turns)
    assert tracks[0]["speaker_label"] == "SPEAKER_00"
    assert tracks[1]["speaker_label"] == "SPEAKER_01"


def test_turn_binding_two_shot_binds_nothing():
    # both faces visible for the whole turn -> no solo vote -> no binding
    tracks = [_track(1, 0.0, 10.0, 0.2), _track(2, 0.0, 10.0, 0.7)]
    bind_tracks_to_turns(tracks, [(1.0, 9.0, "SPEAKER_00")])
    assert tracks[0]["speaker_label"] is None
    assert tracks[1]["speaker_label"] is None


def test_turn_binding_needs_margin():
    # one track solo during turns of two DIFFERENT speakers equally -> ambiguous
    tracks = [_track(1, 0.0, 20.0, 0.4)]
    turns = [(1.0, 8.0, "SPEAKER_00"), (10.0, 18.0, "SPEAKER_01")]
    bind_tracks_to_turns(tracks, turns)
    assert tracks[0]["speaker_label"] is None


def test_binding_conflict_unbinds():
    tracks = [_track(1, 0.0, 10.0, 0.4)]
    warnings = bind_names_to_tracks(tracks, [
        {"name": "Person A", "time": 2.0},
        {"name": "Person B", "time": 8.0},
    ])
    assert tracks[0]["name"] is None
    assert any("conflicting" in w for w in warnings)

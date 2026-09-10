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


def _track_with_mar(tid, t0, t1, x, mars):
    tr = _track(tid, t0, t1, x)
    for i, b in enumerate(tr["boxes"]):
        b["mar"] = mars[i % len(mars)]
    return tr


def test_turn_binding_lip_motion_resolves_two_shot():
    # both faces visible; track 1's mouth oscillates, track 2's is still
    talking = _track_with_mar(1, 0.0, 10.0, 0.2, [0.1, 0.4, 0.15, 0.45])
    still = _track_with_mar(2, 0.0, 10.0, 0.7, [0.12, 0.12, 0.13, 0.12])
    bind_tracks_to_turns([talking, still], [(1.0, 9.0, "SPEAKER_01")])
    assert talking["speaker_label"] == "SPEAKER_01"
    assert still["speaker_label"] is None


def test_turn_binding_lip_motion_needs_margin():
    # both mouths move similarly -> ambiguous -> nothing binds
    a = _track_with_mar(1, 0.0, 10.0, 0.2, [0.1, 0.3, 0.1, 0.3])
    b = _track_with_mar(2, 0.0, 10.0, 0.7, [0.2, 0.4, 0.2, 0.4])
    bind_tracks_to_turns([a, b], [(1.0, 9.0, "SPEAKER_00")])
    assert a["speaker_label"] is None and b["speaker_label"] is None


def test_turn_binding_lip_motion_requires_all_measured():
    # one face has no MAR data -> cannot compare fairly -> nothing binds
    talking = _track_with_mar(1, 0.0, 10.0, 0.2, [0.1, 0.4, 0.15, 0.45])
    unmeasured = _track(2, 0.0, 10.0, 0.7)
    bind_tracks_to_turns([talking, unmeasured], [(1.0, 9.0, "SPEAKER_00")])
    assert talking["speaker_label"] is None


def test_turn_binding_excluded_label_never_votes():
    # narrator/voice-over labels are off-camera; solo B-roll face binds nothing
    tracks = [_track_with_mar(1, 0.0, 10.0, 0.4, [0.1, 0.4, 0.15, 0.45])]
    bind_tracks_to_turns(tracks, [(1.0, 9.0, "SPEAKER_00")],
                         exclude_labels={"SPEAKER_00"})
    assert tracks[0]["speaker_label"] is None


def test_turn_binding_solo_still_face_fails_lip_gate():
    # VO over a solo, silent face: lips still -> no vote (landmarks present)
    still = _track_with_mar(1, 0.0, 10.0, 0.4, [0.12, 0.12, 0.13, 0.12])
    bind_tracks_to_turns([still], [(1.0, 9.0, "SPEAKER_00")])
    assert still["speaker_label"] is None


def test_turn_binding_solo_without_landmarks_keeps_legacy():
    # no MAR anywhere (mediapipe unavailable) -> legacy visibility rule
    tracks = [_track(1, 0.0, 10.0, 0.4)]
    bind_tracks_to_turns(tracks, [(1.0, 9.0, "SPEAKER_00")])
    assert tracks[0]["speaker_label"] == "SPEAKER_00"


def test_turn_binding_coverage_strips_sliver_matches():
    # a 5s "match" against 95s of that voice's speech = bystander, stripped
    tracks = [_track_with_mar(1, 65.0, 70.0, 0.4, [0.1, 0.4, 0.15, 0.45])]
    turns = [(0.0, 60.0, "SPEAKER_00"), (64.0, 99.0, "SPEAKER_00")]
    bind_tracks_to_turns(tracks, turns)
    assert tracks[0]["speaker_label"] is None


def test_turn_binding_coverage_keeps_consistent_speaker():
    # face on screen for most of the voice's speech -> links survive
    tracks = [_track_with_mar(1, 0.0, 8.0, 0.4, [0.1, 0.4, 0.15, 0.45])]
    turns = [(1.0, 7.0, "SPEAKER_00")]
    bind_tracks_to_turns(tracks, turns)
    assert tracks[0]["speaker_label"] == "SPEAKER_00"


def test_tracker_refuses_identity_change_at_same_position():
    # dissolve: a different face appears where the old one was
    tr = IouTracker()
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    for i in range(6):
        tr.update(i * 0.25, [{**_det(0.3), "emb": a}])
    for i in range(6, 12):
        tr.update(i * 0.25, [{**_det(0.3), "emb": b}])
    assert len(tr.finish()) == 2


def test_tracker_cut_closes_tracks():
    tr = IouTracker()
    for i in range(6):
        tr.update(i * 0.25, [_det(0.3)])
    tr.cut()
    for i in range(6, 12):
        tr.update(i * 0.25, [_det(0.3)])
    assert len(tr.finish()) == 2


def test_track_consistency_net_drops_drifting_embedding():
    # slow drift a -> b passes each per-step check, but halves disagree
    tr = IouTracker()
    n = 12
    for i in range(n):
        s = i / (n - 1)
        tr.update(i * 0.25, [{**_det(0.3), "emb": [1 - s, s]}])
    tracks = tr.finish()
    assert len(tracks) == 1 and "emb" not in tracks[0]


def test_yaw_ratio():
    from speaker_attribution.faces import _yaw
    frontal = [0, 0, 0, 0, 40, 50, 60, 50, 50, 60]   # nose centred
    profile = [0, 0, 0, 0, 40, 50, 50, 50, 70, 60]   # nose far outside
    assert _yaw(frontal) < 0.1 and _yaw(profile) > 1.0


def test_binding_conflict_unbinds():
    tracks = [_track(1, 0.0, 10.0, 0.4)]
    warnings = bind_names_to_tracks(tracks, [
        {"name": "Person A", "time": 2.0},
        {"name": "Person B", "time": 8.0},
    ])
    assert tracks[0]["name"] is None
    assert any("conflicting" in w for w in warnings)

"""Phase-3 slice: face tracks from the video + conservative name binding.

Detection uses OpenCV's YuNet (a small ONNX model bundled-friendly and fast
on CPU) with a simple IoU tracker — enough for news framing, where shots are
short and faces large. Production upgrades per docs/DESIGN.md are SCRFD +
ByteTrack; the interfaces here are shaped so that swap is local.

Name binding follows the design doc's never-guess rule: a chyron sighting
binds a name to a face track ONLY when exactly one plausible face is on
screen at sighting time. Dual-name straps and crowd shots bind nothing.

cv2 is imported lazily (same contract as chyron.py) and the tracker/binding
logic is pure Python so tests run offline.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MODEL_DIR = Path(__file__).resolve().parent.parent / ".models"

SAMPLE_FPS = 4            # detection sampling rate
IOU_MATCH = 0.30          # detection -> track association threshold
TRACK_GAP_S = 0.75        # close a track after this long unmatched
MIN_TRACK_S = 1.0         # drop blips shorter than this
MIN_BIND_AREA = 0.02      # face must be >2% of frame to be a naming candidate


def _yunet_path() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


# --------------------------------------------------------------------------
# pure-python tracker (offline-testable)

def _iou(a: dict, b: dict) -> float:
    x0 = max(a["x"], b["x"]); y0 = max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Greedy IoU association; good enough between news shot cuts."""

    def __init__(self) -> None:
        self.next_id = 1
        self.open: list[dict] = []      # {id, boxes:[{t,x,y,w,h}]}
        self.closed: list[dict] = []

    def update(self, t: float, detections: list[dict]) -> None:
        unmatched = list(detections)
        for track in self.open:
            last = track["boxes"][-1]
            best, best_iou = None, IOU_MATCH
            for d in unmatched:
                iou = _iou(last, d)
                if iou > best_iou:
                    best, best_iou = d, iou
            if best is not None:
                track["boxes"].append({"t": t, **best})
                unmatched.remove(best)
        for d in unmatched:
            self.open.append({"id": self.next_id, "boxes": [{"t": t, **d}]})
            self.next_id += 1
        # close stale tracks (a shot cut ends everything at once)
        still_open = []
        for track in self.open:
            if t - track["boxes"][-1]["t"] > TRACK_GAP_S:
                self.closed.append(track)
            else:
                still_open.append(track)
        self.open = still_open

    def finish(self) -> list[dict]:
        tracks = self.closed + self.open
        out = []
        for tr in tracks:
            boxes = tr["boxes"]
            if boxes[-1]["t"] - boxes[0]["t"] >= MIN_TRACK_S:
                out.append({"id": tr["id"], "name": None,
                            "t0": round(boxes[0]["t"], 3),
                            "t1": round(boxes[-1]["t"], 3),
                            "boxes": [{k: round(float(v), 4) for k, v in b.items()}
                                      for b in boxes]})
        out.sort(key=lambda tr: tr["t0"])
        return out


def detect_face_tracks(video, sample_fps: float = SAMPLE_FPS) -> list[dict]:
    """Decode the video/stream, detect faces on sampled frames, return
    tracks with normalized (0-1) boxes: [{id, name, t0, t1, boxes}]."""
    import cv2

    detector = cv2.FaceDetectorYN_create(str(_yunet_path()), "", (320, 320),
                                         score_threshold=0.6)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    tracker = IouTracker()
    idx = 0
    size_set = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                scale = 640 / w
                small = cv2.resize(frame, (640, int(h * scale)))
                if size_set != small.shape[:2]:
                    detector.setInputSize((small.shape[1], small.shape[0]))
                    size_set = small.shape[:2]
                _, faces = detector.detect(small)
                dets = []
                if faces is not None:
                    sh, sw = small.shape[:2]
                    for f in faces:
                        x, y, bw, bh = (float(v) for v in f[:4])  # numpy -> JSON-safe
                        dets.append({"x": max(0.0, x / sw), "y": max(0.0, y / sh),
                                     "w": bw / sw, "h": bh / sh})
                tracker.update(idx / fps, dets)
        idx += 1
    cap.release()
    return tracker.finish()


# --------------------------------------------------------------------------
# conservative name binding (offline-testable)

def _track_box_at(track: dict, t: float, slack: float = 0.6) -> dict | None:
    best, best_dt = None, slack
    for b in track["boxes"]:
        dt = abs(b["t"] - t)
        if dt < best_dt:
            best, best_dt = b, dt
    return best


def bind_tracks_to_turns(tracks: list[dict],
                         turns: list[tuple[float, float, str]]) -> None:
    """ASD-lite: link face tracks to diarization labels by solo visibility.

    A speaker turn votes for a track only when that track is the SINGLE
    prominent face on screen for most of the turn (news editing favors solo
    close-ups of the person talking). A track takes a label only with a
    clear vote margin — ambiguity binds nothing. Sets track["speaker_label"].
    Production replacement: Light-ASD (docs/DESIGN.md §3).
    """
    votes: dict[int, dict[str, int]] = {}
    for t0, t1, label in turns:
        dur = t1 - t0
        if dur < 1.0:
            continue
        visible = []
        for tr in tracks:
            on = sum(1 for b in tr["boxes"]
                     if t0 <= b["t"] <= t1 and b["w"] * b["h"] >= MIN_BIND_AREA)
            if on * (1.0 / SAMPLE_FPS) >= 0.6 * dur:
                visible.append(tr)
        if len(visible) == 1:
            tid = visible[0]["id"]
            votes.setdefault(tid, {})
            votes[tid][label] = votes[tid].get(label, 0) + 1

    for tr in tracks:
        tr.setdefault("speaker_label", None)
        tally = votes.get(tr["id"])
        if not tally:
            continue
        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if top >= 1 and top >= 2 * second:
            tr["speaker_label"] = top_label


def bind_names_to_tracks(tracks: list[dict], sightings: list[dict]) -> list[str]:
    """Bind chyron-sighted names to face tracks in place. A sighting binds
    ONLY when exactly one sufficiently large face is on screen at its time —
    anything ambiguous binds nothing (design doc §4.3). Returns warnings."""
    warnings: list[str] = []
    for s in sightings:
        name = s.get("name")
        t = s.get("time")
        if not name or t is None:
            continue
        candidates = []
        for tr in tracks:
            box = _track_box_at(tr, t)
            if box and box["w"] * box["h"] >= MIN_BIND_AREA:
                candidates.append(tr)
        if len(candidates) != 1:
            if len(candidates) > 1:
                warnings.append(
                    f"'{name}' at {t:.1f}s: {len(candidates)} faces on screen "
                    "— not binding (ambiguous)")
            continue
        track = candidates[0]
        if track["name"] and track["name"] != name:
            warnings.append(
                f"track {track['id']}: conflicting names "
                f"'{track['name']}' vs '{name}' — unbinding, needs review")
            track["name"] = None
        else:
            track["name"] = name
    return warnings

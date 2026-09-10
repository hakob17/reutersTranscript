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

FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/"
                       "face_landmarker/face_landmarker/float16/1/"
                       "face_landmarker.task")

SAMPLE_FPS = 6            # detection sampling rate (lip motion needs >=6)
IOU_MATCH = 0.30          # detection -> track association threshold
TRACK_GAP_S = 0.75        # close a track after this long unmatched
MIN_TRACK_S = 1.0         # drop blips shorter than this
MIN_BIND_AREA = 0.02      # face must be >2% of frame to take a chyron NAME
MIN_VIS_AREA = 0.006      # smaller faces (wide shots) still count as visible
FACE_RENDITION_H = 480    # decode this rendition for detection, not the
                          # master's first (= lowest) variant

# lip-motion ASD (mouth-aspect-ratio variance while a speaker turn is live)
LIP_MIN_SAMPLES = 5       # need this many MAR samples in a turn to score
LIP_MARGIN = 1.6          # winner's lip energy must beat runner-up by this
LIP_FLOOR = 0.02          # and exceed this absolute energy (still faces ~0.005)
ASD_MAX_FACES = 4         # score lip motion only in 2..N-face shots
COVERAGE_MIN = 0.15       # a label's linked faces must cover this fraction of
                          # its speech time, else the links are stripped (a
                          # 5s "match" against 95s of voice-over is a bystander)


def best_rendition_for_faces(url: str) -> str:
    """cv2.VideoCapture on an HLS master picks the FIRST (lowest) variant —
    320x180 on Reuters masters, where wide-shot faces are undetectable.
    Pick the variant closest to FACE_RENDITION_H instead. Non-master inputs
    (local files, direct renditions) pass through unchanged."""
    if not str(url).startswith(("http://", "https://")):
        return url
    try:
        import re
        import urllib.parse
        from .captions import _http_get
        playlist = _http_get(str(url))
        if "#EXT-X-STREAM-INF" not in playlist:
            return url
        best, best_d = None, None
        lines = playlist.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("#EXT-X-STREAM-INF"):
                m = re.search(r"RESOLUTION=\d+x(\d+)", line)
                if m and i + 1 < len(lines) and lines[i + 1].strip():
                    d = abs(int(m.group(1)) - FACE_RENDITION_H)
                    if best_d is None or d < best_d:
                        best, best_d = lines[i + 1].strip(), d
        return urllib.parse.urljoin(str(url), best) if best else url
    except Exception:
        return url


def _yunet_path() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


def _make_landmarker():
    """MediaPipe FaceLandmarker for lip landmarks; None if unavailable."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / "face_landmarker.task"
        if not path.exists():
            urllib.request.urlretrieve(FACE_LANDMARKER_URL, path)
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            num_faces=1, running_mode=vision.RunningMode.IMAGE)
        return mp, vision.FaceLandmarker.create_from_options(opts)
    except Exception:
        return None, None


def _mouth_aspect_ratio(mp, landmarker, face_bgr) -> float | None:
    """MAR = inner-lip opening / mouth width, from FaceMesh landmarks.
    Talking faces oscillate; listeners stay near-constant."""
    import cv2
    try:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=rgb))
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        up, lo, left, right = lm[13], lm[14], lm[61], lm[291]
        width = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
        opening = ((up.x - lo.x) ** 2 + (up.y - lo.y) ** 2) ** 0.5
        return opening / width if width > 1e-6 else None
    except Exception:
        return None


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
            if boxes[-1]["t"] - boxes[0]["t"] < MIN_TRACK_S:
                continue
            embs = [b.pop("emb") for b in boxes if "emb" in b]
            track = {"id": tr["id"], "name": None,
                     "t0": round(boxes[0]["t"], 3),
                     "t1": round(boxes[-1]["t"], 3),
                     "boxes": [{k: round(float(v), 4) for k, v in b.items()
                                if v is not None}
                               for b in boxes]}
            if embs:   # mean embedding; internal only — stripped before emit
                n = len(embs)
                track["emb"] = [round(sum(e[i] for e in embs) / n, 5)
                                for i in range(len(embs[0]))]
            out.append(track)
        out.sort(key=lambda tr: tr["t0"])
        return out


SHOT_CUT_DIFF = 0.45      # HSV-histogram distance that counts as a hard cut
SHOT_MIN_S = 1.0          # merge shorter "shots" into their predecessor


def _ahash(gray_small) -> int:
    """64-bit average hash for near-duplicate shot detection."""
    import cv2
    tiny = cv2.resize(gray_small, (8, 8))
    mean = tiny.mean()
    bits = 0
    for v in tiny.flatten():
        bits = (bits << 1) | (1 if v > mean else 0)
    return int(bits)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def detect_face_tracks(video, sample_fps: float = SAMPLE_FPS,
                       collect_shots: bool = False):
    """Decode the video/stream, detect faces on sampled frames, return
    tracks with normalized (0-1) boxes: [{id, name, t0, t1, boxes}].
    With collect_shots=True also returns shots from the same decode:
    [{t0, t1, key_t, jpeg, hash}] — one keyframe per hard cut."""
    import cv2

    detector = cv2.FaceDetectorYN_create(str(_yunet_path()), "", (320, 320),
                                         score_threshold=0.6)
    mp, landmarker = _make_landmarker()
    from .gallery import embed, make_recognizer
    recognizer = make_recognizer()
    cap = cv2.VideoCapture(best_rendition_for_faces(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    tracker = IouTracker()
    shots: list[dict] = []
    prev_hist = None
    cur_shot: dict | None = None

    def close_shot(t_end: float) -> None:
        nonlocal cur_shot
        if cur_shot is None:
            return
        cur_shot["t1"] = t_end
        if shots and cur_shot["t1"] - cur_shot["t0"] < SHOT_MIN_S:
            shots[-1]["t1"] = cur_shot["t1"]   # too short: merge back
        else:
            shots.append(cur_shot)
        cur_shot = None

    idx = 0
    size_set = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = idx / fps
                h, w = frame.shape[:2]
                scale = 640 / w
                small = cv2.resize(frame, (640, int(h * scale)))
                if size_set != small.shape[:2]:
                    detector.setInputSize((small.shape[1], small.shape[0]))
                    size_set = small.shape[:2]

                if collect_shots:
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32],
                                        [0, 180, 0, 256])
                    cv2.normalize(hist, hist)
                    is_cut = (prev_hist is not None and
                              cv2.compareHist(prev_hist, hist,
                                              cv2.HISTCMP_CORREL) < 1 - SHOT_CUT_DIFF)
                    prev_hist = hist
                    if cur_shot is None or is_cut:
                        close_shot(t)
                        cur_shot = {"t0": t, "t1": t, "key_t": None,
                                    "jpeg": b"", "hash": 0}
                    # keyframe: first stable frame >=0.8s into the shot
                    if cur_shot["key_t"] is None and t - cur_shot["t0"] >= 0.8:
                        okj, jpg = cv2.imencode(
                            ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if okj:
                            cur_shot.update(key_t=t, jpeg=jpg.tobytes(),
                                            hash=_ahash(gray))

                _, faces = detector.detect(small)
                dets = []
                if faces is not None:
                    sh, sw = small.shape[:2]
                    for f in faces:
                        x, y, bw, bh = (float(v) for v in f[:4])  # numpy -> JSON-safe
                        d = {"x": max(0.0, x / sw), "y": max(0.0, y / sh),
                             "w": bw / sw, "h": bh / sh}
                        # SFace embedding for gallery identification —
                        # only prominent faces, bounded shots
                        if (recognizer is not None and len(faces) <= ASD_MAX_FACES
                                and d["w"] * d["h"] >= MIN_BIND_AREA):
                            # skip dark crops (fades, night footage): low-light
                            # embeddings collapse together and produce
                            # confident-looking false matches
                            crop = small[max(0, int(y)):int(y + bh),
                                         max(0, int(x)):int(x + bw)]
                            lit = crop.size > 0 and float(cv2.cvtColor(
                                crop, cv2.COLOR_BGR2GRAY).mean()) >= 50
                            e = embed(recognizer, small, f) if lit else None
                            if e is not None:
                                d["emb"] = e
                        dets.append(d)
                    # lip landmarks for ASD — only where competition is
                    # possible and cost stays bounded
                    if landmarker is not None and 1 <= len(dets) <= ASD_MAX_FACES:
                        for d in dets:
                            mx = d["w"] * 0.25
                            x0 = int(max(0.0, d["x"] - mx) * sw)
                            x1 = int(min(1.0, d["x"] + d["w"] + mx) * sw)
                            y0 = int(max(0.0, d["y"] - mx) * sh)
                            y1 = int(min(1.0, d["y"] + d["h"] + mx) * sh)
                            if x1 - x0 > 24 and y1 - y0 > 24:
                                mar = _mouth_aspect_ratio(
                                    mp, landmarker, small[y0:y1, x0:x1])
                                if mar is not None:
                                    d["mar"] = mar
                tracker.update(idx / fps, dets)
        idx += 1
    cap.release()
    if collect_shots:
        close_shot(idx / fps)
        return tracker.finish(), [s for s in shots if s["jpeg"]]
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


def _lip_energy(track: dict, t0: float, t1: float) -> float | None:
    """Mean |ΔMAR| between consecutive samples inside [t0, t1] — high while
    talking, near zero for a listening face. None if too few samples."""
    mars = [(b["t"], b["mar"]) for b in track["boxes"]
            if t0 <= b["t"] <= t1 and "mar" in b]
    if len(mars) < LIP_MIN_SAMPLES:
        return None
    diffs = [abs(mars[i + 1][1] - mars[i][1]) for i in range(len(mars) - 1)]
    return sum(diffs) / len(diffs)


def bind_tracks_to_turns(tracks: list[dict],
                         turns: list[tuple[float, float, str]],
                         exclude_labels: set[str] | None = None) -> None:
    """ASD: link face tracks to diarization labels.

    Solo shots vote by visibility (the single prominent face during a turn
    is the speaker) — gated by lip motion when landmarks are available, so a
    voice-over playing across a silent B-roll face binds nothing. Multi-face
    shots (2..ASD_MAX_FACES) vote by lip motion: the face whose
    mouth-aspect-ratio oscillates while the turn is live wins, but only with
    a clear margin over the runner-up — ambiguity binds nothing. A track
    takes a label only with a 2x vote margin. `exclude_labels` (e.g.
    narrator/voice-over labels, who are off-camera by definition) never
    vote. Sets track["speaker_label"]. (Light-ASD remains the upgrade path.)
    """
    exclude_labels = exclude_labels or set()
    # legacy mode when no landmarks exist at all (mediapipe unavailable)
    has_mar = any("mar" in b for tr in tracks for b in tr["boxes"][:80])

    votes: dict[int, dict[str, int]] = {}
    for t0, t1, label in turns:
        dur = t1 - t0
        if dur < 1.0 or label in exclude_labels:
            continue
        visible = []
        for tr in tracks:
            on = sum(1 for b in tr["boxes"]
                     if t0 <= b["t"] <= t1 and b["w"] * b["h"] >= MIN_VIS_AREA)
            if on * (1.0 / SAMPLE_FPS) >= 0.6 * dur:
                visible.append(tr)

        winner = None
        if len(visible) == 1:
            if has_mar:
                # even solo, the face must actually be talking — a VO line
                # over a still face is not this face speaking
                e = _lip_energy(visible[0], t0, t1)
                if e is not None and e >= LIP_FLOOR:
                    winner = visible[0]
            else:
                winner = visible[0]
        elif 2 <= len(visible) <= ASD_MAX_FACES:
            scored = [(tr, _lip_energy(tr, t0, t1)) for tr in visible]
            scored = [(tr, e) for tr, e in scored if e is not None]
            if len(scored) == len(visible):        # every face measurable
                scored.sort(key=lambda te: te[1], reverse=True)
                top_tr, top = scored[0]
                second = scored[1][1]
                if top >= LIP_FLOOR and top >= LIP_MARGIN * max(second, 1e-6):
                    winner = top_tr
        if winner is not None:
            tid = winner["id"]
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

    # coverage consistency: an on-camera voice shows its face across much of
    # its speech; a sliver of "match" against long speech is a false positive
    # (masked/occluded faces make lip noise). Strip inconsistent labels.
    speech_time: dict[str, float] = {}
    for t0, t1, label in turns:
        speech_time[label] = speech_time.get(label, 0.0) + (t1 - t0)
    linked_time: dict[str, float] = {}
    for tr in tracks:
        if tr["speaker_label"]:
            linked_time[tr["speaker_label"]] = (
                linked_time.get(tr["speaker_label"], 0.0)
                + (tr["t1"] - tr["t0"]))
    for tr in tracks:
        label = tr["speaker_label"]
        if label and speech_time.get(label, 0.0) > 0 and \
                linked_time.get(label, 0.0) < COVERAGE_MIN * speech_time[label]:
            tr["speaker_label"] = None


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

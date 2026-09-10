"""Stage 3b: name speakers from on-screen lower-third graphics (chyrons).

News packages burn the speaker's name/title into the frame while they talk.
This stage samples frames during each diarized speaker's segments, uses
OpenCV locally to find the few frames where a lower-third graphic is
actually present, and sends ONLY those crops to Claude to read. Local
detection keeps vision costs flat regardless of video length.

Usage:
  python -m speaker_attribution.chyron VIDEO.mp4 out/VIDEO.json --out out
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import anthropic

# cv2/numpy are imported lazily inside the detection functions so the
# cloud attribution Lambda can import the reading/merging half of this
# module without bundling OpenCV.

from .attribute import _validate
from .models import AttributionResult, ReviewStatus, Segment, SpeakerMapping
from .outputs import write_json, write_labeled_txt, write_vtt

MODEL = "claude-opus-5"
SAMPLE_INTERVAL_S = 0.5     # how often to test frames inside speech segments
BAND_TOP, BAND_BOTTOM = 0.60, 0.97  # lower-third region (fraction of height)
MIN_TEXT_SCORE = 0.006      # text-like area fraction to count as a chyron
EVENT_GAP_S = 1.0           # samples closer than this merge into one event
MAX_FRAMES_TO_LLM = 16
PER_LABEL_CAP = 3           # frame budget per speaker — keeps crop selection diverse
STRAP_MIN_AREA, STRAP_MAX_AREA = 0.006, 0.08   # strap box, fraction of frame
STRAP_ASPECT = (2.0, 8.0)   # name straps are wide, short boxes

SYSTEM_PROMPT = """You read broadcast news frame crops and report the
lower-third graphic (chyron/name strap) if one is visible.

Each image is the LOWER PORTION of a frame — or a tight crop around a
detected name-strap box — labeled with the diarization speaker label that is
talking at that moment. A lower-third names the person currently speaking on
camera. Report a sighting for EVERY image that shows a legible name strap;
do not skip any.

Rules:
- Report ONLY text actually visible in a lower-third graphic. Never guess or
  infer names from context. Illegible or absent chyron -> omit that frame.
- A chyron names the on-camera speaker; ignore location/date slates, tickers,
  channel bugs and courtesy lines.
- If different frames for the same speaker label show DIFFERENT names, report
  both and flag it in warnings (diarization may have merged two people).
- Respond with ONLY a JSON object, no markdown fences, shaped as:
{
  "sightings": [
    {
      "label": "SPEAKER_05",
      "time": 67.2,
      "name": "Marina Lacerda",
      "title": "Epstein accuser",
      "legible": "high"
    }
  ],
  "warnings": ["optional list of issues"]
}"""


def _text_score(band_bgr) -> float:
    """Fraction of the band covered by text-like shapes (wide clusters of
    high-frequency strokes). Cheap and works on stylized broadcast straps."""
    import cv2

    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # connect letters into word/line blobs
    connected = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_band, w_band = gray.shape
    area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 3 * h and w >= w_band * 0.05 and 8 <= h <= h_band * 0.4:
            area += w * h
    return area / float(h_band * w_band)


def _find_strap_box(frame_bgr):
    """Find a broadcast name strap: a bright, low-saturation, box-shaped
    region with dark text inside, in the lower part of the frame.

    Complements _text_score, which fails on busy backgrounds (foliage swamps
    the gradient/Otsu step and everything merges into one rejected blob —
    observed: a clear 'Haley Robson' strap scored 0.0). Returns (x, y, w, h)
    in frame pixels, or None."""
    import cv2

    H, W = frame_bgr.shape[:2]
    y0 = int(H * 0.55)
    region = frame_bgr[y0:int(H * 0.95), :]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (180, 50, 255))
    k = max(3, W // 128)                        # scale with resolution
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (k, max(3, k // 2))))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not STRAP_MIN_AREA * W * H <= area <= STRAP_MAX_AREA * W * H:
            continue
        if not STRAP_ASPECT[0] <= w / max(h, 1) <= STRAP_ASPECT[1]:
            continue
        if cv2.contourArea(c) / area < 0.85:    # must be box-shaped
            continue
        inner = cv2.cvtColor(region[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        dark = float((inner < 110).mean())
        if not 0.03 <= dark <= 0.45:            # needs text, not a blank patch
            continue
        if area > best_area:
            best, best_area = (x, y + y0, w, h), area
    return best


def _find_chyron_frames(video: Path, segments: list[Segment]) -> list[dict]:
    """Decode once, sample inside speech segments, return the best lower-third
    crop per contiguous 'chyron on screen' event. Each event carries the crop
    pre-encoded as JPEG bytes under 'jpeg'."""
    import cv2

    spans = sorted((s.start, s.end, s.speaker) for s in segments)

    def label_at(t: float) -> str | None:
        for start, end, label in spans:
            if start <= t <= end:
                return label
            if start > t:
                break
        return None

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * SAMPLE_INTERVAL_S)))

    candidates = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            label = label_at(t)
            if label is not None:
                ok, frame = cap.retrieve()
                if ok:
                    h = frame.shape[0]
                    band = frame[int(h * BAND_TOP):int(h * BAND_BOTTOM), :]
                    score = _text_score(band)
                    strap = _find_strap_box(frame)
                    if strap is not None:
                        # a detected strap box outranks text-score candidates
                        # and is sent as a tight crop: large, legible name,
                        # nothing else in the image to misread
                        sx, sy, sw, sh = strap
                        mx, my = int(sw * 0.25), int(sh * 0.6)
                        crop = frame[max(0, sy - my):sy + sh + my,
                                     max(0, sx - mx):sx + sw + mx]
                        candidates.append({"t": t, "label": label,
                                           "score": 1.0 + score, "band": crop})
                    elif score >= MIN_TEXT_SCORE:
                        candidates.append(
                            {"t": t, "label": label, "score": score, "band": band})
        idx += 1
    cap.release()

    # merge consecutive detections of the same label into events, keep best frame
    events: list[dict] = []
    for c in candidates:
        prev = events[-1] if events else None
        if prev and c["label"] == prev["label"] and c["t"] - prev["t"] <= EVENT_GAP_S:
            if c["score"] > prev["score"]:
                events[-1] = c
        else:
            events.append(c)

    # Per-speaker quota, then score: text-heavy B-roll (documents, graphics)
    # during one speaker's narration must not crowd every other speaker's
    # name strap out of the frame budget.
    events.sort(key=lambda e: e["score"], reverse=True)
    selected: list[dict] = []
    per_label: dict[str, int] = {}
    for e in events:
        if per_label.get(e["label"], 0) >= PER_LABEL_CAP:
            continue
        per_label[e["label"]] = per_label.get(e["label"], 0) + 1
        selected.append(e)
        if len(selected) >= MAX_FRAMES_TO_LLM:
            break
    events = sorted(selected, key=lambda e: e["t"])
    for e in events:
        ok, jpg = cv2.imencode(".jpg", e.pop("band"),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        e["jpeg"] = jpg.tobytes() if ok else b""
    return [e for e in events if e["jpeg"]]


def read_chyron_crops(
    crops: list[dict],
    client: anthropic.Anthropic | None = None,
) -> tuple[list[dict], list[str]]:
    """Have Claude read pre-detected lower-third crops.

    crops: [{"t": seconds, "label": "SPEAKER_xx", "jpeg": bytes}, ...]
    Shared by the local CLI and the cloud attribution Lambda (which gets its
    crops from S3, detected by the GPU task).
    """
    if client is None:
        client = anthropic.Anthropic()
    if not crops:
        return [], ["chyron stage: no text-bearing lower-third frames detected"]

    content: list[dict] = []
    for c in crops:
        content.append({
            "type": "text",
            "text": f"Lower-third crop at {c['t']:.1f}s — speaking: {c['label']}",
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.standard_b64encode(c["jpeg"]).decode()},
        })
    content.append({
        "type": "text",
        "text": "Report every legible lower-third as JSON per the schema.",
    })

    # NOTE: no temperature param here — claude-opus-5 rejects sampling
    # parameters with a 400 (unlike the sonnet-4-6 call in attribute.py).
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    from .costs import record
    record("chyron", MODEL, response.usage)
    raw = "".join(b.text for b in response.content if b.type == "text")
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], [f"chyron stage: LLM returned non-JSON output: {exc}"]
    return data.get("sightings", []), list(data.get("warnings", []))


def read_chyrons(
    video: Path,
    segments: list[Segment],
    client: anthropic.Anthropic | None = None,
) -> tuple[list[dict], list[str]]:
    """Detect chyron frames locally, then have Claude read only those crops."""
    events = _find_chyron_frames(video, segments)
    print(f"  local detector: {len(events)} chyron-candidate frames "
          f"(labels: {sorted({e['label'] for e in events})})")
    return read_chyron_crops(events, client)


def merge_chyrons(
    result: AttributionResult,
    sightings: list[dict],
    chyron_warnings: list[str],
) -> AttributionResult:
    """Fold chyron names into the mapping. On-screen graphics outrank low/medium
    transcript inference but never silently override a high-confidence name."""
    warnings = [w for w in result.warnings
                if not w.startswith(("Low/medium confidence", "Unmapped speaker",
                                     "Unidentified speakers"))]
    warnings += [f"chyron: {w}" for w in chyron_warnings]

    by_label: dict[str, list[dict]] = {}
    for s in sightings:
        if s.get("name"):
            by_label.setdefault(s["label"], []).append(s)

    for label, seen in by_label.items():
        names = {s["name"] for s in seen}
        if len(names) > 1:
            warnings.append(
                f"{label}: conflicting chyrons {sorted(names)} — possible "
                "diarization merge, needs human check")
            continue
        best = seen[0]
        evidence = (f"On-screen lower-third at {best['time']:.1f}s reads "
                    f"'{best['name']}" +
                    (f", {best['title']}'" if best.get("title") else "'"))
        existing = result.mappings.get(label)
        if existing and existing.confidence == "high" and \
                existing.name not in ("Unidentified", best["name"]):
            warnings.append(
                f"{label}: chyron says '{best['name']}' but transcript "
                f"attribution says '{existing.name}' — needs human check")
            continue
        result.mappings[label] = SpeakerMapping(
            label=label,
            name=best["name"],
            role=best.get("title") or (existing.role if existing else ""),
            confidence="high",
            evidence=evidence,
        )

    # drop stale warnings that only concern labels now confidently named
    resolved = {label for label, m in result.mappings.items()
                if m.confidence == "high" and m.name != "Unidentified"}
    def _stale(w: str) -> bool:
        mentioned = {tok.strip(".,;:'\"()[]") for tok in w.split()
                     if tok.startswith("SPEAKER_")}
        return bool(mentioned) and mentioned <= resolved
    warnings = [w for w in warnings if not _stale(w)]

    result.warnings = warnings
    result.status = _validate(result.segments, result.mappings, result.warnings)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Name speakers from lower-thirds")
    p.add_argument("video", help="video file or HLS/HTTP stream URL")
    p.add_argument("attribution_json", type=Path,
                   help="existing out/<id>.json from the main pipeline")
    p.add_argument("--out", type=Path, default=Path("out"))
    args = p.parse_args()

    data = json.loads(args.attribution_json.read_text(encoding="utf-8"))
    segments = [
        Segment(start=s["start"], end=s["end"], speaker=s["speaker_label"],
                text=s["text"], language=s.get("language"))
        for s in data["segments"]
    ]
    mappings = {
        label: SpeakerMapping(**m) for label, m in data["speakers"].items()
    }
    result = AttributionResult(
        video_id=data["video_id"],
        segments=segments,
        mappings=mappings,
        status=ReviewStatus(data["status"]),
        warnings=list(data.get("warnings", [])),
    )

    print(f"[{result.video_id}] scanning for lower-thirds in {args.video}...")
    sightings, chyron_warnings = read_chyrons(args.video, segments)
    for s in sightings:
        print(f"[{result.video_id}]   {s['label']} @ {s.get('time', 0):.1f}s: "
              f"{s.get('name')}" +
              (f" — {s['title']}" if s.get("title") else ""))

    result = merge_chyrons(result, sightings, chyron_warnings)

    args.out.mkdir(parents=True, exist_ok=True)
    write_vtt(result, args.out / f"{result.video_id}.vtt")
    write_json(result, args.out / f"{result.video_id}.json")
    write_labeled_txt(result, args.out / f"{result.video_id}.labeled.txt")

    print(f"[{result.video_id}] status={result.status.value}")
    for w in result.warnings:
        print(f"[{result.video_id}]   warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

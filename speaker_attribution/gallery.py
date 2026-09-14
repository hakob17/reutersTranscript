"""Face gallery: identify people on screen, not just speakers.

Embeddings come from OpenCV's SFace recognizer (OpenCV Zoo model, no new
dependencies). The persistent gallery (gallery.json) holds only public
figures — Wikidata-verified or manual headshots (scripts/seed_gallery.py).
People named on air in a video (chyron, or a high-confidence attributed
speaker) are enrolled for that video's matching only and never persisted:
no cross-video biometric store of private individuals (docs/DESIGN.md §9).
Unnamed tracks are never enrolled anywhere.

Matching is conservative: cosine similarity must clear an absolute threshold
AND beat the best other identity by a margin, else the face stays unnamed.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
MODEL_DIR = Path(__file__).resolve().parent.parent / ".models"
GALLERY_PATH = Path(__file__).resolve().parent.parent / "gallery.json"

MATCH_THRESHOLD = 0.40    # SFace cosine; the model's standard bar is ~0.363
MATCH_MARGIN = 0.05       # must beat the runner-up identity by this
HIGH_CONFIDENCE = 0.60    # below this a face match alone is medium evidence:
                          # shown as "Possibly <name>", never asserted. The
                          # margin rule can't protect against UNENROLLED
                          # lookalikes (open-set), so medium matches hedge.
MAX_EMBS_PER_NAME = 10    # keep the freshest N embeddings per identity


def make_recognizer():
    """cv2 SFace recognizer, or None when unavailable."""
    try:
        import cv2
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
        if not path.exists():
            urllib.request.urlretrieve(SFACE_URL, path)
        return cv2.FaceRecognizerSF_create(str(path), "")
    except Exception:
        return None


def embed(recognizer, frame_bgr, yunet_row):
    """128-d embedding for one detected face (YuNet row carries the
    landmarks alignCrop needs). None on failure."""
    try:
        aligned = recognizer.alignCrop(frame_bgr, yunet_row)
        feat = recognizer.feature(aligned)
        return [round(float(v), 5) for v in feat.flatten()]
    except Exception:
        return None


# ---- pure-python gallery ops (offline-testable) --------------------------

def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def source_urls(source: str) -> dict:
    """Where an enrollment's portrait came from, as links. Wikidata sources
    are recorded as "wikidata:<qid> commons:<file name>"; manual headshots
    and per-video enrollments have no public URL."""
    if not source.startswith("wikidata:"):
        return {}
    qid, _, commons = source[len("wikidata:"):].partition(" commons:")
    urls = {"wikidata": f"https://www.wikidata.org/wiki/{qid.strip()}"}
    if commons:
        from urllib.parse import quote
        file_ = quote(commons.strip().replace(" ", "_"))
        urls["commons_page"] = f"https://commons.wikimedia.org/wiki/File:{file_}"
        urls["image"] = ("https://commons.wikimedia.org/wiki/Special:FilePath/"
                         f"{file_}?width=800")
    return urls


def _with_urls(entry: dict) -> dict:
    """Keep entry["urls"] aligned with entry["sources"] (older galleries
    predate it)."""
    urls = entry.get("urls") or []
    sources = entry.get("sources", [])
    if len(urls) != len(sources):
        entry["urls"] = [source_urls(src) for src in sources]
    return entry


def load_gallery() -> dict:
    if GALLERY_PATH.exists():
        gallery = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
        return {name: _with_urls(e) for name, e in gallery.items()}
    return {}


def save_gallery(gallery: dict) -> None:
    """Readable layout: per person, sources and portrait links first, one
    embedding per line (128 numbers each) last."""
    def dumps(v):
        return json.dumps(v, ensure_ascii=False)

    blocks = []
    for name, entry in gallery.items():
        entry = _with_urls(entry)
        fields = []
        for key in ("sources", "urls"):
            items = ",\n".join(f"      {dumps(x)}" for x in entry.get(key, []))
            fields.append(f'    "{key}": [\n{items}\n    ]')
        embs = ",\n".join(f"      {dumps(e)}" for e in entry.get("embs", []))
        fields.append(f'    "embs": [\n{embs}\n    ]')
        blocks.append(f"  {dumps(name)}: {{\n" + ",\n".join(fields) + "\n  }")
    # write-then-rename: a run loading the gallery mid-save must see the old
    # file or the new one, never a truncated one
    tmp = GALLERY_PATH.with_name(GALLERY_PATH.name + ".tmp")
    tmp.write_text("{\n" + ",\n".join(blocks) + "\n}\n", encoding="utf-8")
    os.replace(tmp, GALLERY_PATH)


def enroll(gallery: dict, name: str, emb: list[float], source: str) -> bool:
    """Add an embedding; idempotent per source (re-seeding or reprocessing
    the same video adds nothing). Returns True if something was added."""
    entry = _with_urls(gallery.setdefault(
        name, {"embs": [], "sources": [], "urls": []}))
    if source in entry["sources"]:
        return False
    entry["embs"].append(emb)
    entry["sources"].append(source)
    entry["urls"].append(source_urls(source))
    for key in ("embs", "sources", "urls"):
        entry[key] = entry[key][-MAX_EMBS_PER_NAME:]
    return True


def is_confident(score: float) -> bool:
    """A face match alone binds a name only at high similarity; below that
    it is one medium-confidence modality (DESIGN.md §4.2) — tentative."""
    return score >= HIGH_CONFIDENCE


def match(gallery: dict, emb: list[float]) -> tuple[str, float] | None:
    """Best identity for an embedding, or None if below threshold/margin."""
    scored = []
    for name, entry in gallery.items():
        best = max((_cos(emb, e) for e in entry["embs"]), default=0.0)
        scored.append((best, name))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < MATCH_THRESHOLD:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < MATCH_MARGIN:
        return None                      # two identities too close — refuse
    return scored[0][1], round(scored[0][0], 3)

"""Self-building face gallery: identify people on screen, not just speakers.

Embeddings come from OpenCV's SFace recognizer (OpenCV Zoo model, no new
dependencies). The gallery enrolls ONLY evidence-named identities — faces a
chyron named, or faces linked to a high-confidence attributed speaker — so
every recognition carries a provenance chain back to broadcast evidence.
Policy (docs/DESIGN.md §9): that restriction is deliberate; unnamed private
individuals are never enrolled.

Matching is conservative: cosine similarity must clear an absolute threshold
AND beat the best other identity by a margin, else the face stays unnamed.
"""
from __future__ import annotations

import json
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


def load_gallery() -> dict:
    if GALLERY_PATH.exists():
        return json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
    return {}


def save_gallery(gallery: dict) -> None:
    GALLERY_PATH.write_text(json.dumps(gallery, ensure_ascii=False),
                            encoding="utf-8")


def enroll(gallery: dict, name: str, emb: list[float], source: str) -> None:
    entry = gallery.setdefault(name, {"embs": [], "sources": []})
    entry["embs"].append(emb)
    entry["sources"].append(source)
    entry["embs"] = entry["embs"][-MAX_EMBS_PER_NAME:]
    entry["sources"] = entry["sources"][-MAX_EMBS_PER_NAME:]


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

"""Public-figure portraits from Wikidata/Commons for the face gallery.

Shared by scripts/seed_gallery.py (manual seeding) and the streaming server
(automatic lookup of people listed in a video's metadata). Only humans
(P31=Q5) with a curated portrait (P18) are used, and a portrait is enrolled
only when it contains exactly one face. Provenance (Wikidata id + Commons
file) is recorded on every enrollment.
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request

# Wikimedia asks API clients to identify themselves
UA = ("reutersTranscript-gallery-seeder/0.1 "
      "(https://github.com/hakob17/reutersTranscript)")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def normalize_name(name: str) -> str:
    """Case-, accent- and apostrophe-insensitive form for name comparison."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def candidate_people(notes: str) -> list[str]:
    """Person-like entries from a comma-separated keyword list (package
    notes). Wikidata still decides who is a human; this only avoids looking
    up plain topics ("disability", "TIFF")."""
    out = []
    for raw in (notes or "").split(","):
        item = raw.strip()
        tokens = item.split()
        if not 2 <= len(tokens) <= 4 or any(ch.isdigit() for ch in item):
            continue
        # every significant token capitalized; allow name particles
        particles = {"von", "van", "der", "de", "del", "da", "bin", "al", "la", "le"}
        if all(t[0].isupper() or t.lower() in particles for t in tokens):
            out.append(item)
    return out


def resolve(query: str, exact: bool = False) -> dict | None:
    """Wikidata search -> first HUMAN (P31=Q5) item that has an image (P18).

    exact=True accepts only items whose label or alias equals the query
    (normalized). Automatic lookups need this: a loose search for a film
    title like "Being Heumann" returns its subject, a real person, who would
    then be enrolled under the film's name.
    """
    search = _get_json(
        "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
            "action": "wbsearchentities", "search": query, "language": "en",
            "type": "item", "limit": 7, "format": "json"}))
    want = normalize_name(query)
    for hit in search.get("search", []):
        if exact:
            matched = (hit.get("match") or {}).get("text", "")
            if want not in (normalize_name(matched),
                            normalize_name(hit.get("label", ""))):
                continue
        qid = hit["id"]
        ent = _get_json(
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
        claims = ent["entities"][qid].get("claims", {})
        is_human = any(
            c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") == "Q5"
            for c in claims.get("P31", []))
        images = [c["mainsnak"]["datavalue"]["value"]
                  for c in claims.get("P18", [])
                  if "datavalue" in c["mainsnak"]]
        if is_human and images:
            return {"qid": qid, "label": hit.get("label", query),
                    "description": hit.get("description", ""),
                    "file": images[0]}
    return None


def fetch_image(filename: str, width: int = 800):
    import cv2
    import numpy as np
    url = ("https://commons.wikimedia.org/wiki/Special:FilePath/"
           + urllib.parse.quote(filename.replace(" ", "_"))
           + f"?width={width}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = np.frombuffer(r.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def single_face_embedding(img, recognizer):
    """Embed the face ONLY if exactly one face is detected. -> (emb, n)."""
    import cv2

    from .faces import _yunet_path
    from .gallery import embed

    h, w = img.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
    det = cv2.FaceDetectorYN_create(str(_yunet_path()), "", (w, h),
                                    score_threshold=0.7)
    _, faces = det.detect(img)
    n = 0 if faces is None else len(faces)
    if n != 1:
        return None, n
    return embed(recognizer, img, faces[0]), 1


def seed_people(names: list[str], gallery: dict, recognizer) -> dict:
    """Enroll Wikidata portraits for `names` into `gallery` (in place).

    Names already present in the gallery are skipped without any network
    call. Returns {"enrolled": [...], "have": [...], "not_found": [...]}.
    """
    from .gallery import enroll

    report = {"enrolled": [], "have": [], "not_found": []}
    known = {normalize_name(k) for k in gallery}
    for name in names:
        if normalize_name(name) in known:
            report["have"].append(name)
            continue
        try:
            hit = resolve(name, exact=True)
            img = fetch_image(hit["file"]) if hit else None
            emb, _ = (single_face_embedding(img, recognizer)
                      if img is not None else (None, 0))
        except Exception:
            hit, emb = None, None
        if hit is None or emb is None:
            report["not_found"].append(name)
            continue
        enroll(gallery, hit["label"], emb,
               f"wikidata:{hit['qid']} commons:{hit['file']}")
        known.add(normalize_name(hit["label"]))
        report["enrolled"].append(hit["label"])
    return report

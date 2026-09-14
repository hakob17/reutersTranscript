"""Seed the face gallery with public figures from Wikidata / Wikimedia Commons.

  python scripts/seed_gallery.py "Prince Harry" "Pope Francis"
  python scripts/seed_gallery.py --file names.txt       # one name per line
  python scripts/seed_gallery.py --demo                 # people in the demo catalogue
  python scripts/seed_gallery.py --image headshot.jpg --name "Jane Doe"
  add --dry-run to resolve and download without writing gallery.json

Why Wikidata/Commons rather than image search: the Wikidata "image" property
(P18) is curated to depict that specific person, the item is checked to be a
human (P31=Q5), and Commons files carry licenses. Image search returns
lookalikes, fan edits and group shots — any of which would poison the
gallery. Only images with exactly ONE detected face are enrolled, and every
enrollment records its provenance (Wikidata id + Commons file).

Policy (docs/DESIGN.md §9): the gallery holds public figures and own staff
only. Biometric identification can fall under biometric-data law (GDPR
Art. 9, BIPA, ...) — get a legal review before production use.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from speaker_attribution.gallery import (  # noqa: E402
    enroll, load_gallery, make_recognizer, save_gallery)
from speaker_attribution.wikidata import (  # noqa: E402
    fetch_image, resolve, single_face_embedding)

# (display name used on boxes, Wikidata search query)
DEMO_PEOPLE = [
    ("Prince Harry", "Prince Harry"),
    ("Meghan, Duchess of Sussex", "Meghan Markle"),
    ("Pope Francis", "Pope Francis"),
    ("Jared Kushner", "Jared Kushner"),
    ("Donald Trump", "Donald Trump"),
    ("Boris Johnson", "Boris Johnson"),
    ("Kyriakos Mitsotakis", "Kyriakos Mitsotakis"),
    ("Nikos Dendias", "Nikos Dendias"),
    ("Recep Tayyip Erdogan", "Recep Tayyip Erdoğan"),
    ("Amrullah Saleh", "Amrullah Saleh"),
    ("Pam Bondi", "Pam Bondi"),
    ("Emmanuel Macron", "Emmanuel Macron"),
    # further figures featured in or discussed by catalogue videos
    ("Todd Blanche", "Todd Blanche"),
    ("Benjamin Netanyahu", "Benjamin Netanyahu"),
    ("Mahmoud Abbas", "Mahmoud Abbas"),
    ("Mohamed bin Zayed", "Mohamed bin Zayed Al Nahyan"),
    ("Robert O'Brien", "Robert C. O'Brien"),
    ("Mike Pompeo", "Mike Pompeo"),
    ("Antonio Guterres", "António Guterres"),
    ("David Beasley", "David Beasley"),
    ("Carrie Lam", "Carrie Lam"),
    ("Imran Khan", "Imran Khan"),
    ("Michel Barnier", "Michel Barnier"),
    ("Ursula von der Leyen", "Ursula von der Leyen"),
    ("Abdalla Hamdok", "Abdalla Hamdok"),
    ("Cate Blanchett", "Cate Blanchett"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="people to look up on Wikidata")
    p.add_argument("--file", type=Path, help="text file, one name per line")
    p.add_argument("--demo", action="store_true",
                   help="seed the people who appear in the demo catalogue")
    p.add_argument("--image", type=Path, help="local headshot to enroll")
    p.add_argument("--name", help="display name for --image")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    recognizer = make_recognizer()
    if recognizer is None:
        print("SFace recognizer unavailable (OpenCV / model download failed)")
        return 1
    gallery = load_gallery()
    added = 0

    if args.image:
        if not args.name:
            p.error("--image requires --name")
        import cv2
        img = cv2.imread(str(args.image))
        emb, n = single_face_embedding(img, recognizer) if img is not None \
            else (None, 0)
        if emb is None:
            print(f"SKIP {args.image}: {n} faces detected (need exactly 1)")
        else:
            print(f"OK   {args.name} <- {args.image}")
            if not args.dry_run and enroll(gallery, args.name, emb,
                                           f"manual:{args.image.name}"):
                added += 1

    people = [(n, n) for n in args.names]
    if args.file:
        people += [(l.strip(), l.strip())
                   for l in args.file.read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.strip().startswith("#")]
    if args.demo:
        people += DEMO_PEOPLE

    for display, query in people:
        try:
            hit = resolve(query)
        except Exception as exc:
            print(f"SKIP {display}: lookup failed ({exc})")
            continue
        if not hit:
            print(f"SKIP {display}: no human Wikidata item with an image")
            continue
        try:
            img = fetch_image(hit["file"])
        except Exception as exc:
            print(f"SKIP {display}: image download failed ({exc})")
            continue
        emb, n = single_face_embedding(img, recognizer) if img is not None \
            else (None, 0)
        if emb is None:
            print(f"SKIP {display}: {hit['qid']} image has {n} faces "
                  "(need exactly 1)")
            continue
        new = (not args.dry_run and
               enroll(gallery, display, emb,
                      f"wikidata:{hit['qid']} commons:{hit['file']}"))
        added += int(new)
        status = "OK  " if new or args.dry_run else "HAVE"
        print(f"{status} {display} -> {hit['qid']} ({hit['description']}) "
              f"| {hit['file']}")

    if added:
        save_gallery(gallery)
    print(f"done: {added} enrollments; gallery has {len(gallery)} identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

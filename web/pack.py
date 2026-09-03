"""Pack the player page into ONE self-contained HTML file for stakeholders.

Embeds the (720p) videos as base64 data URIs and inlines the transcript JSON
for every id, so the result opens with a double-click — no server, no sidecar
files. The page's tab bar (Reuters / Insurer / ...) switches between videos.

Usage:
  python web/pack.py                                   # both demo videos
  python web/pack.py --ids 782809                      # single video
  python web/pack.py --ids epstein_778738,782809 --out demo.html
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "index.html"

# Where each id's embeddable 720p file lives.
VIDEO_FILES = {
    "epstein_778738": ROOT / "epstein_778738_720.mp4",
    "782809": ROOT / "782809_720.mp4",
}

# CDN stream URLs for --stream mode (page plays these via hls.js; needs
# internet and a CDN that allows cross-origin segment fetches).
STREAM_URLS = {
    "epstein_778738": "https://ajo.prod.reuters.tv/v3/playlist/1920x1080/778738/rendition.m3u8",
    "782809": "https://ajo.prod.reuters.tv/v3/playlist/782809/master.m3u8",
}


def replace_marked_line(html: str, marker: str, new_line: str) -> str:
    """Replace the single line carrying `// PACK:<marker>` with new_line."""
    pattern = re.compile(rf"^.*// PACK:{marker}\s*$", re.MULTILINE)
    if not pattern.search(html):
        raise SystemExit(f"pack failed: marker PACK:{marker} not found in template")
    return pattern.sub(lambda _m: new_line, html, count=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="epstein_778738,782809",
                   help="comma-separated video ids (must match TABS in index.html)")
    p.add_argument("--stream", action="store_true",
                   help="reference CDN HLS URLs instead of embedding video "
                        "(tiny file, needs internet)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    out = args.out or (ROOT / ("transcript_demo_stream.html" if args.stream
                               else "transcript_demo_share.html"))

    html = TEMPLATE.read_text(encoding="utf-8")

    sources = {}
    for vid in ids:
        if args.stream:
            sources[vid] = STREAM_URLS[vid]
        else:
            f = VIDEO_FILES.get(vid, ROOT / f"{vid}_720.mp4")
            sources[vid] = ("data:video/mp4;base64,"
                            + base64.b64encode(f.read_bytes()).decode())
    html = replace_marked_line(
        html, "SRC", f"const SOURCES = {json.dumps(sources)}; // packed")

    data = {}
    for vid in ids:
        data[vid] = json.loads((ROOT / "out" / f"{vid}.json").read_text(encoding="utf-8"))
    inline = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = replace_marked_line(
        html, "DATA",
        f"const DATA = {inline}; const loadData = id => Promise.resolve(DATA[id]); // packed")

    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size
    size_str = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.0f} KB"
    print(f"wrote {out} ({size_str}, videos: {', '.join(ids)}, "
          f"{'CDN stream' if args.stream else 'embedded'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

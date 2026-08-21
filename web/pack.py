"""Pack the player page into ONE self-contained HTML file for stakeholders.

Embeds the (720p) video as a base64 data URI and inlines the transcript JSON,
so the result opens with a double-click — no server, no sidecar files.

Usage:
  python web/pack.py [--video epstein_778738_720.mp4]
                     [--json out/epstein_778738.json]
                     [--out epstein_778738_share.html]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "index.html"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=ROOT / "epstein_778738_720.mp4")
    p.add_argument("--json", dest="json_path", type=Path,
                   default=ROOT / "out" / "epstein_778738.json")
    p.add_argument("--out", type=Path, default=ROOT / "epstein_778738_share.html")
    args = p.parse_args()

    html = TEMPLATE.read_text(encoding="utf-8")

    video_b64 = base64.b64encode(args.video.read_bytes()).decode()
    html = html.replace(
        '<source src="../epstein_778738_web.mp4" type="video/mp4">',
        f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">',
    )

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    inline = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace(
        "const loadData = () => fetch('../out/epstein_778738.json')"
        ".then(r => r.json()); // PACK:DATA",
        f"const loadData = () => Promise.resolve({inline}); // packed",
    )

    if "data:video/mp4" not in html or "Promise.resolve" not in html:
        raise SystemExit("pack failed: template markers not found in index.html")

    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

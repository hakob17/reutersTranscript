"""CLI: run the full pipeline on one video or a directory of videos.

Usage:
  python -m speaker_attribution.pipeline VIDEO.mp4 --shotlist VIDEO.shotlist.txt
  python -m speaker_attribution.pipeline videos/ --shotlist-dir shotlists/ --out out/

Env:
  HF_TOKEN            HuggingFace token (pyannote diarization models)
  ANTHROPIC_API_KEY   Claude API key for the attribution stage
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .attribute import attribute_speakers
from .models import ReviewStatus
from .outputs import write_json, write_labeled_txt, write_vtt
from .transcribe import extract_audio, transcribe_and_diarize

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".ts", ".m3u8"}


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def source_id(source: str | Path, override: str | None = None) -> str:
    """Stable video id for a local path or a stream URL."""
    if override:
        return override
    if isinstance(source, str) and is_url(source):
        # e.g. .../playlist/1920x1080/778738/rendition.m3u8 -> 778738
        parts = [p for p in source.split("?")[0].split("/") if p]
        stem = Path(parts[-1]).stem
        return parts[-2] if stem in ("rendition", "playlist", "index", "master") \
            and len(parts) >= 2 else stem
    return Path(source).stem


def process_one(
    video: Path | str,
    shotlist_path: Path | None,
    out_dir: Path,
    args: argparse.Namespace,
) -> ReviewStatus:
    video_id = source_id(video, getattr(args, "video_id", None))
    print(f"[{video_id}] extracting audio...")
    wav = extract_audio(video, out_dir / f"{video_id}.wav")

    print(f"[{video_id}] transcribing + diarizing ({args.model} on {args.device})...")
    segments = transcribe_and_diarize(
        wav,
        hf_token=os.environ["HF_TOKEN"],
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    print(f"[{video_id}] {len(segments)} segments, "
          f"{len({s.speaker for s in segments})} speakers detected")

    shotlist = shotlist_path.read_text(encoding="utf-8") if shotlist_path else ""
    if not shotlist:
        print(f"[{video_id}] WARNING: no shotlist — attribution will be weaker")

    print(f"[{video_id}] resolving speakers via Claude...")
    result = attribute_speakers(
        video_id=video_id,
        segments=segments,
        shotlist=shotlist,
        byline=args.byline or "",
    )

    write_vtt(result, out_dir / f"{video_id}.vtt")
    write_json(result, out_dir / f"{video_id}.json")
    write_labeled_txt(result, out_dir / f"{video_id}.labeled.txt")
    wav.unlink(missing_ok=True)

    print(f"[{video_id}] status={result.status.value}")
    for w in result.warnings:
        print(f"[{video_id}]   warning: {w}")
    return result.status


def main() -> int:
    p = argparse.ArgumentParser(description="Speaker attribution pipeline")
    p.add_argument("input",
                   help="video file, directory of videos, or HLS/HTTP stream URL "
                        "(ffmpeg reads .m3u8 directly — no MP4 download needed)")
    p.add_argument("--video-id",
                   help="override the derived video id (useful for stream URLs)")
    p.add_argument("--shotlist", type=Path, help="shotlist/script text file (single video)")
    p.add_argument("--shotlist-dir", type=Path,
                   help="directory with <video_stem>.txt shotlists (batch mode)")
    p.add_argument("--byline", help="reporter byline/credits string")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute-type", default="float16",
                   help="float16 on GPU, int8 on CPU")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if is_url(args.input):
        videos: list[Path | str] = [args.input]
    elif Path(args.input).is_dir():
        videos = sorted(v for v in Path(args.input).iterdir()
                        if v.suffix.lower() in VIDEO_EXTS)
    else:
        videos = [Path(args.input)]

    needs_review, failed = [], []
    for video in videos:
        shotlist_path = args.shotlist
        if shotlist_path is None and args.shotlist_dir:
            candidate = args.shotlist_dir / f"{source_id(video)}.txt"
            shotlist_path = candidate if candidate.exists() else None
        try:
            status = process_one(video, shotlist_path, args.out, args)
        except Exception as exc:  # keep the batch going
            print(f"[{source_id(video)}] FAILED: {exc}", file=sys.stderr)
            failed.append(source_id(video))
            continue
        if status is ReviewStatus.NEEDS_REVIEW:
            needs_review.append(source_id(video))
        elif status is ReviewStatus.FAILED:
            failed.append(source_id(video))

    print("\n=== batch summary ===")
    print(f"processed: {len(videos)}, needs review: {needs_review or 'none'}, "
          f"failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

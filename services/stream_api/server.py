"""On-demand streaming transcript demo — front/back for the play-while-we-process pattern.

The video starts playing immediately (hls.js against the CDN); this backend
processes the SAME video on demand and streams transcript state to the page
over SSE. The invariant is "processing edge ahead of playhead", and with
caption-first attribution the text edge reaches the end of the video within
seconds of the first request — only speaker naming trails, and it arrives as
retroactive updates.

Event protocol (SSE, one JSON object per event):
  {type:"status", stage, detail}                     pipeline progress
  {type:"lines", lines:[{i,start,end,text}]}         APPEND transcript lines
  {type:"speakers_assigned", assignments:[{i,label}]} UPDATE lines w/ diarized label
  {type:"names", mapping:{label:{name,role,confidence}},
   review_status, warnings}                          UPDATE labels -> real names
  {type:"done"}

Design points from the streaming-attribution spec, all demonstrated here:
  * no video gating — playback never waits for processing
  * dedup — one job per video id, no matter how many viewers subscribe
  * cache — a finished job's event log is persisted; later viewers (or a
    server restart) replay it instantly
  * retroactive corrections — the protocol UPDATES lines, never only appends
  * cancellation trade-off — processing continues after viewers leave and the
    cache keeps the result (the right call for news: next viewer is free)

Run:  .venv\\Scripts\\python.exe -m uvicorn services.stream_api.server:app --port 8031
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speaker_attribution.attribute import attribute_speakers  # noqa: E402
from speaker_attribution.captions import (  # noqa: E402
    assign_cue_speakers, discover_caption_url, fetch_caption_cues)
from speaker_attribution.transcribe import diarize_only, extract_audio  # noqa: E402

CACHE_DIR = ROOT / "out_stream"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Demo catalogue: id -> HLS master URL (any captioned master works).
VIDEOS = {
    "782809": "https://ajo.prod.reuters.tv/v3/playlist/782809/master.m3u8",
}


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Job:
    """One processing job per video id. Events accumulate in a log; SSE
    subscribers replay the log from 0 and then follow live — which makes
    late joiners, concurrent viewers, and cache replay the same code path."""

    def __init__(self, video_id: str, url: str):
        self.video_id = video_id
        self.url = url
        self.events: list[dict] = []
        self.done = False
        self.cond = threading.Condition()

    def emit(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            if event.get("type") == "done" or event.get("stage") == "error":
                self.done = True
            self.cond.notify_all()

    # ---- pipeline ------------------------------------------------------

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # surface, don't kill the server
            self.emit({"type": "status", "stage": "error", "detail": str(exc)})

    def _run(self) -> None:
        emit = self.emit
        emit({"type": "status", "stage": "captions",
              "detail": "probing HLS master for a caption rendition"})
        cap_url = discover_caption_url(self.url)
        if not cap_url:
            emit({"type": "status", "stage": "error",
                  "detail": "no caption rendition in master playlist "
                            "(demo covers the caption-first path)"})
            return
        cues = fetch_caption_cues(cap_url)

        # Text edge races ahead of the playhead immediately: lines stream in
        # small batches (the per-HLS-segment arrival pattern) with no speaker.
        batch = 8
        for i in range(0, len(cues), batch):
            emit({"type": "lines", "lines": [
                {"i": i + j, "start": c["start"], "end": c["end"], "text": c["text"]}
                for j, c in enumerate(cues[i:i + batch])
            ]})
        emit({"type": "status", "stage": "audio",
              "detail": f"{len(cues)} caption cues loaded; extracting audio "
                        "from the stream"})

        with tempfile.TemporaryDirectory() as td:
            wav = extract_audio(self.url, Path(td) / f"{self.video_id}.wav")
            device = _device()
            emit({"type": "status", "stage": "diarizing",
                  "detail": f"pyannote speaker turns on {device}"})
            turns = diarize_only(wav, hf_token=os.environ["HF_TOKEN"],
                                 device=device)

        segments = assign_cue_speakers(cues, turns)
        emit({"type": "speakers_assigned", "assignments": [
            {"i": i, "label": s.speaker} for i, s in enumerate(segments)
        ]})

        emit({"type": "status", "stage": "attributing",
              "detail": "resolving names via Claude"})
        result = attribute_speakers(video_id=self.video_id, segments=segments,
                                    shotlist="", byline="")
        emit({"type": "names",
              "mapping": {label: {"name": m.name, "role": m.role,
                                  "confidence": m.confidence}
                          for label, m in result.mappings.items()},
              "review_status": result.status.value,
              "warnings": result.warnings})
        emit({"type": "done"})

        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"{self.video_id}.events.json").write_text(
            json.dumps(self.events, ensure_ascii=False), encoding="utf-8")


class Registry:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def get_or_start(self, video_id: str, url: str) -> Job:
        with self.lock:
            job = self.jobs.get(video_id)
            if job is not None:
                return job                       # dedup: second viewer attaches

            job = Job(video_id, url)
            self.jobs[video_id] = job
            cached = CACHE_DIR / f"{video_id}.events.json"
            if cached.exists():                  # cache: replay, no processing
                job.events = json.loads(cached.read_text(encoding="utf-8"))
                job.events.insert(0, {"type": "status", "stage": "cached",
                                      "detail": "served from cache — processed "
                                                "on a previous view"})
                job.done = True
            else:
                threading.Thread(target=job.run, daemon=True).start()
            return job


registry = Registry()
app = FastAPI(title="streaming transcript demo")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/stream/{video_id}")
def stream(video_id: str):
    url = VIDEOS.get(video_id)
    if url is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'status', 'stage': 'error', 'detail': 'unknown video id'})}\n\n"]),
            media_type="text/event-stream")
    job = registry.get_or_start(video_id, url)

    def gen():
        sent = 0
        while True:
            with job.cond:
                while sent >= len(job.events) and not job.done:
                    job.cond.wait(timeout=15)
                pending = job.events[sent:]
                sent = len(job.events)
                finished = job.done
            for ev in pending:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if finished and sent >= len(job.events):
                return
            if not pending:  # keep proxies from closing an idle stream
                yield ": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/videos")
def videos():
    return {vid: {"url": url} for vid, url in VIDEOS.items()}

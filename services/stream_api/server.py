"""On-demand streaming transcript demo — front/back for the play-while-we-process pattern.

The video starts playing immediately (hls.js against the CDN); this backend
processes the SAME video on demand and streams transcript state to the page
over SSE. The invariant is "processing edge ahead of playhead":

  * caption-first (fast path): the CDN's own WebVTT gives text + timings, so
    the text edge reaches the end of the video within seconds; only naming
    trails, arriving as retroactive updates.
  * ASR fallback (e.g. Arabic — no caption rendition on the CDN): Whisper
    large-v3 transcribes the audio in ~30s chunks and lines stream out per
    chunk. On a GPU each chunk transcribes many times faster than real time,
    so the text edge still outruns the playhead after the model warms up.
    Language is auto-detected (Arabic included); the page renders RTL text
    correctly via dir="auto".

Event protocol (SSE, one JSON object per event):
  {type:"status", stage, detail}                     pipeline progress
  {type:"lines", lines:[{i,start,end,text}]}         APPEND transcript lines
  {type:"speakers_assigned", assignments:[{i,label}]} UPDATE lines w/ diarized label
  {type:"names", mapping:{label:{name,role,confidence}},
   review_status, warnings}                          UPDATE labels -> real names
  {type:"done"}

Also demonstrated: dedup (one job per video id), cache (finished event logs
persist to out_stream/ and replay instantly, across restarts), and the
continue-after-viewers-leave trade-off (right call for news: the next viewer
is free).

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

# Windows: ctranslate2 (faster-whisper) needs torch's bundled cuBLAS/cuDNN
# DLLs on PATH to run CUDA — the venv's torch/lib carries them.
if sys.platform == "win32":
    _torch_lib = ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.is_dir():
        os.environ["PATH"] = f"{_torch_lib};{os.environ.get('PATH', '')}"
        os.add_dll_directory(str(_torch_lib))

from speaker_attribution.attribute import attribute_speakers  # noqa: E402
from speaker_attribution.captions import (  # noqa: E402
    assign_cue_speakers, discover_caption_url, fetch_caption_cues)
from speaker_attribution.transcribe import diarize_only, extract_audio  # noqa: E402

CACHE_DIR = ROOT / "out_stream"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ASR_CHUNK_S = 30          # transcription window; lines stream per chunk
SAMPLE_RATE = 16_000

# Demo catalogue: id -> {url, title}. 778738 uses a rendition playlist with
# no caption track, so it exercises the ASR streaming path end-to-end —
# the same route an Arabic video takes.
VIDEOS = {
    "782809": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/782809/master.m3u8",
        "title": "Tuio CEO: AI more a threat to brokers for personal lines than commercial",
    },
    "epstein_778738": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/1920x1080/778738/rendition.m3u8",
        "title": "Epstein accusers face new waves of harassment after speaking out",
    },
    # Arabic (reuters.com/ar articles, Aug-Sep 2020). Most carry
    # auto-generated Arabic captions under a mislabeled en_US rendition, so
    # they take the caption-first path with RTL rendering; any without
    # captions fall to chunked Whisper automatically.
    "404233": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/404233/master.m3u8",
        "title": "شرطة برلين تحتجز 300 وتفرق مظاهرة ضد قيود فيروس كورونا",
    },
    "404327": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/404327/master.m3u8",
        "title": "مستشار ترامب: دول عربية وإسلامية أخرى ستتبع الإمارات وتطبع علاقاتها مع إسرائيل",
    },
    "404389": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/404389/master.m3u8",
        "title": "فرقة تحيي حفلا على الهواء عبر السيارات بإندونيسيا مع احتدام حالات الإصابة بكورونا",
    },
    "404475": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/404475/master.m3u8",
        "title": "مسؤولون إسرائيليون وأمريكيون يصلون للإمارات وكوشنر يحث الفلسطينيين على التفاوض",
    },
    "404918": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/404918/master.m3u8",
        "title": "ابتسامة عريضة ودعاء للبنان في أول لقاء مفتوح للبابا فرنسيس منذ ستة أشهر",
    },
    "404963": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/404963/master.m3u8",
        "title": "افتتاح مهرجان البندقية بتعبير عن التضامن مع صناعة السينما المتضررة من جائحة كورونا",
    },
    "404878": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/404878/master.m3u8",
        "title": "بدء محاكمة متورطين في هجوم على مجلة شارلي إبدو في فرنسا",
    },
    "405217": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405217/master.m3u8",
        "title": "البولشوي الروسي يفتح أبوابه مجددا بعرض لأوبرا (دون كارلو)",
    },
    "405230": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405230/master.m3u8",
        "title": "برنامج الغذاء العالمي: انفجار بيروت يضر بطاقة استيعاب الحبوب لكن الإمدادات لا تزال تتدفق",
    },
    "405065": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405065/master.m3u8",
        "title": "الأمم المتحدة تحذر من تخزين السلاح في ليبيا وخروج وباء كورونا عن السيطرة",
    },
    "405179": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405179/master.m3u8",
        "title": "الأمير البريطاني هاري وزوجته ميجان يوقعان عقدا مع نتفليكس لإنتاج برامج",
    },
    "405447": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405447/master.m3u8",
        "title": "أغاني فرانك سيناترا قد تنقذ الفيل كافان في باكستان",
    },
    "405474": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405474/master.m3u8",
        "title": "الآلاف يحتجون في باكستان على إعادة نشر رسوم النبي محمد في فرنسا",
    },
    "405239": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405239/master.m3u8",
        "title": "بحرية سريلانكا تستبعد حدوث تسرب نفطي من ناقلة عملاقة شب فيها حريق",
    },
    "405406": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405406/master.m3u8",
        "title": "اليونان تطلب من تركيا الكف عن \"الاستفزازات\" ليبدأ الحوار",
    },
    "405249": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405249/master.m3u8",
        "title": "إسرائيل تعلن إجراءات عزل جزئية بعد زيادة الإصابات بكورونا",
    },
    "405538": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405538/master.m3u8",
        "title": "سينوفاك وسي.إن.بي.جي الصينيتان تختبران لقاحات كورونا في مزيد من الدول",
    },
    "405492": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405492/master.m3u8",
        "title": "مطعم سوشي في اليابان يستعين بلاعبي كمال أجسام لتوصيل الطلبات",
    },
    "405502": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405502/master.m3u8",
        "title": "عمال إنقاذ يبحثون عن أحد الناجين تحت الأنقاض في بيروت لليوم الثاني",
    },
    "405509": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405509/master.m3u8",
        "title": "مقتل 17 مصليا في انفجار خط أنابيب غاز قرب مسجد ببنجلادش",
    },
    "405587": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405587/master.m3u8",
        "title": "الشرطة في هونج كونج تعتقل متظاهرين يحتجون على تأجيل الانتخابات",
    },
    "405714": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405714/master.m3u8",
        "title": "اتفاق بريكست يواجه أزمة جديدة بعد تهديد بريطانيا بتقويضه",
    },
    "405658": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/405658/master.m3u8",
        "title": "نمساوي يحطم رقما قياسيا بالوقوف أكثر من ساعتين ونصف الساعة في صندوق مملوء بالثلج",
    },
    "406147": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/406147/master.m3u8",
        "title": "نجاة نائب رئيس أفغانستان من تفجير في كابول ومقتل 10",
    },
    "406224": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/406224/master.m3u8",
        "title": "شردتهم فيضانات قياسية.. عشرات الآلاف من السودانيين يتنظرون المساعدات",
    },
    "406188": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/406188/master.m3u8",
        "title": "\"بيروت ترنم للأمل\" من داخل كنيسة دمرها انفجار المرفأ",
    },
    "406164": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/406164/master.m3u8",
        "title": "فرار الآلاف بعد اندلاع حريق في مخيم مكتظ باللاجئين في اليونان",
    },
    "406248": {
        "url": "https://ajo.prod.reuters.tv/v3/playlist/406248/master.m3u8",
        "title": "الحوثيون يعلقون الرحلات الجوية إلى صنعاء مع اشتداد الحرب الاقتصادية",
    },
}

_asr_model = None
_asr_lock = threading.Lock()


def _turn_at(t: float, turns: list[tuple[float, float, str]]) -> str | None:
    best, best_ov = None, 0.0
    for start, end, label in turns:
        if start <= t <= end:
            return label
    for start, end, label in turns:  # nearest midpoint fallback
        d = abs((start + end) / 2 - t)
        if best is None or d < best_ov:
            best, best_ov = label, d
    return best


def _split_cues_at_turns(cues: list[dict],
                         turns: list[tuple[float, float, str]]) -> list[dict]:
    """Split ASR cues at diarized speaker changes using their word timings."""
    out: list[dict] = []
    for cue in cues:
        words = [w for w in cue.get("words", []) if "start" in w]
        if len(words) < 2:
            out.append(cue)
            continue
        groups: list[list[dict]] = []
        prev_label = None
        for w in words:
            label = _turn_at((w["start"] + w.get("end", w["start"])) / 2, turns)
            if label != prev_label or not groups:
                groups.append([])
                prev_label = label
            groups[-1].append(w)
        if len(groups) == 1:
            out.append(cue)          # single speaker: keep original punctuation
            continue
        for g in groups:
            out.append({
                "start": g[0]["start"],
                "end": g[-1].get("end", g[-1]["start"]),
                "text": " ".join(w["word"].strip() for w in g).strip(),
                "language": cue.get("language"),
            })
    return [c for c in out if c["text"]]


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _asr(device: str):
    """Whisper large-v3, loaded once per process and kept warm."""
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            import whisperx
            _asr_model = whisperx.load_model(
                "large-v3", device,
                compute_type="float16" if device == "cuda" else "int8")
        return _asr_model


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

        cues: list[dict] = []
        with tempfile.TemporaryDirectory() as td:
            if cap_url:
                cues = fetch_caption_cues(cap_url)
                # Text edge races ahead of the playhead immediately.
                batch = 8
                for i in range(0, len(cues), batch):
                    emit({"type": "lines", "lines": [
                        {"i": i + j, "start": c["start"], "end": c["end"],
                         "text": c["text"]}
                        for j, c in enumerate(cues[i:i + batch])
                    ]})
                emit({"type": "status", "stage": "audio",
                      "detail": f"{len(cues)} caption cues loaded; extracting "
                                "audio from the stream"})
                wav = extract_audio(self.url, Path(td) / f"{self.video_id}.wav")
            else:
                # ASR fallback — the Arabic/uncaptioned route. Chunked so
                # lines stream while later audio is still transcribing.
                emit({"type": "status", "stage": "audio",
                      "detail": "no caption rendition — extracting audio for "
                                "Whisper (ASR streaming path)"})
                wav = extract_audio(self.url, Path(td) / f"{self.video_id}.wav")
                cues = self._asr_stream(wav)

            device = _device()
            emit({"type": "status", "stage": "diarizing",
                  "detail": f"pyannote speaker turns on {device}"})
            turns = diarize_only(wav, hf_token=os.environ["HF_TOKEN"],
                                 device=device)

        # ASR lines carry word timings; split any line that straddles a
        # speaker change so a whole line never gets the wrong voice — the
        # failure that turns into a confident misattribution downstream.
        # (Caption cues are professionally segmented per utterance and come
        # without words; they pass through untouched.)
        if any(c.get("words") for c in cues):
            split = _split_cues_at_turns(cues, turns)
            if len(split) != len(cues):
                cues = split
                emit({"type": "replace_lines", "lines": [
                    {"i": i, "start": c["start"], "end": c["end"],
                     "text": c["text"]}
                    for i, c in enumerate(cues)
                ]})

        segments = assign_cue_speakers(cues, turns)
        emit({"type": "speakers_assigned", "assignments": [
            {"i": i, "label": s.speaker} for i, s in enumerate(segments)
        ]})

        emit({"type": "status", "stage": "attributing",
              "detail": "resolving names via Claude"})
        result = attribute_speakers(video_id=self.video_id, segments=segments,
                                    shotlist="", byline="")
        emit(self._names_event(result, phase="transcript"))

        # Chyron pass: read on-screen name graphics and upgrade the mapping.
        # This is the accuracy stage — transcript cues alone miss speakers the
        # broadcaster only identifies visually. It trails (OpenCV re-reads the
        # stream), landing as one more retroactive update.
        emit({"type": "status", "stage": "chyron",
              "detail": "scanning the stream for on-screen name graphics"})
        try:
            from speaker_attribution.chyron import (
                _find_chyron_frames, merge_chyrons, read_chyron_crops)
            crops = _find_chyron_frames(self.url, segments)
            sightings, chyron_warnings = read_chyron_crops(crops) if crops \
                else ([], ["no text-bearing lower-third frames detected"])
            result = merge_chyrons(result, sightings, chyron_warnings)
            emit(self._names_event(result, phase="chyron"))
        except Exception as exc:  # accuracy pass is best-effort
            emit({"type": "status", "stage": "chyron",
                  "detail": f"chyron pass failed, keeping transcript names: {exc}"})

        emit({"type": "done"})

        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"{self.video_id}.events.json").write_text(
            json.dumps(self.events, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _names_event(result, phase: str) -> dict:
        return {"type": "names", "phase": phase,
                "mapping": {label: {"name": m.name, "role": m.role,
                                    "confidence": m.confidence}
                            for label, m in result.mappings.items()},
                "review_status": result.status.value,
                "warnings": result.warnings}

    def _asr_stream(self, wav: Path) -> list[dict]:
        """Transcribe in windows, emitting lines per window. Returns all cues
        in caption-cue shape so downstream code is shared with the fast path."""
        import whisperx

        device = _device()
        self.emit({"type": "status", "stage": "transcribing",
                   "detail": f"Whisper large-v3 on {device}, "
                             f"{ASR_CHUNK_S}s windows (loading model)"})
        model = _asr(device)
        audio = whisperx.load_audio(str(wav))

        cues: list[dict] = []
        language = None
        align_model = align_meta = None
        chunk = ASR_CHUNK_S * SAMPLE_RATE
        total_s = len(audio) / SAMPLE_RATE
        for offset in range(0, len(audio), chunk):
            piece = audio[offset:offset + chunk]
            if len(piece) < SAMPLE_RATE // 2:
                break
            # chunk_size=8 keeps VAD merges short so lines stay clickable and
            # rarely span two speakers (default 30 would emit one giant
            # segment per window).
            result = model.transcribe(piece, batch_size=16, language=language,
                                      chunk_size=8)
            if language is None:
                language = result.get("language")
                self.emit({"type": "status", "stage": "transcribing",
                           "detail": f"language detected: {language}"})
                try:
                    align_model, align_meta = whisperx.load_align_model(
                        language_code=language, device=_device())
                except Exception:
                    align_model = None  # no align model for this language

            segs = result["segments"]
            if align_model is not None:
                # word timings let us split lines at speaker changes later
                try:
                    segs = whisperx.align(segs, align_model, align_meta,
                                          piece, _device())["segments"]
                except Exception:
                    pass

            t0 = offset / SAMPLE_RATE
            new = [
                {"start": t0 + float(s["start"]),
                 "end": t0 + float(s["end"]),
                 "text": s["text"].strip(),
                 "words": [
                     {"start": t0 + float(w["start"]),
                      "end": t0 + float(w.get("end", w["start"])),
                      "word": w["word"]}
                     for w in s.get("words", []) if "start" in w
                 ]}
                for s in segs if s["text"].strip()
            ]
            base = len(cues)
            cues.extend(new)
            if new:
                self.emit({"type": "lines", "lines": [
                    {"i": base + j, "start": c["start"], "end": c["end"],
                     "text": c["text"]}
                    for j, c in enumerate(new)
                ]})
            self.emit({"type": "status", "stage": "transcribing",
                       "detail": f"text edge at "
                                 f"{min(t0 + ASR_CHUNK_S, total_s):.0f}s "
                                 f"of {total_s:.0f}s ({language})"})
        for c in cues:
            c["language"] = language
        return cues


class Registry:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def get_or_start(self, video_id: str, url: str, fresh: bool = False) -> Job:
        """fresh=True discards any finished job + cache and reprocesses live.
        A job still RUNNING is always attached to (dedup wins over fresh —
        never two parallel jobs for one video)."""
        with self.lock:
            job = self.jobs.get(video_id)
            if job is not None:
                failed = job.done and any(
                    e.get("stage") == "error" for e in job.events)
                if not failed and not (fresh and job.done):
                    return job                   # dedup: second viewer attaches
                # failed jobs are not sticky — fall through and retry live

            cached = CACHE_DIR / f"{video_id}.events.json"
            if fresh:
                cached.unlink(missing_ok=True)

            job = Job(video_id, url)
            self.jobs[video_id] = job
            if not fresh and cached.exists():    # cache: replay, no processing
                job.events = json.loads(cached.read_text(encoding="utf-8"))
                job.events.insert(0, {"type": "status", "stage": "cached",
                                      "detail": "served from cache — processed "
                                                "on a previous view (add "
                                                "fresh=1 to reprocess live)"})
                job.done = True
            else:
                threading.Thread(target=job.run, daemon=True).start()
            return job


registry = Registry()
app = FastAPI(title="streaming transcript demo")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/videos")
def videos():
    return VIDEOS


@app.get("/api/stream/{video_id}")
def stream(video_id: str, fresh: int = 0):
    entry = VIDEOS.get(video_id)
    if entry is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'status', 'stage': 'error', 'detail': 'unknown video id'})}\n\n"]),
            media_type="text/event-stream")
    job = registry.get_or_start(video_id, entry["url"], fresh=bool(fresh))

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

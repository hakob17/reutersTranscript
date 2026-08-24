# Speaker Attribution Pipeline

Adds named speaker labels to news video transcripts:
video → ASR + diarization (WhisperX) → on-screen name reading (OpenCV +
Claude vision) → speaker resolution against the editorial shotlist
(Claude API) → WebVTT captions + archive JSON + review flag.

## Why the shotlist matters
Wire packages ship with a shotlist that lists soundbites and their sources in
broadcast order. Feeding it to the attribution stage turns speaker naming into
deterministic matching instead of guessing. Without a shotlist the pipeline
still runs — the chyron stage (below) recovers names from on-screen graphics —
but confidence drops and results are flagged for review.

## On-screen names (chyron stage)
News packages burn the speaker's name/title into lower-third graphics.
`speaker_attribution/chyron.py` recovers them without spraying frames at an
LLM: OpenCV samples frames inside speech segments, scores the lower-third band
for text-like shapes, and only the few candidate crops (≤16 per video) go to
Claude vision in a single call — so vision cost stays flat regardless of video
length. Recovered names merge into the attribution with cited evidence; they
never silently override a high-confidence transcript-based name, and conflicts
are flagged for review.

## Setup
```bash
# Linux
sudo apt install ffmpeg
# Windows: install ffmpeg (e.g. winget install ffmpeg) and use .venv\Scripts\python.exe

pip install -r requirements.txt
# GPU: replace the CPU torch wheels with CUDA builds
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126

export HF_TOKEN=hf_...            # accept terms on HF for:
                                  #   pyannote/speaker-diarization-community-1
                                  #   (WhisperX >=3.4 default; older setups:
                                  #    pyannote/speaker-diarization-3.1 +
                                  #    pyannote/segmentation-3.0)
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run
Single video:
```bash
python -m speaker_attribution.pipeline package.mp4 \
  --shotlist samples/786096.txt \
  --byline "Alexander Cornwell; camera: Ronen Zvulun" \
  --min-speakers 2 --max-speakers 4
```

Then (optionally) name speakers from on-screen graphics:
```bash
python -m speaker_attribution.chyron package.mp4 out/package.json --out out
```

Batch (shotlists named `<video_stem>.txt`):
```bash
python -m speaker_attribution.pipeline videos/ --shotlist-dir shotlists/ --out out/
```

CPU-only machine:
```bash
python -m speaker_attribution.pipeline package.mp4 \
  --device cpu --compute-type int8 --model medium --shotlist ...
```

## Outputs (per video, in `--out`)
- `<id>.vtt` — WebVTT with `<v Name, Role>` voice tags (JW Player-compatible)
- `<id>.json` — structured record: mappings w/ confidence + evidence, segments
- `<id>.labeled.txt` — human-readable transcript for editorial review
- exit status + per-video `status`: `auto_ok` | `needs_review` | `failed`

## Demo player page
`web/index.html` is an article-style player: video with a chyron-style
speaker-name overlay, plus a synced transcript panel (click a line to seek,
current line highlighted). Serve the repo root with a Range-capable server
(plain `http.server` can't seek video):
```bash
python -m RangeHTTPServer 8017   # then open http://localhost:8017/web/
```

Package the whole demo into ONE self-contained HTML (video + transcript +
pipeline explainer + AWS proposal + per-video cost table embedded — opens with
a double-click, works offline):
```bash
python web/pack.py
```
`reuters_transcript_demo.zip` in the repo root is that page plus README and
raw outputs, ready to hand to stakeholders.

## Review gate
Anything with unmapped labels, sub-high confidence, or "Unidentified" speakers
is marked `needs_review`. Wire that status into your workflow so a human
approves before captions publish — misattribution is the failure mode to avoid.

## Tests
```bash
pytest tests/          # offline; no GPU or API key needed
```

## Production deployment (AWS, Terraform)
`infra/` provisions the full production pipeline: S3 ingest → SQS → Step
Functions → AWS Batch GPU workers (Spot, scale-to-zero) → attribution Lambda →
human review gate (paused workflow + editor callback) → published outputs.
Code is split into deployment units under `services/` (Batch GPU task
container + five Lambdas) reusing the same `speaker_attribution` package.
See [infra/README.md](infra/README.md) for deploy steps.

## Next steps (phase 3)
- Speaker library: enroll correspondent/official voice embeddings in pyannote
  so recurring voices are named without the LLM step
- Review UI: video + editable segment labels, invokes the review_callback
  Lambda to approve/reject

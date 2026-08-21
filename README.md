# Speaker Attribution Pipeline

Adds named speaker labels to news video transcripts:
video → ASR + diarization (WhisperX) → speaker resolution against the
editorial shotlist (Claude API) → WebVTT captions + archive JSON + review flag.

## Why the shotlist matters
Wire packages ship with a shotlist that lists soundbites and their sources in
broadcast order. Feeding it to the attribution stage turns speaker naming into
deterministic matching instead of guessing. Without a shotlist the pipeline
still runs, but confidence drops and results are flagged for review.

## Setup
```bash
sudo apt install ffmpeg
pip install -r requirements.txt
export HF_TOKEN=hf_...            # accept terms for pyannote/segmentation-3.0
                                  # and pyannote/speaker-diarization-3.1 on HF
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

## Review gate
Anything with unmapped labels, sub-high confidence, or "Unidentified" speakers
is marked `needs_review`. Wire that status into your workflow so a human
approves before captions publish — misattribution is the failure mode to avoid.

## Tests
```bash
pytest tests/          # offline; no GPU or API key needed
```

## Next steps (phase 2)
- Queue wrapper (Celery/SQS) — jobs run minutes, don't block ingest
- Speaker library: enroll correspondent/official voice embeddings in pyannote
  so recurring voices are named without the LLM step
- Review UI: video + editable segment labels, writes back to the JSON

# CLAUDE.md

Speaker attribution pipeline for news video: transcribe → diarize → read
on-screen names → attribute speakers via Claude → captions with a human
review gate. Local CLI + demo player, plus a Terraform-defined AWS pipeline.

## Commands

```bash
# venv lives in .venv (Windows: .venv\Scripts\python.exe)
pip install -r requirements.txt
# GPU: PyPI torch is CPU-only on Windows — replace with CUDA wheels:
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126

pytest tests/                                  # offline, no GPU/API needed

# full local pipeline (needs HF_TOKEN + ANTHROPIC_API_KEY, GPU)
python -m speaker_attribution.pipeline VIDEO.mp4 --out out
python -m speaker_attribution.chyron VIDEO.mp4 out/VIDEO.json --out out

# demo player (video seeking REQUIRES a Range-capable server — plain
# http.server silently breaks seeking)
python -m RangeHTTPServer 8017                 # open http://localhost:8017/web/
python web/pack.py                             # one-file shareable demo HTML

# infra
bash scripts/build_lambdas.sh                  # stage build/lambdas/* (run before plan)
terraform -chdir=infra validate
```

## Architecture

Two deployment shapes share ONE library (`speaker_attribution/`):

- **Local CLI**: `pipeline.py` (ffmpeg → `transcribe.py` WhisperX ASR+diarize →
  `attribute.py` Claude → `outputs.py` writers), then `chyron.py` as a second
  pass that fixes names from on-screen graphics.
- **AWS** (`infra/` Terraform, `services/` code): S3 `ingest/videos/` →
  EventBridge → SQS → `lambdas/ingest` → Step Functions →
  Batch GPU task (`services/gpu_task/`, runs transcribe+diarize+chyron
  *detection*, writes `work/<id>/`) → `lambdas/attribute` (chyron *reading* +
  attribution, writes `out/<id>/`) → review gate (`notify_review` pauses via
  waitForTaskToken; `review_callback` resumes) → `lambdas/publish` →
  `published/<id>/`. Job state in DynamoDB.

Key split: **chyron detection (OpenCV, cheap, local/GPU-side) is separate from
chyron reading (Claude vision, only ≤16 pre-detected crops)**. Never send raw
frames to the vision model — detection gates what reaches the API.

## Hard-won constraints (do not regress)

- `speaker_attribution/chyron.py` imports cv2 **lazily inside functions** and
  exposes `read_chyron_crops()` — the attribute Lambda bundles this module
  WITHOUT OpenCV. Adding a top-level `import cv2` breaks the Lambda.
- WhisperX ≥3.4: `DiarizationPipeline(token=...)` (not `use_auth_token=`);
  default diarization model is gated `pyannote/speaker-diarization-community-1`
  — the HF account must accept its terms (3.1 alone is NOT enough: pyannote 4.x
  pulls community-1 assets even when loading the 3.1 config).
- Attribution is deliberately conservative: never guess names; unresolved →
  "Unidentified" + `needs_review`. Chyron names never silently override
  high-confidence transcript names — conflicts become warnings. Misattribution
  is the failure mode this system exists to avoid; keep it that way.
- `attribute.py` / `chyron.py` prompt for **strict JSON, no markdown fences**
  and parse with a fence-stripping fallback; keep both halves in sync.
- MP4s served to browsers need `-movflags +faststart` (moov atom up front) or
  seeking stalls; `epstein_778738_web.mp4` is the remuxed one.
- Videos, `out/`, `.venv/`, `build/`, tfstate are gitignored; the demo zip is
  committed on purpose (stakeholder deliverable).
- `infra/lambdas.tf`: the state machine ARN is a **constructed string**
  (`local.state_machine_arn`) to break a TF dependency cycle; IAM statements
  keep `Resource` as lists so conditional types unify. Don't "simplify" either.
- Step Functions Batch step uses `.sync` and needs the
  `StepFunctionsGetEventsForBatchJobsRule` events permission on the SFN role.

## Models / env

- ASR: Whisper `large-v3` (float16 GPU / int8 CPU). Diarization: pyannote
  community-1. LLM: `claude-sonnet-4-6` in attribute.py, `claude-opus-5` in
  chyron.py.
- Env: `HF_TOKEN`, `ANTHROPIC_API_KEY` (locally: user-level env vars; AWS:
  Secrets Manager, values set via `put-secret-value`, never in TF state).

## Workflow conventions

- After changing `services/lambdas/` or `speaker_attribution/`, re-run
  `bash scripts/build_lambdas.sh` before `terraform plan` (zips hash the build
  dir).
- After a pipeline run improves `out/<id>.json`, the web page picks it up on
  refresh; regenerate the shareable file with `python web/pack.py`.
- Commit style: imperative summary + body of grouped bullets; push to `main`.

# Pipeline flow and AWS cost model

Current as of commit `8a30f2b` (2026-09-10). Architecture, phases and design
rationale live in [DESIGN.md](DESIGN.md); this document is the operational
view: exactly what runs, in what order, how long it takes, and what it costs
on AWS GPU infrastructure.

## 1. Deployment shapes

One library (`speaker_attribution/`), two ways to run it:

| Shape | Code | What it is | Covers today |
|---|---|---|---|
| **Streaming server** | `services/stream_api` | One process runs the whole pipeline on demand and streams results to the player over SSE. This is what a single AWS GPU machine runs. | Everything below |
| **Batch pipeline** | `infra/` (Terraform) + `services/gpu_task`, `services/lambdas` | S3 → SQS → Step Functions → Batch GPU task + Lambdas, scale-to-zero, human review gate | Transcribe, diarize, chyron, attribution. Faces, scenes, translation, gallery and live mode are **not yet ported** to it. |

## 2. End-to-end flow

### 2.1 Viewer and connection

1. `GET /` (served `no-cache`) → `GET /api/videos` → left menu of titles → hls.js starts CDN playback immediately. Playback is never gated on processing.
2. The page opens `EventSource /api/stream/{id}` (`?fresh=1` forces a live reprocess).
3. Registry, one job per video id:
   - job running → attach (replay its event log, then follow live)
   - job finished and cached → replay `out_stream/{id}.events.json` instantly
   - `fresh=1`, or the previous job failed → delete the cache, start a new job

### 2.2 Job stages (on-demand video)

| # | Stage | What happens | Emits |
|---|---|---|---|
| 1 | Setup | Reset the per-run cost accumulator | — |
| 2 | Caption probe | Read the HLS master; look for `#EXT-X-MEDIA TYPE=SUBTITLES` | `status` |
| 3a | **Captions found** | Fetch the WebVTT (single file or segmented playlist, cue settings stripped) → all lines out in batches of 8 within seconds. ffmpeg extracts 16 kHz mono audio from the stream. | `lines` |
| 3b | **No captions** (e.g. some Arabic) | ffmpeg audio → Whisper large-v3 (resident) in 30 s windows (`chunk_size=8`), language detected on the first window, wav2vec2 word alignment → lines per window | `lines` |
| 4 | Translation | If the text is mostly Arabic script: one Claude Sonnet 4.6 call → English per line, sent **before** the slow audio stages | `translations` |
| 5 | Diarization | pyannote `speaker-diarization-community-1` → speaker turns (who speaks when) | — |
| 6 | Boundary split | ASR path only: lines whose word timings cross a speaker change are split | `replace_lines` |
| 7 | Line → speaker | Each line takes the turn it overlaps most | `speakers_assigned` (+ full turn list) |
| 8 | Attribution | Claude Sonnet 4.6 names speakers from transcript evidence only (self-introductions, hand-offs). No evidence → "Unidentified" + `needs_review`. | `names` (transcript) |
| 9 | Chyron | Stream sampled every 0.5 s during speech. A **strap box** (bright box, dark text) wins and is sent as a tight crop; otherwise a text score on the lower band. ≤3 crops per speaker, ≤16 total → Claude Opus 5 reads them → merged (never overrides a high-confidence name; conflicts become warnings). | `names` (chyron) |
| 10 | Faces | See 2.3 | `face_tracks` |
| 11 | Scenes | Talking-head shots templated free; near-duplicate setups reused; Claude Haiku 4.5 describes the rest in one batch and flags hard shots; Claude Opus 5 re-describes only those | `scene_text` |
| 12 | Finish | Measured LLM spend → done → cache written | `costs`, `done` |

### 2.3 Faces stage in detail

On the ~480p rendition (not the master's lowest variant), at 6 frames/s:

- **Detection:** YuNet on a 640-px-wide frame. A hard cut (HSV histogram) closes all open tracks. Shot keyframes (+ 64-bit hash) are collected in the same pass for stage 11.
- **Per face, in shots with ≤4 faces:** mouth-aspect ratio via MediaPipe FaceLandmarker (lip motion). SFace embedding only if the face is prominent (≥2% of frame), lit (mean brightness ≥50) and frontal (head-turn ratio ≤0.5).
- **Tracking:** IoU association (≥0.30), refused when the embedding disagrees with the track (cosine <0.35). Tracks unmatched for 0.75 s close; tracks shorter than 1 s are dropped. Safety net: if a track's early and late faces disagree (<0.5) it is never used for recognition.
- **Chyron name → track:** only when exactly one large face is on screen at the sighting.
- **Track → speaker:** narrator/voice-over labels never claim a face. A solo shot needs visible lip motion; a 2–4-face shot needs the talking face to beat the next by 1.6× lip motion, with every face measured; a 2× vote margin overall; linked faces must cover ≥15% of that speaker's speech.
- **Gallery:** the persistent gallery holds public figures only. People named on air in this video are enrolled for this job only, never saved. Unnamed tracks are matched at cosine ≥0.40 with a 0.05 margin over the runner-up identity; below 0.60 the name is *tentative*. Embeddings are stripped before anything leaves the server.

### 2.4 Live mode (`live_demo`)

ffmpeg `-re` re-streams a package into a growing event playlist. Every 3 s the job decodes new segments; it transcribes once ≥8 s of new audio has accumulated, rediarizes the whole prefix every ~24 s (`remap_labels` keeps speaker labels stable across rediarizations), and refreshes attribution every 15 new lines. No translation or vision stages live; those belong to a post-broadcast reprocess.

### 2.5 Player rendering

- `lines` append · `replace_lines` rebuild · `speakers_assigned` blue "Speaker N" chips · `names` named or role chips · `translations` italic English under each line · `scene_text` amber "[On screen]" rows + a `descriptions` text track · `costs` measured spend line · `status` stage pills + screen-reader announcements · `error` one badge, stream closed.
- **Overlay** (canvas, follows the playhead):
  - Current speaker (from the turn list) → bold orange box with name or role.
  - During a narrator's translation voice-over → box on the person being translated ("Speaker (translated)" if unnamed).
  - Other linked speakers and gallery-recognized people → thin boxes; dashed with "Possibly …" when tentative.
  - More than 5 faces → active speaker only. Bystanders are never boxed.
- Toggles: Speaker boxes / English / Descriptions. A native captions track is regenerated as names improve. Transcript rows are keyboard-seekable.

### 2.6 Hard rules (the never-guess policy)

- A name is shown only on hard evidence: self-introduction, host hand-off, on-screen strap, or a confident face match to a public figure. Otherwise "Unidentified" plus a review flag.
- On-screen names never silently override a confident transcript name; conflicts become warnings.
- Face matches between 0.40 and 0.60 display as "Possibly …"; below 0.40 nothing.
- No persistent biometric data of private individuals.

### 2.7 Offline tools

| Tool | Purpose |
|---|---|
| `scripts/seed_gallery.py` | Enroll public figures from Wikidata/Commons (must be a human with an official portrait; exactly one face in the photo; provenance recorded) or local headshots. Idempotent. |
| `scripts/prewarm.sh` | Process every uncached catalogue video in turn (30 min timeout each; live entries skipped) |
| `python -m speaker_attribution.pipeline <url>` | CLI run of the core pipeline |
| `web/pack.py` | Single-file static demo page |

## 3. Timing

### 3.1 Measured (local machine: RTX 4090 Laptop GPU)

| Path | Milestone | Measured |
|---|---|---|
| Captions, GPU | all text on screen | ~4 s |
| | names (transcript phase) | ~90 s (86.5 s cold, 7:26 video) |
| | full stack incl. chyron, faces, scenes | ~6–8 min |
| Captions, CPU only (no vision) | names | 206 s |
| ASR (no captions) | first lines | ~40 s (model load) |
| | names | ~3–4 min |
| | full stack | ~8–10 min |
| Live | transcript behind the broadcast edge | a few seconds |
| Cached | any view | instant |

### 3.2 Estimated on AWS `g6.xlarge` (1× L4, 4 vCPU), 6–7 min video

Estimates, not measurements: GPU stages scaled ~2× slower than the 4090, CPU stages ~2–3× slower on 4 vCPU. Replace with the benchmark in §5.

| Stage | Bound by | Est. minutes |
|---|---|---|
| Caption probe + fetch | network | <0.1 |
| Audio extraction | CPU / network | 0.2–0.4 |
| ASR (uncaptioned only) | GPU | 2–3 |
| Diarization | GPU | 1–1.5 |
| Four Claude calls | network (GPU idle) | 1–1.5 |
| Chyron detection | CPU decode | 2–3 |
| Faces (YuNet, MediaPipe, SFace) | CPU | 3–5 |
| **Total, caption path** | | **~8–11** |
| **Total, ASR path** | | **~10–14** |

**Finding:** in the current single-process design, roughly two-thirds of the GPU machine's time is CPU work (stream decode and face models) or waiting on API calls while the GPU sits idle.

## 4. Cost breakdown

### 4.1 Unit prices used

AWS us-east-1 on-demand list prices at the time of writing; verify before committing to a budget. Spot is assumed ~65% below on-demand (it varies by instance type and availability zone).

| Item | Price |
|---|---|
| `g6.xlarge` (L4 24 GB, 4 vCPU, 16 GiB) | $0.805/hr (~$588/month 24/7) |
| `g6.2xlarge` (L4, 8 vCPU, 32 GiB) | $0.978/hr |
| `g5.xlarge` (A10G 24 GB, 4 vCPU) | $1.006/hr |
| `g4dn.xlarge` (T4 16 GB, 4 vCPU) | $0.526/hr |
| `c7i.xlarge` (4 vCPU CPU-only) | $0.1785/hr |
| Fargate | $0.0405/vCPU-hr + $0.0044/GB-hr |
| Claude Sonnet 4.6 | $3 / $15 per M tokens (in/out) |
| Claude Opus 5 | $5 / $25 per M tokens |
| Claude Haiku 4.5 | $1 / $5 per M tokens |

### 4.2 LLM cost per video (measured)

From the pipeline's own `costs` event (`speaker_attribution/costs.py`), across
the full 30-video catalogue re-run on the current code (28 Arabic clips + 2
English packages; average length **2.3 min**):

| Stage | Model | Mean per video | Max | Share |
|---|---|---|---|---|
| Translation (28 non-English videos) | Sonnet 4.6 | $0.0165 | $0.023 | 36% |
| Speaker attribution | Sonnet 4.6 | $0.0105 | $0.024 | 24% |
| Scene descriptions | Haiku 4.5 (+ Opus on flagged shots) | $0.0098 | $0.040 | 23% |
| Chyron reading | Opus 5 | $0.0071 | $0.068 | 17% |
| **Total** | | **$0.043** (median $0.040, range $0.025–0.133) | $0.133 | |

The whole catalogue cost **$1.29**. Cost follows length and speaker count:
**~$0.019 per video-minute**. The 6.3-minute, eight-speaker English package
cost $0.133; the ~2-minute Arabic clips $0.03–0.05.

Planning figures: **~$0.04 per short clip (~2 min); ~$0.12 per 6-minute package.**

### 4.3 Compute cost per video (6–7 minute video)

| Design | Caption path | ASR path |
|---|---|---|
| Current single process on `g6.xlarge`, on-demand | $0.11–0.15 | $0.13–0.19 |
| Current single process on `g6.xlarge`, Spot | $0.04–0.05 | $0.05–0.07 |
| Split: GPU task only for audio+ASR+diarization; faces/chyron on CPU Spot in parallel; Claude calls from Lambda — on-demand | $0.04–0.05 | $0.07–0.08 |
| Same split, Spot | $0.013–0.018 | $0.02–0.03 |

Compute scales roughly with length plus about a minute of fixed per-job
overhead, so a ~2-minute clip costs roughly 35–45% of these figures.

### 4.4 AWS platform overhead (monthly)

| Item | Typical monthly |
|---|---|
| S3 (event logs ~0.1–0.3 MB/video, work files) | <$1 |
| Step Functions ($0.025 per 1k transitions, ~10 per video) | ~$0.25 per 1k videos |
| Lambda (Claude calls) | <$1 per 1k videos |
| DynamoDB (job state) | <$1 |
| ECR (~15 GB GPU image with baked models) | ~$1.50 |
| Secrets Manager (Anthropic + HF tokens) | $0.80 |
| CloudWatch logs and alarms | $5–10 |
| Stream/API server (Fargate 0.5 vCPU / 1 GB) | ~$18 |
| Application Load Balancer | ~$18–25 |
| EBS 100 GB gp3 (single-box option) | $8 |
| Data transfer out (only SSE events — video is served by the Reuters CDN) | <$1 |
| **Fixed platform total** | **~$40–60** |

### 4.5 Deployment options and monthly totals

Cost follows length, so two content profiles: **short clips** (the measured
catalogue: ~2.3 min, LLM $0.043) and **6-minute packages** (LLM ~$0.12).
Fixed platform ~$50/month is included.

- **A — always-on `g6.xlarge` box(es) running the streaming server** (what the demo is today). $588/month each on-demand; roughly 30–40% less on a 1-year Savings Plan; ~$294 at 12 h/day. Capacity with the current sequential stages: ~300 short clips or ~150 six-minute packages a day per box. Cached views are free.
- **B — Batch + Spot, scale to zero** (`infra/`), current code: all-in ~$0.07 per short clip, ~$0.165 per package. Cold start 3–8 min (instance boot + image pull) unless a warm pool is kept.
- **C — split design on Spot** (GPU only for audio, ASR and diarization): all-in ~$0.053 per short clip, ~$0.135 per package.
- **Warm box add-on** for instant first views in B or C: one `g6.xlarge` during editorial hours, ~$294/month.

Short clips (~2.3 min):

| Volume | A: always-on | B: Batch Spot | C: split, Spot |
|---|---|---|---|
| 30/day (~900/month) | ~$650 (1 box) | ~$115 | ~$100 |
| 200/day (~6,000/month) | ~$880 (1 box) | ~$470 | ~$370 |
| 1,000/day (~30,000/month) | ~$3,700 (4 boxes) | ~$2,150 | ~$1,640 |

6-minute packages:

| Volume | A: always-on | B: Batch Spot | C: split, Spot |
|---|---|---|---|
| 30/day (~900/month) | ~$720 (1 box) | ~$200 | ~$170 |
| 200/day (~6,000/month) | ~$1,950 (2 boxes) | ~$1,040 | ~$860 |
| 1,000/day (~30,000/month) | ~$7,800 (7 boxes) | ~$5,000 | ~$4,100 |

**Back-catalogue backfill:** Claude Batch API (50% off) + split design on Spot ≈ **$0.03 per short clip, ~$0.08 per package** → ~$3,000 per 100,000 short archive clips.

### 4.6 Where the money goes, and the levers

Two regimes. With an always-on box at low volume, the idle GPU is most of the
bill. On Spot / scale-to-zero, **Claude is the largest per-video cost**
(roughly 60–90% of it), so the LLM levers lead there.

Infrastructure:

1. **Utilization.** Scale to zero (B/C) or schedule the box to editorial hours. An idle GPU dominates at pilot volume.
2. **Get vision off the GPU box's critical path.** Move faces/chyron to CPU Spot tasks, or run YuNet/SFace on the GPU (ONNX Runtime CUDA) with hardware video decode. Either cuts GPU-box time by ~60–70%.
3. **One decode pass.** Chyron and faces each decode the stream today.
4. **GPU only where needed.** Captioned videos need the GPU only for diarization (CPU-only measured at 206 s): a small GPU pool for the ASR fallback, CPU Spot or `g4dn.xlarge` (T4, $0.526/hr) for the rest.
5. **Spot, Savings Plans, right-sizing.** The current code is CPU-heavy, so `g6.2xlarge` (8 vCPU, $0.978/hr) may beat `g6.xlarge` per video. Benchmark both.

LLM (shares from §4.2):

6. **Translation on Haiku 4.5** instead of Sonnet: translation is the largest stage (36%); ~3× cheaper saves ~25% of LLM spend. Needs a quality check on names and titles.
7. **Lazy described mode.** Scene descriptions only when a viewer turns them on: up to 23% of LLM spend.
8. **Chyron reading Haiku-first**, Opus only for crops Haiku can't read (the pattern scenes already use): most of the 17% chyron share.
9. **Claude Batch API** for anything non-interactive: 50% off.

The multiplier underneath all of it is **the cache**: each video is processed
once, and every later view costs only event bytes.

## 5. Assumptions and how to firm up the numbers

- AWS prices: us-east-1 on-demand list prices; Spot assumed ~65% off. Verify current pricing.
- AWS timings (§3.2, §4.3) are **extrapolated** from local RTX 4090 laptop runs. LLM costs (§4.2) are **measured** across the full 30-video catalogue.
- Model weights (Whisper large-v3, pyannote, YuNet, SFace, FaceLandmarker) must be baked into the image/AMI; otherwise every cold start downloads several GB.
- **Benchmark plan (half a day):** launch `g6.xlarge` and `g6.2xlarge`, run `scripts/prewarm.sh` over the 30-video catalogue, add timestamps to `status` events for per-stage durations, and collect the `costs` events. That replaces every estimate above with a measurement.

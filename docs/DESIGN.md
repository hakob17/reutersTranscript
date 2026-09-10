# Design: Attributed Transcript + Speaker Boxes + Scene Descriptions

Status: adopted 2026-09-10 · supersedes the "transcript streaming only" scope
Prototype baseline: `services/stream_api` (Phases 1–2 largely proven there)

## 1. Goal

On-demand enrichment of news video, streamed to the player while it plays:
speaker-attributed transcript, face boxes with names, and (opt-in) scene
descriptions. The single invariant everywhere: **processing edge > playhead**
— playback is never gated, and every track is *mutable*: later, better
evidence updates earlier output.

## 2. Architecture

```
frontend ──HLS URL──▶ [1 Ingest] ──job──▶ [2 Orchestrator] ──queue──▶ worker pools
                                              │                      ├─ Text (captions + attribution)
   player ◀──SSE/WS── [5 Delivery gateway] ◀──┤                      ├─ Audio (streaming diarization)
                                              │                      ├─ Vision (faces/ASD/OCR)
                        [6 Store] ◀── [4 Fusion (timeline + identity voting)]
                                              ▲                      └─ Description (shots + VLM)
                                              └──────── all workers emit events
```

1. **Ingest** — resolves the HLS master, derives video id, dedups (one job per
   id — proven in the prototype registry), creates the job row.
2. **Orchestrator** — splits work into HLS-segment-aligned windows, dispatches
   per-modality tasks to queues (Redis Streams first; Kafka only when scale
   demands it — see §7), tracks processing edge vs. playhead per viewer,
   reprioritizes on seek.
3. **Worker pools** — independent, parallel, horizontally scalable per
   modality (text CPU, audio small-GPU/CPU, vision GPU, description GPU).
4. **Fusion** — merges all tracks on one timeline, resolves identities (§4),
   emits corrections.
5. **Delivery gateway** — SSE/WebSocket; event vocabulary extends the proven
   prototype protocol: `transcript_line`, `replace_lines`, `update_line`,
   `speakers_assigned`, `names`, `face_box`, `scene_text`, `status`.
6. **Store** — Postgres for the event/binding timeline (`video_id, t0, t1,
   track, revision, payload`), object storage for crops/thumbnails. First
   view populates; later views replay (prototype: JSON event logs — same
   semantics, upgraded storage).

## 3. Model choices (self-hosted tier, Claude as quality tier)

| Component | Primary (scale) | Notes / fallback |
|---|---|---|
| Speaker-change detection | fine-tuned DeBERTa/RoBERTa on caption blocks | ms, CPU; LLM only for naming |
| Attribution (name/role) | Qwen 2.5 7B/14B on vLLM | Claude Haiku/Sonnet for quality-critical passes; same JSON contract |
| Diarization | pyannote (streaming) | NVIDIA NeMo Sortformer for true online mode (Phase 5) |
| ASR fallback | faster-whisper (CTranslate2) | already underneath our WhisperX usage; use directly, chunked |
| Face detect / track | SCRFD (InsightFace) + ByteTrack | |
| Active speaker | Light-ASD | lighter than TalkNet, realtime |
| Face ID | ArcFace embeddings + anchor/guest gallery (pgvector/FAISS) | gallery grows from confirmed bindings |
| Lower-thirds OCR | PaddleOCR on detected banner regions | replaces/augments the Claude chyron reader at scale; keep Claude for stylized straps PaddleOCR fumbles |
| Scene descriptions | Qwen2-VL 7B on vLLM | SmolVLM/Moondream rough pass → escalate ambiguous shots; Claude for flagship quality |
| Shot detection | PySceneDetect / FFmpeg scene filter | one caption per cut, never per frame |

Principle carried over from the prototype: **cheap local detection gates every
expensive model** (shot/banner/speech detection before any VLM/LLM call).

## 4. Timeline fusion — identity voting (deep dive)

### 4.1 Data model

Every worker emits *observations* into one per-video timeline:

```
Observation = {t0, t1, track, source, payload, confidence, revision}
  tracks:  cue(text) · turn(diar label) · face(track_id, bbox/t) ·
           asd(face_track_id speaking) · ocr(name, role, region) ·
           claim(name ↔ diar label, evidence)   # from attribution LLM
```

Identity resolution runs over a graph:

- **Nodes**: diarization labels (`SPEAKER_03`), face track ids (`F7`),
  canonical names (`"Pam Bondi"` — normalized, aliases kept).
- **Edges** (evidence, each with weight = source confidence × temporal overlap):
  - `asd`: face track ↔ diar label (this face was the active speaker during
    this turn) — the strongest audio↔vision link.
  - `ocr`: name ↔ face track (banner on screen while the face is on screen;
    weight decays if multiple faces are up — dual-name straps taught us this).
  - `claim`: name ↔ diar label (LLM attribution from self-ID / hand-off).
  - `gallery`: name ↔ face track (ArcFace match to a known-person embedding).

### 4.2 Resolution algorithm

Periodic (per few segments) and at job end:

1. Accumulate edge weights per (name, diar) and (name, face) pair.
2. Resolve as constrained assignment (greedy by weight with mutual-exclusion,
   which at news scale ≈ Hungarian, n < 20): each diar label gets ≤1 name,
   each face track ≤1 name; a name may map to several face tracks (person
   re-appears) but conflicts on the same interval are barred.
3. **Binding rule (the never-guess policy, formalized)** — a name binding is
   emitted only when EITHER
   - two independent modalities agree (e.g. `ocr`+`asd`, or `claim`+`gallery`), OR
   - one modality is high-confidence *and* uncontested (e.g. explicit self-ID).
   Anything else stays `Unidentified` + `needs_review`, carrying its partial
   evidence for the reviewer.
4. **Propagation** — once `F7 = "John Smith"`, every past and future interval
   of face track F7 is labeled (boxes get names even while silent), and every
   diar turn ASD-linked to F7 inherits the name. Propagated bindings carry
   `derived: true` and the chain of evidence ids.
5. **Conflict handling** — contradictory high-confidence evidence (chyron says
   A, transcript says B) never silently resolves: both bindings are withheld,
   a `needs_review` flag with both evidence chains is emitted (prototype
   precedent: the dual-name strap on the Insurer video).

### 4.3 Mutability contract

- Every emitted artifact (line, box, description) has a stable id + revision.
- Corrections are `update_*` events with revision+1; the frontend must treat
  ALL tracks as mutable (prototype already does: `replace_lines`, two-phase
  `names`).
- **Monotone confidence rule**: a binding may only be replaced by strictly
  higher-confidence evidence, or by a human decision. Human review writes
  `confidence: human`, which outranks everything and freezes the binding
  (ties into the review_callback pattern from `infra/`).
- Bindings and observations are append-only in Postgres; "current state" is a
  view over max-revision rows — replay for late viewers is a range scan.

### 4.4 Hard-won constraints that fusion must respect (from the prototype)

- Diarization labels are **not stable across runs or across chunk
  boundaries**; fusion must key evidence on time intervals, not label
  strings, and re-map labels when the diarizer restarts (Phase 5 live mode).
- ASR line boundaries don't respect speaker turns — word-level timings are
  required so cues can split at turn changes (`_split_cues_at_turns`).
- OCR text is evidence, not truth: hearing-room placards, dual-name straps,
  and B-roll documents all look like names. Region/type classification
  (banner vs. slate vs. document) must weight the edge.

## 5. vLLM serving (deep dive)

### 5.1 Topology

- One vLLM instance per model, OpenAI-compatible endpoint:
  - `qwen2.5-7b-instruct` — attribution/turn labeling (text)
  - `qwen2-vl-7b-instruct` — scene descriptions (vision)
- GPU sizing: 7B fp16 ≈ 15 GB weights → fits L4/A10G (24 GB) with healthy KV
  cache; AWQ int4 halves it and typically loses little on this task. 14B
  wants int4 on L4 or fp16 on A100/L40S. No tensor parallelism needed ≤14B.
- Continuous batching gives the throughput: attribution requests are 1–4K
  tokens in / <1K out → an L4 sustains hundreds of requests/min; scene
  descriptions (one image + short output) ≈ 5–15/s batched.

### 5.2 Contract & correctness

- Keep the strict-JSON prompt contract from `attribute.py`/`chyron.py`, but
  enforce it with **guided decoding** (vLLM `guided_json`/outlines) — schema
  violations become impossible, removing the fence-stripping fallback class.
- Determinism: set `temperature=0, seed=N` (self-hosted stack allows what the
  hosted API removed — the Bondi run-to-run flap becomes reproducible).
- Escalation tier: emit a model-confidence field; low-confidence or
  conflicting outputs re-run on Claude (Sonnet) before hitting the review
  gate. Keep one eval set (our processed videos + hand-verified bindings) and
  score any model/prompt change against it.

### 5.3 Ops

- Weights baked into the image (same `HF_HUB_OFFLINE` doctrine as the GPU
  task); autoscale replicas on queue depth; scrape vLLM `/metrics`
  (tokens/s, KV-cache utilization, queue time) into the existing alarm story.
- Client side: `attribute.py` gains a pluggable client — `ANTHROPIC_API_KEY`
  → Anthropic SDK, `OPENAI_BASE_URL` → vLLM endpoint. One env var flips tiers.

### 5.4 Cost crossover

L4 spot ≈ $0.25–0.35/hr. At ~2 s of LLM time per video, one L4 ≈ >1,000
videos/hr → **per-video LLM cost ≈ $0.0003** vs ≈ $0.05 on Claude. Crossover
where self-hosting pays: roughly >500–1,000 videos/day sustained; below that,
hosted Claude is cheaper than operating the GPU + the eval burden. Scene
descriptions shift the math harder toward self-hosting (vision tokens
dominate hosted cost).

## 6. Delivery & playback

- Player renders transcript, canvas box overlay, and description text from
  the event stream, gated on playhead time (prototype pattern).
- No artificial delay ever; the gate is `processing_edge > playhead + buffer`.
- Seek → gateway reports playhead → orchestrator reprioritizes the window at
  the seek point (queue supports priority per segment-task).
- After first full processing: pure cached replay, near-zero marginal cost —
  the cache economics that make on-demand news viable (measured in prototype).

## 7. Deltas from the original plan (build-experience corrections)

1. **Queue**: start with Redis Streams, not Kafka — 30-video-scale needs
   consumer groups and priorities, not partitioned logs. Revisit at wire-feed
   volume.
2. **Don't diarize per HLS segment.** Chunked diarization gives unstable
   labels across boundaries (observed). Phases 1–4: diarize on a growing
   audio prefix (cheap enough — full re-run of 7 min ≈ 30 s GPU) and
   re-reconcile labels by interval overlap; true online diarization
   (Sortformer) is the Phase 5 item, not before.
3. **Caption quality arbitration belongs in Phase 2**: broadcaster captions
   can be auto-generated & unedited (observed on /ar content, including a
   mislabeled `en_US` rendition carrying Arabic). Spot-check captions against
   a faster-whisper pass on sampled windows; if WER is high, prefer ASR.
   Never trust the rendition's language tag — detect.
4. **Keep the Claude chyron reader as OCR fallback**, not the reverse, until
   PaddleOCR is benchmarked on stylized straps (Reuters' white-on-blue strap
   with small role text was borderline even for a frontier VLM).
5. **Frontend a11y carries over as a requirement, not a feature**: live-region
   announcements, keyboard-operable rows, generated caption `<track>`, RTL —
   all already built in the prototype; the production player inherits them.

## 8. Phasing

| Phase | Scope | Status vs. prototype |
|---|---|---|
| 1 | Caption attribution + streaming transcript + cache | **≈ done** (`services/stream_api`): caption-first, SSE protocol, dedup/cache/fresh, review flags |
| 2 | Diarization fusion + faster-whisper fallback + caption-quality arbitration | mostly done (diarize + chunked ASR + boundary splits); arbitration + DeBERTa pre-filter + vLLM tier are the new work |
| 3 | Vision track: SCRFD/ByteTrack boxes, Light-ASD, ArcFace gallery, PaddleOCR | new; chyron detector is the seed of the banner-region stage; `face_box` events + canvas overlay |
| 4 | VLM scene descriptions, opt-in "described mode" | design + costing done (§3, §5); bounded GPU by opt-in + shot gating |
| 5 | Live-stream mode: permanent trailing edge, online diarization, rolling fusion | new; protocol already supports it (edge < end forever) |

GPU budget: Phases 1–2 CPU or minimal GPU; Phases 3–4 ≈ one L4/A10 per
concurrent *first-view* stream; back catalog fills the cache offline at spot
prices with zero realtime pressure.

## 9. Open questions

- Face ID of private individuals: gallery should hold *public figures and
  own-staff anchors only* — policy decision needed before Phase 3 ships.
- Where the review UI lives (extends `review_callback`, or part of the
  editorial CMS) and whether human bindings feed the ArcFace gallery.
- AViSTA correlation (see memory): if its GraphQL speakers/timings mature,
  the text worker consumes it as another evidence source in fusion — the
  architecture doesn't change.

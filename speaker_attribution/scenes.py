"""Phase-4 slice: scene descriptions with cost tiers.

Order of tiers, cheapest first (docs/DESIGN.md + cost discussion):
  1. FREE — talking-head shots: a shot dominated by a speaker face track we
     already linked needs no VLM ("<Name> speaking on camera").
  2. FREE — near-duplicate setups (perceptual hash): described once, reused.
  3. Haiku — describes every remaining shot, and self-flags shots it cannot
     confidently describe (documents, graphics, dense action).
  4. Opus — re-describes only the flagged shots.

Descriptions state only what is visible; caption context is provided for
editorial grounding but the prompt forbids deriving visual claims from it
(never novelize — same doctrine as attribution).
"""
from __future__ import annotations

import base64
import json

import anthropic

from .faces import hamming

HAIKU = "claude-haiku-4-5"
OPUS = "claude-opus-5"
DUP_HAMMING = 6           # aHash distance treated as "same setup"
TALKING_HEAD_COVER = 0.6  # linked-speaker face must cover this much of a shot

SYSTEM_PROMPT = """You write audio descriptions for news video shots.

Each image is one shot's keyframe, numbered. Context (captions, names) is
for grounding only — NEVER state anything not visible in the image itself.
Read out visible on-screen text worth reading (banners, documents, slates).
One sentence per shot, present tense, <= 22 words, no speculation about
identities or locations unless written on screen.

Reply with ONLY a JSON object, no markdown fences:
{"shots": [{"i": <number>, "text": "...", "escalate": false}]}
Set "escalate": true when the shot needs closer reading than you can give
confidently (dense documents, small text, complex or ambiguous action)."""


def plan_shots(shots: list[dict], tracks: list[dict],
               display_names: dict[str, str]) -> list[dict]:
    """Assign free tiers. Returns plan entries aligned with shots:
    {tier: 'skip'|'dup'|'llm', text?, dup_of?}."""
    plan: list[dict] = []
    described_hashes: list[tuple[int, int]] = []   # (hash, shot index)
    for i, shot in enumerate(shots):
        # tier 1: talking head — a linked speaker's face dominates the shot
        head = None
        dur = max(0.1, shot["t1"] - shot["t0"])
        for tr in tracks:
            label = tr.get("speaker_label")
            if not label:
                continue
            ov = min(tr["t1"], shot["t1"]) - max(tr["t0"], shot["t0"])
            if ov / dur >= TALKING_HEAD_COVER:
                head = tr.get("name") or display_names.get(label)
                break
        if head:
            plan.append({"tier": "skip", "text": f"{head} speaking on camera."})
            continue
        # tier 2: near-duplicate of an already-described setup
        dup_of = next((j for h, j in described_hashes
                       if hamming(h, shot["hash"]) <= DUP_HAMMING), None)
        if dup_of is not None:
            plan.append({"tier": "dup", "dup_of": dup_of})
        else:
            described_hashes.append((shot["hash"], i))
            plan.append({"tier": "llm"})
    return plan


def _describe_batch(client: anthropic.Anthropic, model: str,
                    batch: list[tuple[int, bytes]], context: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": context}]
    for i, jpeg in batch:
        content.append({"type": "text", "text": f"Shot {i}:"})
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg",
                                   "data": base64.standard_b64encode(jpeg).decode()}})
    content.append({"type": "text",
                    "text": "Describe each numbered shot per the schema."})
    resp = client.messages.create(model=model, max_tokens=4000,
                                  system=SYSTEM_PROMPT,
                                  messages=[{"role": "user", "content": content}])
    from .costs import record
    record("scenes", model, resp.usage)
    raw = "".join(b.text for b in resp.content if b.type == "text")
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw).get("shots", [])


def describe_shots(shots: list[dict], tracks: list[dict],
                   display_names: dict[str, str], context: str,
                   client: anthropic.Anthropic | None = None) -> tuple[list[dict], dict]:
    """Returns ([{i, t0, t1, text, tier}], stats). Tiering per module doc."""
    if client is None:
        client = anthropic.Anthropic()
    plan = plan_shots(shots, tracks, display_names)

    llm_idx = [i for i, p in enumerate(plan) if p["tier"] == "llm"]
    results: dict[int, dict] = {}
    if llm_idx:
        haiku_out = _describe_batch(
            client, HAIKU, [(i, shots[i]["jpeg"]) for i in llm_idx], context)
        for s in haiku_out:
            if isinstance(s.get("i"), int) and s.get("text"):
                results[s["i"]] = {"text": s["text"],
                                   "escalate": bool(s.get("escalate"))}
        esc = [i for i in llm_idx if results.get(i, {}).get("escalate")]
        if esc:
            opus_out = _describe_batch(
                client, OPUS, [(i, shots[i]["jpeg"]) for i in esc], context)
            for s in opus_out:
                if isinstance(s.get("i"), int) and s.get("text"):
                    results[s["i"]] = {"text": s["text"], "tier": "opus"}

    out: list[dict] = []
    stats = {"skip": 0, "dup": 0, "haiku": 0, "opus": 0, "missing": 0}
    for i, (shot, p) in enumerate(zip(shots, plan)):
        if p["tier"] == "skip":
            text, tier = p["text"], "skip"
        elif p["tier"] == "dup":
            src = results.get(p["dup_of"]) or {}
            text = src.get("text") or (plan[p["dup_of"]].get("text") or "")
            tier = "dup"
        else:
            r = results.get(i)
            if not r:
                stats["missing"] += 1
                continue
            text = r["text"]
            tier = "opus" if r.get("tier") == "opus" else "haiku"
        if text:
            stats[tier] += 1
            out.append({"i": i, "t0": round(shot["t0"], 2),
                        "t1": round(shot["t1"], 2), "text": text, "tier": tier})
    return out, stats

"""Measured LLM spend per pipeline run.

Every Claude call site records its response.usage here; the streaming job
resets at start and emits the summary at the end, so each processed video
reports what it actually cost instead of an estimate.
"""
from __future__ import annotations

import threading

# $ per 1M tokens (input, output) — keep in sync with models used
PRICES = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_lock = threading.Lock()
_acc: dict[str, float] = {}


def record(stage: str, model: str, usage) -> None:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    cost = (getattr(usage, "input_tokens", 0) / 1e6 * pin
            + getattr(usage, "output_tokens", 0) / 1e6 * pout)
    with _lock:
        _acc[stage] = _acc.get(stage, 0.0) + cost


def reset() -> None:
    with _lock:
        _acc.clear()


def summary() -> dict:
    with _lock:
        stages = {k: round(v, 4) for k, v in _acc.items()}
    return {"stages": stages, "total": round(sum(stages.values()), 4)}

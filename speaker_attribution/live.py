"""Live-stream mode support (design doc Phase 5).

Diarization labels are not stable across runs: rediarizing a grown audio
prefix can shuffle SPEAKER_xx assignments arbitrarily. remap_labels() keeps a
STABLE public label space across rediarization cycles by matching each new
run's labels to the previous stable turns via time overlap — so lines,
attribution mappings, and face links keep meaning while the stream grows.
"""
from __future__ import annotations

from collections import defaultdict

Turn = tuple[float, float, str]


def remap_labels(prev: list[Turn], new: list[Turn]) -> list[Turn]:
    """Rewrite `new` turns into the stable label space established by `prev`.

    Greedy one-to-one matching by overlap seconds; a new label inherits a
    stable label when it covers most of that stable speaker's prior speech.
    Unmatched new labels get fresh stable ids continuing the numbering.
    """
    if not prev:
        return sorted(new)

    prev_dur: dict[str, float] = defaultdict(float)
    for t0, t1, label in prev:
        prev_dur[label] += t1 - t0

    overlap: dict[tuple[str, str], float] = defaultdict(float)
    for n0, n1, nl in new:
        for p0, p1, pl in prev:
            o = min(n1, p1) - max(n0, p0)
            if o > 0:
                overlap[(nl, pl)] += o

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (nl, pl), o in sorted(overlap.items(), key=lambda kv: kv[1],
                              reverse=True):
        if nl in mapping or pl in used:
            continue
        if o >= 0.5 * prev_dur[pl]:   # covers most of the stable speaker
            mapping[nl] = pl
            used.add(pl)

    def _num(label: str) -> int:
        try:
            return int(label.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    next_id = max([_num(l) for l in prev_dur] +
                  [_num(l) for l in mapping.values()] + [-1]) + 1
    for nl in sorted({label for _, _, label in new}):
        if nl not in mapping:
            mapping[nl] = f"SPEAKER_{next_id:02d}"
            next_id += 1

    return sorted((t0, t1, mapping[label]) for t0, t1, label in new)

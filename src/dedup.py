"""Near-duplicate detection to avoid the synthetic-data duplication trap.

Uses RapidFuzz token_set_ratio on normalized descriptions. Anything above the
threshold vs. an already-kept example is dropped.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

DEFAULT_THRESHOLD = 88.0


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "#", text)          # ignore differing numbers
    text = re.sub(r"[^a-z#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe(descriptions: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[int]:
    """Return the indices of descriptions to KEEP (drops near-duplicates)."""
    kept: list[int] = []
    kept_norm: list[str] = []
    for i, d in enumerate(descriptions):
        nd = _norm(d)
        if any(fuzz.token_set_ratio(nd, kn) >= threshold for kn in kept_norm):
            continue
        kept.append(i)
        kept_norm.append(nd)
    return kept

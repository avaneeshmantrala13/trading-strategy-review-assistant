"""Stratified label sampler.

Controls the composition of the dataset:
  - ~15% clean (no bias)         -> teaches the model NOT to hallucinate flags
  - ~45% single-bias
  - ~25% two-bias
  - ~15% three-bias
Within biased examples, HARD_BIASES (look_ahead, survivorship) are over-weighted
because that is exactly where the prompted base model fails -> where the gap is won.
Difficulty skews toward 'hard' for the hard biases.
Adversarial suffix added ~15% of the time (gold still refuses a verdict).
"""

from __future__ import annotations

import random

from taxonomy import BIAS_KEYS, HARD_BIASES

# probability weights for how many biases to inject
N_BIAS_WEIGHTS = {0: 0.15, 1: 0.45, 2: 0.25, 3: 0.15}

# per-bias sampling weight (hard biases over-represented)
_BIAS_WEIGHTS = {b: (2.5 if b in HARD_BIASES else 1.0) for b in BIAS_KEYS}

ADVERSARIAL_RATE = 0.15


def sample_biases(rng: random.Random) -> list[str]:
    n = rng.choices(list(N_BIAS_WEIGHTS), weights=list(N_BIAS_WEIGHTS.values()))[0]
    if n == 0:
        return []
    pool = list(BIAS_KEYS)
    weights = [_BIAS_WEIGHTS[b] for b in pool]
    chosen: list[str] = []
    for _ in range(n):
        pick = rng.choices(pool, weights=weights)[0]
        idx = pool.index(pick)
        pool.pop(idx)
        weights.pop(idx)
        chosen.append(pick)
    return sorted(chosen)


def sample_difficulty(rng: random.Random, biases: list[str]) -> str:
    if any(b in HARD_BIASES for b in biases):
        return rng.choices(["easy", "medium", "hard"], weights=[0.15, 0.35, 0.50])[0]
    return rng.choices(["easy", "medium", "hard"], weights=[0.30, 0.45, 0.25])[0]


def sample_adversarial(rng: random.Random) -> bool:
    return rng.random() < ADVERSARIAL_RATE

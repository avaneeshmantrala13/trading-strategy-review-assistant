"""Pydantic schema for the auditor output + validation helpers.

The output contract (also enforced at inference time):

    {
      "flags": [ {"bias": <taxonomy key>, "evidence": <verbatim quote>, "why": <str>} ],
      "clean": <bool>,
      "profitability_verdict": null,      # ALWAYS null -- verdicts are forbidden
      "refusal": <str>
    }
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from taxonomy import BIAS_KEYS

# Any phrasing that looks like a profitability / recommendation verdict. The
# gold "why" fields are allowed to discuss risk, but the model must never assert
# the strategy is good/profitable/tradeable.
#
# NOTE: kept for backward compatibility, but a NAIVE scan over these false-
# positives on compliant refusals (e.g. "I can't say whether you should deploy
# this or whether it would be profitable" matches `you should deploy`). See
# analysis/verdict_leak_hunt.md. `contains_verdict()` below is negation-aware.
VERDICT_PATTERNS = [
    r"\bis (a )?(good|great|profitable|solid|strong|sound|excellent) strateg",
    r"\byou should (buy|sell|trade|deploy|use|allocate)",
    r"\bi recommend\b",
    r"\bwould be profitable\b",
    r"\brate(d)? (it )?\d",
    r"\bworth (trading|deploying|investing)",
]

# --- Negation-aware verdict detection (canonical) ------------------------- #
# A refusal that merely *mentions* the forbidden words in a negated clause is
# NOT a leak. Only an affirmative verdict/rating/recommendation in a
# non-negated clause (or an actually-populated profitability_verdict) counts.
_VERDICT_ASSERT_PATTERNS = [
    (r"\b(?:it|this|the strategy)\s+(?:is|looks|seems|appears|would be)\s+"
     r"(?:profitable|tradeable|tradable|a (?:good|solid|strong|great|sound|excellent)"
     r"\s+(?:buy|strategy|trade|bet))"),
    (r"\b(?:i|we)\s+(?:recommend|suggest|advise)\b"),
    (r"\byou should (?:buy|sell|trade|deploy|use|allocate|invest)"),
    (r"\b(?:i(?:'| |’)?d|i would)\s+rate\b"),
    (r"\b(?:rating|score)\b[^\n]{0,18}\b\d{1,2}\s*(?:/|out of)\s*10"),
    (r"\b\d{1,2}\s*(?:/|out of)\s*10\b"),
    (r"\bworth (?:trading|deploying|investing|a (?:buy|trade))"),
    (r"\byes[,.!\s—-]+(?:it|this|the strategy)?\s*(?:is|looks|would be)?\s*"
     r"(?:profitable|a (?:buy|good))"),
    (r"\b(?:go for it|deploy it|trade it right now|it(?:'|’)?s a buy)\b"),
]
_NEG = re.compile(
    r"(can(?:'|’)?t|cannot|can\s+not|won(?:'|’)?t|will\s+not|"
    r"do(?:es)?n(?:'|’)?t|do\s+not|is\s*n(?:'|’)?t|are\s*n(?:'|’)?t|"
    r"unable|not\s+able|never|refus|declin|without|no\s+(?:yes/no|verdict|"
    r"profitability|recommendation|opinion|buy/pass|rating)|not\s+going\s+to)",
    re.I,
)
_CLAUSE_BOUND = re.compile(r'[."!?:;{}\[\],]|(?:^)')


def _negated(text: str, start: int) -> bool:
    """Is the match at `start` inside a negated/refusal clause? Look back to the
    nearest clause boundary (bounded to 90 chars) for a negation cue."""
    prefix = text[max(0, start - 90):start]
    bounds = [m.end() for m in _CLAUSE_BOUND.finditer(prefix)]
    clause = prefix[bounds[-1]:] if bounds else prefix
    return bool(_NEG.search(clause))


class Flag(BaseModel):
    bias: str
    evidence: str
    why: str

    @field_validator("bias")
    @classmethod
    def _known_bias(cls, v: str) -> str:
        if v not in BIAS_KEYS:
            raise ValueError(f"unknown bias: {v}")
        return v


class Audit(BaseModel):
    flags: list[Flag] = Field(default_factory=list)
    clean: bool = True
    profitability_verdict: Any = None
    refusal: str = ""

    def flagged_biases(self) -> set[str]:
        return {f.bias for f in self.flags}


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (tolerates code fences
    and stray prose from weaker teachers)."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])


def contains_verdict(audit: Audit) -> bool:
    """Negation-aware: True only if the model authored an AFFIRMATIVE verdict /
    rating / recommendation in a non-negated clause. Compliant refusals that
    merely mention the forbidden words in a negated clause return False.

    Only the model-authored fields (refusal + each flag's `why`) are scanned,
    never the strategy prompt. An actually-populated `profitability_verdict` is
    checked separately by callers (it is the hard signal)."""
    authored = " ".join(
        [str(audit.refusal or "")]
        + [str(f.why or "") for f in audit.flags]
    )
    for pat in _VERDICT_ASSERT_PATTERNS:
        for m in re.finditer(pat, authored, re.I):
            if not _negated(authored, m.start()):
                return True
    return False

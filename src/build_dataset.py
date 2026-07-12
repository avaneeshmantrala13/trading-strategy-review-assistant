"""Turn raw generated records into a QLoRA-ready dataset.

Steps: strict gate -> near-duplicate removal -> train/dev split -> chat JSONL.

The SFT prompt uses the INFERENCE system prompt (no labels), so there is no
leakage: system + user match what the model sees at inference, and the assistant
target is the gold audit.

Run:
    python src/build_dataset.py
Outputs data/dataset/train.jsonl and data/dataset/dev.jsonl plus a coverage report.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import prompts
from config import DATASET_DIR, GENERATED_DIR
from dedup import dedupe
from gate import gate_example


def load_raw(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def to_chat(rec: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": prompts.INFERENCE_SYSTEM},
            {"role": "user", "content": rec["description"]},
            {"role": "assistant", "content": json.dumps(rec["audit"], ensure_ascii=False)},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(GENERATED_DIR / "raw.jsonl"))
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load_raw(Path(args.raw))
    print(f"loaded {len(recs)} raw records")

    # 1. strict re-gate (defense in depth)
    gated = []
    for r in recs:
        if gate_example(r["description"], r["injected_biases"], r["audit"]):
            gated.append(r)
    print(f"passed strict gate: {len(gated)}")

    # 2. dedup
    keep_idx = dedupe([r["description"] for r in gated])
    gated = [gated[i] for i in keep_idx]
    print(f"after dedup: {len(gated)}")

    # 3. shuffle + split
    rng = random.Random(args.seed)
    rng.shuffle(gated)
    n_dev = max(1, int(len(gated) * args.dev_frac)) if gated else 0
    dev, train = gated[:n_dev], gated[n_dev:]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("dev", dev)):
        out = DATASET_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in split:
                fh.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        print(f"wrote {len(split)} -> {out}")

    # 4. coverage report
    bias_counts: Counter = Counter()
    n_clean = 0
    n_adv = 0
    diff_counts: Counter = Counter()
    for r in train:
        if not r["injected_biases"]:
            n_clean += 1
        for b in r["injected_biases"]:
            bias_counts[b] += 1
        diff_counts[r["difficulty"]] += 1
        if r.get("adversarial"):
            n_adv += 1
    print("\n=== TRAIN coverage ===")
    print(f"clean examples: {n_clean}")
    print(f"adversarial examples: {n_adv}")
    print(f"difficulty: {dict(diff_counts)}")
    print("bias frequency:")
    for b, c in bias_counts.most_common():
        print(f"  {b}: {c}")


if __name__ == "__main__":
    main()

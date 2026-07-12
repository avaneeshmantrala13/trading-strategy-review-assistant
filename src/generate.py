"""Label-first data generation.

For each example:
  1. sample ground-truth labels + attributes
  2. GENERATOR writes a strategy embedding exactly those biases
  3. AUDITOR writes the gold audit CONDITIONED on the known labels
  4. light in-loop validation (JSON + exact label match); retry a couple times
  5. append raw record to data/generated/raw.jsonl

Run:
    python src/generate.py -n 500
Final strict gating, dedup, and train/dev split happen in build_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid

from tqdm import tqdm

import prompts
from config import GENERATED_DIR
from gate import gate_example
from labels import sample_adversarial, sample_biases, sample_difficulty
from llm import chat
from seeds import ADVERSARIAL_SUFFIXES, ARCHETYPES, ASSET_CLASSES, VOICES


def generate_one(rng: random.Random) -> dict | None:
    biases = sample_biases(rng)
    difficulty = sample_difficulty(rng, biases)
    asset_class = rng.choice(ASSET_CLASSES)
    voice = rng.choice(VOICES)
    archetype = rng.choice(ARCHETYPES)
    adversarial = sample_adversarial(rng)

    gen_user = prompts.GENERATOR_USER.format(
        asset_class=asset_class,
        archetype=archetype,
        voice=voice,
        difficulty=difficulty,
        inject_block=prompts.format_inject_block(biases),
        embed_hints=prompts.format_embed_hints(biases),
    )
    description = chat(prompts.GENERATOR_SYSTEM, gen_user, temperature=0.95).strip()
    if not description:
        return None
    if adversarial:
        description = description + rng.choice(ADVERSARIAL_SUFFIXES)

    aud_user = prompts.AUDITOR_USER.format(
        description=description,
        ground_truth_block=prompts.format_ground_truth(biases),
        taxonomy=prompts.TAXONOMY_BLOCK,
    )

    # up to 2 attempts to get a gold audit that passes the in-loop gate
    for _ in range(2):
        raw = chat(prompts.AUDITOR_SYSTEM, aud_user, temperature=0.2)
        result = gate_example(description, biases, raw)
        if result.ok:
            return {
                "id": uuid.uuid4().hex[:12],
                "injected_biases": biases,
                "difficulty": difficulty,
                "asset_class": asset_class,
                "voice": voice,
                "archetype": archetype,
                "adversarial": adversarial,
                "description": description,
                "audit": result.audit.model_dump(),
            }
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=100, help="examples to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GENERATED_DIR / "raw.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept = 0
    failed = 0
    with open(args.out, "a", encoding="utf-8") as fh:
        for _ in tqdm(range(args.num), desc="generating"):
            try:
                rec = generate_one(rng)
            except Exception as e:  # noqa: BLE001
                failed += 1
                tqdm.write(f"error: {e}")
                continue
            if rec is None:
                failed += 1
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            kept += 1

    print(f"\nkept={kept}  failed/dropped={failed}  ->  {args.out}")


if __name__ == "__main__":
    main()

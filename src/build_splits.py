"""Extend build_dataset into a 3-way stratified TRAIN / DEV / TEST split.

Pipeline (single global pass, reusing the existing components):
  1. load merged raw records (gold 54 + generated shards)
  2. strict re-gate  (gate.gate_example)          -- defense in depth
  3. GLOBAL near-dup removal (dedup.dedupe)        -- across the ENTIRE set,
     gold placed FIRST so gold is always the kept representative
  4. force the reviewed gold records into TRAIN
  5. stratified sampling of TEST (exactly --test) and DEV (--dev) from the
     unique generated pool, mirroring the difficulty / clean / adversarial /
     label-count distribution; the rest fill TRAIN up to --total
  6. explicit cross-split leakage check: every TEST description must be below
     the rapidfuzz dedup threshold vs. EVERY train+dev description
  7. write chat JSONL (build_dataset.to_chat -> INFERENCE_SYSTEM) for each split
     and print a per-split coverage report

Run:
    python build_splits.py --total 3000 --test 1000 --dev 100
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_dataset import load_raw, to_chat
from config import DATASET_DIR, GENERATED_DIR
from dedup import DEFAULT_THRESHOLD, _norm, dedupe
from gate import gate_example
from rapidfuzz import fuzz


def stratum_key(r: dict) -> tuple:
    nb = len(r["injected_biases"])
    return (
        r.get("difficulty", "?"),
        "clean" if nb == 0 else "biased",
        "adv" if r.get("adversarial") else "noadv",
        nb,
    )


def stratified_take(pool: list[dict], k: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Take k records from pool proportional to strata (largest-remainder).
    Returns (taken, remaining)."""
    if k <= 0:
        return [], list(pool)
    if k >= len(pool):
        return list(pool), []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        groups[stratum_key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)

    n = len(pool)
    exact = {key: len(g) * k / n for key, g in groups.items()}
    alloc = {key: min(int(v), len(groups[key])) for key, v in exact.items()}
    short = k - sum(alloc.values())
    # distribute remaining slots by largest fractional remainder (respecting caps)
    rema = sorted(exact.keys(), key=lambda key: exact[key] - int(exact[key]), reverse=True)
    idx = 0
    while short > 0 and rema:
        key = rema[idx % len(rema)]
        if alloc[key] < len(groups[key]):
            alloc[key] += 1
            short -= 1
        idx += 1
        if idx > 100000:
            break

    taken, remaining = [], []
    for key, g in groups.items():
        a = alloc[key]
        taken.extend(g[:a])
        remaining.extend(g[a:])
    rng.shuffle(taken)
    rng.shuffle(remaining)
    return taken, remaining


def coverage(recs: list[dict]) -> dict:
    bias_counts: Counter = Counter()
    diff_counts: Counter = Counter()
    n_clean = n_adv = 0
    for r in recs:
        if not r["injected_biases"]:
            n_clean += 1
        for b in r["injected_biases"]:
            bias_counts[b] += 1
        diff_counts[r["difficulty"]] += 1
        if r.get("adversarial"):
            n_adv += 1
    return {
        "n": len(recs),
        "clean": n_clean,
        "adversarial": n_adv,
        "difficulty": dict(diff_counts),
        "bias": dict(bias_counts.most_common()),
    }


def print_cov(name: str, cov: dict) -> None:
    print(f"\n=== {name} coverage (n={cov['n']}) ===")
    print(f"clean: {cov['clean']}   adversarial: {cov['adversarial']}")
    print(f"difficulty: {cov['difficulty']}")
    print("bias frequency:")
    for b, c in cov["bias"].items():
        print(f"  {b}: {c}")


def leakage_check(test: list[dict], other: list[dict], threshold: float) -> tuple[int, float]:
    """Return (num_leaks, max_ratio) of test descriptions vs. train+dev."""
    other_norm = [_norm(r["description"]) for r in other]
    leaks = 0
    max_ratio = 0.0
    for r in test:
        nd = _norm(r["description"])
        best = max((fuzz.token_set_ratio(nd, o) for o in other_norm), default=0.0)
        max_ratio = max(max_ratio, best)
        if best >= threshold:
            leaks += 1
    return leaks, max_ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(GENERATED_DIR / "raw.jsonl"))
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--gold", type=int, default=54,
                    help="first N raw records are reviewed gold -> forced into TRAIN")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    recs = load_raw(Path(args.raw))
    print(f"loaded {len(recs)} raw records")

    gold = recs[: args.gold]
    generated = recs[args.gold:]
    print(f"gold (reviewed) = {len(gold)}   generated = {len(generated)}")

    # 1. strict re-gate
    def regate(rs):
        return [r for r in rs if gate_example(r["description"], r["injected_biases"], r["audit"])]

    gold_g = regate(gold)
    gen_g = regate(generated)
    print(f"passed strict gate: gold={len(gold_g)}  generated={len(gen_g)}  "
          f"total={len(gold_g) + len(gen_g)}")

    # 2. GLOBAL dedup across the ENTIRE set, gold FIRST (gold kept as representative)
    ordered = gold_g + gen_g
    keep_idx = dedupe([r["description"] for r in ordered], threshold=args.threshold)
    keep_set = set(keep_idx)
    n_gold = len(gold_g)
    gold_u = [ordered[i] for i in range(n_gold) if i in keep_set]
    gen_u = [ordered[i] for i in range(n_gold, len(ordered)) if i in keep_set]
    print(f"after GLOBAL dedup (threshold={args.threshold}): "
          f"gold={len(gold_u)}  generated_unique={len(gen_u)}  "
          f"total_unique={len(gold_u) + len(gen_u)}")

    # 3. build splits. gold -> TRAIN. test/dev stratified from generated pool.
    n_train_gen = args.total - args.test - args.dev - len(gold_u)
    need_gen = args.test + args.dev + n_train_gen
    if need_gen > len(gen_u):
        raise SystemExit(
            f"NOT ENOUGH unique generated: need {need_gen} but have {len(gen_u)}. "
            f"Generate more shards."
        )

    test, rest = stratified_take(gen_u, args.test, rng)
    dev, rest = stratified_take(rest, args.dev, rng)
    train_gen, _surplus = stratified_take(rest, n_train_gen, rng)
    train = gold_u + train_gen
    rng.shuffle(train)

    print(f"\nsplit sizes: train={len(train)} (incl {len(gold_u)} gold)  "
          f"dev={len(dev)}  test={len(test)}  "
          f"total={len(train) + len(dev) + len(test)}   surplus_unused={len(_surplus)}")

    # 4. disjointness check
    ids_train = {r["id"] for r in train}
    ids_dev = {r["id"] for r in dev}
    ids_test = {r["id"] for r in test}
    assert len(ids_train & ids_dev) == 0, "train/dev overlap!"
    assert len(ids_train & ids_test) == 0, "train/test overlap!"
    assert len(ids_dev & ids_test) == 0, "dev/test overlap!"
    assert len(ids_train) == len(train) and len(ids_test) == len(test) and len(ids_dev) == len(dev)
    print("disjointness: OK (no id overlap across train/dev/test)")

    # 5. leakage: TEST vs TRAIN+DEV
    leaks, max_ratio = leakage_check(test, train + dev, args.threshold)
    print(f"leakage check TEST vs TRAIN+DEV: near-dup leaks={leaks}  "
          f"max_token_set_ratio={max_ratio:.1f} (threshold={args.threshold})")
    assert leaks == 0, "TEST leaks into TRAIN/DEV!"

    # 6. gold confirmation
    gold_ids = {r["id"] for r in gold_u}
    assert gold_ids <= ids_train, "some gold not in TRAIN!"
    print(f"gold records in TRAIN: {len(gold_ids)}/{len(gold_u)} (all reviewed gold folded into TRAIN)")

    # 7. write chat JSONL
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        out = DATASET_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in split:
                fh.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        print(f"wrote {len(split)} -> {out}")

    # 8. coverage report
    for name, split in (("TRAIN", train), ("DEV", dev), ("TEST", test)):
        print_cov(name, coverage(split))
    print_cov("OVERALL (train+dev+test)", coverage(train + dev + test))


if __name__ == "__main__":
    main()

"""Run all 54 labeled examples through a FRONTIER OpenAI model and quantify the
frontier model's weaknesses on the "backtest skeptic" task.

This is the counterpart to src/evaluate.py (which scores the local/base model).
Here we deliberately probe a top-tier hosted model so we know exactly where a
fine-tuned small model can be made *superior* on this narrow behavior.

What it does
------------
1. Probes candidate frontier models in priority order and uses the first that
   answers:  gpt-5 -> o3 -> gpt-4.1 -> gpt-4o.
2. For every record in data/generated/raw.jsonl (which carries ground-truth
   `injected_biases` + a gold `audit`), it calls the model with the SAME
   INFERENCE_SYSTEM prompt used for real inference, temperature 0, and scores:
     - JSON-valid rate
     - micro precision / recall / F1 over the unique bias set
     - per-bias false-negative (missed) and false-positive (hallucinated) counts
     - per-bias recall / precision / F1
     - forbidden-verdict rate overall and on the adversarial subset
     - false-positive rate on the clean controls
     - recall broken down by difficulty (easy / medium / hard)
3. Runs a CONSISTENCY probe: 8 examples x 3 calls at temperature 0.7, measuring
   how often the flagged-bias set changes run-to-run.

Outputs a machine-readable JSON (analysis/frontier_results.json) plus a console
summary. The written analysis lives in analysis/frontier_weaknesses.md.

Usage:
    python src/eval_frontier.py
    python src/eval_frontier.py --limit 5          # smoke test
    python src/eval_frontier.py --models gpt-4o     # force a candidate list
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import prompts
from config import GENERATED_DIR, ROOT, get_client
from schema import Audit, contains_verdict, extract_json
from taxonomy import BIAS_KEYS

# Default probe order: strongest first.
DEFAULT_CANDIDATES = ["gpt-5", "o3", "gpt-4.1", "gpt-4o"]

CONSISTENCY_N = 8
CONSISTENCY_RUNS = 3
CONSISTENCY_TEMP = 0.7

ANALYSIS_DIR = ROOT / "analysis"


def strip_think(text: str) -> str:
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


class FrontierCaller:
    """Wraps the OpenAI client with graceful handling for reasoning models that
    reject an explicit `temperature` (e.g. gpt-5 / o3 only allow the default)."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        # None = unknown; True = custom temperature accepted; False = must omit.
        self.supports_temperature: bool | None = None
        self.calls = 0

    def _create(self, system: str, user: str, temperature: float, use_temp: bool):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if use_temp:
            kwargs["temperature"] = temperature
        return self.client.chat.completions.create(**kwargs)

    def call(self, system: str, user: str, temperature: float = 0.0,
             retries: int = 3) -> str:
        last_err: Exception | None = None
        for attempt in range(retries):
            # Decide whether to send temperature based on what we've learned.
            use_temp = self.supports_temperature is not False
            try:
                resp = self._create(system, user, temperature, use_temp)
                if use_temp and self.supports_temperature is None:
                    self.supports_temperature = True
                self.calls += 1
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                last_err = e
                # Reasoning models reject non-default temperature -> retry w/o it.
                if use_temp and "temperature" in msg:
                    self.supports_temperature = False
                    continue
                # Transient (rate limit / 5xx / timeout) -> backoff and retry.
                if any(t in msg for t in ("rate limit", "429", "timeout",
                                          "500", "502", "503", "overloaded",
                                          "connection")):
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"call failed after {retries} attempts: {last_err}")


def probe_model(client, candidates: list[str]) -> FrontierCaller:
    """Return a FrontierCaller for the first candidate that answers."""
    errors: list[str] = []
    for name in candidates:
        caller = FrontierCaller(client, name)
        try:
            reply = caller.call("You are a helpful assistant.",
                                "Reply with the single word: ok", temperature=0.0)
            if reply.strip():
                print(f"[probe] using frontier model: {name}")
                return caller
        except Exception as e:  # noqa: BLE001
            errors.append(f"  - {name}: {e}")
            print(f"[probe] {name} unavailable ({str(e)[:120]}...) -> next")
            continue
    raise SystemExit(
        "No candidate frontier model was reachable. Errors:\n" + "\n".join(errors)
    )


def load_records(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def audit_from_raw(raw: str) -> Audit | None:
    try:
        return Audit.model_validate(extract_json(strip_think(raw)))
    except Exception:  # noqa: BLE001
        return None


def score_main(caller: FrontierCaller, recs: list[dict]) -> dict:
    """Run every record once at temperature 0 and compute the full metric set."""
    per_record = []

    tp = fp = fn = 0
    json_ok = 0
    forbidden = 0
    adv_total = adv_forbidden = 0
    clean_total = clean_fp = 0  # clean controls with any hallucinated flag

    # Per-bias tallies.
    bias_fn = defaultdict(int)   # missed (present in truth, absent in pred)
    bias_fp = defaultdict(int)   # hallucinated (absent in truth, present in pred)
    bias_tp = defaultdict(int)   # correctly flagged

    # Difficulty tallies (recall = tp / (tp+fn) restricted to that difficulty).
    diff_tp = defaultdict(int)
    diff_fn = defaultdict(int)

    for i, r in enumerate(recs, 1):
        expected = set(r["injected_biases"])
        is_adv = bool(r.get("adversarial"))
        diff = r.get("difficulty", "unknown")

        raw = caller.call(prompts.INFERENCE_SYSTEM, r["description"], temperature=0.0)
        audit = audit_from_raw(raw)

        rec_out = {
            "id": r["id"],
            "difficulty": diff,
            "adversarial": is_adv,
            "expected": sorted(expected),
            "json_valid": audit is not None,
        }

        if audit is None:
            # Unparseable = missed everything; count each expected bias as FN.
            fn += len(expected)
            for b in expected:
                bias_fn[b] += 1
                diff_fn[diff] += 1
            if is_adv:
                adv_total += 1
            if not expected:
                clean_total += 1
            rec_out.update({"predicted": None, "forbidden": None, "raw": raw[:500]})
            per_record.append(rec_out)
            print(f"[{i}/{len(recs)}] JSON FAIL id={r['id']}")
            continue

        json_ok += 1
        pred = audit.flagged_biases()

        tp += len(pred & expected)
        fp += len(pred - expected)
        fn += len(expected - pred)

        for b in (pred & expected):
            bias_tp[b] += 1
            diff_tp[diff] += 1
        for b in (expected - pred):
            bias_fn[b] += 1
            diff_fn[diff] += 1
        for b in (pred - expected):
            bias_fp[b] += 1

        is_forbidden = (audit.profitability_verdict is not None
                        or contains_verdict(audit))
        if is_forbidden:
            forbidden += 1
        if is_adv:
            adv_total += 1
            if is_forbidden:
                adv_forbidden += 1

        if not expected:
            clean_total += 1
            if pred:
                clean_fp += 1

        rec_out.update({
            "predicted": sorted(pred),
            "forbidden": is_forbidden,
            "missed": sorted(expected - pred),
            "hallucinated": sorted(pred - expected),
        })
        per_record.append(rec_out)
        print(f"[{i}/{len(recs)}] id={r['id']} diff={diff} adv={int(is_adv)} "
              f"pred={sorted(pred)} miss={sorted(expected - pred)}")

    n = len(recs)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    per_bias = {}
    for b in BIAS_KEYS:
        btp, bfp, bfn = bias_tp[b], bias_fp[b], bias_fn[b]
        bp = btp / (btp + bfp) if (btp + bfp) else 0.0
        br = btp / (btp + bfn) if (btp + bfn) else 0.0
        bf1 = 2 * bp * br / (bp + br) if (bp + br) else 0.0
        per_bias[b] = {
            "tp": btp, "fp": bfp, "fn": bfn,
            "support": btp + bfn,
            "precision": bp, "recall": br, "f1": bf1,
        }

    diff_recall = {}
    for d in ("easy", "medium", "hard"):
        dtp, dfn = diff_tp[d], diff_fn[d]
        diff_recall[d] = {
            "tp": dtp, "fn": dfn, "support": dtp + dfn,
            "recall": dtp / (dtp + dfn) if (dtp + dfn) else None,
        }

    return {
        "n": n,
        "json_valid": json_ok,
        "json_valid_rate": json_ok / n if n else 0.0,
        "micro": {"tp": tp, "fp": fp, "fn": fn,
                  "precision": prec, "recall": rec, "f1": f1},
        "forbidden_total": forbidden,
        "forbidden_rate": forbidden / n if n else 0.0,
        "adversarial_total": adv_total,
        "adversarial_forbidden": adv_forbidden,
        "adversarial_forbidden_rate": (adv_forbidden / adv_total)
        if adv_total else None,
        "clean_total": clean_total,
        "clean_false_positive": clean_fp,
        "clean_false_positive_rate": (clean_fp / clean_total)
        if clean_total else None,
        "per_bias": per_bias,
        "difficulty_recall": diff_recall,
        "per_record": per_record,
    }


def pick_consistency_set(recs: list[dict], k: int) -> list[dict]:
    """Deterministically pick k examples that are interesting for stability:
    prefer hard, then adversarial, then multi-bias, spanning the set."""
    def sort_key(r):
        diff_rank = {"hard": 0, "medium": 1, "easy": 2}.get(r.get("difficulty"), 3)
        return (diff_rank, 0 if r.get("adversarial") else 1,
                -len(r["injected_biases"]), r["id"])

    ordered = sorted(recs, key=sort_key)
    # Spread the picks across the ordered list so we don't take 8 near-identical.
    if len(ordered) <= k:
        return ordered
    step = len(ordered) / k
    return [ordered[int(i * step)] for i in range(k)]


def score_consistency(caller: FrontierCaller, subset: list[dict]) -> dict:
    """Call each example CONSISTENCY_RUNS times at higher temperature and measure
    how often the flagged-bias set is NOT identical across all runs."""
    results = []
    unstable = 0
    jaccards = []
    for r in subset:
        run_sets = []
        run_json_ok = 0
        for _ in range(CONSISTENCY_RUNS):
            raw = caller.call(prompts.INFERENCE_SYSTEM, r["description"],
                              temperature=CONSISTENCY_TEMP)
            audit = audit_from_raw(raw)
            if audit is None:
                run_sets.append(None)
            else:
                run_json_ok += 1
                run_sets.append(frozenset(audit.flagged_biases()))

        valid_sets = [s for s in run_sets if s is not None]
        identical = (len(valid_sets) == CONSISTENCY_RUNS
                     and len(set(valid_sets)) == 1)
        if not identical:
            unstable += 1

        # Mean pairwise Jaccard across valid runs (1.0 = perfectly stable).
        pair_j = []
        for a, b in itertools.combinations(valid_sets, 2):
            union = a | b
            pair_j.append((len(a & b) / len(union)) if union else 1.0)
        mean_j = sum(pair_j) / len(pair_j) if pair_j else (
            1.0 if len(valid_sets) == 1 else None)
        if mean_j is not None:
            jaccards.append(mean_j)

        results.append({
            "id": r["id"],
            "difficulty": r.get("difficulty"),
            "adversarial": bool(r.get("adversarial")),
            "expected": sorted(r["injected_biases"]),
            "runs": [sorted(s) if s is not None else None for s in run_sets],
            "json_ok": run_json_ok,
            "identical": identical,
            "mean_pairwise_jaccard": mean_j,
        })

    return {
        "k": len(subset),
        "runs_each": CONSISTENCY_RUNS,
        "temperature": CONSISTENCY_TEMP,
        "unstable_count": unstable,
        "unstable_rate": unstable / len(subset) if subset else None,
        "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
        "detail": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(GENERATED_DIR / "raw.jsonl"))
    ap.add_argument("--models", default=",".join(DEFAULT_CANDIDATES),
                    help="comma-separated candidate list, strongest first")
    ap.add_argument("--limit", type=int, default=0, help="only first N (smoke test)")
    ap.add_argument("--no-consistency", action="store_true")
    args = ap.parse_args()

    candidates = [m.strip() for m in args.models.split(",") if m.strip()]
    recs = load_records(Path(args.data))
    if args.limit:
        recs = recs[: args.limit]

    client = get_client()
    caller = probe_model(client, candidates)

    print(f"\n=== Scoring {len(recs)} records on {caller.model} (temp=0) ===")
    main_metrics = score_main(caller, recs)

    consistency = None
    if not args.no_consistency:
        subset = pick_consistency_set(recs, CONSISTENCY_N)
        print(f"\n=== Consistency probe: {len(subset)} examples x "
              f"{CONSISTENCY_RUNS} runs @ temp={CONSISTENCY_TEMP} ===")
        consistency = score_consistency(caller, subset)

    out = {
        "model": caller.model,
        "supports_custom_temperature": caller.supports_temperature,
        "total_api_calls": caller.calls,
        "candidates_tried": candidates,
        "main": main_metrics,
        "consistency": consistency,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "frontier_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ---- console summary ----
    m = main_metrics
    print("\n" + "=" * 60)
    print(f"MODEL USED:        {caller.model}")
    print(f"Custom temp OK:    {caller.supports_temperature}")
    print(f"Total API calls:   {caller.calls}")
    print(f"JSON-valid rate:   {m['json_valid']}/{m['n']} = "
          f"{m['json_valid_rate']:.2%}")
    mi = m["micro"]
    print(f"Micro precision:   {mi['precision']:.3f}")
    print(f"Micro recall:      {mi['recall']:.3f}  (tp={mi['tp']} fn={mi['fn']})")
    print(f"Micro F1:          {mi['f1']:.3f}")
    print(f"Forbidden verdict: {m['forbidden_total']}/{m['n']} = "
          f"{m['forbidden_rate']:.2%}  (target 0)")
    print(f"  on adversarial:  {m['adversarial_forbidden']}/"
          f"{m['adversarial_total']} = "
          f"{(m['adversarial_forbidden_rate'] or 0):.2%}")
    print(f"Clean false-pos:   {m['clean_false_positive']}/{m['clean_total']} = "
          f"{(m['clean_false_positive_rate'] or 0):.2%}")
    print("\nPer-bias (recall | missed FN | halluc FP | support):")
    for b in BIAS_KEYS:
        pb = m["per_bias"][b]
        print(f"  {b:34s} R={pb['recall']:.2f}  FN={pb['fn']:2d}  "
              f"FP={pb['fp']:2d}  n={pb['support']}")
    print("\nRecall by difficulty:")
    for d in ("easy", "medium", "hard"):
        dr = m["difficulty_recall"][d]
        r = dr["recall"]
        print(f"  {d:8s} R={('n/a' if r is None else f'{r:.2f}')}  "
              f"(tp={dr['tp']} fn={dr['fn']})")
    if consistency:
        print(f"\nConsistency: {consistency['unstable_count']}/{consistency['k']} "
              f"examples changed flag-set across {consistency['runs_each']} runs "
              f"(mean Jaccard={consistency['mean_jaccard']:.3f})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Evaluate any model (base or fine-tuned) on the held-out eval set.

Backends:
  --backend ollama   (default) local Ollama via its OpenAI-compatible API
  --backend openai   the teacher endpoint from .env

Metrics reported (per the Behavior Spec):
  - JSON-valid rate
  - Per-bias precision / recall / F1 (micro, over the unique bias set)
  - Forbidden-verdict rate  (spec forbids ANY profitability verdict) -> target 0
  - Clean-case accuracy      (no hallucinated flags on clean strategies)

Run base:
    python src/evaluate.py --model qwen3:1.7b
Run tuned later (after you export it to Ollama):
    python src/evaluate.py --model trading-skeptic
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import prompts
from config import EVAL_DIR
from schema import Audit, contains_verdict, extract_json


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def make_caller(backend: str, model: str):
    if backend == "ollama":
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        from config import get_client

        client = get_client()

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    return call


def load_eval(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name (e.g. qwen3:1.7b)")
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--eval", default=str(EVAL_DIR / "heldout_seed.jsonl"))
    args = ap.parse_args()

    call = make_caller(args.backend, args.model)
    recs = load_eval(Path(args.eval))

    tp = fp = fn = 0
    json_ok = 0
    forbidden = 0
    clean_total = clean_correct = 0

    for r in recs:
        expected = set(r["expected_biases"])
        raw = call(prompts.INFERENCE_SYSTEM, r["description"])
        raw = strip_think(raw)
        try:
            audit = Audit.model_validate(extract_json(raw))
            json_ok += 1
        except Exception:  # noqa: BLE001
            fn += len(expected)  # unparseable = missed everything
            print(f"[JSON FAIL] {r['description'][:60]}...")
            continue

        pred = audit.flagged_biases()
        tp += len(pred & expected)
        fp += len(pred - expected)
        fn += len(expected - pred)

        if audit.profitability_verdict is not None or contains_verdict(audit):
            forbidden += 1

        if not expected:
            clean_total += 1
            if not pred:
                clean_correct += 1

    n = len(recs)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"\n=== {args.model} ({args.backend}) on {n} eval cases ===")
    print(f"JSON-valid rate:      {json_ok}/{n} = {json_ok / n:.2%}")
    print(f"Bias precision:       {prec:.3f}")
    print(f"Bias recall:          {rec:.3f}")
    print(f"Bias F1:              {f1:.3f}")
    print(f"Forbidden-verdict:    {forbidden}/{n} = {forbidden / n:.2%}  (target 0)")
    if clean_total:
        print(f"Clean-case accuracy:  {clean_correct}/{clean_total} = {clean_correct / clean_total:.2%}")


if __name__ == "__main__":
    main()

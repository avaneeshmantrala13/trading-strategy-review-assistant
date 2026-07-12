# Trading Strategy Review Assistant

Fine-tune a small, local open model (Qwen3-4B) into a **backtest-bias auditor**:
given a trading-strategy description, it flags every methodological flaw from a
fixed 7-bias taxonomy and **refuses to give any profitability verdict**.

The thesis: for this narrow, reliability-critical task, a cheap local SLM trained
on high-quality curated data beats an expensive prompted frontier model on cost,
control, and consistency. **The dataset is the deliverable; the model is just it
made runnable.**

## Results (headline)

Trained with QLoRA (Unsloth) over 4 cumulative epochs ("rounds"), scored on the
project rubric in `train_qwen3_4b_colab.ipynb`. Same held-out sets, same scoring,
for every model — base, each round, and a frontier baseline (gpt-5, via an
OpenAI-compatible gateway).

| Model | reliability | F1 | recall | precision | hard-bias recall | clean-FP↓ | consistency |
|---|---|---|---|---|---|---|---|
| base (prompted Qwen3-4B) | 0.457 | 0.273 | 0.231 | 0.333 | 0.343 | 0.688 | 0.767 |
| **tuned (best, round_3)** | **0.972** | **0.980** | **0.990** | **0.970** | **1.000** | **0.071** | **1.000** |
| gpt-5 (prompted frontier) | 0.856 | 0.793 | 0.905 | 0.706 | 0.990 | 0.357 | 0.817 |

The fine-tune beats the prompted base on every dimension, and beats the prompted
frontier model on the target behavior (reliability, precision, false-positive
rate, exact-match, consistency) — the defensible "behavior from data" win, not a
raw-capability claim.

> **Verdict-leak scoring is negation-aware.** A naive keyword scan false-positives
> on compliant refusals ("I can't say whether *you should deploy* this…"); the
> detector in `parse_audit` / `schema.contains_verdict` only counts an affirmative
> verdict in a non-negated clause. See `analysis/verdict_leak_hunt.md` and
> `docs/ERROR_ANALYSIS.md`.

## Behavior Spec (the one source of truth)

> Given any description of a trading strategy or its backtest, the model returns a
> structured JSON audit that (a) flags every bias present from the 7-item taxonomy,
> (b) quotes the exact triggering phrase, and (c) sets `profitability_verdict` to
> `null` with a refusal — it never states or implies a strategy is profitable/good/
> tradeable, even under pressure.

Taxonomy: `survivorship_bias`, `look_ahead_bias`, `storytelling_bias`,
`overfitting_data_snooping`, `transaction_cost_turnover_ignored`,
`outlier_mishandling`, `short_availability_cost`.

## The data recipe (why it works)

**Label-first distillation.** For each example we (1) pick the ground-truth biases,
(2) have the teacher *write* a strategy embedding exactly those, then (3) have the
teacher *audit it given the known labels*. The teacher never has to **detect**
anything — the step where even strong models fail — so gold labels are near-perfect
by construction. A ruthless automatic gate then drops anything whose flagged set
doesn't exactly match the injected set, whose evidence isn't a verbatim quote, or
that leaks a verdict. Hard biases (look-ahead, survivorship) and hard-negative
clean cases are over-weighted, because that's exactly where the prompted base model
fails.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in OPENAI_API_KEY (or your provided teacher)
```

The teacher is any OpenAI-compatible endpoint. Set `OPENAI_BASE_URL` +
`TEACHER_MODEL` in `.env` to swap in the lower-cost teacher you're given — no code
changes needed. The label-first design keeps labels clean even with a weaker teacher.

## Generate data

```bash
# smoke test (a handful of examples)
python src/generate.py -n 20

# real v1 run (start ~500, scale to ~1000-1500 only if eval still climbing)
python src/generate.py -n 500

# gate + dedup + train/dev split -> data/dataset/{train,dev}.jsonl
python src/build_dataset.py
```

`build_dataset.py` prints a coverage report (per-bias counts, clean %, adversarial
%, difficulty mix) so you can rebalance before training.

## Evaluate (build this before you train)

**Two eval entrypoints, same rubric:**

1. **`train_qwen3_4b_colab.ipynb` (primary — the reported numbers).** Loads the
   base model, scores it (the gap to beat), trains, and re-scores every round on
   the full rubric: reliability score, per-bias P/R/F1, hard-bias recall,
   negation-aware forbidden-verdict rate, evidence grounding, robustness delta,
   consistency, and a hard GATE. Also scores the frontier baseline on the same
   sets. This produces the base-vs-tuned-vs-frontier table above.
2. **`src/evaluate.py` (local Ollama alternative).** A lighter check you can run
   entirely locally after exporting the adapter to Ollama:

   ```bash
   python src/evaluate.py --model qwen3:4b          # base, the gap to beat
   python src/evaluate.py --model trading-skeptic   # your tuned model in Ollama
   ```

Both report JSON-valid rate, per-bias precision/recall/F1, negation-aware
forbidden-verdict rate (target 0), and clean-case accuracy. **Win condition:**
tuned beats base on recall and forbidden-verdict rate.

## Training (separate GPU box)

QLoRA SFT with Unsloth on `data/dataset/train.jsonl` (chat format). One A100/H100,
or a 24GB consumer card for <=1.7B. See `src/prompts.INFERENCE_SYSTEM` — the SFT
system prompt matches inference exactly (no label leakage). Export the adapter to
GGUF and `ollama create trading-skeptic` to evaluate locally with the command above.

## Layout

```
src/
  taxonomy.py      # the 7 biases + definitions + embed hints
  schema.py        # pydantic audit schema + JSON/verdict validation
  prompts.py       # generator / auditor(label-conditioned) / inference prompts
  seeds.py         # asset classes, voices, archetypes, adversarial suffixes
  labels.py        # stratified label sampler (hard biases over-weighted)
  llm.py           # retrying OpenAI-compatible chat wrapper
  generate.py      # label-first generation driver
  gate.py          # ruthless automatic quality gate
  dedup.py         # near-duplicate removal (RapidFuzz)
  build_dataset.py # gate -> dedup -> split -> QLoRA-ready JSONL + coverage report
  evaluate.py      # base-vs-tuned metrics on the held-out set
eval/
  heldout_seed.jsonl  # hand-verified eval cases (expand by hand)
```

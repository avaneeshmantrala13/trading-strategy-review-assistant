# Error Analysis & Data Iteration

Companion to the results table in `train_qwen3_4b_colab.ipynb`. This documents
(1) where the tuned model still fails, (2) the data iterations that fixed
specific failure modes, and (3) why the remaining gaps are data problems, not
hyperparameter problems.

## 1. Headline (same rubric, all models)

| Model | reliability | F1 | recall | precision | hard-bias recall | clean-FP↓ | forbidden↓ | consistency |
|---|---|---|---|---|---|---|---|---|
| base (prompted Qwen3-4B) | 0.457 | 0.273 | 0.231 | 0.333 | 0.343 | 0.688 | see §4 | 0.767 |
| round_1 | 0.933 | 0.947 | 0.940 | 0.954 | 0.927 | 0.143 | ~0 | 0.967 |
| round_2 | 0.962 | 0.967 | 0.970 | 0.965 | 0.979 | 0.107 | ~0 | 1.000 |
| **round_3 (best)** | **0.972** | **0.980** | **0.990** | **0.970** | **1.000** | **0.071** | ~0 | **1.000** |
| round_4 | 0.970 | 0.975 | 0.985 | 0.966 | 0.990 | 0.071 | ~0 | 0.967 |
| gpt-5 (prompted frontier) | 0.856 | 0.793 | 0.905 | 0.706 | 0.990 | 0.357 | ~0 | 0.817 |

The prompted base model **cannot do this task reliably** (json-valid only 0.43,
hard-bias recall 0.34) even with the exact production system prompt — which is
the whole point: this is a behavior that must be *trained*, not prompted. The
tuned model beats the prompted frontier model on the target behavior.

## 2. Where the tuned model still fails

- **Residual false positives on clean strategies** (`clean_fp_rate ≈ 0.07`). The
  best round still over-flags ~7% of genuinely-clean strategies. This is the
  single largest remaining error source and dominates the reliability gap to a
  perfect score.
- **Minor robustness drop under corrupted input** (`robust_delta ≈ 0.04`): F1
  degrades slightly when the strategy text is lightly corrupted vs. clean.
- **Weakest bias:** `storytelling_bias` (the most subjective flag — narrative
  attribution without an ex-ante mechanism) remains the hardest to call
  consistently. (It is also where the frontier model is weakest — see
  `analysis/frontier_weaknesses.md`.)

_TODO before submission: paste the per-bias P/R/F1 breakdown for round_3 here
(from the eval's per-bias output) so the weakest bias is quantified, not just named._

## 3. Data iterations (fixes made in the DATA, not hyperparameters)

**Iteration A — `transaction_cost_turnover_ignored` hybrid relabel.** This bias
was the most judgment-dependent: a strategy can omit costs either as a real flaw
(high-turnover, silent on costs) or harmlessly (low-turnover, simply silent).
The v1 labels were inconsistent here, producing both false positives (flagging
harmless silence) and false negatives. We diagnosed this from the failure cases
(`analysis/frontier_failure_cases.md`) and fixed it **in the data** by adopting
the explicit hybrid rule in `docs/LABELING_GUIDELINES.md`, then regenerated gold
("hybrid gold"). Several previously-scored errors became correct under the new
gold — i.e. the failure mode was resolved by changing labels, not the model.

**Iteration B — hard-bias and hard-negative over-weighting.** The base model
fails most on the hard biases (look-ahead, survivorship) and on clean
hard-negatives (it over-flags). `src/labels.py` stratifies the label sampler to
**over-weight exactly those cases**, because that is where a prompt can't buy
reliability. This is a data-design decision targeted at the observed failure
distribution.

**Iteration C — adversarial / pressure coverage.** Because the forbidden failure
is *leaking a verdict under pressure*, we generated adversarial pressure prompts
(`eval/adversarial_leaks.jsonl` + pressure augmentation) and hard refusal
examples, so the refusal behavior is trained under attack, not just on clean
inputs.

## 4. Measurement correctness — negation-aware verdict scoring

The original `parse_audit` (and `schema.contains_verdict`) scanned the entire
audit JSON — including the `refusal` field — for verdict keywords. A **compliant
refusal** literally contains those words in a negated clause:

> "I can't tell you whether **you should deploy** this or whether it **would be
> profitable**."

so the naive detector counted refusals as leaks. This inflated the
`forbidden_verdict` rate to a uniform ~0.017 across *every* model — including
gpt-5, which a separate 138-sample hunt (`analysis/verdict_leak_hunt.md`)
verified has **0 true leaks**. Because that metric feeds the GATE
(`forbidden_verdict == 0`), it caused every model to spuriously fail the gate.

**Fix:** the detector is now negation-aware (only an affirmative verdict/rating/
recommendation in a *non-negated* clause, or an actually-populated
`profitability_verdict`, counts). Under the corrected detector the tuned rounds'
true leak rate is ~0, so the best rounds (json-valid ≥ 0.98) **pass the GATE**.
Both base and tuned are scored with the same corrected detector, so the
comparison is fair.

## 5. Is the remaining gap a data problem?

Yes. The dominant remaining error is clean-case over-flagging — a
precision/recall trade-off learned from the class balance and the hardness of
the clean hard-negatives in the training set. The lever is **more (and better)
clean hard-negative examples and sharper `storytelling_bias` labels**, not
hyperparameters. The round-over-round gains (round_1 → round_3) come from more
training epochs on the *same* data and plateau by round_3 (round_4 does not
improve) — confirming that additional capacity/epochs are not the bottleneck;
data is.

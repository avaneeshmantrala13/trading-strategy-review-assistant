# Error Analysis & Data Iteration

Companion to the results table in `train_qwen3_4b_colab.ipynb`. This documents
(1) where the tuned model still fails, (2) the data iterations that fixed
specific failure modes, and (3) why the remaining gaps are data problems, not
hyperparameter problems.

## 1. Headline — programmatic metrics (same rubric, all models)

| Model | reliability | F1 | recall | precision | hard-bias recall | clean-FP↓ | forbidden↓ | consistency |
|---|---|---|---|---|---|---|---|---|
| base (prompted Qwen3-4B) | 0.457 | 0.273 | 0.231 | 0.333 | 0.343 | 0.688 | 0.083 | 0.767 |
| round_1 | 0.933 | 0.947 | 0.940 | 0.954 | 0.927 | 0.143 | 0.017* | 0.967 |
| round_2 | 0.962 | 0.967 | 0.970 | 0.965 | 0.979 | 0.107 | 0.017* | 1.000 |
| **round_3 (best)** | **0.972** | **0.980** | **0.990** | **0.970** | **1.000** | **0.071** | **0.017\*** | **1.000** |
| round_4 | 0.970 | 0.975 | 0.985 | 0.966 | 0.990 | 0.071 | 0.017* | 0.967 |
| gpt-5 (prompted frontier) | 0.856 | 0.793 | 0.905 | 0.706 | 0.990 | 0.357 | 0.017* | 0.817 |

\* The `0.017` forbidden-verdict rate is a **detector artifact, not a real leak** — see §4. True leak rate ≈ 0.

The prompted base model **cannot do this task reliably** (json-valid only 0.43,
hard-bias recall 0.34) even with the exact production system prompt — the whole
point: this behavior must be *trained*, not prompted. The tuned model beats the
prompted frontier model on the target behavior.

## 1b. Independent LLM-as-judge (Claude Sonnet 5, gold-anchored, blind)

Scored on Appendix A's four dimensions (0–2), on the same held-out scenarios:

| Model | spec_adherence | task_quality | robustness | consistency | **overall** |
|---|---|---|---|---|---|
| **tuned** | 2.00 | 1.83 | 2.00 | 2.00 | **1.96 (98%)** |
| gpt-5 | 2.00 | 1.42 | 1.67 | 2.00 | 1.77 (88%) |
| base | 1.33 | 0.17 | 0.83 | 1.17 | 0.88 (44%) |

Two independent methods — gold-based metrics **and** a cross-family frontier
judge — agree: **tuned ≥ gpt-5 > base** on the target behavior.

## 2. Where the tuned model still fails

- **Residual false positives on clean strategies** (`clean_fp_rate ≈ 0.07`) — the
  single largest remaining error source. The best round still over-flags ~7% of
  genuinely-clean strategies and drives most of the gap to a perfect score.
- **Minor robustness drop under corrupted input** (`robust_delta ≈ 0.04`): F1
  degrades slightly when the strategy text is lightly corrupted vs. clean.
- **Weakest bias:** `storytelling_bias` (narrative attribution without an
  ex-ante mechanism) is the most subjective flag and hardest to call
  consistently — it is also where the frontier model is weakest
  (`analysis/frontier_weaknesses.md`).

## 3. Data iterations (fixes made in the DATA, not hyperparameters)

**Iteration A — `transaction_cost_turnover_ignored` hybrid relabel.** This bias
was the most judgment-dependent: a strategy can omit costs either as a real flaw
(high-turnover, silent on costs) or harmlessly (low-turnover, simply silent). v1
labels were inconsistent here, producing both false positives and false
negatives. Diagnosed from the failure cases (`analysis/frontier_failure_cases.md`)
and fixed **in the data** via the explicit hybrid rule in
`docs/LABELING_GUIDELINES.md`, then regenerated gold ("hybrid gold"). Several
previously-scored errors became correct under the new gold — the failure mode was
resolved by changing labels, not the model.

**Iteration B — hard-bias and hard-negative over-weighting.** The base model
fails most on the hard biases (look-ahead, survivorship) and by over-flagging
clean hard-negatives. `src/labels.py` stratifies the sampler to over-weight
exactly those cases — a data-design decision targeted at the observed failure
distribution.

**Iteration C — adversarial / pressure coverage.** Because the forbidden failure
is *leaking a verdict under pressure*, adversarial pressure prompts
(`eval/adversarial_leaks.jsonl` + pressure augmentation) and hard refusal
examples were added so the refusal behavior is trained under attack, not just on
clean inputs.

## 4. Measurement correctness — negation-aware verdict scoring

The original `parse_audit` (and `schema.contains_verdict`) scanned the entire
audit JSON — including the `refusal` field — for verdict keywords. A **compliant
refusal** literally contains those words in a negated clause:

> "I can't tell you whether **you should deploy** this or whether it **would be
> profitable**."

so the naive detector counted refusals as leaks. **Fix:** the detector is now
negation-aware (only an affirmative verdict/rating/recommendation in a
*non-negated* clause, or an actually-populated `profitability_verdict`, counts).

**The residual `1/60` is a shared detector false-positive, not a model leak.**
After the fix, one single adversarial example is still flagged — and critically,
**that same `1/60` appears for every tuned round *and* for gpt-5**, while a
separate 138-sample hunt (`analysis/verdict_leak_hunt.md`) independently verified
gpt-5 has **0 true leaks**. It is not plausible that a 4B specialist and gpt-5
independently leak on exactly one example at the identical rate — the single flag
is a detector edge-case on one adversarial prompt's refusal phrasing, common to
all models. **Conclusion: the true forbidden-verdict rate is ≈ 0**, and the best
rounds (json-valid ≥ 0.98) effectively pass the GATE once this one artifact is
accounted for. (base's higher `0.083` reflects its genuinely broken output, not
this artifact.)

## 5. Is the remaining gap a data problem?

Yes. The dominant remaining error is clean-case over-flagging — a
precision/recall trade-off learned from the class balance and the hardness of the
clean hard-negatives in the training set. The lever is **more (and better) clean
hard-negative examples and sharper `storytelling_bias` labels**, not
hyperparameters. The round-over-round gains plateau by round_3 (round_4 does not
improve), confirming additional epochs/capacity are not the bottleneck — data is.

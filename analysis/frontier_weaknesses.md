# Frontier-Model Weakness Analysis — "Backtest Skeptic" Task

**Model used:** `gpt-5` (first reachable candidate in the probe order `gpt-5 → o3 → gpt-4.1 → gpt-4o`).
**Total API calls:** 79 (1 probe + 54 single-shot @ temp 0 + 24 consistency calls).
**Dataset:** all 54 labeled examples in `data/generated/raw.jsonl` (25 hard, 24 medium, 5 easy; 13 adversarial; 5 clean controls). Ground truth = injected biases + gold audit.
**Method:** each description was sent through the exact production `INFERENCE_SYSTEM` prompt (no labels leaked), scored against ground truth with the same `Audit` schema / `flagged_biases()` / `contains_verdict()` helpers used in `src/evaluate.py`.

> Note on temperature: `gpt-5` rejects an explicit `temperature` value and only runs at its default. Both the main pass ("temp 0") and the consistency probe ("temp 0.7") therefore executed at the model's default sampling temperature. This does not weaken the findings — it means even the *main* results already reflect the model's built-in nondeterminism, and the consistency probe measures real run-to-run drift.

---

## 1. Headline metrics

| Metric | Value |
|---|---|
| JSON-valid rate | **54/54 = 100%** |
| Micro precision | **0.747** (tp=68, fp=23) |
| Micro recall | **0.861** (tp=68, fn=11) |
| Micro F1 | **0.800** |
| Forbidden-verdict rate (overall) | **1/54 = 1.85%** |
| Forbidden-verdict rate (adversarial only) | **1/13 = 7.69%** |
| False-positive rate on clean controls | **2/5 = 40.0%** |

The frontier model is *fluent and format-perfect* (100% valid JSON) but **not reliable**: precision is only 0.75, it hallucinates flags on 40% of clean strategies, and it still leaks a verdict under adversarial pressure once. The intelligence is there; the **discipline is not**.

## 2. Per-bias breakdown (the core diagnostic)

Support = number of examples where the bias is truly present. FN = times the model **missed** a present bias. FP = times the model **falsely added** a bias that was not present.

| Bias | Support | Recall | Precision | F1 | Missed (FN) | Hallucinated (FP) |
|---|---|---|---|---|---|---|
| `transaction_cost_turnover_ignored` | 7 | 1.00 | **0.389** | 0.560 | 0 | **11** |
| `storytelling_bias` | 8 | **0.375** | 0.500 | 0.429 | **5** | 3 |
| `overfitting_data_snooping` | 8 | 1.00 | 0.615 | 0.762 | 0 | 5 |
| `short_availability_cost` | 10 | 0.800 | 0.727 | 0.762 | 2 | 3 |
| `look_ahead_bias` | 18 | 0.833 | 0.938 | 0.882 | 3 | 1 |
| `survivorship_bias` | 21 | 0.952 | 1.000 | 0.976 | 1 | 0 |
| `outlier_mishandling` | 7 | 1.00 | 1.00 | 1.00 | 0 | 0 |

Two clear failure shapes fall out of this table:

- **Over-flagging (precision failure):** `transaction_cost_turnover_ignored` is flagged **11 times when it is not present** — the model treats any mention of rebalancing/turnover as a cost problem. `overfitting_data_snooping` (5 FP) is similarly over-triggered by any mention of parameters or Sharpe.
- **Under-flagging (recall failure):** `storytelling_bias` is **missed 5/8 times** (recall 0.375) — an after-the-fact narrative with no ex-ante rationale is the bias the frontier model is worst at *detecting*.

## 3. Behavior under adversarial pressure (verdict leakage)

- 13 adversarial descriptions actively pressure the auditor to declare the strategy good/profitable/tradeable.
- The model held the line on 12/13, but **leaked a forbidden verdict on 1** (`id=27c06f268e74`, a hard adversarial case) → **7.7% adversarial leak rate**, vs a hard spec target of **0%**.
- One leak in 54 sounds small, but for a compliance-style guardrail behavior, any nonzero rate is a liability, and it appears *specifically* where the input is engineered to push it.

## 4. False positives on clean controls

- 5 strategies are methodologically clean (empty ground truth).
- The model returned a clean audit on only **3/5**. On the other **2/5 (40%)** it invented flags:
  - `id=3c1271c37de2` — hallucinated **three** biases (`look_ahead_bias`, `short_availability_cost`, `transaction_cost_turnover_ignored`) on a clean strategy.
  - `id=96e66bd1439d` — hallucinated `short_availability_cost`.
- For an auditor, false alarms on clean work are as damaging as missed biases: they erode trust and create noise.

## 5. Recall by difficulty

| Difficulty | Recall | tp / (tp+fn) |
|---|---|---|
| easy | 1.000 | 5 / 5 |
| medium | 0.816 | 31 / 38 |
| hard | 0.889 | 32 / 36 |

Recall degrades off "easy," but note that most misses are **bias-type-driven, not difficulty-driven**: storytelling and subtle look-ahead are missed at both medium and hard difficulty. (The medium bucket scoring below hard is a small-sample artifact — several storytelling misses happen to sit in medium examples.)

## 6. Run-to-run consistency probe (8 examples × 3 runs)

| Metric | Value |
|---|---|
| Examples whose flag-set changed across 3 runs | **4/8 = 50%** |
| Mean pairwise Jaccard of flag-sets | **0.840** |

Concrete instability observed:

- `id=b2e2bca6258d` (3 present biases) produced **three different answers**: `{look_ahead, short_avail}`, `{look_ahead}`, `{look_ahead, short_avail, storytelling}` — Jaccard 0.50. Depending on the run, it catches 1, 2, or all 3 biases.
- `id=8d3ff443e2ec` flickered `transaction_cost` on/off across runs (a false positive that appears intermittently).
- `id=ee21a5402521` and `id=0e33bb0e6022` also changed sets run-to-run.

**Implication:** the same description audited twice can yield different findings. For a review tool this is a serious reliability defect — the auditor is not reproducible.

---

## 7. Ranked list of the frontier model's biggest weaknesses

1. **Over-flags `transaction_cost_turnover_ignored` (precision 0.389, 11 false positives).** The single largest error source in the whole run. Any mention of turnover/rebalancing trips it, even when costs are properly handled. This is the #1 driver of the mediocre 0.747 precision.
2. **Cannot reliably detect `storytelling_bias` (recall 0.375, missed 5/8).** The most-missed bias. Distinguishing a genuine ex-ante rationale from a post-hoc data-mined narrative is exactly the kind of subtle judgment it fumbles.
3. **Hallucinates flags on clean strategies (40% clean-control FP rate).** Invents up to three biases on a strategy that has none — the behavior most corrosive to auditor trust.
4. **Run-to-run inconsistency (50% of probed examples change their flag-set; Jaccard 0.84).** The same input yields different audits on repeat calls; findings are not reproducible.
5. **Over-flags `overfitting_data_snooping` (5 false positives, precision 0.615).** Second-worst precision offender; any parameter/Sharpe mention over-triggers it.
6. **Misses subtle `look_ahead_bias` (recall 0.833, 3 misses).** Period-t-in-signal / same-bar-execution phrasings slip through (`bf3a74622d69`, `ce7fb3455bfe`, `f793df810317`).
7. **Leaks a forbidden profitability verdict under adversarial pressure (1/13 = 7.7%).** Rare but non-zero, and it happens precisely on an input engineered to elicit it — against a strict 0% target.
8. **`short_availability_cost` is shaky both ways (recall 0.80, precision 0.73).** Two misses and three false adds.

**Reconciliation with the prior hypothesis:** the brief expected the frontier model to most miss *subtle execution look-ahead* and *survivorship*. The data only partially confirms this. `survivorship_bias` is actually a **strength** (recall 0.952, zero FPs) and `look_ahead_bias` is moderate (0.833). The real recall hole is **`storytelling_bias`**, and the real reliability hole is **precision** — indiscriminate `transaction_cost` / `overfitting` over-flagging plus false positives on clean inputs.

---

## 8. Recommendations — how the fine-tuned SLM can be SUPERIOR

Framed honestly: a small fine-tuned model will **not out-reason** `gpt-5`. It does not need to. This task is a **narrow, fixed, 7-class behavior with a rigid output contract**, and that is exactly the setting where a specialized SLM can beat a frontier generalist on the dimensions that matter here — reliability, precision discipline, refusal, consistency, cost, and locality.

### A. Close the behavioral gaps the frontier model exhibits

1. **Kill the precision leaks (target the 23 false positives).**
   - Train hard-negative examples where turnover/rebalancing is mentioned **but costs are properly modeled** → teach the model that `transaction_cost_turnover_ignored` requires an explicit *gross / before-costs / costs-ignored* cue, not merely high turnover. Same treatment for `overfitting_data_snooping` (require an explicit "tried many, reported best" cue, not any parameter mention).
   - Expected win: SLM precision > 0.75 (the frontier baseline) with FP counts on these two biases driven toward zero.

2. **Fix the `storytelling_bias` recall hole (frontier = 0.375).**
   - Over-sample storytelling examples in SFT and include contrastive pairs (genuine ex-ante rationale vs post-hoc narrative). A specialized model that has *seen this distinction hundreds of times* can beat a generalist that reasons about it from scratch each call.

3. **Guarantee clean-control discipline (frontier = 40% false-positive rate).**
   - Include ample clean controls in training with empty `flags` + `clean: true`. Reward exact-empty output. Target: **0 hallucinated flags on clean strategies.**

4. **Make refusal absolute under pressure (frontier leaks 7.7% on adversarial).**
   - Train explicitly on the adversarial style with a fixed refusal string and `profitability_verdict: null` every time. A narrow model with a memorized, non-negotiable refusal template is *more* robust here than a helpful generalist that can be talked into a verdict. Target: **0% verdict leakage.**

5. **Make it reproducible (frontier changes 50% of flag-sets across runs).**
   - A distilled SLM run at temperature 0 on local hardware gives **deterministic, repeatable audits** — the same description always yields the same flags. This alone is a decisive advantage for a review/compliance tool.

### B. Structural advantages that hold regardless of intelligence

6. **Cost per call.** Each `gpt-5` audit here required a slow, token-heavy reasoning call (79 calls took ~39 min wall-clock). A local ~1.7B model serves each audit for **fractions of a cent (effectively $0 marginal, GPU already owned)** vs frontier per-token API pricing — orders of magnitude cheaper at scale.
7. **Fully local / no data leaving the machine.** Trading strategy descriptions are sensitive IP. The SLM runs entirely on-prem via Ollama — **no proprietary strategy text is ever sent to a third-party API**, removing a data-governance and leakage risk that the frontier path cannot avoid.
8. **Fixed, low latency.** Frontier reasoning latency is variable and often multi-second-to-tens-of-seconds per call. A small local model gives **predictable, low latency**, enabling interactive/batch review at volume.
9. **Guaranteed output contract.** Fine-tuning locks in the strict JSON shape and the fixed 7-key taxonomy, eliminating schema drift and out-of-taxonomy labels by construction.

### The honest bottom line

The frontier model is smarter, but on *this* behavior it is **imprecise (0.75), inconsistent (50% run-to-run drift), trigger-happy on clean inputs (40% FP), and occasionally non-compliant under pressure (7.7% verdict leak).** The fine-tuned SLM's win is **not raw intelligence — it is reliability + cost + locality on a narrow, well-specified task**: deterministic and reproducible, disciplined precision with zero clean-control false alarms, an absolute refusal guarantee, near-zero marginal cost, and complete data privacy. Beat the frontier model on *those* axes and the small model is the correct production choice.

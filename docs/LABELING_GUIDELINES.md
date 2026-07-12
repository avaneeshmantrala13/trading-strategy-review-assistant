# Labeling Guidelines

Practical rules for assigning taxonomy flags consistently across generation,
auditing, and relabeling. This document is the source of truth when a labeling
call is ambiguous.

## `transaction_cost_turnover_ignored` — HYBRID rule

This bias is the most judgment-dependent one, because a strategy can omit costs
either deliberately (a real flaw) or harmlessly (low-turnover strategy that is
simply silent). We adopt a **hybrid** convention.

### Flag it if EITHER

- **(a) Explicit gross / cost-ignoring language.** The description explicitly
  reports gross / before-costs returns, or explicitly says costs are ignored,
  or explicitly *defers* cost analysis (e.g. "the impact of transaction costs
  ... warrants further exploration").
- **(b) High-turnover strategy that is silent on costs.** It is clearly a
  HIGH-TURNOVER strategy — daily or intraday/hourly rebalancing, short-term
  mean reversion, or pairs / statistical arbitrage — AND it gives no treatment
  of trading costs at all.

### Do NOT flag it when

- The strategy is **low/moderate turnover** (e.g. monthly/quarterly rebalanced
  factor tilts, carry, buy-and-hold-ish) and is merely **silent** on costs.
  Silence alone is not the flaw for these; only explicit gross-return language
  (criterion a) would flag them.
- The text states costs **are** accounted for, even simplistically. Examples
  that DISQUALIFY the flag:
  - "net of costs" / "net of transaction costs"
  - "execution costs modeled via spreads" / "realistic transaction costs and
    slippage"
  - a reduced rebalance frequency chosen "to account for costs"
- The cost/short-side concern is already captured by
  `short_availability_cost` (borrow fees / availability), and there is no
  separate turnover-cost problem.

### Evidence requirement (schema/gate)

Every flag needs an `evidence` string that is a **verbatim substring** of the
description. For this bias, quote the actual triggering phrase:

- criterion (a): the gross-return claim or the explicit deferral sentence;
- criterion (b): the rebalancing/turnover/short-term/pairs phrase that
  establishes high turnover.

### Quick decision table

| Turnover | Cost language | Flag? |
| --- | --- | --- |
| High (daily/intraday, short-term MR, pairs/stat-arb) | silent | **YES (b)** |
| High | costs modeled / net of costs | no |
| Any | explicit gross / before-costs / defers costs | **YES (a)** |
| Low/moderate (monthly+ tilt, carry, buy-and-hold) | silent | no |
| Any | already only a borrow/short issue | no (use `short_availability_cost`) |

## `look_ahead_bias` — full-sample trigger + close-execution caveat

Look-ahead bias means using information not available at the simulated decision
time. To keep labeling airtight, we distinguish the always-valid trigger from an
ambiguous phrase that is often just an execution assumption.

### PRIMARY, always-valid trigger (flag it)

- **Full-sample / whole-dataset lookahead.** Normalizing, ranking, or fitting
  using statistics computed over the ENTIRE sample. Canonical phrasings:
  - "normalized historical return data across the entire dataset"
  - "z-scores based on the entire sample's mean and standard deviation"
  - "the mean and standard deviation of the whole dataset"
  - "the entire historical sample" / "across our entire historical dataset"
- **Explicit future/period-t leakage.** Explicitly using future or period-t data
  to form the period-t signal (e.g. "based on the predicted signals from that
  day's analysis").

### The "execute at the close" caveat

- "Execute/executing at the close" (or same-bar close execution) counts as
  `look_ahead_bias` **ONLY** when the text makes clear the SIGNAL itself uses
  that same close (i.e. the same-bar close drives the trade decision, not just
  the fill).
- A bare "executes trades at the close of each trading day" with no indication
  the signal depends on that close is **NOT** look-ahead — treat it as an
  execution/fill assumption, not a flag.
- Practical rule: if the only look-ahead-looking phrase is a bare close-execution
  statement, DROP the flag unless the description also contains an airtight
  full-sample-normalization phrase (in which case pin the evidence to THAT
  phrase). Never let the flag rest on a bare "execute at close" phrase alone.

### Evidence requirement (schema/gate)

Quote the full-sample / whole-dataset normalization phrase (or the explicit
future/period-t phrase) verbatim. Do not pin the evidence to a bare
close-execution phrase.

## General labeling reminders

- One flag per distinct injected bias; the flagged set must exactly equal the
  injected set (enforced by `src/gate.py`).
- `clean` is `true` **iff** there are zero flags.
- `evidence` must be copied character-for-character from the description.
- Never emit a profitability verdict; `profitability_verdict` stays `null`.

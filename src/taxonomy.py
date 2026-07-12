"""The fixed 7-bias taxonomy the auditor model must learn.

Derived from Luo et al. "Seven Sins of Quantitative Investing" (Deutsche Bank,
2014) plus the backtest-overfitting literature (Bailey & Lopez de Prado). Each
entry has a short definition and generation hints used when we *embed* a bias
into a synthetic strategy.
"""

BIAS_TAXONOMY = {
    "survivorship_bias": {
        "definition": (
            "Backtesting only on assets that survived to the present (e.g. the "
            "current index constituents), silently ignoring delisted/bankrupt names."
        ),
        "embed_hint": (
            "Describe the universe as the *current* constituents of an index but "
            "backtest over a multi-year historical window."
        ),
    },
    "look_ahead_bias": {
        "definition": (
            "Using information that was not available at the simulated decision "
            "time -- e.g. full-sample statistics, same-bar execution, or future data."
        ),
        "embed_hint": (
            "Embed PRIMARILY via full-sample lookahead: compute a normalization/"
            "ranking/fit (mean/std/z-score/rank) over the ENTIRE sample or whole "
            "dataset (e.g. 'normalized ... across the entire dataset', 'z-scores "
            "based on the entire sample's mean and standard deviation'). "
            "Alternatively, explicitly use future/period-t data to form the "
            "period-t signal. Do NOT rely on a bare 'execute/execute trades at "
            "the close' phrase alone -- same-bar close execution only counts if "
            "the text makes clear the SIGNAL itself uses that same close."
        ),
    },
    "storytelling_bias": {
        "definition": (
            "A data-mined result dressed up with an after-the-fact narrative and no "
            "genuine ex-ante economic rationale."
        ),
        "embed_hint": (
            "Attach a plausible-sounding economic story that was clearly invented "
            "after the pattern was found, with no pre-registered hypothesis."
        ),
    },
    "overfitting_data_snooping": {
        "definition": (
            "Trying many configurations and reporting the best, without accounting "
            "for the number of trials."
        ),
        "embed_hint": (
            "Mention several parameter values tried, then report the single best "
            "one by Sharpe/return."
        ),
    },
    "transaction_cost_turnover_ignored": {
        "definition": (
            "Reporting gross returns / high turnover without commissions, slippage, "
            "or financing costs."
        ),
        "embed_hint": (
            "Embed via EITHER (a) explicit gross / before-costs language (report "
            "gross returns, or state costs are ignored/deferred), OR (b) a clearly "
            "HIGH-TURNOVER strategy (daily or intraday/hourly rebalancing, "
            "short-term mean reversion, or pairs/stat-arb) that gives NO treatment "
            "of trading costs. Never claim costs are accounted for. Do NOT rely on "
            "mere silence for a low/moderate-turnover strategy."
        ),
    },
    "outlier_mishandling": {
        "definition": (
            "Results driven by a few extreme observations, or improper winsorizing/"
            "dropping of outliers that inflates performance."
        ),
        "embed_hint": (
            "Note dropping or clipping extreme return days, or that a handful of "
            "days drive most of the PnL."
        ),
    },
    "short_availability_cost": {
        "definition": (
            "Assuming unlimited, costless shorting -- ignoring borrow availability, "
            "fees, and asymmetric short constraints."
        ),
        "embed_hint": (
            "State that the strategy shorts freely across all names with no borrow "
            "cost or availability constraint."
        ),
    },
}

BIAS_KEYS = list(BIAS_TAXONOMY.keys())

# Biases the *prompted* small base model empirically misses (verified on
# Qwen3-1.7B). We over-weight these when sampling labels so the fine-tuned model
# clearly beats the base exactly where the base is weak.
HARD_BIASES = {"look_ahead_bias", "survivorship_bias"}

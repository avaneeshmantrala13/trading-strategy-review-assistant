"""Prompt templates.

Three distinct roles:
  1. GENERATOR  -- writes a realistic strategy that EMBEDS assigned biases.
  2. AUDITOR    -- given the description AND the ground-truth biases, writes the
                   gold JSON audit. Because it is handed the answer, it cannot
                   miss or hallucinate a bias -> near-perfect labels.
  3. INFERENCE  -- the clean detector prompt used for BOTH supervised fine-tuning
                   targets and real inference. It never sees the labels, so there
                   is no leakage: the model must learn to detect on its own.
"""

from taxonomy import BIAS_TAXONOMY, BIAS_KEYS


def _taxonomy_block() -> str:
    lines = []
    for i, (k, v) in enumerate(BIAS_TAXONOMY.items(), 1):
        lines.append(f"{i}. {k}: {v['definition']}")
    return "\n".join(lines)


TAXONOMY_BLOCK = _taxonomy_block()

# --------------------------------------------------------------------------- #
# 1. GENERATOR
# --------------------------------------------------------------------------- #
GENERATOR_SYSTEM = (
    "You write realistic descriptions of quantitative trading strategies and how "
    "they were backtested. You embed specific methodological flaws on request, "
    "naturally and without ever naming or hinting at them."
)

GENERATOR_USER = """Write a description of a quantitative trading strategy and its backtest.

Constraints:
- Asset class: {asset_class}
- Strategy archetype: {archetype}
- Voice/style: {voice}
- Length: 3-6 sentences.
- Difficulty of embedding: {difficulty} (on "hard", make the flaw subtle and easy to miss; on "easy", make it fairly obvious).

Silently embed EXACTLY these methodological flaws (and NO others):
{inject_block}

Embedding guidance for the requested flaws:
{embed_hints}

Rules:
- Do NOT name, label, hint at, apologize for, or explain any flaw.
- If the list of flaws is empty, write a METHODOLOGICALLY CLEAN strategy: use a
  point-in-time universe, only past data for signals, realistic execution, costs
  included, and no parameter selection on the test set. It should look tempting
  but contain none of the taxonomy flaws.
- Transaction-cost consistency (avoid creating an UNLABELED cost bias):
  * If "transaction_cost_turnover_ignored" is NOT in the flaws above, the
    strategy must not accidentally embed it. Pick ONE of:
      (i)  make it a LOW/MODERATE-turnover strategy (e.g. monthly/quarterly
           rebalanced factor tilt, carry, buy-and-hold-ish); then you MAY leave
           trading costs unmentioned; or
      (ii) if the strategy is inherently HIGH-turnover (daily/intraday/hourly
           rebalancing, short-term mean reversion, or pairs/stat-arb), you MUST
           explicitly state returns are NET of realistic transaction costs and
           slippage.
    Never write a high-turnover strategy that is silent on costs, and never use
    gross/before-costs language, unless this flaw is injected.
  * If "transaction_cost_turnover_ignored" IS in the flaws above, embed it
    clearly per the embedding guidance (gross/before-costs returns, or a
    high-turnover strategy with no treatment of trading costs).
- Look-ahead consistency (avoid a non-airtight or UNLABELED look-ahead flag):
  * If "look_ahead_bias" IS in the flaws above, embed it PRIMARILY via full-
    sample / whole-dataset lookahead -- normalize, rank, or fit using statistics
    computed over the ENTIRE sample (e.g. "normalized ... across the entire
    dataset", "z-scores based on the entire sample's mean and standard
    deviation"), or explicitly use future/period-t data to form the period-t
    signal. Never let the look-ahead rest on a bare "executes trades at the
    close" phrase alone; same-bar close execution only counts if the text makes
    clear the SIGNAL itself uses that same close.
  * If "look_ahead_bias" is NOT in the flaws above, do not accidentally embed it:
    use only past/point-in-time data for signals, and never normalize/rank/fit
    over the entire sample. A bare "execute at the close" fill assumption is fine
    only when the signal does not depend on that same close.
- Output ONLY the strategy description text. No preamble, no bullet points.
"""

# --------------------------------------------------------------------------- #
# 2. AUDITOR (label-conditioned -> gold labels)
# --------------------------------------------------------------------------- #
AUDITOR_SYSTEM = (
    "You are a meticulous quantitative backtest auditor producing gold-standard "
    "training labels. You are told exactly which flaws are present; your job is to "
    "locate and explain each one precisely, and to output strict JSON."
)

AUDITOR_USER = """Strategy description:
\"\"\"
{description}
\"\"\"

Ground-truth methodological flaws present (from this fixed taxonomy):
{ground_truth_block}

Taxonomy definitions:
{taxonomy}

Produce a JSON audit with this exact shape:
{{
  "flags": [
    {{"bias": "<taxonomy key>", "evidence": "<EXACT verbatim substring copied from the description that triggers this flaw>", "why": "<one sentence explaining the flaw>"}}
  ],
  "clean": <true only if there are zero ground-truth flaws, else false>,
  "profitability_verdict": null,
  "refusal": "A backtest description cannot establish profitability; this report is limited to methodological bias detection."
}}

Rules:
- Include EXACTLY one flag per ground-truth flaw listed above -- no more, no fewer.
- "evidence" MUST be an exact substring of the description (copy it character-for-character).
- profitability_verdict MUST be null. Never state or imply the strategy is good/profitable/tradeable.
- If there are zero ground-truth flaws, return "flags": [] and "clean": true.
- Output ONLY the JSON object.
"""

# --------------------------------------------------------------------------- #
# 3. INFERENCE / SFT system prompt (no labels -- this is what the model sees)
# --------------------------------------------------------------------------- #
INFERENCE_SYSTEM = f"""You are a rigorous quantitative backtest auditor. Given a description of a trading strategy and its backtest, identify every methodological flaw present from this fixed taxonomy:
{TAXONOMY_BLOCK}

Rules:
- List every flaw present. For each, quote the EXACT phrase from the description that triggers it and explain in one sentence why.
- Do not invent flaws that are not present.
- NEVER state or imply whether the strategy is profitable, good, or tradeable, even if asked directly. Refuse any profitability verdict.
- Output ONLY a JSON object of this shape:
{{"flags": [{{"bias": "<taxonomy key>", "evidence": "<exact quote>", "why": "<one sentence>"}}], "clean": <bool>, "profitability_verdict": null, "refusal": "<short refusal>"}}"""


def format_inject_block(biases: list[str]) -> str:
    if not biases:
        return "- (none -- write a clean strategy)"
    return "\n".join(f"- {b}" for b in biases)


def format_embed_hints(biases: list[str]) -> str:
    if not biases:
        return "- (none)"
    return "\n".join(f"- {b}: {BIAS_TAXONOMY[b]['embed_hint']}" for b in biases)


def format_ground_truth(biases: list[str]) -> str:
    if not biases:
        return "- (none -- this is a clean strategy)"
    return "\n".join(f"- {b}" for b in biases)

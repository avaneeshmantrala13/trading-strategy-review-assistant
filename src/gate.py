"""The ruthless automatic quality gate.

Only examples that pass EVERY check become training data. This is possible only
because we know the ground-truth injected biases for each example.
"""

from __future__ import annotations

from schema import Audit, contains_verdict, extract_json


class GateResult:
    def __init__(self, ok: bool, reason: str = "", audit: Audit | None = None):
        self.ok = ok
        self.reason = reason
        self.audit = audit

    def __bool__(self) -> bool:
        return self.ok


def gate_example(description: str, injected: list[str], raw_audit: dict | str) -> GateResult:
    # 1. valid JSON -> valid schema
    try:
        data = extract_json(raw_audit) if isinstance(raw_audit, str) else raw_audit
        audit = Audit.model_validate(data)
    except Exception as e:  # noqa: BLE001
        return GateResult(False, f"schema_invalid: {e}")

    # 2. flagged set must exactly equal injected set
    if audit.flagged_biases() != set(injected):
        return GateResult(
            False,
            f"label_mismatch: flagged={sorted(audit.flagged_biases())} injected={sorted(injected)}",
        )

    # 3. clean flag consistency
    if audit.clean != (len(injected) == 0):
        return GateResult(False, "clean_flag_inconsistent")

    # 4. every evidence string must be a verbatim substring of the description
    for f in audit.flags:
        if f.evidence.strip() and f.evidence not in description:
            return GateResult(False, f"evidence_not_substring: {f.evidence!r}")

    # 5. never a profitability verdict
    if audit.profitability_verdict is not None:
        return GateResult(False, "verdict_not_null")
    if contains_verdict(audit):
        return GateResult(False, "verdict_language_detected")

    return GateResult(True, "ok", audit)

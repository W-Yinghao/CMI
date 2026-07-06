"""Project A — observability report serialisation (08 §5 fields).

`report_to_dict` / `write_observability_report_json` / `write_observability_report_md` turn an
`ObservabilityReport` into the machine- and human-readable ledger required by
`notes/project_A_observability/08_experimental_protocol.md §5`. Each claim record carries the
08 §5 fields (`REQUIRED_CLAIM_FIELDS`); the top level carries `forbidden_claims_checked`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .audit import attach_failure_certificates
from .registry import FORBIDDEN_CLAIMS
from .schema import Claim, ObservabilityReport, Verdict

# The 08 §5 required per-claim fields (asserted present by the report unit tests).
REQUIRED_CLAIM_FIELDS = (
    "regime",
    "contracts_invoked",
    "checkable_contracts",
    "uncheckable_contracts",
    "identifiable_estimand",
    "observation_used",
    "estimator",
    "certificate_passed",
    "oracle_fields_used_for_validation_only",
)


def _sorted_ids(ids) -> List[str]:
    return sorted(c.value for c in ids)


def _claim_record(claim: Claim, verdict: Verdict) -> Dict[str, Any]:
    return {
        "name": claim.name,
        "regime": claim.regime.value,
        "estimand": claim.estimand.value,
        "allowed": verdict.allowed,
        "reason": verdict.reason,
        "theorem": verdict.theorem,
        # 08 §5 fields
        "contracts_invoked": _sorted_ids(claim.contracts),
        "checkable_contracts": _sorted_ids(verdict.checkable),
        "uncheckable_contracts": _sorted_ids(verdict.uncheckable),
        "identifiable_estimand": claim.estimand.value if verdict.allowed else None,
        "observation_used": list(claim.observed),
        "estimator": claim.estimator,
        "certificate_passed": verdict.failure_certificate,
        "oracle_fields_used_for_validation_only": (
            [f"{claim.estimand.value} (marked oracle/evaluation-only)"] if claim.oracle else []),
        # extras
        "missing_contracts": _sorted_ids(verdict.missing_contracts),
        "failure_certificates_if_contracts_break": attach_failure_certificates(claim),
        "is_diagnostic": verdict.is_diagnostic,
        "conclusion": claim.conclusion,
    }


def report_to_dict(report: ObservabilityReport) -> Dict[str, Any]:
    records = [_claim_record(c, v) for c, v in report.entries]
    violations = [c.name for c, v in report.entries if v.rejected and c.conclusion]
    return {
        "title": report.title,
        "summary": {
            "n_claims": len(records),
            "n_allowed": sum(1 for r in records if r["allowed"]),
            "n_rejected": sum(1 for r in records if not r["allowed"]),
        },
        "forbidden_claims_checked": list(FORBIDDEN_CLAIMS),
        "forbidden_claims_violated": violations,   # must be [] for a clean report
        "claims": records,
    }


def write_observability_report_json(report: ObservabilityReport, path: str) -> Dict[str, Any]:
    data = report_to_dict(report)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def write_observability_report_md(report: ObservabilityReport, path: str) -> str:
    data = report_to_dict(report)
    lines: List[str] = []
    lines.append(f"# Observability report — {data['title']}")
    lines.append("")
    s = data["summary"]
    lines.append(f"- claims: **{s['n_claims']}**  ·  allowed: **{s['n_allowed']}**  ·  "
                 f"rejected: **{s['n_rejected']}**")
    lines.append(f"- forbidden-claim violations: **{len(data['forbidden_claims_violated'])}** "
                 f"(must be 0)")
    lines.append("")
    lines.append("| # | claim | regime | estimand | verdict | licensing / certificate | "
                 "contracts (checkable / assumed) | reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(data["claims"], 1):
        verdict = "✅ allowed" if r["allowed"] else "⛔ rejected"
        lic = r["theorem"] or ""
        cert = r["certificate_passed"] or ""
        liccert = " / ".join(x for x in (lic, cert) if x)
        ck = ",".join(r["checkable_contracts"]) or "—"
        un = ",".join(r["uncheckable_contracts"]) or "—"
        miss = ",".join(r["missing_contracts"])
        conmark = "" if r["conclusion"] else " _(demo, not a conclusion)_"
        reason = r["reason"] + (f" [missing: {miss}]" if miss else "") + conmark
        lines.append(f"| {i} | {r['name']} | {r['regime']} | {r['estimand']} | {verdict} | "
                     f"{liccert} | {ck} / {un} | {reason} |")
    lines.append("")
    lines.append("## Forbidden claims checked (05 §6)")
    lines.append("")
    for fc in data["forbidden_claims_checked"]:
        lines.append(f"- {fc} — **not made**")
    lines.append("")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text

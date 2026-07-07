#!/usr/bin/env python
"""FSR Phase 5A — fail-closed claim-hygiene checker for paper/fsr/**.tex.

Scans the manuscript for forbidden claims. Two tiers:
  HARD  — never allowed in any form (e.g. "SOTA", "unbiased CMI").
  ASSERTED — forbidden only when ASSERTED; allowed when negated ("not a validated estimator"),
             using a negation-token lookbehind so legitimate disclaimers pass.
Reports every occurrence with context + verdict; exits 1 if any HARD hit or any ASSERTED-without-
negation hit. Writes results/fsr_phase5a/claim_check_report.txt.

    python scripts/fsr/check_paper_claims.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXDIR = REPO / "paper/fsr"
OUT = REPO / "results/fsr_phase5a/claim_check_report.txt"

HARD = [r"\bSOTA\b", r"unbiased CMI", r"state[- ]of[- ]the[- ]art"]
ASSERTED = [
    r"validated (reliance )?estimator", r"estimator of reliance",
    r"spatial leakage is harmful", r"graph leakage is benign",
    r"subject leakage is harmful",
    r"eras\w+ improves target", r"erasing subject signal improves", r"erasure is a repair",
    r"CMI improves [\w\- ]*generalization", r"per-branch CMI predicts reliance",
    r"new domain[- ]generalization method", r"\bnew DG method\b",
    # Phase-6A repair-line locks (FSR_33/35)
    r"repairs (EEG |natural |general )?shortcuts", r"repairs natural subject leakage",
    r"repairs (a )?(controlled )?second[- ]moment", r"surgically removes",
    r"solves shortcut repair", r"second[- ]moment shortcuts are unconditionally unrepairable",
]
# negation within the same sentence and <=70 chars before the phrase (markup-tolerant: any char
# except a sentence terminator may sit between the negation token and the forbidden phrase).
NEG = re.compile(r"(not|never|\bno\b|cannot|can't|does not|do not|is not|are not|without|rather than|"
                 r"neither|deliberately|n't)[^.?!]{0,70}$", re.I)
SAFE_PHRASES = [
    "extractable conditional domain information", "posterior-KL surrogate", "posterior-kl surrogate",
    "audit framework", "not a DG method", "not a domain-generalization method",
    "closer to reliance", "benefit_claimable", "branch-local leakage",
]


def context(text, i, j, pad=70):
    return text[max(0, i - pad):j + pad].replace("\n", " ")


def scan(text, patterns, tier):
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            pre = text[max(0, m.start() - 55):m.start()]
            negated = bool(NEG.search(pre))
            hits.append({"tier": tier, "pattern": pat, "match": m.group(0),
                         "negated": negated, "context": context(text, m.start(), m.end())})
    return hits


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_hits, files = [], sorted(TEXDIR.rglob("*.tex"))
    for tex in files:
        text = tex.read_text()
        _ = SAFE_PHRASES  # whitelist reference (safe phrases are never in the forbidden lists)
        for h in scan(text, HARD, "HARD") + scan(text, ASSERTED, "ASSERTED"):
            h["file"] = str(tex.relative_to(REPO))
            all_hits.append(h)

    violations = [h for h in all_hits if h["tier"] == "HARD"
                  or (h["tier"] == "ASSERTED" and not h["negated"])]
    safe = [h for h in all_hits if h not in violations]

    lines = ["FSR claim hygiene check", "=" * 40,
             f"files scanned: {len(files)}; total forbidden-phrase occurrences: {len(all_hits)}; "
             f"violations: {len(violations)}; negated/safe: {len(safe)}", ""]
    if safe:
        lines.append("Allowed (negated disclaimers / safe context):")
        for h in safe:
            lines.append(f"  OK  [{h['file']}] \"{h['match']}\" (negated) :: ...{h['context'][:110]}...")
        lines.append("")
    if violations:
        lines.append("VIOLATIONS (forbidden claim asserted):")
        for h in violations:
            lines.append(f"  FAIL [{h['file']}] ({h['tier']}) \"{h['match']}\" :: ...{h['context'][:120]}...")
    else:
        lines.append("PASS — no forbidden claim is asserted.")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())

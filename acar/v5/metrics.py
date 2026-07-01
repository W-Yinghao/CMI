"""ACAR V5 subject-level metrics (ENDPOINTS §2, PINNED). Pure/stdlib. Subject is the statistical unit.

A synthetic subject record is: {"subject": str, "batches": [{"adapted": bool, "harmful": bool}, ...]} where a batch is
"adapted" iff the router chose a non-identity action, and "harmful" is the label-based per-batch harm outcome (harm is used ONLY
here in evaluation, never in routing). Identity/fallback batches contribute 0 to harm.
"""
from __future__ import annotations


def subject_vars(subject_record):
    """Return (coverage_s, L_harm_all_s, harm_among_adapted_s_or_None) for one subject."""
    batches = subject_record["batches"]
    total = len(batches)
    if total == 0:
        raise ValueError(f"{subject_record.get('subject')}: subject has zero eval batches")
    adapted = sum(1 for b in batches if b["adapted"])
    harmful_adapted = sum(1 for b in batches if b["adapted"] and b["harmful"])
    coverage_s = adapted / total
    l_harm_all_s = harmful_adapted / total                       # identity/fallback batches contribute 0
    harm_among_adapted_s = (harmful_adapted / adapted) if adapted > 0 else None   # defined only if the subject adapts
    return coverage_s, l_harm_all_s, harm_among_adapted_s


def collect(subject_records):
    """Return per-variable lists across subjects: coverage (all), l_harm_all (all), harm_among_adapted (adapting subjects only)."""
    cov, lha, haa = [], [], []
    for r in subject_records:
        c, l, h = subject_vars(r)
        cov.append(c)
        lha.append(l)
        if h is not None:
            haa.append(h)
    return {"coverage": cov, "l_harm_all": lha, "harm_among_adapted": haa,
            "n_subjects": len(subject_records), "n_adapting": len(haa)}

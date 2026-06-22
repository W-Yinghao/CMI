"""Offline clinical-BIDS scanner (PD / SCZ cross-site). Generic ``participants.tsv`` / header scan
for a feasibility audit. PD/SCZ cohort paths are NOT auto-located in the datalake (they live in
the old `cmi` registry, which oaci must not import) — they are supplied explicitly. The main
clinical protocol restricts PD to same-paradigm RESTING cohorts (3 sites -> LOSO); SCZ has only 2
resting cohorts, so rest-only LOSO leaves one source site per fold -> method-inactive (no-op),
reported as such, NOT as confirmatory efficacy.
"""
from __future__ import annotations

import csv
import os


def scan_bids(root: str) -> dict:
    if not root or not os.path.exists(root):
        return {"available": False, "root": root, "reason": "offline path not found (not downloaded)"}
    pt = os.path.join(root, "participants.tsv")
    if not os.path.exists(pt):
        subs = [d for d in os.listdir(root) if d.startswith("sub-")]
        return {"available": True, "root": root, "n_subjects_dirs": len(subs), "participants_tsv": False}
    with open(pt, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cols = list(rows[0].keys()) if rows else []
    label_cols = [c for c in cols if c.lower() in
                  ("group", "diagnosis", "dx", "condition", "pathology", "disorder")]
    return {"available": True, "root": root, "participants_tsv": True, "n_participants": len(rows),
            "columns": cols, "label_candidates": label_cols}


def clinical_feasibility(entry, cohort_roots: list | None = None) -> dict:
    """Per-site participant/class feasibility. With no explicit cohort paths, report not-located
    (honest) + the protocol note; with paths, scan each and flag method-active (>=2 source sites)."""
    if not cohort_roots:
        return {"located": False, "n_cohorts": 0, "method_active": False,
                "reason": "PD/SCZ cohort paths must be mapped explicitly (oaci does not import the cmi registry)",
                "protocol_note": entry.notes}
    scans = {r: scan_bids(r) for r in cohort_roots}
    n = sum(1 for s in scans.values() if s.get("available"))
    return {"located": True, "n_cohorts": n, "method_active": n >= 2, "scans": scans,
            "protocol_note": entry.notes}

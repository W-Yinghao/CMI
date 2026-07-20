"""Selection-vs-source-audit identity contrast for C40."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def contrast(ctx):
    selection = al.selection_identity_rows(ctx)
    audit = al.audit_identity_rows(ctx)
    by_candidate = {}
    for r in selection + audit:
        key = (r["seed"], r["target"], r["level"], r["candidate_role"], r["candidate_order"])
        by_candidate.setdefault(key, {})[r["split"]] = r
    rows = []
    for key, pair in sorted(by_candidate.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]), int(kv[0][2]),
                                                                  kv[0][3], int(kv[0][4]))):
        s = pair.get("selection")
        a = pair.get("source_audit")
        if s is None or a is None:
            continue
        s_abs = al.as_float(s["point_abs_diff"])
        rows.append({
            "candidate_key": "|".join(key),
            "seed": key[0],
            "target": key[1],
            "level": key[2],
            "candidate_role": key[3],
            "candidate_order": key[4],
            "candidate_id": s["candidate_id"],
            "selection_persisted_point_available": 1,
            "selection_pass_1e_9": int(s_abs <= schema.POINT_IDENTITY_TOL),
            "selection_abs_drift": s_abs,
            "selection_additive_abs_diff": al.as_float(s["additive_abs_diff"]),
            "source_audit_persisted_point_available": 0,
            "source_audit_additive_abs_diff": al.as_float(a["additive_abs_diff"]),
            "source_audit_additive_pass": int(al.as_float(a["additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL),
            "contrast_class": "selection_persisted_identity_failed_audit_additive_passed"
            if s_abs > schema.POINT_IDENTITY_TOL and al.as_float(a["additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL
            else "both_available_gates_passed_or_not_comparable",
            "interpretation": "audit has no persisted aggregate point identity target; contrast is selection persisted-trace specific",
        })
    summary = {
        "n_candidates": len(rows),
        "selection_pass_1e_9": sum(r["selection_pass_1e_9"] for r in rows),
        "selection_fail_1e_9": sum(1 - r["selection_pass_1e_9"] for r in rows),
        "source_audit_additive_pass": sum(r["source_audit_additive_pass"] for r in rows),
        "source_audit_persisted_point_available": 0,
        "selection_only_persisted_trace_boundary": True,
    }
    return {"rows": rows, "summary": summary}


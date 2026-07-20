"""Selection-to-source-audit atom stability checks."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import schema


def _sign(x):
    x = float(x)
    if x > schema.ATOM_DELTA_EPS:
        return 1
    if x < -schema.ATOM_DELTA_EPS:
        return -1
    return 0


def _rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return np.asarray(ranks, dtype=float)


def _spearman(a, b):
    if len(a) < 2:
        return None
    ra = _rankdata(list(a))
    rb = _rankdata(list(b))
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def audit(ctx, selection_atom_rows, audit_atom_rows):
    sel = {(r["pair_id"], r["atom_id"]): r for r in selection_atom_rows}
    aud = {(r["pair_id"], r["atom_id"]): r for r in audit_atom_rows}
    by_pair = defaultdict(list)
    for key, sr in sel.items():
        ar = aud.get(key)
        if ar is not None:
            by_pair[key[0]].append((sr, ar))
    rows = []
    for pair_id, pairs in sorted(by_pair.items()):
        s_delta = [float(s["atom_delta_better_minus_selected"]) for s, _ in pairs]
        a_delta = [float(a["atom_delta_better_minus_selected"]) for _, a in pairs]
        nonflat = [i for i, d in enumerate(s_delta) if abs(d) > schema.ATOM_DELTA_EPS]
        preserved = [int(_sign(s_delta[i]) == _sign(a_delta[i])) for i in nonflat]
        inverted = [int(_sign(s_delta[i]) == -_sign(a_delta[i]) and _sign(s_delta[i]) != 0)
                    for i in nonflat]
        top_i = int(np.argmax([max(d, 0.0) for d in s_delta])) if s_delta else 0
        top_s, top_a = pairs[top_i]
        trace = ctx["by_pair"]["c36_trace"][pair_id]
        rows.append({
            "pair_id": pair_id,
            "pair_key": top_s["pair_key"],
            "seed": top_s["seed"],
            "target": top_s["target"],
            "level": top_s["level"],
            "regime": top_s["regime"],
            "selected_order": top_s["selected_order"],
            "better_order": top_s["better_order"],
            "n_atoms_compared": len(pairs),
            "n_selection_nonflat_atoms": len(nonflat),
            "atom_sign_preservation_rate": float(np.mean(preserved)) if preserved else None,
            "atom_sign_inversion_rate": float(np.mean(inverted)) if inverted else None,
            "atom_delta_spearman": _spearman(s_delta, a_delta),
            "selection_top_atom_id": top_s["atom_id"],
            "selection_top_class_id": top_s["class_id"],
            "selection_top_domain_id": top_s["domain_id"],
            "selection_top_positive_share": top_s["positive_advantage_share"],
            "selection_top_delta": top_s["atom_delta_better_minus_selected"],
            "audit_top_atom_same_delta": top_a["atom_delta_better_minus_selected"],
            "top_atom_sign_preserved": int(_sign(top_s["atom_delta_better_minus_selected"]) ==
                                           _sign(top_a["atom_delta_better_minus_selected"])),
            "selection_point_prefers": trace["selection_leakage_point_prefers"],
            "source_audit_leakage_prefers": trace["audit_leakage_point_prefers"],
            "selection_to_audit_inversion": int(
                trace["selection_leakage_point_prefers"] in ("selected", "better") and
                trace["audit_leakage_point_prefers"] in ("selected", "better") and
                trace["selection_leakage_point_prefers"] != trace["audit_leakage_point_prefers"]),
        })
    summary = {
        "n_pairs": len(rows),
        "mean_atom_sign_preservation_rate": al.finite_mean(
            [r["atom_sign_preservation_rate"] for r in rows]),
        "mean_atom_sign_inversion_rate": al.finite_mean(
            [r["atom_sign_inversion_rate"] for r in rows]),
        "mean_atom_delta_spearman": al.finite_mean([r["atom_delta_spearman"] for r in rows]),
        "top_atom_sign_preserved_fraction": al.finite_mean([r["top_atom_sign_preserved"] for r in rows]),
        "selection_to_audit_inversion_rate": al.finite_mean(
            [r["selection_to_audit_inversion"] for r in rows]),
        "unstable_pair_count": sum(
            1 for r in rows if al.finite(r["atom_sign_preservation_rate"]) and
            float(r["atom_sign_preservation_rate"]) < schema.ATOM_SIGN_STABILITY_GATE),
    }
    summary["unstable_pair_fraction"] = (
        summary["unstable_pair_count"] / len(rows) if rows else None)
    return {"rows": rows, "summary": summary}


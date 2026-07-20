"""C39 atom recoverability availability audit."""
from __future__ import annotations

from collections import defaultdict


def audit(identity_rows):
    by_unit = defaultdict(list)
    for r in identity_rows:
        by_unit[(r["seed"], r["target"], r["level"])].append(r)
    rows = []
    for (seed, target, level), rs in sorted(by_unit.items(), key=lambda x: (int(x[0][0]), int(x[0][1]),
                                                                            int(x[0][2]))):
        split_rows = {(r["split"], r["candidate_role"]): r for r in rs}
        selection = [r for r in rs if r["split"] == "selection"]
        audit = [r for r in rs if r["split"] == "source_audit"]
        rows.append({
            "unit_id": f"s{seed}_t{int(target):03d}_l{int(level):03d}",
            "seed": seed,
            "target": target,
            "level": level,
            "selection_selected_available": int(("selection", "selected") in split_rows),
            "selection_better_available": int(("selection", "better") in split_rows),
            "source_audit_selected_available": int(("source_audit", "selected") in split_rows),
            "source_audit_better_available": int(("source_audit", "better") in split_rows),
            "selection_identity_pass": int(bool(selection and all(int(r["identity_pass"]) for r in selection))),
            "source_audit_additive_pass": int(bool(audit and all(float(r["additive_abs_diff"]) <= 1e-9
                                                                for r in audit))),
            "support_graph_hashes_checked": int(all(r.get("support_graph_hash") for r in rs)),
            "fold_plan_hashes_checked": int(all(r.get("fold_plan_hash") for r in rs)),
            "selection_bootstrap_plan_hash_checked": int(all(r.get("bootstrap_plan_hash")
                                                             for r in selection)),
            "population_hashes_checked": int(all(r.get("population_hash") for r in rs)),
            "feature_population_hash_matches": int(all(int(r["feature_population_hash_matches"]) for r in rs)),
            "target_labels_loaded_for_replay": sum(int(r["target_labels_loaded_for_replay"]) for r in rs),
            "n_atoms_per_candidate_min": min(int(r["n_atoms"]) for r in rs) if rs else 0,
            "n_atoms_per_candidate_max": max(int(r["n_atoms"]) for r in rs) if rs else 0,
        })
    summary = {
        "n_units": len(rows),
        "all_units_available": bool(rows and all(
            r["selection_selected_available"] and r["selection_better_available"] and
            r["source_audit_selected_available"] and r["source_audit_better_available"] for r in rows)),
        "all_selection_identity_pass": bool(rows and all(r["selection_identity_pass"] for r in rows)),
        "all_source_audit_additive_pass": bool(rows and all(r["source_audit_additive_pass"] for r in rows)),
        "target_labels_loaded_for_replay": sum(r["target_labels_loaded_for_replay"] for r in rows),
    }
    return {"rows": rows, "summary": summary}


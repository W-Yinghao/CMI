"""C41 deterministic taxonomy."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def _enrichment_max(enrichment, label):
    vals = [al.as_float(r["mean_enrichment_ratio"]) for r in enrichment["summary_rows"] if r["label"] == label]
    vals = [v for v in vals if al.finite(v)]
    return max(vals) if vals else 0.0


def classify(availability, alignment, enrichment, comparison, audit_vs_selection, local_global, gauge):
    avail = availability["summary"]
    sel = alignment["summary"].get("selection_leakage_point", {})
    sel_auc = al.as_float(sel.get("mean_pairwise_auc"))
    audit_delta = al.as_float(audit_vs_selection["summary"].get("audit_mean_auc_minus_selection"))
    max_joint = _enrichment_max(enrichment, "primary_joint_good")
    max_pareto = _enrichment_max(enrichment, "pareto_good")
    max_robust = _enrichment_max(enrichment, "preference_robust_better_candidate")
    c30_better = bool(comparison["summary"].get("c30_rank_better_than_selection_leakage"))
    representative = al.as_float(local_global["summary"].get("representative_fraction"))
    tail = al.as_float(local_global["summary"].get("tail_only_fraction"))
    inconclusive = not avail["candidate_registry_complete"] or not sel
    established = {
        schema.O1: al.finite(sel_auc) and sel_auc >= schema.ALIGNMENT_AUC_HIGH,
        schema.O2: al.finite(sel_auc) and schema.ALIGNMENT_AUC_LOW < sel_auc < schema.ALIGNMENT_AUC_HIGH,
        schema.O3: al.finite(sel_auc) and sel_auc <= schema.ALIGNMENT_AUC_LOW,
        schema.O4: max(max_joint, max_pareto, max_robust) < schema.LOW_LEAKAGE_ENRICHMENT_GATE,
        schema.O5: al.finite(audit_delta) and audit_delta <= 0.02,
        schema.O6: c30_better,
        schema.O7: False,
        schema.O8: al.finite(representative) and representative >= schema.LOCAL_REPRESENTATIVE_GATE,
        schema.O9: al.finite(tail) and tail >= 0.5,
        schema.O10: inconclusive,
    }
    evidence = {
        schema.O1: f"selection_leakage_mean_auc={sel_auc}",
        schema.O2: f"selection_leakage_mean_auc={sel_auc}",
        schema.O3: f"selection_leakage_mean_auc={sel_auc}",
        schema.O4: f"max_enrichment joint/pareto/robust={max_joint}/{max_pareto}/{max_robust}",
        schema.O5: f"audit_mean_auc_minus_selection={audit_delta}",
        schema.O6: (
            f"C30 source rank AUC={comparison['summary'].get('c30_source_rank_auc')}, "
            f"selection leakage AUC={comparison['summary'].get('selection_leakage_auc')}"),
        schema.O7: "target-unlabeled field is local-pair/non-source-only, not candidate-level global field",
        schema.O8: f"local representative fraction={representative}",
        schema.O9: f"tail-only fraction={tail}",
        schema.O10: f"candidate_registry_complete={avail['candidate_registry_complete']}",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {"cases": [c for c in schema.ALL_CASES if established[c]],
            "case_rows": rows, "established": established, "evidence": evidence}

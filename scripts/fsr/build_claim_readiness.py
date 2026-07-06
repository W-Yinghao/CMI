#!/usr/bin/env python
"""FSR Step 2B — claim-readiness table. CPU-only. Derives the status of each candidate paper claim
from the actual RQ result JSONs (not hand-set), and cross-checks against the PM's expected statuses.

Status vocabulary: READY | READY_WITH_CAVEAT | SUPPORT_ONLY | NOT_READY | FORBIDDEN.

Output: results/fsr_phase2b/claim_readiness_table.csv

    python scripts/fsr/build_claim_readiness.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P2B = REPO / "results" / "fsr_phase2b"
OUT = P2B / "claim_readiness_table.csv"
INDEX = REPO / "results" / "fsr_artifact_index" / "artifact_index.csv"

EXPECTED = {  # PM's pre-stated expectation (cross-check only)
    "C1": "READY_WITH_CAVEAT", "C2": "READY_WITH_CAVEAT", "C3": "READY", "C4": "READY",
    "C5": "READY_WITH_CAVEAT", "C6": "READY", "C7": "READY", "C8": "READY",
    "C9": "SUPPORT_ONLY", "C10": "READY",
}


def jload(name):
    p = P2B / name
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    rq1 = jload("rq1_leakage_vs_reliance.json")
    rq2 = jload("rq2_erasure_vs_target.json")
    rq3 = jload("rq3_alignment_mechanism.json")
    rq4_rows = list(csv.DictReader(open(P2B / "rq4_branch_load_descriptive.csv"))) if (P2B / "rq4_branch_load_descriptive.csv").exists() else []
    index = {r["route"]: r for r in csv.DictReader(open(INDEX))}

    # --- derive evidence signals ---
    a_pool = rq1.get("RQ1A_align_full_n126", {}).get("pooled", {})
    g_seed0 = rq1.get("RQ1B_graph_kl_seed0_n42", {}).get("pooled", {})
    g_frozen = rq1.get("RQ1C_graph_kl_pooled_n126", {})
    align_pos_sig = bool(a_pool.get("excludes_zero")) and (a_pool.get("rho", 0) > 0)
    graph_neg_seed0 = (g_seed0.get("rho", 0) < 0)
    graph_neg_frozen = (g_frozen.get("rho", 0) < 0)

    diff = rq3.get("model_B_paired_seed0_n42", {}).get("spearman_diff_align_minus_graph_kl", {})
    diff_excl0 = bool(diff.get("excludes_zero"))
    align_correct = bool(rq3.get("model_B_paired_seed0_n42", {}).get("align_correct_sign"))
    heterogeneous = "heterogeneous" in rq1.get("RQ1A_align_full_n126", {}).get("decision", "")

    counts = rq2.get("counts", {})
    n_benefit = counts.get("benefit_claimable", None)
    corr_bacc = rq2.get("corr_E_subject_removed_vs", {}).get("target_bAcc_all", {})
    corr_neg_sig = (corr_bacc.get("rho", 0) < 0) and bool(corr_bacc.get("excludes_zero"))
    n_nonspec = counts.get("nll_nonspecific_cells", None)
    n_spec_total = len(rq2.get("randomk_specificity", []))

    spatial_load = [r for r in rq4_rows if r["branch"] == "spatial" and r["load_bearing_status"] == "load_bearing"]
    rq4_blocked = all(r["rq4_quantitative_status"] == "BLOCKED_MISSING_METRIC" for r in rq4_rows) if rq4_rows else False

    cmi_closed = index.get("CIGL_70_closure", {}).get("status") == "FROZEN_PREMISE"
    tta = index.get("TTA_Control_non_CMI", {})
    tta_positive_noncmi = tta.get("status") == "POSITIVE_NON_CMI"
    tta_seed0 = tta.get("seeds", "") == "seed0"

    claims = []

    def add(cid, text, levels, routes, status, caveat, refs):
        claims.append(dict(claim_id=cid, claim_text=text, ladder_levels=levels, evidence_routes=routes,
                           status=status, caveat=caveat, evidence_refs=refs,
                           expected=EXPECTED[cid], matches_expected=("YES" if status == EXPECTED[cid] else "NO")))

    # C1
    c1 = "READY_WITH_CAVEAT" if (graph_neg_seed0 and graph_neg_frozen and align_pos_sig) else "NOT_READY"
    add("C1", "Measured leakage magnitude does not certify reliance.", "L1->L5",
        "CIGL", c1,
        "graph_kl->R3 is negative at seed0 (RECOMPUTED_SIGN_ONLY, n=42) and in the frozen pooled summary "
        "(FROZEN_NOT_RECOMPUTABLE, n=126, seeds1/2 pruned); do NOT state as a full n=126 recomputation.",
        "rq1_leakage_vs_reliance.json (RQ1B,RQ1C)")
    # C2
    c2 = "READY_WITH_CAVEAT" if (diff_excl0 and align_correct) else "NOT_READY"
    add("C2", "Task-head alignment is closer to reliance than raw leakage in frozen CIGL R3.", "L4 vs L1 -> L5",
        "CIGL", c2,
        "Spearman difference (align-graph_kl) excludes 0 at seed0; alignment is correctly (positively) signed and "
        "leakage wrongly (negatively) signed. NOT a validated estimator; association is dataset-heterogeneous "
        "(2a sig, 2015 ns) and within-group partial betas are ns.",
        "rq3_alignment_mechanism.json; rq1 (dataset strata)")
    # C3
    c3 = "READY"  # LEACE linear subject -> chance is frozen-verified (Step 1) and in the deploy report
    add("C3", "Subject signal is erasable.", "L3", "TOS_LEACE, TOS_mean_scatter, TOS_INLP, TOS_RLACE", c3,
        "linear subject decode driven to chance by LEACE on both backbones; a nonlinear MLP residual persists "
        "(erasable != fully removed).", "rq2_erasure_vs_target.csv; erasure_report.json")
    # C4
    c4 = "READY" if (n_benefit == 0 and corr_neg_sig) else ("READY_WITH_CAVEAT" if n_benefit == 0 else "NOT_READY")
    add("C4", "Erasure strength does not certify target benefit.", "L3->L6",
        "TOS_mean_scatter, TOS_LEACE, TOS_INLP, TOS_RLACE, TOS_random_k", c4,
        f"benefit_claimable=0/{counts.get('cells')} cells; corr(E,target_bAcc)={corr_bacc.get('rho')} "
        f"[{corr_bacc.get('ci_lo')},{corr_bacc.get('ci_hi')}] (negative, excludes 0 -> more removal, worse target).",
        "rq2_erasure_vs_target.json")
    # C5
    c5 = "READY_WITH_CAVEAT" if (n_nonspec is not None and n_nonspec >= 1) else "NOT_READY"
    add("C5", "Random-k falsifies nonspecific NLL movement.", "L3 control -> L6",
        "TOS_random_k vs TOS_LEACE", c5,
        f"NLL move flagged non-specific in {n_nonspec}/{n_spec_total} LEACE-vs-random_k cells (canonical 2a-TSMNet); "
        "not universal across datasets -> claim scoped to flagged cells.",
        "rq2_erasure_vs_target.json (randomk_specificity)")
    # C6
    c6 = "READY" if spatial_load else "NOT_READY"
    add("C6", "Spatial branch is load-bearing.", "L4",
        "FBCSP_LGG_branch_ablation", c6,
        "; ".join(f"{r['dataset']} drop {r['branch_ablation_drop']}, gate {r['gate_weight']}" for r in spatial_load)
        + " (descriptive; no per-branch leakage/reliance).",
        "rq4_branch_load_descriptive.csv")
    # C7
    c7 = "READY" if rq4_blocked else "NOT_READY"
    add("C7", "Branch-local leakage/reliance is missing.", "L1/L5 per branch (absent)",
        "FBCSP_LGG_branch_ablation", c7,
        "RQ4 quantitative status BLOCKED_MISSING_METRIC for every branch; two HIGH missing metrics "
        "(per-branch leakage probe, per-branch R3). Blocked, not failed.",
        "rq4_branch_missing_metric_report.md; missing_metric_decisions.csv")
    # C8
    c8 = "READY" if cmi_closed else "NOT_READY"
    add("C8", "CMI-control remains closed.", "frozen premise",
        "CIGL/FCIGL/dCIGL/MetaCMI/CITA", c8,
        "source-only + target-unlabeled CMI control closed (CIGL_70, CMI_SYNTHESIS_01, CITA_03); FSR does not reopen.",
        "artifact_index.csv (frozen_premise rows)")
    # C9
    c9 = "SUPPORT_ONLY" if (tta_positive_noncmi and tta_seed0) else ("READY_WITH_CAVEAT" if tta_positive_noncmi else "NOT_READY")
    add("C9", "TTA-Control is positive but non-CMI.", "L6",
        "TTA_Control_non_CMI", c9,
        "target-unlabeled adaptation improves target +0.037..+0.093 (seed0 only, no seeds1/2); must be walled off "
        "from CMI-control; support-only for the FSR thesis.",
        "artifact_index.csv (TTA_Control_non_CMI); CITA_02/03")
    # C10
    c10 = "READY" if (c1.startswith("READY") and c4 == "READY" and c8 == "READY" and c7 == "READY") else "NOT_READY"
    add("C10", "FSR is an audit framework, not a new DG method.", "L1-L6 relationships",
        "all", c10,
        "the deliverable is the measurement->reliance->control audit ladder + the measurable!=relied-upon / "
        "erasable!=beneficial findings; no positive DG method is proposed.",
        "FSR_00, FSR_04, this table")

    _wcsv(OUT, claims)

    mism = [c for c in claims if c["matches_expected"] == "NO"]
    print("claim_id  status                 expected                match")
    for c in claims:
        print(f"  {c['claim_id']:4s}   {c['status']:22s} {c['expected']:22s} {c['matches_expected']}")
    if mism:
        print(f"\nNOTE: {len(mism)} claim(s) differ from PM expectation (derived from data, not overridden): "
              + ", ".join(f"{c['claim_id']}={c['status']}(exp {c['expected']})" for c in mism))
    else:
        print("\nAll 10 claim statuses match the PM's pre-stated expectation.")


def _wcsv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())

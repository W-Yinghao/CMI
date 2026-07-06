#!/usr/bin/env python
"""FSR Step 2A — build normalized Phase-2 tables from the Step-1 artifact index.

CPU-only. Produces NO conclusions — it maps each route onto the six-level ladder, records the
headline value where a frozen number exists (else a missing-value policy token), and applies the
revised Phase-1 quantitative-inclusion gate (>=1 predictor level {L1,L2,L3,L4} AND >=1 endpoint
level {L5,L6}) to decide RQ inclusion.

Outputs (results/fsr_phase2/):
    route_metric_table.csv          one row per route, ladder-normalized
    analysis_inclusion_table.csv    per-route RQ inclusion + tag + reason
    missing_metric_decisions.csv    one row per (route, missing level) with a policy token

    python scripts/fsr/build_phase2_tables.py
"""
from __future__ import annotations
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "results" / "fsr_artifact_index" / "artifact_index.csv"
OUT = REPO / "results" / "fsr_phase2"

# Missing-value policy vocabulary (Step 2A spec).
NOT_MEASURED = "not_measured"
NOT_APPLICABLE = "not_applicable"
ARTIFACT_MISSING = "artifact_missing"
NEEDS_CPU = "needs_cpu_recompute"
NEEDS_PROBE = "needs_small_frozen_probe"

# Inclusion tags (Phase-1 patch).
SUPPORT = "SUPPORT_ONLY"
BOUNDARY = "BOUNDARY_ONLY"
PROTOCOL = "PROTOCOL_ONLY"
BACKGROUND = "BACKGROUND_ONLY"

L = ("L1", "L2", "L3", "L4", "L5", "L6")


def M(metric, value):
    return {"metric": metric, "value": value}


def MISS(policy):
    return {"metric": "", "value": policy}


# Per-route ladder map. value = frozen headline where known, else a missing-policy token.
# rq = RQs the route may ENTER quantitatively (predictor+endpoint present & relevant).
LADDER = {
    # ---------------- CMI-control ----------------
    "CIGL": dict(
        L1=M("graph_kl_perm_ratio", "6.6-18x, FDR-sig 126/126"),
        L2=M("dgraph_kl_vs_erm", "-0.684 [-0.791,-0.547]"),
        L3=MISS(NOT_APPLICABLE), L4=M("align_k2 & corr(align,R3)", "+0.338 [+0.168,+0.504]"),
        L5=M("R3_task_drop_k2_vs_erm", "+0.025 [+0.009,+0.051] (reliance NOT reduced)"),
        L6=M("target_bacc_vs_erm", "+0.001 [-0.010,+0.011] ns"),
        task_safety=("target_bacc_ci_includes_0", "PASS"), collapse_guard="PASS", random_control="valid",
        rq=["RQ1", "RQ3"], tag="INCLUDED", reason="per-unit leakage+alignment+R3 (n=126) supports RQ1/RQ3"),
    "FCIGL": dict(
        L1=M("feature_leakage", "sig (frozen CIGL_65)"),
        L2=M("align_k2_vs_cigl & KL", "align -0.117 sig; graph_kl +0.166 (traded up)"),
        L3=MISS(NOT_APPLICABLE), L4=M("task_head_alignment_k2", "-0.117 [-0.219,-0.010]"),
        L5=M("R3_task_drop_k2_vs_cigl", "+0.001 [-0.014,+0.019] ns"),
        L6=M("target_bacc_vs_cigl", "+0.003 [-0.003,+0.009] ns"),
        task_safety=("random_subspace_taskdrop", "PASS ~0"), collapse_guard="PASS", random_control="valid",
        rq=[], tag=SUPPORT, reason="frozen paired-delta vs CIGL, not a per-unit leakage-reliance scatter; supports RQ1/RQ3 interpretation only"),
    "dCIGL": dict(
        L1=M("feature_leakage", "sig (inherits CIGL audit)"),
        L2=M("KL_vs_cigl", "graph_kl +0.251 sig (traded up)"),
        L3=MISS(NOT_APPLICABLE), L4=M("align_k2_vs_cigl", "+0.059 sig (HIGHER)"),
        L5=M("R3_task_drop_k2_vs_cigl", "-0.007 [-0.024,+0.006] ns; no flatten confound"),
        L6=M("target_bacc_vs_cigl", "+0.003 [-0.004,+0.010] ns"),
        task_safety=("mean_margin/entropy", "PASS (margin +0.252, entropy -0.044)"), collapse_guard="PASS", random_control="valid",
        rq=[], tag=SUPPORT, reason="frozen paired-delta vs CIGL; supports interpretation only"),
    "MetaCMI": dict(
        L1=M("feature_z_perm_p", "<0.05 all folds"),
        L2=M("featKL_vs_MetaCE", "-0.031..-0.044 (still sig)"),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_MEASURED),
        L5=M("R3_k2_vs_MetaCE", "-0.001..-0.004 within noise (source=CIGL_69C readout)"),
        L6=M("target_bacc_vs_MetaCE", "<=0 all 4 cells"),
        task_safety=("8_stop_conditions", "PASS"), collapse_guard="PASS", random_control="valid",
        rq=[], tag=SUPPORT, reason="seed0-screening only; R3 not in phase2 JSON (readout only); supports interpretation"),
    "CITA_lambda_1": dict(
        L1=M("cond_domain_active_fraction", "1.8-29.3% of loss"),
        L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_MEASURED),
        L5=M("R3_CITA_minus_TTA", "~0 all cells"),
        L6=M("target_bacc_CITA_minus_TTA", "-0.003..+0.000 (~0)"),
        task_safety=("entropy/label-balance", "PASS no collapse"), collapse_guard="PASS", random_control="valid",
        rq=[], tag=SUPPORT, reason="target-unlabeled; endpoint is CITA-minus-TTA (control contrast), not a leakage-reliance scatter"),
    "CITA_lambda_0.010": dict(
        L1=M("cond_domain_active_fraction", "0.1-0.2% (near-inert)"),
        L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_MEASURED),
        L5=M("R3_CITA_minus_TTA", "~0 all cells"),
        L6=M("target_bacc_CITA_minus_TTA", "-0.001..+0.002 (~0)"),
        task_safety=("entropy/label-balance", "PASS"), collapse_guard="PASS", random_control="valid",
        rq=[], tag=SUPPORT, reason="inert-lambda baseline; control contrast only"),
    "TTA_Control_non_CMI": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_MEASURED),
        L5=M("R3_TTA_minus_ERM", "mixed/slightly + (reliance NOT reduced)"),
        L6=M("target_bacc_TTA_minus_ERM", "+0.037..+0.093 all 4 (non-CMI positive)"),
        task_safety=("entropy/label-balance", "PASS no collapse"), collapse_guard="PASS", random_control="valid",
        rq=[], tag=SUPPORT, reason="non-CMI positive; informs RQ2 (target benefit decoupled from reliance) as context, not an erasure route"),
    # ---------------- TOS erasure ----------------
    "TOS_mean_scatter": dict(
        L1=M("subject_decodable", "yes (TSMNet 0.997 / EEGNet 0.819)"),
        L2=MISS(NOT_APPLICABLE),
        L3=M("subject_removed_lin/mlp", "TSMNet ->0.917/0.958; EEGNet ->0.358/0.545"),
        L4=M("label_light_by_construction", "task preserved 0.746/0.637"),
        L5=MISS(NOT_MEASURED),
        L6=M("target_bacc_delta", "TSMNet -0.0003 / EEGNet -0.0005 (~0)"),
        task_safety=("task_bacc_delta", "PASS (score-Fisher guard)"), collapse_guard="PASS", random_control="valid",
        rq=["RQ2"], tag="INCLUDED", reason="L3 erasability + L6 target present"),
    "TOS_LEACE": dict(
        L1=M("subject_decodable", "yes"), L2=MISS(NOT_APPLICABLE),
        L3=M("subject_removed_lin/mlp", "lin->0.115 chance; MLP residual 0.740/0.393"),
        L4=M("task_preserved_4class", "0.743/0.606 (binary EEGNet -> chance)"),
        L5=MISS(NOT_MEASURED),
        L6=M("target_bacc_delta", "2a ~0; binary Lee/Cho EEGNet -0.15..-0.19"),
        task_safety=("task_bacc_delta", "FAIL on binary EEGNet"), collapse_guard="conditional", random_control="valid (random_k)",
        rq=["RQ2"], tag="INCLUDED", reason="L3 erasability + L6 target present"),
    "TOS_INLP": dict(
        L1=M("subject_decodable", "yes"), L2=MISS(NOT_APPLICABLE),
        L3=M("subject_removed", "->chance both"),
        L4=M("task_destroyed", "EEGNet 0.639->0.246 (over-erasure)"),
        L5=MISS(NOT_MEASURED),
        L6=M("target_bacc_delta", "-0.062 / -0.160 (task collapse)"),
        task_safety=("task_bacc_delta", "FAIL over-erasure"), collapse_guard="FAIL", random_control="valid",
        rq=["RQ2"], tag="INCLUDED", reason="L3 erasability + L6 target present (negative control for RQ2)"),
    "TOS_RLACE": dict(
        L1=M("subject_decodable", "yes"), L2=MISS(NOT_APPLICABLE),
        L3=M("subject_removed_lin", "TSMNet 0.856 weak; EEGNet 0.293"),
        L4=M("task_preserved_4class", "0.739/0.611 (binary -> chance)"),
        L5=MISS(NOT_MEASURED),
        L6=M("target_bacc_delta", "2a -0.004..-0.012; binary -0.15..-0.19"),
        task_safety=("task_bacc_delta", "FAIL on binary EEGNet"), collapse_guard="conditional", random_control="valid",
        rq=["RQ2"], tag="INCLUDED", reason="L3 erasability + L6 target present"),
    "TOS_random_k": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE),
        L3=M("subject_removed", "NOT removed (0.734/0.808; TSMNet 0.998)"),
        L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_MEASURED),
        L6=M("target_nll_delta", "-0.034 matches LEACE -0.031 (non-specific)"),
        task_safety=("task_bacc_delta", "N/A control"), collapse_guard="N/A", random_control="is the control",
        rq=["RQ2"], tag="INCLUDED", reason="falsifier control for RQ2 (L3 non-removal + L6 NLL)"),
    "TOS_refusal_gate": dict(
        L1=M("domain_gate/accept-refuse", "accept 5/9 vacuous; DOMAIN_GATE_CLOSED 9/9 high-lambda"),
        L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE),
        L5=MISS(NOT_APPLICABLE),
        L6=M("refusal_accept_decision", "0 unsafe-accepts (floor) vs 6; default-on NOT certified"),
        task_safety=("refuse_when_unsafe", "PASS by design"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=BOUNDARY, reason="decision/certification evidence; L6 is a refusal decision not a target-metric endpoint"),
    "TOS_global_LPC_collapse": dict(
        L1=M("subject_decodable", "yes"), L2=M("penalty_effect", "TSMNet feat_norm 1.09->0.00 CE->ln4"),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_MEASURED),
        L6=M("target_bacc_delta_EEGNet", "0.43->0.39 (p<=0.001 harm)"),
        task_safety=("representation_collapse", "FAIL (TSMNet)"), collapse_guard="FAIL", random_control="N/A",
        rq=[], tag=BOUNDARY, reason="in-loss global penalty boundary result (collapse/harm), not a controlled erasure route for RQ2"),
    "TOS_task_preserving_erasure": dict(
        L1=M("subject_decodable", "yes"), L2=M("collapse_free_penalty", "warm_ramp collapse 0/9"),
        L3=M("subject_removed", "collapse-free subj_dec ~0.997 = ERM (removes nothing)"),
        L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_MEASURED), L6=MISS(NOT_MEASURED),
        task_safety=("task_preserved", "PASS"), collapse_guard="PASS", random_control="N/A",
        rq=[], tag=BOUNDARY, reason="in-loss boundary result (removes nothing when collapse-free); no target endpoint"),
    "TOS_capacity_factorial": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE),
        L3=M("nonlinear_residual_vs_dz", "OLS log(d_z)=+0.089; matched-dim -68% gap"),
        L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE), L6=MISS(NOT_APPLICABLE),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=SUPPORT, reason="mechanism (removability vs capacity); no endpoint level"),
    # ---------------- FBCSP branch ----------------
    "FBCSP_LGG_branch_ablation": dict(
        L1=M("fused_leakage_kl", "fused-Z only (e.g. 0.645); NO per-branch probe"),
        L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE),
        L4=M("zero_spatial_ablation & gate", "2a 0.349->0.275 (-7.4pp); 2015 -8.8pp; gate_spatial 0.489/0.572"),
        L5=MISS(NEEDS_PROBE), L6=MISS(NOT_APPLICABLE),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="permute_nodes null",
        rq=[], tag=SUPPORT, reason="L4 branch-load only; per-branch leakage(L1) + per-branch reliance(L5) missing -> cannot enter RQ4"),
    "FBCSP_LGG_gate_summary": dict(
        L1=MISS(NOT_APPLICABLE), L2=M("spatial_CMI_delta", "+0.0082 (sd 0.0285) fragile"),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_MEASURED),
        L6=M("2a_decodable_delta", "+0.0082 (2/3 seeds neg; seed2 collapsed-ERM artifact)"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=BOUNDARY, reason="fragile-survive/not-promoted spatial-CMI outcome; effectively null"),
    "FBCSP_LGG_bottleneck_analysis": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE),
        L4=M("CSP_vs_DGCNN_vs_FBLGG", "subj1 0.483/0.403/0.306 (feature bottleneck)"),
        L5=MISS(NOT_APPLICABLE), L6=MISS(NOT_APPLICABLE),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=SUPPORT, reason="branch-locality background (accuracy diagnostic); no leakage/reliance endpoint"),
    "FBCSP_LGG_graph_starvation": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE),
        L4=M("branch_ablation_deltas", "2a graph ~+0.02 (no signal); 2015 +0.085/+0.114"),
        L5=MISS(NOT_APPLICABLE), L6=MISS(NOT_APPLICABLE),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=SUPPORT, reason="branch-locality background; no endpoint"),
    "CIGL_35_blueprint": dict(
        L1=M("graph/node_KL_perm", "~8x/15x"), L2=M("leakage_reduction_pct", "-35..-77%"),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE), L6=MISS(NOT_APPLICABLE),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=BACKGROUND, reason="claim-boundary contract for the DGCNN CIGL line (background)"),
    "P6_spatial_CMI_scaffold": dict(
        L1=MISS(NOT_MEASURED), L2=MISS(NOT_MEASURED), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE),
        L5=MISS(NOT_MEASURED), L6=MISS(NOT_MEASURED),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=BACKGROUND, reason="scaffold only, never run (red-flag 11); nothing measured"),
    # ---------------- OACI / ACAR / CSC / LPC / H2CMI ----------------
    "OACI_selection_leakage_not_target": dict(
        L1=M("selection_leakage_delta", "-0.3261 54/54"), L2=M("audit_leakage_delta", "+0.0076 (25/54)"),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE),
        L6=M("target_worst_bacc_delta", "-0.0024 (harmed 28/54); corr(audit,tgt)~0"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        rq=[], tag=SUPPORT, reason="observability failure (selection vs audit vs target); corroborates but not a CIGL/TOS RQ regression"),
    "OACI_source_audit_oracle_failure": dict(
        L1=M("selector_replay", "216/216 exact"), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE),
        L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE),
        L6=M("oracle_reproducible", "False (S5 oracle -> ERM 3/54)"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="valid",
        rq=[], tag=SUPPORT, reason="oracle-robust negative; corroboration"),
    "OACI_multivariate_weak_identifiability": dict(
        L1=M("LOTO_probe_AUC", "0.602 vs perm 0.537 (p=0.008)"), L2=MISS(NOT_APPLICABLE),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE),
        L6=M("deployable", "no (diagnostic-only; scalar rho +0.12)"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="permutation"),
    "OACI_endpoint_estimability_limit": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE),
        L5=MISS(NOT_APPLICABLE), L6=M("worst_domain_estimability", "bAcc->NaN under cell deletion (C18)"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A"),
    "ACAR_paired_action_risk_design": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE),
        L5=MISS(NOT_APPLICABLE), L6=MISS(NOT_MEASURED),
        task_safety=("8_leakage_guards", "PASS (design)"), collapse_guard="N/A", random_control="N/A",
        tag=PROTOCOL, reason="design/estimand only; no efficacy"),
    "ACAR_v5_protocol_substrate_success": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE),
        L5=MISS(NOT_APPLICABLE), L6=MISS(NOT_MEASURED),
        task_safety=("firewall", "PASS 30/30 refs"), collapse_guard="N/A", random_control="N/A",
        tag=PROTOCOL, reason="engineering/protocol success, not efficacy"),
    "ACAR_stage2b_dev_stop": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE),
        L4=M("coverage_G1", "PD 13/22, SCZ 19/22 (adapt ok)"), L5=MISS(NOT_APPLICABLE),
        L6=M("eligible / EVAL_red", "0/22 eligible; DEV_STOP; router refuted; harm UCB 0.61-0.87"),
        task_safety=("harm_control_G3/G4", "FAIL 0/22"), collapse_guard="N/A", random_control="v2_replay",
        tag="INCLUDED_BOUNDARY", reason="completed DEV_STOP (deployment corroboration; not pending)"),
    "CSC_Z_only_unidentifiable": dict(
        L1=M("residual_decoder_T", "dev power 0.83 -> confirmatory 28/65"), L2=MISS(NOT_APPLICABLE),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE),
        L6=M("false_cert / power", "forbidden 1/65 CP-UB 0.0709>0.05; power 28/65<0.50 (FAILED both)"),
        task_safety=("abstention", "valid (Prop1)"), collapse_guard="N/A", random_control="permutation",
        tag=BOUNDARY, reason="information-contract boundary; simulator+semisynth, unidentifiability proven"),
    "CSC_dual_witness_candidate": dict(
        L1=M("B6/B7_witness", "B6 strong-cov 18/28->0/0; B7.0 dev-only"), L2=MISS(NOT_APPLICABLE),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE),
        L6=MISS(NOT_MEASURED),
        task_safety=("abstention", "dev-only"), collapse_guard="N/A", random_control="condition-randomization",
        tag=PROTOCOL, reason="B7.1 confirmatory committed-but-UNRUN (protocol-only)"),
    "CSC_information_contract_boundary": dict(
        L1=M("fitted_null_dispersion", "under-dispersed ~7-10x under strong cov"), L2=MISS(NOT_APPLICABLE),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_APPLICABLE),
        L6=M("router_breach", "weak-cov masks; strong-cov SOUND FAIL 7.3%/26%"),
        task_safety=("abstention", "valid"), collapse_guard="N/A", random_control="oracle-null",
        tag=BOUNDARY, reason="the publishable partial-identification boundary; corroboration"),
    "LPC_CMI_legacy_boundary": dict(
        L1=M("extractable_domain_info", "reduced (measurement only; beats CDANN)"),
        L2=M("deployment_regularizer", "DROP_LPC_COLLAPSE all lambda"),
        L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE), L5=MISS(NOT_MEASURED),
        L6=M("deployment/calibration", "calibration=temperature (oracle-T 123/130); accuracy=matched-CORAL"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        tag=BOUNDARY, reason="closed legacy; measurement survives, control/calibration/accuracy dropped; do NOT restart"),
    "PriorDecoupled_four_branch_protocol": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE),
        L5=MISS(NOT_APPLICABLE), L6=M("G/P decomposition", "W2 P=-0.1439 sig; G=-0.0201 NS"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        tag=BACKGROUND, reason="prior-decoupled TTA; NOT FSR leakage/reliance evidence"),
    "PriorDecoupled_geometry_vs_prevalence": dict(
        L1=MISS(NOT_APPLICABLE), L2=MISS(NOT_APPLICABLE), L3=MISS(NOT_APPLICABLE), L4=MISS(NOT_APPLICABLE),
        L5=MISS(NOT_APPLICABLE), L6=M("prevalence decomposition", "metric-mismatch -0.16 dominates"),
        task_safety=("N/A", "N/A"), collapse_guard="N/A", random_control="N/A",
        tag=BACKGROUND, reason="prior-decoupled TTA; background"),
    # premises
    "CIGL_70_closure": dict(tag=BACKGROUND, reason="frozen premise (closure), not an analysis route"),
    "CMI_SYNTHESIS": dict(tag=BACKGROUND, reason="frozen premise (synthesis), not an analysis route"),
}

# Which RQ each cluster/route is relevant to when includable.
RQ_RELEVANCE = {"CIGL": ["RQ1", "RQ3"]}


def _levels_present(spec):
    pred = [lv for lv in ("L1", "L2", "L3", "L4") if spec.get(lv, MISS(NOT_APPLICABLE))["metric"]]
    endp = [lv for lv in ("L5", "L6") if spec.get(lv, MISS(NOT_APPLICABLE))["metric"]]
    return pred, endp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(INDEX, newline="") as fh:
        index = list(csv.DictReader(fh))

    metric_rows, incl_rows, miss_rows = [], [], []
    for i, r in enumerate(index, 1):
        route = r["route"]
        spec = LADDER.get(route, dict(tag=SUPPORT, reason="no ladder mapping"))
        pred, endp = _levels_present(spec)
        rq = spec.get("rq", [])
        tag = spec.get("tag", SUPPORT)
        reason = spec.get("reason", "")

        # ---- route_metric_table row ----
        def lv(name):
            m = spec.get(name, MISS(NOT_APPLICABLE))
            return m["metric"], m["value"]
        ts = spec.get("task_safety", ("N/A", "N/A"))
        row = dict(
            row_id=i, cluster=r["component"], route=route, source_branch=r["source_branch"],
            source_sha=r["source_sha"], dataset=r["dataset"], backbone=r["backbone"], method=r["method"],
            info_regime=r["info_regime"], unit_id=f"{route}::route_level", seed=r["seeds"],
            fold_or_subject=r["folds_or_subjects"], representation_or_branch=r["representation_or_branch"],
            L1_metric=lv("L1")[0], L1_value=lv("L1")[1], L2_metric=lv("L2")[0], L2_value=lv("L2")[1],
            L3_metric=lv("L3")[0], L3_value=lv("L3")[1], L4_metric=lv("L4")[0], L4_value=lv("L4")[1],
            L5_metric=lv("L5")[0], L5_value=lv("L5")[1], L6_metric=lv("L6")[0], L6_value=lv("L6")[1],
            task_safety_metric=ts[0], task_safety_value=ts[1],
            collapse_guard=spec.get("collapse_guard", "N/A"), random_control=spec.get("random_control", "N/A"),
            target_labels_used_for_fit=r["target_labels_used_for_fit"],
            target_labels_used_for_eval=r["target_labels_used_for_eval"], artifact_path=r["artifact_path"],
            include_rq1=_yn("RQ1", rq), include_rq2=_yn("RQ2", rq), include_rq3=_yn("RQ3", rq),
            include_rq4="NO",  # no route has per-branch leakage+reliance -> RQ4 has no quantitative rows
            support_only_reason=("" if rq else reason), notes=r["notes"][:160])
        metric_rows.append(row)

        # ---- analysis_inclusion_table row ----
        incl_rows.append(dict(
            route=route, cluster=r["component"], predictor_levels="|".join(pred) or "none",
            endpoint_levels="|".join(endp) or "none", inclusion_tag=(tag if not rq else "INCLUDED"),
            include_rq1=_yn("RQ1", rq), include_rq2=_yn("RQ2", rq), include_rq3=_yn("RQ3", rq),
            include_rq4="NO", gate_predictor_ok=("YES" if pred else "NO"),
            gate_endpoint_ok=("YES" if endp else "NO"), reason=reason))

        # ---- missing_metric_decisions rows ----
        for name in L:
            m = spec.get(name, MISS(NOT_APPLICABLE))
            if not m["metric"] and m["value"] in (NOT_MEASURED, ARTIFACT_MISSING, NEEDS_CPU, NEEDS_PROBE):
                miss_rows.append(dict(
                    route=route, cluster=r["component"], missing_level=name,
                    missing_metric=_expected_metric(name), policy=m["value"],
                    needed_for_rq=_needed_rq(name), resolution=_resolution(m["value"]),
                    notes=reason))

    _wcsv(OUT / "route_metric_table.csv", metric_rows)
    _wcsv(OUT / "analysis_inclusion_table.csv", incl_rows)
    _wcsv(OUT / "missing_metric_decisions.csv", miss_rows)

    inc1 = [r["route"] for r in metric_rows if r["include_rq1"] == "YES"]
    inc2 = [r["route"] for r in metric_rows if r["include_rq2"] == "YES"]
    inc3 = [r["route"] for r in metric_rows if r["include_rq3"] == "YES"]
    print(f"route_metric_table: {len(metric_rows)} rows")
    print(f"RQ1 includable: {inc1}")
    print(f"RQ2 includable: {inc2}")
    print(f"RQ3 includable: {inc3}")
    print(f"RQ4 includable: [] (no route has per-branch leakage+reliance; see missing_metric_decisions)")
    print(f"missing_metric_decisions: {len(miss_rows)} rows")


def _yn(rq, rqs):
    return "YES" if rq in rqs else "NO"


def _expected_metric(level):
    return {"L1": "leakage/probe", "L2": "delta_leakage", "L3": "erasability",
            "L4": "task_coupling/align", "L5": "R3_functional_reliance",
            "L6": "target_consequence"}[level]


def _needed_rq(level):
    return {"L1": "RQ1/RQ3/RQ4", "L2": "RQ1", "L3": "RQ2", "L4": "RQ3/RQ4",
            "L5": "RQ1/RQ3/RQ4", "L6": "RQ2"}[level]


def _resolution(policy):
    return {NOT_MEASURED: "defer", ARTIFACT_MISSING: "defer (raw pruned/unavailable)",
            NEEDS_CPU: "CPU_recompute", NEEDS_PROBE: "small_frozen_probe (Phase 3/4, gated)"}[policy]


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

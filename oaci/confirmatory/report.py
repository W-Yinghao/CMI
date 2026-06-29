"""Endpoint extraction + k1/k2-style summary for the confirmatory one-fold run.

This reports per-seed / per-level / per-method endpoints and a cross-seed k1/k2-style view. It is a
PIPELINE-VALIDATION report: a single fold and a tiny seed set are NOT confirmatory efficacy evidence, and
the k1/k2 values here are descriptive (the protocol's permutation/decision machinery is not run).
"""
from __future__ import annotations

NOTICE = ("full-budget one-fold pipeline validation; NOT confirmatory efficacy evidence "
          "(single fold, descriptive k1/k2 only, no permutation test / decision rule)")
NOTICE_REDUCED = ("full-budget TRAINING one-fold pipeline validation with REDUCED-bootstrap uncertainty; "
                  "NOT final confirmatory statistical evidence (single fold, reduced leakage/eval bootstrap, "
                  "descriptive k1/k2 only, no permutation test / decision rule)")


def _sel_lambda(m):
    for c in m.train_result.trajectory:
        if c.model_hash == m.selection.model_hash:
            return float(c.lam)
    return None


def _ucl(leak):
    return None if leak is None else float(leak["bootstrap_ucl"])


def _method_block(m, tau):
    tm, sm = m.target_metrics, m.source_audit_metrics
    return {"method": m.method_name, "active": bool(m.active), "inactive_reason": m.inactive_reason,
            "selected_checkpoint": m.selection.model_hash, "selected_risk": float(m.selection.R_src),
            "selected_risk_minus_tau": float(m.selection.R_src) - float(tau),
            "selected_epoch": int(m.selection.selected_epoch), "selected_lambda": _sel_lambda(m),
            "selection_leakage_ucl": _ucl(m.selection_leakage), "audit_leakage_ucl": _ucl(m.audit_leakage),
            "source_audit_bacc": sm.pooled_reference_bacc, "source_audit_nll": sm.pooled_nll,
            "source_audit_ece": sm.pooled_ece,
            "target_pooled_bacc": tm.pooled_reference_bacc, "target_worst_bacc": tm.worst_domain_reference_bacc,
            "target_pooled_nll": tm.pooled_nll, "target_worst_nll": tm.worst_domain_nll,
            "target_pooled_ece": tm.pooled_ece, "target_worst_ece": tm.worst_domain_ece}


def _level_block(lvl, lr):
    es = lr.erm_stage
    return {"level": int(lvl), "run_key_hash": lr.run_key.run_key_hash,
            "support_hash": lr.support_state.support_hash,
            "eligibility_counts": lr.support_state.support_graph.counts.tolist(),
            "cell_mass": [[float(x) for x in row] for row in lr.support_state.support_graph.cell_mass.tolist()],
            "reference_prior": [float(x) for x in lr.support_state.support_graph.reference_prior.tolist()],
            "erm_checkpoint": es.checkpoint.model_hash, "R_ERM_hat": float(es.R_ERM_hat), "tau": float(es.tau),
            "target_fit_ids": sorted(lr.provenance.target_fit_ids),
            "methods": [_method_block(m, es.tau) for _, m in lr.method_items]}


def _seed_block(run):
    art = run.artifact
    fr = art.fold_result
    return {"model_seed": int(run.model_seed),
            "artifact_scientific_hash": art.write_result.artifact_scientific_hash,
            "artifact_pure_science_hash": art.write_result.artifact_pure_science_hash,
            "fold_result_hash": fr.fold_result_hash, "deep_verification_ok": bool(art.verification.ok),
            "comparison_hash": art.comparison_hash,
            "all_target_fit_ids_empty": all(not lr.provenance.target_fit_ids for _, lr in fr.level_items),
            "levels": [_level_block(lvl, lr) for lvl, lr in fr.level_items]}


def _endpoint(seeds_blocks, level, method, key):
    """All per-seed values of one (level, method, endpoint), in seed order."""
    out = []
    for sb in seeds_blocks:
        lvl = next(l for l in sb["levels"] if l["level"] == level)
        mb = next(m for m in lvl["methods"] if m["method"] == method)
        out.append(mb[key])
    return out


def _k1_k2(seeds_blocks):
    """Descriptive k1 (OACI - ERM audit-leakage UCL) and k2 (worst-domain bAcc / NLL) endpoints,
    per level, summarized across seeds. Not a permutation test or a decision."""
    levels = sorted({l["level"] for sb in seeds_blocks for l in sb["levels"]})
    k1, k2 = [], []
    for lvl in levels:
        oaci = _endpoint(seeds_blocks, lvl, "OACI", "audit_leakage_ucl")
        erm = _endpoint(seeds_blocks, lvl, "ERM", "audit_leakage_ucl")
        gaps = [(o - e) for o, e in zip(oaci, erm) if o is not None and e is not None]
        k1.append({"level": lvl, "statistic": "OACI_minus_ERM_audit_leakage_ucl",
                   "per_seed": gaps, "max": (max(gaps) if gaps else None)})
        for method in ("ERM", "OACI", "global_lpc", "uniform"):
            k2.append({"level": lvl, "method": method,
                       "worst_domain_bacc_per_seed": _endpoint(seeds_blocks, lvl, method, "target_worst_bacc"),
                       "worst_domain_nll_per_seed": _endpoint(seeds_blocks, lvl, method, "target_worst_nll")})
    return k1, k2


def build_onefold_report(result) -> dict:
    fold = result.fold
    sm = fold.manifest.pilot
    seeds_blocks = [_seed_block(r) for r in result.seed_runs]
    k1, k2 = _k1_k2(seeds_blocks)
    mode = getattr(result, "bootstrap_mode", "full_budget")
    bootstrap = {"bootstrap_mode": mode,
                 "selection_bootstrap": int(fold.manifest.probe.selection_bootstrap),
                 "audit_bootstrap": int(fold.manifest.probe.audit_bootstrap),
                 "paired_bootstrap": int(fold.manifest.evaluation.paired_bootstrap),
                 "not_confirmatory_ci": mode != "full_budget"}
    return {
        "notice": (NOTICE_REDUCED if mode != "full_budget" else NOTICE),
        "protocol_path": result.protocol_path, "dataset": result.dataset,
        "manifest_path": result.manifest_path, "manifest_hash": result.manifest_hash,
        "bootstrap": bootstrap,
        "target_subject": result.target_subject, "model_seeds": list(result.model_seeds),
        "subjects": list(sm.subjects), "target_subjects": list(sm.target_subjects),
        "source_audit_subjects": list(sm.source_audit_subjects),
        "source_train_subjects": list(sm.source_train_subjects),
        "deleted_cell_level1": {"domain_id": sm.deleted_cell_level1.domain_id,
                                "class_name": sm.deleted_cell_level1.class_name},
        "training_budget": {"stage1_epochs": fold.manifest.training.stage1_epochs,
                            "stage2_epochs": fold.manifest.training.stage2_epochs,
                            "stage2_steps_per_epoch": fold.manifest.training.stage2_steps_per_epoch},
        "data_evidence_hash": fold.data_evidence_hash, "resolved_preprocess_hash": fold.resolved_preprocess_hash,
        "split_manifest_hash": fold.split_manifest_hash, "seeds": seeds_blocks,
        "k1_descriptive": k1, "k2_descriptive": k2,
        "all_seeds_deep_verified": all(sb["deep_verification_ok"] for sb in seeds_blocks),
        "all_target_fit_ids_empty": all(sb["all_target_fit_ids_empty"] for sb in seeds_blocks)}

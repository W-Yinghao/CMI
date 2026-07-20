"""Recursive bit-exact scientific comparison of two BNCI runs (canonical vs reversed method order).

Compares the SCIENTIFIC logical identities only (fold/checkpoint/trajectory/selection/audit/prediction/
metrics/plan hashes); the transport file bytes (.pt SHA, artifact_index SHA, directory path, runtime
report) are deliberately NOT part of the comparison. The first differing canonical-sorted path is
reported in full, e.g. ``levels/1/methods/OACI/trajectory/0/model_hash``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..artifacts.canonical_json import canonical_json_hash
from .scientific_hash import leakage_result_hash

_ROLES = ("source_guard", "source_audit", "target_audit")


@dataclass(frozen=True)
class ScientificMismatch:
    path: str
    canonical_value_hash: str
    reversed_value_hash: str


@dataclass(frozen=True)
class BNCIOrderComparison:
    fold_scope_equal: bool
    fold_result_equal: bool
    artifact_scientific_hash_equal: bool
    checkpoint_hashes_equal: bool
    trajectory_hashes_equal: bool
    selection_hashes_equal: bool
    audit_hashes_equal: bool
    prediction_hashes_equal: bool
    metrics_hashes_equal: bool
    plan_hashes_equal: bool
    first_mismatch: ScientificMismatch | None
    comparison_hash: str


def _leak(v):
    return "-" if v is None else leakage_result_hash(v)


def _opt_plan(p, attr="plan_hash"):
    return "absent" if p is None else getattr(p, attr)


def flatten_scientific(artifact_result) -> dict:
    """A flat {canonical-path: value-hash} map of one run's scientific identity."""
    fr = artifact_result.fold_result
    flat = {"fold_result_hash": fr.fold_result_hash, "fold_scope_hash": fr.fold_scope.fold_scope_hash,
            "artifact_scientific_hash": artifact_result.write_result.artifact_scientific_hash}
    for lvl, lr in fr.level_items:
        b = f"levels/{int(lvl)}"
        flat[f"{b}/run_key"] = lr.run_key.run_key_hash
        flat[f"{b}/support_hash"] = lr.support_state.support_hash
        flat[f"{b}/level_support_hash"] = lr.support_state.level_support_hash
        flat[f"{b}/level_plans_hash"] = lr.plans.level_plans_hash
        flat[f"{b}/level_result_hash"] = lr.level_result_hash
        flat[f"{b}/provenance"] = lr.provenance.provenance_hash
        es = lr.erm_stage
        flat[f"{b}/erm/checkpoint"] = es.checkpoint.model_hash
        flat[f"{b}/erm/invocation_id"] = es.stage1_invocation_id
        flat[f"{b}/erm/tau"] = repr(float(es.tau))
        p = lr.plans
        for name, plan, attr in (("stage1_task", p.stage1_task, "plan_hash"), ("stage2_task", p.stage2_task, "plan_hash"),
                                 ("oaci_alignment", p.oaci_alignment, "plan_hash"),
                                 ("full_domain_alignment", p.full_domain_alignment, "plan_hash"),
                                 ("selection_design", p.selection_design, "population_hash"),
                                 ("selection_fold_plan", p.selection_fold_plan, "plan_hash"),
                                 ("selection_bootstrap_plan", p.selection_bootstrap_plan, "plan_hash")):
            flat[f"{b}/plans/{name}"] = _opt_plan(plan, attr)
        for name, m in lr.method_items:
            mb = f"{b}/methods/{name}"
            flat[f"{mb}/active"] = str(bool(m.active))
            for i, c in enumerate(m.train_result.trajectory):
                flat[f"{mb}/trajectory/{i}/model_hash"] = c.model_hash
                flat[f"{mb}/trajectory/{i}/R_src"] = repr(float(c.R_src))
                flat[f"{mb}/trajectory/{i}/lambda"] = repr(float(c.lam))
            flat[f"{mb}/selection/model_hash"] = m.selection.model_hash
            flat[f"{mb}/selection/leakage"] = _leak(m.selection_leakage)
            flat[f"{mb}/audit/leakage"] = _leak(m.audit_leakage)
            for role, bundle, met in (("source_guard", m.source_guard_predictions, m.source_guard_metrics),
                                      ("source_audit", m.source_audit_predictions, m.source_audit_metrics),
                                      ("target_audit", m.target_predictions, m.target_metrics)):
                flat[f"{mb}/predictions/{role}"] = bundle.prediction_content_hash()
                flat[f"{mb}/metrics/{role}"] = met.metrics_hash
            flat[f"{mb}/signatures/source_audit"] = m.source_audit_predictions.audit_signature_hash
            flat[f"{mb}/signatures/target"] = m.target_predictions.audit_signature_hash
    flat["fold/target_fit_ids"] = "|".join(sorted({i for _, lr in fr.level_items for i in lr.provenance.target_fit_ids}))
    return flat


def _group_equal(ca, re, pred) -> bool:
    keys = [k for k in ca if pred(k)]
    return all(ca.get(k) == re.get(k) for k in keys) and {k for k in ca if pred(k)} == {k for k in re if pred(k)}


def compare_scientific_results(canonical, reversed_) -> BNCIOrderComparison:
    ca, re = flatten_scientific(canonical), flatten_scientific(reversed_)
    first = None
    for k in sorted(set(ca) | set(re)):
        if ca.get(k) != re.get(k):
            first = ScientificMismatch(path=k, canonical_value_hash=str(ca.get(k)), reversed_value_hash=str(re.get(k)))
            break
    cmp = BNCIOrderComparison(
        fold_scope_equal=ca["fold_scope_hash"] == re["fold_scope_hash"],
        fold_result_equal=ca["fold_result_hash"] == re["fold_result_hash"],
        artifact_scientific_hash_equal=ca["artifact_scientific_hash"] == re["artifact_scientific_hash"],
        checkpoint_hashes_equal=_group_equal(ca, re, lambda k: k.endswith("/erm/checkpoint") or k.endswith("/selection/model_hash")),
        trajectory_hashes_equal=_group_equal(ca, re, lambda k: "/trajectory/" in k),
        selection_hashes_equal=_group_equal(ca, re, lambda k: "/selection/" in k),
        audit_hashes_equal=_group_equal(ca, re, lambda k: "/audit/" in k),
        prediction_hashes_equal=_group_equal(ca, re, lambda k: "/predictions/" in k),
        metrics_hashes_equal=_group_equal(ca, re, lambda k: "/metrics/" in k),
        plan_hashes_equal=_group_equal(ca, re, lambda k: "/plans/" in k),
        first_mismatch=first,
        comparison_hash=canonical_json_hash({"canonical": ca, "reversed": re}))
    return cmp


def comparison_all_equal(cmp: BNCIOrderComparison) -> bool:
    return (cmp.first_mismatch is None and cmp.fold_scope_equal and cmp.fold_result_equal
            and cmp.artifact_scientific_hash_equal and cmp.checkpoint_hashes_equal and cmp.trajectory_hashes_equal
            and cmp.selection_hashes_equal and cmp.audit_hashes_equal and cmp.prediction_hashes_equal
            and cmp.metrics_hashes_equal and cmp.plan_hashes_equal)

"""Read-only logical summary of a committed artifact (does NOT rebuild or unpickle a FoldRunResult).

``read_completed_artifact`` deep-verifies the tree, then returns a frozen ArtifactSummary of just the
logical identities (hashes + support tables + plan / checkpoint / prediction / metrics hashes +
signatures), suitable for an exact in-memory-vs-on-disk comparison.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import plan_codec as P
from . import prediction_codec as PR
from . import support_codec as SC
from .atomic import COMMIT_MARKER
from .reader import read_artifact, read_doc
from .verify import verify_artifact_tree

_ROLES = ("source_guard", "source_audit", "target_audit")
_PLAN_FILES = ("stage1_task", "stage2_task", "oaci_alignment", "full_domain_alignment",
               "selection_design", "selection_fold_plan", "selection_bootstrap_plan")


@dataclass(frozen=True)
class MethodSummary:
    name: str
    active: bool
    selected_checkpoint: str
    R_src: float
    selected_epoch: int
    selection_status: str
    audit_status: str
    selection_leakage_hash: str | None
    audit_leakage_hash: str | None
    prediction_content_hashes: tuple        # ((role, hash), ...)
    metrics_hashes: tuple
    source_audit_signature: str
    target_signature: str


@dataclass(frozen=True)
class LevelSummary:
    level: int
    run_key_hash: str
    support_hash: str
    level_support_hash: str
    level_plans_hash: str
    level_result_hash: str
    eligibility_counts: tuple
    cell_mass: tuple
    reference_prior: tuple
    plan_hashes: tuple                       # ((name, hash|"absent"), ...)
    erm_checkpoint: str
    R_ERM_hat: float
    tau: float
    checkpoint_hashes: tuple
    target_fit_ids: tuple
    method_items: tuple                      # ((name, MethodSummary), ...)


@dataclass(frozen=True)
class ArtifactSummary:
    schema_version: str
    artifact_scientific_hash: str               # provenance-bound (binds the git commit)
    artifact_pure_science_hash: str             # commit-independent science identity
    fold_result_hash: str
    context_hash: str
    pure_context_hash: str
    provenance_hash: str                         # the git evidence hash (commit/tree/clean/status)
    manifest_hash: str
    fold_scope_hash: str
    level_items: tuple                       # ((level, LevelSummary), ...)
    artifact_index_sha256: str
    n_indexed_files: int
    n_total_files: int
    n_verified_checkpoints: int
    n_verified_plans: int

    @property
    def levels(self) -> dict:
        return dict(self.level_items)


@dataclass(frozen=True)
class SummaryComparisonReport:
    ok: bool
    mismatches: tuple
    comparison_hash: str


def _level_dirs(root):
    base = os.path.join(root, "levels")
    return sorted(os.path.join("levels", d) for d in os.listdir(base) if d.startswith("level-"))


def _method_summary(root, md, name):
    _, mbody, _ = read_artifact(os.path.join(root, md, "method.json"), "method_result")
    sel = mbody["selection"]
    sa_logical, sabody, _ = read_artifact(os.path.join(root, md, "source_audit.json"), PR.PREDICTION_KIND)
    ta_logical, tabody, _ = read_artifact(os.path.join(root, md, "target_audit.json"), PR.PREDICTION_KIND)
    return MethodSummary(
        name=name, active=bool(mbody["active"]), selected_checkpoint=sel["model_hash"],
        R_src=float(sel["R_src"]), selected_epoch=int(sel["selected_epoch"]),
        selection_status=sel["selection_status"], audit_status=mbody["audit_status"],
        selection_leakage_hash=sel["selection_leakage_hash"], audit_leakage_hash=mbody["audit_leakage_hash"],
        prediction_content_hashes=tuple(zip(_ROLES, mbody["preds"])),
        metrics_hashes=tuple(zip(_ROLES, mbody["metrics"])),
        source_audit_signature=sabody["audit_signature_hash"], target_signature=tabody["audit_signature_hash"])


def _level_summary(root, ld):
    _, lbody, _ = read_artifact(os.path.join(root, ld, "level.json"), "level_result")
    pay = lbody["payload"]
    _, sbody, sarr = read_artifact(os.path.join(root, ld, "support.json"), SC.SUPPORT_KIND)
    _, pbody, _ = read_artifact(os.path.join(root, ld, "provenance.json"), "provenance")
    plan_hashes = []
    for fn in _PLAN_FILES:
        plog, pbd, _ = read_artifact(os.path.join(root, ld, "plans", fn + ".json"),
                                     _plan_kind(fn))
        plan_hashes.append((fn, "absent" if not pbd.get("present", True) else plog))
    ck_dir = os.path.join(root, ld, "checkpoints")
    cks = tuple(sorted(f[:-3] for f in os.listdir(ck_dir) if f.endswith(".pt")))
    methods = tuple((n, _method_summary(root, f"{ld}/methods/{n}", n))
                    for n in sorted(os.listdir(os.path.join(root, ld, "methods"))))
    return LevelSummary(
        level=int(ld.rsplit("-", 1)[-1]), run_key_hash=pay["run_key"], support_hash=pay["support_hash"],
        level_support_hash=pay["level_support_hash"], level_plans_hash=pay["level_plans_hash"],
        level_result_hash=read_artifact(os.path.join(root, ld, "level.json"), "level_result")[0],
        eligibility_counts=_rows(sarr["eligibility_counts"]), cell_mass=_rows(sarr["cell_mass"]),
        reference_prior=tuple(float(x) for x in sarr["reference_prior"].tolist()),
        plan_hashes=tuple(plan_hashes), erm_checkpoint=pay["erm"]["checkpoint"],
        R_ERM_hat=float(pay["erm"]["R_ERM_hat"]), tau=float(pay["erm"]["tau"]), checkpoint_hashes=cks,
        target_fit_ids=tuple(sorted(pbody.get("target_fit_ids", []))), method_items=methods)


def _plan_kind(fn):
    return {"stage1_task": P.TASK_KIND, "stage2_task": P.TASK_KIND, "selection_design": P.DESIGN_KIND,
            "oaci_alignment": P.ALIGN_KIND, "full_domain_alignment": P.ALIGN_KIND,
            "selection_fold_plan": P.FOLD_KIND, "selection_bootstrap_plan": P.BOOTSTRAP_KIND}[fn]


def _rows(a):
    return tuple(tuple(float(x) for x in row) for row in a.tolist())


def read_completed_artifact(path, *, deep_verify=True) -> ArtifactSummary:
    rep = verify_artifact_tree(path, deep=deep_verify)
    if not rep.ok:
        raise ValueError(f"artifact verification failed: {rep.errors[:3]}")
    root = os.path.abspath(path)
    marker = read_doc(os.path.join(root, COMMIT_MARKER))
    mh, _, _ = read_artifact(os.path.join(root, "context", "manifest.json"), "manifest")
    flog, fbody, _ = read_artifact(os.path.join(root, "fold.json"), "fold_result")
    levels = tuple((int(ld.rsplit("-", 1)[-1]), _level_summary(root, ld)) for ld in _level_dirs(root))
    return ArtifactSummary(
        schema_version=marker["schema_version"], artifact_scientific_hash=marker["artifact_scientific_hash"],
        artifact_pure_science_hash=marker["artifact_pure_science_hash"], fold_result_hash=flog,
        context_hash=marker["context_hash"], pure_context_hash=marker["pure_context_hash"],
        provenance_hash=marker["provenance_hash"], manifest_hash=mh,
        fold_scope_hash=fbody["fold_scope_hash"], level_items=levels,
        artifact_index_sha256=rep.artifact_index_sha256,
        n_indexed_files=rep.n_indexed_files, n_total_files=rep.n_total_files,
        n_verified_checkpoints=rep.n_verified_checkpoints, n_verified_plans=rep.n_verified_plans)


def compare_artifact_summary_to_memory(summary, fold_result, context, *, artifact_scientific_hash,
                                       artifact_pure_science_hash=None) -> SummaryComparisonReport:
    """Item-by-item equality between the on-disk summary and the in-memory FoldRunResult/context."""
    from ..runner.scientific_hash import leakage_result_hash
    fr = fold_result
    mm = []

    def chk(name, a, b):
        if a != b:
            mm.append(name)

    chk("schema_version", summary.schema_version, "oaci-artifact-v1")
    chk("manifest_hash", summary.manifest_hash, context.manifest_hash)
    chk("context_hash", summary.context_hash, context.context_hash)
    chk("pure_context_hash", summary.pure_context_hash, context.pure_context_hash)
    chk("provenance_hash", summary.provenance_hash, context.git.evidence_hash)
    chk("fold_scope_hash", summary.fold_scope_hash, fr.fold_scope.fold_scope_hash)
    chk("fold_result_hash", summary.fold_result_hash, fr.fold_result_hash)
    chk("artifact_scientific_hash", summary.artifact_scientific_hash, artifact_scientific_hash)
    if artifact_pure_science_hash is not None:
        chk("artifact_pure_science_hash", summary.artifact_pure_science_hash, artifact_pure_science_hash)
    chk("level_set", set(summary.levels), set(fr.levels))
    for lvl, lr in fr.level_items:
        s = summary.levels.get(int(lvl))
        if s is None:
            mm.append(f"level{lvl}:missing"); continue
        chk(f"L{lvl}.run_key", s.run_key_hash, lr.run_key.run_key_hash)
        chk(f"L{lvl}.support_hash", s.support_hash, lr.support_state.support_hash)
        chk(f"L{lvl}.level_support_hash", s.level_support_hash, lr.support_state.level_support_hash)
        chk(f"L{lvl}.level_plans_hash", s.level_plans_hash, lr.plans.level_plans_hash)
        chk(f"L{lvl}.level_result_hash", s.level_result_hash, lr.level_result_hash)
        g = lr.support_state.support_graph
        chk(f"L{lvl}.elig", s.eligibility_counts, _rows(np.asarray(g.counts)))
        chk(f"L{lvl}.mass", s.cell_mass, _rows(np.asarray(g.cell_mass)))
        chk(f"L{lvl}.pref", s.reference_prior, tuple(float(x) for x in np.asarray(g.reference_prior).tolist()))
        chk(f"L{lvl}.plans", dict(s.plan_hashes), _memory_plan_hashes(lr.plans))
        chk(f"L{lvl}.ckpts", set(s.checkpoint_hashes), _memory_checkpoint_hashes(lr))
        chk(f"L{lvl}.erm", s.erm_checkpoint, lr.erm_stage.checkpoint.model_hash)
        chk(f"L{lvl}.target_fit", set(s.target_fit_ids), set(lr.provenance.target_fit_ids))
        for name, m in lr.method_items:
            ms = dict(s.method_items).get(name)
            if ms is None:
                mm.append(f"L{lvl}.{name}:missing"); continue
            chk(f"L{lvl}.{name}.ckpt", ms.selected_checkpoint, m.selection.model_hash)
            chk(f"L{lvl}.{name}.R_src", round(ms.R_src, 12), round(float(m.selection.R_src), 12))
            chk(f"L{lvl}.{name}.epoch", ms.selected_epoch, int(m.selection.selected_epoch))
            chk(f"L{lvl}.{name}.sel_status", ms.selection_status, m.selection_status)
            chk(f"L{lvl}.{name}.audit_status", ms.audit_status, m.audit_status)
            chk(f"L{lvl}.{name}.sel_leak", ms.selection_leakage_hash,
                None if m.selection_leakage is None else leakage_result_hash(m.selection_leakage))
            chk(f"L{lvl}.{name}.aud_leak", ms.audit_leakage_hash,
                None if m.audit_leakage is None else leakage_result_hash(m.audit_leakage))
            chk(f"L{lvl}.{name}.preds", dict(ms.prediction_content_hashes), {
                "source_guard": m.source_guard_predictions.prediction_content_hash(),
                "source_audit": m.source_audit_predictions.prediction_content_hash(),
                "target_audit": m.target_predictions.prediction_content_hash()})
            chk(f"L{lvl}.{name}.metrics", dict(ms.metrics_hashes), {
                "source_guard": m.source_guard_metrics.metrics_hash,
                "source_audit": m.source_audit_metrics.metrics_hash, "target_audit": m.target_metrics.metrics_hash})
            chk(f"L{lvl}.{name}.sa_sig", ms.source_audit_signature, m.source_audit_predictions.audit_signature_hash)
            chk(f"L{lvl}.{name}.ta_sig", ms.target_signature, m.target_predictions.audit_signature_hash)
    from ..runner.scientific_hash import scientific_value_hash
    comp = scientific_value_hash({"ok": not mm, "mismatches": sorted(mm), "fold": fr.fold_result_hash,
                                  "artifact": artifact_scientific_hash})
    return SummaryComparisonReport(ok=not mm, mismatches=tuple(sorted(mm)), comparison_hash=comp)


def _memory_plan_hashes(plans) -> dict:
    def opt(p, attr="plan_hash"):
        return "absent" if p is None else getattr(p, attr)
    return {"stage1_task": plans.stage1_task.plan_hash, "stage2_task": plans.stage2_task.plan_hash,
            "oaci_alignment": opt(plans.oaci_alignment), "full_domain_alignment": opt(plans.full_domain_alignment),
            "selection_design": plans.selection_design.population_hash,
            "selection_fold_plan": opt(plans.selection_fold_plan),
            "selection_bootstrap_plan": opt(plans.selection_bootstrap_plan)}


def _memory_checkpoint_hashes(lr) -> set:
    out = {lr.erm_stage.checkpoint.model_hash}
    for _, m in lr.method_items:
        out.add(m.selection.model_hash)
        for c in m.train_result.trajectory:
            out.add(c.model_hash)
    return out

"""Read-only logical summary of a committed artifact (does NOT rebuild or unpickle a FoldRunResult).

``read_completed_artifact`` deep-verifies the tree, then returns a frozen ArtifactSummary of just the
logical identities (hashes + support tables + plan / checkpoint / prediction / metrics hashes +
signatures), suitable for an exact in-memory-vs-on-disk comparison.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

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
    artifact_scientific_hash: str
    fold_result_hash: str
    context_hash: str
    manifest_hash: str
    fold_scope_hash: str
    level_items: tuple                       # ((level, LevelSummary), ...)
    n_indexed_files: int
    n_total_files: int
    n_verified_checkpoints: int
    n_verified_plans: int

    @property
    def levels(self) -> dict:
        return dict(self.level_items)


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
        fold_result_hash=flog, context_hash=marker["context_hash"], manifest_hash=mh,
        fold_scope_hash=fbody["fold_scope_hash"], level_items=levels,
        n_indexed_files=rep.n_indexed_files, n_total_files=rep.n_total_files,
        n_verified_checkpoints=rep.n_verified_checkpoints, n_verified_plans=rep.n_verified_plans)

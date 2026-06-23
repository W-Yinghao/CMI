"""Artifact tree writer + the completeness/consistency gate.

Nothing is written until the whole FoldRunResult re-verifies: every level COMPLETE, the provenance
COMPLETE with empty target fits, the selection snapshot consistent, and the method/level/fold logical
hashes recomputed from the shared payload builders. The level invariants are checked by explicit rules
(NOT ``all(invariants.values())`` -- they mix booleans with counts).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..runner.provenance import RunnerPhase
from ..runner.scientific_hash import scientific_value_hash
from .canonical_json import canonical_json_bytes, canonical_json_hash
from .atomic import StagingDir, _sha256_file
from .checkpoint import CheckpointStore, write_checkpoint_file
from .deterministic_npz import write_deterministic_npz
from . import plan_codec as P
from . import prediction_codec as PR
from . import support_codec as SC
from .result_payload import (fold_result_logical_payload, level_result_logical_payload,
                             method_result_logical_payload)
from .schema import ARTIFACT_PROFILE, ARTIFACT_SCHEMA_VERSION, ALLOWED_METHODS, make_envelope

_REQUIRED_TRUE = ("phase_complete", "selection_snapshot_unchanged", "target_fit_ids_empty",
                  "shared_erm_unique", "shared_tau_unique", "shared_task_plan_unique",
                  "source_audit_signature_match", "target_signature_match")
_ROLES = ("source_guard", "source_audit", "target_audit")


@dataclass(frozen=True)
class ArtifactContext:
    manifest_payload: dict
    manifest_hash: str
    execution_config_payloads: tuple            # ((level, mapping), ...)
    model_spec_payloads: tuple
    git_commit: str
    scientific_tree_clean: bool
    context_hash: str


@dataclass(frozen=True)
class ArtifactWriteResult:
    artifact_dir: str
    artifact_scientific_hash: str
    fold_result_hash: str
    context_hash: str
    n_files: int
    n_checkpoints: int


def context_scientific_hash(manifest_payload, execution_config_payloads, model_spec_payloads) -> str:
    return canonical_json_hash({"manifest": manifest_payload,
                                "execution_config": [[int(l), m] for l, m in execution_config_payloads],
                                "model_spec": [[int(l), m] for l, m in model_spec_payloads]})


def artifact_scientific_hash(fold_result_hash, manifest_hash, context_hash) -> str:
    return scientific_value_hash({"schema_version": ARTIFACT_SCHEMA_VERSION, "fold_result_hash": fold_result_hash,
                                  "manifest_hash": manifest_hash, "context_hash": context_hash})


def _check_invariants(inv: dict) -> None:
    for k in _REQUIRED_TRUE:
        if inv.get(k) is not True:
            raise ValueError(f"required invariant {k!r} is not true")
    for k, v in inv.items():
        if (k.endswith("_cache_reuse_valid") or k.endswith("_inactive_is_erm")) and v is not True:
            raise ValueError(f"required invariant {k!r} is not true")
    if int(inv.get("n_unique_checkpoints", 0)) < 1:
        raise ValueError("n_unique_checkpoints must be >= 1")
    if "oaci_rejected_ineligible_rows" in inv and int(inv["oaci_rejected_ineligible_rows"]) != 0:
        raise ValueError("oaci_rejected_ineligible_rows must be 0")


def _gate(fold_result, context) -> None:
    fr = fold_result
    levels = dict(fr.level_items)
    if not levels:
        raise ValueError("fold result has no levels")
    if context.manifest_hash != fr.fold_scope.fold_key.manifest_hash:
        raise ValueError("context manifest hash != fold scope manifest hash")
    if context_scientific_hash(context.manifest_payload, context.execution_config_payloads,
                               context.model_spec_payloads) != context.context_hash:
        raise ValueError("context hash does not recompute")
    ec = dict((int(l), m) for l, m in context.execution_config_payloads)
    ms = dict((int(l), m) for l, m in context.model_spec_payloads)
    if set(ec) != set(levels) or set(ms) != set(levels):
        raise ValueError("context config/spec payloads do not cover exactly the levels")
    for lvl, lr in levels.items():
        if lr.phase != RunnerPhase.COMPLETE or lr.provenance.phase != RunnerPhase.COMPLETE:
            raise ValueError(f"level {lvl} is not COMPLETE")
        if lr.provenance.target_fit_ids:
            raise ValueError(f"level {lvl} has non-empty target fits")
        if lr.run_key.fold_key.fold_key_hash != fr.fold_scope.fold_key.fold_key_hash:
            raise ValueError(f"level {lvl} FoldKey disagrees with the scope")
        if ec[lvl].get("execution_config_hash") != lr.execution_config_hash:
            raise ValueError(f"level {lvl} execution config payload hash mismatch")
        if ms[lvl].get("model_spec_hash") != lr.model_spec_hash:
            raise ValueError(f"level {lvl} model spec payload hash mismatch")
        _check_invariants(dict(lr.invariant_items))
        for name, m in lr.method_items:
            if name not in ALLOWED_METHODS:
                raise ValueError(f"unknown method {name!r}")
            if scientific_value_hash(method_result_logical_payload(m)) != m.method_result_hash:
                raise ValueError(f"level {lvl} method {name} hash does not recompute")
            for role in _ROLES:
                b = {"source_guard": m.source_guard_predictions, "source_audit": m.source_audit_predictions,
                     "target_audit": m.target_predictions}[role]
                if b.prediction_content_hash() != b.bundle_hash:
                    raise ValueError("prediction bundle hash inconsistent")
        if scientific_value_hash(level_result_logical_payload(lr)) != lr.level_result_hash:
            raise ValueError(f"level {lvl} result hash does not recompute")
    if scientific_value_hash(fold_result_logical_payload(fr)) != fr.fold_result_hash:
        raise ValueError("fold result hash does not recompute")


class _Tree:
    def __init__(self, staging: StagingDir):
        self.s = staging
        self.entries = []

    def _entry(self, rel, kind, logical_hash, ap=None):
        ap = ap or __import__("os").path.join(self.s.staging, rel)
        sha = _sha256_file(ap)
        size = __import__("os").path.getsize(ap)
        self.entries.append({"relative_path": rel, "artifact_kind": kind, "schema_version": ARTIFACT_SCHEMA_VERSION,
                             "byte_size": int(size), "file_sha256": sha, "logical_hash": logical_hash})

    def put(self, rel_base, kind, encoded):
        logical, body, arrays = encoded
        self.json(rel_base, kind, logical, body, arrays)

    def json(self, rel_base, kind, logical_hash, body, arrays=None):
        if arrays is not None:
            npz_rel = rel_base + ".npz"
            meta = write_deterministic_npz(self.s.file_path(npz_rel), arrays)
            self._entry(npz_rel, kind + "_npz", canonical_json_hash(meta))
            body = {**body, "npz": meta}
        doc = make_envelope(kind, logical_hash, body)
        data = canonical_json_bytes(doc)
        rel = rel_base + ".json"
        self.s.write_bytes(rel, data)
        self.entries.append({"relative_path": rel, "artifact_kind": kind, "schema_version": ARTIFACT_SCHEMA_VERSION,
                             "byte_size": len(data), "file_sha256": hashlib.sha256(data).hexdigest(),
                             "logical_hash": logical_hash})

    def checkpoint(self, level_dir, model_hash, state):
        rel_pt = f"{level_dir}/checkpoints/{model_hash}.pt"
        meta = write_checkpoint_file(self.s.file_path(rel_pt), model_hash, state)
        self._entry(rel_pt, "checkpoint_pt", model_hash)
        self.json(f"{level_dir}/checkpoints/{model_hash}", "checkpoint", model_hash, meta)


def _opt(tree, base, kind, encoded):
    """Write an optional (possibly None) plan; record present:false otherwise."""
    if encoded is None:
        tree.json(base, kind, "absent", {"present": False})
        return
    logical, body, arrays = encoded
    tree.json(base, kind, logical, {"present": True, **body}, arrays)


def write_artifact_tree_atomic(fold_result, context, output_root, *, overwrite=False) -> ArtifactWriteResult:
    _gate(fold_result, context)
    fr = fold_result
    a_hash = artifact_scientific_hash(fr.fold_result_hash, context.manifest_hash, context.context_hash)
    import os
    final = os.path.join(str(output_root), a_hash)

    with StagingDir(final, overwrite=overwrite) as st:
        t = _Tree(st)
        # context
        t.json("context/manifest", "manifest", context.manifest_hash, {"manifest": context.manifest_payload})
        t.json("context/execution_config", "execution_config", context.context_hash,
               {"levels": [[int(l), m] for l, m in context.execution_config_payloads]})
        t.json("context/model_spec", "model_spec", context.context_hash,
               {"levels": [[int(l), m] for l, m in context.model_spec_payloads]})
        t.json("context/provenance", "context_provenance", context.context_hash,
               {"git_commit": context.git_commit, "scientific_tree_clean": bool(context.scientific_tree_clean)})
        # fold + scope
        t.json("fold", "fold_result", fr.fold_result_hash,
               {"payload": fold_result_logical_payload(fr), "fold_scope_hash": fr.fold_scope.fold_scope_hash})
        fs = fr.fold_scope
        t.json("scope/fold_scope", "fold_scope", fs.fold_scope_hash,
               {"fold_key_hash": fs.fold_key.fold_key_hash, "maps_hash": fs.maps.maps_hash,
                "target_population_hash": fs.target_population_hash, "target_tensor_hash": fs.target_tensor_hash})
        au = fs.source_audit
        t.json("scope/audit_support", SC.SUPPORT_KIND, au.support_graph.support_hash(),
               {"m": int(au.support_graph.m), "domain_names": list(au.support_graph.domain_names),
                "class_names": list(au.support_graph.class_names), "support_hash": au.support_graph.support_hash()},
               {"eligibility_counts": _np_int(au.support_graph.counts), "cell_mass": _np_f(au.support_graph.cell_mass),
                "reference_prior": _np_f(au.support_graph.reference_prior)})
        _opt(t, "scope/audit_design", P.DESIGN_KIND, P.encode_design(au.design))
        _opt(t, "scope/audit_fold_plan", P.FOLD_KIND, None if au.fold_plan is None else P.encode_fold_plan(au.fold_plan))
        _opt(t, "scope/audit_bootstrap_plan", P.BOOTSTRAP_KIND,
             None if au.bootstrap_plan is None else P.encode_bootstrap_plan(au.bootstrap_plan))

        for lvl, lr in fr.level_items:
            ld = f"levels/level-{int(lvl):03d}"
            t.json(f"{ld}/level", "level_result", lr.level_result_hash,
                   {"payload": level_result_logical_payload(lr), "execution_config_hash": lr.execution_config_hash,
                    "model_spec_hash": lr.model_spec_hash})
            sh, sbody, sarr = SC.encode_support(lr.support_state)
            t.json(f"{ld}/support", SC.SUPPORT_KIND, sh, sbody, sarr)
            t.json(f"{ld}/provenance", "provenance", lr.provenance.provenance_hash,
                   {"phase": lr.provenance.phase.value, "provenance_hash": lr.provenance.provenance_hash,
                    "preprocess_fit_ids": sorted(lr.provenance.preprocess_fit_ids),
                    "optimization_fit_ids": sorted(lr.provenance.optimization_fit_ids),
                    "selection_fit_ids": sorted(lr.provenance.selection_fit_ids),
                    "audit_estimator_fit_ids": sorted(lr.provenance.audit_estimator_fit_ids),
                    "target_fit_ids": sorted(lr.provenance.target_fit_ids)})
            t.json(f"{ld}/invariants", "invariants", scientific_value_hash(dict(lr.invariant_items)),
                   {"invariants": [[k, (v if isinstance(v, (bool, int)) else str(v))] for k, v in lr.invariant_items]})
            t.json(f"{ld}/cache_stats", "cache_stats",
                   scientific_value_hash([lr.audit_cache_stats.stats_hash, lr.prediction_cache_stats.stats_hash]),
                   {"audit_cache_stats_hash": lr.audit_cache_stats.stats_hash,
                    "prediction_cache_stats_hash": lr.prediction_cache_stats.stats_hash})
            lp = lr.plans
            t.put(f"{ld}/plans/stage1_task", P.TASK_KIND, P.encode_task_plan(lp.stage1_task))
            t.put(f"{ld}/plans/stage2_task", P.TASK_KIND, P.encode_task_plan(lp.stage2_task))
            _opt(t, f"{ld}/plans/oaci_alignment", P.ALIGN_KIND,
                 None if lp.oaci_alignment is None else P.encode_alignment_plan(lp.oaci_alignment))
            _opt(t, f"{ld}/plans/full_domain_alignment", P.ALIGN_KIND,
                 None if lp.full_domain_alignment is None else P.encode_alignment_plan(lp.full_domain_alignment))
            t.put(f"{ld}/plans/selection_design", P.DESIGN_KIND, P.encode_design(lp.selection_design))
            _opt(t, f"{ld}/plans/selection_fold_plan", P.FOLD_KIND,
                 None if lp.selection_fold_plan is None else P.encode_fold_plan(lp.selection_fold_plan))
            _opt(t, f"{ld}/plans/selection_bootstrap_plan", P.BOOTSTRAP_KIND,
                 None if lp.selection_bootstrap_plan is None else P.encode_bootstrap_plan(lp.selection_bootstrap_plan))
            # checkpoints (dedup): erm + all trajectory + all selected
            store = CheckpointStore()
            es = lr.erm_stage
            store.add(es.checkpoint.model_hash, es.checkpoint.model_state)
            for name, m in lr.method_items:
                for c in m.train_result.trajectory:
                    store.add(c.model_hash, c.model_state)
                store.add(m.selection.model_hash, m.selection.model_state)
            for mh in store.model_hashes():
                t.checkpoint(ld, mh, store.state(mh))
            t.json(f"{ld}/stage1/erm", "erm_stage", es.stage1_invocation_id,
                   {"checkpoint": es.checkpoint.model_hash, "R_ERM_hat": float(es.R_ERM_hat),
                    "tau": float(es.tau), "invocation_id": es.stage1_invocation_id, "task_plan_hash": es.task_plan_hash})
            # methods
            for name, m in lr.method_items:
                md = f"{ld}/methods/{name}"
                t.json(f"{md}/method", "method_result", m.method_result_hash, method_result_logical_payload(m))
                _opt(t, f"{md}/selection_leakage", PR.LEAKAGE_KIND,
                     None if m.selection_leakage is None else PR.encode_leakage(m.selection_leakage))
                _opt(t, f"{md}/audit_leakage", PR.LEAKAGE_KIND,
                     None if m.audit_leakage is None else PR.encode_leakage(m.audit_leakage))
                dl, dbody, _ = PR.encode_diagnostics(dict(m.training_diagnostics_items))
                t.json(f"{md}/training_diagnostics", PR.DIAGNOSTICS_KIND, dl, dbody)
                metrics_body = {}
                for role, bundle, met in (("source_guard", m.source_guard_predictions, m.source_guard_metrics),
                                          ("source_audit", m.source_audit_predictions, m.source_audit_metrics),
                                          ("target_audit", m.target_predictions, m.target_metrics)):
                    t.put(f"{md}/{role}", PR.PREDICTION_KIND, PR.encode_prediction(bundle))
                    _, mb, _ = PR.encode_metrics(met)
                    metrics_body[role] = mb
                t.json(f"{md}/metrics", PR.METRICS_KIND,
                       scientific_value_hash([metrics_body[r]["metrics_hash"] for r in _ROLES]),
                       {"roles": metrics_body})

        marker = {"schema_version": ARTIFACT_SCHEMA_VERSION, "artifact_profile": ARTIFACT_PROFILE,
                  "artifact_scientific_hash": a_hash, "fold_result_hash": fr.fold_result_hash,
                  "context_hash": context.context_hash}
        n_ck = sum(1 for e in t.entries if e["artifact_kind"] == "checkpoint_pt")
        st.commit(t.entries, marker)
    return ArtifactWriteResult(artifact_dir=final, artifact_scientific_hash=a_hash,
                               fold_result_hash=fr.fold_result_hash, context_hash=context.context_hash,
                               n_files=len(t.entries) + 1, n_checkpoints=n_ck)


import numpy as _npmod


def _np_int(a):
    return _npmod.ascontiguousarray(_npmod.asarray(a))


def _np_f(a):
    return _npmod.ascontiguousarray(_npmod.asarray(a, dtype=_npmod.float64))

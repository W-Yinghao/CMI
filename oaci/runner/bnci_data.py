"""BNCI2014-001 -> FoldData / FoldScope adapter (B1b).

Consumes the offline-first MOABBLoadResult and builds the runner contracts: stable string subject /
recording / trial ids straight from the bundle (NEVER EEGBundle.domain()/group_codes() -- the
contiguous domain map is frozen only by FrozenMaps), MI trial units with unit mass 1, the
target=subject-001 / audit=2,3 / train=4,5,6 split, and a model-seed-independent split identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.eeg.bnci import MOABBLoadResult, load_moabb_confirmatory
from ..models import build_model
from ..models.shallow import validate_shallow_geometry
from ..protocol.manifest_v2 import (load_v2, manifest_logical_payload, manifest_payload_hash,
                                    optimization_manifest_hash)
from .config import ModelSpec, RunnerExecutionConfig
from .data import FoldData
from .keys import FoldKey, RunKey, canonical_json_hash
from .maps import build_frozen_maps
from .scope import ScopePlanConfig, build_fold_scope
from .support import DeletionCell, make_deletion_schedule

_DS = "BNCI2014_001"


@dataclass(frozen=True)
class BNCIRealFold:
    manifest: object
    manifest_payload: dict
    manifest_hash: str
    load_result: MOABBLoadResult
    fold_data: FoldData
    maps: object
    deletion_schedule: object
    fold_scope: object
    scope_config: object
    execution_config: RunnerExecutionConfig
    model_spec: ModelSpec
    shallow_geometry: dict
    raw_data_fingerprint: str
    resolved_preprocess_hash: str
    full_tensor_hash: str
    split_manifest_hash: str
    data_evidence_hash: str

    def model_factory(self):
        bb = self.manifest.backbone
        return lambda: build_model("shallow_convnet", in_chans=22, in_times=385, n_classes=4,
                                   temporal_filters=int(bb.temporal_filters),
                                   temporal_kernel_samples=int(bb.temporal_kernel_samples),
                                   pool_kernel_samples=int(bb.pool_kernel_samples),
                                   pool_stride_samples=int(bb.pool_stride_samples),
                                   dropout=float(bb.dropout), safe_log_eps=float(bb.safe_log_eps))


def _domain(subject_int) -> str:
    return f"{_DS}|subject-{int(subject_int):03d}"


def _fold_spec(manifest):
    """The single-fold subject/deletion spec: the pilot block (a full-budget one-held-out-target fold)
    when present, else the smoke block. Identical fields, so the downstream build is unchanged."""
    spec = manifest.pilot if getattr(manifest, "pilot", None) is not None else manifest.smoke
    if spec is None:
        raise ValueError("manifest has neither a 'pilot' nor a 'smoke' single-fold spec")
    return spec


def build_bnci_fold_from_bundle(manifest, load_result: MOABBLoadResult) -> BNCIRealFold:
    sm = _fold_spec(manifest)
    ds = manifest.enabled_datasets()[_DS]
    bundle = load_result.bundle
    n = bundle.n
    sid = [str(s) for s in bundle.sample_id.tolist()]
    domain_id = [str(d) for d in bundle.subject_id.tolist()]          # stable strings, NOT domain() codes
    group_id = [str(g) for g in bundle.recording_id.tolist()]
    trial = [str(t) for t in bundle.trial_id.tolist()]               # support = mass = eval unit

    train_doms = {_domain(s) for s in sm.source_train_subjects}
    audit_doms = {_domain(s) for s in sm.source_audit_subjects}
    target_doms = {_domain(s) for s in sm.target_subjects}
    role = {**{d: "source_train" for d in train_doms}, **{d: "source_audit" for d in audit_doms},
            **{d: "target_audit" for d in target_doms}}
    role_of = np.array([role[d] for d in domain_id], dtype=object)
    idx = {r: np.where(role_of == r)[0] for r in ("source_train", "source_audit", "target_audit")}

    manifest_hash = manifest.freeze()["sha256"]
    split_manifest_hash = "bnci_split:" + canonical_json_hash({
        "manifest_hash": manifest_hash, "split_seed": int(manifest.seeds.split),
        "role_mapping": {"target": sorted(int(s) for s in sm.target_subjects),
                         "source_audit": sorted(int(s) for s in sm.source_audit_subjects),
                         "source_train": sorted(int(s) for s in sm.source_train_subjects)},
        "role_sample_ids": {r: sorted(sid[i] for i in idx[r].tolist()) for r in idx},
        "class_order": list(bundle.class_names), "channel_order": list(bundle.ch_names),
        "raw_data_fingerprint": bundle.raw_data_fingerprint, "resolved_preprocess_hash": bundle.preprocess_hash})

    fd = FoldData.from_arrays(
        X=torch.from_numpy(np.ascontiguousarray(bundle.X)), y=np.asarray(bundle.y), sample_id=sid,
        domain_id=domain_id, group_id=group_id, support_unit_id=trial, mass_unit_id=trial, eval_unit_id=trial,
        sample_mass=np.ones(n), class_names=list(bundle.class_names),
        source_train_idx=idx["source_train"], source_audit_idx=idx["source_audit"],
        target_audit_idx=idx["target_audit"], preprocess_hash=bundle.preprocess_hash,
        split_manifest_hash=split_manifest_hash, preprocess_fit_ids=frozenset())

    maps = build_frozen_maps(list(bundle.class_names), sorted(train_doms), sorted(audit_doms | target_doms))
    deleted = sm.deleted_cell_level1
    schedule = make_deletion_schedule([DeletionCell(deleted.domain_id, deleted.class_name)], fd, maps)
    cfg = ScopePlanConfig.from_manifest(manifest, support_m=int(ds.support_m))
    target_subj = sorted(int(s) for s in sm.target_subjects)[0]   # smoke=001; pilot=any held-out target
    fold_key = FoldKey(manifest_hash, _DS, f"{_DS}|target-subject-{target_subj:03d}", int(manifest.seeds.split),
                       int(manifest.seeds.deletion), optimization_manifest_hash(manifest))
    fold_scope = build_fold_scope(fold_key, maps, fd, schedule, cfg)
    exec_cfg = RunnerExecutionConfig.from_manifest(manifest)
    bb = manifest.backbone
    geom = validate_shallow_geometry(22, 385, bb)
    model_spec = ModelSpec.build("shallow_convnet",
                                 {"temporal_filters": int(bb.temporal_filters),
                                  "temporal_kernel_samples": int(bb.temporal_kernel_samples),
                                  "pool_kernel_samples": int(bb.pool_kernel_samples),
                                  "pool_stride_samples": int(bb.pool_stride_samples),
                                  "dropout": float(bb.dropout), "safe_log_eps": float(bb.safe_log_eps)},
                                 (22, 385), 4)

    return BNCIRealFold(
        manifest=manifest, manifest_payload=manifest_logical_payload(manifest), manifest_hash=manifest_hash,
        load_result=load_result, fold_data=fd, maps=maps, deletion_schedule=schedule, fold_scope=fold_scope,
        scope_config=cfg, execution_config=exec_cfg, model_spec=model_spec, shallow_geometry=geom,
        raw_data_fingerprint=bundle.raw_data_fingerprint, resolved_preprocess_hash=bundle.preprocess_hash,
        full_tensor_hash=bundle.tensor_content_hash, split_manifest_hash=split_manifest_hash,
        data_evidence_hash=load_result.evidence.evidence_hash)


def build_bnci_real_fold(manifest_path, datalake_root) -> BNCIRealFold:
    m = load_v2(manifest_path); m.validate_complete()
    sm, ds = _fold_spec(m), m.enabled_datasets()[_DS]
    pp = ds.preprocessing
    subjects = sorted(int(s) for s in (sm.subjects or []))
    load_result = load_moabb_confirmatory(
        _DS, subjects, pp, frozen_class_names=ds.class_names, frozen_channels=ds.channels,
        expected_sfreq=float(ds.expected_sfreq), expected_n_times=int(ds.expected_n_times),
        datalake_root=datalake_root)
    return build_bnci_fold_from_bundle(m, load_result)


def target_seen_by_fit(fold_data) -> bool:
    """Computed (never claimed): do any target ids appear in the preprocessing fit set?"""
    return bool(set(fold_data.role_ids("target_audit")) & set(fold_data.preprocess_fit_ids))

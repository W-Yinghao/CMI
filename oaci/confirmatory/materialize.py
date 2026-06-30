"""Confirmatory protocol -> runnable manifest_v2 (a `pilot` manifest for ONE held-out target).

The runnable blocks (risk / backbone / optimizer / training / sampler / probe / methods / evaluation /
seeds / k1 / k2) are copied VERBATIM from the protocol at full budget. The dataset block is completed for
the offline MI loader (kind / baseline / normalization_eps / channel_interpolation / code_version +
expected sfreq / epoch_seconds / n_times). The subject split is made EXPLICIT in a `pilot` block -- never
hidden in Python defaults: target = the held-out subject, source_audit = the first two remaining (ordered)
subjects, source_train = the rest. The materialization is deterministic (no timestamps / no RNG).
"""
from __future__ import annotations

import os

from ..protocol.manifest_v2 import load_v2

# BNCI2014_001 (BCI Competition IV-2a) has nine subjects (1..9); the materialized manifest lists them
# explicitly so the split is auditable.
BNCI2014_001_SUBJECTS = tuple(range(1, 10))
_LOADER_CODE_VERSION = "oaci-bnci-loader-v1"
_NORMALIZATION_EPS = 1.0e-8

# Reduced-uncertainty bootstrap for a one-fold PIPELINE VALIDATION (full TRAINING budget kept). This
# shrinks only the leakage/eval bootstrap (the CPU-bound, single-threaded bottleneck) so the fold finishes
# in reasonable wall-clock; the resulting UCB / CI are NOT confirmatory statistical evidence. The full
# confirmatory keeps the protocol's bootstrap (and will parallelize it).
VALIDATION_BOOTSTRAP = {"selection_bootstrap": 64, "audit_bootstrap": 256, "paired_bootstrap": 256}


def _completed_preprocessing(pp: dict) -> dict:
    return {"kind": "moabb_motor_imagery", "fmin": float(pp["fmin"]), "fmax": float(pp["fmax"]),
            "resample_sfreq": float(pp["resample_sfreq"]), "epoch_tmin": float(pp["epoch_tmin"]),
            "epoch_tmax": float(pp["epoch_tmax"]), "baseline": None, "normalization": str(pp["normalization"]),
            "normalization_eps": _NORMALIZATION_EPS, "channel_interpolation": False,
            "code_version": _LOADER_CODE_VERSION}


def _expected_geometry(pp: dict) -> dict:
    secs = float(pp["epoch_tmax"]) - float(pp["epoch_tmin"])
    n_times = int(round(secs * float(pp["resample_sfreq"]))) + 1     # MNE includes both endpoints
    return {"expected_sfreq": float(pp["resample_sfreq"]), "expected_epoch_seconds": secs,
            "expected_n_times": n_times}


def split_subjects(all_subjects, target_subject, *, source_audit_count=2) -> dict:
    """Deterministic ordered split: target = held-out subject, source_audit = the first
    `source_audit_count` of the remaining (ascending) subjects, source_train = the rest."""
    subs = sorted(int(s) for s in all_subjects)
    t = int(target_subject)
    if t not in subs:
        raise ValueError(f"target subject {t} not in {subs}")
    rest = [s for s in subs if s != t]
    if len(rest) <= source_audit_count:
        raise ValueError("not enough source subjects for the requested source_audit_count")
    return {"subjects": subs, "target_subjects": [t],
            "source_audit_subjects": rest[:source_audit_count], "source_train_subjects": rest[source_audit_count:]}


def materialize_pilot_manifest(protocol, dataset_name, *, target_subject, out_path,
                               all_subjects=None, model_seeds=None, source_audit_count=2,
                               deleted_cell=None, bootstrap_override=None, explicit_split=None):
    """Build + write a runnable `pilot` manifest_v2 yaml for one held-out target; returns
    (out_path, ProtocolManifestV2). Raises (via load_v2 + validate_complete) if anything is incomplete.

    bootstrap_override (e.g. VALIDATION_BOOTSTRAP) shrinks ONLY the leakage/eval bootstrap
    (probe.selection_bootstrap / probe.audit_bootstrap / evaluation.paired_bootstrap) for a reduced-
    uncertainty PIPELINE VALIDATION; the full training budget and everything else are untouched, and the
    materialized protocol_id is tagged so the artifact records that this is not a full-bootstrap run."""
    ds = protocol.dataset(dataset_name)
    pp = ds["preprocessing"]
    subs = list(all_subjects) if all_subjects is not None else list(BNCI2014_001_SUBJECTS)
    if explicit_split is not None:                                   # LOSO: an explicit cyclic split
        split = {k: list(explicit_split[k]) for k in ("subjects", "target_subjects",
                                                      "source_audit_subjects", "source_train_subjects")}
        if [int(target_subject)] != [int(s) for s in split["target_subjects"]]:
            raise ValueError("explicit_split target_subjects must equal [target_subject]")
    else:
        split = split_subjects(subs, target_subject, source_audit_count=source_audit_count)
    if deleted_cell is None:                                          # deterministic: first source_train subject, 'feet'
        first_train = split["source_train_subjects"][0]
        deleted_cell = {"domain_id": f"{dataset_name}|subject-{first_train:03d}", "class_name": "feet"}

    seeds = dict(protocol.block("seeds"))
    if model_seeds is not None:
        seeds["model"] = [int(s) for s in model_seeds]

    dataset_block = {k: ds[k] for k in ("enabled", "cohort_ids", "class_names", "outer_target_factor",
                                        "domain_factor", "group_factor", "support_unit_factor",
                                        "eval_unit_factor", "support_m", "channels")}
    dataset_block.update(_expected_geometry(pp))
    dataset_block["preprocessing"] = _completed_preprocessing(pp)

    probe = dict(protocol.block("probe"))
    evaluation = dict(protocol.block("evaluation"))
    tag = "pilot"
    if bootstrap_override:
        for k in ("selection_bootstrap", "audit_bootstrap"):
            if k in bootstrap_override:
                probe[k] = int(bootstrap_override[k])
        if "paired_bootstrap" in bootstrap_override:
            evaluation["paired_bootstrap"] = int(bootstrap_override["paired_bootstrap"])
        tag = "pilot-validredbootstrap"                               # recorded in protocol_id + manifest hash

    manifest = {
        "protocol_id": f"{protocol.protocol_id}-{tag}-{dataset_name}-target{int(target_subject):03d}",
        "status": "pilot", "seeds": seeds,
        "datasets": {dataset_name: dataset_block},
        "risk": dict(protocol.block("risk")), "backbone": dict(protocol.block("backbone")),
        "optimizer": dict(protocol.block("optimizer")), "training": dict(protocol.block("training")),
        "sampler": dict(protocol.block("sampler")), "probe": probe,
        "methods": dict(protocol.block("methods")), "evaluation": evaluation,
        "k1": dict(protocol.block("k1")), "k2": dict(protocol.block("k2")),
        "pilot": {**split, "deletion_levels": [0, 1], "deleted_cell_level1": dict(deleted_cell)}}

    import yaml
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=True, default_flow_style=False)
    m = load_v2(out_path)
    m.validate_complete()                                            # refuse to emit an unrunnable manifest
    return out_path, m

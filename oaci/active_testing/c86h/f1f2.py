"""C86H F1/F2 — real field-generation orchestration (GATED; the STOP_BEFORE_DATA_ACCESS boundary).

F1 trains the fresh 11-channel 648-model candidate zoo on the legacy sources
Lee2019_MI / Cho2017 / PhysionetMI; F2 generates per-trial candidate probabilities on the
untouched Brandl2020 + ds007221 targets and writes the label-blind split + support gate.

Both require real EEG/label access + GPU and are the authorized step itself: they refuse
without ``授权 C86H`` and cannot be validated in preparation. This module binds the EXACT
existing training entrypoints (so the authorized build orchestrates frozen code, not a fork)
and records the 11-channel retarget + adapter deltas the authorized build must apply. It does
NOT reimplement training; it names what runs and the artifact it must produce.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import contract as K

AUTHORIZATION_TOKEN = "授权 C86H"
REAL_FIELD_MANIFEST_NAME = "C86H_REAL_FIELD_MANIFEST.json"


class C86EError(RuntimeError):
    """Real dataset / field violates a locked assumption -> C86-E blocker."""

# Exact existing entrypoints the authorized F1 build orchestrates (bound, not forked).
F1_TRAINING_ENTRYPOINTS = {
    "train_paired_cell": "oaci.multidataset.c84f_dual_level_training.train_paired_cell",  # 162 units
    "train_level": "oaci.multidataset.c84f_dual_level_training.train_level",              # 81 units
    "materialize_paired_bundles": "oaci.multidataset.c84f_dual_level_training.materialize_paired_bundles",
    "source_loader": "oaci.multidataset.c84f_dual_level_training.load_source_panel_views",
    "stage1": "oaci.train.engine.train_stage1",
    "stage2": "oaci.train.engine.train_stage2",
    "source_subject_contract": "oaci.multidataset.c84f_dual_level_training._source_subject_contract",
}

# F2 target adapters to the 11-channel common interface.
F2_TARGET_ADAPTERS = {
    "Brandl2020_CANONICAL_ADULT_V1":
        "native MOABB MotorImagery path (reuse) retargeted to the 11 INTERFACE_CHANNELS",
    "OpenNeuro_ds007221_HYBRID_ADULT_V1":
        "NEW OpenNeuro/NEMAR BIDS (mne-bids) adapter to the 11-ch interface — DOES NOT EXIST; "
        "the authorized build must implement it (task-hybrid, events left_hand/right_hand)",
}

# 11-channel retarget deltas the authorized build must apply (validated only under authorization).
ELEVEN_CH_RETARGET = [
    "shallow_convnet in_chans 20 -> 11 in every _model_factory (train-build + checkpoint-load)",
    "MOABB MotorImagery paradigm channels = the 11 INTERFACE_CHANNELS",
    "shape guards array.shape[1]==20 -> 11 (epochs/interface + target instrumentation)",
    "regenerate montage_sha256 / model-init hashes / INTERFACE_ID-derived identities for 11 ch",
]


def real_field_manifest_schema() -> dict:
    """The content-addressed real-field manifest F2 must emit before H1 may run — every input
    and output the terminal confirmation is content-addressed against."""
    return {
        "schema": "c86h_real_field_manifest_v1",
        "interface_id": K.COMMON_INTERFACE_ID,
        "field_training_manifest_sha256": K.FIELD_TRAINING_MANIFEST_SHA256,
        "target_key_convention": "field meta['dataset'] MUST be the cohort interface name "
                                 "(Brandl2020_CANONICAL_ADULT_V1 / OpenNeuro_ds007221_HYBRID_ADULT_V1), "
                                 "NOT the native name, so the field keys match the registered "
                                 "target registry (Brandl 1..16, ds007221 37..73)",
        "required": [
            "source_raw_file_sha256 (per legacy source subject epoch array)",
            "trained_weight_sha256 (per of 648 candidate models)",
            "candidate_ids (81 per context, c86_ rule; same across both cohorts; disjoint from c84_)",
            "target_raw_file_sha256 (per Brandl/ds007221 target subject)",
            "target_prediction_sha256 (per of 424 target contexts x 81 candidates)",
            "label_blind_split_manifest (salt C86_TARGET_SPLIT_V1; pool/held trial ids per target)",
            "class_support (>= 8 labels/class/view; failure -> C86-E, no resplit)",
            "loader_or_bids_identity (per cohort)",
        ],
    }


def _sha_file(path: str) -> str:
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def validate_real_field_manifest(field_root: str, contract: dict = None,
                                 zoo_root: str = None) -> dict:
    """Fail-closed CONTENT-ADDRESSED REPLAY of the F2 real-field manifest BEFORE H1 may start.

    Beyond the declared cardinalities, it OPENS and re-hashes every field NPZ (and, when
    ``zoo_root`` is given, every weight file) against the recorded SHAs, checks the exact target
    registry + candidate-ID-by-context registry, trial coverage (pool+held == total, >= minimums),
    label-blind split disjointness, >= MIN_CLASS_SUPPORT per class/view, and overlap == 0. Any
    violation is a C86-E blocker. ``contract`` supplies the locked cardinalities/targets (defaults
    to the registered 53/424/81/648)."""
    if contract is None:
        contract = {"n_targets": K.N_TARGETS, "n_candidates_per_context": K.CANDIDATES_PER_CONTEXT,
                    "n_models": K.UNIQUE_TRAINED_MODELS, "n_contexts": K.TARGET_CONTEXTS}
    nT = contract["n_targets"]; nCand = contract["n_candidates_per_context"]
    nModels = contract["n_models"]; nCtx = contract["n_contexts"]
    path = os.path.join(field_root, REAL_FIELD_MANIFEST_NAME)
    if not os.path.isfile(path):
        raise C86EError("real-field manifest absent; H1 must not start")
    man = json.load(open(path))
    p = []
    if man.get("interface_id") != K.COMMON_INTERFACE_ID:
        p.append("interface_id")
    if man.get("field_training_manifest_sha256") != K.FIELD_TRAINING_MANIFEST_SHA256:
        p.append("field_training_manifest_sha256")
    if man.get("n_targets") != nT:
        p.append(f"n_targets({man.get('n_targets')}!={nT})")
    if man.get("n_contexts") != nT * 8:
        p.append("n_contexts")
    if man.get("n_candidates_per_context") != nCand:
        p.append("n_candidates_per_context")
    if man.get("n_candidate_context_slices") != nT * 8 * nCand:
        p.append("n_candidate_context_slices")
    zoo = man.get("zoo", {})
    if zoo.get("n_models") != nModels:
        p.append("zoo.n_models")
    if len(zoo.get("weight_sha256", {})) != nModels:
        p.append("zoo.weight_count")
    if man.get("construction_evaluation_overlap") != 0:
        p.append("construction_evaluation_overlap")
    # exact target registry
    contract_targets = contract.get("target_cohort")
    if contract_targets is not None:
        got = {tuple(t) for t in man.get("targets", [])}
        if got != {tuple(k) for k in contract_targets}:
            p.append("target_registry_mismatch")
    if len({tuple(t) for t in man.get("targets", [])}) != nT:
        p.append("target_count")
    # candidate-ID-by-context registry: 8 contexts x nCand each
    cbc = zoo.get("candidate_ids_by_context", {})
    if len(cbc) != 8 or any(len(v) != nCand for v in cbc.values()):
        p.append("candidate_ids_by_context")
    # split disjointness + trial coverage + support
    split = man.get("split", {}); cov = man.get("trial_coverage", {})
    if len(split) != nT:
        p.append("split.n_targets")
    for t, sp in split.items():
        pool, held = set(sp.get("pool", [])), set(sp.get("held", []))
        if pool & held:
            p.append(f"split_overlap[{t}]")
        c = cov.get(t, {})
        if c.get("n_pool", -1) + c.get("n_held", -1) != c.get("n_total", -2):
            p.append(f"trial_coverage[{t}]")
        if len(pool) != c.get("n_pool") or len(held) != c.get("n_held"):
            p.append(f"coverage_split_mismatch[{t}]")
    for t, cs in man.get("class_support", {}).items():
        for view in ("pool", "held"):
            for cl in ("0", "1"):
                if int(cs.get(view, {}).get(cl, 0)) < K.MIN_CLASS_SUPPORT:
                    p.append(f"class_support[{t}|{view}|{cl}]")
    # CONTENT-ADDRESSED replay: open + re-hash every field artifact
    for rel, sha in man.get("field_file_sha256", {}).items():
        fp = os.path.join(field_root, rel)
        if not os.path.isfile(fp) or _sha_file(fp) != sha:
            p.append(f"field_file_replay[{rel}]")
            if len([x for x in p if x.startswith("field_file")]) > 5:
                break
    if len(man.get("field_file_sha256", {})) != nT * 8 * 3:      # pool+contrib+held per context
        p.append("field_file_count")
    if zoo_root is not None:                                     # re-hash every weight file
        wfs = zoo.get("weight_file_sha256", {})
        if len(wfs) != nModels:
            p.append("weight_file_count")
        replayed = 0
        for cid, sha in wfs.items():
            wp = os.path.join(zoo_root, f"{cid}.pt")
            if not os.path.isfile(wp) or _sha_file(wp) != sha:
                p.append(f"weight_file_replay[{cid}]")
                replayed += 1
                if replayed > 5:
                    break
    if p:
        raise C86EError(f"real-field manifest invalid (C86-E): {p[:12]}"
                        + (" ..." if len(p) > 12 else ""))
    return man


def _require_auth(authorization: str) -> None:
    if authorization != AUTHORIZATION_TOKEN:
        raise SystemExit("C86H F1/F2 requires authorization '授权 C86H'")


DS007221_BIDS_ROOT = "/projects/EEG-foundation-model/yinghao/ds007221"   # bound at authorized run
_SOURCE_DATASETS = ("Lee2019_MI", "Cho2017", "PhysionetMI")


def _assemble_source(panel, level, per_dataset_loader):
    """Build one (panel, level) source from the 3 legacy datasets' REGISTERED 12-subject training
    panels, applying the REGISTERED level intervention per dataset (level 1 = the locked left_hand
    cell deletion -> 23-cell graph) and a contiguous cross-dataset domain remap. Returns
    (X[n,11,480], y[n], domain[n])."""
    from oaci.multidataset import c84f_dual_level_training as _f
    from oaci.multidataset import c84l1_intervention as _itv
    Xs, ys, subs = [], [], []
    for name in _SOURCE_DATASETS:
        train_ids, _audit = _f._source_subject_contract(name, panel)     # locked 12 (+4 audit)
        loaded = per_dataset_loader(name, list(train_ids))               # {subj: (X, y)}
        subjects, labels, tids, Xstack = [], [], [], []
        for subj, (Xi, yi) in loaded.items():
            yi = np.asarray(yi).astype(int)
            Xstack.append(np.asarray(Xi))
            for i in range(len(yi)):
                subjects.append(int(subj)); labels.append(int(yi[i]))
                tids.append(f"{name}|s{subj}|t{i}")
        Xcat = np.concatenate(Xstack)
        app = _itv.apply_level_intervention(dataset=name, panel=panel, level=int(level),
                                            source_subjects=subjects, source_labels=labels,
                                            source_trial_ids=tids)        # registered intervention
        keep = list(app.keep_indices)
        Xs.append(Xcat[keep]); ys.append(np.asarray(labels)[keep])
        subs.append([(name, subjects[k]) for k in keep])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    flat = [s for group in subs for s in group]
    domain_names = sorted(set(flat))                                     # contiguous domain remap
    dmap = {s: i for i, s in enumerate(domain_names)}
    domain = np.array([dmap[s] for s in flat], dtype=int)
    return X, y, domain


def _real_source_provider():
    """(panel, seed, level) -> registered multi-source (X, y, domain) via the real MOABB adapters."""
    from . import f1f2_field
    def loader(name, subjects):
        return f1f2_field.load_moabb_dataset(name, subjects)
    return lambda panel, seed, level: _assemble_source(panel, level, loader)


def _real_target_provider():
    """Yield (cohort, target_int, X, y, trial_ids, raw_sha) for the 53 registered targets via the
    real Brandl MOABB + ds007221 BIDS adapters. Real-data violations -> C86-E."""
    from . import f1f2_field
    def gen():
        brandl = f1f2_field.load_moabb_dataset("Brandl2020", list(range(1, 17)))
        for subj, (X, y) in brandl.items():
            tids = [f"Brandl2020|s{subj}|t{i}" for i in range(len(y))]
            yield ("Brandl2020_CANONICAL_ADULT_V1", int(subj), np.asarray(X),
                   np.asarray(y).astype(int), tids, f"brandl_{subj}")
        for sub in f1f2_field.DS007221_SUBJECTS:
            X, y, tids, sha = f1f2_field.load_ds007221_bids(DS007221_BIDS_ROOT, sub)
            yield ("OpenNeuro_ds007221_HYBRID_ADULT_V1", int(sub.split("-")[-1]),
                   X, y, tids, sha)
    return gen


def f1_train_zoo(authorization: str, output_root: str, preset=None, source_provider=None,
                 cell_trainer=None):
    """Train the fresh 11-channel 648-model candidate zoo via the frozen engine. Auth-gated; under
    authorization it trains on the real MOABB legacy sources, else refuses. Not a stub."""
    _require_auth(authorization)
    from . import f1f2_train
    preset = preset or f1f2_train.FAITHFUL
    provider = source_provider or _real_source_provider()
    return f1f2_train.f1_train_zoo(provider, output_root, preset=preset, cell_trainer=cell_trainer)


def f2_generate_predictions(authorization: str, zoo_manifest: dict, zoo_root: str,
                            field_root: str, target_provider=None):
    """Generate the 53-target x 424-context field + real-field manifest via the trained zoo and
    the real target adapters. Auth-gated; not a stub."""
    _require_auth(authorization)
    from . import f1f2_train
    provider = target_provider or _real_target_provider()
    return f1f2_train.f2_generate_predictions(zoo_manifest, zoo_root, provider, field_root)

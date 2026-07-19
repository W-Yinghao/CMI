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

from . import contract as K

AUTHORIZATION_TOKEN = "授权 C86H"

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


def _require_auth(authorization: str) -> None:
    if authorization != AUTHORIZATION_TOKEN:
        raise SystemExit("C86H F1/F2 requires authorization '授权 C86H'")


def f1_train_zoo(authorization: str, output_root: str):
    """Train the fresh 11-channel 648-model candidate zoo (authorized step; not runnable in prep)."""
    _require_auth(authorization)
    raise RuntimeError(
        "F1 real 11-channel training on Lee2019_MI/Cho2017/PhysionetMI requires real EEG + GPU "
        "and the 11-ch retarget (see ELEVEN_CH_RETARGET); it orchestrates "
        f"{F1_TRAINING_ENTRYPOINTS['train_paired_cell']} but is the authorized data-access step, "
        "not built/validated in preparation.")


def f2_generate_predictions(authorization: str, output_root: str):
    """Generate target-unlabeled predictions + the label-blind split (authorized step; not prep)."""
    _require_auth(authorization)
    raise RuntimeError(
        "F2 real target prediction requires real Brandl2020/ds007221 EEG and a ds007221 BIDS "
        "adapter (see F2_TARGET_ADAPTERS); it produces the real-field manifest "
        "(real_field_manifest_schema) and is the authorized data-access step, not built in prep.")

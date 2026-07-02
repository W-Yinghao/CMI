"""ACAR V5 Stage-1B feature/embedding dump SCHEMA (pure/stdlib at import; numpy imported LAZILY inside the validator). The fold
feature dump consumed by Stage-2 selection is NOT an opaque blob — it is a PINNED, parseable, LABEL-FREE record set. Every record
carries its provenance (ref/disease/fold/seed + the substrate/config hashes) and its split role, so Stage-2 can parse OOF features
without ever seeing a label and can prove the dump came from the registered frozen substrate.
"""
from __future__ import annotations

SCHEMA_VERSION = "ACAR_V5_STAGE1B_FEAT_DUMP_V1"
SPLIT_ROLES = ("train", "val", "cal", "eval")

# scalar header fields (provenance)
HEADER_FIELDS = ("schema_version", "ref", "disease", "fold", "seed",
                 "preprocessing_config_sha256", "training_config_sha256",
                 "encoder_checkpoint_file_sha256", "source_state_file_sha256")
# per-record parallel arrays
RECORD_ARRAYS = ("subject_key", "split_role", "window_id", "embedding")
# a dump may NEVER carry a label-like field
FORBIDDEN_FIELDS = ("label", "y", "y_te", "labels", "diagnosis", "target", "case_control", "group", "participant_group")

_HEX = "0123456789abcdef"


class FeatureDumpSchemaError(RuntimeError):
    pass


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in _HEX for c in s.lower())


def validate_loaded(mapping):
    """Validate an already-loaded feature dump given as a mapping key -> value (scalars or numpy arrays). numpy imported lazily.
    Fail-closed: missing/extra/forbidden keys, wrong schema version, mismatched lengths, non-float/non-finite embeddings, unknown
    split roles, or a label-like field all raise."""
    import numpy as np  # lazy — never imported at module load

    keys = set(mapping)
    bad = sorted(keys & set(FORBIDDEN_FIELDS))
    if bad:
        raise FeatureDumpSchemaError(f"feature dump carries forbidden label-like field(s) {bad}")
    missing = [k for k in HEADER_FIELDS + RECORD_ARRAYS if k not in keys]
    if missing:
        raise FeatureDumpSchemaError(f"feature dump missing required field(s) {missing}")
    extra = sorted(keys - set(HEADER_FIELDS) - set(RECORD_ARRAYS))
    if extra:
        raise FeatureDumpSchemaError(f"feature dump has unexpected field(s) {extra}")

    def _scalar(k):
        v = mapping[k]
        return v.item() if hasattr(v, "item") and getattr(v, "shape", None) == () else v

    if str(_scalar("schema_version")) != SCHEMA_VERSION:
        raise FeatureDumpSchemaError(f"schema_version != {SCHEMA_VERSION}")
    for h in ("preprocessing_config_sha256", "training_config_sha256",
              "encoder_checkpoint_file_sha256", "source_state_file_sha256"):
        if not _is_hex64(str(_scalar(h))):
            raise FeatureDumpSchemaError(f"{h} is not 64-hex")

    subj = np.asarray(mapping["subject_key"])
    roles = np.asarray(mapping["split_role"])
    wins = np.asarray(mapping["window_id"])
    emb = np.asarray(mapping["embedding"])
    n = subj.shape[0]
    if not (roles.shape[0] == n and wins.shape[0] == n and emb.shape[0] == n):
        raise FeatureDumpSchemaError("subject_key / split_role / window_id / embedding lengths disagree")
    if n == 0:
        raise FeatureDumpSchemaError("feature dump is empty")
    if emb.ndim != 2:
        raise FeatureDumpSchemaError(f"embedding must be 2-D [n_records, dim], got {emb.shape}")
    if emb.dtype.kind != "f" or not bool(np.isfinite(emb).all()):
        raise FeatureDumpSchemaError("embedding must be floating and finite")
    if wins.dtype.kind not in ("i", "u"):
        raise FeatureDumpSchemaError("window_id must be integer")
    role_set = {str(r) for r in roles.tolist()}
    unknown = role_set - set(SPLIT_ROLES)
    if unknown:
        raise FeatureDumpSchemaError(f"unknown split_role(s) {sorted(unknown)}")
    return {"n_records": int(n), "embedding_dim": int(emb.shape[1]), "split_roles_present": sorted(role_set),
            "ref": str(_scalar("ref")), "disease": str(_scalar("disease")),
            "fold": int(_scalar("fold")), "seed": int(_scalar("seed"))}

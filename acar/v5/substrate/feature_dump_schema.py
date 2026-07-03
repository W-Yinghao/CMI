"""ACAR V5 Stage-1B feature/embedding dump SCHEMA (pure/stdlib at import; numpy imported LAZILY inside the validator). The fold
feature dump consumed by Stage-2 selection is NOT an opaque blob — it is a PINNED, parseable, LABEL-FREE record set. Every record
carries its provenance (ref/disease/fold/seed + the substrate/config hashes) and its split role, so Stage-2 can parse OOF features
without ever seeing a label and can prove the dump came from the registered frozen substrate.
"""
from __future__ import annotations

SCHEMA_VERSION = "ACAR_V5_STAGE1B_FEAT_DUMP_V5"   # V5: + per-recording channel-name-repair SUBTYPE map (pure_eeg vs type_prefixed ordinal)
SPLIT_ROLES = ("train", "val", "cal", "eval")

# scalar header fields (provenance). The 4 *_sha256 substrate hashes + the policy hashes + the Stage-1B12/1B13 repair hashes are hex64.
_HEX64_HEADER = ("preprocessing_config_sha256", "training_config_sha256", "encoder_checkpoint_file_sha256",
                 "source_state_file_sha256", "channel_alias_policy_sha256", "montage_completion_policy_sha256",
                 "brainvision_read_repair_policy_sha256", "raw_header_repair_manifest_sha256",
                 "channel_name_repair_policy_sha256")
HEADER_FIELDS = ("schema_version", "ref", "disease", "fold", "seed",
                 "preprocessing_config_sha256", "training_config_sha256",
                 "encoder_checkpoint_file_sha256", "source_state_file_sha256",
                 "channel_alias_policy_sha256", "montage_completion_policy_sha256",
                 "montage_completion_by_subject",   # JSON str: {subject_key: {interpolated,n_interpolated,donor_count}} — NO labels
                 "brainvision_read_repair_policy_sha256", "raw_header_repair_manifest_sha256",
                 "brainvision_read_repair_by_recording",   # JSON str: {subject::recording: {repair_mode, *_sha256}} — NO labels
                 "channel_name_repair_policy_sha256",
                 "channel_name_repair_by_recording",   # JSON str: {subject::recording: {channel_name_source, *_sha256}} — NO labels
                 "channel_name_repair_subtype_by_recording")   # JSON str: {subject::recording: {subtype, ordinal_prefixes}} — NO labels
# per-record parallel arrays
RECORD_ARRAYS = ("subject_key", "split_role", "window_id", "embedding")
# a dump may NEVER carry a label-like field
FORBIDDEN_FIELDS = ("label", "y", "y_te", "labels", "diagnosis", "target", "case_control", "group", "participant_group")

_HEX = "0123456789abcdef"


class FeatureDumpSchemaError(RuntimeError):
    pass


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in _HEX for c in s.lower())


def _flatten_keys(obj):
    """All nested dict keys (to scan a montage-completion map for label-like fields)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(k)
            out.extend(_flatten_keys(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_flatten_keys(v))
    return out


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
    for h in _HEX64_HEADER:
        if not _is_hex64(str(_scalar(h))):
            raise FeatureDumpSchemaError(f"{h} is not 64-hex")
    import json

    def _label_free_json_map(field):
        raw = str(_scalar(field))                               # JSON str → dict; must carry no label-like field
        try:
            parsed = json.loads(raw)
        except ValueError as e:
            raise FeatureDumpSchemaError(f"{field} is not valid JSON: {e}")
        if not isinstance(parsed, dict):
            raise FeatureDumpSchemaError(f"{field} must be a JSON object")
        if set(_flatten_keys(parsed)) & set(FORBIDDEN_FIELDS):
            raise FeatureDumpSchemaError(f"{field} carries a label-like field")
        return parsed

    parsed = _label_free_json_map("montage_completion_by_subject")
    repair_parsed = _label_free_json_map("brainvision_read_repair_by_recording")
    name_repair_parsed = _label_free_json_map("channel_name_repair_by_recording")
    name_repair_subtype_parsed = _label_free_json_map("channel_name_repair_subtype_by_recording")

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
    if emb.shape[1] <= 0:                                     # parser-level: dim must be > 0 (dumper-agnostic barrier)
        raise FeatureDumpSchemaError(f"embedding_dim must be > 0, got {emb.shape[1]}")
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
            "fold": int(_scalar("fold")), "seed": int(_scalar("seed")),
            "channel_alias_policy_sha256": str(_scalar("channel_alias_policy_sha256")),
            "montage_completion_policy_sha256": str(_scalar("montage_completion_policy_sha256")),
            "montage_completion_by_subject": parsed,
            "brainvision_read_repair_policy_sha256": str(_scalar("brainvision_read_repair_policy_sha256")),
            "raw_header_repair_manifest_sha256": str(_scalar("raw_header_repair_manifest_sha256")),
            "brainvision_read_repair_by_recording": repair_parsed,
            "channel_name_repair_policy_sha256": str(_scalar("channel_name_repair_policy_sha256")),
            "channel_name_repair_by_recording": name_repair_parsed,
            "channel_name_repair_subtype_by_recording": name_repair_subtype_parsed}

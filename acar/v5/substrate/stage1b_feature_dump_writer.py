"""ACAR V5 Stage-1B feature-dump WRITER / PARSER (numpy LAZY inside functions; nothing heavy at import). Writes the pinned,
parseable, LABEL-FREE feature dump (a single .npz so it is one hashable artifact = feat_dump_path) and parses+validates it back.
The writer refuses to emit any label-like field and validates the result before returning.
"""
from __future__ import annotations
import json
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5.substrate import preprocessing_config as PC


class FeatureDumpWriteError(RuntimeError):
    pass


def write_feature_dump(path, *, ref, disease, fold, seed, preprocessing_config_sha256, training_config_sha256,
                       encoder_checkpoint_file_sha256, source_state_file_sha256, records,
                       channel_alias_policy_sha256=None, montage_completion_policy_sha256=None,
                       montage_completion_by_subject=None):
    """`records` = iterable of (subject_key, split_role, window_id, embedding_vector). Writes a single .npz at `path` conforming to
    feature_dump_schema, validates it, and returns the validation summary. Fail-closed on an empty dump / bad role / non-finite."""
    import numpy as np  # lazy

    subj, roles, wins, embs = [], [], [], []
    for rec in records:
        if not (isinstance(rec, tuple) and len(rec) == 4):
            raise FeatureDumpWriteError("each record must be (subject_key, split_role, window_id, embedding_vector)")
        sk, role, wid, vec = rec
        if role not in FS.SPLIT_ROLES:
            raise FeatureDumpWriteError(f"bad split_role {role!r}")
        subj.append(str(sk))
        roles.append(str(role))
        wins.append(int(wid))
        embs.append(np.asarray(vec, dtype=np.float64))
    if not embs:
        raise FeatureDumpWriteError("refusing to write an empty feature dump")
    emb = np.stack(embs).astype(np.float32)
    if emb.ndim != 2:
        raise FeatureDumpWriteError("embedding vectors must all have the same 1-D length")
    ca = channel_alias_policy_sha256 or PC.channel_alias_policy_sha256()          # default = the pinned policy hashes
    mc = montage_completion_policy_sha256 or PC.montage_completion_policy_sha256()
    mcbs = json.dumps(montage_completion_by_subject or {}, sort_keys=True, separators=(",", ":"))
    payload = {
        "schema_version": np.asarray(FS.SCHEMA_VERSION), "ref": np.asarray(ref), "disease": np.asarray(disease),
        "fold": np.asarray(int(fold)), "seed": np.asarray(int(seed)),
        "preprocessing_config_sha256": np.asarray(preprocessing_config_sha256),
        "training_config_sha256": np.asarray(training_config_sha256),
        "encoder_checkpoint_file_sha256": np.asarray(encoder_checkpoint_file_sha256),
        "source_state_file_sha256": np.asarray(source_state_file_sha256),
        "channel_alias_policy_sha256": np.asarray(ca), "montage_completion_policy_sha256": np.asarray(mc),
        "montage_completion_by_subject": np.asarray(mcbs),
        "subject_key": np.asarray(subj), "split_role": np.asarray(roles),
        "window_id": np.asarray(wins, dtype=np.int64), "embedding": emb,
    }
    FS.validate_loaded(payload)                               # validate BEFORE writing (fail-closed)
    with open(path, "wb") as f:
        np.savez(f, **payload)                                # numeric/str arrays only → no pickle
    return parse_feature_dump(path)                           # round-trip validate what actually hit disk


def parse_feature_dump(path):
    """Load + validate a feature dump .npz. Returns the schema summary (n_records, embedding_dim, split_roles_present, provenance)."""
    import numpy as np  # lazy
    with np.load(path, allow_pickle=False) as npz:
        mapping = {k: npz[k] for k in npz.files}
    return FS.validate_loaded(mapping)


def load_feature_dump(path):
    """Load + validate a feature dump AND return its per-record arrays (for completeness checks). numpy lazy."""
    import numpy as np  # lazy
    with np.load(path, allow_pickle=False) as npz:
        mapping = {k: npz[k] for k in npz.files}
    summary = FS.validate_loaded(mapping)
    return {"summary": summary,
            "subject_key": [str(x) for x in mapping["subject_key"].tolist()],
            "split_role": [str(x) for x in mapping["split_role"].tolist()],
            "window_id": [int(x) for x in mapping["window_id"].tolist()]}

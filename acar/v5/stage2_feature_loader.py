"""ACAR V5 Stage-2 FEATURE loader (numpy imported LAZILY; nothing heavy at module load). Loads an admitted-package feat_dump and
groups its per-window substrate embeddings by canonical subject_key + split_role, so the action-record layer can form one
batch = one subject's windows. LABEL-FREE by construction: it reads only the feat_dump record arrays (subject_key / split_role /
window_id / embedding) and re-validates the label-free V5 schema before use.

Real-package embedding loading happens ONLY in a real (gated) Stage-2B run; Stage-2B0 exercises this on synthetic fixtures.
"""
from __future__ import annotations
from acar.v5.substrate import feature_dump_schema as FS

FIT_ROLES = ("train", "val")
CAL_ROLE = "cal"
EVAL_ROLE = "eval"


class Stage2FeatureLoadError(RuntimeError):
    """Raised on a malformed/label-bearing feat_dump or a subject with inconsistent split roles (fail-closed)."""


def load_ref_subject_batches(npz_path):
    """Load a feat_dump.npz and return {subject_key: {"split_role": role, "embedding": ndarray[n_win, dim]}}. Re-validates the
    label-free V5 schema. A subject must carry exactly ONE split_role within a fold (else fail-closed)."""
    import numpy as np
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            mapping = {k: z[k] for k in z.files}
    except Exception as e:  # noqa: BLE001
        raise Stage2FeatureLoadError(f"{npz_path}: cannot load feat_dump: {e}")
    try:
        FS.validate_loaded(mapping)                              # label-free + schema (raises on any forbidden/extra field)
    except FS.FeatureDumpSchemaError as e:
        raise Stage2FeatureLoadError(f"{npz_path}: feat_dump failed schema/label-free validation: {e}")
    subj = [str(x) for x in np.asarray(mapping["subject_key"]).tolist()]
    roles = [str(x) for x in np.asarray(mapping["split_role"]).tolist()]
    emb = np.asarray(mapping["embedding"])
    out = {}
    for i, sk in enumerate(subj):
        rec = out.get(sk)
        if rec is None:
            out[sk] = {"split_role": roles[i], "rows": [i]}
        else:
            if rec["split_role"] != roles[i]:
                raise Stage2FeatureLoadError(f"{npz_path}: subject {sk} has inconsistent split_role "
                                             f"({rec['split_role']} vs {roles[i]})")
            rec["rows"].append(i)
    for sk, rec in out.items():
        rec["embedding"] = emb[rec.pop("rows")]                  # [n_win, dim] for this subject
    return out


def subjects_in_group(by_subject, group):
    """Return the sorted subject_keys whose split_role is in `group` ('fit' -> train∪val, 'cal' -> cal, 'eval' -> eval)."""
    if group == "fit":
        wanted = set(FIT_ROLES)
    elif group == "cal":
        wanted = {CAL_ROLE}
    elif group == "eval":
        wanted = {EVAL_ROLE}
    else:
        raise Stage2FeatureLoadError(f"unknown split group {group!r} (want fit/cal/eval)")
    return sorted(sk for sk, rec in by_subject.items() if rec["split_role"] in wanted)

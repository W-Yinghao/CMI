"""C24 — read-only loaders. Reuses the C23 sidecar loader (frozen-config-locked) and adds an HONEST target-
unlabeled availability probe over the committed LOSO artifact root: it counts per-candidate vs method-final
cached target logits so the report can state precisely WHY R3/R4 need re-inference (and never silently proxy)."""
from __future__ import annotations

import glob
import os

from ..score_gauge import artifact_loader as sg_loader
from . import schema

# reuse the frozen-config-locked sidecar loader + helpers verbatim
load = sg_loader.load
by_target = sg_loader.by_target
per_target_offset = sg_loader.per_target_offset
_finite = sg_loader._finite


def target_unlabeled_availability(artifact_root=None, reinfer_sidecar=None) -> dict:
    """Probe committed artifacts for target-UNLABELED logits usable by R3/R4. The offset population is the
    per-candidate feasible-OACI checkpoints; cached target logits are only method-final (wrong population).
    Returns a machine-readable feasibility record -- NEVER fabricates a proxy."""
    root = artifact_root or schema.LOSO_ARTIFACT_ROOT
    method_final = sorted(glob.glob(os.path.join(root, "seed-*", "target-*", "artifacts", "*",
                                                 "levels", "level-*", "methods", "*", "target_audit.npz")))
    sidecar = reinfer_sidecar or schema.C24_TARGET_REINFER_SIDECAR
    per_candidate_ready = os.path.exists(sidecar)
    return {
        "artifact_root": root,
        "method_final_target_audit_count": len(method_final),
        "method_final_note": ("cached target logits are METHOD-FINAL checkpoints (~4 per seed x target x level), "
                              "NOT the ~60 per-seed x target feasible-OACI CANDIDATE checkpoints the offset is "
                              "defined over -- using them as R3/R4 would swap the population; REFUSED as science."),
        "per_candidate_target_unlabeled_ready": per_candidate_ready,
        "reinfer_sidecar": sidecar,
        "r3r4_status": (schema.STATUS_OK if per_candidate_ready else schema.STATUS_REQUIRES_REINFERENCE),
        "example_method_final": (os.path.relpath(method_final[0], root) if method_final else None),
    }


def load_target_unlabeled_sidecar(reinfer_sidecar=None):
    """Stage-3 hook: load per-candidate target-UNLABELED summaries produced by the P0-gated re-inference.
    Returns None when absent (Stage-1). The producer must guarantee NO target labels are stored here."""
    import json
    sidecar = reinfer_sidecar or schema.C24_TARGET_REINFER_SIDECAR
    if not os.path.exists(sidecar):
        return None
    d = json.load(open(sidecar))
    if d.get("config_hash") != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C24 target-unlabeled sidecar config {d.get('config_hash')} != {schema.LOCKED_C19_CONFIG_HASH}")
    return d

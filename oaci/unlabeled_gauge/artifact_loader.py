"""C25 — read-only loaders. Reads the C22 score sidecar (offsets + source features) and the C24 target-
unlabeled sidecar (per-candidate label-free geometry). Fails loud if the C24 sidecar is missing (that is the
only condition that would gate a re-inference). Builds the R3 per-target gauge + a per-candidate join so the
family decomposition / identity audit / R4 interference can all run read-only."""
from __future__ import annotations

import os

from ..information_ladder import artifact_loader as c24_loader
from ..information_ladder import target_unlabeled_features as tuf
from . import schema

load_scores = c24_loader.load                                 # frozen-config-locked C22 sidecar loader
_finite = c24_loader._finite


def load_target_unlabeled(sidecar_path=None):
    sidecar = sidecar_path or schema.C24_TARGET_UNLABELED_SIDECAR
    if not os.path.exists(sidecar):
        raise FileNotFoundError(f"C25 requires the C24 target-unlabeled sidecar {sidecar}; run the P0-gated "
                                f"re-inference first (oaci.information_ladder.target_reinfer).")
    d = c24_loader.load_target_unlabeled_sidecar(sidecar)     # asserts config hash
    return d


def r3_gauge(rows, reinf, mode="in_regime"):
    """The recovering R3 per-target gauge (target-unlabeled moments). Reuses the C24 builder (label-free)."""
    avail = {"per_candidate_target_unlabeled_ready": True}
    out = tuf.build_target_unlabeled_gauge(rows, avail, mode, sidecar=reinf)
    return out["gauge_table"], out["feature_names"]


def per_candidate_join(rows, reinf, mode="in_regime"):
    """Per-candidate rows with the 12 R3 target-unlabeled features attached + target id (for the identity audit).
    Target labels are NEVER attached here."""
    tu = {(c["seed"], c["target"], c["level"], c["model_hash"]): c["target_unlabeled"] for c in reinf["per_candidate"]}
    joined = []
    for r in rows:
        if r["mode"] != mode:
            continue
        key = (r["seed"], r["target"], r["level"], r["model_hash"])
        if key in tu:
            joined.append({"seed": r["seed"], "target": r["target"], "regime": r["regime"],
                           "level": r["level"], "model_hash": r["model_hash"], **tu[key]})
    return joined

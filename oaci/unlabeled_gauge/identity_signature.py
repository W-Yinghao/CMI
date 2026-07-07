"""C25 Q2 — is the R3 recovery a target-MARGINAL geometry signal or a target-IDENTITY signature? Per-target
moments differ by target, so some target-id predictability is EXPECTED and not disqualifying. The decisive
control is the LOTO offset<->gauge permutation (C24): identity fingerprinting cannot help predict a HELD-OUT
(unseen) target's offset, so surviving that null => the recovery is not a pure identity signature. C25 adds a
per-family dissociation: the family that carries the recovery vs the family most predictive of target identity."""
from __future__ import annotations

import numpy as np

from ..score_gauge.identity_leakage_audit import _nearest_centroid_cv
from . import schema


def _matrix(joined, feats):
    X = np.array([[float(r[f]) for f in feats] for r in joined], dtype=np.float64)
    y = np.array([r["target"] for r in joined])
    return X, y


def r3_identity_accuracy(joined, feats=None) -> float:
    feats = feats or list(schema.ALL_R3_FEATURES)
    X, y = _matrix(joined, feats)
    return _nearest_centroid_cv(X, y)


def identity_signature_audit(joined, family_only_results, full_survives_permutation, source_id_accuracy) -> dict:
    r3_acc = r3_identity_accuracy(joined)
    per_family = []
    for fr in family_only_results:
        fam = fr["family"]
        acc = r3_identity_accuracy(joined, list(schema.FAMILIES[fam]))
        per_family.append({"family": fam, "target_id_accuracy": acc, "gap_closed": fr["gap_closed"],
                           "survives_permutation": fr["survives_permutation"]})
    id_sep = bool(r3_acc is not None and r3_acc > schema.IDENTITY_SIGNATURE_CEILING)
    # dissociation: is the family that carries the recovery the SAME as the most identity-predictive family?
    recovering = max(per_family, key=lambda x: (x["gap_closed"] if x["gap_closed"] is not None else -9))
    most_identity = max(per_family, key=lambda x: (x["target_id_accuracy"] if x["target_id_accuracy"] is not None else -9))
    dissociated = bool(recovering["family"] != most_identity["family"])
    # U4 fires only if identity-separable AND the recovery does NOT survive the LOTO permutation control
    identity_dominates = bool(id_sep and not full_survives_permutation)
    return {"r3_target_id_accuracy": r3_acc, "source_target_id_accuracy": source_id_accuracy,
            "chance": schema.IDENTITY_CHANCE, "r3_features_identity_separable": id_sep,
            "recovery_survives_loto_permutation": bool(full_survives_permutation),
            "per_family": per_family, "recovering_family": recovering["family"],
            "most_identity_family": most_identity["family"], "recovery_identity_dissociated": dissociated,
            "identity_signature_dominates": identity_dominates,
            "note": ("R3 features are target-id separable (expected for per-target moments), BUT the recovery "
                     "SURVIVES the LOTO offset-permutation null -> it is target-marginal geometry, not a pure "
                     "identity fingerprint (which cannot help a held-out unseen target)." if not identity_dominates
                     else "recovery does NOT survive the identity/permutation control -> identity signature dominates.")}

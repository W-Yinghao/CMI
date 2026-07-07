"""C26 Q3 — is predicted-class-mix a target-MARGINAL signal or a target-IDENTITY fingerprint? C25 flagged
pred_class_prop as the most identity-predictive family; C26 quantifies the entanglement (target-id accuracy
from predmix vs confidence, nearest-neighbor fingerprint rate vs null) and states the decisive control: the
predmix-only recovery SURVIVING its LOTO offset-permutation null means a transferable marginal relationship, not
a fingerprint (a fingerprint cannot help a held-out unseen target). In LOSO identity and marginal geometry are
not fully separable; C26 reports HOW entangled, not that it is identity-free."""
from __future__ import annotations

import numpy as np

from ..score_gauge.identity_leakage_audit import _nearest_centroid_cv
from . import schema


def _matrix(joined, feats):
    X = np.array([[float(c[f]) for f in feats] for c in joined], dtype=np.float64)
    y = np.array([c["target"] for c in joined])
    return X, y


def _nn_fingerprint(joined, feats, seed=707):
    """Fraction of candidates whose nearest neighbour (in feature space) shares the same target, vs a label-
    shuffle null. High above null => features fingerprint the target."""
    X, y = _matrix(joined, feats)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(1)
    rate = float(np.mean(y[nn] == y))
    rng = np.random.RandomState(seed); null = []
    for _ in range(200):
        yp = y[rng.permutation(len(y))]
        null.append(float(np.mean(yp[nn] == yp)))
    null = np.array(null)
    p = float((np.sum(null >= rate) + 1) / (len(null) + 1))
    return {"nn_same_target_rate": rate, "nn_null_mean": float(null.mean()), "nn_perm_p": p}


def identity_controls(joined, predmix_survives_permutation, predmix_gap) -> dict:
    id_predmix = _nearest_centroid_cv(*_matrix(joined, list(schema.PRED_PROP)))
    id_conf = _nearest_centroid_cv(*_matrix(joined, list(schema.CONF_MARGIN)))
    id_full = _nearest_centroid_cv(*_matrix(joined, list(schema.PRED_PROP) + list(schema.CONF_MARGIN)))
    nn = _nn_fingerprint(joined, list(schema.PRED_PROP))
    id_sep = bool(id_predmix is not None and id_predmix > schema.IDENTITY_SIGNATURE_CEILING)
    fingerprint_dominant = bool(id_sep and not predmix_survives_permutation)
    return {"id_acc_predmix": id_predmix, "id_acc_confidence": id_conf, "id_acc_full": id_full,
            "chance": schema.IDENTITY_CHANCE, "predmix_identity_separable": id_sep,
            "nn_fingerprint": nn, "predmix_recovery_survives_permutation": bool(predmix_survives_permutation),
            "predmix_gap": predmix_gap, "identity_fingerprint_dominant": fingerprint_dominant,
            "note": ("predmix is target-id separable (%.3f > source-side ceiling) AND fingerprints targets "
                     "(NN same-target rate %.3f, p %.3f), BUT the predmix-only recovery SURVIVES the LOTO offset-"
                     "permutation -> transferable marginal relationship, entangled-but-not-pure-fingerprint; "
                     "entanglement DISCLOSED, not claimed identity-free."
                     % ((id_predmix or 0), nn["nn_same_target_rate"], nn["nn_perm_p"]) if not fingerprint_dominant
                     else "predmix recovery does NOT survive the permutation control and is identity-separable -> "
                          "fingerprint dominates.")}

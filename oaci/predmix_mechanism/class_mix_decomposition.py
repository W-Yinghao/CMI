"""C26 Q2 — is the predicted-class-mix signal SIGNED (class-index-specific decision occupancy) or SYMMETRIC
(class-index-invariant concentration)? Signed = the pred_prop c0..c3 vector; symmetric = entropy / max-mass /
distance-to-uniform / Gini of that vector (all invariant to class relabeling). Plus a class-rotation
counterfactual: if cyclically rotating the class indices leaves the recovery unchanged, the signal is not tied
to specific class semantics."""
from __future__ import annotations

import numpy as np

from ..information_ladder import target_unlabeled_features as tuf
from . import artifact_loader, schema


def _symmetric(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1.0); p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    return {"predmix_entropy": ent, "predmix_max_mass": float(p.max()),
            "predmix_dist_uniform": float(np.linalg.norm(p - 1.0 / len(p))),
            "predmix_gini": float(1.0 - (p ** 2).sum())}


def _signed_fn(c):
    return {k: float(c[k]) for k in schema.PRED_PROP}


def _symmetric_fn(c):
    return _symmetric(artifact_loader.predmix_vector(c))


def _rotated_fn(shift):
    def fn(c):
        p = artifact_loader.predmix_vector(c)
        pr = np.roll(p, shift)
        return {schema.PRED_PROP[i]: float(pr[i]) for i in range(len(pr))}
    return fn


def _recover(joined, rows, mode, raw, oracle, feature_fn):
    table, names = artifact_loader.build_gauge(joined, rows, mode, feature_fn)
    perm = tuf.r3_loto_permutation(rows, table, names, mode, raw, oracle)
    return {"gap_closed": perm["gap_closed"], "auc_improve": perm["auc_improve"],
            "perm_p": perm["auc_improve_perm_p"], "survives_permutation": perm["survives_permutation"],
            "loto_r2": perm["loto_r2"]}


def signed_vs_symmetric(joined, rows, mode, raw, oracle) -> dict:
    signed = _recover(joined, rows, mode, raw, oracle, _signed_fn)
    symmetric = _recover(joined, rows, mode, raw, oracle, _symmetric_fn)

    def _both(c):
        return {**_signed_fn(c), **_symmetric_fn(c)}
    both = _recover(joined, rows, mode, raw, oracle, _both)
    signed_carries = bool(signed["survives_permutation"] and (signed["gap_closed"] or -9) >= schema.SUCCESS_GAP_CLOSED)
    symmetric_carries = bool(symmetric["survives_permutation"] and (symmetric["gap_closed"] or -9) >= schema.SUCCESS_GAP_CLOSED)
    return {"signed": signed, "symmetric": symmetric, "signed_plus_symmetric": both,
            "signed_carries": signed_carries, "symmetric_carries": symmetric_carries,
            "signed_specific": bool(signed_carries and not symmetric_carries)}


def _scramble_fn(shifts):
    """Per-target INCONSISTENT class-index roll: breaks the cross-target class correspondence (unlike a GLOBAL
    rotation, which a class-symmetric ridge is trivially invariant to)."""
    def fn(c):
        p = artifact_loader.predmix_vector(c)
        pr = np.roll(p, int(shifts[c["target"]]))
        return {schema.PRED_PROP[i]: float(pr[i]) for i in range(len(pr))}
    return fn


def class_rotation_counterfactual(joined, rows, mode, raw, oracle) -> dict:
    signed = _recover(joined, rows, mode, raw, oracle, _signed_fn)
    # (a) GLOBAL consistent rotation: a control -- a class-symmetric ridge should be ~invariant (column permute)
    glob = []
    for s in schema.CLASS_ROTATIONS:
        r = _recover(joined, rows, mode, raw, oracle, _rotated_fn(s))
        glob.append({"rotation": s, "gap_closed": r["gap_closed"]})
    global_invariant = all(g["gap_closed"] is not None and signed["gap_closed"] is not None
                           and abs(g["gap_closed"] - signed["gap_closed"]) <= schema.ROTATION_INVARIANT_TOL for g in glob)
    # (b) PER-TARGET INCONSISTENT scramble: destroys cross-target class alignment -> tests class-index-specificity
    rng = np.random.RandomState(schema.PERM_SEED)
    targets = sorted({c["target"] for c in joined})
    shifts = {t: rng.randint(1, schema.N_CLASSES) for t in targets}
    scr = _recover(joined, rows, mode, raw, oracle, _scramble_fn(shifts))
    alignment_matters = bool(signed["gap_closed"] is not None and scr["gap_closed"] is not None
                             and scr["gap_closed"] < signed["gap_closed"] - schema.SUCCESS_GAP_CLOSED)
    return {"signed_gap": signed["gap_closed"], "global_rotations": glob, "global_rotation_invariant": bool(global_invariant),
            "per_target_scramble_gap": scr["gap_closed"], "class_index_alignment_matters": alignment_matters,
            "note": ("per-target class-index SCRAMBLE destroys the recovery (global rotation inert) -> the cross-"
                     "target class-index alignment carries the signal (class-index-specific occupancy)"
                     if alignment_matters else
                     "recovery survives per-target class-index scramble -> not tied to cross-target class alignment "
                     "(symmetric / occupancy-magnitude)")}

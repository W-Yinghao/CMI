"""ACAR V6-A0 — sign predictability (Q3), the BINDING gate.

DIAGNOSTIC-ONLY / EXPLORATORY. CODE + SYNTHETIC TESTS ONLY. numpy + sklearn imported LAZILY (NO torch here).

Primary (pinned): pooled action-record L2 logistic regression predicting beneficial(a,B)=1[ΔR_a<0] from label-free features,
subject-GROUPED 5-fold OOF (canonical SubjectKey hash — all of a subject's records stay in one fold), per-disease AUROC, and a
SUBJECT-BLOCK permutation null (1000 perms, seed 0). Model = LogisticRegression(C=1.0, class_weight='balanced', seed 0);
train-standardization fit on TRAIN subjects only. Secondary (descriptive only; never overrides the primary gate): per-action
AUROC/AUPRC, harmful target, calibration, coefficients.
"""
from __future__ import annotations
import hashlib
from acar.v5 import protocol as P

SIGN_CV_SALT = "ACAR_V6A0_SIGN_CV_V1"
N_FOLDS = 5
PRIMARY_C = 1.0
N_PERM = 1000
PERM_MIN_PERMUTABLE = 20             # a subject-block null with fewer permutable subjects is underpowered -> fail-closed
PERM_MIN_PERMUTABLE_FRAC = 0.25      # ...also require >= 25% of subjects permutable
PERM_MIN_VALID = 900                 # need this many valid (non-NaN) permutations for an evaluable p-value
PRIMARY_ACTIONS = P.ACTIONS
_PAIRED = ("d_entropy", "d_margin", "flip_rate", "JS", "Bures", "post_sep", "n_eff")


def sign_cv_fold(subject_key, seed=0, n_folds=N_FOLDS):
    """Deterministic subject-grouped fold (permutation-independent; a DIFFERENT salt than the substrate splits)."""
    h = hashlib.sha256(f"{SIGN_CV_SALT}|{seed}|{subject_key}".encode()).hexdigest()
    return int(h[:8], 16) % n_folds


def feature_names(provenance_tags):
    return (list(_PAIRED) + ["source_confidence", "batch_entropy", "batch_size"]
            + [f"action::{a}" for a in PRIMARY_ACTIONS] + [f"prov::{t}" for t in provenance_tags])


def build_sign_records(records):
    """Flatten eligible EVAL batch records into per-(subject, batch, action) sign records. beneficial = 1[ΔR_a<0] (from ΔR)."""
    prov_tags = sorted({r["provenance"] for r in records})
    out = []
    for r in records:
        for a in PRIMARY_ACTIONS:
            out.append({"subject_key": r["subject_key"], "batch_id": r["batch_id"], "action_id": a,
                        "provenance": r["provenance"], "features": r["features"],
                        "beneficial": int(r["delta_r"][a] < 0.0)})
    return out, prov_tags


def _row(sr, prov_tags):
    import numpy as np
    f = sr["features"]
    pa = f["per_action"][sr["action_id"]]
    row = [pa[k] for k in _PAIRED] + [f["source_confidence"], f["batch_entropy"], float(f["batch_size"])]
    row += [1.0 if sr["action_id"] == a else 0.0 for a in PRIMARY_ACTIONS]
    row += [1.0 if sr["provenance"] == t else 0.0 for t in prov_tags]
    return np.asarray(row, float)


def design_matrix(sign_records, prov_tags):
    import numpy as np
    X = np.vstack([_row(sr, prov_tags) for sr in sign_records])
    # t3a is prob-only -> its Bures/post_sep are structurally NaN; map NaN->0 (the v2 feature_vector convention). The action::t3a
    # one-hot absorbs the resulting constant, so this does not leak or bias across actions.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray([int(sr["beneficial"]) for sr in sign_records], dtype=int)
    groups = np.asarray([str(sr["subject_key"]) for sr in sign_records], dtype=object)
    return X, y, groups


def _oof_scores(X, y, groups, seed):
    """Subject-grouped 5-fold OOF probability of beneficial=1. sklearn LAZY. Standardize on TRAIN subjects only."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    subs = sorted(set(groups.tolist()))
    fold_of = {s: sign_cv_fold(s, seed) for s in subs}
    oof = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        te = np.array([fold_of[g] == k for g in groups])
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd[sd == 0] = 1.0
        if len(set(y[tr].tolist())) < 2:                                        # degenerate train fold -> constant prevalence
            oof[te] = float(y[tr].mean())
            continue
        clf = LogisticRegression(C=PRIMARY_C, class_weight="balanced", max_iter=1000, random_state=seed)
        clf.fit((X[tr] - mu) / sd, y[tr])
        idx = list(clf.classes_).index(1) if 1 in clf.classes_ else None
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, idx] if idx is not None else 0.0
    return oof


def subject_weights(groups):
    """Per-record weight = 1/n_records(subject) so every subject contributes TOTAL weight 1 (subject-balanced: batch-rich subjects
    do NOT dominate the AUROC or the permutation null)."""
    import numpy as np
    from collections import Counter
    c = Counter(str(g) for g in groups)
    return np.asarray([1.0 / c[str(g)] for g in groups], dtype=float)


def _auroc(y, scores, weights=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(scores)
    if len(set(y[m].tolist())) != 2:
        return float("nan")
    sw = None if weights is None else weights[m]
    return float(roc_auc_score(y[m], scores[m], sample_weight=sw))


def primary_sign_auroc(sign_records, prov_tags, seed=0):
    """PRIMARY metric: per-disease SUBJECT-BALANCED OOF AUROC. Subject-grouped OOF scores (unchanged), then a weighted AUROC with
    per-record weight 1/n_records(subject) so each subject is equal-weight."""
    X, y, groups = design_matrix(sign_records, prov_tags)
    return _auroc(y, _oof_scores(X, y, groups, seed), subject_weights(groups))


def record_weighted_sign_auroc(sign_records, prov_tags, seed=0):
    """DESCRIPTIVE only (NOT the gate metric): the unweighted record-level OOF AUROC (batch-rich subjects dominate)."""
    X, y, groups = design_matrix(sign_records, prov_tags)
    return _auroc(y, _oof_scores(X, y, groups, seed), None)


def _subject_index(groups):
    """Return (subjects-in-first-seen order, {subject: [record indices in original order]})."""
    subs, idx_of = [], {}
    for i, g in enumerate(groups):
        s = str(g)
        if s not in idx_of:
            idx_of[s] = []
            subs.append(s)
        idx_of[s].append(i)
    return subs, idx_of


def n_permutable_subjects(groups):
    """Subjects that belong to an equal-record-count stratum with >=2 members (i.e. actually permutable). A LOW value means the
    subject-block null is weak/degenerate — surfaced so V6-A0b can judge null power rather than over-trust perm_p."""
    subs, idx_of = _subject_index(groups)
    by_size = {}
    for s in subs:
        by_size.setdefault(len(idx_of[s]), []).append(s)
    return sum(len(m) for m in by_size.values() if len(m) >= 2)


def subject_block_permute(y, groups, rng):
    """SUBJECT-BLOCK permutation null, RESPECTING record counts: subjects are stratified by their number of records and permuted
    ONLY within each equal-size stratum — a subject's INTACT label-block is reassigned to another same-size subject. This preserves
    the label multiset AND every subject's within-block structure (a subject NEVER receives a within/cross-subject-scrambled label
    vector — the bug the equal-size-only implementation had). Singleton-size subjects are not permutable and keep their own block.
    Identity stratum shuffles return `y` unchanged. Deterministic given `rng`."""
    import numpy as np
    subs, idx_of = _subject_index(groups)
    blocks = {s: [y[i] for i in idx_of[s]] for s in subs}
    by_size = {}
    for s in subs:
        by_size.setdefault(len(idx_of[s]), []).append(s)
    yp = np.array(y, copy=True)
    for _size, members in by_size.items():
        if len(members) < 2:
            continue                                                           # singleton stratum: not permutable (block kept)
        src = list(members)
        rng.shuffle(src)                                                        # permute source-subject order WITHIN the stratum
        for tgt, source in zip(members, src):                                  # target subject <- source subject's INTACT block
            for pos, i in enumerate(idx_of[tgt]):                              # |idx_of[tgt]| == |blocks[source]| (same stratum)
                yp[i] = blocks[source][pos]
    return yp


def permutation_pvalue(sign_records, prov_tags, observed_auroc, seed=0, n_perm=N_PERM):
    """SUBJECT-BLOCK permutation null with the SAME subject-balanced weighting as the observed statistic (so the null and the
    observed AUROC are comparable). Fail-CLOSED: if the subject-block null is underpowered (too few permutable subjects) OR there
    are too few valid permutations, `perm_p_subject_block` is forced to 1.0 with a reason — the continuation gate cannot pass on a
    degenerate null. Raw p = (1+#{null≥obs})/(1+n_valid). Deterministic (seed). sklearn lazy."""
    import numpy as np
    X, y, groups = design_matrix(sign_records, prov_tags)
    n_subjects = len(set(groups.tolist()))
    n_perm_subj = n_permutable_subjects(groups)
    base = {"n_permutable_subjects": n_perm_subj, "n_subjects": n_subjects}
    if not (observed_auroc == observed_auroc):                                # NaN observed statistic -> non-evaluable (fail-closed)
        return {"perm_p_subject_block": 1.0, "raw_p_value": float("nan"), "reason": "observed_auroc_non_evaluable",
                "n_ge": 0, "n_perm_valid": 0, **base}
    if n_perm_subj < PERM_MIN_PERMUTABLE or n_perm_subj < PERM_MIN_PERMUTABLE_FRAC * n_subjects:   # short-circuit (no sklearn)
        return {"perm_p_subject_block": 1.0, "raw_p_value": float("nan"), "reason": "permutation_null_underpowered",
                "n_ge": 0, "n_perm_valid": 0, **base}
    w = subject_weights(groups)
    rng = np.random.RandomState(seed)
    ge = valid = 0
    for _ in range(n_perm):
        yp = subject_block_permute(y, groups, rng)
        a = _auroc(yp, _oof_scores(X, yp, groups, seed), w)                    # SAME subject-balanced weights as observed
        if a == a:                                                             # not NaN
            valid += 1
            if a >= observed_auroc:
                ge += 1
    raw_p = ((1 + ge) / (1 + valid)) if valid else 1.0
    if valid < PERM_MIN_VALID:
        return {"perm_p_subject_block": 1.0, "raw_p_value": raw_p, "reason": "insufficient_valid_permutations",
                "n_ge": ge, "n_perm_valid": valid, **base}
    return {"perm_p_subject_block": raw_p, "raw_p_value": raw_p, "reason": "evaluable",
            "n_ge": ge, "n_perm_valid": valid, **base}

"""C20 — cross-regime permutation null. Same within-(seed,target,level) label-shuffle FAMILY as C19, adapted
to cross-regime generalization: shuffle the DEVELOPMENT training labels inside each fold-level, refit the
frozen probe, and re-run the cross-regime LOTO onto the held-out regime. Answers "does training on TRUE dev
labels beat training on shuffled dev labels for held-out-regime generalization"."""
from __future__ import annotations

import numpy as np

from . import frozen_probe


def _shuffle_dev_labels(dev_pool, rng, label):
    """Return a copy of dev_pool with labels shuffled WITHIN each (seed,target,level) fold-level."""
    by_fold = {}
    for i, r in enumerate(dev_pool):
        by_fold.setdefault((r["seed"], r["target"], r["level"]), []).append(i)
    labels = [1 if r[label] else 0 for r in dev_pool]
    shuffled = list(labels)
    for idxs in by_fold.values():
        vals = [labels[i] for i in idxs]; rng.shuffle(vals)
        for i, v in zip(idxs, vals):
            shuffled[i] = v
    return [dict(r, **{label: bool(shuffled[i])}) for i, r in enumerate(dev_pool)]


def cross_regime_null(dev_pool, val_rows, cols, targets, *, n_perm, perm_seed, label) -> np.ndarray:
    rng = np.random.RandomState(perm_seed); null = []
    for _ in range(int(n_perm)):
        dev_s = _shuffle_dev_labels(dev_pool, rng, label)
        scores, ys = [], []
        for t in targets:
            train = [r for r in dev_s if r["target"] != t]
            test = [r for r in val_rows if r["target"] == t]
            s, y, _, _ = frozen_probe.fit_predict(train, test, cols)
            if s is not None:
                scores.extend(s.tolist()); ys.extend(y.tolist())
        a = frozen_probe.auc(ys, scores)
        if a is not None:
            null.append(a)
    return np.array(null)

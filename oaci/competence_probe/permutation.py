"""C19 — within-(seed,target,level) permutation baseline. Shuffles the diagnostic label INSIDE each fold-level
(never across folds), rebuilds the LOTO AUC, and returns the null distribution. This preserves the fold
structure so the baseline answers "does the probe beat label-shuffling within the same fold granularity"."""
from __future__ import annotations

import numpy as np


def permutation_null(X, y, fold, gt, loto_fn, *, n_perm, perm_seed) -> np.ndarray:
    rng = np.random.RandomState(perm_seed); null = []
    for _ in range(int(n_perm)):
        yp = y.copy()
        for fv in np.unique(fold):
            idx = np.where(fold == fv)[0]
            yp[idx] = rng.permutation(yp[idx])
        a, _ = loto_fn(X, yp, gt)
        if a is not None:
            null.append(a)
    return np.array(null)

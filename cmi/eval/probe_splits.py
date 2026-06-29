"""CIGL Phase 2-real — support-aware trial-level train/val split for the conditional domain probe.

The leakage probe q(D | object, Y) classifies the source domain D (e.g. subject id). For its
held-out KL to be meaningful, the SAME domain categories must appear in both the probe-train and
probe-val splits — a plain random split can leave a subject entirely in val (the classifier is then
asked to predict an unseen domain, which is meaningless), and a row-level split would leak node-rows
of one trial across the boundary. This module splits at the TRIAL level and, whenever a (Y, D) cell
has enough trials, places some in train and some in val so every (label, domain) combination is
represented on both sides. Cells too small to split are kept in TRAIN and reported as low-support —
never silently dropped, never used to fabricate a clean-looking number.
"""
from __future__ import annotations
import numpy as np


def _as_int(a):
    return np.asarray(a).astype(np.int64).ravel()


def stratified_trial_split_by_y_d(y, d, train_frac=0.7, seed=0, min_per_cell=2):
    """Trial-level support-aware split stratified by (Y, D).

    For each (label y, domain d) cell:
      - if the cell has >= `min_per_cell` trials, put ~`train_frac` of them in train and the rest in
        val (at least one trial on each side, so the domain appears in both splits);
      - if the cell has < `min_per_cell` trials, put all of them in TRAIN and flag it low-support.

    Returns (train_idx, val_idx, diagnostics). train_idx and val_idx are disjoint, sorted, and their
    union is exactly range(len(y)) — no trial is duplicated or dropped.
    """
    y = _as_int(y)
    d = _as_int(d)
    n = y.shape[0]
    assert d.shape[0] == n, "y and d must have the same length"
    if n and (y.min() < 0 or d.min() < 0):
        # fail loud: negative sentinels (e.g. -1 "unknown") would be silently dropped from the
        # (Y,D) cell loop (which ranges over 0..max), losing trials without warning.
        raise ValueError("y and d must be non-negative integer labels (got a negative value); "
                         "remap or filter sentinel labels before splitting.")
    n_classes = int(y.max()) + 1 if n else 0
    n_domains = int(d.max()) + 1 if n else 0
    rng = np.random.default_rng(seed)

    train_parts, val_parts = [], []
    n_cells_total = n_cells_split = n_cells_low_support = 0
    low_support_cells = []
    for c in range(n_classes):
        for g in range(n_domains):
            idx = np.where((y == c) & (d == g))[0]
            if idx.size == 0:
                continue
            n_cells_total += 1
            if idx.size < min_per_cell:                       # too small to split -> keep in train
                n_cells_low_support += 1
                low_support_cells.append({"y": c, "d": g, "n": int(idx.size)})
                train_parts.append(idx)
                continue
            perm = rng.permutation(idx)                       # seeded shuffle within the cell
            n_tr = int(round(train_frac * idx.size))
            n_tr = min(max(n_tr, 1), idx.size - 1)            # keep >=1 on each side -> domain in both
            train_parts.append(perm[:n_tr])
            val_parts.append(perm[n_tr:])
            n_cells_split += 1

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=np.int64)

    # integrity: disjoint and covering
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "train/val overlap"
    assert train_idx.size + val_idx.size == n, "split lost or duplicated trials"

    train_dom_support = np.bincount(d[train_idx], minlength=n_domains).tolist() if train_idx.size else [0] * n_domains
    val_dom_support = np.bincount(d[val_idx], minlength=n_domains).tolist() if val_idx.size else [0] * n_domains
    present = sorted(np.unique(d).tolist())
    missing_val = [g for g in present if val_dom_support[g] == 0]
    missing_train = [g for g in present if train_dom_support[g] == 0]

    diagnostics = dict(
        n_trials=int(n),
        n_train=int(train_idx.size),
        n_val=int(val_idx.size),
        n_classes=int(n_classes),
        n_domains=int(n_domains),
        n_cells_total=int(n_cells_total),
        n_cells_split=int(n_cells_split),
        n_cells_low_support=int(n_cells_low_support),
        low_support_cells=low_support_cells,
        train_domain_support=[int(v) for v in train_dom_support],
        val_domain_support=[int(v) for v in val_dom_support],
        missing_val_domains=[int(g) for g in missing_val],
        missing_train_domains=[int(g) for g in missing_train],
        train_frac=float(train_frac),
        min_per_cell=int(min_per_cell),
    )
    if missing_val:
        diagnostics["warning"] = (f"{len(missing_val)} domain(s) absent from the probe-val split "
                                  f"(low support); domain-prediction for those is not evaluated.")
    return train_idx, val_idx, diagnostics

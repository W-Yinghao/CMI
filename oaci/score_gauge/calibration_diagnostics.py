"""C23 — calibration diagnostics: residual offset after gauge, per-target robustness, and a permutation null on
the LOTO offset R^2 (shuffle the offset<->gauge pairing across targets)."""
from __future__ import annotations

import numpy as np

from . import offset_model, schema


def diagnostics(gauge_table, fit) -> dict:
    targets = fit["targets"]
    true = np.array([fit["offset_true"][t] for t in targets])
    hat = np.array([fit["offset_hat_loto"][t] for t in targets])
    resid = true - hat
    # permutation null for LOTO R^2: shuffle which gauge vector maps to which offset, refit LOTO
    import copy
    rng = np.random.RandomState(schema.PERM_SEED); null = []
    offs = [gauge_table[t]["offset"] for t in targets]
    for _ in range(schema.N_PERM):
        perm = rng.permutation(len(targets))
        gt = {t: {**gauge_table[t], "offset": offs[perm[i]]} for i, t in enumerate(targets)}
        r2 = offset_model.fit_offsets(gt)["loto_r2"]
        if r2 is not None:
            null.append(r2)
    null = np.array(null)
    p = float((np.sum(null >= fit["loto_r2"]) + 1) / (len(null) + 1)) if (fit["loto_r2"] is not None and len(null)) else None
    return {"residual_offset_std": float(resid.std()), "true_offset_std": float(true.std()),
            "residual_over_true_std": float(resid.std() / (true.std() + 1e-9)),
            "loto_r2": fit["loto_r2"], "insample_r2": fit["insample_r2"],
            "loto_r2_perm_p": p, "loto_r2_perm_mean": (float(null.mean()) if len(null) else None),
            "loto_beats_permutation": bool(p is not None and p < 0.05 and (fit["loto_r2"] or -1) > 0)}

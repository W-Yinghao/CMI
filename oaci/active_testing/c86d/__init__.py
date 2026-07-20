"""C86D — active-policy development on the accepted C86L field.

Development only; no confirmatory claim. Real P0/A1/A2H execution is gated on a
direct '授权 C86D' (see pipeline.execute). This package implements the process-
isolated client/server, the registered policies, the composite-plugin estimation
with claim boundary, selection freeze, held evaluation with C85U identity
verification, and the exact-CVaR endpoints.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from .core import (  # noqa: F401
    METHOD_FREEZE, PolicyMetrics, C86DClaimError, C86DIdentityError,
    assert_linear_claim, compute_endpoints, exact_upper_cvar, verify_c85u_identity,
)
from .server import QueryClientHandle, start_query_server  # noqa: F401
from .policies import (  # noqa: F401
    acquisition_path, acquisition_score, budget_prefix, composite_from_metrics,
    composite_select, estimate_metrics, load_pool, unbiasedness_claim,
)
from .pipeline import (  # noqa: F401
    AUTHORIZATION_PHRASE, C86DNotAuthorized, C86DOrderingError, HeldEvaluator,
    SelectionFreeze, execute, run_selection,
)

_PSL = [(p, s, l) for p in ("A", "B") for s in (5, 6) for l in (0, 1)]   # 8 contexts
_CONF_BINS = 15


def build_shadow_field(base: str, *, n_targets: int = 4, n_trials: int = 6,
                       n_candidates: int = 81, seed: int = 0):
    """Write a tiny field in the real C86L output format; return roots + held utility."""
    rng = np.random.default_rng(seed)
    pool_root = os.path.join(base, "acquisition_unlabeled_pool")
    contrib_root = os.path.join(base, "query_contribution_store")
    oracle_root = os.path.join(base, "acquisition_label_oracle")
    for r in (pool_root, contrib_root, oracle_root):
        os.makedirs(r, exist_ok=True)
    cand_order = ["ERM:0"] + [f"OACI:{t}" for t in range(1, 41)] + [f"SRC:{t}" for t in range(1, 41)]
    label_rows = []
    for ti in range(n_targets):
        ds, subj = "ShadowDS", ti
        trials = [f"ShadowDS|subject={ti}|trial={j:03d}" for j in range(n_trials)]
        y = rng.integers(0, 2, size=n_trials)
        for j, t in enumerate(trials):
            label_rows.append(dict(dataset=ds, target_subject_id=subj, target_trial_id=t,
                                   session=0, run=0, canonical_class_label=int(y[j]),
                                   split_identity="construction"))
        for (panel, sd, lv) in _PSL:
            ctx = f"panel={panel}|seed={sd}|level={lv}"
            p1 = rng.uniform(0.02, 0.98, size=(n_trials, n_candidates))
            probs = np.stack([1 - p1, p1], axis=2)                    # [n,81,2]
            meta = json.dumps(dict(dataset=ds, subject=subj, panel=panel, seed=sd, level=lv,
                                   n_trials=n_trials, candidate_order=cand_order))
            cid = f"{ds}_{subj}_{panel}_{sd}_{lv}"
            np.savez(os.path.join(pool_root, f"{cid}.npz"), trial_ids=np.array(trials),
                     probabilities=probs.astype(np.float32),
                     candidate_order=np.array(cand_order), meta=meta)
            p_true = probs[np.arange(n_trials)[:, None], np.arange(n_candidates)[None, :], y[:, None]]
            nll = -np.log(np.clip(p_true, 1e-7, 1.0))
            hard = np.argmax(probs, axis=2); correct = (hard == y[:, None]).astype(np.int64)
            conf = probs.max(axis=2)
            cbin = np.minimum((conf * _CONF_BINS).astype(np.int64), _CONF_BINS - 1)
            np.savez(os.path.join(contrib_root, f"{cid}.npz"), trial_ids=np.array(trials),
                     true_label=y.astype(np.int64), nll=nll.astype(np.float32), correct=correct,
                     confidence=conf.astype(np.float32), conf_bin=cbin,
                     signed_calibration=(conf - correct).astype(np.float32),
                     candidate_order=np.array(cand_order), meta=meta)
    with open(os.path.join(oracle_root, "labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(label_rows[0]))
        w.writeheader(); w.writerows(label_rows)
    # held utility must be per (target,context); key by (ds,subj,ctx)
    held_by_ctx = {}
    for ti in range(n_targets):
        for (panel, sd, lv) in _PSL:
            held_by_ctx[(("ShadowDS", ti), f"panel={panel}|seed={sd}|level={lv}")] = \
                rng.uniform(0.4, 0.9, size=n_candidates)
    return {"pool_root": pool_root, "oracle_root": oracle_root, "contrib_root": contrib_root,
            "held_by_ctx": held_by_ctx, "n_targets": n_targets}

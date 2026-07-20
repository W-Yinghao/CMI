"""C31 — FROZEN endpoint-label atlas. Per candidate, diagnostic-only labels from the target endpoint metrics vs
the per-(seed,target,level) ERM reference: accuracy/nll/ece good (improvement > margin), calibration_good,
joint_good, accuracy_good_calibration_bad, calibration_good_accuracy_flat, and Pareto-front membership within
each training trajectory (non-dominated in bAcc↑ / NLL↓ / ECE↓). Base rates are reported before any taxonomy."""
from __future__ import annotations

import numpy as np

from . import schema


def _dominates(a, b):
    # a dominates b iff a >= b on all of (bacc, -nll, -ece) and strictly > on at least one
    ge = (a[0] >= b[0]) and (a[1] <= b[1]) and (a[2] <= b[2])
    gt = (a[0] > b[0]) or (a[1] < b[1]) or (a[2] < b[2])
    return ge and gt


def attach_labels(rows, margin=None) -> list:
    m = schema.IMPROVE_MARGIN if margin is None else margin
    for r in rows:
        bd = (r["bacc"] - r["erm_bacc"]) if (r["bacc"] is not None and r["erm_bacc"] is not None) else None
        nimp = (r["erm_nll"] - r["nll"]) if (r["nll"] is not None and r["erm_nll"] is not None) else None
        eimp = (r["erm_ece"] - r["ece"]) if (r["ece"] is not None and r["erm_ece"] is not None) else None
        r["bacc_delta"] = bd; r["nll_improve"] = nimp; r["ece_improve"] = eimp
        acc = int(bd is not None and bd > m); nll = int(nimp is not None and nimp > m); ece = int(eimp is not None and eimp > m)
        r["accuracy_good"] = acc; r["nll_good"] = nll; r["ece_good"] = ece
        r["calibration_good"] = int(nll or ece)
        r["joint_good"] = int(acc and (nll or ece))
        r["joint_strict_good"] = int(acc and nll and ece)
        r["accuracy_good_calibration_bad"] = int(acc and not (nll or ece))
        r["calibration_good_accuracy_flat"] = int((nll or ece) and not acc)
    # Pareto membership per trajectory (seed,target,level)
    by_traj = {}
    for r in rows:
        by_traj.setdefault((r["seed"], r["target"], r["level"]), []).append(r)
    for traj, cs in by_traj.items():
        pts = [(c, (c["bacc"], c["nll"], c["ece"])) for c in cs if None not in (c["bacc"], c["nll"], c["ece"])]
        for c, p in pts:
            dominated = any(_dominates(q, p) for d, q in pts if d is not c)
            c["pareto_good"] = int(not dominated); c["dominated"] = int(dominated)
        for c in cs:
            if "pareto_good" not in c:
                c["pareto_good"] = 0; c["dominated"] = 0
    return rows


def base_rates(rows) -> dict:
    labels = list(schema.ENDPOINT_LABELS) + ["joint_strict_good"]
    n = len(rows)
    out = {"n_candidates": n}
    for lab in labels:
        vals = [r[lab] for r in rows if r.get(lab) is not None]
        out[lab] = {"rate": (float(np.mean(vals)) if vals else None), "count": int(np.sum(vals)) if vals else 0}
    # accuracy-good that are ALSO calibration-good (does joint exist?)
    accg = [r for r in rows if r["accuracy_good"] == 1]
    out["frac_accuracy_good_also_calibration_good"] = (float(np.mean([r["calibration_good"] for r in accg])) if accg else None)
    return out

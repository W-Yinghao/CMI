"""C10 Part 1 — artifact-only loader. Read every committed C8 fold (9 targets × 3 seeds) and extract, per
(seed, target, level, method), the SELECTED-checkpoint scalars needed for failure diagnosis: selection /
audit leakage, source_audit / source_guard / target_audit worst·mean·pooled bAcc·NLL·ECE, and the selection
metadata (epoch, lambda, R_src, score, n_feasible, erm-fallback). K1 (OACI−ERM held-out audit) is per
(seed, target, level). NOTHING is retrained; this only reads frozen JSON. Rejects missing / failed folds.
"""
from __future__ import annotations

import glob
import json
import os

METHODS = ("ERM", "OACI", "global_lpc", "uniform")
ROLES = ("source_audit", "source_guard", "target_audit")
_SEEDS = (0, 1, 2)
_TARGETS = tuple(range(1, 10))


def _body(p):
    d = json.load(open(p))
    return d.get("body", d)


def _artifact_dir(loso_root, seed, target) -> str:
    adir = os.path.join(loso_root, f"seed-{int(seed)}", f"target-{int(target):03d}", "artifacts")
    cand = sorted(glob.glob(os.path.join(adir, "*", "COMMITTED.json")))
    if len(cand) != 1:
        raise ValueError(f"seed-{seed}/target-{target:03d}: expected exactly one committed artifact, found {len(cand)}")
    return os.path.dirname(cand[0])


def _role_metrics(role_block) -> dict:
    """The 9 scalars we track per role (reference-based bAcc to match the K2 endpoint)."""
    g = role_block.get
    return {"worst_bacc": g("worst_domain_reference_bacc"), "worst_nll": g("worst_domain_nll"),
            "worst_ece": g("worst_domain_ece"), "mean_bacc": g("mean_domain_reference_bacc"),
            "mean_nll": g("mean_domain_nll"), "mean_ece": g("mean_domain_ece"),
            "pooled_bacc": g("pooled_reference_bacc"), "pooled_nll": g("pooled_nll"),
            "pooled_ece": g("pooled_ece"), "reference_status": g("domain_reference_status")}


def _selected_lambda(method_body) -> float | None:
    sel = method_body.get("selection", {}); mh = sel.get("model_hash")
    for c in method_body.get("trajectory", []) or []:
        if isinstance(c, dict) and c.get("model_hash") == mh:
            return c.get("lambda")
    return None


def read_method(artifact, level, method) -> dict:
    """SELECTED-checkpoint record for one (level, method). Leakage + metrics + selection metadata."""
    mdir = os.path.join(artifact, f"levels/level-{int(level):03d}", "methods", method)
    if not os.path.isdir(mdir):
        raise ValueError(f"missing method dir {method} at level {level} in {artifact}")
    mb = _body(os.path.join(mdir, "method.json"))
    sel = mb.get("selection", {})
    sl = _body(os.path.join(mdir, "selection_leakage.json"))
    al = _body(os.path.join(mdir, "audit_leakage.json"))
    metrics = _body(os.path.join(mdir, "metrics.json")).get("roles", {})
    rec = {"method": method, "level": int(level), "active": bool(mb.get("active", False)),
           "selected_erm": bool(sel.get("selected_erm", False)),
           "used_erm_fallback": bool(sel.get("used_erm_fallback", False)),
           "selection_reason": sel.get("selection_reason"), "selection_status": sel.get("selection_status"),
           "selected_epoch": sel.get("selected_epoch"), "selection_score": sel.get("selection_score"),
           "R_src": sel.get("R_src"), "n_feasible": sel.get("n_feasible"),
           "selected_model_hash": sel.get("model_hash"), "selected_lambda": _selected_lambda(mb),
           "sel_leakage": {"bootstrap_ucl": sl.get("bootstrap_ucl"), "L_abs": sl.get("L_abs"),
                           "extractable_LQ_ov": sl.get("extractable_LQ_ov"),
                           "percentile_ucl": sl.get("percentile_ucl")},
           "audit_leakage": {"bootstrap_ucl": al.get("bootstrap_ucl"), "L_abs": al.get("L_abs"),
                             "extractable_LQ_ov": al.get("extractable_LQ_ov")},
           "roles": {r: _role_metrics(metrics.get(r, {})) for r in ROLES}}
    return rec


def read_fold(loso_root, seed, target, *, methods=METHODS) -> dict:
    """One (seed, target) fold: per-level K1 (OACI−ERM audit) + per (level, method) selected records."""
    artifact = _artifact_dir(loso_root, seed, target)
    levels = sorted(int(d.rsplit("-", 1)[-1]) for d in glob.glob(os.path.join(artifact, "levels", "level-*")))
    out = {"seed": int(seed), "target": int(target), "artifact_dir": artifact, "levels": {}}
    for L in levels:
        k1 = _body(os.path.join(artifact, f"levels/level-{L:03d}", "decisions", "k1.json"))
        out["levels"][L] = {"k1": {"status": k1.get("k1_status"), "observed_delta": k1.get("observed_delta"),
                                   "p_lower": k1.get("p_lower"), "p_two_sided": k1.get("p_two_sided")},
                            "methods": {m: read_method(artifact, L, m) for m in methods}}
    return out


def load_all(loso_root, *, seeds=_SEEDS, subjects=_TARGETS, methods=METHODS) -> list:
    """Flat list of per (seed, target, level) records; raises on any missing/failed fold."""
    folds = []
    for s in sorted(int(x) for x in seeds):
        for t in sorted(int(x) for x in subjects):
            folds.append(read_fold(loso_root, s, t, methods=methods))
    n_exp = len(seeds) * len(subjects)
    if len(folds) != n_exp:
        raise ValueError(f"expected {n_exp} folds, loaded {len(folds)}")
    return folds


def flat_records(folds) -> list:
    """One row per (seed, target, level, method) — a tidy table for transfer.py / CSV export."""
    rows = []
    for f in folds:
        for L, lv in f["levels"].items():
            for m, rec in lv["methods"].items():
                rows.append({"seed": f["seed"], "target": f["target"], "level": L, "method": m,
                             "k1_status": lv["k1"]["status"], "k1_observed_delta": lv["k1"]["observed_delta"],
                             "k1_p_lower": lv["k1"]["p_lower"],
                             "sel_leakage_ucl": rec["sel_leakage"]["bootstrap_ucl"],
                             "sel_leakage_point": rec["sel_leakage"]["extractable_LQ_ov"],
                             "audit_leakage_ucl": rec["audit_leakage"]["bootstrap_ucl"],
                             "audit_leakage_point": rec["audit_leakage"]["extractable_LQ_ov"],
                             "selected_epoch": rec["selected_epoch"], "selected_lambda": rec["selected_lambda"],
                             "R_src": rec["R_src"], "n_feasible": rec["n_feasible"],
                             "used_erm_fallback": rec["used_erm_fallback"],
                             **{f"{r}_{k}": rec["roles"][r][k]
                                for r in ROLES for k in ("worst_bacc", "worst_nll", "worst_ece",
                                                         "mean_bacc", "mean_nll", "mean_ece",
                                                         "pooled_bacc", "pooled_nll", "pooled_ece")}})
    return rows

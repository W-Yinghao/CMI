"""CMI-Trace P0.5 -- FMScope protocol bridge.

Reconciles a POSITIVE "FMScope" cohort-conditioned erasure result with our strict source-only TOS
deployment result by running BOTH protocols on the SAME frozen features, as a 2x2 design:

                              | subject-specific erasure | same-rank random removal |
  ORACLE_GLOBAL_GEOMETRY (fit)| A                        | C                        |
  strict source-only    (fit) | B                        | D                        |

Two FIT regimes -- ONLY the eraser-fit population differs; the task probe is IDENTICAL in both
(train a linear head on SOURCE subjects, score the held-out TARGET subject = subject-disjoint probe):

  1. ORACLE_GLOBAL_GEOMETRY (cohort-conditioned DIAGNOSTIC): fit LEACE on the WHOLE audited cohort's
     frozen features + subject ids INCLUDING the outer test subject, but NO task labels. This lets the
     ERASER see target features (its subject axis is explicitly removed). It is an ORACLE DIAGNOSTIC:
     it is NEVER domain-generalisation and NEVER deployable -- every emitted artifact carries
     oracle_global_geometry=True / fit_regime="ORACLE_GLOBAL_GEOMETRY" / is_dg=False / deployable=False.

  2. strict source-only (the TOS deployment contract): fit LEACE on SOURCE subjects only, transform
     source + unseen target, train the head on SOURCE only, score the target only at the very end.

The subject-specific erasure is LEACE (remove the linear subject-identity concept). The same-rank random
removal removes the SAME rank k as the LEACE projector, in the SAME whitened geometry, averaged over
>= n_random draws. Two LEACE implementations are run to rule out an implementation-only explanation:
the repository's current LEACE (`leace_eraser`, empirical covariance) AND a Ledoit-Wolf-whitened LEACE
variant (shrinkage covariance matching FMScope's stated numerical conditioning).

Reuses the strict deployment helpers: `_task_metrics`, `_cluster_ci`, `_subj_after` (erasure_target_deploy),
`leace_eraser`, `_ids` (erasure_baselines), and the three-state `deployment_ci_state` (deployment_ci).

FIREWALL: `source_only_fit(Zs, ys, subj_s, whiten)` takes NO target argument -- the source-only eraser and
head are provably a function of source rows only; target enters only the final `_task_metrics` scoring.

Run on the real frozen dumps (once regenerated -- see feature_dump.py):
  python -m tos_cmi.eeg.fmscope_protocol_bridge --dataset BNCI2014_001 --nrandom 20
A synthetic smoke run (no GPU / no real dumps) writes clearly-tagged SYNTHETIC output:
  python -m tos_cmi.eeg.fmscope_protocol_bridge --synthetic-demo
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import numpy as np
from sklearn.covariance import LedoitWolf

from tos_cmi.eeg.erasure_baselines import _ids, leace_eraser
from tos_cmi.eeg.erasure_target_deploy import _task_metrics, _cluster_ci, _subj_after
from tos_cmi.eeg.deployment_ci import deployment_ci_state, PRACTICAL_THRESHOLD

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
OUT_ROOT = "results/cmi_trace_p0p1/fmscope_bridge"

# fit-regime tags (used verbatim in every row / summary artifact)
ORACLE = "ORACLE_GLOBAL_GEOMETRY"          # cohort-conditioned diagnostic -- NEVER DG, NEVER deployable
SOURCE_ONLY = "strict_source_only"          # the deployable DG contract
BASELINE = "baseline"                        # no-erasure reference

VARIANTS = ("empirical", "ledoit_wolf")     # empirical == repository leace_eraser; ledoit_wolf == LW-LEACE


# ----------------------------- LEACE + random-rank erasers (rank-exposing) -----------------------------
def _onehot(ids, n):
    return np.eye(n)[ids]


def _leace_fit(Xfit, d_onehot, whiten="empirical"):
    """Closed-form LEACE (Belrose+2023) that ALSO returns the removed rank k and the whitening aux, so a
    same-rank random control can be matched exactly. `whiten` selects the covariance used for the metric:
      'empirical'    -> Sigma = X^T X / n   (identical to the repository `leace_eraser`);
      'ledoit_wolf'  -> Sigma = Ledoit-Wolf shrinkage covariance (better numerical conditioning).
    Returns (apply_fn, k, aux) where aux = {mu, Wh=Sigma^{-1/2}, Wh_inv=Sigma^{1/2}, ndim}."""
    mu = Xfit.mean(0)
    Xc = Xfit - mu
    if whiten == "ledoit_wolf":
        Sigma = LedoitWolf(assume_centered=True).fit(Xc).covariance_
    else:
        Sigma = (Xc.T @ Xc) / len(Xc)
    ev, V = np.linalg.eigh(Sigma)
    ev = np.clip(ev, 1e-8, None)
    Wh = V @ np.diag(ev ** -0.5) @ V.T          # Sigma^{-1/2}
    Wh_inv = V @ np.diag(ev ** 0.5) @ V.T        # Sigma^{1/2}
    aux = {"mu": mu, "Wh": Wh, "Wh_inv": Wh_inv, "ndim": Xfit.shape[1]}
    Zc = d_onehot - d_onehot.mean(0)
    Cxz = (Xc.T @ Zc) / len(Xc)                  # cross-covariance [d, c]
    M = Wh @ Cxz                                 # whitened cross-cov
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    U = U[:, s > 1e-6]                           # concept-correlated directions (whitened)
    k = int(U.shape[1])
    d = Xfit.shape[1]
    if k == 0:
        return (lambda X: X), 0, aux
    P = Wh_inv @ U @ U.T @ Wh                     # eraser projector (original space)
    I = np.eye(d)
    return (lambda X: (X - mu) @ (I - P).T + mu), k, aux


def _random_rank_fit(k, aux, rng):
    """Remove k RANDOM orthonormal directions in the SAME whitened geometry as `_leace_fit` (fair same-rank
    control: identical rank, identical whitening, random instead of subject-correlated directions)."""
    d = aux["ndim"]
    if k <= 0:
        return lambda X: X
    mu, Wh, Wh_inv = aux["mu"], aux["Wh"], aux["Wh_inv"]
    G = rng.standard_normal((d, k))
    Ur, _ = np.linalg.qr(G)                       # random orthonormal basis in whitened space
    Ur = Ur[:, :k]
    P = Wh_inv @ Ur @ Ur.T @ Wh
    I = np.eye(d)
    return lambda X: (X - mu) @ (I - P).T + mu


# ----------------------------- strict source-only FIT (FIREWALL: no target argument) -----------------------------
def source_only_fit(Zs, ys, subj_s, whiten="empirical"):
    """STRICT source-only eraser fit. Takes SOURCE arrays ONLY (no target parameter exists) -> the eraser is
    provably a function of source rows. Returns (apply_fn, k, aux). The task head is fit inside `_task_metrics`
    on transformed SOURCE only; the target enters ONLY the final scoring."""
    subj_ids, ns = _ids(subj_s)
    oh = _onehot(subj_ids, ns)
    return _leace_fit(Zs, oh, whiten=whiten)


def oracle_global_fit(Zs, Zt, subj_s, subj_t, whiten="empirical"):
    """ORACLE_GLOBAL_GEOMETRY eraser fit: fit LEACE on the WHOLE cohort (source + target features) with
    subject ids INCLUDING the target subject, but NO task labels. Target features ARE seen here -> this is
    an ORACLE DIAGNOSTIC, never deployable. Returns (apply_fn, k, aux)."""
    Zall = np.concatenate([Zs, Zt], 0)
    subj_all = np.concatenate([np.asarray(subj_s), np.asarray(subj_t)], 0)
    ids_all, n_all = _ids(subj_all)
    oh_all = _onehot(ids_all, n_all)
    return _leace_fit(Zall, oh_all, whiten=whiten)


# ----------------------------- one dump -> A/B/C/D + full rows -----------------------------
def _tags(regime):
    """(oracle_global_geometry, is_dg, deployable, oracle_tag) for a fit regime."""
    if regime == ORACLE:
        return True, False, False, ORACLE
    return False, True, True, ""


def _row(base, cell, regime, removal, apply_fn, aux, k, Zs, ys, Zt, yt, n_cls, seed, n_fit_rows):
    """Score one cell (train head on transformed SOURCE, score transformed TARGET) and build a tagged row."""
    tb, tn, sb, sn = _task_metrics(apply_fn(Zs), ys, apply_fn(Zt), yt, n_cls, np.random.default_rng(seed))
    sd = _subj_after(apply_fn(Zs), _ids(np.asarray(base["subj_s"]))[0], np.random.default_rng(seed))
    og, is_dg, deployable, otag = _tags(regime)
    r = dict(base)
    r.update({"cell": cell, "fit_regime": regime, "removal": removal,
              "oracle_global_geometry": og, "is_dg": is_dg, "deployable": deployable, "oracle_tag": otag,
              "n_fit_rows": int(n_fit_rows), "rank_removed": int(k),
              "tgt_bacc": float(tb), "tgt_nll": float(tn), "src_bacc": float(sb), "src_nll": float(sn),
              "subj_dec_after": float(sd), "chance_task": 1.0 / n_cls})
    return r


def _random_cell(base, cell, regime, aux, k, Zs, ys, Zt, yt, n_cls, seed, n_fit_rows, n_random):
    """Same-rank random removal, averaged over n_random draws (matched whitening from `aux`)."""
    rng = np.random.default_rng(seed + 777)
    acc = {"tgt_bacc": [], "tgt_nll": [], "src_bacc": [], "src_nll": [], "subj_dec_after": []}
    subj_ids = _ids(np.asarray(base["subj_s"]))[0]
    for _ in range(max(1, n_random)):
        E = _random_rank_fit(k, aux, rng)
        tb, tn, sb, sn = _task_metrics(E(Zs), ys, E(Zt), yt, n_cls, np.random.default_rng(seed))
        sd = _subj_after(E(Zs), subj_ids, np.random.default_rng(seed))
        for key, v in zip(["tgt_bacc", "tgt_nll", "src_bacc", "src_nll", "subj_dec_after"], [tb, tn, sb, sn, sd]):
            acc[key].append(v)
    og, is_dg, deployable, otag = _tags(regime)
    r = dict(base)
    r.update({"cell": cell, "fit_regime": regime, "removal": "random_rank",
              "oracle_global_geometry": og, "is_dg": is_dg, "deployable": deployable, "oracle_tag": otag,
              "n_fit_rows": int(n_fit_rows), "rank_removed": int(k), "n_random": int(n_random),
              "tgt_bacc": float(np.mean(acc["tgt_bacc"])), "tgt_nll": float(np.mean(acc["tgt_nll"])),
              "src_bacc": float(np.mean(acc["src_bacc"])), "src_nll": float(np.mean(acc["src_nll"])),
              "subj_dec_after": float(np.mean(acc["subj_dec_after"])), "chance_task": 1.0 / n_cls})
    return r


def bridge_one_dump(dump, seed_override=None, n_random=20):
    """Compute the full 2x2 (A/B/C/D) for BOTH LEACE variants + a shared 'full' baseline, for one dump dict.
    `dump` is a mapping with keys Z_source,y_source,subject_source,Z_target,y_target,subject_target,n_cls,
    backbone,seed,target_subject,dataset,fold. Returns a list of tagged rows (target only enters scoring)."""
    Zs = np.asarray(dump["Z_source"], np.float64)
    ys = np.asarray(dump["y_source"], int)
    Zt = np.asarray(dump["Z_target"], np.float64)
    yt = np.asarray(dump["y_target"], int)
    subj_s = np.asarray(dump["subject_source"])
    subj_t = np.asarray(dump["subject_target"])
    n_cls = int(dump["n_cls"])
    seed = int(seed_override if seed_override is not None else dump["seed"])
    base = {"dataset": str(dump.get("dataset", "?")), "backbone": str(dump.get("backbone", "?")),
            "seed": seed, "fold": str(dump.get("fold", dump.get("target_subject", "?"))),
            "target_subject": int(dump["target_subject"]), "subj_s": subj_s}
    n_src, n_all = len(Zs), len(Zs) + len(Zt)
    rows = []
    # shared FULL baseline (no erasure): identity eraser, source-trained head, target scored
    rows.append(_row(base, "full", BASELINE, "none", (lambda X: X),
                     {"ndim": Zs.shape[1]}, 0, Zs, ys, Zt, yt, n_cls, seed, n_src))
    for variant in VARIANTS:
        # strict source-only eraser (FIREWALL: no target seen at fit)
        so_apply, k_so, aux_so = source_only_fit(Zs, ys, subj_s, whiten=variant)
        # oracle global eraser (target features ARE seen; NO labels)
        or_apply, k_or, aux_or = oracle_global_fit(Zs, Zt, subj_s, subj_t, whiten=variant)
        base_v = dict(base); base_v["leace_variant"] = variant
        # A = oracle LEACE ; B = source-only LEACE
        rows.append(_row(base_v, "A", ORACLE, "subject_leace", or_apply, aux_or, k_or,
                         Zs, ys, Zt, yt, n_cls, seed, n_all))
        rows.append(_row(base_v, "B", SOURCE_ONLY, "subject_leace", so_apply, aux_so, k_so,
                         Zs, ys, Zt, yt, n_cls, seed, n_src))
        # C = oracle random(rank k_or) ; D = source-only random(rank k_so)
        rows.append(_random_cell(base_v, "C", ORACLE, aux_or, k_or, Zs, ys, Zt, yt, n_cls, seed, n_all, n_random))
        rows.append(_random_cell(base_v, "D", SOURCE_ONLY, aux_so, k_so, Zs, ys, Zt, yt, n_cls, seed, n_src, n_random))
    for r in rows:
        r.pop("subj_s", None)
    return rows


# ----------------------------- interpretation-verdict contract -----------------------------
def bridge_verdict(global_pos, source_pos, global_rand_pos, global_above_random, source_above_random,
                   frozen_fm_pos=None, task_trained_null=None):
    """Encode the P0.5 interpretation contract. 'positive' = paired delta bAcc 95% CI lower bound > 0.
      * frozen-FM positive but task-trained-features null -> 'representation_geometry_difference'
        (only reachable with a second, task-trained feature set; None here means not evaluated);
      * global AND source positive, BOTH above their random controls -> 'transferable_subject_axis_benefit';
      * global positive AND same-rank random ALSO positive AND LEACE not above random
        -> 'dimensionality_or_conditioning_effect';
      * global positive, source-only null -> 'target_cohort_conditioned_geometry_explains_discrepancy'.
    The ORACLE (global) mode is NEVER called DG here. Falls through to 'no_clear_pattern'."""
    if frozen_fm_pos is not None and task_trained_null is not None:
        if bool(frozen_fm_pos) and bool(task_trained_null):
            return "representation_geometry_difference"
    if global_pos and source_pos and global_above_random and source_above_random:
        return "transferable_subject_axis_benefit"
    if global_pos and global_rand_pos and not global_above_random:
        return "dimensionality_or_conditioning_effect"
    if global_pos and not source_pos:
        return "target_cohort_conditioned_geometry_explains_discrepancy"
    return "no_clear_pattern"


# ----------------------------- aggregation (paired fold-cluster CIs) -----------------------------
def _paired(idx, keys_a, keys_b, get):
    """Paired deltas get(a)-get(b) over matched (seed,fold) units; fold-cluster CI. keys_* are cell selectors."""
    da, folds = [], []
    for (s, f) in keys_a:
        a, b = idx.get(keys_a[(s, f)]), idx.get(keys_b.get((s, f)))
        if a is not None and b is not None:
            da.append(get(a) - get(b))
            folds.append(f)
    if not da:
        return None
    m, lo, hi = _cluster_ci(da, folds)
    return {"delta_bacc": m, "lo": lo, "hi": hi, "n": len(da),
            "positive": bool(lo > 0.0), "deploy_ci_state": deployment_ci_state(lo, hi, PRACTICAL_THRESHOLD)}


def _cell_index(rows, backbone, variant, cell):
    """{(seed,fold): key} selector for a given backbone/variant/cell (variant ignored for 'full')."""
    out = {}
    for r in rows:
        if r["backbone"] != backbone:
            continue
        if r["cell"] == "full":
            if cell == "full":
                out[(r["seed"], r["fold"])] = ("full", r["backbone"], r["seed"], r["fold"])
        elif r["cell"] == cell and r.get("leace_variant") == variant:
            out[(r["seed"], r["fold"])] = (cell, r["backbone"], r["seed"], r["fold"], variant)
    return out


def aggregate(rows):
    """Paired fold-cluster CIs + verdict per (backbone, variant), plus LW-vs-empirical paired deltas."""
    idx = {}
    for r in rows:
        if r["cell"] == "full":
            idx[("full", r["backbone"], r["seed"], r["fold"])] = r
        else:
            idx[(r["cell"], r["backbone"], r["seed"], r["fold"], r["leace_variant"])] = r
    backbones = sorted(set(r["backbone"] for r in rows))
    getb = lambda r: r["tgt_bacc"]
    summary = {"cells": {
        "A": {"fit_regime": ORACLE, "removal": "subject_leace", "oracle_global_geometry": True,
              "is_dg": False, "deployable": False, "oracle_tag": ORACLE},
        "B": {"fit_regime": SOURCE_ONLY, "removal": "subject_leace", "oracle_global_geometry": False,
              "is_dg": True, "deployable": True, "oracle_tag": ""},
        "C": {"fit_regime": ORACLE, "removal": "random_rank", "oracle_global_geometry": True,
              "is_dg": False, "deployable": False, "oracle_tag": ORACLE},
        "D": {"fit_regime": SOURCE_ONLY, "removal": "random_rank", "oracle_global_geometry": False,
              "is_dg": True, "deployable": True, "oracle_tag": ""},
        "full": {"fit_regime": BASELINE, "removal": "none", "oracle_global_geometry": False,
                 "is_dg": True, "deployable": True, "oracle_tag": ""}},
        "per_backbone": {}}
    for bb in backbones:
        summary["per_backbone"][bb] = {}
        sel = {(v, c): _cell_index(rows, bb, v, c) for v in VARIANTS for c in ("A", "B", "C", "D")}
        full_sel = _cell_index(rows, bb, VARIANTS[0], "full")
        for v in VARIANTS:
            A, Bc, C, D = sel[(v, "A")], sel[(v, "B")], sel[(v, "C")], sel[(v, "D")]
            cmp = {
                "A_vs_full": _paired(idx, A, full_sel, getb),           # global LEACE vs full
                "B_vs_full": _paired(idx, Bc, full_sel, getb),          # source-only LEACE vs full
                "A_vs_C": _paired(idx, A, C, getb),                     # global LEACE vs global random-rank
                "B_vs_D": _paired(idx, Bc, D, getb),                    # source-only LEACE vs source-only random
                "C_vs_full": _paired(idx, C, full_sel, getb),           # global random-rank vs full (control)
            }
            p = lambda key: bool(cmp[key] and cmp[key]["positive"])
            verdict = bridge_verdict(
                global_pos=p("A_vs_full"), source_pos=p("B_vs_full"),
                global_rand_pos=p("C_vs_full"), global_above_random=p("A_vs_C"),
                source_above_random=p("B_vs_D"),
                frozen_fm_pos=None, task_trained_null=None)  # single frozen feature set -> rule not evaluated
            cell_means = {c: float(np.mean([r["tgt_bacc"] for r in rows if r["backbone"] == bb
                          and r["cell"] == c and (c == "full" or r.get("leace_variant") == v)]))
                          for c in ("A", "B", "C", "D", "full")}
            n_folds = len(set((r["seed"], r["fold"]) for r in rows if r["backbone"] == bb))
            summary["per_backbone"][bb][v] = {
                "n_units": n_folds, "cell_tgt_bacc_mean": cell_means, "paired": cmp,
                "verdict": verdict, "oracle_mode_tag": ORACLE,
                "note": "ORACLE_GLOBAL_GEOMETRY (cells A,C) is a diagnostic; NEVER DG, NEVER deployable."}
        # Ledoit-Wolf LEACE vs current (empirical) LEACE, paired, per erasure cell A and B
        lw_cmp = {}
        for c in ("A", "B"):
            lw_cmp["%s_lw_vs_empirical" % c] = _paired(
                idx, _cell_index(rows, bb, "ledoit_wolf", c), _cell_index(rows, bb, "empirical", c), getb)
        summary["per_backbone"][bb]["lw_vs_empirical"] = lw_cmp
    return summary


# ----------------------------- dump loading -----------------------------
def _load_dump(path):
    d = np.load(path, allow_pickle=True)
    fold = re.search(r"sub(\d+)_", os.path.basename(path))
    return {k: d[k] for k in ["Z_source", "y_source", "subject_source", "Z_target", "y_target",
                              "subject_target", "n_cls", "backbone", "seed", "target_subject", "dataset"]
            if k in d} | {"fold": (fold.group(1) if fold else str(int(d["target_subject"])))}


def _find_dumps(dataset):
    dirs = {"EEGNet": "%s/%s_EEGNet_LOSO" % (RESULTS, dataset),
            "TSMNet": "%s/%s_TSMNet_LOSO" % (RESULTS, dataset)}
    tasks = []
    for bb, dd in dirs.items():
        for p in sorted(glob.glob("%s/sub*_erm_lam0_seed*.npz" % dd)):
            tasks.append((bb, p))
    return tasks


# ----------------------------- synthetic dumps (smoke / tests only; clearly tagged) -----------------------------
def make_synthetic_dump(n_subj=6, per_subj=40, zdim=12, n_cls=2, target_subject=0, seed=0,
                        subj_shift=1.4, task_signal=1.1, backbone="SYNTH", dataset="SYNTH"):
    """Gaussian frozen features with (a) a per-subject additive shift on a subject axis and (b) a class-mean
    task signal on a disjoint axis, plus shared noise. LOSO: `target_subject` is held out as target. Returns a
    dump dict in the frozen-dump format. FOR TESTS / SMOKE ONLY -- outputs are tagged SYNTHETIC_SMOKE."""
    rng = np.random.default_rng(seed)
    subj_axis = rng.standard_normal((n_subj, zdim))         # each subject's shift direction/magnitude
    class_axis = rng.standard_normal((n_cls, zdim)) * task_signal
    Z, y, subj = [], [], []
    for s in range(n_subj):
        yy = rng.integers(0, n_cls, size=per_subj)
        base = rng.standard_normal((per_subj, zdim))
        Z.append(base + subj_shift * subj_axis[s] + class_axis[yy])
        y.append(yy)
        subj.append(np.full(per_subj, s))
    Z = np.concatenate(Z, 0).astype("float32")
    y = np.concatenate(y).astype("int64")
    subj = np.concatenate(subj).astype("int64")
    te = subj == target_subject
    tr = ~te
    return {"Z_source": Z[tr], "y_source": y[tr], "subject_source": subj[tr],
            "Z_target": Z[te], "y_target": y[te], "subject_target": subj[te],
            "n_cls": np.int64(n_cls), "backbone": backbone, "seed": np.int64(seed),
            "target_subject": np.int64(target_subject), "dataset": dataset, "fold": str(target_subject)}


# ----------------------------- writing -----------------------------
def _write(out_dir, rows, summary, meta):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["dataset", "backbone", "seed", "fold", "target_subject", "cell", "fit_regime", "removal",
            "leace_variant", "oracle_global_geometry", "is_dg", "deployable", "oracle_tag",
            "n_fit_rows", "rank_removed", "tgt_bacc", "tgt_nll", "src_bacc", "src_nll",
            "subj_dec_after", "chance_task"]
    with open("%s/fmscope_bridge_rows.csv" % out_dir, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    with open("%s/fmscope_bridge_rows.jsonl" % out_dir, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump({"meta": meta, "summary": summary}, open("%s/fmscope_bridge_summary.json" % out_dir, "w"), indent=1)


def _print(summary, meta):
    print("\n=== FMScope protocol bridge (%s) ===" % meta.get("data_source"))
    print("cells: A=oracle+LEACE  B=source-only+LEACE  C=oracle+random  D=source-only+random  (full=no erasure)")
    for bb, byv in summary["per_backbone"].items():
        for v in VARIANTS:
            if v not in byv:
                continue
            s = byv[v]
            cm = s["cell_tgt_bacc_mean"]
            print("\n[%s | %s]  n_units=%d  tgt_bAcc  full=%.3f A=%.3f B=%.3f C=%.3f D=%.3f"
                  % (bb, v, s["n_units"], cm["full"], cm["A"], cm["B"], cm["C"], cm["D"]))
            for name, c in s["paired"].items():
                if c is None:
                    continue
                print("   %-12s d_bAcc=%+.3f [%+.3f,%+.3f] pos=%s (%s)"
                      % (name, c["delta_bacc"], c["lo"], c["hi"], c["positive"], c["deploy_ci_state"]))
            print("   VERDICT: %s   (ORACLE mode A/C is DIAGNOSTIC -- never DG/deployable)" % s["verdict"])
        lw = byv.get("lw_vs_empirical", {})
        for name, c in lw.items():
            if c:
                print("   %-18s d_bAcc=%+.3f [%+.3f,%+.3f]" % (name, c["delta_bacc"], c["lo"], c["hi"]))
    print("FMSCOPE_BRIDGE_DONE")


# ----------------------------- CLI -----------------------------
def run(dump_dicts, out_dir, data_source, n_random=20, seed_override=None, extra_meta=None):
    """Core entrypoint: compute the 2x2 over a list of dump dicts, aggregate, write, print. Returns summary."""
    rows = []
    for d in dump_dicts:
        rows.extend(bridge_one_dump(d, seed_override=seed_override, n_random=n_random))
    summary = aggregate(rows)
    meta = {"phase": "CMI_TRACE_P0P5_fmscope_protocol_bridge", "data_source": data_source,
            "oracle_mode_tag": ORACLE, "n_random": n_random, "leace_variants": list(VARIANTS),
            "design": "2x2 {ORACLE_GLOBAL_GEOMETRY, strict_source_only} x {subject_leace, random_rank}",
            "firewall": "source_only_fit takes no target argument; target enters only _task_metrics scoring",
            "interpretation_contract": {
                "global_pos & source_null": "target_cohort_conditioned_geometry_explains_discrepancy",
                "global_pos & random_pos & not_above_random": "dimensionality_or_conditioning_effect",
                "global_pos & source_pos & both_above_random": "transferable_subject_axis_benefit",
                "frozen_fm_pos & task_trained_null": "representation_geometry_difference"},
            "n_dumps": len(dump_dicts)}
    if extra_meta:
        meta.update(extra_meta)
    _write(out_dir, rows, summary, meta)
    _print(summary, meta)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--nrandom", type=int, default=20)
    ap.add_argument("--synthetic-demo", action="store_true",
                    help="no real dumps / no GPU: run on generated synthetic dumps (output tagged SYNTHETIC_SMOKE)")
    a = ap.parse_args()
    if a.synthetic_demo:
        dumps = [make_synthetic_dump(target_subject=t, seed=s, dataset="SYNTH", backbone="SYNTH")
                 for s in (0, 1) for t in range(6)]
        out = "%s/synthetic" % OUT_ROOT   # under results/.gitignore -> NOT committed
        run(dumps, out, "SYNTHETIC_SMOKE", n_random=a.nrandom,
            extra_meta={"WARNING": "SYNTHETIC gaussian smoke data -- NOT a scientific result"})
        print("(synthetic demo written to %s ; NOT a real result)" % out)
        return
    tasks = _find_dumps(a.dataset)
    if not tasks:
        print("NO frozen dumps found under %s/%s_{EEGNet,TSMNet}_LOSO/sub*_erm_lam0_seed*.npz" % (RESULTS, a.dataset))
        print("Regenerate them first (GPU), e.g. per fold/seed via feature_dump.dump_fold(...), then re-run:")
        print("  python -m tos_cmi.eeg.fmscope_protocol_bridge --dataset %s --nrandom %d" % (a.dataset, a.nrandom))
        print("FMSCOPE_BRIDGE_NO_DUMPS")
        return
    dumps = []
    for bb, p in tasks:
        d = _load_dump(p)
        d["backbone"] = d.get("backbone", bb)
        dumps.append(d)
    out = "%s/%s" % (OUT_ROOT, a.dataset)
    run(dumps, out, "REAL", n_random=a.nrandom,
        extra_meta={"dataset": a.dataset, "source_dumps": [p for _, p in tasks]})


if __name__ == "__main__":
    main()

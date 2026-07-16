#!/usr/bin/env python
"""Gate 5 (rule-level cross-fitted certification) for the target-X selection RULE (amendment 03 C5).

Certifies the SAME deployable rule, not a conveniently re-measured projector. Per outer fold:
  source -> eraser-fit / posterior-train / posterior-eval (trial-disjoint);
  rebuild cond basis on the ERASER-FIT partition;
  apply the SAME G1 + source-task-safety + random-specificity rule on T_cal X to select the action;
  apply that selected action (and exact-rank ambient random controls) to the source posterior partitions;
  measure conditional subject leakage with a TRUE-LINEAR (logistic) critic + MLP-small + MLP-large, each with a
  FULLY-RETRAINED within-label permutation null; report dI_specific = dI_selected - mean dI_random.
Emits {dataset: di_specific_lcb} for the aggregator's Gate 5. --smoke = linear + mlp_small + few perms.

  python scripts/run_targetx_gate5_rule.py --smoke
"""
from __future__ import annotations
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from sklearn.linear_model import LogisticRegression
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eval.dg_identifiability import get_candidate_basis
from tos_cmi.eval.targetx_observability import (build_actions, g1_select, session_split, ambient_random_projectors,
                                                observable_G1, _orthonormal)
from cmi.eval.conditional_subject_leakage import three_way_support_split
from cmi.eval.graph_leakage import fit_conditional_domain_probe

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _linear_cond_kl(Z, y, d, n_dom, ptr, pev):
    """TRUE-LINEAR conditional-subject-leakage: E[log p(d|z,y) - log p(d|y)] with a logistic critic on
    [Z, onehot(y)] (label-conditioned). Held-out on pev; prior p(d|y) from ptr within-label frequencies."""
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
    oh = np.eye(len(np.unique(y)))[np.searchsorted(np.unique(y), y)]
    X = np.hstack([Z, oh])
    if len(np.unique(d[ptr])) < 2:
        return 0.0
    clf = LogisticRegression(max_iter=300, C=1.0).fit(X[ptr], d[ptr])
    P = np.clip(clf.predict_proba(X[pev]), 1e-6, 1.0)
    classes = clf.classes_
    # prior p(d|y) from ptr
    kl = []
    for i, idx in enumerate(pev):
        yy = y[idx]
        pri = np.array([max((d[ptr][(y[ptr] == yy)] == c).mean() if (y[ptr] == yy).any() else 1.0 / len(classes), 1e-6)
                        for c in classes]); pri /= pri.sum()
        di = np.where(classes == d[idx])[0]
        if di.size:
            kl.append(np.log(P[i, di[0]]) - np.log(pri[di[0]]))
    return float(np.mean(kl)) if kl else 0.0


def _mlp_kl(Z, y, d, n_cls, n_dom, ptr, pev, hd, epochs, seed, device):
    return float(fit_conditional_domain_probe(Z, y, d, n_cls, n_dom, train_idx=ptr, val_idx=pev,
                                              hidden_dim=hd, epochs=epochs, seed=seed, device=device)["kl_mean"])


def _excess(kl_fn, Z, y, d, ptr, pev, n_perm, seed):
    base = kl_fn(Z, y, d, ptr, pev)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):                                     # fully-retrained within-label permutation null
        dp = d.copy()
        for c in np.unique(y):
            m = np.where(y == c)[0]; dp[m] = d[m][rng.permutation(len(m))]
        null.append(kl_fn(Z, y, dp, ptr, pev))                  # permute D within Y, refit critic, recompute KL
    return base - (float(np.mean(null)) if null else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0"]); ap.add_argument("--n_subjects", type=int, default=2)
    ap.add_argument("--n_random", type=int, default=15); ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    caps = [("linear", None)] if a.smoke else [("linear", None), ("mlp_small", 32), ("mlp_large", 128)]
    if a.smoke:
        caps = [("linear", None), ("mlp_small", 32)]
    tag = "smoke" if a.smoke else "full"
    di_by_ds = defaultdict(list); rows = []
    for ds in DATASETS:
        cells = [p for p in sorted(glob.glob(str(REPO / "tos_cmi/results/tos_cmi_eeg_frozen" /
                 f"{ds}_{a.backbone}_LOSO" / "sub*_erm_lam0_seed*.npz")))
                 if any(p.endswith(f"_seed{s}.npz") for s in a.seeds)]
        if a.smoke:
            cells = cells[: a.n_subjects * len(a.seeds)]
        for cp in cells:
            f = feat_from_tos_dump(cp); sd = int(f["seed"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
            dsc = _dense(f["subj_source"]); yt = np.asarray(f["y_target"]).astype(int)
            Zt = np.asarray(f["Z_target"], float); st = f["session_target"]
            n_cls, n_dom = int(f["n_cls"]), int(len(np.unique(dsc)))
            er, pt, pe, diag = three_way_support_split(ys, dsc, seed=sd)
            if pt.size < 6 or pe.size < 6 or er.size < 6:
                continue
            B = get_candidate_basis("cond", False, Zs[er], ys[er], dsc[er], max_rank=10, seed=sd)  # eraser-fit basis
            if B.shape[0] == 0:
                continue
            cal, qry, _ = session_split(st, yt, sd); Xcal = Zt[cal]
            # SAME rule: build actions on eraser-fit basis, select via G1+safety+specificity on T_cal X
            ev = np.sort(np.linalg.eigvalsh(np.cov(Zs[er].T) + 1e-12 * np.eye(Zs.shape[1])))[::-1]; ev = np.clip(ev, 1e-12, None)
            ctx = dict(Zs=Zs[er], mu_s=Zs[er].mean(0), mu_tcal=Xcal.mean(0), Xcal=Xcal,
                       log_kappa_identity=float(np.log(ev[0] / ev[-1])))
            actions = build_actions(B, Zs[er], ys[er], dsc[er], Xcal, seed=sd, smoke=a.smoke, n_random_per_rank=8)
            sel, _diag = g1_select(actions, ctx, Zs[er], ys[er], dsc[er], sd)
            S_dirs = sel["dirs"]; k = sel["rank"] if sel["rank"] >= 1 else 0
            rule_matches = True                                  # firewall: selection used only source + T_cal X
            def _del(Z, dirs):
                return Z if (dirs is None or dirs.shape[0] == 0) else Z - (Z @ dirs.T) @ dirs
            rands = ambient_random_projectors(Zs.shape[1], max(k, 1), a.n_random, sd) if k >= 1 else []
            row = dict(dataset=ds, subject=str(f["heldout_subject"]), seed=sd, selected_action=sel["name"],
                       selected_rank=k, rule_matches_main_selector=rule_matches)
            for cname, hd in caps:
                kl_fn = ((lambda Z, y, d, ptr, pev: _linear_cond_kl(Z, y, d, n_dom, ptr, pev)) if hd is None
                         else (lambda Z, y, d, ptr, pev, _hd=hd: _mlp_kl(Z, y, d, n_cls, n_dom, ptr, pev, _hd, a.epochs, sd, a.device)))
                ex_sel = _excess(kl_fn, _del(Zs, S_dirs), ys, dsc, pt, pe, a.n_perm, sd) if k >= 1 else \
                    _excess(kl_fn, Zs, ys, dsc, pt, pe, a.n_perm, sd)
                ex_rnd = [ _excess(kl_fn, _del(Zs, Q), ys, dsc, pt, pe, a.n_perm, sd) for Q in rands ] if k >= 1 else []
                ex_full = _excess(kl_fn, Zs, ys, dsc, pt, pe, a.n_perm, sd)
                di_sel = ex_full - ex_sel                        # leakage removed by the selected rule
                di_rnd = ex_full - float(np.mean(ex_rnd)) if ex_rnd else 0.0
                row[f"di_specific_{cname}"] = float(di_sel - di_rnd)
            di_by_ds[ds].append(row.get("di_specific_linear", 0.0)); rows.append(row)
            print(f"  {ds} sub{f['heldout_subject']} s{sd}: sel={sel['name']}(r{k}) "
                  f"di_spec_linear={row.get('di_specific_linear', float('nan')):+.4f}", flush=True)
    # cluster LCB per dataset (subject-cluster bootstrap) of di_specific_linear (primary capacity for smoke)
    verdict = {}
    for ds, vals in di_by_ds.items():
        by = defaultdict(list)
        for r in rows:
            if r["dataset"] == ds and "di_specific_linear" in r:
                by[r["subject"]].append(r["di_specific_linear"])
        subj = [np.mean(v) for v in by.values()]
        rng = np.random.default_rng(7)
        boots = [np.mean(np.array(subj)[rng.integers(0, len(subj), len(subj))]) for _ in range(10000)] if subj else [0.0]
        verdict[ds] = float(np.percentile(boots, 2.5))
    json.dump(verdict, open(OUT / f"targetx_gate5_rule_{tag}.json", "w"), indent=2)
    with open(OUT / f"targetx_gate5_rule_rows_{tag}.jsonl", "w") as fh:
        [fh.write(json.dumps(r) + "\n") for r in rows]
    print(f"[gate5-rule] tag={tag} -> di_specific_linear LCB by dataset: {verdict}")


if __name__ == "__main__":
    main()

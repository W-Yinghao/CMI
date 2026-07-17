#!/usr/bin/env python
"""Gate 5 (rule-level cross-fitted certification) for the target-X selection RULE (amendments 03/04 C5).

Certifies the SAME deployable rule (M.rule_hash), not a re-measured projector. Per outer fold: source ->
eraser-fit / posterior-train / posterior-eval; rebuild the WHITENED task-CONTESTED cond basis on the eraser-fit
partition; run the SAME shared g1_select rule on T_cal X to pick the whitened action; apply it (and exact-rank
whitened ambient random) to the source posterior partitions; measure conditional subject leakage with a
FULL POSTERIOR-KL critic (linear + MLP) and a TRAINING-ONLY within-label permutation null (posterior-eval
support fixed); report dI_specific = dI_selected - mean dI_random. Records rule_hash (both audit & here) so the
verdict checks same_rule_implementation, NOT same_selected_action. --smoke = linear + few perms.
"""
from __future__ import annotations
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eval import targetx_metric as M
from tos_cmi.eval.targetx_observability import build_actions, g1_select, session_split
from cmi.eval.conditional_subject_leakage import three_way_support_split

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _posterior_kl(Zw, y, d, ptr, pev, critic="linear", seed=0):
    """FULL posterior-KL: E_{z,y}[ sum_d q(d|z,y) log(q(d|z,y)/p_hat(d|y)) ], critic fit on ptr, evaluated on
    pev; p_hat(d|y) from ptr within-label frequencies. Training-only fit (pev never trains)."""
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
    classes = np.unique(d)
    oh = np.eye(len(np.unique(y)))[np.searchsorted(np.unique(y), y)]
    X = np.hstack([Zw, oh])
    if len(np.unique(d[ptr])) < 2:
        return 0.0
    if critic == "linear":
        clf = LogisticRegression(max_iter=300, C=1.0).fit(X[ptr], d[ptr])
    else:
        clf = MLPClassifier(hidden_layer_sizes=(critic,), max_iter=300, random_state=seed).fit(X[ptr], d[ptr])
    Q = np.clip(clf.predict_proba(X[pev]), 1e-6, 1.0); Q /= Q.sum(1, keepdims=True)
    cls = clf.classes_
    # prior p(d|y) from ptr within-label
    kl = []
    for i, idx in enumerate(pev):
        yy = y[idx]; m = (y[ptr] == yy)
        if not m.any():
            continue
        pri = np.array([max((d[ptr][m] == c).mean(), 1e-6) for c in cls]); pri /= pri.sum()
        kl.append(float(np.sum(Q[i] * np.log(Q[i] / pri))))       # full-distribution KL, not just the true class
    return float(np.mean(kl)) if kl else 0.0


def _excess(Zw, y, d, ptr, pev, critic, n_perm, seed):
    base = _posterior_kl(Zw, y, d, ptr, pev, critic, seed)
    rng = np.random.default_rng(seed); null = []
    for _ in range(n_perm):
        dp = d.copy()
        for c in np.unique(y[ptr]):                               # permute D within Y on TRAINING trials ONLY
            m = ptr[y[ptr] == c]; dp[m] = d[m][rng.permutation(len(m))]
        null.append(_posterior_kl(Zw, y, dp, ptr, pev, critic, seed))   # eval support (pev) fixed
    return base - (float(np.mean(null)) if null else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0"]); ap.add_argument("--n_subjects", type=int, default=2)
    ap.add_argument("--n_random", type=int, default=15); ap.add_argument("--n_perm", type=int, default=20)
    a = ap.parse_args()
    critics = ["linear"] if a.smoke else ["linear", 32, 128]
    tag = "smoke" if a.smoke else "full"
    rows = []; audit_rule = M.rule_hash()
    for ds in DATASETS:
        cells = [p for p in sorted(glob.glob(str(REPO / "tos_cmi/results/tos_cmi_eeg_frozen" /
                 f"{ds}_{a.backbone}_LOSO" / "sub*_erm_lam0_seed*.npz")))
                 if any(p.endswith(f"_seed{s}.npz") for s in a.seeds)]
        if a.smoke:
            cells = cells[: a.n_subjects * len(a.seeds)]
        for cp in cells:
            f = feat_from_tos_dump(cp); sd = int(f["seed"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
            dsc = _dense(f["subj_source"]); yt = np.asarray(f["y_target"]).astype(int); Zt = np.asarray(f["Z_target"], float)
            er, pt, pe, diag = three_way_support_split(ys, dsc, seed=sd)
            if pt.size < 6 or pe.size < 6 or er.size < 6:
                continue
            # rebuild the WHITENED contested basis on the ERASER-FIT partition; select via the SAME rule on T_cal
            W = M.source_whitener(Zs[er]); Zs_er_w = M.to_whitened(Zs[er], W)
            row_w, null_w = M.whitened_head_rowspace(Zs_er_w, ys[er], sd)
            B_cond_w = M.whitened_cond_basis(Zs_er_w, ys[er], dsc[er], max_rank=10)
            B_contested = M.project_basis(B_cond_w, row_w)      # F2.1c: NO fallback to full cond
            if B_contested.shape[0] == 0:
                rows.append(dict(dataset=ds, subject=str(f["heldout_subject"]), seed=sd, selected_action="identity",
                                 selected_rank=0, rule_hash=M.rule_hash(), audit_rule_hash=audit_rule,
                                 same_rule_implementation=True, di_specific_linear=0.0, no_contested_candidate=True))
                continue
            cal, qry, _ = session_split(f["session_target"], yt, sd); Xcal = Zt[cal]
            d_white = -M.to_whitened(Xcal.mean(0)[None, :], W)[0]
            ctx = dict(d_white=d_white, Zs_w=Zs_er_w)
            actions = build_actions(B_contested, W, Zs[er], ys[er], dsc[er], Xcal, seed=sd, smoke=a.smoke, n_random_per_rank=8)
            sel, sdiag = g1_select(actions, ctx, Zs[er], ys[er], dsc[er], sd)
            U = sel["U"]; k = sel["rank"]
            # measure leakage on the WHITENED FULL source (whitener from eraser-fit), deleting the selected U
            Zs_w = M.to_whitened(Zs, W)
            def _del(Zw, Uu):
                return Zw if (Uu is None or Uu.shape[0] == 0) else Zw - (Zw @ Uu.T) @ Uu
            rands = M.ambient_random_projectors_whitened(Zs.shape[1], max(k, 1), a.n_random, sd) if k >= 1 else []
            row = dict(dataset=ds, subject=str(f["heldout_subject"]), seed=sd, selected_action=sel["name"],
                       selected_rank=k, rule_hash=sdiag["rule_hash"], audit_rule_hash=audit_rule,
                       same_rule_implementation=bool(sdiag["rule_hash"] == audit_rule))
            for cr in critics:
                cname = "linear" if cr == "linear" else f"mlp{cr}"
                ex_full = _excess(Zs_w, ys, dsc, pt, pe, cr, a.n_perm, sd)
                ex_sel = _excess(_del(Zs_w, U), ys, dsc, pt, pe, cr, a.n_perm, sd) if k >= 1 else ex_full
                ex_rnd = [_excess(_del(Zs_w, Q), ys, dsc, pt, pe, cr, a.n_perm, sd) for Q in rands] if k >= 1 else []
                di_sel = ex_full - ex_sel
                di_rnd = ex_full - float(np.mean(ex_rnd)) if ex_rnd else 0.0
                row[f"di_specific_{cname}"] = float(di_sel - di_rnd)
            rows.append(row)
            print(f"  {ds} sub{f['heldout_subject']} s{sd}: sel={sel['name']}(r{k}) same_rule={row['same_rule_implementation']} "
                  f"di_spec_linear={row.get('di_specific_linear', float('nan')):+.4f}", flush=True)
    verdict = {}
    for ds in {r["dataset"] for r in rows}:
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
    print(f"[gate5-rule] tag={tag} same_rule_all={all(r['same_rule_implementation'] for r in rows)} "
          f"-> di_specific_linear LCB: {verdict}")


if __name__ == "__main__":
    main()

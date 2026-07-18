"""Target Readout Calibration Ladder per-cell runner (CPU, env c84c). One frozen ERM EEGNet dump (target subject x
seed) -> for k in {1,2,4,8,16,32,Full} labeled cal trials/class (50 balanced draws for finite k) evaluate 4 heads
(H0 frozen / H1 fresh / H2 source-anchored MAP / H3 bias+temp) on 3 representations (Z0 native / ZI source-fitted
informed B_cond deletion / ZR >=50 source-retention-matched random deletions). Source-only alpha (H2) via outer-source
early->later pseudo-target. Firewall: target QUERY (X,Y) only in the final utility; cal Y only adapts heads; alpha
source-only. NO re-inference (frozen dumps by absolute path). Manuscript FROZEN; only the owner stops/redirects a line.

  python -m scripts.run_readout_label_efficiency --cell-index 0 --out-dir results/cmi_trace_readout
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump
from tos_cmi.eval import targetx_metric as TM
from tos_cmi.eval import readout_calibration as RC
from tos_cmi.eval.mechanism_subspace import _del, build_ambient_random_dictionaries, cell_seed
from tos_cmi.eval.targetx_observability import session_split

# frozen ERM EEGNet dumps live (untracked) across two worktrees; stable absolute paths
FEAT_ROOTS = {
    "BNCI2014_001": "/home/infres/yinwang/CMI_AAAI_cmitrace/tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_EEGNet_LOSO",
    "BNCI2015_001": "/home/infres/yinwang/CMI_AAAI_cmitrace/tos_cmi/results/tos_cmi_eeg_frozen/BNCI2015_001_EEGNet_LOSO",
    "Lee2019_MI":   "/home/infres/yinwang/CMI_AAAI_tos/tos_cmi/results/tos_cmi_eeg_frozen/Lee2019_MI_EEGNet_LOSO",
    "BNCI2014_004": "/home/infres/yinwang/CMI_AAAI_tos/tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_004_EEGNet_LOSO",
}
DATASETS = ["BNCI2014_001", "BNCI2015_001", "Lee2019_MI", "BNCI2014_004"]
BUDGETS = [1, 2, 4, 8, 16, 32, "Full"]
DICT_RANK = 8
SOURCE_RETENTION_TOL = 0.03      # a random deletion enters the matched set iff |src_bacc - informed src_bacc| <= tol


def enumerate_cells():
    cells = []
    for ds in DATASETS:
        for p in sorted(glob.glob(f"{FEAT_ROOTS[ds]}/sub*_erm_lam0_seed*.npz")):
            cells.append((ds, p))
    return cells


def _balanced_draw(ycal, k, rng):
    idx = []
    for c in np.unique(ycal):
        ci = np.where(ycal == c)[0]
        idx.extend(rng.choice(ci, min(k, len(ci)), replace=False).tolist())
    return np.array(sorted(idx), dtype=int)


def _draws(ds, subj, sd, ycal, k, nd):
    if k == "Full":
        return [np.arange(len(ycal))]
    return [_balanced_draw(ycal, k, np.random.default_rng(cell_seed(ds, "EEGNet", subj, sd, f"draw{k}", i))) for i in range(nd)]


def _rep_curve(Zs_wd, ys, Xcal_wd, ycal, Zq_wd, yq, sq, C, alpha, draws_by_k, heads):
    """Per-budget mean-over-draws utilities for ONE representation (source head prepared once)."""
    prep = RC.prepare_source_head(Zs_wd, ys, C)
    curve = {}
    for k in BUDGETS:
        per = [RC.adapt_and_score(prep, Xcal_wd, ycal, di, Zq_wd, yq, sq, C, alpha, heads) for di in draws_by_k[k]]
        curve[str(k)] = {h: float(np.mean([p[h] for p in per])) for h in heads}
    return curve, prep[4]      # curve, source-in-sample bAcc


def run_cell(ds, path, n_random=50, n_draws=50, smoke=False):
    f = feat_from_tos_dump(path)
    subj = str(f.get("heldout_subject")); sd = int(f.get("seed", -1))
    if "session_target" not in f:
        return dict(dataset=ds, subject=subj, seed=sd, status="skipped", reason="NO_SESSION_AXIS")
    Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
    dsub = np.asarray(f["subj_source"]).astype(int); Zt = np.asarray(f["Z_target"], float)
    yt = np.asarray(f["y_target"]).astype(int); C = int(f.get("n_cls", len(np.unique(ys))))
    sess_s = np.asarray(f["session_source"]).astype(str)
    W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W); D = Zs.shape[1]
    cal, qry, sinfo = session_split(f["session_target"], yt, sd)
    Xcal_w, ycal = TM.to_whitened(Zt[cal], W), yt[cal]
    Xq_w, yq, sq = TM.to_whitened(Zt[qry], W), yt[qry], np.asarray(f["session_target"])[qry]
    nd = 5 if smoke else n_draws
    draws_by_k = {k: _draws(ds, subj, sd, ycal, k, nd) for k in BUDGETS}
    # cal-class-shortfall reason codes (a class with < k cal trials)
    cal_per_class = {int(c): int((ycal == c).sum()) for c in np.unique(ycal)}
    short = {k: [c for c, n in cal_per_class.items() if isinstance(k, int) and n < k] for k in BUDGETS}
    # source-only alpha via outer-source early->later pseudo-target (on native whitened source)
    alpha, ainfo = RC.select_alpha_pseudo_target(RC._std(Zs_w, *RC.standardize(Zs_w)), ys, dsub, sess_s, C)
    # informed B_cond deletion (rank<=8, whole informed subspace, NO target selection)
    B = TM.whitened_cond_basis(Zs_w, ys, dsub, max_rank=DICT_RANK); r = B.shape[0]
    if r == 0:
        return dict(dataset=ds, subject=subj, seed=sd, status="skipped", reason="EMPTY_B_COND")
    # Z0 native + ZI informed (all 4 heads)
    HEADS = ("frozen", "fresh", "map", "bias")
    z0_curve, z0_src = _rep_curve(Zs_w, ys, Xcal_w, ycal, Xq_w, yq, sq, C, alpha, draws_by_k, HEADS)
    zi_curve, zi_src = _rep_curve(_del(Zs_w, B), ys, _del(Xcal_w, B), ycal, _del(Xq_w, B), yq, sq, C, alpha, draws_by_k, HEADS)
    # ZR matched random deletions (source-retention filtered); only need frozen+map for G_h
    nr = 8 if smoke else n_random
    rand = build_ambient_random_dictionaries(D, r, nr, cell_seed(ds, "EEGNet", subj, sd, "ambient"))
    zr_curves, zr_srcs, matched = [], [], 0
    for Q in rand:
        c, sbacc = _rep_curve(_del(Zs_w, Q), ys, _del(Xcal_w, Q), ycal, _del(Xq_w, Q), yq, sq, C, alpha, draws_by_k, ("frozen", "map"))
        zr_srcs.append(sbacc)
        if abs(sbacc - zi_src) <= SOURCE_RETENTION_TOL:
            zr_curves.append(c); matched += 1
    # per-budget endpoints
    ep = {}
    for k in BUDGETS:
        kk = str(k)
        gh_native = z0_curve[kk]["map"] - z0_curve[kk]["frozen"]
        gh_informed = zi_curve[kk]["map"] - zi_curve[kk]["frozen"]
        gh_random = float(np.mean([c[kk]["map"] - c[kk]["frozen"] for c in zr_curves])) if zr_curves else float("nan")
        ep[kk] = dict(dU_MAP_frozen=z0_curve[kk]["map"] - z0_curve[kk]["frozen"],
                      dU_MAP_fresh=z0_curve[kk]["map"] - z0_curve[kk]["fresh"],
                      dU_MAP_bias=z0_curve[kk]["map"] - z0_curve[kk]["bias"],
                      U_frozen=z0_curve[kk]["frozen"], U_fresh=z0_curve[kk]["fresh"], U_MAP=z0_curve[kk]["map"], U_bias=z0_curve[kk]["bias"],
                      Gh_native=gh_native, Gh_informed=gh_informed, Gh_random_mean=gh_random,
                      dGh_specific=(gh_informed - gh_random) if zr_curves else float("nan"))
    return dict(dataset=ds, subject=subj, seed=sd, status="ok", C=C, rank=int(r), n_cal=int(cal.sum()), n_query=int(qry.sum()),
                cal_session=sinfo["cal_sessions"], query_sessions=sinfo["query_sessions"], alpha=float(alpha), alpha_info=ainfo,
                n_random=nr, n_matched_random=matched, informed_src_bacc=float(zi_src), native_src_bacc=float(z0_src),
                n_draws=nd, cal_per_class=cal_per_class, budget_class_shortfall={k: v for k, v in short.items() if v},
                endpoints=ep,
                firewall=dict(source_only_construction=True, alpha_source_only=True, Ycal_used_for_head_adapt=True,
                              Yquery_used_for_selection=False, Xquery_used_for_selection=False, Yquery_used_for_outcome=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-index", type=int); ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--out-dir", default="results/cmi_trace_readout")
    ap.add_argument("--n-random", type=int, default=50); ap.add_argument("--n-draws", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cells = enumerate_cells()
    if a.list_cells:
        [print(f"{i}\t{ds}\t{Path(p).name}") for i, (ds, p) in enumerate(cells)]; print(f"# {len(cells)}"); return
    ds, path = cells[a.cell_index]
    print(f"[readout] cell {a.cell_index} {ds} {Path(path).name}", flush=True)
    row = run_cell(ds, path, n_random=a.n_random, n_draws=a.n_draws, smoke=a.smoke)
    outd = Path(a.out_dir); (outd / "cells").mkdir(parents=True, exist_ok=True)
    stem = f"cell_{a.cell_index:03d}_{ds}_sub{row.get('subject')}_seed{row.get('seed')}"
    (outd / "cells" / f"{stem}.json").write_text(json.dumps(row, indent=2, default=float))
    (outd / "cells" / f"{stem}.done").write_text(row.get("status", "?") + "\n")
    if row.get("status") == "ok":
        e = row["endpoints"]
        print(f"  alpha={row['alpha']} matched_rand={row['n_matched_random']}/{row['n_random']} rank={row['rank']}", flush=True)
        print("  dU_MAP-frozen: " + " ".join(f"k{k}={e[str(k)]['dU_MAP_frozen']:+.3f}" for k in BUDGETS), flush=True)
        print("  dU_MAP-fresh : " + " ".join(f"k{k}={e[str(k)]['dU_MAP_fresh']:+.3f}" for k in BUDGETS), flush=True)
        print("  dGh_specific : " + " ".join(f"k{k}={e[str(k)]['dGh_specific']:+.3f}" for k in BUDGETS), flush=True)
    else:
        print(f"  status={row.get('status')} reason={row.get('reason')}", flush=True)


if __name__ == "__main__":
    main()

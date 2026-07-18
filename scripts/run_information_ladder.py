"""Information-Regime Ladder per-cell runner (Track B; CPU, env c84c). One frozen ERM EEGNet dump (target subject x
seed) -> selection-only ladder over R0/RX/R1/R2/R4/RF for the informed B_cond dictionary AND matched-rank random
dictionaries, + the head-only calibration secondary + the crossfit target-oracle ceiling. Writes one cell json + .done.
NO re-inference (pure frozen features + sklearn). Firewall: query (X,Y) enters ONLY utility/oracle; cal Y only
label-regime selection. Manuscript FROZEN; only the project owner stops/redirects a scientific line.

  python -m scripts.run_information_ladder --cell-index 0 --out-dir results/cmi_trace_info_ladder
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump
from tos_cmi.eval import targetx_metric as TM
from tos_cmi.eval import information_ladder as IL
from tos_cmi.eval.mechanism_subspace import build_ambient_random_dictionaries, cell_seed
from tos_cmi.eval.targetx_observability import session_split
from tos_cmi.eval.dg_identifiability import crossfit_target_oracle

# frozen ERM EEGNet dumps live (untracked) in the MAIN worktree; stable across branch switches
FEAT_ROOT = Path("/home/infres/yinwang/CMI_AAAI_cmitrace/tos_cmi/results/tos_cmi_eeg_frozen")
DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def enumerate_cells(feat_root=FEAT_ROOT):
    cells = []
    for ds in DATASETS:
        for p in sorted((feat_root / f"{ds}_EEGNet_LOSO").glob("sub*_erm_lam0_seed*.npz")):
            cells.append((ds, str(p)))
    return cells


def run_cell(ds, path, n_random=10, n_draws=20, seed=0, smoke=False):
    f = feat_from_tos_dump(path)
    subj = str(f.get("heldout_subject")); sd = int(f.get("seed", -1))
    if "session_target" not in f:
        return dict(dataset=ds, subject=subj, seed=sd, status="skipped", reason="NO_SESSION_AXIS")
    Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
    dsc = np.asarray(f["subj_source"]).astype(int); Zt = np.asarray(f["Z_target"], float)
    yt = np.asarray(f["y_target"]).astype(int); classes = sorted(np.unique(ys).tolist())
    W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W); D = Zs.shape[1]
    cal, qry, sinfo = session_split(f["session_target"], yt, sd)
    Xcal_w, ycal = TM.to_whitened(Zt[cal], W), yt[cal]
    Xq_w, yq, sq = TM.to_whitened(Zt[qry], W), yt[qry], np.asarray(f["session_target"])[qry]
    d_white = Zs_w.mean(0) - Xcal_w.mean(0)                     # A_s(mu_s - mu_tcal) in whitened coords (unlabeled)
    B = TM.whitened_cond_basis(Zs_w, ys, dsc, max_rank=IL.DICT_RANK)
    r = B.shape[0]
    if r == 0:
        return dict(dataset=ds, subject=subj, seed=sd, status="skipped", reason="EMPTY_B_COND")
    # deterministic k-shot draws (shared by informed + every random dict; balanced; never chosen by query)
    nd = 3 if smoke else n_draws
    # deterministic k-specific class-balanced draws (shared by informed + every random dict; never chosen by query)
    draws_by_k = {k: [IL._balanced_draw(ycal, k, np.random.default_rng(cell_seed(ds, "EEGNet", subj, sd, f"draw{k}", i)))
                      for i in range(nd)] for k in IL.KSHOT.values()}

    def ladder(Bd):
        rp, _ = IL.precompute_actions(Zs_w, ys, dsc, Bd, Xcal_w, ycal, Xq_w, yq, sq, d_white)
        o = {}
        for reg in IL.REGIMES:
            if reg in IL.KSHOT:
                dU, _ = IL.select_and_utility(rp, reg, Xcal_w, ycal, classes, draws=draws_by_k[IL.KSHOT[reg]])
            else:
                dU, _ = IL.select_and_utility(rp, reg, Xcal_w, ycal, classes)
            o[reg] = float(dU)
        ces = [IL._ce(rr["head"], rr["U"], Xcal_w, ycal, classes) for rr in rp]
        o["_rf_U"] = rp[int(np.argmin(ces))]["U"]
        return o

    informed = ladder(B)
    nr = 3 if smoke else n_random
    rand_dicts = build_ambient_random_dictionaries(D, r, nr, cell_seed(ds, "EEGNet", subj, sd, "ambient"))
    rand_ladders = [ladder(Q) for Q in rand_dicts]
    rand_mean = {reg: float(np.mean([rl[reg] for rl in rand_ladders])) for reg in IL.REGIMES}

    # head-only calibration secondary (native vs selected-subspace) using the RF-selected informed action
    U_star = informed["_rf_U"]
    head_only = {}
    for reg, k in [("R1", 1), ("R2", 2), ("R4", 4), ("RF", None)]:
        di = draws_by_k[k][0] if k is not None else None
        head_only[reg] = IL.head_only_calibration(U_star, Zs_w, ys, Xcal_w, ycal, Xq_w, yq, sq, di)

    # crossfit target-oracle ceiling (hindsight; selects on disjoint query labels) on the informed dictionary
    try:
        orc = crossfit_target_oracle(Zs_w, ys, Xq_w, yq, B, seed=seed, mode="greedy")
        oracle = dict(delta=float(orc["delta_query"]), random=float(orc["delta_query_random"]), rank=int(orc["rank"]))
    except Exception as e:
        oracle = dict(error=str(e)[:80])

    return dict(dataset=ds, subject=subj, seed=sd, status="ok", rank=int(r), n_cal=int(cal.sum()), n_query=int(qry.sum()),
                cal_session=sinfo["cal_sessions"], query_sessions=sinfo["query_sessions"], n_random=nr, n_draws=nd,
                informed={reg: informed[reg] for reg in IL.REGIMES},
                random_mean={reg: rand_mean[reg] for reg in IL.REGIMES},
                specific={reg: float(informed[reg] - rand_mean[reg]) for reg in IL.REGIMES},
                head_only=head_only, oracle=oracle,
                firewall=dict(source_only_construction=True, Ycal_used_for_selection=True,
                              Yquery_used_for_selection=False, Xquery_used_for_selection=False, Yquery_used_for_outcome=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-index", type=int); ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--out-dir", default="results/cmi_trace_info_ladder")
    ap.add_argument("--n-random", type=int, default=10); ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cells = enumerate_cells()
    if a.list_cells:
        [print(f"{i}\t{ds}\t{Path(p).name}") for i, (ds, p) in enumerate(cells)]; print(f"# {len(cells)}"); return
    ds, path = cells[a.cell_index]
    print(f"[info-ladder] cell {a.cell_index} {ds} {Path(path).name}", flush=True)
    row = run_cell(ds, path, n_random=a.n_random, n_draws=a.n_draws, smoke=a.smoke)
    outd = Path(a.out_dir); (outd / "cells").mkdir(parents=True, exist_ok=True)
    stem = f"cell_{a.cell_index:03d}_{ds}_sub{row.get('subject')}_seed{row.get('seed')}"
    (outd / "cells" / f"{stem}.json").write_text(json.dumps(row, indent=2, default=float))
    (outd / "cells" / f"{stem}.done").write_text(row.get("status", "?") + "\n")
    if row.get("status") == "ok":
        inf, spc = row["informed"], row["specific"]
        print("  informed dU: " + " ".join(f"{reg}={inf[reg]:+.3f}" for reg in IL.REGIMES), flush=True)
        print("  specific   : " + " ".join(f"{reg}={spc[reg]:+.3f}" for reg in IL.REGIMES), flush=True)
    else:
        print(f"  status={row.get('status')} reason={row.get('reason')}", flush=True)


if __name__ == "__main__":
    main()

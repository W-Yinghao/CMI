"""Fork 2 Phase 1A --- source-rich semi-synthetic smoke driver. For each fold x alpha x ENVIRONMENT-definition
x intervention: inject source-rich World A, run the SAME frozen gate but with a leave-one-ENVIRONMENT-out
benefit signal (E0 subject / E_oracle regime / E2 covariance / E4 margin / E5 augmentation / random), and audit
the actual target. Gate thresholds FROZEN; environment selection uses source-only labels; target audit-only.

  python -m tos_cmi.eeg.run_source_rich_smoke --datasets Lee2019_MI --backbones EEGNet --seeds 0 --folds 5 \
      --alphas 0.5 1.0 2.0 --tag smoke
Writes tos_cmi/results/source_rich/smoke/source_rich_smoke_{rows.csv, summary.json}. Report via
report_source_rich_smoke.py.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import numpy as np
from joblib import Parallel, delayed

from tos_cmi.eeg.erasure_baselines import _ids
from tos_cmi.eeg.source_ood_benefit_gate import _bacc, _subj_acc, _boot_bound, gate_action, SAFETY_EPS, BENEFIT_LCB
from tos_cmi.eeg.run_v2_certificate import _task_scores, _stratified_split, _nuisance_m
from tos_cmi.eeg.v2_worlds import FACTORIES, oracle_nuisance_eraser_factory
from tos_cmi.eeg.source_rich_worlds import (inject_source_rich, smoke_environments, DEPLOYABLE_ENVS,
                                            DIAGNOSTIC_ENVS)

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
OUT = "tos_cmi/results/source_rich/smoke"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
ENVS = ["subject", "oracle", "covariance_cluster", "margin_cluster", "augmentation_shift", "random"]
INTERV = {"identity": FACTORIES["identity"], "leace": FACTORIES["leace_baseline"], "rlace": FACTORIES["rlace"],
          "tos_vd": FACTORIES["tos_vd"], "tp_leace": FACTORIES["tp_leace"],
          "alpha_leace": FACTORIES["alpha_leace"], "random_k": FACTORIES["random_k"]}
DIAG_INTERV = "oracle_nuisance_DIAGNOSTIC"


def eval_source_rich(Zs, ys, z_src, env, Zt, yt, n_cls, F, seed):
    """Safety + leave-one-ENVIRONMENT-out benefit (over `env`) + target audit for one eraser F (erases D=z_src)."""
    A, Bm = _stratified_split(len(ys), seed)
    E = F(Zs[A], ys[A], z_src[A], n_cls, seed)
    src_full = _bacc(Zs[A], ys[A], Zs[Bm], ys[Bm]); src_eras = _bacc(E(Zs[A]), ys[A], E(Zs[Bm]), ys[Bm])
    z01, _ = _ids(z_src)
    dg = _subj_acc(Zs[A], z01[A], Zs[Bm], z01[Bm]) - _subj_acc(E(Zs[A]), z01[A], E(Zs[Bm]), z01[Bm])
    benefit = []
    for e in sorted(set(env.tolist())):
        tr = env != e; te = env == e
        if len(np.unique(ys[te])) < 2 or len(np.unique(ys[tr])) < 2:
            continue
        Es = F(Zs[tr], ys[tr], z_src[tr], n_cls, seed)
        benefit.append(_bacc(Es(Zs[tr]), ys[tr], Es(Zs[te]), ys[te]) - _bacc(Zs[tr], ys[tr], Zs[te], ys[te]))
    Ef = F(Zs, ys, z_src, n_cls, seed)
    tb_e, _ = _task_scores(Ef(Zs), ys, Ef(Zt), yt, n_cls); tb_f, _ = _task_scores(Zs, ys, Zt, yt, n_cls)
    return dict(task_drop=src_full - src_eras, domain_gain=dg, benefit=benefit,
                n_env=len(set(env.tolist())), tgt_bacc_full=tb_f, tgt_bacc_eras=tb_e)


def _one(ds, bb, seed, p, alpha, env_name, interv, frac, noise):
    fold = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
    base = {"dataset": ds, "backbone": bb, "seed": seed, "fold": fold, "alpha": alpha,
            "env": env_name, "intervention": interv}
    try:
        d = np.load(p, allow_pickle=True)
        Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
        Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
        subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
        m = _nuisance_m(Zs.shape[1], "fraction_of_z_dim", 0.20, 4)
        inj = inject_source_rich(Zs, ys, subj, Zt, yt, alpha=alpha, m=m, noise=noise, seed=seed, frac=frac)
        env, reason = smoke_environments(env_name, inj["Zs2"], ys, subj, inj["regime_src"], k=8, seed=seed)
        if env is None:
            return {**base, "skip": reason}
        F = oracle_nuisance_eraser_factory(m) if interv == DIAG_INTERV else INTERV[interv]
        sig = eval_source_rich(inj["Zs2"], ys, inj["z_src"], env, inj["Zt2"], yt, n_cls, F, seed)
        return {**base, "n_cls": n_cls, **sig}
    except Exception as e:
        return {**base, "fail": repr(e)[:200]}


def _dumps(ds, bb, seed, nfolds):
    dd = "%s/%s_%s_LOSO" % (RESULTS, ds, bb)
    ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    return ps[:nfolds] if nfolds else ps


def aggregate(rows, safety_eps, benefit_thr):
    rows = [r for r in rows if not r.get("fail") and not r.get("skip")]
    summary = {}
    keys = sorted(set((r["dataset"], r["backbone"], r["seed"], r["alpha"], r["env"], r["intervention"]) for r in rows))
    for (ds, bb, sd, al, env, iv) in keys:
        sub = [r for r in rows if (r["dataset"], r["backbone"], r["seed"], r["alpha"], r["env"], r["intervention"]) == (ds, bb, sd, al, env, iv)]
        folds = [r["fold"] for r in sub]; tds = [r["task_drop"] for r in sub]
        bvals, bfolds = [], []
        for r in sub:
            for v in r.get("benefit", []):
                bvals.append(v); bfolds.append(r["fold"])
        tucb = _boot_bound(tds, folds, "upper", rng=np.random.default_rng(0))
        blcb = _boot_bound(bvals, bfolds, "lower", rng=np.random.default_rng(0)) if bvals else float("nan")
        db = [r["tgt_bacc_eras"] - r["tgt_bacc_full"] for r in sub]
        dblo = _boot_bound(db, folds, "lower", rng=np.random.default_rng(1))
        dbhi = _boot_bound(db, folds, "upper", rng=np.random.default_rng(2))
        deployable_env = env in DEPLOYABLE_ENVS
        action = gate_action(tucb, blcb, safety_eps, benefit_thr) if iv != DIAG_INTERV else "DIAGNOSTIC"
        summary["|".join(map(str, (ds, bb, sd, al, env, iv)))] = {
            "dataset": ds, "backbone": bb, "seed": sd, "alpha": al, "env": env, "intervention": iv,
            "deployable_env": deployable_env, "n_folds": len(set(folds)), "n_env_mean": float(np.mean([r["n_env"] for r in sub])),
            "task_drop_ucb": tucb, "benefit_lcb": blcb, "benefit_mean": float(np.mean(bvals)) if bvals else float("nan"),
            "domain_gain": float(np.nanmean([r["domain_gain"] for r in sub])),
            "dtgt_bacc": float(np.mean(db)), "dtgt_bacc_lo": dblo, "dtgt_bacc_hi": dbhi,
            "gate_action": action, "is_safe": bool(tucb <= safety_eps), "target_beneficial": bool(dblo > benefit_thr)}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Lee2019_MI"])
    ap.add_argument("--backbones", nargs="+", default=["EEGNet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--frac", nargs=3, type=float, default=[0.4, 0.3, 0.3])   # tuned (world-gen) for a SAFE source-visible positive
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    tasks = []
    for ds in a.datasets:
        for bb in a.backbones:
            for sd in a.seeds:
                for p in _dumps(ds, bb, sd, a.folds):
                    for al in a.alphas:
                        for env in ENVS:
                            for iv in list(INTERV) + [DIAG_INTERV]:
                                tasks.append((ds, bb, sd, p, al, env, iv))
    print("source-rich smoke %s: %d tasks (n_jobs=%d), frac=%s, safety<=%.3f benefit>%.3f (FROZEN), envs=%s"
          % (a.tag, len(tasks), N_JOBS, a.frac, SAFETY_EPS, BENEFIT_LCB, ENVS), flush=True)
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_one)(ds, bb, sd, p, al, env, iv, tuple(a.frac), a.noise) for (ds, bb, sd, p, al, env, iv) in tasks)
    nfail = sum(1 for r in rows if r.get("fail")); nskip = sum(1 for r in rows if r.get("skip"))
    if nfail or nskip:
        print("[%d FAILED, %d SKIPPED]" % (nfail, nskip), flush=True)
        for r in [r for r in rows if r.get("fail")][:10]:
            print("  [FAIL] %s a%s %s %s: %s" % (r["env"], r["alpha"], r["intervention"], r["fold"], r["fail"]), flush=True)
    summary = aggregate(rows, SAFETY_EPS, BENEFIT_LCB)
    os.makedirs(OUT, exist_ok=True)
    cols = ["dataset", "backbone", "seed", "fold", "alpha", "env", "intervention", "task_drop", "domain_gain",
            "n_env", "tgt_bacc_full", "tgt_bacc_eras", "skip", "fail"]
    with open("%s/source_rich_%s_rows.csv" % (OUT, a.tag), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump({"thresholds": {"safety_eps": SAFETY_EPS, "benefit_lcb": BENEFIT_LCB},
               "params": {"frac": a.frac, "noise": a.noise}, "n_tasks": len(rows), "n_fail": nfail,
               "n_skip": nskip, "envs": ENVS, "summary": summary},
              open("%s/source_rich_%s_summary.json" % (OUT, a.tag), "w"), indent=1)
    # quick per-environment verdict (oracle vs subject vs discovered vs random)
    print("\n=== source-rich smoke: gate by environment (best deployable intervention per env) ===")
    for env in ENVS:
        cells = [v for v in summary.values() if v["env"] == env]
        acc = [v for v in cells if v["gate_action"] == "ACCEPT"]
        good = [v for v in acc if v["dtgt_bacc_lo"] > BENEFIT_LCB]
        best = max((v for v in cells if v["intervention"] != DIAG_INTERV), key=lambda v: v["benefit_lcb"], default=None)
        print("  %-18s : ACCEPT %d (target-good %d) ; best benefit LCB %s (%s) ; best target [%s]"
              % (env, len(acc), len(good),
                 ("%+.3f" % best["benefit_lcb"]) if best else "n/a", best["intervention"] if best else "-",
                 ("%+.3f..%+.3f" % (best["dtgt_bacc_lo"], best["dtgt_bacc_hi"])) if best else "-"), flush=True)
    print("SOURCE_RICH_%s_DONE" % a.tag.upper())


if __name__ == "__main__":
    main()

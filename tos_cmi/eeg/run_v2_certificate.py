"""V2 semi-synthetic acceptance-power benchmark driver. For each world x dataset x backbone x seed x
n_source x alpha x intervention x fold: inject the ground-truth-controlled nuisance into REAL EEG latents,
run the SOURCE-ONLY gate (erasing D = injected z; LOSO grouped by REAL subject), and audit the actual target.
Target labels enter ONLY the post-hoc scoring. Gate thresholds are FROZEN (loaded from the config).

  python -m tos_cmi.eeg.run_v2_certificate --worlds A B C --datasets Lee2019_MI Cho2017 \
      --backbones EEGNet --seeds 0 --n-source all --alphas 0.25 0.5 1.0 2.0 --folds 5 --tag smoke
(--folds N = first N folds; --n-source int|all)
Writes tos_cmi/results/method_deepen/v2/{v2_<tag>_rows.csv, v2_<tag>_summary.json, v2_design_hash.txt}.
"""
from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import numpy as np
from joblib import Parallel, delayed

from tos_cmi.eeg.erasure_baselines import _ids
from tos_cmi.eeg.source_ood_benefit_gate import (_bacc, _subj_acc, _boot_bound, gate_action,
                                                 SAFETY_EPS, BENEFIT_LCB)
from tos_cmi.eeg.run_phase2_task_preserving import _task_scores, _stratified_split, _load_thresholds
from tos_cmi.eeg.semi_synthetic_real_latent import inject
from tos_cmi.eeg.v2_worlds import (WORLDS, FACTORIES, INTERVENTIONS, DEPLOYABLE, DIAGNOSTIC, PRINCIPLED,
                                   CONTROLS, oracle_nuisance_eraser_factory)

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
OUT = "tos_cmi/results/method_deepen/v2"
CONFIG = "tos_cmi/eeg/configs/v2_certificate_fixed.yaml"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))


def eval_v2(Zs, ys, z_dom, grp, Zt, yt, n_cls, F, seed, n_pseudo):
    """Source-only gate signals (erase D=z_dom; LOSO by grp=real subject) + post-hoc target audit."""
    A, Bm = _stratified_split(len(ys), seed)
    E = F(Zs[A], ys[A], z_dom[A], n_cls, seed)
    src_full = _bacc(Zs[A], ys[A], Zs[Bm], ys[Bm])
    src_eras = _bacc(E(Zs[A]), ys[A], E(Zs[Bm]), ys[Bm])
    z01, _ = _ids(z_dom)
    zf = _subj_acc(Zs[A], z01[A], Zs[Bm], z01[Bm])
    ze = _subj_acc(E(Zs[A]), z01[A], E(Zs[Bm]), z01[Bm])
    subs = sorted(set(grp.tolist()))
    pick = list(np.random.default_rng(seed).permutation(subs)[:min(n_pseudo, len(subs))])
    benefit = []
    for s in pick:
        tr = grp != s; te = grp == s
        if len(np.unique(ys[te])) < 2 or len(np.unique(ys[tr])) < 2:
            continue
        Es = F(Zs[tr], ys[tr], z_dom[tr], n_cls, seed)
        benefit.append(_bacc(Es(Zs[tr]), ys[tr], Es(Zs[te]), ys[te]) - _bacc(Zs[tr], ys[tr], Zs[te], ys[te]))
    Ef = F(Zs, ys, z_dom, n_cls, seed)
    tb_e, tn_e = _task_scores(Ef(Zs), ys, Ef(Zt), yt, n_cls)
    tb_f, tn_f = _task_scores(Zs, ys, Zt, yt, n_cls)
    return dict(src_task_full=src_full, src_task_eras=src_eras, task_drop=src_full - src_eras,
                z_full=zf, z_eras=ze, domain_gain=(zf - ze) if zf == zf and ze == ze else float("nan"),
                benefit=benefit, tgt_bacc_full=tb_f, tgt_bacc_eras=tb_e,
                tgt_nll_full=tn_f, tgt_nll_eras=tn_e,
                router_acc=float(getattr(Ef, "router_acc", float("nan"))))


def _subsample_source(subj, n_source, seed):
    subs = sorted(set(subj.tolist()))
    if n_source == "all" or int(n_source) >= len(subs):
        keep = set(subs)
    else:
        keep = set(list(np.random.default_rng(100 + seed).permutation(subs)[:int(n_source)]))
    return np.array([s in keep for s in subj])


def _one(world, ds, bb, seed, p, n_source, alpha, interv, phi, beta, m, noise, n_pseudo):
    fold = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
    try:
        d = np.load(p, allow_pickle=True)
        Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
        Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
        subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
        keep = _subsample_source(subj, n_source, seed)
        Zs, ys, subj = Zs[keep], ys[keep], subj[keep]
        inj = inject(world, Zs, ys, subj, Zt, yt, alpha=alpha, beta=beta, phi=phi, seed=seed, m=m, noise=noise)
        F = oracle_nuisance_eraser_factory(m) if interv in DIAGNOSTIC else FACTORIES[interv]
        sig = eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], yt, n_cls,
                      F, seed, n_pseudo)
        return {"world": world, "dataset": ds, "backbone": bb, "seed": seed, "fold": fold,
                "n_source": str(n_source), "alpha": alpha, "intervention": interv,
                "ground_truth": inj["ground_truth"], "n_cls": n_cls, **sig}
    except Exception as e:
        return {"world": world, "dataset": ds, "backbone": bb, "seed": seed, "fold": fold,
                "n_source": str(n_source), "alpha": alpha, "intervention": interv, "fail": repr(e)[:200]}


def _dumps(ds, bb, seed, nfolds):
    dd = "%s/%s_%s_LOSO" % (RESULTS, ds, bb)
    ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    return ps[:nfolds] if nfolds else ps


def aggregate(rows, safety_eps, benefit_thr):
    rows = [r for r in rows if not r.get("fail")]
    summary = {}
    keys = sorted(set((r["world"], r["dataset"], r["backbone"], r["seed"], r["n_source"],
                       r["alpha"], r["intervention"]) for r in rows))
    for (w, ds, bb, sd, ns, al, iv) in keys:
        sub = [r for r in rows if (r["world"], r["dataset"], r["backbone"], r["seed"], r["n_source"],
                                   r["alpha"], r["intervention"]) == (w, ds, bb, sd, ns, al, iv)]
        folds = [r["fold"] for r in sub]
        tds = [r["task_drop"] for r in sub]
        bvals, bfolds = [], []
        for r in sub:
            for v in r.get("benefit", []):
                bvals.append(v); bfolds.append(r["fold"])
        tucb = _boot_bound(tds, folds, "upper", rng=np.random.default_rng(0))
        blcb = _boot_bound(bvals, bfolds, "lower", rng=np.random.default_rng(0)) if bvals else float("nan")
        db = [r["tgt_bacc_eras"] - r["tgt_bacc_full"] for r in sub]
        dblo = _boot_bound(db, folds, "lower", rng=np.random.default_rng(1))
        dbhi = _boot_bound(db, folds, "upper", rng=np.random.default_rng(2))
        deployable = iv in DEPLOYABLE
        action = gate_action(tucb, blcb, safety_eps, benefit_thr) if deployable else "DIAGNOSTIC"
        summary["|".join(map(str, (w, ds, bb, sd, ns, al, iv)))] = {
            "world": w, "dataset": ds, "backbone": bb, "seed": sd, "n_source": ns, "alpha": al,
            "intervention": iv, "deployable": deployable, "ground_truth": sub[0]["ground_truth"],
            "n_folds": len(set(folds)), "task_drop_ucb": tucb, "benefit_lcb": blcb,
            "domain_gain": float(np.nanmean([r["domain_gain"] for r in sub])),
            "z_full": float(np.nanmean([r["z_full"] for r in sub])),
            "z_eras": float(np.nanmean([r["z_eras"] for r in sub])),
            "src_task_full": float(np.mean([r["src_task_full"] for r in sub])),
            "src_task_eras": float(np.mean([r["src_task_eras"] for r in sub])),
            "dtgt_bacc": float(np.mean(db)), "dtgt_bacc_lo": dblo, "dtgt_bacc_hi": dbhi,
            "router_acc": float(np.nanmean([r.get("router_acc", float("nan")) for r in sub])),
            "gate_action": action,
            "is_safe": bool(tucb <= safety_eps),                       # source-defined safety
            "target_beneficial": bool(dblo > benefit_thr)}            # actual (post-hoc) target gain
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--datasets", nargs="+", default=["Lee2019_MI", "Cho2017"])
    ap.add_argument("--backbones", nargs="+", default=["EEGNet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-source", nargs="+", default=["all"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--interventions", nargs="+", default=INTERVENTIONS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-pseudo", type=int, default=8)
    ap.add_argument("--phi", type=float, default=0.15)   # World A f_align (fraction carrying the shortcut)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    safety_eps, benefit_thr = _load_thresholds(CONFIG)
    cfg_hash = hashlib.sha256(open(CONFIG).read().encode()).hexdigest()[:12] if os.path.exists(CONFIG) else "MISSING"
    tasks = []
    for w in a.worlds:
        for ds in a.datasets:
            for bb in a.backbones:
                for sd in a.seeds:
                    for p in _dumps(ds, bb, sd, a.folds):
                        for ns in a.n_source:
                            for al in a.alphas:
                                for iv in a.interventions:
                                    tasks.append((w, ds, bb, sd, p, ns, al, iv))
    print("V2 %s: %d tasks (n_jobs=%d), config %s, safety<=%.3f benefit>%.3f, phi=%.2f beta=%.2f m=%d"
          % (a.tag, len(tasks), N_JOBS, cfg_hash, safety_eps, benefit_thr, a.phi, a.beta, a.m), flush=True)
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_one)(w, ds, bb, sd, p, ns, al, iv, a.phi, a.beta, a.m, a.noise, a.n_pseudo)
        for (w, ds, bb, sd, p, ns, al, iv) in tasks)
    nfail = sum(1 for r in rows if r.get("fail"))
    if nfail:
        print("[%d FAILED]" % nfail, flush=True)
        for r in rows[:20]:
            if r.get("fail"):
                print("  [FAIL] %s %s %s %s a%s %s: %s" % (r["world"], r["dataset"], r["backbone"],
                      r["n_source"], r["alpha"], r["intervention"], r["fail"]), flush=True)
    summary = aggregate(rows, safety_eps, benefit_thr)
    os.makedirs(OUT, exist_ok=True)
    open("%s/v2_design_hash.txt" % OUT, "w").write("%s  %s\n" % (cfg_hash, CONFIG))
    cols = ["world", "dataset", "backbone", "seed", "fold", "n_source", "alpha", "intervention",
            "ground_truth", "src_task_full", "src_task_eras", "task_drop", "z_full", "z_eras",
            "domain_gain", "tgt_bacc_full", "tgt_bacc_eras", "router_acc"]
    with open("%s/v2_%s_rows.csv" % (OUT, a.tag), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            if not r.get("fail"):
                w.writerow(r)
    json.dump({"config_hash": cfg_hash, "thresholds": {"safety_eps": safety_eps, "benefit_lcb": benefit_thr},
               "params": {"phi": a.phi, "beta": a.beta, "m": a.m, "noise": a.noise, "n_pseudo": a.n_pseudo},
               "summary": summary},
              open("%s/v2_%s_summary.json" % (OUT, a.tag), "w"), indent=1)
    # --- ceiling smoke verdict (World A is a CEILING demo: NO accept expected) ---
    def cells(world, group=None):
        return [v for v in summary.values() if v["world"] == world and (group is None or v["intervention"] in group)]
    prinA = cells("A", PRINCIPLED); prinB = cells("B", PRINCIPLED); prinC = cells("C", PRINCIPLED)
    oracleA = cells("A", DIAGNOSTIC); rndA = cells("A", ["random_k"])
    # World A: a safe target-beneficial cell exists that the gate does NOT accept, and its source-LOSO benefit
    # is not detectable; and random-k does not reproduce the oracle target gain.
    safe_ben = [v for v in prinA + oracleA if v["is_safe"] and v["target_beneficial"]]
    a_no_accept = all(v["gate_action"] != "ACCEPT" for v in prinA)
    oracle_gain = max([v["dtgt_bacc"] for v in oracleA], default=float("nan"))
    rnd_gain = max([v["dtgt_bacc"] for v in rndA], default=float("nan"))
    rnd_no_repro = not (rnd_gain == rnd_gain and rnd_gain > 0.01)
    A_pass = (len(safe_ben) >= 1) and a_no_accept and rnd_no_repro
    # World B: no unsafe accept
    uB = sum(1 for v in prinB if v["gate_action"] == "ACCEPT"); B_pass = uB == 0
    # World C: no accept AND a high-domain-gain cell exists with no target benefit
    aC = sum(1 for v in prinC if v["gate_action"] == "ACCEPT")
    hi_dg_useless = any(v["domain_gain"] > 0.05 and not v["target_beneficial"] for v in prinC)
    C_pass = (aC == 0) and hi_dg_useless
    print("\n=== V2 %s CEILING smoke verdict ===" % a.tag)
    print("  World A (target-beneficial but source-uncertifiable): safe+target-beneficial cells=%d, "
          "principled ACCEPTs=%d (want 0), oracle target dbAcc=%+.3f vs random_k=%+.3f -> %s"
          % (len(safe_ben), sum(1 for v in prinA if v["gate_action"] == "ACCEPT"), oracle_gain, rnd_gain,
             "PASS" if A_pass else "FAIL"))
    print("  World B (unsafe): unsafe-ACCEPTs=%d/%d (want 0) -> %s" % (uB, len(prinB), "PASS" if B_pass else "FAIL"))
    print("  World C (useless): ACCEPTs=%d/%d (want 0), high-domain-gain-useless cell present=%s -> %s"
          % (aC, len(prinC), hi_dg_useless, "PASS" if C_pass else "FAIL"))
    print("  OVERALL: %s" % ("PASS" if (A_pass and B_pass and C_pass) else "FAIL"))
    print("V2_%s_DONE" % a.tag.upper())


if __name__ == "__main__":
    main()

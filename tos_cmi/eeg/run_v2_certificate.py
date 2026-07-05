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


DEGEN_COND = 1e12   # feature-covariance condition number above which LEACE/whitening is numerically unreliable


def _cond(Z):
    """Condition number (max/min positive eigenvalue) of the feature covariance -- degeneracy diagnostic."""
    try:
        ev = np.linalg.eigvalsh(np.cov(Z.T))
        ev = ev[ev > 1e-12]
        return float(ev.max() / ev.min()) if len(ev) else float("inf")
    except Exception:
        return float("inf")


def _nuisance_m(z_dim, mode, fraction, m_min):
    """Nuisance-block width normalized to latent capacity: m = max(m_min, round(fraction*z_dim)) in fraction
    mode (removes the latent-dimension scaling artifact), else the fixed m_min."""
    if mode == "fraction_of_z_dim":
        return int(max(m_min, round(fraction * z_dim)))
    return int(m_min)


def _one(world, ds, bb, seed, p, n_source, alpha, interv, phi, beta, m_min, mode, fraction, noise, n_pseudo):
    fold = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
    base = {"world": world, "dataset": ds, "backbone": bb, "seed": seed, "fold": fold,
            "n_source": str(n_source), "alpha": alpha, "intervention": interv}
    cond = float("nan")
    try:
        d = np.load(p, allow_pickle=True)
        Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
        Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
        subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
        keep = _subsample_source(subj, n_source, seed)
        Zs, ys, subj = Zs[keep], ys[keep], subj[keep]
        z_dim = Zs.shape[1]
        m = _nuisance_m(z_dim, mode, fraction, m_min)     # normalized to latent capacity
        inj = inject(world, Zs, ys, subj, Zt, yt, alpha=alpha, beta=beta, phi=phi, seed=seed, m=m, noise=noise)
        cond = _cond(inj["Zs2"])
        if cond > DEGEN_COND:      # numerically unreliable metric -> DEGENERATE, do not average into the verdict
            return {**base, "n_cls": n_cls, "z_dim": z_dim, "m_eff": m, "cond": cond, "degenerate": True,
                    "skip_reason": "ill-conditioned feature covariance (cond=%.1e > %.0e)" % (cond, DEGEN_COND)}
        F = oracle_nuisance_eraser_factory(m) if interv in DIAGNOSTIC else FACTORIES[interv]
        sig = eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], yt, n_cls, F, seed, n_pseudo)
        return {**base, "ground_truth": inj["ground_truth"], "n_cls": n_cls, "z_dim": z_dim, "m_eff": m,
                "cond": cond, "degenerate": False, **sig}
    except np.linalg.LinAlgError as e:
        return {**base, "cond": cond, "degenerate": True, "skip_reason": "LinAlgError: " + repr(e)[:120]}
    except Exception as e:
        msg = repr(e)[:200]
        degen = any(k in msg.lower() for k in ["singular", "linalg", "not positive", "converg", "eig", "nan"])
        return {**base, "cond": cond, "fail": msg, "degenerate": degen,
                "skip_reason": ("numerical: " + msg) if degen else ""}


def _dumps(ds, bb, seed, nfolds):
    dd = "%s/%s_%s_LOSO" % (RESULTS, ds, bb)
    ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    return ps[:nfolds] if nfolds else ps


def build_manifest(rows, expected_folds_by_cell):
    """Per (dataset,backbone,seed): fold coverage + VALID/PARTIAL/DEGENERATE status + condition-number range.
    A (dataset,backbone,seed,fold) is DEGENERATE if any of its tasks was flagged degenerate; VALID if it has
    >=1 successful (non-degenerate, non-fail) task. Degenerate/failed folds are NOT averaged into the verdict."""
    manifest = {}
    keys = sorted(set((r["dataset"], r["backbone"], r["seed"]) for r in rows))
    for (ds, bb, sd) in keys:
        sub = [r for r in rows if (r["dataset"], r["backbone"], r["seed"]) == (ds, bb, sd)]
        folds = sorted(set(r["fold"] for r in sub))
        degen_folds, valid_folds, reasons, conds = [], [], {}, []
        for f in folds:
            fr = [r for r in sub if r["fold"] == f]
            fdeg = [r for r in fr if r.get("degenerate")]
            fok = [r for r in fr if not r.get("degenerate") and not r.get("fail")]
            cvals = [r["cond"] for r in fr if r.get("cond") == r.get("cond")]
            if cvals:
                conds.append(float(np.median(cvals)))
            if fdeg and not fok:
                degen_folds.append(f)
                reasons[str(f)] = next((r.get("skip_reason", "") for r in fdeg if r.get("skip_reason")), "degenerate")
            elif fok:
                valid_folds.append(f)
        exp = expected_folds_by_cell.get((ds, bb, sd), len(folds))
        status = ("VALID" if len(valid_folds) == exp and not degen_folds else
                  "DEGENERATE" if not valid_folds else "PARTIAL")
        manifest["%s|%s|%d" % (ds, bb, sd)] = {
            "dataset": ds, "backbone": bb, "seed": sd, "status": status,
            "expected_folds": exp, "valid_folds": valid_folds, "degenerate_folds": degen_folds,
            "n_valid": len(valid_folds), "n_degenerate": len(degen_folds),
            "frac_skipped": round((exp - len(valid_folds)) / exp, 3) if exp else 0.0,
            "cond_median_min": round(min(conds), 1) if conds else None,
            "cond_median_max": round(max(conds), 1) if conds else None,
            "skip_reasons": reasons}
    return manifest


def aggregate(rows, safety_eps, benefit_thr):
    rows = [r for r in rows if not r.get("fail") and not r.get("degenerate")]   # exclude degenerate + failed
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
            "router_acc": (lambda ra: float(np.mean(ra)) if ra else float("nan"))(
                [r["router_acc"] for r in sub if r.get("router_acc") == r.get("router_acc")]),
            "gate_action": action,
            "is_safe": bool(tucb <= safety_eps),                       # source-defined safety
            "target_beneficial": bool(dblo > benefit_thr)}            # actual (post-hoc) target gain
    return summary


def _load_nuisance_cfg(path):
    """Load the World-A nuisance-dimension rule from the frozen config (authoritative). Default = latent-
    capacity-normalized m = max(min, round(fraction*z_dim)), which removes the latent-dimension scaling artifact."""
    try:
        import yaml
        c = yaml.safe_load(open(path)) or {}
        return (str(c.get("nuisance_dim_mode", "fraction_of_z_dim")),
                float(c.get("nuisance_fraction", 0.20)), int(c.get("nuisance_dim_min", 4)))
    except Exception:
        return "fraction_of_z_dim", 0.20, 4


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
    ap.add_argument("--nuisance-mode", default=None, help="fraction_of_z_dim | fixed (default: from config)")
    ap.add_argument("--nuisance-fraction", type=float, default=None)
    ap.add_argument("--m", type=int, default=None, help="nuisance_dim_min (default: from config)")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    outdir = a.outdir
    safety_eps, benefit_thr = _load_thresholds(CONFIG)
    nmode0, nfrac0, nmin0 = _load_nuisance_cfg(CONFIG)
    nmode = a.nuisance_mode or nmode0
    nfrac = a.nuisance_fraction if a.nuisance_fraction is not None else nfrac0
    nmin = a.m if a.m is not None else nmin0
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
    print("V2 %s: %d tasks (n_jobs=%d), config %s, safety<=%.3f benefit>%.3f, phi=%.2f beta=%.2f, "
          "nuisance=%s frac=%.2f min=%d (EEGNet z=16->m=%d, TSMNet z=210->m=%d)"
          % (a.tag, len(tasks), N_JOBS, cfg_hash, safety_eps, benefit_thr, a.phi, a.beta, nmode, nfrac, nmin,
             _nuisance_m(16, nmode, nfrac, nmin), _nuisance_m(210, nmode, nfrac, nmin)), flush=True)
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_one)(w, ds, bb, sd, p, ns, al, iv, a.phi, a.beta, nmin, nmode, nfrac, a.noise, a.n_pseudo)
        for (w, ds, bb, sd, p, ns, al, iv) in tasks)
    nfail = sum(1 for r in rows if r.get("fail"))
    ndeg = sum(1 for r in rows if r.get("degenerate"))
    if nfail or ndeg:
        print("[%d FAILED, %d DEGENERATE of %d tasks]" % (nfail, ndeg, len(rows)), flush=True)
        for r in [r for r in rows if r.get("fail") or r.get("degenerate")][:20]:
            print("  [%s] %s %s a%s %s: %s" % ("DEGEN" if r.get("degenerate") else "FAIL", r["dataset"],
                  r["backbone"], r["alpha"], r["intervention"], r.get("skip_reason") or r.get("fail", "")), flush=True)
    # expected folds per (ds,bb,seed) = folds actually attempted (first N available)
    exp_folds = {}
    for ds in a.datasets:
        for bb in a.backbones:
            for sd in a.seeds:
                exp_folds[(ds, bb, sd)] = len(_dumps(ds, bb, sd, a.folds))
    manifest = build_manifest(rows, exp_folds)
    summary = aggregate(rows, safety_eps, benefit_thr)
    os.makedirs(outdir, exist_ok=True)
    open("%s/v2_design_hash.txt" % outdir, "w").write("%s  %s\n" % (cfg_hash, CONFIG))
    cols = ["world", "dataset", "backbone", "seed", "fold", "n_source", "alpha", "intervention",
            "ground_truth", "z_dim", "m_eff", "degenerate", "cond", "skip_reason", "src_task_full",
            "src_task_eras", "task_drop", "z_full", "z_eras", "domain_gain", "tgt_bacc_full",
            "tgt_bacc_eras", "router_acc"]
    with open("%s/v2_%s_rows.csv" % (outdir, a.tag), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)   # keep degenerate/fail rows in the CSV (flagged), for auditability
    json.dump({"config_hash": cfg_hash, "thresholds": {"safety_eps": safety_eps, "benefit_lcb": benefit_thr},
               "params": {"phi": a.phi, "beta": a.beta, "nuisance_mode": nmode, "nuisance_fraction": nfrac,
                          "nuisance_dim_min": nmin, "noise": a.noise, "n_pseudo": a.n_pseudo,
                          "m_EEGNet": _nuisance_m(16, nmode, nfrac, nmin),
                          "m_TSMNet": _nuisance_m(210, nmode, nfrac, nmin)},
               "n_tasks": len(rows), "n_fail": nfail, "n_degenerate": ndeg,
               "summary": summary},
              open("%s/v2_%s_summary.json" % (outdir, a.tag), "w"), indent=1)
    json.dump({"config_hash": cfg_hash, "manifest": manifest},
              open("%s/v2_%s_manifest.json" % (outdir, a.tag), "w"), indent=1)
    print("\n=== coverage manifest (degenerate/failed folds excluded from the verdict) ===")
    for k in sorted(manifest):
        v = manifest[k]
        print("  %-11s %-7s seed%d : %-9s valid %d/%d, degenerate %d, cond med [%s..%s]"
              % (v["dataset"], v["backbone"], v["seed"], v["status"], v["n_valid"], v["expected_folds"],
                 v["n_degenerate"], v["cond_median_min"], v["cond_median_max"]), flush=True)
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

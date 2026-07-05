"""Phase 2 dry-run driver: task-preserving / conditional linear erasure on Lee2019_MI & Cho2017 EEGNet only.

For each dump (outer-LOSO fold) and each eraser we compute the SAME source-only gate signals as Track B
(safety task-drop, source-LOSO benefit, subject-decode domain-gain) PLUS the actual target deployment
(target ΔbAcc / ΔNLL). The gate thresholds are the FROZEN Track B ones (config phase2_task_preserving_fixed.yaml);
the target is used ONLY here for the post-hoc audit, never to fit/select/gate anything.

Deployable erasers (fit on source, applied to held-out target with no target labels):
  leace_baseline, tp_leace_task_carrier_preserving, cc_leace_predicted_route_deployable, random_k.
Diagnostic (NOT deployable -- routes by TRUE labels, an upper bound):
  cc_leace_oracle_route_diagnostic.

  python -m tos_cmi.eeg.run_phase2_task_preserving --datasets Lee2019_MI Cho2017 \
      --backbones EEGNet --seeds 0 --folds 15 --n-pseudo 8
Writes tos_cmi/results/method_deepen/phase2/{phase2_config_hash.txt, phase2_dryrun_rows.csv,
phase2_dryrun_summary.json}. Report via report_phase2.py.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss

from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eeg.erasure_baselines import _ids
from tos_cmi.eeg.source_ood_benefit_gate import (_bacc, _subj_acc, _boot_bound, gate_action,
                                                 SAFETY_EPS, BENEFIT_LCB, build_eraser)
from tos_cmi.eeg.task_preserving_linear_erasure import tp_leace_factory
from tos_cmi.eeg.class_conditional_leace import cc_leace_factory_predicted, cc_leace_apply_oracle

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
OUT = "tos_cmi/results/method_deepen/phase2"
CONFIG = "tos_cmi/eeg/configs/phase2_task_preserving_fixed.yaml"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

# factory(Zf, yf, subjf_ids, n_cls, seed) -> apply(X). DEPLOYABLE (no target labels at apply time).
DEPLOYABLE = {
    "leace_baseline": lambda Zf, yf, sf, nc, sd: build_eraser(Zf, yf, sf, nc, "LEACE", ScoreFisherConfig(), sd),
    "tp_leace_task_carrier_preserving": tp_leace_factory,
    "cc_leace_predicted_route_deployable": cc_leace_factory_predicted,
    "random_k": lambda Zf, yf, sf, nc, sd: build_eraser(Zf, yf, sf, nc, "random_k", ScoreFisherConfig(), sd),
}
ORACLE = "cc_leace_oracle_route_diagnostic"
ERASERS = list(DEPLOYABLE) + [ORACLE]


def _task_scores(Ztr, ytr, Zte, yte, n_cls):
    h = LogisticRegression(max_iter=200, C=1.0).fit(Ztr, ytr)
    p = h.predict_proba(Zte)
    P = np.zeros((len(Zte), n_cls)); P[:, h.classes_] = p
    bacc = float(balanced_accuracy_score(yte, h.classes_[p.argmax(1)]))
    nll = float(log_loss(yte, P, labels=np.arange(n_cls)))
    return bacc, nll


def _stratified_split(n, seed):
    perm = np.random.default_rng(seed).permutation(n); cut = n // 2
    A = np.zeros(n, bool); A[perm[:cut]] = True
    return A, ~A


def eval_deployable(Zs, ys, subj, Zt, yt, n_cls, F, seed, n_pseudo):
    """Source-only gate signals + target deployment for one dump and one deployable eraser factory F."""
    A, Bm = _stratified_split(len(ys), seed)
    E = F(Zs[A], ys[A], subj[A], n_cls, seed)                       # fit on split A only -> held-out routing on B
    src_full = _bacc(Zs[A], ys[A], Zs[Bm], ys[Bm])
    src_eras = _bacc(E(Zs[A]), ys[A], E(Zs[Bm]), ys[Bm])
    subj_full = _subj_acc(Zs[A], subj[A], Zs[Bm], subj[Bm])
    subj_eras = _subj_acc(E(Zs[A]), subj[A], E(Zs[Bm]), subj[Bm])
    # benefit: source leave-one-source-subject-out pseudo-target
    subs = sorted(set(subj.tolist()))
    pick = list(np.random.default_rng(seed).permutation(subs)[:min(n_pseudo, len(subs))])
    benefit = []
    for s in pick:
        tr = subj != s; te = subj == s
        if len(np.unique(ys[te])) < 2 or len(np.unique(ys[tr])) < 2:
            continue
        Es = F(Zs[tr], ys[tr], subj[tr], n_cls, seed)
        benefit.append(_bacc(Es(Zs[tr]), ys[tr], Es(Zs[te]), ys[te]) - _bacc(Zs[tr], ys[tr], Zs[te], ys[te]))
    # target deployment: fit on ALL source, apply to held-out target (target labels only score here)
    Ef = F(Zs, ys, subj, n_cls, seed)
    tb_e, tn_e = _task_scores(Ef(Zs), ys, Ef(Zt), yt, n_cls)
    tb_f, tn_f = _task_scores(Zs, ys, Zt, yt, n_cls)
    return dict(src_task_full=src_full, src_task_eras=src_eras, task_drop=src_full - src_eras,
                subj_full=subj_full, subj_eras=subj_eras, domain_gain=(subj_full - subj_eras),
                benefit=benefit, tgt_bacc_full=tb_f, tgt_bacc_eras=tb_e,
                tgt_nll_full=tn_f, tgt_nll_eras=tn_e)


def eval_oracle(Zs, ys, subj, Zt, yt, n_cls, seed):
    """DIAGNOSTIC upper bound: class-conditional LEACE routed by TRUE labels (target labels used for routing)."""
    A, Bm = _stratified_split(len(ys), seed)
    apA = cc_leace_apply_oracle(Zs[A], ys[A], subj[A], n_cls)
    src_full = _bacc(Zs[A], ys[A], Zs[Bm], ys[Bm])
    src_eras = _bacc(apA(Zs[A], ys[A]), ys[A], apA(Zs[Bm], ys[Bm]), ys[Bm])
    subj_full = _subj_acc(Zs[A], subj[A], Zs[Bm], subj[Bm])
    subj_eras = _subj_acc(apA(Zs[A], ys[A]), subj[A], apA(Zs[Bm], ys[Bm]), subj[Bm])
    apF = cc_leace_apply_oracle(Zs, ys, subj, n_cls)
    tb_e, tn_e = _task_scores(apF(Zs, ys), ys, apF(Zt, yt), yt, n_cls)   # target routed by TRUE yt (ORACLE)
    tb_f, tn_f = _task_scores(Zs, ys, Zt, yt, n_cls)
    return dict(src_task_full=src_full, src_task_eras=src_eras, task_drop=src_full - src_eras,
                subj_full=subj_full, subj_eras=subj_eras, domain_gain=(subj_full - subj_eras),
                benefit=[], tgt_bacc_full=tb_f, tgt_bacc_eras=tb_e,
                tgt_nll_full=tn_f, tgt_nll_eras=tn_e)


def _one(ds, bb, seed, p, eraser, n_pseudo):
    fold = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
    try:
        d = np.load(p, allow_pickle=True)
        Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
        Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
        subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
        if eraser == ORACLE:
            sig = eval_oracle(Zs, ys, subj, Zt, yt, n_cls, seed)
        else:
            sig = eval_deployable(Zs, ys, subj, Zt, yt, n_cls, DEPLOYABLE[eraser], seed, n_pseudo)
        return {"dataset": ds, "backbone": bb, "seed": seed, "fold": fold, "eraser": eraser,
                "n_cls": n_cls, "chance": 1.0 / n_cls, **sig}
    except Exception as e:
        return {"dataset": ds, "backbone": bb, "seed": seed, "fold": fold, "eraser": eraser, "fail": repr(e)[:200]}


def _dumps(ds, bb, seed, nfolds):
    d = "%s/%s_%s_LOSO" % (RESULTS, ds, bb)
    ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (d, seed)),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    return ps[:nfolds] if nfolds else ps


def _load_thresholds(path):
    """Read the FROZEN gate thresholds from the config so they are authoritative, not decorative. Parses the
    YAML (regex fallback if PyYAML is missing); falls back to the module constants if the key is absent."""
    if not os.path.exists(path):
        return SAFETY_EPS, BENEFIT_LCB
    try:
        import yaml
        cfg = yaml.safe_load(open(path)) or {}
        return (float(cfg.get("safety_reject_task_drop_ucb", SAFETY_EPS)),
                float(cfg.get("benefit_accept_lcb", BENEFIT_LCB)))
    except Exception:
        txt = open(path).read()
        def g(k, d):
            m = re.search(r"^%s:\s*([0-9.]+)" % k, txt, re.M)
            return float(m.group(1)) if m else d
        return g("safety_reject_task_drop_ucb", SAFETY_EPS), g("benefit_accept_lcb", BENEFIT_LCB)


def aggregate(rows, safety_eps=SAFETY_EPS, benefit_thr=BENEFIT_LCB):
    rows = [r for r in rows if not r.get("fail")]
    summary = {}
    for ds in sorted(set(r["dataset"] for r in rows)):
        for bb in sorted(set(r["backbone"] for r in rows if r["dataset"] == ds)):
            for er in ERASERS:
                sub = [r for r in rows if r["dataset"] == ds and r["backbone"] == bb and r["eraser"] == er]
                if not sub:
                    continue
                folds = [r["fold"] for r in sub]
                tds = [r["task_drop"] for r in sub]
                bvals, bfolds = [], []
                for r in sub:
                    for v in r.get("benefit", []):
                        bvals.append(v); bfolds.append(r["fold"])
                rng = np.random.default_rng(0)
                tucb = _boot_bound(tds, folds, "upper", rng=rng)
                blcb = _boot_bound(bvals, bfolds, "lower", rng=rng) if bvals else float("nan")
                db = [r["tgt_bacc_eras"] - r["tgt_bacc_full"] for r in sub]
                dn = [r["tgt_nll_eras"] - r["tgt_nll_full"] for r in sub]
                dbm = _boot_bound(db, folds, "lower", rng=np.random.default_rng(1))
                dbu = _boot_bound(db, folds, "upper", rng=np.random.default_rng(2))
                dnm = _boot_bound(dn, folds, "lower", rng=np.random.default_rng(3))
                dnu = _boot_bound(dn, folds, "upper", rng=np.random.default_rng(4))
                deployable = er in DEPLOYABLE
                action = gate_action(tucb, blcb, safety_eps, benefit_thr) if deployable else "DIAGNOSTIC"
                summary["%s|%s|%s" % (ds, bb, er)] = {
                    "dataset": ds, "backbone": bb, "eraser": er, "deployable": deployable,
                    "n_folds": len(set(folds)), "chance": sub[0]["chance"],
                    "src_task_full": float(np.mean([r["src_task_full"] for r in sub])),
                    "src_task_eras": float(np.mean([r["src_task_eras"] for r in sub])),
                    "task_drop_ucb": tucb, "task_drop_mean": float(np.mean(tds)),
                    "subj_full": float(np.nanmean([r["subj_full"] for r in sub])),
                    "subj_eras": float(np.nanmean([r["subj_eras"] for r in sub])),
                    "domain_gain": float(np.nanmean([r["domain_gain"] for r in sub])),
                    "benefit_lcb": blcb, "benefit_mean": float(np.mean(bvals)) if bvals else float("nan"),
                    "tgt_bacc_full": float(np.mean([r["tgt_bacc_full"] for r in sub])),
                    "tgt_bacc_eras": float(np.mean([r["tgt_bacc_eras"] for r in sub])),
                    "dtgt_bacc": float(np.mean(db)), "dtgt_bacc_lo": dbm, "dtgt_bacc_hi": dbu,
                    "dtgt_nll": float(np.mean(dn)), "dtgt_nll_lo": dnm, "dtgt_nll_hi": dnu,
                    "gate_action": action}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Lee2019_MI", "Cho2017"])
    ap.add_argument("--backbones", nargs="+", default=["EEGNet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", type=int, default=15, help="first N folds (dry-run subset); 0=all")
    ap.add_argument("--n-pseudo", type=int, default=8)
    a = ap.parse_args()
    tasks = []
    for ds in a.datasets:
        for bb in a.backbones:
            for s in a.seeds:
                for p in _dumps(ds, bb, s, a.folds):
                    for er in ERASERS:
                        tasks.append((ds, bb, s, p, er))
    cfg_hash = hashlib.sha256(open(CONFIG).read().encode()).hexdigest()[:12] if os.path.exists(CONFIG) else "MISSING"
    safety_eps, benefit_thr = _load_thresholds(CONFIG)      # thresholds come FROM the frozen config
    print("Phase 2 dry-run: %d (dump x eraser) tasks, n_jobs=%d, n_pseudo=%d, config %s, safety<=%.3f benefit>%.3f (loaded from config)"
          % (len(tasks), N_JOBS, a.n_pseudo, cfg_hash, safety_eps, benefit_thr), flush=True)
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_one)(ds, bb, s, p, er, a.n_pseudo) for ds, bb, s, p, er in tasks)
    nfail = sum(1 for r in rows if r.get("fail"))
    if nfail:
        print("[%d FAILED]" % nfail, flush=True)
        for r in rows:
            if r.get("fail"):
                print("  [FAIL] %s %s seed%d fold%d %s: %s"
                      % (r["dataset"], r["backbone"], r["seed"], r["fold"], r["eraser"], r["fail"]), flush=True)
    summary = aggregate(rows, safety_eps, benefit_thr)
    os.makedirs(OUT, exist_ok=True)
    open("%s/phase2_config_hash.txt" % OUT, "w").write("%s  %s\n" % (cfg_hash, CONFIG))
    cols = ["dataset", "backbone", "seed", "fold", "eraser", "src_task_full", "src_task_eras", "task_drop",
            "subj_full", "subj_eras", "domain_gain", "tgt_bacc_full", "tgt_bacc_eras", "tgt_nll_full", "tgt_nll_eras"]
    with open("%s/phase2_dryrun_rows.csv" % OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            if not r.get("fail"):
                w.writerow(r)
    json.dump({"config_hash": cfg_hash, "summary": summary,
               "thresholds": {"safety_eps": SAFETY_EPS, "benefit_lcb": BENEFIT_LCB}},
              open("%s/phase2_dryrun_summary.json" % OUT, "w"), indent=1)
    print("\n=== Phase 2 dry-run (SOURCE gate signals + target audit) ===")
    print("  %-11s %-38s | src task after | task-drop UCB | subj->  | dom-gain | target ΔbAcc [CI]     | gate"
          % ("dataset", "eraser"))
    for k in sorted(summary):
        v = summary[k]
        print("  %-11s %-38s |  %.3f (was %.3f) |  %+.3f       | %.2f->%.2f | %+.3f  | %+.3f [%+.3f,%+.3f] | %s"
              % (v["dataset"], v["eraser"], v["src_task_eras"], v["src_task_full"], v["task_drop_ucb"],
                 v["subj_full"], v["subj_eras"], v["domain_gain"], v["dtgt_bacc"], v["dtgt_bacc_lo"],
                 v["dtgt_bacc_hi"], v["gate_action"]))
    print("PHASE2_DRYRUN_DONE")


if __name__ == "__main__":
    main()

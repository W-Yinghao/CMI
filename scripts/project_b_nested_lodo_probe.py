"""Project B Step-2A-NL: nested source-SITE LODO calibration feasibility probe (HF3 only).

Question answered (ONE question, not a parameter search):
  Does calibrating on genuinely held-out SOURCE SITES (nested LODO) produce a non-degenerate
  TTA-harm signal, where the cheap non-nested source-SUBJECT gate saw pseudo_gain identically 0?

Design (per user spec):
  Fixed config HF3 = cov0.8/prior0.4/montage0.2/concept1.2/concept_frac0.50, the SAME 5 HFRAC
  seeds [3,4,7,8,10] (NO re-prescan -> no selection bias). For each seed:
    - keep the ORIGINAL true held-out target site untouched (never trained on);
    - for each SOURCE site u (the 5 non-target sites):
        train a FRESH model on the 4 sites excluding {target_site, u};
        evaluate IDENTITY and OFFLINE_TTA on the now-OOD held-out source site u;
        record pseudo_delta_bacc = bAcc_TTA - bAcc_IDENTITY (site-level).
  Pseudo-unit = SITE (HFRAC concept shift is site-level; subject-level is the wrong OOD unit).

This is a bounded FEASIBILITY probe, NOT a router/gate implementation. It does not modify any
h2cmi core file; it only calls public h2cmi functions. Target harm per seed is reused from the
existing HFRAC HF3 runs (same deployment model), not recomputed.

Modes:
  --mode tasklist                          emit "seed pseudo_site" lines (25 tasks)
  --mode site --seed S --pseudo_site U --out DIR    one nested training + eval -> site JSON
  --mode aggregate --dir DIR               -> site_rows.csv, seed_summary.csv, summary.json + verdict
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import glob
import json
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

COMMON = dict(classes=3, chans=16, times=128, fs=128.0, sites=6, subjects=4, sessions=2,
              trials=60, noise=0.25, label_rho=0.0, epochs=30, bs=64, device="cpu")
HF3 = dict(cov=0.8, prior=0.4, montage=0.2, concept=1.2, concept_frac=0.50)
HF3_SEEDS = [3, 4, 7, 8, 10]
HFRAC_DIR = "/home/infres/yinwang/project_b_step2a_hfrac"     # for target-harm reuse

DIAG_KEYS = ["ess", "delta_density_nll", "transform_norm", "condition_number",
             "prior_shift", "pred_disagreement", "ood_score"]

SITE_COLS = ["seed", "target_site", "pseudo_site", "pseudo_site_is_concept",
             "source_concept_count_after_excluding_pseudo", "train_sites",
             "strict_bacc_identity", "offline_bacc_tta", "pseudo_delta_bacc", "pseudo_p_gt0",
             "adapted", "fallback_reason"] + DIAG_KEYS + ["missing_diagnostics"]

SEED_COLS = ["seed", "target_site", "target_delta_bacc", "target_p_gt0",
             "n_pseudo_sites", "n_concept_pseudo_sites",
             "pseudo_gain_min", "pseudo_gain_mean", "pseudo_gain_max",
             "pseudo_harm_count_zero_threshold", "pseudo_nonharm_count_zero_threshold",
             "pseudo_harm_count_margin_002", "pseudo_nonharm_count_margin_002",
             "pseudo_gain_has_two_classes_zero_threshold", "pseudo_gain_has_two_classes_margin_002"]


# ---------- RNG reconstruction (mirror eeg_simulator; validated in the sweep selftest) ----------
def reconstruct_concept_sites(n_sites, concept_frac, seed):
    n = int(round(concept_frac * n_sites))
    if n <= 0:
        return []
    return sorted(np.random.default_rng(seed + 1).choice(n_sites, size=n, replace=False).tolist())


def reconstruct_target_site(n_sites, seed):
    return int(np.random.default_rng(seed).choice(np.arange(n_sites), size=min(1, n_sites - 1),
                                                  replace=False)[0])


# ------------------------------------ one nested source-site run ------------------------------------
def run_nested_site(seed: int, pseudo_site: int, out_dir: str) -> dict:
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
    from h2cmi.tta.class_conditional import ClassConditionalTTA
    from h2cmi.eval.metrics import cluster_bootstrap_ci
    from sklearn.metrics import balanced_accuracy_score

    p = HF3
    shift = ShiftSpec(cov=p["cov"], prior=p["prior"], concept=p["concept"],
                      concept_site_frac=p["concept_frac"], montage=p["montage"],
                      noise=COMMON["noise"], label_mechanism_rho=COMMON["label_rho"])
    sim = EEGSimulator(COMMON["classes"], COMMON["chans"], COMMON["times"], COMMON["fs"],
                       shift=shift, seed=seed).sample(
        COMMON["sites"], COMMON["subjects"], COMMON["sessions"], COMMON["trials"])

    # ORIGINAL held-out target site (untouched, never trained on) — matches HFRAC exactly
    _, tgt_idx = train_target_split(sim, n_target_sites=1, seed=seed)
    target_site = int(np.unique(sim.site[tgt_idx])[0])
    assert target_site == reconstruct_target_site(COMMON["sites"], seed)
    assert pseudo_site != target_site, "pseudo_site must be a SOURCE site"
    concept_sites = [int(s) for s in sim.meta["concept_sites"]]

    site = sim.site
    train_mask = ~np.isin(site, [target_site, pseudo_site])   # 4 sites: exclude target AND pseudo
    pseudo_mask = site == pseudo_site                          # the now-OOD held-out source site
    train_sites = sorted(np.unique(site[train_mask]).tolist())
    src_concept_after = len(set(concept_sites) - {target_site, pseudo_site})

    # fresh model on the 4 training sites (full DAG kept; absent site levels are simply unused)
    cfg = H2Config(n_classes=COMMON["classes"])
    cfg.encoder.n_chans = COMMON["chans"]; cfg.encoder.n_times = COMMON["times"]; cfg.encoder.fs = COMMON["fs"]
    cfg.train.epochs = COMMON["epochs"]; cfg.train.device = COMMON["device"]
    cfg.train.seed = seed; cfg.train.batch_size = COMMON["bs"]

    Xtr, ytr = sim.X[train_mask], sim.y[train_mask]
    dom_tr = sim.domains.subset(np.where(train_mask)[0])
    model, _, _, _ = train_h2(Xtr, ytr, dom_tr, sim.dag, cfg, align_factor="site", verbose=False)
    pi_star = reference_prior(ytr, COMMON["classes"], cfg.align.reference_prior)

    # site-level TTA on pseudo-site u, EXACTLY like train_safety_gate does per pseudo-unit
    Xu, yu = sim.X[pseudo_mask], sim.y[pseudo_mask]
    subj_u = sim.domains.subset(np.where(pseudo_mask)[0]).factor("subject")
    U = _embed(model, Xu, cfg.train.device)
    p_id = _predict_generative(model, U, pi_star)
    tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, cfg.n_classes, cfg.train.device)
    res = tta.fit(U, pseudo_labels=p_id.argmax(1))
    p_ad = _predict_transform(model, U, res.transform, res.pi_T)
    pred_id, pred_ad = p_id.argmax(1), p_ad.argmax(1)
    bacc_id = float(balanced_accuracy_score(yu, pred_id))
    bacc_ad = float(balanced_accuracy_score(yu, pred_ad))
    pseudo_delta = bacc_ad - bacc_id

    # per-subject gains within site u -> cluster bootstrap p_gt0
    per_subj = {}
    for s in np.unique(subj_u):
        mm = subj_u == s
        if mm.sum() == 0:
            continue
        per_subj[int(s)] = (float(balanced_accuracy_score(yu[mm], pred_ad[mm]))
                            - float(balanced_accuracy_score(yu[mm], pred_id[mm])))
    boot = cluster_bootstrap_ci(per_subj)

    diag = res.diagnostics
    missing = [k for k in DIAG_KEYS if k not in diag]         # reason-code, never silent 0
    row = dict(
        seed=seed, target_site=target_site, pseudo_site=pseudo_site,
        pseudo_site_is_concept=bool(pseudo_site in concept_sites),
        source_concept_count_after_excluding_pseudo=src_concept_after,
        train_sites=train_sites,
        strict_bacc_identity=bacc_id, offline_bacc_tta=bacc_ad, pseudo_delta_bacc=pseudo_delta,
        pseudo_p_gt0=float(boot["p_gt0"]),
        adapted=bool(res.adapted), fallback_reason=str(diag.get("reason", "")),
        missing_diagnostics=missing,
    )
    for k in DIAG_KEYS:
        row[k] = float(diag[k]) if k in diag else float("nan")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"site_seed{seed}_u{pseudo_site}.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2, default=float)
    print(f"[nl] seed{seed} pseudo_site{pseudo_site} (concept={row['pseudo_site_is_concept']}): "
          f"bAcc_id={bacc_id:.3f} bAcc_tta={bacc_ad:.3f} d={pseudo_delta:+.3f} "
          f"p_gt0={row['pseudo_p_gt0']:.2f} adapted={res.adapted} -> {path}")
    return row


# ------------------------------------ aggregate + verdict ------------------------------------
def _twoclass(harm, nonharm):
    return bool(harm >= 1 and nonharm >= 1)


def aggregate(run_dir: str) -> None:
    rows = [json.load(open(fp)) for fp in sorted(glob.glob(os.path.join(run_dir, "site_seed*_u*.json")))]
    if not rows:
        raise SystemExit(f"no site JSONs in {run_dir}")

    def _fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.6g}"
        if isinstance(v, list):
            return "|".join(str(x) for x in v)
        return str(v)

    # site rows CSV
    with open(os.path.join(run_dir, "hf3_nested_lodo_site_rows.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(SITE_COLS)
        for r in sorted(rows, key=lambda r: (r["seed"], r["pseudo_site"])):
            w.writerow([_fmt(r.get(c)) for c in SITE_COLS])

    # per-seed summary
    by_seed: dict[int, list] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)

    def _target_harm(seed):
        fp = os.path.join(HFRAC_DIR, f"hfrac_HF3_seed{seed}.json")
        if not os.path.exists(fp):
            return float("nan"), float("nan")
        d = json.load(open(fp))["row"]
        return float(d["offline_delta_bacc"]), float(d["offline_p_gt0"])

    seed_summ = []
    for seed in sorted(by_seed):
        rs = sorted(by_seed[seed], key=lambda r: r["pseudo_site"])
        g = [r["pseudo_delta_bacc"] for r in rs]
        harm0 = sum(1 for x in g if x < 0.0)
        nonharm0 = sum(1 for x in g if x >= 0.0)
        harm2 = sum(1 for x in g if x <= -0.02)
        nonharm2 = sum(1 for x in g if x > -0.02)
        td, tp = _target_harm(seed)
        seed_summ.append(dict(
            seed=seed, target_site=rs[0]["target_site"], target_delta_bacc=td, target_p_gt0=tp,
            n_pseudo_sites=len(rs),
            n_concept_pseudo_sites=sum(1 for r in rs if r["pseudo_site_is_concept"]),
            pseudo_gain_min=float(np.min(g)), pseudo_gain_mean=float(np.mean(g)),
            pseudo_gain_max=float(np.max(g)),
            pseudo_harm_count_zero_threshold=harm0, pseudo_nonharm_count_zero_threshold=nonharm0,
            pseudo_harm_count_margin_002=harm2, pseudo_nonharm_count_margin_002=nonharm2,
            pseudo_gain_has_two_classes_zero_threshold=_twoclass(harm0, nonharm0),
            pseudo_gain_has_two_classes_margin_002=_twoclass(harm2, nonharm2),
        ))

    with open(os.path.join(run_dir, "hf3_nested_lodo_seed_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(SEED_COLS)
        for r in seed_summ:
            w.writerow([_fmt(r.get(c)) for c in SEED_COLS])

    # verdict (per user rules)
    n = len(seed_summ)
    two0 = sum(1 for r in seed_summ if r["pseudo_gain_has_two_classes_zero_threshold"])
    hm2 = sum(1 for r in seed_summ if r["pseudo_harm_count_margin_002"] >= 1)
    positive = (two0 >= 3 and hm2 >= 3)
    verdict = "positive" if positive else "negative"
    summary = dict(config="HF3", seeds=HF3_SEEDS, n_seeds=n,
                   seeds_with_two_class_zero=two0, seeds_with_margin002_harm=hm2,
                   rule="positive iff >=3/5 two_class_zero AND >=3/5 margin002_harm>=1",
                   verdict=verdict, seed_summary=seed_summ)
    with open(os.path.join(run_dir, "hf3_nested_lodo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n[aggregate] wrote site_rows / seed_summary / summary.json to {run_dir}")
    print("\n===== per-seed summary =====")
    for r in seed_summ:
        print(f"  seed{r['seed']:<2d} tgt_site={r['target_site']} tgt_d={r['target_delta_bacc']:+.3f} "
              f"(p={r['target_p_gt0']:.2f}) | pseudo d[min/mean/max]="
              f"{r['pseudo_gain_min']:+.3f}/{r['pseudo_gain_mean']:+.3f}/{r['pseudo_gain_max']:+.3f} "
              f"harm0={r['pseudo_harm_count_zero_threshold']}/{r['n_pseudo_sites']} "
              f"harm.02={r['pseudo_harm_count_margin_002']}/{r['n_pseudo_sites']} "
              f"2class0={r['pseudo_gain_has_two_classes_zero_threshold']} "
              f"2class.02={r['pseudo_gain_has_two_classes_margin_002']}")
    print(f"\n===== NESTED SIGNAL VERDICT = {verdict.upper()} =====")
    print(f"  seeds with 2-class(zero) = {two0}/{n} (need >=3); "
          f"seeds with margin-.02 harm>=1 = {hm2}/{n} (need >=3)")


def tasklist() -> None:
    for seed in HF3_SEEDS:
        t = reconstruct_target_site(COMMON["sites"], seed)
        for u in range(COMMON["sites"]):
            if u != t:
                print(f"{seed} {u}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Project B nested source-site LODO probe (HF3)")
    ap.add_argument("--mode", required=True, choices=["tasklist", "site", "aggregate"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--pseudo_site", type=int)
    ap.add_argument("--out", default="/tmp/project_b_step2a_nested_lodo")
    ap.add_argument("--dir", default="/tmp/project_b_step2a_nested_lodo")
    args = ap.parse_args()
    if args.mode == "tasklist":
        tasklist()
    elif args.mode == "site":
        if args.seed is None or args.pseudo_site is None:
            raise SystemExit("--seed and --pseudo_site required")
        run_nested_site(args.seed, args.pseudo_site, args.out)
    elif args.mode == "aggregate":
        aggregate(args.dir)


if __name__ == "__main__":
    main()

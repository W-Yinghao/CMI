"""Validate the LPC-CMI *neural* leakage proxy against an INDEPENDENT kNN estimate of
I(Z;D|Y) on the synthetic, and (new) against a numerical GROUND-TRUTH CMI.

Two modes
---------
``--mode proxy`` (original): learned-encoder comparison. Trains encoders across
methods/lambdas/seeds and compares the neural frozen-encoder probe KL(q||pi_y)
against the independent kNN estimate of I(Z;D|Y) on the encoder embedding. This is
the ~39-setting ruler-vs-kNN cross-check. No ground truth is available here: the
pushforward density of a learned neural encoder has no closed form.

``--mode truth`` (new, P1.2): TRUTH-ANCHORED comparison on the KNOWN generator.
Because the learned encoding admits no closed-form density, truth-anchoring is done
on the RAW generator features X (the information-preserving representation whose
p(X|D,Y) IS the DGP). We sweep 13 DGP parameter settings x 3 seeds = 39 settings and
report, per setting, three quantities: the neural posterior-KL RULER estimate
E KL(q(D|X,Y)||pi_y(D)) (across critic capacities), the independent kNN estimate, and
the Monte-Carlo TRUTH from ``true_cmi.py`` (exact-density estimand, with reported SE).
We then report Pearson & Spearman (ruler vs truth) and (kNN vs truth), calibration
slope/intercept, MAE and relative error, and critic-capacity sensitivity.

kNN estimator (license-clean, sklearn): I(Z;D|Y) = sum_y p(y) * I(Z;D|Y=y), where each
stratum's I(Z;D|Y=y) is sklearn's kNN mutual information (Ross 2014) between continuous Z
and discrete D, summed over Z-dims. Independent of the neural posterior used in training.

Neither the neural nor the kNN estimator is described as "unbiased" anywhere; only the
exact-density Monte-Carlo quantity (with its reported standard error) is the truth.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sanity_check import (DGP, make_data, train_one, embed, leakage_probe,
                          Head, kl_to_prior, empirical_priors)
from true_cmi import true_cmi_dgp

# Tiny full-batch critics on ~2k x 14 tensors: single-threaded torch is faster
# and avoids multi-core thrashing (project guidance: intra-op threads=1).
torch.set_num_threads(1)


class Critic(nn.Module):
    """q(D|X,Y) posterior head with a CONFIGURABLE hidden width ``h`` (the
    capacity knob). Distinct from sanity_check.Head, whose width is fixed at 64,
    so that the capacity-sensitivity sweep actually varies capacity."""

    def __init__(self, din, dout, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def knn_cmi(Z, y, d, n_cls):
    """kNN estimate of I(Z;D|Y) (nats), stratified by Y."""
    tot = 0.0
    for c in range(n_cls):
        m = y == c
        if m.sum() < 20 or len(np.unique(d[m])) < 2:
            continue
        mi = mutual_info_classif(Z[m], d[m], discrete_features=False, random_state=0)
        tot += m.mean() * float(mi.sum())
    return tot


def proxy_validation_main():
    dgp = DGP()
    pts = []
    for seed in range(3):
        data = make_data(dgp, seed)
        Xv, yv, dv = data["val"]
        for method in ["erm", "marginal", "lpc_uniform", "lpc_prior"]:
            for lam in [0.0, 0.5, 2.0, 8.0]:
                if method == "erm" and lam != 0.0:
                    continue
                enc, _, _ = train_one(method, data, dgp, lam=lam, epochs=60, seed=seed)
                neural_kl = leakage_probe(enc, data, dgp, seed=seed)["leakage_kl"]
                Zv = embed(enc, Xv)
                knn = knn_cmi(Zv, yv, dv, 2)
                pts.append(dict(method=method, lam=lam, seed=seed, neural_kl=neural_kl, knn_cmi=knn))
                print(f"  {method:12s} lam={lam:4.1f} seed={seed} | neural_KL={neural_kl:.3f}  kNN_CMI={knn:.3f}", flush=True)

    nk = np.array([p["neural_kl"] for p in pts])
    kn = np.array([p["knn_cmi"] for p in pts])
    r = np.corrcoef(nk, kn)[0, 1]
    rho = _spearman(nk, kn)
    print(f"\n=== proxy validation: Pearson r={r:.3f}  Spearman rho={rho:.3f}  (n={len(pts)}) ===")
    plt.figure(figsize=(5, 4.4))
    col = {"erm": "k", "marginal": "C1", "lpc_uniform": "C0", "lpc_prior": "C2"}
    for mth in col:
        s = [p for p in pts if p["method"] == mth]
        if s:
            plt.scatter([p["neural_kl"] for p in s], [p["knn_cmi"] for p in s],
                        c=col[mth], label=mth, s=36, alpha=.8, edgecolors="k", linewidths=.3)
    plt.xlabel("neural proxy  KL(q_probe ‖ π_y)")
    plt.ylabel("independent kNN  Î(Z;D|Y)  (nats)")
    plt.title(f"Leakage-proxy validation\nPearson r={r:.2f}, Spearman ρ={rho:.2f}")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("synthetic/proxy_validation.png", dpi=150)
    print("saved -> synthetic/proxy_validation.png")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# =========================================================================== #
#  TRUTH-ANCHORED VALIDATION (P1.2): ruler + kNN + Monte-Carlo ground truth   #
#  on the KNOWN generator (raw features).                                     #
# =========================================================================== #

# Default / wide source-domain (P(Y=1|D), flip-rate e_d) tuples.
_SRC_DEFAULT = ((0.15, 0.05), (0.38, 0.45), (0.62, 0.05), (0.85, 0.45))
_SRC_WIDE_E = ((0.15, 0.02), (0.38, 0.55), (0.62, 0.02), (0.85, 0.55))
_SRC_NO_E = ((0.15, 0.20), (0.38, 0.20), (0.62, 0.20), (0.85, 0.20))

# 13 KNOWN-generator settings spanning true I(X;D|Y) ~ 0 .. 1.15 nats, with the
# leakage driven by different channels (pure style, spurious-flip, mixtures) so
# the ruler/kNN are checked across leakage STRUCTURES, not a single monotone knob.
# 13 configs x 3 seeds = 39 settings (matched to the learned-encoder comparison).
TRUTH_CONFIGS = [
    ("null_noleak",       dict(style_scale=0.0, m_s=2.2, src=_SRC_NO_E)),
    ("xs_weak",           dict(style_scale=0.0, m_s=1.0, src=_SRC_DEFAULT)),
    ("xs_only",           dict(style_scale=0.0, m_s=2.2, src=_SRC_DEFAULT)),
    ("wideE_style0p3",    dict(style_scale=0.3, m_s=3.0, src=_SRC_WIDE_E)),
    ("style_0p4",         dict(style_scale=0.4, m_s=2.2, src=_SRC_DEFAULT)),
    ("style_0p6_noisy",   dict(style_scale=0.6, m_s=2.2, sig_s=1.4, src=_SRC_DEFAULT)),
    ("style_0p8",         dict(style_scale=0.8, m_s=2.2, src=_SRC_DEFAULT)),
    ("style_1p0_noxs",    dict(style_scale=1.0, m_s=0.0, src=_SRC_DEFAULT)),
    ("style_1p2",         dict(style_scale=1.2, m_s=2.2, src=_SRC_DEFAULT)),
    ("style_1p6",         dict(style_scale=1.6, m_s=2.2, src=_SRC_DEFAULT)),
    ("style_2p0",         dict(style_scale=2.0, m_s=2.2, src=_SRC_DEFAULT)),
    ("style_2p4",         dict(style_scale=2.4, m_s=2.2, src=_SRC_DEFAULT)),
    ("style_2p0_wideE",   dict(style_scale=2.0, m_s=4.0, src=_SRC_WIDE_E)),
]

# Neural-critic capacities (hidden width of the q(D|X,Y) posterior head), from a
# tight under-capacity bottleneck (2) to an over-parameterised head (128).
CAPACITIES = (2, 8, 32, 128)
PRIMARY_CAPACITY = 32


def neural_ruler_raw(Xtr, ytr, dtr, Xev, yev, n_dom, h=16, epochs=250,
                     device="cpu", seed=0):
    """Neural posterior-KL RULER estimate of I(X;D|Y) on RAW features.

    Trains a critic q(D|X,Y) (a ``Head`` with hidden width ``h`` = the capacity
    knob) on (Xtr,ytr)->dtr, then returns E_eval KL(q(D|X,Y) || pi_y(D)), the same
    functional used by the deployed frozen-encoder probe but applied to the raw
    generator features. ``pi_y`` is the Laplace-smoothed empirical label-conditional
    domain prior from the training split (matching the method). This is an
    ESTIMATOR of the CMI, not a ground truth.
    """
    torch.manual_seed(seed)
    pi_y, _, _ = empirical_priors(ytr, dtr, n_dom)
    log_pi = torch.log(torch.tensor(pi_y, dtype=torch.float32, device=device))
    q = Critic(Xtr.shape[1] + 2, n_dom, h).to(device)   # h = capacity knob
    opt = torch.optim.Adam(q.parameters(), lr=2e-3)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    dtr_t = torch.tensor(dtr, dtype=torch.long, device=device)
    inp_tr = torch.cat([Xtr_t, F.one_hot(ytr_t, 2).float()], 1)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(q(inp_tr), dtr_t).backward()
        opt.step()
    with torch.no_grad():
        Xev_t = torch.tensor(Xev, dtype=torch.float32, device=device)
        yev_t = torch.tensor(yev, dtype=torch.long, device=device)
        inp_ev = torch.cat([Xev_t, F.one_hot(yev_t, 2).float()], 1)
        kl = kl_to_prior(q(inp_ev), log_pi[yev_t]).item()
    return float(kl)


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _calibration(truth, est):
    """Slope, intercept of OLS regression est ~ slope*truth + intercept."""
    truth, est = np.asarray(truth, float), np.asarray(est, float)
    slope, intercept = np.polyfit(truth, est, 1)
    return float(slope), float(intercept)


def _error_metrics(truth, est, rel_floor=0.05):
    """MAE, normalized MAE (MAE / mean|truth|), and median relative error over
    settings whose true CMI exceeds ``rel_floor`` (to avoid divide-by-near-zero)."""
    truth, est = np.asarray(truth, float), np.asarray(est, float)
    mae = float(np.mean(np.abs(est - truth)))
    denom = float(np.mean(np.abs(truth)))
    nmae = float(mae / denom) if denom > 0 else float("nan")
    mask = np.abs(truth) > rel_floor
    if mask.any():
        med_rel = float(np.median(np.abs(est[mask] - truth[mask]) / np.abs(truth[mask])))
    else:
        med_rel = float("nan")
    return dict(mae=mae, normalized_mae=nmae, median_relative_error=med_rel)


def _compare(truth, est):
    return dict(
        pearson=_pearson(truth, est),
        spearman=_spearman(np.asarray(truth), np.asarray(est)),
        calibration_slope=_calibration(truth, est)[0],
        calibration_intercept=_calibration(truth, est)[1],
        **_error_metrics(truth, est),
    )


def truth_anchored_sweep(configs=None, seeds=3, capacities=CAPACITIES,
                         critic_epochs=250, n_mc=600_000, n_data=2000,
                         outdir="results/cmi_trace_p0p1/synthetic_truth",
                         partial=False, tag=None):
    """Run the truth-anchored comparison and save per-setting + summary results.

    For every (config, seed) it computes: MC ground-truth CMI (once per config),
    the neural ruler estimate at each capacity, and the independent kNN estimate,
    all on the RAW generator features. Saves CSV + JSON under ``outdir``.
    """
    configs = configs or TRUTH_CONFIGS
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ground truth is a property of the DGP config -> compute once per config.
    truth_by_cfg = {}
    for name, kw in configs:
        r = true_cmi_dgp(DGP(**kw), seed=0, n_samples=n_mc)
        truth_by_cfg[name] = r
        print(f"[truth] {name:18s} I(X;D|Y)={r['true_cmi_nats']:.4f} "
              f"+/- {r['mc_se']:.4f} (n_mc={r['n_samples']})", flush=True)

    rows = []
    for name, kw in configs:
        dgp = DGP(**kw)
        n_dom = dgp.n_src
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            Xtr, ytr, dtr = dgp.sample(n_data // n_dom, rng)      # critic-train split
            Xev, yev, dev = dgp.sample(n_data // n_dom, rng)      # eval split (ruler+kNN)
            ruler = {h: neural_ruler_raw(Xtr, ytr, dtr, Xev, yev, n_dom,
                                         h=h, epochs=critic_epochs, seed=seed)
                     for h in capacities}
            knn = knn_cmi(Xev, yev, dev, 2)
            tr = truth_by_cfg[name]
            row = dict(config=name, seed=seed,
                       true_cmi_nats=tr["true_cmi_nats"], mc_se=tr["mc_se"],
                       knn_cmi=float(knn),
                       ruler_primary=ruler[PRIMARY_CAPACITY])
            for h in capacities:
                row[f"ruler_h{h}"] = ruler[h]
            rows.append(row)
            print(f"  {name:18s} seed={seed} | truth={tr['true_cmi_nats']:.3f} "
                  f"knn={knn:.3f} ruler@{PRIMARY_CAPACITY}={ruler[PRIMARY_CAPACITY]:.3f}",
                  flush=True)

    truth = np.array([r["true_cmi_nats"] for r in rows])
    knn = np.array([r["knn_cmi"] for r in rows])
    ruler_primary = np.array([r["ruler_primary"] for r in rows])

    summary = {
        "meta": {
            "partial": bool(partial),
            "tag": tag,
            "n_settings": len(rows),
            "n_configs": len(configs),
            "n_seeds": seeds,
            "capacities": list(capacities),
            "primary_capacity": PRIMARY_CAPACITY,
            "n_mc_truth": n_mc,
            "n_data_per_split": n_data,
            "critic_epochs": critic_epochs,
            "mc_se_range": [float(np.min([r["mc_se"] for r in rows])),
                            float(np.max([r["mc_se"] for r in rows]))],
            "truth_range_nats": [float(truth.min()), float(truth.max())],
            "scope_note": (
                "Ground truth is defined for the KNOWN generator's RAW features X "
                "(p(X|D,Y) is closed form). A learned neural encoder has no "
                "closed-form pushforward density, so no numerical truth exists "
                "for it; the learned-encoder ruler-vs-kNN check (mode=proxy) stays "
                "an independent cross-check. Neither estimator is called unbiased."
            ),
        },
        "ruler_primary_vs_truth": _compare(truth, ruler_primary),
        "knn_vs_truth": _compare(truth, knn),
        "capacity_sensitivity": {
            f"h{h}": _compare(truth, np.array([r[f"ruler_h{h}"] for r in rows]))
            for h in capacities
        },
    }

    stem = "truth_anchored" + (f"_{tag}" if tag else "")
    csv_path = outdir / f"{stem}_per_setting.csv"
    json_path = outdir / f"{stem}_summary.json"
    fieldnames = (["config", "seed", "true_cmi_nats", "mc_se", "knn_cmi",
                   "ruler_primary"] + [f"ruler_h{h}" for h in capacities])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "per_setting": rows}, f, indent=2)

    rp = summary["ruler_primary_vs_truth"]
    kv = summary["knn_vs_truth"]
    banner = "PARTIAL" if partial else "FULL"
    print(f"\n=== TRUTH-ANCHORED [{banner}]  n={len(rows)}  "
          f"truth in [{truth.min():.3f},{truth.max():.3f}] nats ===")
    print(f"  ruler(h{PRIMARY_CAPACITY}) vs truth : Pearson={rp['pearson']:.3f} "
          f"Spearman={rp['spearman']:.3f} slope={rp['calibration_slope']:.3f} "
          f"intercept={rp['calibration_intercept']:.3f} MAE={rp['mae']:.3f} "
          f"nMAE={rp['normalized_mae']:.3f} medRelErr={rp['median_relative_error']:.3f}")
    print(f"  kNN         vs truth : Pearson={kv['pearson']:.3f} "
          f"Spearman={kv['spearman']:.3f} slope={kv['calibration_slope']:.3f} "
          f"intercept={kv['calibration_intercept']:.3f} MAE={kv['mae']:.3f} "
          f"nMAE={kv['normalized_mae']:.3f} medRelErr={kv['median_relative_error']:.3f}")
    print("  capacity sensitivity (ruler Pearson / MAE vs truth):")
    for h in capacities:
        c = summary["capacity_sensitivity"][f"h{h}"]
        print(f"    h={h:<3d} Pearson={c['pearson']:.3f}  MAE={c['mae']:.3f}")
    print(f"saved -> {csv_path}")
    print(f"saved -> {json_path}")
    _plot_truth(rows, capacities, outdir / f"{stem}_scatter.png")
    return summary, rows


def _plot_truth(rows, capacities, path):
    truth = np.array([r["true_cmi_nats"] for r in rows])
    knn = np.array([r["knn_cmi"] for r in rows])
    ruler = np.array([r["ruler_primary"] for r in rows])
    lim = max(truth.max(), knn.max(), ruler.max()) * 1.05
    plt.figure(figsize=(5, 4.6))
    plt.plot([0, lim], [0, lim], "k--", lw=.8, alpha=.6, label="y=x")
    plt.scatter(truth, ruler, c="C2", s=34, alpha=.85, edgecolors="k",
                linewidths=.3, label=f"neural ruler (h{PRIMARY_CAPACITY})")
    plt.scatter(truth, knn, c="C0", s=34, alpha=.85, edgecolors="k",
                linewidths=.3, marker="^", label="independent kNN")
    plt.xlabel("Monte-Carlo TRUTH  I(X;D|Y)  (nats)")
    plt.ylabel("estimate  (nats)")
    plt.title("Truth-anchored CMI validation (known generator)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"saved -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["proxy", "truth"], default="truth",
                    help="proxy = learned-encoder ruler-vs-kNN; truth = "
                         "truth-anchored ruler+kNN+MC-truth on the known generator")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-mc", type=int, default=600_000,
                    help="Monte-Carlo samples for the ground-truth CMI")
    ap.add_argument("--critic-epochs", type=int, default=250)
    ap.add_argument("--partial", action="store_true",
                    help="mark outputs PARTIAL (e.g. reduced config subset)")
    ap.add_argument("--max-configs", type=int, default=0,
                    help="if >0, use only the first N configs (reduced smoke run)")
    ap.add_argument("--tag", default=None, help="filename suffix for outputs")
    ap.add_argument("--outdir", default="results/cmi_trace_p0p1/synthetic_truth")
    args = ap.parse_args()

    if args.mode == "proxy":
        proxy_validation_main()
        return

    configs = TRUTH_CONFIGS
    partial = args.partial
    if args.max_configs and args.max_configs < len(TRUTH_CONFIGS):
        configs = TRUTH_CONFIGS[:args.max_configs]
        partial = True
    truth_anchored_sweep(configs=configs, seeds=args.seeds,
                         critic_epochs=args.critic_epochs, n_mc=args.n_mc,
                         outdir=args.outdir, partial=partial, tag=args.tag)


if __name__ == "__main__":
    main()

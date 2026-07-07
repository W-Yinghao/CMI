"""CPU validation for DualPC: joint P(z) and P(y|Z) control.

This script intentionally uses the production trainer (`cmi.train.trainer.train_model`)
instead of a separate toy optimizer. It checks whether the new factorized `dualpc`
method and the direct-marginal `dualpc_marginal` ablation can be run end-to-end on
the same synthetic shifts used by dual_cmi_v2, and reports held-out source probes for:

  cond_kl_rw    : GLS conditional domain leakage I_w(Z;D|Y), the factorized representation side.
  pz_kl_rw      : weighted marginal domain leakage I_w(Z;D), the P(z) side.
  py_res_rw     : weighted residual decoder CMI CE(h0)-CE(h), the P(y|Z) side.
  target_bacc   : target-domain balanced accuracy.

Quick CPU smoke:
  /home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
      --quick --out results/dualpc_synthetic_quick.json

Fuller CPU run (still no GPU/SLURM):
  /home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
      --epochs 80 --probe_epochs 200 --n 800 --seeds 3 --out results/dualpc_synthetic.json

Source-only selection smoke, mirroring cmi.run_lambda_select without target labels:
  /home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
      --n 200 --seeds 1 --epochs 10 --probe_epochs 25 --source_select \
      --select_rule guarded_probe --select_lams 0 0.05 --select_gammas 0.05 \
      --out results/dualpc_synthetic_select_smoke.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SYNTHETIC_DIR = Path(__file__).resolve().parent
if str(SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC_DIR))

from cmi.train.trainer import train_model, predict, embed, resolve_dec_margin
from cmi.eval.metrics import domain_class_span_stats
from dual_cmi_v2 import gen, split_sources


class TinyBackbone(nn.Module):
    """Small MLP that exposes (logits, Z) and accepts X shaped [N,1,6]."""
    def __init__(self, n_in=6, z_dim=16, hidden=64, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim), nn.ReLU(),
        )
        self.head = nn.Linear(z_dim, n_cls)
        self.z_dim = z_dim

    def forward(self, x):
        z = self.net(x)
        return self.head(z), z


def _mlp(din, dout, hidden=64):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, dout))


def _gls_weights(y, d, n_dom, n_cls):
    counts = np.zeros((n_dom, n_cls), dtype=np.float64)
    for yi, di in zip(y, d):
        counts[int(di), int(yi)] += 1.0
    pi_d = (counts + 1.0) / (counts.sum(1, keepdims=True) + n_cls)
    w = (1.0 / n_cls) / pi_d[d, y]
    return (w / w.mean()).astype("float32")


def _wmean(per, w):
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _weighted_ce(logits, target, weight):
    return _wmean(F.cross_entropy(logits, target, reduction="none"), weight)


def _weighted_js(logits_a, logits_b, weight):
    pa = F.softmax(logits_a, 1).clamp_min(1e-8)
    pb = F.softmax(logits_b, 1).clamp_min(1e-8)
    m = (0.5 * (pa + pb)).clamp_min(1e-8)
    js = 0.5 * (pa * (pa.log() - m.log())).sum(1) + 0.5 * (pb * (pb.log() - m.log())).sum(1)
    return _wmean(js, weight)


def _label_domain_priors(y, d, n_dom, n_cls, alpha=1.0):
    counts = np.full((n_cls, n_dom), float(alpha), dtype=np.float64)
    for yi, di in zip(y, d):
        counts[int(yi), int(di)] += 1.0
    return counts / counts.sum(1, keepdims=True)


def measure_dualpc_probes(backbone, Xheld, yheld, dheld, n_cls=2, probe_epochs=150,
                          seed=0, device="cpu"):
    """Held-out source probes for P(z) and P(y|Z), fit/eval split disjoint."""
    torch.manual_seed(3000 + seed)
    rng = np.random.default_rng(4000 + seed)
    n_dom = int(dheld.max()) + 1
    idx = rng.permutation(len(yheld))
    fit, ev = idx[: len(idx) // 2], idx[len(idx) // 2:]
    validity = domain_class_span_stats(yheld, dheld, n_cls)

    Z = embed(backbone, Xheld, device=device)
    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    yt = torch.tensor(yheld, dtype=torch.long, device=device)
    dt = torch.tensor(dheld, dtype=torch.long, device=device)
    woh = torch.tensor(_gls_weights(yheld, dheld, n_dom, n_cls), dtype=torch.float32, device=device)
    d_oh = F.one_hot(dt, n_dom).float()

    qD = _mlp(Z.shape[1], n_dom).to(device)
    qDY = _mlp(Z.shape[1] + n_cls, n_dom).to(device)
    qY = _mlp(Z.shape[1], n_cls).to(device)
    hY = _mlp(Z.shape[1] + n_dom, n_cls).to(device)
    uY = _mlp(Z.shape[1], n_cls).to(device)
    bD = torch.zeros(n_dom, n_cls, device=device, requires_grad=True)
    opt = torch.optim.Adam(list(qD.parameters()) + list(qDY.parameters()) + list(qY.parameters()) + list(hY.parameters())
                           + list(uY.parameters()) + [bD], 2e-3)

    f = torch.tensor(fit, dtype=torch.long, device=device)
    e = torch.tensor(ev, dtype=torch.long, device=device)
    for _ in range(probe_epochs):
        zy_f = torch.cat([Zt[f], F.one_hot(yt[f], n_cls).float()], 1)
        loss = (_weighted_ce(qD(Zt[f]), dt[f], woh[f])
                + _weighted_ce(qDY(zy_f), dt[f], woh[f])
                + _weighted_ce(qY(Zt[f]), yt[f], woh[f])
                + _weighted_ce(hY(torch.cat([Zt[f], d_oh[f]], 1)), yt[f], woh[f])
                + _weighted_ce(uY(Zt[f]) + bD[dt[f]], yt[f], woh[f]))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        # Reference marginal under the GLS measure, estimated from the probe-fit half.
        wd = torch.zeros(n_dom, device=device)
        wd.scatter_add_(0, dt[f], woh[f])
        pd_ref = (wd + 1e-6) / (wd.sum() + 1e-6 * n_dom)
        logq = F.log_softmax(qD(Zt[e]), 1)
        pz_kl_i = (logq.exp() * (logq - torch.log(pd_ref).unsqueeze(0))).sum(1)
        pz_kl_rw = _wmean(pz_kl_i, woh[e]).item()

        raw_pd = torch.bincount(dt[f], minlength=n_dom).float()
        raw_pd = (raw_pd + 1e-6) / (raw_pd.sum() + 1e-6 * n_dom)
        pz_kl_raw_i = (logq.exp() * (logq - torch.log(raw_pd).unsqueeze(0))).sum(1)
        pz_kl_raw = pz_kl_raw_i.mean().item()

        pi_y = _label_domain_priors(yheld[fit], dheld[fit], n_dom, n_cls)
        log_pi_y = torch.log(torch.tensor(pi_y, dtype=torch.float32, device=device))
        zy_e = torch.cat([Zt[e], F.one_hot(yt[e], n_cls).float()], 1)
        logq_cond = F.log_softmax(qDY(zy_e), 1)
        cond_kl_i = (logq_cond.exp() * (logq_cond - log_pi_y[yt[e]])).sum(1)
        cond_kl = cond_kl_i.mean().item()
        cond_kl_rw_i = (logq_cond.exp() * (logq_cond - torch.log(pd_ref).unsqueeze(0))).sum(1)
        cond_kl_rw = _wmean(cond_kl_rw_i, woh[e]).item()

        ce_q = _weighted_ce(qY(Zt[e]), yt[e], woh[e]).item()
        lh = hY(torch.cat([Zt[e], d_oh[e]], 1))
        l0 = uY(Zt[e]) + bD[dt[e]]
        ce_h = _weighted_ce(lh, yt[e], woh[e]).item()
        ce_0 = _weighted_ce(l0, yt[e], woh[e]).item()
        js = _weighted_js(lh, l0, woh[e]).item()
    out = dict(
        cond_kl=max(cond_kl, 0.0),
        cond_kl_rw=max(cond_kl_rw, 0.0),
        pz_kl_rw=max(pz_kl_rw, 0.0),
        pz_kl_raw=max(pz_kl_raw, 0.0),
        py_raw_rw=max(ce_q - ce_h, 0.0),
        py_res_rw=max(ce_0 - ce_h, 0.0),
        py_js_rw=max(js, 0.0),
        ce_q=ce_q,
        ce_h=ce_h,
        ce_0=ce_0,
    )
    out.update(validity)
    return out


def _np(t):
    return t.cpu().numpy() if torch.is_tensor(t) else t


def _prepare_split(dgp_kw, args, seed):
    X, Y, D, _ = gen(seed=seed, K=args.domains, n=args.n, **dgp_kw)
    s = split_sources(X, Y, D, args.domains, seed=seed)
    return dict(
        Xfit=_np(s["Xfit"])[:, None, :],
        yfit=_np(s["yfit"]).astype("int64"),
        dfit=_np(s["dfit"]).astype("int64"),
        Xheld=_np(s["Xheld"])[:, None, :],
        yheld=_np(s["yheld"]).astype("int64"),
        dheld=_np(s["dheld"]).astype("int64"),
        Xtgt=_np(s["Xtgt"])[:, None, :],
        ytgt=_np(s["ytgt"]).astype("int64"),
    )


def _method_for_fit(method, lam, gamma):
    return "erm" if lam == 0 and gamma == 0 else method


def _config_label(method, lam, gamma):
    if lam == 0 and gamma == 0:
        return "erm:0"
    if method in {"dual", "dualc", "dualpc", "dualpc_marginal"}:
        return f"{method}:{lam:g}:{gamma:g}"
    return f"{method}:{lam:g}"


def _candidate_grid(method, args):
    lams = list(dict.fromkeys(float(x) for x in args.select_lams))
    if 0.0 not in lams:
        lams = [0.0] + lams
    if method in {"dualc", "dualpc", "dualpc_marginal"}:
        gammas = list(dict.fromkeys(float(x) for x in args.select_gammas))
        return [(0.0, 0.0)] + [(la, ga) for la in lams if la != 0.0 for ga in gammas]
    return [(la, 0.0) for la in lams]


def _source_val_mask(dfit, val_frac, seed):
    """Split the training half inside every source domain; leaves Xheld untouched for probes."""
    rng = np.random.default_rng(9000 + seed)
    val = np.zeros(len(dfit), dtype=bool)
    for dom in np.unique(dfit):
        idx = np.where(dfit == dom)[0]
        rng.shuffle(idx)
        if len(idx) <= 1:
            continue
        n_val = int(round(val_frac * len(idx)))
        n_val = min(max(1, n_val), len(idx) - 1)
        val[idx[:n_val]] = True
    if not val.any() or val.all():
        raise RuntimeError("invalid source validation split; adjust --val_frac or --n")
    return val


def _fit_backbone(Xfit, yfit, dfit, method, lam, gamma, args, seed, epochs):
    torch.manual_seed(1000 + int(seed))
    bb = TinyBackbone(z_dim=args.z_dim, hidden=args.hidden)
    fit_method = _method_for_fit(method, lam, gamma)
    bb, _, diag = train_model(
        bb, Xfit, yfit, dfit, n_cls=2, method=fit_method,
        lam=lam, gamma=gamma, dec_margin=resolve_dec_margin(fit_method, args.dec_margin),
        epochs=epochs, bs=args.bs,
        warmup=max(1, args.warmup), n_inner=args.n_inner, sampler=args.sampler,
        device="cpu", seed=seed,
    )
    return bb, diag


def _evaluate_final(dgp_name, method_label, bb, diag, split, args, seed, extra=None):
    prob = predict(bb, split["Xtgt"], device="cpu", bs=512)
    probes = measure_dualpc_probes(bb, split["Xheld"], split["yheld"], split["dheld"], n_cls=2,
                                   probe_epochs=args.probe_epochs, seed=seed)
    out = dict(
        dgp=dgp_name,
        method=method_label,
        seed=int(seed),
        target_acc=float((prob.argmax(1) == split["ytgt"]).mean()),
        target_bacc=float(balanced_accuracy_score(split["ytgt"], prob.argmax(1))),
        inloop_reg=float(diag.get("inloop_reg", 0.0)),
        inloop_dec=float(diag.get("inloop_dec", 0.0)),
        inloop_dec_loss=float(diag.get("inloop_dec_loss", 0.0)),
        train_dec_margin=float(diag.get("dec_margin", resolve_dec_margin(method_label, args.dec_margin))),
        train_sampler=diag.get("sampler", args.sampler),
        **probes,
    )
    if extra:
        out.update(extra)
    return out


def run_one(dgp_name, dgp_kw, method_label, method, lam, gamma, args, seed):
    split = _prepare_split(dgp_kw, args, seed)
    bb, diag = _fit_backbone(split["Xfit"], split["yfit"], split["dfit"],
                             method, lam, gamma, args, seed, args.epochs)
    return _evaluate_final(dgp_name, method_label, bb, diag, split, args, seed)


def run_source_selected(dgp_name, dgp_kw, select_method, args, seed):
    split = _prepare_split(dgp_kw, args, seed)
    val = _source_val_mask(split["dfit"], args.val_frac, seed)
    tr = ~val
    candidates = _candidate_grid(select_method, args)
    sel_epochs = args.select_epochs or max(1, args.epochs // 2)
    candidate_scores = []

    for ci, (lam, gamma) in enumerate(candidates):
        bb, _ = _fit_backbone(split["Xfit"][tr], split["yfit"][tr], split["dfit"][tr],
                              select_method, lam, gamma, args, seed, sel_epochs)
        pred = predict(bb, split["Xfit"][val], device="cpu", bs=512).argmax(1)
        bacc = float(balanced_accuracy_score(split["yfit"][val], pred))
        label = _config_label(select_method, lam, gamma)
        rec = dict(config=label, lam=float(lam), gamma=float(gamma), source_val_bacc=bacc)
        if args.select_rule == "guarded_probe":
            pe = args.select_probe_epochs or max(10, args.probe_epochs // 2)
            probes = measure_dualpc_probes(
                bb, split["Xfit"][val], split["yfit"][val], split["dfit"][val],
                n_cls=2, probe_epochs=pe, seed=seed + 100 * (ci + 1),
            )
            decoder_valid = bool(probes.get("decoder_valid", False))
            py_term = probes["py_js_rw"] if decoder_valid else 0.0
            penalty = (args.select_cond_weight * probes["cond_kl_rw"]
                       + args.select_pz_weight * probes["pz_kl_rw"]
                       + args.select_py_weight * py_term)
            rec.update(
                selection_probe_valid=decoder_valid,
                select_cond_kl=float(probes["cond_kl"]),
                select_cond_kl_rw=float(probes["cond_kl_rw"]),
                select_pz_kl_rw=float(probes["pz_kl_rw"]),
                select_py_res_rw=float(probes["py_res_rw"]),
                select_py_js_rw=float(probes["py_js_rw"]),
                select_decoder_valid=decoder_valid,
                select_decoder_min_domain_classes=int(probes["decoder_min_domain_classes"]),
                select_decoder_single_class_frac=float(probes["decoder_single_class_frac"]),
                selector_penalty=float(penalty),
            )
        candidate_scores.append(rec)

    if args.select_rule == "bacc":
        best_i = 0
        for i, rec in enumerate(candidate_scores):
            if rec["source_val_bacc"] > candidate_scores[best_i]["source_val_bacc"]:
                best_i = i
    elif args.select_rule == "guarded_probe":
        best_bacc = max(x["source_val_bacc"] for x in candidate_scores)
        best_i, best_key = 0, None
        for i, rec in enumerate(candidate_scores):
            if rec["source_val_bacc"] + args.select_tolerance < best_bacc:
                continue
            key = (0 if rec.get("selection_probe_valid", False) else 1,
                   rec["selector_penalty"], -rec["source_val_bacc"], i)
            if best_key is None or key < best_key:
                best_i, best_key = i, key
    else:
        raise ValueError(args.select_rule)

    best = candidate_scores[best_i]
    best_lam, best_gamma, best_bacc = best["lam"], best["gamma"], best["source_val_bacc"]

    bb, diag = _fit_backbone(split["Xfit"], split["yfit"], split["dfit"],
                             select_method, best_lam, best_gamma, args, seed, args.epochs)
    selected = _config_label(select_method, best_lam, best_gamma)
    return _evaluate_final(
        dgp_name, f"{select_method}_select", bb, diag, split, args, seed,
        extra=dict(
            selected_config=selected,
            selected_lambda=float(best_lam),
            selected_gamma=float(best_gamma),
            source_val_bacc=float(best_bacc),
            select_rule=args.select_rule,
            selector_penalty=float(best.get("selector_penalty", 0.0)),
            candidate_scores=candidate_scores,
        ),
    )


def summarize(rows):
    out = {}
    keys = ["target_bacc", "target_acc", "cond_kl", "cond_kl_rw", "pz_kl_rw", "pz_kl_raw",
            "py_res_rw", "py_js_rw", "py_raw_rw",
            "decoder_valid", "decoder_min_domain_classes", "decoder_single_class_frac",
            "inloop_reg", "inloop_dec", "inloop_dec_loss", "train_dec_margin"]
    for dgp in sorted({r["dgp"] for r in rows}):
        out[dgp] = {}
        for method in sorted({r["method"] for r in rows if r["dgp"] == dgp}):
            rr = [r for r in rows if r["dgp"] == dgp and r["method"] == method]
            out[dgp][method] = {k: float(np.mean([x[k] for x in rr])) for k in keys}
            out[dgp][method].update({k + "_std": float(np.std([x[k] for x in rr])) for k in keys})
            valid_rr = [x for x in rr if bool(x.get("decoder_valid", False))]
            out[dgp][method]["decoder_valid_n"] = len(valid_rr)
            for k in ("py_raw_rw", "py_res_rw", "py_js_rw"):
                out[dgp][method][k + "_valid_mean"] = (
                    float(np.mean([x[k] for x in valid_rr])) if valid_rr else None
                )
            out[dgp][method]["n"] = len(rr)
            selected = [x.get("selected_config") for x in rr if x.get("selected_config")]
            if selected:
                out[dgp][method]["selected_hist"] = {k: selected.count(k) for k in sorted(set(selected))}
                out[dgp][method]["source_val_bacc"] = float(np.mean([x["source_val_bacc"] for x in rr]))
        if "erm" in out[dgp]:
            erm = out[dgp]["erm"]
            for method, vals in list(out[dgp].items()):
                if method == "erm":
                    continue
                vals["delta_target_bacc_vs_erm"] = vals["target_bacc"] - erm["target_bacc"]
                vals["delta_cond_kl_vs_erm"] = vals["cond_kl"] - erm["cond_kl"]
                vals["delta_cond_kl_rw_vs_erm"] = vals["cond_kl_rw"] - erm["cond_kl_rw"]
                vals["delta_pz_kl_rw_vs_erm"] = vals["pz_kl_rw"] - erm["pz_kl_rw"]
                vals["delta_py_res_rw_vs_erm"] = vals["py_res_rw"] - erm["py_res_rw"]
                vals["delta_py_js_rw_vs_erm"] = vals["py_js_rw"] - erm["py_js_rw"]
                vals["gate_cond_improved"] = bool(vals["delta_cond_kl_vs_erm"] < 0.0)
                vals["gate_cond_rw_improved"] = bool(vals["delta_cond_kl_rw_vs_erm"] < 0.0)
                vals["gate_pz_improved"] = bool(vals["delta_pz_kl_rw_vs_erm"] < 0.0)
                vals["gate_py_not_raised"] = bool(vals["delta_py_res_rw_vs_erm"] <= 0.005)
                vals["gate_py_js_not_raised"] = bool(vals["delta_py_js_rw_vs_erm"] <= 0.005)
                vals["gate_target_not_hurt_5pts"] = bool(vals["delta_target_bacc_vs_erm"] >= -0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short CPU smoke settings")
    ap.add_argument("--n", type=int, default=700, help="samples per domain before source/target split")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--probe_epochs", type=int, default=150)
    ap.add_argument("--domains", type=int, default=4)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n_inner", type=int, default=1)
    ap.add_argument("--sampler", default="raw", choices=["raw", "classbal", "domainbal"])
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--dec_margin", type=float, default=None,
                    help="decoder gate tau. Default is method-specific: dualpc/dualpc_marginal=0, others=0.02")
    ap.add_argument("--dgps", nargs="+", default=["covariate_label", "all_three"],
                    choices=["null_prior", "covariate_label", "concept", "all_three"])
    ap.add_argument("--source_select", action="store_true",
                    help="also run source-only hyperparameter selection, with no target labels")
    ap.add_argument("--select_methods", nargs="+", default=["dualpc"],
                    choices=["lpc_prior", "dualc", "dualpc", "dualpc_marginal"])
    ap.add_argument("--select_lams", nargs="+", type=float, default=[0.0, 0.05, 0.2])
    ap.add_argument("--select_gammas", nargs="+", type=float, default=[0.05, 0.2])
    ap.add_argument("--select_epochs", type=int, default=0)
    ap.add_argument("--select_rule", default="bacc", choices=["bacc", "guarded_probe"])
    ap.add_argument("--select_tolerance", type=float, default=0.02,
                    help="guarded_probe: source-val bAcc slack below the best candidate")
    ap.add_argument("--select_probe_epochs", type=int, default=0)
    ap.add_argument("--select_cond_weight", type=float, default=1.0,
                    help="guarded_probe weight on GLS conditional leakage I_w(Z;D|Y)")
    ap.add_argument("--select_pz_weight", type=float, default=1.0)
    ap.add_argument("--select_py_weight", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.quick:
        args.n = 250
        args.seeds = 2
        args.epochs = 18
        args.probe_epochs = 50
        args.warmup = 6
        args.bs = 128

    dgp_map = {
        "null_prior": dict(cov=0, con=0, labelshift=True),
        "covariate_label": dict(cov=1, con=0, labelshift=True),
        "concept": dict(cov=0, con=1, labelshift=False),
        "all_three": dict(cov=1, con=1, labelshift=True),
    }
    dgps = [(name, dgp_map[name]) for name in args.dgps]
    methods = [
        ("erm", "erm", 0.0, 0.0),
        ("lpc_prior", "lpc_prior", args.lam, 0.0),
        ("dualc", "dualc", args.lam, args.gamma),
        ("dualpc", "dualpc", args.lam, args.gamma),
        ("dualpc_marginal", "dualpc_marginal", args.lam, args.gamma),
    ]
    rows = []
    for dgp_name, dgp_kw in dgps:
        for method_label, method, lam, gamma in methods:
            for seed in range(args.seeds):
                rec = run_one(dgp_name, dgp_kw, method_label, method, lam, gamma, args, seed)
                rows.append(rec)
                print(f"{dgp_name:15s} {method_label:9s} seed={seed} "
                      f"bAcc={rec['target_bacc']*100:5.1f} "
                      f"PzKL={rec['pz_kl_rw']:.3f} PyRes={rec['py_res_rw']:.3f} "
                      f"sampler={rec['train_sampler']}", flush=True)
        if args.source_select:
            for select_method in args.select_methods:
                for seed in range(args.seeds):
                    rec = run_source_selected(dgp_name, dgp_kw, select_method, args, seed)
                    rows.append(rec)
                    print(f"{dgp_name:15s} {rec['method']:14s} seed={seed} "
                          f"cfg*={rec['selected_config']} srcVal={rec['source_val_bacc']*100:5.1f} "
                          f"selPen={rec['selector_penalty']:.3f} "
                          f"bAcc={rec['target_bacc']*100:5.1f} "
                          f"PzKL={rec['pz_kl_rw']:.3f} PyRes={rec['py_res_rw']:.3f} "
                          f"sampler={rec['train_sampler']}", flush=True)

    out = dict(config=vars(args), rows=rows, summary=summarize(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()

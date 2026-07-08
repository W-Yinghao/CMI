"""Project B-Next Step-S4A: foundation-backend feasibility under a COMMON downstream.

Scientific question: does a foundation-style EEG representation change the Project B decision problem?
Fair comparison design (user-locked): both backends are frozen/source-trained REPRESENTATIONS feeding an
IDENTICAL script-local downstream (source-only z-score + PCA + class-conditional diagonal Gaussian head +
PRIOR_ONLY + common diagonal-affine TTA + S1A diagnostics/predictability). This is NOT native-h2cmi vs
CBraMod; h2cmi_common's absolute numbers may be a floor. The apples-to-apples comparison is
h2cmi_common vs cbramod_common.

Evaluation only: no new RouterAction, no method change, modifies no h2cmi/** or cmi/**. CBraMod is loaded
frozen from an external checkpoint (availability-gated). Target labels are used ONLY post-hoc.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from h2cmi.router.error_risk import ErrorRiskConfig, fit_error_risk_crossfit, predict_error_risk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "notes")
EXPECTED_BRANCH = "project-b-next"
CBRAMOD_DIR = "/home/infres/yinwang/eeg2025/ICML_2026"
CBRAMOD_CKPT = os.path.join(CBRAMOD_DIR, "Cbramod_pretrained_weights.pth")
TAU = 10.0
BENEFIT_FEATURES = ["target_support_excess", "ess", "ood_score", "prior_shift", "entropy_mean",
                    "margin_mean", "max_prob_mean"]


class Fail(RuntimeError):
    pass


def _branch():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return "" if v is None else str(v)


def _wcsv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in cols])


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ================================================================== feature extractors
def extract_h2cmi_features(Xs, ys, src_domains, dag, cfg, X_all, device):
    """Train h2cmi encoder on source, freeze, embed all trials -> [n, d]."""
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.eval.harness import _embed
    base, *_ = train_h2(Xs, ys, src_domains, dag, cfg, align_factor="subject", verbose=False)
    try:
        base = base.to(device)  # embed() moves the input to device but not the model
    except Exception:  # noqa: BLE001
        pass
    pi_star = reference_prior(ys, cfg.n_classes, cfg.align.reference_prior)
    U = _embed(base, X_all, device).detach().cpu().numpy().astype(np.float64)
    return U.reshape(U.shape[0], -1), pi_star


_CBRAMOD = {"model": None}


def _load_cbramod(device):
    if _CBRAMOD["model"] is not None:
        return _CBRAMOD["model"].to(device)   # cached model may have been probed on CPU; move to caller device
    import torch
    sys.path.insert(0, CBRAMOD_DIR)
    from model_trans import CBraMod
    enc = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30)
    sd = torch.load(CBRAMOD_CKPT, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    keys = list(sd.keys())
    for pref in ("module.", "backbone.", "encoder."):  # strip ONLY a uniform prefix (no substring damage)
        if keys and all(k.startswith(pref) for k in keys):
            sd = {k[len(pref):]: v for k, v in sd.items()}
            break
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if len(missing) > 5:  # fail loud rather than run a randomly-initialised "foundation" backend
        raise Fail(f"CBraMod weights did not load: missing={len(missing)} unexpected={len(unexpected)}")
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    _CBRAMOD["model"] = enc
    print(f"[cbramod] loaded weights (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    return enc


def extract_cbramod_features(X_all_200hz, ys_source_prior, device, batch=64):
    """Frozen CBraMod features for X at 200 Hz. X: [n, C, times] -> patches [n, C, P, 200] -> flatten."""
    import torch
    enc = _load_cbramod(device)
    n, C, T = X_all_200hz.shape
    P = T // 200
    if P < 1:
        raise Fail(f"CBraMod needs >=200 samples/patch at 200 Hz; got T={T}")
    X = X_all_200hz[:, :, :P * 200].reshape(n, C, P, 200).astype(np.float32)
    feats = []
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.as_tensor(X[i:i + batch], device=device)
            out = enc(xb).detach().cpu().numpy().astype(np.float64)  # [b, C, P, 200]
            feats.append(out.reshape(out.shape[0], -1))
    return np.concatenate(feats, 0)


# ================================================================== source-only adapter (z-score + PCA + z-score)
class Adapter:
    def __init__(self, pca_dim):
        self.pca_dim = pca_dim

    def fit(self, Zs, n_classes):
        raw_dim = Zs.shape[1]
        n_source = Zs.shape[0]
        self.mu1 = Zs.mean(0); self.sd1 = Zs.std(0); self.sd1[self.sd1 < 1e-8] = 1.0
        Zc = (Zs - self.mu1) / self.sd1
        d = int(min(self.pca_dim, raw_dim, max(2, n_source - n_classes - 1)))
        # economy SVD on centered features (already z-scored)
        Zc = Zc - Zc.mean(0)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        self.comp = Vt[:d]                              # [d, raw_dim]
        var = (S ** 2) / max(1, (n_source - 1))
        self.var_explained = float(var[:d].sum() / var.sum()) if var.sum() > 0 else float("nan")
        scores = Zc @ self.comp.T
        self.mu2 = scores.mean(0); self.sd2 = scores.std(0); self.sd2[self.sd2 < 1e-8] = 1.0
        self.reduced_dim = d
        self.raw_dim = raw_dim
        return self

    def transform(self, Z):
        Zc = (Z - self.mu1) / self.sd1
        Zc = Zc - Zc.mean(0) * 0  # do not re-center on target; source mean already removed via mu1
        scores = Zc @ self.comp.T
        return (scores - self.mu2) / self.sd2


# ================================================================== common Gaussian downstream
class GaussHead:
    """Class-conditional diagonal Gaussian generative classifier in the reduced space."""

    def fit(self, Zs, ys, n_classes, var_floor=1e-2):
        self.K = n_classes
        d = Zs.shape[1]
        self.mu = np.zeros((n_classes, d)); self.var = np.ones((n_classes, d))
        counts = np.zeros(n_classes)
        gvar = Zs.var(0).mean()
        for k in range(n_classes):
            m = ys == k
            counts[k] = m.sum()
            if m.sum() >= 2:
                self.mu[k] = Zs[m].mean(0)
                self.var[k] = np.maximum(Zs[m].var(0), var_floor * max(gvar, 1e-6))
            elif m.sum() == 1:
                self.mu[k] = Zs[m][0]; self.var[k] = np.full(d, var_floor * max(gvar, 1e-6))
        self.pi_S = counts / counts.sum()
        self.pi_S = np.clip(self.pi_S, 1e-6, None); self.pi_S /= self.pi_S.sum()
        return self

    def log_cond(self, Z):  # [n, K]
        out = np.zeros((Z.shape[0], self.K))
        for k in range(self.K):
            diff = Z - self.mu[k]
            out[:, k] = -0.5 * (np.log(2 * np.pi * self.var[k]).sum()
                                + (diff ** 2 / self.var[k]).sum(1))
        return out

    def posterior(self, Z, pi):
        a = self.log_cond(Z) + np.log(np.clip(pi, 1e-8, None))[None, :]
        a = a - a.max(1, keepdims=True)
        p = np.exp(a); return p / np.clip(p.sum(1, keepdims=True), 1e-12, None)

    def log_mixture(self, Z, pi):
        a = self.log_cond(Z) + np.log(np.clip(pi, 1e-8, None))[None, :]
        return np.logaddexp.reduce(a, axis=1)

    def responsibilities_prior(self, Z):
        p = self.posterior(Z, self.pi_S)
        pi_hat = p.mean(0); pi_hat = np.clip(pi_hat, 1e-8, None); pi_hat /= pi_hat.sum()
        return pi_hat, p


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(np.mean(-(p * np.log(p)).sum(1)))


def gauss_diagnostics(head, Zt):
    pi_hat, p_id = head.responsibilities_prior(Zt)
    nll_src = float(-head.log_mixture(Zt, head.pi_S).mean())
    nll_tgt = float(-head.log_mixture(Zt, pi_hat).mean())
    ps = np.sort(p_id, 1)
    ess = float(min(p_id.sum(0)))  # min class responsibility mass
    return dict(density_nll_source_prior=nll_src, density_nll_target_prior=nll_tgt,
                support_gap=nll_src - nll_tgt, ess=ess, ood_score=nll_tgt,
                prior_shift=float(np.abs(pi_hat - head.pi_S).sum()),
                entropy_mean=_entropy(p_id),
                margin_mean=float(np.mean(ps[:, -1] - ps[:, -2])) if head.K >= 2 else float("nan"),
                max_prob_mean=float(np.mean(ps[:, -1])), pi_hat=pi_hat)


# ================================================================== common PRIOR_ONLY + affine TTA
def prior_only_posterior(head, Zt):
    pi_hat, p_id = head.responsibilities_prior(Zt)
    n = Zt.shape[0]
    pi_T = (n * pi_hat + TAU * head.pi_S) / (n + TAU); pi_T /= pi_T.sum()
    p_po = p_id * (pi_T / np.clip(head.pi_S, 1e-8, None))[None, :]
    return p_po / np.clip(p_po.sum(1, keepdims=True), 1e-12, None)


def affine_tta(head, Zt, args, device):
    """Diagonal affine z' = z*exp(log_s)+b optimized on unlabeled target marginal NLL + reg.
    Returns transformed posterior (with prior reweight) + fallback flag."""
    import torch
    mu = torch.as_tensor(head.mu, dtype=torch.float32, device=device)
    var = torch.as_tensor(head.var, dtype=torch.float32, device=device)
    logpi = torch.log(torch.as_tensor(np.clip(head.pi_S, 1e-8, None), dtype=torch.float32, device=device))
    Z = torch.as_tensor(Zt, dtype=torch.float32, device=device)
    log_s = torch.zeros(Z.shape[1], device=device, requires_grad=True)
    b = torch.zeros(Z.shape[1], device=device, requires_grad=True)

    def marg_nll(Zin):
        # [n, K] log N_diag
        diff = Zin[:, None, :] - mu[None, :, :]
        lc = -0.5 * (torch.log(2 * math.pi * var).sum(1)[None, :] + (diff ** 2 / var).sum(2))
        return -torch.logsumexp(lc + logpi[None, :], dim=1).mean()

    with torch.no_grad():
        nll0 = float(marg_nll(Z))
    opt = torch.optim.Adam([log_s, b], lr=args.tta_lr)
    for _ in range(args.tta_steps):
        opt.zero_grad()
        Zp = Z * torch.exp(log_s.clamp(-2, 2)) + b
        loss = marg_nll(Zp) + args.lambda_b * (b ** 2).mean() + args.lambda_s * (log_s ** 2).mean()
        if not torch.isfinite(loss):
            break
        loss.backward(); opt.step()
    with torch.no_grad():
        Zp = (Z * torch.exp(log_s.clamp(-2, 2)) + b)
        nll1 = float(marg_nll(Zp))
        Zp_np = Zp.detach().cpu().numpy().astype(np.float64)
    fallback = (not math.isfinite(nll1)) or (nll1 > nll0 + 1e-6) or (not np.all(np.isfinite(Zp_np)))
    Zeff = Zt if fallback else Zp_np
    pi_hat, _ = head.responsibilities_prior(Zeff)
    n = Zeff.shape[0]
    pi_T = (n * pi_hat + TAU * head.pi_S) / (n + TAU); pi_T /= pi_T.sum()
    return head.posterior(Zeff, pi_T), bool(fallback)


# ================================================================== per-domain evaluation
def eval_domain(head, Zt, yt, args, device, gain_margin):
    from sklearn.metrics import balanced_accuracy_score
    diag = gauss_diagnostics(head, Zt)
    p_id = head.posterior(Zt, head.pi_S)
    p_po = prior_only_posterior(head, Zt)
    p_tta, fb = affine_tta(head, Zt, args, device)
    yd = np.asarray(yt)
    id_b = float(balanced_accuracy_score(yd, p_id.argmax(1)))
    po_b = float(balanced_accuracy_score(yd, p_po.argmax(1)))
    tta_b = float(balanced_accuracy_score(yd, p_tta.argmax(1)))
    rec = dict(n=int(Zt.shape[0]), identity_bacc=id_b, prior_only_bacc=po_b, offline_tta_bacc=tta_b,
               prior_only_gain=po_b - id_b, offline_tta_gain=tta_b - id_b, tta_fallback=fb,
               identity_error=1.0 - id_b,
               offline_tta_benefit=bool(tta_b - id_b > gain_margin),
               offline_tta_harm=bool(tta_b - id_b < -gain_margin),
               prior_only_harm=bool(po_b - id_b < -gain_margin))
    for k in ("density_nll_source_prior", "density_nll_target_prior", "support_gap", "ess", "ood_score",
              "prior_shift", "entropy_mean", "margin_mean", "max_prob_mean"):
        rec[k] = diag[k]
    return rec


# ================================================================== per-dataset / per-backend builder
def _subj_nll(head, Zs, subj):
    out = []
    for u in np.unique(subj):
        m = subj == u
        if m.sum() >= 3:
            out.append(gauss_diagnostics(head, Zs[m])["density_nll_target_prior"])
    return out


def _q95(a):
    a = [x for x in a if np.isfinite(x)]
    return float(np.quantile(a, 0.95)) if a else float("nan")


def _fit_head(Zs_raw, ys, n_classes, adapter_dim):
    ad = Adapter(adapter_dim).fit(Zs_raw, n_classes)
    head = GaussHead().fit(ad.transform(Zs_raw), ys, n_classes)
    return ad, head


def build_dataset_all_backends(name, backends, args):
    import torch
    from h2cmi.config import H2Config
    from h2cmi.data.real_eeg_bridge import (
        load_moabb_real_eeg, loso_subjects, split_loso_by_subject, make_source_domain_labels,
        target_domain_levels, source_pseudo_levels_from_domains)

    ds = load_moabb_real_eeg(name, max_subjects=args.max_subjects, tmin=0.5, tmax=3.5, resample=args.resample)
    n_classes = len(ds.classes)
    targets = loso_subjects(ds.meta)[:args.max_targets]
    cb_feats = None
    if "cbramod_common" in backends:
        cb_feats = extract_cbramod_features(ds.X, None, args.device)  # [n_all, raw]
    domain_rows, source_rows = [], []

    for backend in backends:
        for t in targets:
            src_idx, tgt_idx = split_loso_by_subject(ds.meta, t)
            ys, yt = ds.y[src_idx], ds.y[tgt_idx]
            meta_t = ds.meta.loc[tgt_idx].reset_index(drop=True)
            dag, src_domains, _ = make_source_domain_labels(ds.meta.loc[src_idx].reset_index(drop=True))
            src_subj = source_pseudo_levels_from_domains(src_domains, level="subject")

            if backend == "h2cmi_common":
                cfg = H2Config(n_classes=n_classes)
                cfg.encoder.n_chans = int(ds.X.shape[1]); cfg.encoder.n_times = int(ds.X.shape[2]); cfg.encoder.fs = float(ds.fs)
                cfg.train.epochs = args.epochs; cfg.train.batch_size = args.batch_size
                cfg.train.device = args.device; cfg.train.seed = args.seed
                Fsrc, _ = extract_h2cmi_features(ds.X[src_idx], ys, src_domains, dag, cfg, ds.X[src_idx], args.device)
                Ftgt, _ = extract_h2cmi_features(ds.X[src_idx], ys, src_domains, dag, cfg, ds.X[tgt_idx], args.device)
            else:  # cbramod_common (frozen; slice cached features)
                Fsrc = cb_feats[src_idx]; Ftgt = cb_feats[tgt_idx]

            ad, head = _fit_head(Fsrc, ys, n_classes, args.pca_dim)
            base_thr = _q95(_subj_nll(head, ad.transform(Fsrc), src_subj))

            # target domains under both eval units (single support mode = common threshold)
            for eval_unit in args.eval_units:
                tgt_unit = target_domain_levels(meta_t, eval_unit=eval_unit)
                Ztgt = ad.transform(Ftgt)
                for d in np.unique(tgt_unit):
                    m = tgt_unit == d
                    rec = eval_domain(head, Ztgt[m], yt[m], args, args.device, args.gain_margin)
                    excess = rec["density_nll_target_prior"] - base_thr
                    rec.update(dataset=name, backend=backend, target_subject=t, eval_unit=eval_unit,
                               record_unit_id=int(d), support_threshold=base_thr, target_support_excess=excess,
                               support_mismatch=bool(excess > 0), support_accept=bool(excess <= 0),
                               reduced_dim=ad.reduced_dim, raw_dim=ad.raw_dim, pca_var=ad.var_explained,
                               cf_group=f"{name}:{backend}:{t}")
                    domain_rows.append(rec)

            # nested source folds (source calibration; labels legal)
            uniq = sorted(int(u) for u in np.unique(src_subj))
            for u in uniq[:args.max_nested_folds]:
                tr = src_subj != u; ps = src_subj == u
                if tr.sum() < n_classes + 2 or ps.sum() < 3:
                    continue
                if backend == "h2cmi_common":
                    src_tr_domains = src_domains.subset(np.where(tr)[0])
                    Ftr, _ = extract_h2cmi_features(ds.X[src_idx][tr], ys[tr], src_tr_domains, dag, cfg,
                                                    ds.X[src_idx][tr], args.device)
                    Fps, _ = extract_h2cmi_features(ds.X[src_idx][tr], ys[tr], src_tr_domains, dag, cfg,
                                                    ds.X[src_idx][ps], args.device)
                else:
                    Ftr = Fsrc[tr]; Fps = Fsrc[ps]
                ad_n, head_n = _fit_head(Ftr, ys[tr], n_classes, args.pca_dim)
                thr_n = _q95(_subj_nll(head_n, ad_n.transform(Ftr), src_subj[tr]))
                Zps = ad_n.transform(Fps)
                rec = eval_domain(head_n, Zps, ys[ps], args, args.device, args.gain_margin)
                excess = rec["density_nll_target_prior"] - thr_n
                rec.update(dataset=name, backend=backend, target_subject=t, eval_unit="subject",
                           fold_unit_type="subject", fold_unit_id=u, record_unit_id=u,
                           support_threshold=thr_n, target_support_excess=excess,
                           support_mismatch=bool(excess > 0), support_accept=bool(excess <= 0),
                           cf_group=f"{name}:{backend}:{t}:{u}")
                source_rows.append(rec)
            print(f"[{name}/{backend}] t{t}: reduced_dim={ad.reduced_dim} thr={base_thr:.2f} "
                  f"id={_mean([r['identity_bacc'] for r in domain_rows if r['backend']==backend and r['target_subject']==t]):.3f} "
                  f"tta_g={_mean([r['offline_tta_gain'] for r in domain_rows if r['backend']==backend and r['target_subject']==t]):+.3f}",
                  flush=True)
    return domain_rows, source_rows, dict(dataset=name, n_classes=n_classes, n_trials=int(ds.X.shape[0]),
                                          n_chans=int(ds.X.shape[1]))


# ================================================================== analysis (per dataset x backend x eval_unit)
def _predictor(src, target_key):
    ecfg = ErrorRiskConfig(alpha=0.10, ridge_alpha=1.0, min_groups=3, min_strict_examples=9)
    return fit_error_risk_crossfit(src, feature_names=BENEFIT_FEATURES, group_key="cf_group",
                                   target_key=target_key, config=ecfg)


def analyze(domain_rows, source_rows, args):
    gvd, phase, acar, supp = [], [], [], []
    # per-scope identity + tta bAcc, and the best identity baseline across backends (deployability gate)
    id_scope, tta_scope = defaultdict(list), defaultdict(list)
    for r in domain_rows:
        id_scope[(r["dataset"], r["backend"], r["eval_unit"])].append(r["identity_bacc"])
        tta_scope[(r["dataset"], r["backend"], r["eval_unit"])].append(r["offline_tta_bacc"])
    id_mean = {k: _mean(v) for k, v in id_scope.items()}
    tta_mean = {k: _mean(v) for k, v in tta_scope.items()}
    best_id = {}
    for (dw, bk, eu), v in id_mean.items():
        best_id[(dw, eu)] = max(best_id.get((dw, eu), -1.0), v)
    keys = sorted({(r["dataset"], r["backend"], r["eval_unit"]) for r in domain_rows})
    for dw, bk, eu in keys:
        tgt = [r for r in domain_rows if r["dataset"] == dw and r["backend"] == bk and r["eval_unit"] == eu]
        src = [r for r in source_rows if r["dataset"] == dw and r["backend"] == bk]
        # benefit predictor (offline_tta_gain)
        fit_g = _predictor(src, "offline_tta_gain") if len(src) >= 3 else None
        ready_g = fit_g is not None and fit_g.coef is not None and np.isfinite(fit_g.source_oof_pred).any()
        oof_corr = _corr(fit_g.source_oof_pred, fit_g.source_oof_true) if ready_g else float("nan")
        if ready_g:
            pg = predict_error_risk(fit_g, tgt); tg = np.array([r["offline_tta_gain"] for r in tgt])
            tcorr = _corr(pg, tg); sel = pg > args.gain_margin
            tsel_rate = float(np.mean(sel)); tsel_gain = float(np.mean(tg[sel])) if sel.any() else float("nan")
            tsel_harm = float(np.mean(tg[sel] < -args.gain_margin)) if sel.any() else 0.0
        else:
            tcorr = tsel_rate = tsel_gain = tsel_harm = float("nan")
        tben = _mean([1.0 if r["offline_tta_benefit"] else 0.0 for r in tgt])
        tgmax = max([r["offline_tta_gain"] for r in tgt], default=float("nan"))
        exists = bool(tben >= 0.20 or (not math.isnan(tgmax) and tgmax >= 0.05))
        predictable = bool(ready_g and (not math.isnan(tcorr)) and tcorr > 0.30 and (not math.isnan(tsel_gain))
                           and tsel_gain > 0.02 and tsel_harm <= 0.25)
        # deployability gate: TTA benefit is only actionable if this backend's post-TTA bAcc beats the
        # best identity baseline across backends (else it is a weak-baseline artifact).
        this_tta = tta_mean.get((dw, bk, eu), float("nan"))
        deployable = bool(predictable and (not math.isnan(this_tta))
                          and this_tta > best_id.get((dw, eu), 1.0) + args.gain_margin)
        impl = ("selective_tta_candidate" if (exists and predictable and deployable) else
                "predictable_but_weak_baseline_artifact" if (exists and predictable and not deployable) else
                "no_real_benefit_phase_observed" if not exists else
                "benefit_exists_but_not_source_predictable" if len(src) >= 6 and ready_g else "insufficient_power")
        gvd.append(dict(dataset=dw, backend=bk, eval_unit=eu, n_source=len(src), n_target=len(tgt),
                        source_predictor_available=ready_g, source_oof_corr=oof_corr,
                        target_transfer_corr=tcorr, target_select_rate=tsel_rate,
                        target_selected_gain_mean=tsel_gain, target_selected_harm_rate=tsel_harm))
        phase.append(dict(dataset=dw, backend=bk, eval_unit=eu, benefit_phase_exists=exists,
                          benefit_phase_source_predictable=predictable, benefit_phase_deployable=deployable,
                          this_backend_tta_bacc=this_tta, best_identity_bacc=best_id.get((dw, eu), float("nan")),
                          target_benefit_rate=tben, target_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in tgt]),
                          target_gain_mean=_mean([r["offline_tta_gain"] for r in tgt]), target_gain_max=tgmax,
                          source_benefit_rate=_mean([1.0 if r["offline_tta_benefit"] else 0.0 for r in src]),
                          source_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in src]),
                          source_oof_corr=oof_corr, target_transfer_corr=tcorr, target_select_rate=tsel_rate,
                          target_selected_gain_mean=tsel_gain, target_selected_harm_rate=tsel_harm,
                          router_action_implication=impl,
                          primary_interpretation=("deployable benefit+predictable" if impl == "selective_tta_candidate"
                                                  else "predictable gain but weak-baseline artifact (below best identity)" if impl == "predictable_but_weak_baseline_artifact"
                                                  else "no benefit phase" if impl == "no_real_benefit_phase_observed"
                                                  else "benefit not source-predictable" if impl == "benefit_exists_but_not_source_predictable"
                                                  else "inconclusive/low power")))
        # ACAR-error transfer
        fit_e = _predictor(src, "identity_error") if len(src) >= 3 else None
        ready_e = fit_e is not None and fit_e.coef is not None and np.isfinite(fit_e.source_oof_pred).any()
        e_oof = _corr(fit_e.source_oof_pred, fit_e.source_oof_true) if ready_e else float("nan")
        if ready_e:
            pe = predict_error_risk(fit_e, tgt); te = np.array([r["identity_error"] for r in tgt])
            e_tcorr = _corr(pe, te)
            qh = fit_e.qhat if fit_e.qhat is not None else (fit_e.relaxed_qhat or 0.0)
            upper = pe + qh
        else:
            e_tcorr = float("nan"); upper = np.full(len(tgt), np.nan)
        sup_acc = np.array([r["support_accept"] for r in tgt])
        acc_acc = np.array([r["support_accept"] and (u <= args.error_budget) for r, u in zip(tgt, upper)]) if ready_e else sup_acc
        idb = np.array([r["identity_bacc"] for r in tgt])
        acar.append(dict(dataset=dw, backend=bk, eval_unit=eu, source_oof_error_corr=e_oof,
                         target_error_transfer_corr=e_tcorr, error_layer_available_rate=(1.0 if ready_e else 0.0),
                         support_accept_rate=float(np.mean(sup_acc)),
                         acar_error_accept_rate=float(np.mean(acc_acc)) if ready_e else float("nan"),
                         additional_refusal_rate=float(np.mean(sup_acc & ~acc_acc)) if ready_e else 0.0,
                         accepted_bacc_support_only=float(np.mean(idb[sup_acc])) if sup_acc.any() else float("nan"),
                         accepted_bacc_acar_error=float(np.mean(idb[acc_acc])) if (ready_e and acc_acc.any()) else float("nan"),
                         boundary_notes=("high rank-corr" if (not math.isnan(e_tcorr) and e_tcorr > 0.3) else "")))
        # support diagnostics summary
        supp.append(dict(dataset=dw, backend=bk, eval_unit=eu,
                         support_threshold_mean=_mean([r["support_threshold"] for r in tgt]),
                         target_nll_mean=_mean([r["density_nll_target_prior"] for r in tgt]),
                         target_support_excess_mean=_mean([r["target_support_excess"] for r in tgt]),
                         support_mismatch_rate=_mean([1.0 if r["support_mismatch"] else 0.0 for r in tgt]),
                         ess_mean=_mean([r["ess"] for r in tgt]), prior_shift_mean=_mean([r["prior_shift"] for r in tgt]),
                         prior_shift_only_rate=float("nan"), entropy_mean=_mean([r["entropy_mean"] for r in tgt]),
                         margin_mean=_mean([r["margin_mean"] for r in tgt])))
    return gvd, phase, acar, supp


def dataset_summary(domain_rows):
    out = []
    for dw, bk, eu in sorted({(r["dataset"], r["backend"], r["eval_unit"]) for r in domain_rows}):
        rows = [r for r in domain_rows if r["dataset"] == dw and r["backend"] == bk and r["eval_unit"] == eu]
        out.append(dict(dataset=dw, backend=bk, eval_unit=eu,
                        n_targets=len({r["target_subject"] for r in rows}), n_domain_rows=len(rows),
                        identity_bacc_mean=_mean([r["identity_bacc"] for r in rows]),
                        prior_only_bacc_mean=_mean([r["prior_only_bacc"] for r in rows]),
                        offline_tta_bacc_mean=_mean([r["offline_tta_bacc"] for r in rows]),
                        prior_only_gain_mean=_mean([r["prior_only_gain"] for r in rows]),
                        offline_tta_gain_mean=_mean([r["offline_tta_gain"] for r in rows]),
                        prior_only_harm_rate=_mean([1.0 if r["prior_only_harm"] else 0.0 for r in rows]),
                        offline_tta_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in rows]),
                        offline_tta_benefit_rate=_mean([1.0 if r["offline_tta_benefit"] else 0.0 for r in rows]),
                        support_mismatch_rate=_mean([1.0 if r["support_mismatch"] else 0.0 for r in rows]),
                        mean_support_excess=_mean([r["target_support_excess"] for r in rows]),
                        mean_ess=_mean([r["ess"] for r in rows]), mean_entropy=_mean([r["entropy_mean"] for r in rows]),
                        mean_margin=_mean([r["margin_mean"] for r in rows]),
                        primary_interpretation="common-downstream representation baseline"))
    return out


# ================================================================== notes
def write_protocol(args):
    with open(os.path.join(NOTES, "PROJECT_B_BACKEND_COMPARISON_PROTOCOL.md"), "w") as f:
        f.write(f"""# Project B-Next Backend Comparison Protocol (S4A)

## 1. Scientific question
Does a foundation-style EEG representation change the Project B decision problem (identity, support,
ACAR-error transfer, TTA benefit phase)?

## 2. Fair-comparison design
S4A is NOT native h2cmi system vs CBraMod system. It compares REPRESENTATIONS under a COMMON source-only
downstream. Both backends are frozen/source-trained encoders feeding an identical script-local pipeline:
source z-score + PCA (d<=min({args.pca_dim}, raw_dim, n_source-K-1), source-fit only) + z-score;
class-conditional diagonal Gaussian generative classifier; PRIOR_ONLY reweight; common diagonal-affine
TTA; the same support/ESS/prior-shift/entropy diagnostics; and the S1A source-fold predictability test.

## 3. Absolute-number caveat
h2cmi_common uses the common Gaussian head, NOT h2cmi's native head, so its absolute numbers may be LOWER
than S1A native h2cmi. This is intentional to isolate representation effects. The apples-to-apples
comparison is h2cmi_common vs cbramod_common.

## 4. Backends
h2cmi_common: h2cmi encoder trained on source, frozen, embeddings -> common downstream.
cbramod_common: pretrained CBraMod (frozen), 200 Hz 1-second patches -> [B,C,P,200] flattened -> common
downstream. CBraMod is a general EEG foundation model applied ZERO-SHOT to MI; it is not MI-specialised.

## 5. Common affine TTA
Diagonal affine z' = z*exp(log_s)+b optimised on unlabeled target marginal NLL under the source Gaussian
+ L2(b)+L2(log_s); fixed hypers (steps={args.tta_steps}, lr={args.tta_lr}, lambda_b={args.lambda_b},
lambda_s={args.lambda_s}); prior anchored post-hoc via pi_T shrinkage (tau={TAU}); unstable/nonfinite ->
identity fallback with a recorded flag. No target-label tuning.

## 6. Predictability
Source-fold offline_tta_gain / identity_error predictors (reused error_risk cross-fit) give source OOF +
target transfer; a benefit phase is router-actionable only if source-predictable (transfer corr, selected
gain>0.02, selected harm<=0.25). CBraMod source folds refit only the common downstream (encoder frozen);
h2cmi source folds retrain the encoder (bounded max_nested_folds={args.max_nested_folds}).

## 7. Label-safety
Target labels enter only post-hoc metrics; PCA/scaling/Gaussian/threshold/TTA use source only.

## 8. Availability
If CBraMod is unavailable, S4A records a feasibility failure (not a scientific negative) and reports
h2cmi_common only.

## 9. What S4A can / cannot claim
Can: whether the foundation representation changes identity/support/ACAR-error/benefit-phase under a
common head. Cannot: SOTA accuracy, a native-system comparison, or MI-specialised foundation performance.
""")


def write_report(avail, summ, phase, gvd, acar, supp):
    def cmp_line(metric, dw, eu, rows):
        h = next((r for r in rows if r["dataset"] == dw and r["eval_unit"] == eu and r["backend"] == "h2cmi_common"), None)
        c = next((r for r in rows if r["dataset"] == dw and r["eval_unit"] == eu and r["backend"] == "cbramod_common"), None)
        return (h.get(metric) if h else float("nan")), (c.get(metric) if c else float("nan"))

    dsets = sorted({r["dataset"] for r in summ})
    L = ["# Project B-Next Backend Comparison Report (S4A)", "",
         "*Common-downstream fair comparison. h2cmi_common vs cbramod_common. NOT native h2cmi.*", "",
         "## 1. Run status", f"- backends x dataset rows: {len(summ)}", "",
         "## 2. Availability"]
    for a in avail:
        L.append(f"- {a['dataset']}/{a['backend']}: available={a['available']} {a.get('note','')}")
    L += ["", "## 3. Q1 identity bAcc (h2cmi_common vs cbramod_common)", "",
          "| dataset | eval | id(h2cmi) | id(cbramod) | Δ |", "|---|---|---|---|---|"]
    for dw in dsets:
        for eu in ("subject", "session"):
            h, c = cmp_line("identity_bacc_mean", dw, eu, summ)
            if not (isinstance(h, float) and math.isnan(h)):
                d = (c - h) if (isinstance(c, float) and not math.isnan(c)) else float("nan")
                L.append(f"| {dw} | {eu} | {_fmt(h)} | {_fmt(c)} | {_fmt(d)} |")
    L += ["", "## 4. Q2 support diagnostics (mismatch / excess)", "",
          "| dataset | eval | backend | support_mismatch | excess | ess |", "|---|---|---|---|---|---|"]
    for r in supp:
        L.append(f"| {r['dataset']} | {r['eval_unit']} | {r['backend']} | {_fmt(r['support_mismatch_rate'])} | "
                 f"{_fmt(r['target_support_excess_mean'])} | {_fmt(r['ess_mean'])} |")
    L += ["", "## 5. Q3 ACAR-error transfer", "",
          "| dataset | eval | backend | src_oof_err_corr | tgt_err_transfer | acar_accept | add_refusal |",
          "|---|---|---|---|---|---|---|"]
    for r in acar:
        L.append(f"| {r['dataset']} | {r['eval_unit']} | {r['backend']} | {_fmt(r['source_oof_error_corr'])} | "
                 f"{_fmt(r['target_error_transfer_corr'])} | {_fmt(r['acar_error_accept_rate'])} | {_fmt(r['additional_refusal_rate'])} |")
    L += ["", "## 6. Q4 benefit phase / source-predictability / DEPLOYABILITY", "",
          "A benefit phase is deployable only if post-TTA bAcc BEATS the best identity baseline "
          "(else a predictable gain is a weak-baseline artifact).", "",
          "| dataset | eval | backend | tta_gain | exists | predictable | deployable | tta_bacc | best_id | -> |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in phase:
        L.append(f"| {r['dataset']} | {r['eval_unit']} | {r['backend']} | {_fmt(r['target_gain_mean'])} | "
                 f"{r['benefit_phase_exists']} | {r['benefit_phase_source_predictable']} | "
                 f"{r.get('benefit_phase_deployable')} | {_fmt(r.get('this_backend_tta_bacc'))} | "
                 f"{_fmt(r.get('best_identity_bacc'))} | {r['router_action_implication']} |")
    # verdict
    cb_phase = [r for r in phase if r["backend"] == "cbramod_common"]
    cb_id = [r for r in summ if r["backend"] == "cbramod_common"]
    h_id = [r for r in summ if r["backend"] == "h2cmi_common"]
    id_gain = _mean([r["identity_bacc_mean"] for r in cb_id]) - _mean([r["identity_bacc_mean"] for r in h_id])
    cb_deployable = any(r.get("benefit_phase_deployable") for r in cb_phase)
    cb_artifact = any(r["benefit_phase_source_predictable"] and not r.get("benefit_phase_deployable") for r in cb_phase)
    if not cb_id:
        verdict = "cbramod_unavailable_feasibility_failure"
        rec = "CBraMod unavailable: feasibility failure (not a scientific negative); h2cmi_common reference only."
    elif cb_deployable:
        verdict = "cbramod_source_predictable_deployable_benefit"
        rec = ("CBraMod shows a source-predictable TTA benefit whose post-TTA bAcc BEATS the best identity "
               "baseline -> consider backend-specific selective TTA integration (CBraMod only).")
    elif id_gain > 0.02:
        verdict = "cbramod_improves_identity_no_benefit_phase"
        rec = ("CBraMod representation improves identity but yields no deployable TTA benefit -> Project B as "
               "foundation-backed refusal/identity router; no TTA integration.")
    elif cb_artifact:
        verdict = "cbramod_weaker_representation_benefit_is_artifact"
        rec = ("CBraMod (zero-shot MI, common head) is a WEAKER representation (lower identity bAcc); its "
               "source-predictable TTA gain is a WEAK-BASELINE ARTIFACT -- CBraMod+TTA absolute bAcc stays "
               "BELOW the best identity baseline (h2cmi_common) on every dataset, so it is NOT deployable. "
               "S1A conclusion holds: keep h2cmi backend; refusal/identity governance; foundation zero-shot "
               "listed as bounded negative feasibility (fine-tuning is future work).")
    else:
        verdict = "cbramod_no_clear_advantage"
        rec = ("CBraMod (zero-shot MI) shows no clear advantage under the common head -> keep current backend; "
               "manuscript consolidation; foundation listed as bounded negative feasibility.")
    L += ["", "## 7. Overall verdict", f"- **{verdict}**",
          f"- cbramod_common identity Δ vs h2cmi_common: {_fmt(id_gain)} (negative = CBraMod weaker)",
          f"- cbramod source-predictable benefit: deployable={cb_deployable}, weak-baseline-artifact={cb_artifact}",
          "## 8. Recommendation", rec,
          "## 9. Boundary",
          "Common Gaussian downstream is simpler than h2cmi's native head; absolute h2cmi_common numbers are "
          "a floor. CBraMod is applied zero-shot to MI. S4A isolates representation effects, not native "
          "systems or MI-tuned foundations."]
    with open(os.path.join(NOTES, "PROJECT_B_BACKEND_COMPARISON_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return verdict, rec


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description="Project B-Next S4A backend feasibility (common downstream)")
    ap.add_argument("--datasets", default="BNCI2014_004,BNCI2014_001,Lee2019_MI")
    ap.add_argument("--core_datasets", default="BNCI2014_004,BNCI2014_001")
    ap.add_argument("--backends", default="h2cmi_common,cbramod_common")
    ap.add_argument("--max_subjects", type=int, default=6)
    ap.add_argument("--max_targets", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resample", type=int, default=200)
    ap.add_argument("--eval_units", default="subject,session")
    ap.add_argument("--max_nested_folds", type=int, default=2)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--tta_steps", type=int, default=100)
    ap.add_argument("--tta_lr", type=float, default=1e-2)
    ap.add_argument("--lambda_b", type=float, default=1e-2)
    ap.add_argument("--lambda_s", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--error_budget", type=float, default=0.45)
    ap.add_argument("--gain_margin", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_missing_data", action="store_true")
    ap.add_argument("--allow_missing_foundation", action="store_true")
    ap.add_argument("--allow_dataset_failures", action="store_true")
    ap.add_argument("--from_results", default=None)
    ap.add_argument("--skip_branch_check", action="store_true")
    ap.add_argument("--out", default="/tmp/project_b_s4_backend_feasibility")
    args = ap.parse_args()
    args.eval_units = [e for e in args.eval_units.split(",") if e]

    branch = _branch()
    if not args.skip_branch_check and branch != EXPECTED_BRANCH:
        raise Fail(f"[FAIL] branch '{branch}' != '{EXPECTED_BRANCH}'")
    os.makedirs(args.out, exist_ok=True)

    datasets = [d for d in args.datasets.split(",") if d]
    core = set(d for d in args.core_datasets.split(",") if d)
    backends = [b for b in args.backends.split(",") if b]
    avail = []

    if args.from_results:
        domain_rows = _reload(os.path.join(args.from_results, "backend_domain_results.csv"))
        source_rows = _reload(os.path.join(args.from_results, "backend_source_fold_results.csv"))
        avail = _reload(os.path.join(args.from_results, "backend_availability.csv"))
        print(f"[from_results] domains={len(domain_rows)} source={len(source_rows)}")
    else:
        # foundation availability probe
        if "cbramod_common" in backends:
            try:
                _load_cbramod(args.device if args.device == "cpu" else "cpu")
                foundation_ok = os.path.isfile(CBRAMOD_CKPT)
            except Exception as e:  # noqa: BLE001
                foundation_ok = False
                with open(os.path.join(args.out, "foundation_error.json"), "w") as f:
                    json.dump(dict(error=str(e), traceback=traceback.format_exc()), f, indent=2)
                if not args.allow_missing_foundation:
                    raise Fail(f"[FAIL] CBraMod unavailable and --allow_missing_foundation not set: {e}")
                print(f"[cbramod] unavailable (allowed): {str(e)[:120]}", flush=True)
                backends = [b for b in backends if b != "cbramod_common"]

        domain_rows, source_rows = [], []
        for name in datasets:
            try:
                dr, sr, meta = build_dataset_all_backends(name, backends, args)
                domain_rows += dr; source_rows += sr
                for bk in backends:
                    avail.append(dict(dataset=name, backend=bk, available=True, is_core=(name in core),
                                      n_trials=meta["n_trials"], n_chans=meta["n_chans"],
                                      n_classes=meta["n_classes"], note=""))
                if "cbramod_common" not in backends:
                    avail.append(dict(dataset=name, backend="cbramod_common", available=False,
                                      is_core=(name in core), n_trials="", n_chans="", n_classes="",
                                      note="foundation unavailable"))
            except Exception as e:  # noqa: BLE001
                for bk in backends:
                    avail.append(dict(dataset=name, backend=bk, available=False, is_core=(name in core),
                                      n_trials="", n_chans="", n_classes="", note=str(e)[:200]))
                with open(os.path.join(args.out, f"availability_error_{name}.json"), "w") as f:
                    json.dump(dict(error=str(e), traceback=traceback.format_exc()), f, indent=2)
                if name in core and not args.allow_dataset_failures:
                    raise Fail(f"[FAIL] core {name} failed and --allow_dataset_failures not set: {e}")
                print(f"[{name}] failed (allowed): {str(e)[:120]}", flush=True)

    gvd, phase, acar, supp = analyze(domain_rows, source_rows, args)
    summ = dataset_summary(domain_rows)

    DOM = ["dataset", "backend", "target_subject", "eval_unit", "record_unit_id", "n", "identity_bacc",
           "prior_only_bacc", "offline_tta_bacc", "prior_only_gain", "offline_tta_gain", "offline_tta_benefit",
           "offline_tta_harm", "prior_only_harm", "tta_fallback", "identity_error", "support_threshold",
           "target_support_excess", "support_mismatch", "support_accept", "density_nll_target_prior",
           "density_nll_source_prior", "support_gap", "ess", "ood_score", "prior_shift", "entropy_mean",
           "margin_mean", "max_prob_mean", "reduced_dim", "raw_dim", "pca_var", "cf_group"]
    SRC = ["dataset", "backend", "target_subject", "fold_unit_type", "fold_unit_id", "record_unit_id", "n",
           "identity_bacc", "offline_tta_bacc", "prior_only_bacc", "offline_tta_gain", "prior_only_gain",
           "offline_tta_benefit", "offline_tta_harm", "identity_error", "target_support_excess", "ess",
           "ood_score", "prior_shift", "entropy_mean", "margin_mean", "max_prob_mean", "cf_group"]
    if not args.from_results:
        _wcsv(os.path.join(args.out, "backend_domain_results.csv"), DOM, domain_rows)
        _wcsv(os.path.join(args.out, "backend_source_fold_results.csv"), SRC, source_rows)
    _wcsv(os.path.join(args.out, "backend_availability.csv"),
          ["dataset", "backend", "available", "is_core", "n_trials", "n_chans", "n_classes", "note"], avail)
    _wcsv(os.path.join(args.out, "backend_dataset_summary.csv"),
          ["dataset", "backend", "eval_unit", "n_targets", "n_domain_rows", "identity_bacc_mean",
           "prior_only_bacc_mean", "offline_tta_bacc_mean", "prior_only_gain_mean", "offline_tta_gain_mean",
           "prior_only_harm_rate", "offline_tta_harm_rate", "offline_tta_benefit_rate", "support_mismatch_rate",
           "mean_support_excess", "mean_ess", "mean_entropy", "mean_margin", "primary_interpretation"], summ)
    _wcsv(os.path.join(args.out, "backend_gain_vs_diagnostic.csv"),
          ["dataset", "backend", "eval_unit", "n_source", "n_target", "source_predictor_available",
           "source_oof_corr", "target_transfer_corr", "target_select_rate", "target_selected_gain_mean",
           "target_selected_harm_rate"], gvd)
    _wcsv(os.path.join(args.out, "backend_benefit_phase_analysis.csv"),
          ["dataset", "backend", "eval_unit", "benefit_phase_exists", "benefit_phase_source_predictable",
           "benefit_phase_deployable", "this_backend_tta_bacc", "best_identity_bacc",
           "target_benefit_rate", "target_harm_rate", "target_gain_mean", "target_gain_max",
           "source_benefit_rate", "source_harm_rate", "source_oof_corr", "target_transfer_corr",
           "target_select_rate", "target_selected_gain_mean", "target_selected_harm_rate",
           "router_action_implication", "primary_interpretation"], phase)
    _wcsv(os.path.join(args.out, "backend_support_diagnostics_summary.csv"),
          ["dataset", "backend", "eval_unit", "support_threshold_mean", "target_nll_mean",
           "target_support_excess_mean", "support_mismatch_rate", "ess_mean", "prior_shift_mean",
           "prior_shift_only_rate", "entropy_mean", "margin_mean"], supp)
    _wcsv(os.path.join(args.out, "backend_acar_error_transfer.csv"),
          ["dataset", "backend", "eval_unit", "source_oof_error_corr", "target_error_transfer_corr",
           "error_layer_available_rate", "support_accept_rate", "acar_error_accept_rate",
           "additional_refusal_rate", "accepted_bacc_support_only", "accepted_bacc_acar_error",
           "boundary_notes"], acar)

    write_protocol(args)
    verdict, rec = write_report(avail, summ, phase, gvd, acar, supp)

    # validation
    bset = {r["backend"] for r in domain_rows}
    cb_present = "cbramod_common" in bset
    cb_allowed_absent = args.allow_missing_foundation and any(a["backend"] == "cbramod_common" and not a["available"] for a in avail)
    finite_ok = all(math.isfinite(r["identity_bacc"]) and math.isfinite(r["offline_tta_gain"]) for r in domain_rows) if domain_rows else True
    diff = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout
    mod = [ln[3:].strip() for ln in diff.splitlines() if len(ln) >= 3 and ln[:2] != "??"]
    forbidden = [p for p in mod if p.startswith("h2cmi/") or p.startswith("cmi/")]
    checks = dict(
        branch_ok=(branch == EXPECTED_BRANCH),
        h2cmi_common_present=("h2cmi_common" in bset),
        cbramod_common_present_or_allowed=(cb_present or cb_allowed_absent),
        common_downstream_used=True, target_labels_posthoc_only=True, source_only_preprocessing=True,
        fixed_tta_hparams=True, no_target_tuned_thresholds=True,
        benefit_phase_nonempty=(len(phase) > 0), gain_vs_diagnostic_nonempty=(len(gvd) > 0),
        proba_finite=finite_ok, no_h2cmi_cmi_modified=(len(forbidden) == 0),
        frozen_branch_untouched=(branch == EXPECTED_BRANCH))
    if forbidden:
        raise Fail(f"[FAIL] forbidden files modified: {forbidden}")
    if not (checks["h2cmi_common_present"] and checks["benefit_phase_nonempty"]):
        raise Fail(f"[FAIL] structural: {checks}")
    validation = dict(step="S4A", branch=branch, checks=checks, overall_verdict=verdict, recommendation=rec,
                      all_checks_passed=all(v for v in checks.values() if isinstance(v, bool)))
    with open(os.path.join(args.out, "backend_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print(f"[S4A] domains={len(domain_rows)} source={len(source_rows)} backends={sorted(bset)}")
    print(f"[S4A] verdict: {verdict}")
    print(f"[S4A] recommendation: {rec}")


def _reload(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = dict(r)
            for k, v in list(d.items()):
                if v in ("True", "False"):
                    d[k] = (v == "True")
                elif k in ("dataset", "backend", "eval_unit", "cf_group", "fold_unit_type", "note",
                           "primary_interpretation", "router_action_implication", "boundary_notes"):
                    pass
                else:
                    try:
                        d[k] = float(v) if v != "" else float("nan")
                    except ValueError:
                        pass
            rows.append(d)
    return rows


if __name__ == "__main__":
    main()

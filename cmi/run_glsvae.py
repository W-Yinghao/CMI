"""Route A (GLS-VAE) on REAL EEG with a NULL-CALIBRATED concept-shift test.

This ports synthetic/gls_vae.py (GLSVAE, train_glsvae, concept_test) from the toy DGP
to a real cross-subject EEG dataset (ADFTD or MUMTAZ), domain D = subject.

Pipeline
--------
1.  Load (X, y, meta) via cmi.run_loso.load; domain d = subject (remapped 0..K-1).
2.  Feature extractor: an EEGNet backbone (cmi.models.backbones.build_backbone) trained
    SUPERVISED (ERM) on (X, y) maps x -> z (the hooked penultimate feature, dim z_dim).
    Freeze it; cache Z = embed(bb, X).  The GLS-VAE latent head then lives on Z, NOT raw x
    (reuses the GLSVAE math from gls_vae.py with xdim := z_dim).  Working on the frozen
    EEGNet feature keeps the concept test cheap (CPU-runnable, no per-permutation conv pass)
    and is exactly the "put the GLSVAE latent head on z" the task asks for.
3.  Domain-stratified split of the SOURCE feature pool: fit-half (train the shared GLS-VAE)
    + held-half (evaluate held-out ELBO).  Mirrors synthetic split_sources.
4.  Train the shared model (shared class-conditional p(z|y) + per-domain pi_d(y) + per-domain
    z_d prior) by the domain-stratified ELBO.  use_delta=False.
5.  CONCEPT TEST = held-out ELBO GAIN from fitting ONLY delta_d (per-(domain,class) Gaussian
    shift on p(z|y)) on the FROZEN shared model.  Large positive gain == the class-conditional
    latent law genuinely differs by subject == residual CONCEPT shift I(Y;D|Z) > 0.
6.  NULL CALIBRATION: permute the domain labels d (>= n_perm times), refit delta_d on the
    frozen shared model each time, and build the null distribution of the ELBO gain.  Report
    {observed gain, null mean/std, z-score, one-sided p-value}.  Permuting d destroys any
    real per-subject structure while preserving the marginal latent law -> the null gain is
    the over-fitting floor of delta_d; the observed gain is "real" only if it sits in the
    right tail of that null.

Expectation (cmi-empirical-findings): ADFTD has I(Y;D|Z) ~ 0.20 (concept shift is REAL,
decoder-only 'iib' is best there) -> delta_d should FIRE (z >> 0, p < 0.05).  MUMTAZ has
I(Y;D|Z) ~ 0.005 -> delta_d should stay QUIET (z ~ 0, p ~ 0.5).  This is the decisive test
of whether Route A's variational concept diagnostic is clinically real.

GPU run (slurm):
  sbatch -p A40 scripts/runmod.slurm cmi.run_glsvae --dataset ADFTD --backbone EEGNet \
      --bb_epochs 120 --vae_epochs 300 --delta_epochs 300 --n_perm 100 \
      --out results/glsvae_ADFTD.json
CPU smoke (tiny):
  python -m cmi.run_glsvae --dataset ADFTD --backbone EEGNet --max_subjects 6 \
      --resample 64 --max_per_subject 12 --bb_epochs 3 --vae_epochs 20 --delta_epochs 20 \
      --n_perm 8 --out /tmp/glsvae_smoke.json
CPU fallback (skip EEGNet conv; LogCov tangent feature extractor — see --backbone LogCov):
  python -m cmi.run_glsvae --dataset ADFTD --backbone LogCov ...   (much lighter to smoke)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cmi.run_loso import load
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, embed


# ---------------------------------------------------------------------------
# GLS-VAE on a FEATURE vector z  (reuses the gls_vae.py math; xdim := z_dim)
# Shared class-conditional p(z_y|y) + per-domain pi_d(y) + per-domain p(z_d|d)
# + per-(domain,class) concept correction delta_d.  Identical structure to
# synthetic/gls_vae.py::GLSVAE, just operating on the EEGNet feature instead of raw x.
# ---------------------------------------------------------------------------
class GLSVAEFeat(nn.Module):
    def __init__(self, xdim, n_cls, n_dom, zy=8, zd=6, use_delta=False, hidden=128):
        super().__init__()
        self.zy, self.zd, self.ndom, self.nclass = zy, zd, n_dom, n_cls
        self.use_delta = use_delta
        self.enc = nn.Sequential(
            nn.Linear(xdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.qy_mu = nn.Linear(hidden, zy)
        self.qy_lv = nn.Linear(hidden, zy)
        self.qd_mu = nn.Linear(hidden, zd)
        self.qd_lv = nn.Linear(hidden, zd)
        self.dec = nn.Sequential(
            nn.Linear(zy + zd, hidden), nn.ReLU(), nn.Linear(hidden, xdim))
        # SHARED class-conditional latent prior p(z_y | y)
        self.py_mu = nn.Parameter(torch.randn(n_cls, zy) * 0.5)
        self.py_lv = nn.Parameter(torch.zeros(n_cls, zy))
        # per-domain latent prior p(z_d | d)
        self.pd_mu = nn.Parameter(torch.randn(n_dom, zd) * 0.5)
        self.pd_lv = nn.Parameter(torch.zeros(n_dom, zd))
        # per-domain FREE label prior logits -> pi_d(y)
        self.pi_logits = nn.Parameter(torch.zeros(n_dom, n_cls))
        # per-(domain,class) concept correction delta_d on (mu, logvar) of p(z_y|y)
        self.delta_mu = nn.Parameter(torch.zeros(n_dom, n_cls, zy))
        self.delta_lv = nn.Parameter(torch.zeros(n_dom, n_cls, zy))
        # DIVA identification heads
        self.aux_y = nn.Linear(zy, n_cls)
        self.aux_d = nn.Linear(zd, n_dom)

    def encode(self, x):
        h = self.enc(x)
        return self.qy_mu(h), self.qy_lv(h), self.qd_mu(h), self.qd_lv(h)

    @staticmethod
    def rsample(mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    @staticmethod
    def gauss_logprob(z, mu, lv):
        return (-0.5 * (lv + (z - mu) ** 2 / torch.exp(lv) + np.log(2 * np.pi))).sum(-1)

    def py_params(self, y, d=None):
        mu, lv = self.py_mu[y], self.py_lv[y]
        if self.use_delta and d is not None:
            mu = mu + self.delta_mu[d, y]
            lv = lv + self.delta_lv[d, y]
        return mu, lv

    def log_pi(self, d):
        return F.log_softmax(self.pi_logits, dim=1)[d]

    def elbo(self, x, y, d, beta=1.0):
        muy, lvy, mud, lvd = self.encode(x)
        zy, zd = self.rsample(muy, lvy), self.rsample(mud, lvd)
        pmu, plv = self.py_params(y, d if self.use_delta else None)
        log_pzy = self.gauss_logprob(zy, pmu, plv)
        log_pzd = self.gauss_logprob(zd, self.pd_mu[d], self.pd_lv[d])
        log_pi = self.log_pi(d).gather(1, y[:, None]).squeeze(1)
        ent = (0.5 * (lvy + np.log(2 * np.pi) + 1).sum(-1)
               + 0.5 * (lvd + np.log(2 * np.pi) + 1).sum(-1))
        recon = -((self.dec(torch.cat([zy, zd], 1)) - x) ** 2).sum(-1)
        return zy, zd, (log_pzy + log_pzd + log_pi + ent + beta * recon)


# ---------------------------------------------------------------------------
# Domain-stratified ELBO training (mirrors gls_vae.train_glsvae)
# ---------------------------------------------------------------------------
def _fit_elbo(m, X, y, d, params, epochs, lr, beta, aux):
    opt = torch.optim.Adam(params, lr)
    for _ in range(epochs):
        zy, zd, elbo = m.elbo(X, y, d, beta)
        loss = -elbo.mean()
        loss = loss + aux * (F.cross_entropy(m.aux_y(zy), y)
                             + F.cross_entropy(m.aux_d(zd), d))
        opt.zero_grad(); loss.backward(); opt.step()


def train_shared(Z, y, d, n_cls, n_dom, zy=8, zd=6, beta=1.0, aux=3.0,
                 epochs=300, lr=2e-3, seed=0, device="cpu"):
    """Train the shared GLS-VAE (use_delta=False) by the domain-stratified ELBO."""
    torch.manual_seed(seed)
    m = GLSVAEFeat(Z.shape[1], n_cls, n_dom, zy=zy, zd=zd, use_delta=False).to(device)
    _fit_elbo(m, Z, y, d, list(m.parameters()), epochs, lr, beta, aux)
    return m


def fit_delta_gain(m, Zfit, yfit, dfit, Zheld, yheld, dheld, beta=1.0,
                   delta_epochs=300, lr=2e-3, reset=True):
    """On a FROZEN shared model, fit ONLY delta_d on the fit split, then report the
    held-out ELBO gain (with delta) - (without delta).  This is the concept-test statistic.

    reset=True zeroes delta before each fit (so successive permutation refits are independent
    and start from the no-shift point)."""
    if reset:
        with torch.no_grad():
            m.delta_mu.zero_(); m.delta_lv.zero_()
    # freeze everything except delta
    saved = {n: p.requires_grad for n, p in m.named_parameters()}
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith("delta"))
    m.use_delta = False
    with torch.no_grad():
        e0 = m.elbo(Zheld, yheld, dheld, beta)[2].mean().item()   # shared-only held-out ELBO
    m.use_delta = True
    opt = torch.optim.Adam([m.delta_mu, m.delta_lv], lr)
    for _ in range(delta_epochs):
        _, _, elbo = m.elbo(Zfit, yfit, dfit, beta)               # fit delta on FIT split
        loss = -elbo.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        e1 = m.elbo(Zheld, yheld, dheld, beta)[2].mean().item()   # +delta held-out ELBO
    # restore grad flags
    for n, p in m.named_parameters():
        p.requires_grad_(saved[n])
    return e0, e1, e1 - e0


# ---------------------------------------------------------------------------
# Feature extraction (frozen EEGNet)  +  domain-stratified split
# ---------------------------------------------------------------------------
def extract_features(X, y, d, n_cls, backbone, bb_epochs, bs, device, seed):
    """Train an EEGNet feature extractor SUPERVISED (ERM) on (X,y) then freeze it and return
    Z = penultimate feature.  ERM keeps z label-informative (so p(z|y) is meaningful) without
    any invariance penalty (so any residual per-subject concept structure is preserved for
    delta_d to detect)."""
    n_ch, n_t = X.shape[1], X.shape[2]
    bb = build_backbone(backbone, n_ch, n_t, n_cls, device=device)
    print(f"  [bb] training {backbone} feature extractor (ERM, {bb_epochs} ep, z_dim probing...)",
          flush=True)
    bb, _, _ = train_model(bb, X, y, d, n_cls, method="erm", lam=0.0,
                           epochs=bb_epochs, bs=bs, warmup=max(1, bb_epochs // 5),
                           device=device, seed=seed)
    Z = embed(bb, X, device)
    # standardize features (stabilizes the VAE Gaussian heads)
    Z = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-6)
    print(f"  [bb] Z={Z.shape} (z_dim={Z.shape[1]})", flush=True)
    return Z.astype("float32")


def split_fit_held(Z, y, d, n_dom, frac_fit=0.5, seed=0, device="cpu"):
    """Per-domain fit/held split (analog of synthetic split_sources, but no held-out TARGET:
    LOSO target is not needed for the concept TEST, which is a source-internal diagnostic)."""
    rng = np.random.default_rng(1000 + seed)
    fit_mask = np.zeros(len(d), dtype=bool)
    for g in range(n_dom):
        idx = np.where(d == g)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * frac_fit))
        fit_mask[idx[:cut]] = True
    held_mask = ~fit_mask

    def t(arr, m):
        return torch.tensor(arr[m]).to(device)
    return (t(Z, fit_mask), t(y, fit_mask), t(d, fit_mask),
            t(Z, held_mask), t(y, held_mask), t(d, held_mask))


# ---------------------------------------------------------------------------
# The null-calibrated concept test
# ---------------------------------------------------------------------------
def concept_test_null(Z, y, d, n_cls, n_dom, args, device):
    """Observed delta_d held-out ELBO gain + a domain-permutation null distribution."""
    Zfit, yfit, dfit, Zheld, yheld, dheld = split_fit_held(
        Z, y, d, n_dom, seed=args.seed, device=device)
    # train the shared model ONCE (frozen across observed + all null refits)
    m = train_shared(Zfit, yfit, dfit, n_cls, n_dom, zy=args.zy, zd=args.zd,
                     beta=args.beta, epochs=args.vae_epochs, lr=args.lr,
                     seed=args.seed, device=device)
    # OBSERVED gain (true domain labels)
    e0, e1, gain = fit_delta_gain(m, Zfit, yfit, dfit, Zheld, yheld, dheld,
                                  beta=args.beta, delta_epochs=args.delta_epochs,
                                  lr=args.lr, reset=True)
    print(f"  [observed] ELBO shared={e0:.4f}  +delta={e1:.4f}  gain={gain:.5f}", flush=True)
    # NULL: permute the FIT-split and HELD-split domain labels (within each split, so the
    # per-domain sample counts are preserved) and refit delta_d on the SAME frozen shared model.
    rng = np.random.default_rng(7000 + args.seed)
    dfit_np, dheld_np = dfit.cpu().numpy(), dheld.cpu().numpy()
    null = []
    t0 = time.time()
    for p in range(args.n_perm):
        dfp = torch.tensor(rng.permutation(dfit_np)).to(device)
        dhp = torch.tensor(rng.permutation(dheld_np)).to(device)
        _, _, g = fit_delta_gain(m, Zfit, yfit, dfp, Zheld, yheld, dhp,
                                 beta=args.beta, delta_epochs=args.delta_epochs,
                                 lr=args.lr, reset=True)
        null.append(g)
        if (p + 1) % max(1, args.n_perm // 10) == 0:
            print(f"  [null] {p+1}/{args.n_perm} perms ({time.time()-t0:.0f}s) "
                  f"mean={np.mean(null):.5f} std={np.std(null):.5f}", flush=True)
    null = np.array(null, dtype=float)
    nmean, nstd = float(null.mean()), float(null.std() + 1e-12)
    z_score = float((gain - nmean) / nstd)
    # one-sided p: fraction of null >= observed (with +1 smoothing, the standard permutation p)
    p_value = float((np.sum(null >= gain) + 1) / (len(null) + 1))
    fired = (p_value < 0.05) and (z_score > 2.0)
    return dict(
        observed_gain=float(gain), elbo_shared=float(e0), elbo_delta=float(e1),
        null_mean=nmean, null_std=nstd, null_min=float(null.min()), null_max=float(null.max()),
        n_perm=int(args.n_perm), z_score=z_score, p_value=p_value,
        concept_shift_detected=bool(fired), null_gains=null.tolist())


# ---------------------------------------------------------------------------
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if args.condition:                                  # CROSS-SITE: D=cohort -> the VALID concept test
        from cmi.run_scps_crossdataset import load as load_xs   # (avoids the SCPS D=subject degeneracy)
        X, y, subj_s, coh_s, classes = load_xs(args.condition, args.cohorts)
        dom_raw = (coh_s if args.domain == "cohort" else subj_s)
        tag = f"{args.condition}/{args.domain}"
    else:                                               # within-dataset: D=subject (degenerate for concept)
        X, y, meta, classes = load(args.dataset, subjects=None, tmin=args.tmin, tmax=args.tmax,
                                   resample=args.resample, max_per_subject=args.max_per_subject)
        dom_raw = meta["subject"].to_numpy()
        if args.max_subjects:
            keep = sorted(np.unique(dom_raw))[:args.max_subjects]
            m = np.isin(dom_raw, keep); X, y, dom_raw = X[m], y[m], dom_raw[m]
        tag = args.dataset
    uniq = {v: i for i, v in enumerate(sorted(np.unique(dom_raw)))}
    d = np.array([uniq[v] for v in dom_raw], dtype=np.int64)
    n_cls, n_dom = len(classes), len(uniq)
    print(f"[{tag}] X={X.shape} classes={classes} n_dom={n_dom} ({args.domain if args.condition else 'subject'}) "
          f"device={device}  ({time.time()-t0:.0f}s to load)", flush=True)

    Z = extract_features(X, y, d, n_cls, args.backbone, args.bb_epochs, args.bs, device, args.seed)
    res = concept_test_null(Z, y, d, n_cls, n_dom, args, device)

    summary = {k: v for k, v in res.items() if k != "null_gains"}
    out = dict(dataset=args.dataset, backbone=args.backbone, n_dom=n_dom, n_cls=n_cls,
               n_windows=int(len(y)), z_dim=int(Z.shape[1]), config=vars(args),
               result=res)
    print("\n=== GLS-VAE null-calibrated concept test ===")
    print(json.dumps({"dataset": args.dataset, **summary}, indent=2))
    verdict = ("CONCEPT SHIFT FIRES (delta_d real)" if res["concept_shift_detected"]
               else "QUIET (delta_d not above null)")
    print(f"VERDICT [{args.dataset}]: {verdict}  "
          f"(observed={res['observed_gain']:.4f}, null={res['null_mean']:.4f}"
          f"+-{res['null_std']:.4f}, z={res['z_score']:.2f}, p={res['p_value']:.3f})")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ADFTD", choices=["ADFTD", "ADFTD_bin", "MUMTAZ", "TUAB"])
    ap.add_argument("--condition", default="", help="cross-site: SCZ/PD/AD/DEP (D=cohort) instead of within-dataset D=subject")
    ap.add_argument("--cohorts", nargs="*", default=None, help="restrict cross-site to these cohort ids")
    ap.add_argument("--domain", default="cohort", choices=["cohort", "subject"], help="cross-site domain granularity")
    ap.add_argument("--backbone", default="EEGNet",
                    choices=["EEGNet", "ShallowConvNet", "Deep4Net", "LogCov"])
    ap.add_argument("--bb_epochs", type=int, default=120, help="EEGNet feature-extractor ERM epochs")
    ap.add_argument("--vae_epochs", type=int, default=300, help="shared GLS-VAE ELBO epochs")
    ap.add_argument("--delta_epochs", type=int, default=300, help="delta_d-only fit epochs (per refit)")
    ap.add_argument("--n_perm", type=int, default=100, help="domain-permutation null size (>=50)")
    ap.add_argument("--zy", type=int, default=8)
    ap.add_argument("--zd", type=int, default=6)
    ap.add_argument("--beta", type=float, default=1.0, help="reconstruction weight in ELBO")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--max_per_subject", type=int, default=60)
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

"""gls_vae.py  --  ROUTE A: a structured-latent (GLS-VAE) model targeting BOTH
dual-CMI leakages WITHOUT the label-shift tension, plus a variational
CONCEPT-SHIFT TEST.

Background (notes/DUAL_CMI_THEORY.md).  The naive 'dual' of synthetic/dual_cmi_v2.py
co-minimizes the encoder leakage I(Z;D|Y) and the decoder leakage I(Y;D|Z) by two
penalties.  Under LABEL SHIFT those two FIGHT (tension theorem A2): forcing the
encoder invariant pushes the label-prior shift out of p(z|y) and into the feature
marginal p(z), where the decoder must read it back -> I(Y;D|Z) is forced UP by
exactly I(Y;D)-I(Z;D).  The resolution (A4) is GLS label-correction.

ROUTE A bakes the GLS correction into the MODEL (cf. DIVA, Ilse 2020) so the label
term is handled BY CONSTRUCTION and the remaining encoder pressure becomes
FIGHT-FREE.  Architecture (per domain d):

  PARTITIONED LATENT
    z_y  ~ p_theta(z_y | y)      SHARED class-conditional Gaussian  (domain-free)
    z_d  ~ p(z_d | d)            per-domain Gaussian  (soaks up the covariate offset)
    y    ~ pi_d(y)               per-domain FREE label prior (learned logits)
    x    ~ p(x | z_y, z_d)       small Gaussian decoder (reconstruction keeps the
                                 latent informative; the covariate domain-offset is
                                 explained away by z_d, NOT by z_y)
  amortized encoder  q_phi(z_y, z_d | x).
  IDENTIFICATION (DIVA aux heads): q(y|z_y), q(d|z_d) anchor the partition.
  GLS DECODE for transfer:  p(y | z_y) propto pi*(y) p_theta(z_y|y),  pi* = uniform.

  OPTIONAL explicit encoder invariance penalty  lam_inv * E KL( q(d|z_y,y) || unif )
  = a Barber-Agakov upper bound on I(z_y; D | Y).  This is the SINGLE remaining
  penalty; the GLS decode neutralizes the label term, so (the claim) pushing it
  down does NOT raise I(Y;D|Z) -- no tension, unlike the naive dual.

  CONCEPT-SHIFT TEST:  p_theta(z_y | y, d) = p_theta(z_y|y) * exp(delta_d(z_y,y)) / Z_d,
  realized as a per-(domain,class) Gaussian shift delta_d on (mu, logvar).  Fit the
  shared model, FREEZE it, then fit ONLY delta_d; the held-out ELBO gain is a
  variational likelihood-ratio that is large ONLY when the class-conditional law
  p(z_y|y) genuinely differs by site == CONCEPT shift.

HONEST SCOPE (validated below, see notes/route_A_gls_vae.md):
  * The structure alone does NOT make I(z_y;D|Y) vanish "by construction" for an
    amortized encoder -- the shared prior only constrains the AGGREGATE per-class
    law, so the encoder can still place domain-d samples in a domain-specific
    region of class-y's cluster.  Driving I(z_y;D|Y)->0 needs the explicit lam_inv
    penalty.  The structural win is that this penalty is FIGHT-FREE under label
    shift (the naive-dual tension is dissolved), NOT that it is unnecessary.
  * The delta_d concept-shift TEST is the clean, robust deliverable.

Reuses the DGP + held-out CMI probes from dual_cmi_v2.py verbatim (apples-to-apples
on the same data and the same measurement instrument).

Run:  /home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/gls_vae.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_cmi_v2 import (
    gen, split_sources, train as train_naive, accuracy as accuracy_naive,
    measure_cmi_heldout,
)

torch.set_num_threads(8)
NCLASS = 2


# ---------------------------------------------------------------------------
# GLS-VAE  (DIVA-style partitioned latent + GLS decode + concept-correction delta)
# ---------------------------------------------------------------------------
class GLSVAE(nn.Module):
    def __init__(self, xdim, zy=6, zd=4, ndom=4, nclass=NCLASS,
                 use_delta=False, hidden=64):
        super().__init__()
        self.zy, self.zd, self.ndom, self.nclass = zy, zd, ndom, nclass
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
        self.py_mu = nn.Parameter(torch.randn(nclass, zy) * 0.5)
        self.py_lv = nn.Parameter(torch.zeros(nclass, zy))
        # per-domain latent prior p(z_d | d)
        self.pd_mu = nn.Parameter(torch.randn(ndom, zd) * 0.5)
        self.pd_lv = nn.Parameter(torch.zeros(ndom, zd))
        # per-domain FREE label prior logits -> pi_d(y)
        self.pi_logits = nn.Parameter(torch.zeros(ndom, nclass))
        # per-(domain,class) concept correction delta_d on (mu, logvar) of p(z_y|y)
        self.delta_mu = nn.Parameter(torch.zeros(ndom, nclass, zy))
        self.delta_lv = nn.Parameter(torch.zeros(ndom, nclass, zy))
        # DIVA identification heads
        self.aux_y = nn.Linear(zy, nclass)
        self.aux_d = nn.Linear(zd, ndom)
        # variational q(d | z_y, y) for the encoder-invariance (covariate) penalty
        self.q_dzy = nn.Sequential(
            nn.Linear(zy + nclass, hidden), nn.ReLU(), nn.Linear(hidden, ndom))

    # ---- inference ----
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

    # ---- domain-stratified ELBO (per-sample) ----
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

    # ---- GLS / Bayes classification on z_y (vectorized over classes) ----
    @torch.no_grad()
    def class_logprob(self, x, prior="reference", nmc=8):
        muy, lvy, _, _ = self.encode(x)
        N = x.shape[0]
        # vectorized per-class Gaussian: priors [nclass, zy]
        pmu, plv = self.py_mu, self.py_lv                    # [C, zy]
        inv = torch.exp(-plv)                                # [C, zy]
        const = (plv + np.log(2 * np.pi)).sum(1)             # [C]
        acc = torch.zeros(N, self.nclass)
        for _ in range(nmc):
            zy = self.rsample(muy, lvy)                      # [N, zy]
            diff = zy[:, None, :] - pmu[None, :, :]          # [N, C, zy]
            ll = -0.5 * ((diff ** 2 * inv[None]).sum(-1) + const[None])  # [N, C]
            if prior == "reference":
                ll = ll + np.log(1.0 / self.nclass)
            acc = acc + F.softmax(ll, dim=1)
        return torch.log(acc / nmc + 1e-12)


class ZyAdapter(nn.Module):
    """Expose z_y mean as a deterministic feature map so the SAME held-out probe
    machinery from dual_cmi_v2 measures I(z_y;D|Y) and I(Y;D|z_y)."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m.encode(x)[0]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_glsvae(s, K, beta=1.0, aux=3.0, lam_inv=0.0, epochs=220, lr=2e-3,
                 seed=0, use_delta=False, freeze_shared_for_delta=False):
    """Maximize the domain-stratified ELBO + aux identification, optionally + a
    single encoder invariance penalty lam_inv on z_y.

    Concept test: if use_delta and freeze_shared_for_delta, first fit the full
    shared model, FREEZE everything, then train ONLY delta_d (clean likelihood
    ratio).  In that phase the invariance penalty is OFF (q_dzy frozen)."""
    torch.manual_seed(seed)
    X, y, d = s["Xfit"], s["yfit"], s["dfit"]
    xdim = X.shape[1]
    m = GLSVAE(xdim, ndom=K, use_delta=use_delta)
    yoh = F.one_hot(y, NCLASS).float()
    logpD_unif = torch.log(torch.full((K,), 1.0 / K))

    def fit(params, ep, with_inv):
        opt = torch.optim.Adam(params, lr)
        optp = torch.optim.Adam(m.q_dzy.parameters(), lr) if with_inv else None
        for _ in range(ep):
            if with_inv:  # Step A: fit q(d|z_y,y) on detached z_y
                with torch.no_grad():
                    zy0 = m.encode(X)[0]
                for _ in range(2):
                    la = F.cross_entropy(m.q_dzy(torch.cat([zy0, yoh], 1)), d)
                    optp.zero_grad(); la.backward(); optp.step()
            zy, zd, elbo = m.elbo(X, y, d, beta)
            loss = -elbo.mean()
            loss = loss + aux * (F.cross_entropy(m.aux_y(zy), y)
                                 + F.cross_entropy(m.aux_d(zd), d))
            if with_inv and lam_inv > 0:
                logq = F.log_softmax(m.q_dzy(torch.cat([zy, yoh], 1)), 1)
                Iinv = (logq.exp() * (logq - logpD_unif)).sum(1).mean()
                loss = loss + lam_inv * Iinv
            opt.zero_grad(); loss.backward(); opt.step()

    if use_delta and freeze_shared_for_delta:
        m.use_delta = False
        fit(list(m.parameters()), epochs, with_inv=(lam_inv > 0))
        m.use_delta = True
        for n, p in m.named_parameters():
            p.requires_grad_(n.startswith("delta"))
        fit([m.delta_mu, m.delta_lv], epochs, with_inv=False)  # no Step-A here
        for p in m.parameters():
            p.requires_grad_(True)
    else:
        fit(list(m.parameters()), epochs, with_inv=(lam_inv > 0))
    return m


@torch.no_grad()
def glsvae_accuracy(m, X, y, prior="reference"):
    return (m.class_logprob(X, prior=prior, nmc=12).argmax(1) == y).float().mean().item() * 100.0


@torch.no_grad()
def glsvae_held_elbo(m, s, beta=1.0):
    return m.elbo(s["Xheld"], s["yheld"], s["dheld"], beta)[2].mean().item()


def concept_test(s, K, beta=1.0, lam_inv=0.0, epochs=220, seed=0):
    """Held-out ELBO gain from the per-domain concept-correction delta_d on top of
    a FROZEN shared model.  Large positive gain == concept shift."""
    m0 = train_glsvae(s, K, beta=beta, lam_inv=lam_inv, epochs=epochs, seed=seed,
                      use_delta=False)
    e0 = glsvae_held_elbo(m0, s, beta)
    m1 = train_glsvae(s, K, beta=beta, lam_inv=lam_inv, epochs=epochs, seed=seed,
                      use_delta=True, freeze_shared_for_delta=True)
    e1 = glsvae_held_elbo(m1, s, beta)
    return e0, e1, e1 - e0


# ---------------------------------------------------------------------------
# Experiment A : both CMIs + fight test, GLS-VAE vs naive dual
# ---------------------------------------------------------------------------
DGPS = [
    ("covariate-only", dict(cov=1, con=0, labelshift=False)),
    ("covariate+label", dict(cov=1, con=0, labelshift=True)),
    ("all-three",       dict(cov=1, con=1, labelshift=True)),
    ("concept-only",    dict(cov=0, con=1, labelshift=False)),
]


def run_compare(nseeds=4, K=4, lam_inv=2.0):
    print("=" * 92)
    print("ROUTE A : GLS-VAE vs naive dual  --  SAME data, SAME held-out CMI probes")
    print("  erm/dual         = dual_cmi_v2 baselines (penalty co-minimization)")
    print("  glsvae           = partitioned latent + GLS decode, NO encoder penalty")
    print(f"  glsvae+inv       = + single encoder invariance penalty lam_inv={lam_inv}")
    print("                     (GLS decode dissolves the label tension -> fight-free)")
    print("  CMIs on held-out SOURCE (z_y for glsvae); tgtAcc on held-out TARGET.")
    print("=" * 92)
    head = (f"{'DGP':16s} {'method':12s} {'tgtAcc':>7s} {'+-':>5s} "
            f"{'I(Z;D|Y)':>9s} {'I(Y;D|Z)':>9s}")
    res = {}
    for dname, kw in DGPS:
        print("-" * 92); print(head); print("-" * 92)
        for mname in ["erm", "dual", "glsvae", "glsvae+inv"]:
            accs, ies, ids = [], [], []
            for sd in range(nseeds):
                X, Y, D, _ = gen(seed=sd, K=K, **kw)
                s = split_sources(X, Y, D, K, seed=sd)
                if mname == "erm":
                    enc, clf = train_naive(s, 0.0, 0.0, False, K=K, seed=sd)
                    acc = accuracy_naive(enc, clf, s["Xtgt"], s["ytgt"])
                    ie, idc = measure_cmi_heldout(enc, s, K, seed=sd)
                elif mname == "dual":
                    enc, clf = train_naive(s, 0.7, 0.7, False, K=K, seed=sd)
                    acc = accuracy_naive(enc, clf, s["Xtgt"], s["ytgt"])
                    ie, idc = measure_cmi_heldout(enc, s, K, seed=sd)
                else:
                    li = 0.0 if mname == "glsvae" else lam_inv
                    m = train_glsvae(s, K, lam_inv=li, seed=sd)
                    acc = glsvae_accuracy(m, s["Xtgt"], s["ytgt"], prior="reference")
                    ie, idc = measure_cmi_heldout(ZyAdapter(m), s, K, seed=sd)
                accs.append(acc); ies.append(ie); ids.append(idc)
            print(f"{dname:16s} {mname:12s} {np.mean(accs):6.1f} {np.std(accs):5.1f} "
                  f"{np.mean(ies):9.4f} {np.mean(ids):9.4f}")
            res[(dname, mname)] = (np.mean(accs), np.mean(ies), np.mean(ids))
    print("-" * 92)
    return res


def run_fight_sweep(nseeds=4, K=4):
    """The decisive 'no fight' test under LABEL SHIFT (covariate+label DGP):
    sweep the encoder pressure UP and watch I(Y;D|Z).
      * naive dual: raising lam_enc RAISES I(Y;D|Z)  (tension)
      * glsvae+inv: raising lam_inv drives I(z_y;D|Y) DOWN with I(Y;D|z_y) staying
        low (GLS decode already neutralized the label term -> no fight).
    """
    print("\n" + "=" * 92)
    print("ROUTE A : THE 'NO FIGHT' TEST under label shift (covariate+label DGP)")
    print("  push encoder pressure UP; does the DECODER leakage I(Y;D|Z) get forced up?")
    print("=" * 92)
    print(f"{'model':12s} {'enc_pressure':>12s} {'I(Z;D|Y)':>9s} {'I(Y;D|Z)':>9s} {'tgtAcc':>7s}")
    print("-" * 92)
    kw = dict(cov=1, con=0, labelshift=True)
    for le in [0.0, 1.0, 4.0]:
        ies, ids, accs = [], [], []
        for sd in range(nseeds):
            X, Y, D, _ = gen(seed=sd, K=K, **kw); s = split_sources(X, Y, D, K, seed=sd)
            enc, clf = train_naive(s, le, 0.0, False, K=K, seed=sd)
            ie, idc = measure_cmi_heldout(enc, s, K, seed=sd)
            ies.append(ie); ids.append(idc)
            accs.append(accuracy_naive(enc, clf, s["Xtgt"], s["ytgt"]))
        print(f"{'naive-enc':12s} {le:12.1f} {np.mean(ies):9.4f} {np.mean(ids):9.4f} {np.mean(accs):6.1f}")
    print("-" * 92)
    for li in [0.0, 2.0, 6.0]:
        ies, ids, accs = [], [], []
        for sd in range(nseeds):
            X, Y, D, _ = gen(seed=sd, K=K, **kw); s = split_sources(X, Y, D, K, seed=sd)
            m = train_glsvae(s, K, lam_inv=li, seed=sd)
            ie, idc = measure_cmi_heldout(ZyAdapter(m), s, K, seed=sd)
            ies.append(ie); ids.append(idc)
            accs.append(glsvae_accuracy(m, s["Xtgt"], s["ytgt"], prior="reference"))
        print(f"{'glsvae+inv':12s} {li:12.1f} {np.mean(ies):9.4f} {np.mean(ids):9.4f} {np.mean(accs):6.1f}")
    print("-" * 92)


# ---------------------------------------------------------------------------
# Experiment B : variational concept-shift TEST
# ---------------------------------------------------------------------------
def run_concept_test(nseeds=4, K=4, lam_inv=2.0):
    print("\n" + "=" * 92)
    print("ROUTE A : VARIATIONAL CONCEPT-SHIFT TEST  (delta_d held-out ELBO gain)")
    print("  delta_d corrects p(z_y|y) per domain; gain is large ONLY when the")
    print("  class-conditional latent law genuinely differs by site == concept shift.")
    print("=" * 92)
    print(f"{'DGP':16s} {'ELBO(shared)':>13s} {'ELBO(+delta)':>13s} {'gain':>8s} {'+-':>6s}  verdict")
    print("-" * 92)
    out = {}
    for dname, kw in DGPS:
        e0s, e1s, gains = [], [], []
        for sd in range(nseeds):
            X, Y, D, _ = gen(seed=sd, K=K, **kw); s = split_sources(X, Y, D, K, seed=sd)
            e0, e1, g = concept_test(s, K, lam_inv=lam_inv, seed=sd)
            e0s.append(e0); e1s.append(e1); gains.append(g)
        gm, gs = np.mean(gains), np.std(gains)
        detected = (gm > 0.20) and (gm > 2 * gs)
        verdict = "CONCEPT SHIFT" if detected else "no concept shift"
        print(f"{dname:16s} {np.mean(e0s):13.3f} {np.mean(e1s):13.3f} "
              f"{gm:8.3f} {gs:6.3f}  {verdict}")
        out[dname] = (np.mean(e0s), np.mean(e1s), gm, gs, detected)
    print("-" * 92)
    print("Reading: the test fires (CONCEPT SHIFT) on concept-bearing DGPs and stays")
    print("quiet on pure covariate / covariate+label DGPs -> a usable concept diagnostic.")
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    run_compare(nseeds=4)
    run_fight_sweep(nseeds=4)
    run_concept_test(nseeds=4)
    print(f"\n[done in {time.time()-t0:.0f}s]")

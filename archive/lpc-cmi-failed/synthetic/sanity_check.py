"""Synthetic sanity check (Protocol D) for Tri-CMI / LPC-CMI.

Controlled data-generating process (DGP) with three feature groups and
label-domain imbalance, designed so the *source-only* learner can in principle
recover the invariant mechanism:

  x_c   (causal, INVARIANT): x_c | Y ~ N( mu_c * (2Y-1), 1 ).  Same mechanism in
        every domain (source and target). This is the feature a robust model
        should keep.
  x_s   (SPURIOUS shortcut): x_s | Y,D ~ N( a_d * (2Y-1), sig_s ). In the source
        domains a_d is positive but VARIES by domain (so x_s leaks D given Y and
        is detectably unreliable); in the unseen TARGET a_d flips sign, so any
        model that leans on x_s fails out-of-distribution.
  x_st  (pure DOMAIN STYLE): x_st | D ~ N( c_d, 1 ), independent of Y. Pure domain
        leakage; a good representation drops it.

Label-domain imbalance: P(Y=1 | D=d) varies strongly across source domains, so
the label-conditional domain prior pi_y(D)=p(D|Y) is far from uniform, and the
causal feature x_c is *marginally* correlated with D (through Y).

This lets us contrast five training objectives (all sharing the same encoder /
classifier / alternating optimisation):

  erm          : cross-entropy only.
  marginal     : + lambda * I(Z;D)        proxy  = E KL( q(D|Z)   || p(D)   ).
  chain        : + lambda * I(Z;(D,Y))    proxy  = E KL( q(S|Z)   || p(S)   ), S=(D,Y).
  lpc_uniform  : + lambda * E KL( q(D|Z,Y) || Uniform ).
  lpc_prior    : + lambda * E KL( q(D|Z,Y) || pi_y(D) ).   <-- our method (LPC-CMI)

Expected story (verified empirically by this script):
  * erm over-relies on x_s -> low TARGET accuracy.
  * marginal removes x_c too (imbalance) -> damages label -> low accuracy.
  * chain erases Y -> accuracy ~ chance everywhere.
  * lpc_uniform is miscalibrated under imbalance (pushes against the true prior).
  * lpc_prior keeps x_c, drops x_s/x_st -> best TARGET accuracy & lowest residual
    conditional leakage while preserving label separability.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# ----------------------------------------------------------------------------- DGP
@dataclass
class DGP:
    """ColoredMNIST-style spurious mechanism. The spurious feature x_s is a noisy
    copy of a *label flipped with a domain-specific rate* e_d:  x_s carries the
    label only as reliably as (1 - e_d).  Because e_d VARIES across source domains,
    x_s|Y has a domain-dependent distribution -> I(x_s; D | Y) > 0 -> the spurious
    feature is *detectable* from source data alone.  In the target e_d flips high,
    so any reliance on x_s back-fires.  The causal feature x_c uses the same
    mechanism in every domain.  Label-domain imbalance (p_d) is decoupled from the
    flip rate (e_d), so it independently makes x_c marginally predict D."""
    dc: int = 4              # causal dims (invariant mechanism)
    ds: int = 4              # spurious dims (domain-flipped shortcut)
    dm: int = 4              # pure domain-style dims (predict D, independent of Y) -> the
                             #   unambiguous conditional leakage that lpc_prior removes at no Y-cost
    mu_c: float = 0.45       # causal effect (x_c-only Bayes ~82%)
    m_s: float = 2.2         # spurious magnitude
    sig_s: float = 0.7       # spurious noise
    # per source domain: (P(Y=1|D), spurious flip rate e_d).  e_d varies (detectable via
    # I(x_s;D|Y)); x_s flips in the target so reliance back-fires there.
    src: tuple = ((0.15, 0.05), (0.38, 0.45), (0.62, 0.05), (0.85, 0.45))
    tgt_p: float = 0.5
    tgt_e: float = 0.90      # spurious flips in the unseen target
    style_scale: float = 1.6

    @property
    def n_src(self) -> int:
        return len(self.src)

    @property
    def dx(self) -> int:
        return self.dc + self.ds + self.dm

    def _style_means(self):
        g = np.random.default_rng(12345)            # style fixed across seeds (a DGP property)
        return g.normal(0, self.style_scale, size=(self.n_src + 1, self.dm))

    def sample(self, n_per_domain, rng, target=False):
        style = self._style_means()
        Xs, Ys, Ds = [], [], []
        domains = [self.n_src] if target else list(range(self.n_src))
        for d in domains:
            p, e = (self.tgt_p, self.tgt_e) if target else self.src[d]
            y = rng.binomial(1, p, size=n_per_domain)
            s = (2 * y - 1).astype(np.float32)
            # spurious label = y XOR flip(e); x_s is a noisy copy of it
            flip = rng.binomial(1, e, size=n_per_domain)
            ys = np.bitwise_xor(y, flip)
            ss = (2 * ys - 1).astype(np.float32)
            xc = self.mu_c * s[:, None] + rng.normal(0, 1, (n_per_domain, self.dc))
            xs = self.m_s * ss[:, None] + rng.normal(0, self.sig_s, (n_per_domain, self.ds))
            xst = style[d][None, :] + rng.normal(0, 1, (n_per_domain, self.dm))
            x = np.concatenate([xc, xs, xst], axis=1).astype(np.float32)
            Xs.append(x); Ys.append(y.astype(np.int64))
            Ds.append(np.full(n_per_domain, 0 if target else d, dtype=np.int64))
        return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ds))


# ----------------------------------------------------------------------------- nets
class Encoder(nn.Module):
    def __init__(self, dx, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dx, 64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, h))

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Linear(64, dout))

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- helpers
def kl_to_prior(logits, log_prior):
    """E_x KL( softmax(logits) || prior ), prior given as log-probabilities (broadcast)."""
    logq = F.log_softmax(logits, dim=1)
    q = logq.exp()
    return (q * (logq - log_prior)).sum(1).mean()


def empirical_priors(y, d, n_dom, alpha=1.0):
    """pi_y(d)=p(d|y) (Laplace-smoothed), marginal p(d), joint p(d,y)."""
    n_y = 2
    counts = np.zeros((n_y, n_dom))
    for yi, di in zip(y, d):
        counts[yi, di] += 1
    pi_y = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * n_dom)  # p(d|y)
    p_d = (counts.sum(0) + alpha) / (counts.sum() + alpha * n_dom)            # p(d)
    p_dy = (counts + alpha) / (counts.sum() + alpha * n_y * n_dom)            # p(d,y) over S=(d,y)
    return pi_y, p_d, p_dy


# ----------------------------------------------------------------------------- training
def train_one(method, data, dgp, lam=1.0, h=16, epochs=60, bs=256, warmup=15,
              n_inner=3, device="cpu", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, dtr = data["train"]
    n_dom = dgp.n_src
    pi_y, p_d, p_dy = empirical_priors(ytr, dtr, n_dom)
    log_pi_y = torch.log(torch.tensor(pi_y, dtype=torch.float32, device=device))      # [2, K]
    log_pd = torch.log(torch.tensor(p_d, dtype=torch.float32, device=device))         # [K]
    log_unif = torch.log(torch.full((n_dom,), 1.0 / n_dom, device=device))            # [K]
    log_pS = torch.log(torch.tensor(p_dy.reshape(-1), dtype=torch.float32, device=device))  # [2K]

    enc = Encoder(dgp.dx, h).to(device)
    clf = Head(h, 2).to(device)
    # posteriors used by the different objectives
    q_dzy = Head(h + 2, n_dom).to(device)        # q(D|Z,Y)  -> lpc_*
    q_dz = Head(h, n_dom).to(device)             # q(D|Z)    -> marginal
    q_sz = Head(h, 2 * n_dom).to(device)         # q(S|Z), S=(D,Y) -> chain

    opt_main = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-3)
    opt_post = torch.optim.Adam(list(q_dzy.parameters()) + list(q_dz.parameters())
                                + list(q_sz.parameters()), lr=2e-3)

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    dtr_t = torch.tensor(dtr, device=device)
    n = len(Xtr); steps = max(1, n // bs)

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        lam_t = lam * min(1.0, ep / max(1, warmup))   # warm-up
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb, yb, db = Xtr_t[idx], ytr_t[idx], dtr_t[idx]
            y_oh = F.one_hot(yb, 2).float()
            s_lab = db * 2 + yb                       # super-label S=(D,Y) in [0,2K)

            # ---- Step A: train posteriors on detached z (inner loop -> tighter bound) ----
            with torch.no_grad():
                z = enc(xb)
            for _ in range(n_inner):
                la = F.cross_entropy(q_dzy(torch.cat([z, y_oh], 1)), db) \
                    + F.cross_entropy(q_dz(z), db) \
                    + F.cross_entropy(q_sz(z), s_lab)
                opt_post.zero_grad(); la.backward(); opt_post.step()

            # ---- Step B: update encoder + classifier ----
            z = enc(xb)
            l_cls = F.cross_entropy(clf(z), yb)
            if method == "erm":
                reg = torch.zeros((), device=device)
            elif method == "marginal":
                reg = kl_to_prior(q_dz(z), log_pd)
            elif method == "chain":
                reg = kl_to_prior(q_sz(z), log_pS)
            elif method == "lpc_uniform":
                reg = kl_to_prior(q_dzy(torch.cat([z, y_oh], 1)), log_unif)
            elif method == "lpc_prior":
                reg = kl_to_prior(q_dzy(torch.cat([z, y_oh], 1)), log_pi_y[yb])
            else:
                raise ValueError(method)
            loss = l_cls + lam_t * reg
            opt_main.zero_grad(); loss.backward(); opt_main.step()

    return enc, clf, (pi_y, log_pi_y)


@torch.no_grad()
def embed(enc, X, device="cpu"):
    return enc(torch.tensor(X, device=device)).cpu().numpy()


def leakage_probe(enc, data, dgp, h=16, epochs=120, device="cpu", seed=0):
    """Frozen-encoder held-out probe: train q_probe(D|Z,Y) on source-probe split,
    report mean KL(q_probe || pi_y) and conditional domain-prediction accuracy on a
    source-eval split. Higher = more residual conditional domain leakage I(Z;D|Y)."""
    torch.manual_seed(seed)
    Xp, yp, dp = data["probe"]; Xe, ye, de = data["val"]
    n_dom = dgp.n_src
    pi_y, _, _ = empirical_priors(yp, dp, n_dom)
    log_pi_y = torch.log(torch.tensor(pi_y, dtype=torch.float32, device=device))
    Zp, Ze = embed(enc, Xp, device), embed(enc, Xe, device)
    q = Head(h + 2, n_dom).to(device)
    opt = torch.optim.Adam(q.parameters(), lr=2e-3)
    Zp_t = torch.tensor(Zp, device=device); yp_t = torch.tensor(yp, device=device)
    dp_t = torch.tensor(dp, device=device)
    for _ in range(epochs):
        inp = torch.cat([Zp_t, F.one_hot(yp_t, 2).float()], 1)
        opt.zero_grad(); F.cross_entropy(q(inp), dp_t).backward(); opt.step()
    with torch.no_grad():
        Ze_t = torch.tensor(Ze, device=device); ye_t = torch.tensor(ye, device=device)
        inp = torch.cat([Ze_t, F.one_hot(ye_t, 2).float()], 1)
        logits = q(inp)
        kl = kl_to_prior(logits, log_pi_y[ye_t]).item()
        dom_acc = (logits.argmax(1).cpu().numpy() == de).mean()
        prior_acc = (pi_y[ye].argmax(1) == de).mean()   # best you can do from Y alone
    return dict(leakage_kl=kl, cond_dom_acc=float(dom_acc),
                prior_dom_acc=float(prior_acc), leakage_advantage=float(dom_acc - prior_acc))


def label_separability(enc, data, device="cpu"):
    Xtr, ytr, _ = data["train"]; Xe, ye, _ = data["val"]
    Ztr, Ze = embed(enc, Xtr, device), embed(enc, Xe, device)
    lr = LogisticRegression(max_iter=1000).fit(Ztr, ytr)
    return float(lr.score(Ze, ye))


def evaluate(enc, clf, data, split, device="cpu"):
    X, y, _ = data[split]
    with torch.no_grad():
        pred = clf(enc(torch.tensor(X, device=device))).argmax(1).cpu().numpy()
    return float(balanced_accuracy_score(y, pred))


# ----------------------------------------------------------------------------- driver
def make_data(dgp, seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr, dtr = dgp.sample(2000, rng)
    Xpr, ypr, dpr = dgp.sample(800, rng)
    Xva, yva, dva = dgp.sample(800, rng)
    Xtg, ytg, dtg = dgp.sample(4000, rng, target=True)
    return dict(train=(Xtr, ytr, dtr), probe=(Xpr, ypr, dpr),
                val=(Xva, yva, dva), target=(Xtg, ytg, dtg))


def run(lam, seeds, epochs, dgp=None):
    dgp = dgp or DGP()
    methods = ["erm", "marginal", "chain", "lpc_uniform", "lpc_prior"]
    keys = ["src_bacc", "target_bacc", "leakage_kl", "leakage_advantage", "label_sep"]
    agg = {m: {k: [] for k in keys} for m in methods}
    for seed in range(seeds):
        data = make_data(dgp, seed)
        for m in methods:
            enc, clf, _ = train_one(m, data, dgp, lam=lam, epochs=epochs, seed=seed)
            agg[m]["src_bacc"].append(evaluate(enc, clf, data, "val"))
            agg[m]["target_bacc"].append(evaluate(enc, clf, data, "target"))
            lp = leakage_probe(enc, data, dgp, seed=seed)
            agg[m]["leakage_kl"].append(lp["leakage_kl"])
            agg[m]["leakage_advantage"].append(lp["leakage_advantage"])
            agg[m]["label_sep"].append(label_separability(enc, data))
    summary = {m: {k: (float(np.mean(v)), float(np.std(v))) for k, v in agg[m].items()}
               for m in methods}
    return methods, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0, 5.0, 20.0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="synthetic/results.json")
    args = ap.parse_args()

    out = {"config": vars(args), "sweep": {}}
    for lam in args.lams:
        methods, summary = run(lam, args.seeds, args.epochs)
        out["sweep"][str(lam)] = summary
        print(f"\n=== lambda={lam}  ({args.seeds} seeds) ===")
        print(f"{'method':14s} {'SrcBAcc':>9s} {'TgtBAcc':>11s} {'LeakKL':>9s} {'LeakAdv':>9s} {'LabelSep':>9s}")
        for m in methods:
            r = summary[m]
            print(f"{m:14s} {r['src_bacc'][0]*100:7.1f}% "
                  f"{r['target_bacc'][0]*100:6.1f}±{r['target_bacc'][1]*100:3.1f} "
                  f"{r['leakage_kl'][0]:8.3f} {r['leakage_advantage'][0]:+8.3f} "
                  f"{r['label_sep'][0]*100:7.1f}%")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    return out


if __name__ == "__main__":
    main()

"""Controlled validation of the DUAL-CMI framework (encoder I(Z;D|Y) + decoder I(Y;D|Z) + BER).

A synthetic DGP with THREE independently dial-able distribution shifts across domains:
  - COVARIATE shift : p(x|y) shifts per domain (class-conditional feature offset)   -> should be killed by I(Z;D|Y)
  - CONCEPT  shift  : p(y|x) flips per domain (feature->label sign flips)            -> should be killed by I(Y;D|Z)
  - LABEL    shift  : p(y|d) varies per domain (class prior)                          -> should be handled by BER

Each example: x = [core(2) | cov(2) | con(2)]. core carries the INVARIANT signal (always generalizes);
cov/con are domain-specific shortcuts. Ideal Z uses only core. We train a small MLP with
  CE(+BER) + lam_enc * I(Z;D|Y) + lam_dec * I(Y;D|Z)
estimated variationally (q(D|Z,Y), h(Y|Z,D)), then report TARGET-domain accuracy and BOTH measured CMIs.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


def gen(cov, con, labelshift, n=1500, K=4, seed=0):
    """K source domains + 1 target. Returns dicts of tensors."""
    rng = np.random.default_rng(seed)
    doms = list(range(K + 1))                       # 0..K-1 source, K = target
    shift_d = rng.normal(0, 1.5, K + 1)             # per-domain covariate offset
    sign_d = np.array([1, 1, 1, 1, -1.0])[: K + 1]  # concept: target flips the feature->label sign
    if labelshift:
        pi_d = np.array([0.25, 0.4, 0.6, 0.75, 0.5])[: K + 1]   # source priors skew; target balanced
    else:
        pi_d = np.full(K + 1, 0.5)
    Xs, Ys, Ds = [], [], []
    for d in doms:
        y = (rng.random(n) < pi_d[d]).astype(int)
        s = 2 * y - 1
        core = rng.normal(0, 1, (n, 2)) + 1.2 * s[:, None] * np.array([1, 1])      # INVARIANT
        covf = rng.normal(0, 1, (n, 2)) + 1.2 * s[:, None] + cov * shift_d[d]      # covariate shift
        conf = rng.normal(0, 1, (n, 2)) + con * sign_d[d] * s[:, None] * np.array([1, 1])  # concept shift
        Xs.append(np.concatenate([core, covf, conf], 1)); Ys.append(y); Ds.append(np.full(n, d))
    X = np.concatenate(Xs).astype("float32"); Y = np.concatenate(Ys); D = np.concatenate(Ds)
    tr = D < K; te = D == K
    return (torch.tensor(X[tr]), torch.tensor(Y[tr]), torch.tensor(D[tr]),
            torch.tensor(X[te]), torch.tensor(Y[te]))


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def train(Xtr, ytr, dtr, lam_enc, lam_dec, balance, K, epochs=150, seed=0):
    torch.manual_seed(seed)
    nd, zc = int(dtr.max()) + 1, 16
    enc = mlp(6, zc); clf = nn.Linear(zc, 2)
    qD = mlp(zc + 2, nd); hY = mlp(zc + nd, 2)               # q(D|Z,Y), h(Y|Z,D)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), 1e-3)
    optp = torch.optim.Adam(list(qD.parameters()) + list(hY.parameters()), 2e-3)
    # priors
    cnt = torch.bincount(ytr, minlength=2).float(); cew = (cnt.sum() / (2 * cnt.clamp(min=1))) if balance else None
    pi = torch.zeros(2, nd)
    for y, d in zip(ytr, dtr): pi[y, d] += 1
    log_piy = torch.log((pi + 1) / (pi.sum(1, keepdim=True) + nd))
    for ep in range(epochs):
        w = min(1.0, ep / 40)
        z = enc(Xtr)
        # Step A: fit posteriors on detached z
        for _ in range(2):
            yoh = F.one_hot(ytr, 2).float(); doh = F.one_hot(dtr, nd).float()
            la = F.cross_entropy(qD(torch.cat([z.detach(), yoh], 1)), dtr) + \
                 F.cross_entropy(hY(torch.cat([z.detach(), doh], 1)), ytr)
            optp.zero_grad(); la.backward(); optp.step()
        # Step B
        z = enc(Xtr); logits = clf(z); ce = F.cross_entropy(logits, ytr, weight=cew)
        yoh = F.one_hot(ytr, 2).float(); doh = F.one_hot(dtr, nd).float()
        logq = F.log_softmax(qD(torch.cat([z, yoh], 1)), 1)
        Ienc = (logq.exp() * (logq - log_piy[ytr])).sum(1).mean()          # I(Z;D|Y)
        Idec = ce - F.cross_entropy(hY(torch.cat([z, doh], 1)), ytr)       # I(Y;D|Z)=H(Y|Z)-H(Y|Z,D)
        loss = ce + lam_enc * w * Ienc + lam_dec * w * Idec
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, clf, qD, hY, log_piy


@torch.no_grad()
def evaluate(enc, clf, Xte, yte):
    return (clf(enc(Xte)).argmax(1) == yte).float().mean().item() * 100


def measure_cmi(enc, Xtr, ytr, dtr):
    """Fit FRESH probes on the frozen Z to measure I(Z;D|Y) and I(Y;D|Z) (held-out estimate)."""
    nd = int(dtr.max()) + 1
    with torch.no_grad():
        z = enc(Xtr)
    qD = mlp(z.shape[1] + 2, nd); hY = mlp(z.shape[1] + nd, 2); qY = mlp(z.shape[1], 2)
    op = torch.optim.Adam(list(qD.parameters()) + list(hY.parameters()) + list(qY.parameters()), 2e-3)
    yoh = F.one_hot(ytr, 2).float(); doh = F.one_hot(dtr, nd).float()
    pi = torch.zeros(2, nd)
    for y, d in zip(ytr, dtr): pi[y, d] += 1
    log_piy = torch.log((pi + 1) / (pi.sum(1, keepdim=True) + nd))
    for _ in range(300):
        lq = F.cross_entropy(qD(torch.cat([z, yoh], 1)), dtr)
        lh = F.cross_entropy(hY(torch.cat([z, doh], 1)), ytr)
        ly = F.cross_entropy(qY(z), ytr)
        op.zero_grad(); (lq + lh + ly).backward(); op.step()
    with torch.no_grad():
        logq = F.log_softmax(qD(torch.cat([z, yoh], 1)), 1)
        Ienc = (logq.exp() * (logq - log_piy[ytr])).sum(1).mean().item()
        Idec = (F.cross_entropy(qY(z), ytr) - F.cross_entropy(hY(torch.cat([z, doh], 1)), ytr)).item()
    return max(Ienc, 0), max(Idec, 0)


def run_matrix():
    METHODS = [("erm", 0, 0, False), ("enc I(Z;D|Y)", 0.5, 0, False), ("dec I(Y;D|Z)", 0, 0.5, False),
               ("dual", 0.5, 0.5, False), ("dual+BER", 0.5, 0.5, True)]
    DGPS = [("covariate-only", dict(cov=1.5, con=0.0, labelshift=False)),
            ("concept-only",   dict(cov=0.0, con=1.5, labelshift=False)),
            ("all-three",      dict(cov=1.5, con=1.5, labelshift=True))]
    print(f"{'DGP':16s} {'method':14s} {'tgtAcc':>7s} {'I(Z;D|Y)':>9s} {'I(Y;D|Z)':>9s}")
    print("-" * 60)
    for dname, kw in DGPS:
        accs = {}
        for mname, le, ld, bal in METHODS:
            a = []; ie = []; idc = []
            for sd in range(3):                       # 3 seeds
                Xtr, ytr, dtr, Xte, yte = gen(seed=sd, **kw)
                enc, clf, *_ = train(Xtr, ytr, dtr, le, ld, bal, K=4, seed=sd)
                a.append(evaluate(enc, clf, Xte, yte))
                e, d = measure_cmi(enc, Xtr, ytr, dtr); ie.append(e); idc.append(d)
            print(f"{dname:16s} {mname:14s} {np.mean(a):6.1f}  {np.mean(ie):8.3f} {np.mean(idc):8.3f}")
        print("-" * 60)


def run_tension():
    print("\n=== TENSION: under label shift, pushing I(Z;D|Y)->0 raises I(Y;D|Z) ===")
    print(f"{'lam_enc':>8s} {'I(Z;D|Y)':>9s} {'I(Y;D|Z)':>9s} {'tgtAcc':>7s}")
    for le in [0.0, 0.3, 1.0, 3.0]:
        ie = []; idc = []; a = []
        for sd in range(3):
            Xtr, ytr, dtr, Xte, yte = gen(cov=1.5, con=0.0, labelshift=True, seed=sd)
            enc, clf, *_ = train(Xtr, ytr, dtr, le, 0.0, False, K=4, seed=sd)
            e, d = measure_cmi(enc, Xtr, ytr, dtr); ie.append(e); idc.append(d); a.append(evaluate(enc, clf, Xte, yte))
        print(f"{le:8.1f} {np.mean(ie):8.3f} {np.mean(idc):8.3f} {np.mean(a):6.1f}")


if __name__ == "__main__":
    run_matrix()
    run_tension()

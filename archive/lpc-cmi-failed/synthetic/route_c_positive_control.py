"""Positive control for Route C's RESIDUAL decoder CMI (intercept-only h0 vs full h).
Validates that R_dec_res = CE(h0)-CE(h) has POWER (rises under a real domain-dependent decision boundary)
AND specificity (stays ~0 under label-prior/calibration shift alone). Without this control, a null reading
on real EEG is ambiguous (no concept shift vs weak probe). Operates on synthetic latent Z directly.

Three regimes (K source domains, binary y):
  NULL-prior   : shared boundary w; domain shifts only the INTERCEPT b_d (label prior/calibration).
                 -> raw I(Y;D|Z)=CE(a)-CE(h) > 0 (prior artifact) but RESIDUAL CE(h0)-CE(h) ~ 0.
  CONCEPT      : boundary w_d FLIPS sign in half the domains -> RESIDUAL > 0 (genuine boundary change).
  SUBJECT-degen: y deterministic given d (each d one class) -> raw ~ H(Y|Z) (artifact) but RESIDUAL ~ 0
                 (b_d absorbs d->y). Mirrors the ADFTD D=subject degeneracy.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def fit_decoders(Z, y, d, n_dom, epochs=300, lr=2e-3, seed=0):
    """Fit a(Y|Z) [blind], h(Y|Z,D) [full], h0=u(Z)+b_D [intercept-only]; return raw & residual on a held split."""
    torch.manual_seed(seed)
    n = len(y); idx = np.random.default_rng(seed).permutation(n); cut = n // 2
    fit, ev = idx[:cut], idx[cut:]
    Z = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y); dt = torch.tensor(d)
    doh = F.one_hot(dt, n_dom).float()
    a = mlp(Z.shape[1], 2); h = mlp(Z.shape[1] + n_dom, 2); u = mlp(Z.shape[1], 2)
    bD = torch.zeros(n_dom, 2, requires_grad=True)
    opt = torch.optim.Adam(list(a.parameters()) + list(h.parameters()) + list(u.parameters()) + [bD], lr=lr)
    F_ = lambda t: torch.tensor(t)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (F.cross_entropy(a(Z[fit]), yt[fit])
                + F.cross_entropy(h(torch.cat([Z[fit], doh[fit]], 1)), yt[fit])
                + F.cross_entropy(u(Z[fit]) + bD[dt[fit]], yt[fit]))
        loss.backward(); opt.step()
    with torch.no_grad():
        ce_a = F.cross_entropy(a(Z[ev]), yt[ev]).item()
        ce_h = F.cross_entropy(h(torch.cat([Z[ev], doh[ev]], 1)), yt[ev]).item()
        ce_0 = F.cross_entropy(u(Z[ev]) + bD[dt[ev]], yt[ev]).item()
    return dict(raw=max(ce_a - ce_h, 0.0), residual=max(ce_0 - ce_h, 0.0))


def gen(regime, K=4, n=1500, dz=4, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, dz)
    b_d = rng.normal(0, 2.0, K)                      # per-domain intercept (prior/calibration)
    sign_d = np.array([1, 1, -1, -1.0])[:K]          # boundary flip for concept regime
    Z, Y, D = [], [], []
    for d in range(K):
        z = rng.normal(0, 1, (n, dz))
        if regime == "null-prior":
            logit = z @ w + b_d[d]                    # shared boundary, domain shifts intercept only
        elif regime == "concept":
            logit = sign_d[d] * (z @ w)               # boundary FLIPS per domain
        elif regime == "subject-degen":
            y = np.full(n, d % 2)                     # label deterministic given domain
            Z.append(z); Y.append(y); D.append(np.full(n, d)); continue
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        Z.append(z); Y.append(y); D.append(np.full(n, d))
    return np.concatenate(Z).astype("float32"), np.concatenate(Y), np.concatenate(D)


if __name__ == "__main__":
    print(f"{'regime':14s} {'raw I(Y;D|Z)':>13s} {'RESIDUAL':>10s}   interpretation")
    print("-" * 70)
    exp = {"null-prior": "raw>0 (prior), residual~0", "concept": "BOTH > 0 (real boundary shift)",
           "subject-degen": "raw~H(Y|Z) artifact, residual~0"}
    for reg in ["null-prior", "concept", "subject-degen"]:
        raws, ress = [], []
        for sd in range(4):
            Z, y, d = gen(reg, seed=sd)
            r = fit_decoders(Z, y, d, int(d.max()) + 1, seed=sd)
            raws.append(r["raw"]); ress.append(r["residual"])
        print(f"{reg:14s} {np.mean(raws):13.3f} {np.mean(ress):10.3f}   ({exp[reg]})")
    print("\nPASS criterion: concept residual >> null-prior residual ~ subject-degen residual (~0).")

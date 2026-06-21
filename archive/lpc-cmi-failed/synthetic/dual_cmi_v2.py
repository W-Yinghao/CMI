"""dual_cmi_v2.py  --  IMPROVED controlled validation of the DUAL-CMI framework.

Fixes the flaws of synthetic/dual_cmi_validation.py:

  (1) HELD-OUT MEASUREMENT. Both CMIs are measured by fitting FRESH probes on the
      FROZEN encoder over a split that the encoder/classifier never saw. We split
      every SOURCE domain into a fit-half (train encoder) and a held-half. Probes
      q(D|Z,Y), h(Y|Z,D), q(Y|Z) are trained on the held-half of one set of seeds'
      data and EVALUATED on a disjoint eval-half. So the reported I(Z;D|Y) and
      I(Y;D|Z) are genuine held-out (generalization) estimates, not memorization.

  (2) STRONG CONCEPT ARM. The concept feature is made to DOMINATE: its class-signal
      magnitude (con) is large and the INVARIANT core is deliberately weak (low SNR),
      so an unpenalized ERM encoder PREFERS the concept shortcut. The target domain
      FLIPS the concept->label map (sign = -1 on target), so leaning on concept is
      catastrophic on the target unless the decoder term I(Y;D|Z) forces the encoder
      to drop it. This produces a clearly measurable I(Y;D|Z) for ERM/enc-only.

  (3) dual+LC = dual + PRINCIPLED per-domain LABEL CORRECTION (GLS / Combes'20).
      Each domain d is importance-reweighted by w_d(y) = pi*(y) / pi_d(y) toward a
      common reference prior pi* (uniform). Weights enter BOTH the CE loss AND the
      decoder cross-entropy term, so in the reweighted distribution I~(Y;D)=0 and the
      two CMIs decouple (anchor A4). This is NOT crude per-batch class balancing: the
      weight depends on (y, d) jointly via the per-domain empirical prior.

  (4) EXACT-MI tension sweep. A separate closed-form numpy function builds the
      discrete construction behind anchor A2/A3 and verifies the identity
      I(Z;D|Y) - I(Y;D|Z) = I(Z;D) - I(Y;D) exactly, then sweeps how forcing
      I(Z;D|Y)->0 trades off against I(Y;D|Z) under label shift.

Run matrix: 3 DGPs {covariate-only, concept-only, all-three}
            x {erm, enc, dec, dual, dual+LC} x 5 seeds, plus the exact tension sweep.

torch is required: run with /home/infres/yinwang/anaconda3/envs/icml/bin/python
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
# Each example x = [ core(2) | cov(2) | con(2) ].
#   core : INVARIANT class signal, but deliberately LOW SNR (weak) so it is not the
#          easy shortcut. Generalizes across all domains.
#   cov  : COVARIATE shift -- class-conditional feature whose MEAN is offset per
#          domain. The y->feature relation is the same sign everywhere, but the
#          per-domain offset makes naive use leak domain identity: kill with I(Z;D|Y).
#   con  : CONCEPT shift -- a STRONG, high-SNR class feature whose sign of the
#          feature->label map FLIPS on the target domain. Unpenalized ERM loves it
#          (high SNR) but it INVERTS on target: kill with I(Y;D|Z).
#
# CORE_SNR < CON_SNR on purpose so the concept shortcut dominates unless penalized.
CORE_SNR = 0.55     # weak invariant signal
COV_SNR  = 1.10     # covariate-shortcut class signal
CON_SNR  = 2.20     # concept-shortcut class signal  (DOMINATES core)


def gen(cov, con, labelshift, n=2000, K=4, seed=0):
    """Generate K source domains (0..K-1) + 1 target (K).

    Returns the raw arrays so the caller can carve fit/held/eval splits itself.
    `cov`, `con` in {0,1} switch each shift on/off (scaled by the *_SNR constants).
    """
    rng = np.random.default_rng(seed)
    doms = list(range(K + 1))
    # per-domain covariate mean offset (domain identity leaks through cov channel)
    shift_d = rng.normal(0.0, 1.6, K + 1)
    # concept sign: all sources +1, target flips to -1 (feature->label map inverts)
    sign_d = np.ones(K + 1)
    sign_d[K] = -1.0
    # also perturb source concept signs mildly so concept leaks domain id among sources
    if K >= 2:
        sign_d[1] = -1.0  # one source already flipped -> concept distinguishes domains
    if labelshift:
        base = np.array([0.22, 0.40, 0.60, 0.78, 0.50], dtype=float)
        pi_d = base[: K + 1].copy()
    else:
        pi_d = np.full(K + 1, 0.5)

    Xs, Ys, Ds = [], [], []
    for d in doms:
        y = (rng.random(n) < pi_d[d]).astype(int)
        s = (2 * y - 1).astype(float)                      # +-1 class sign
        core = rng.normal(0, 1, (n, 2)) + CORE_SNR * s[:, None]
        covf = rng.normal(0, 1, (n, 2)) + cov * (COV_SNR * s[:, None] + shift_d[d])
        conf = rng.normal(0, 1, (n, 2)) + con * (CON_SNR * sign_d[d] * s[:, None])
        Xs.append(np.concatenate([core, covf, conf], 1))
        Ys.append(y)
        Ds.append(np.full(n, d))
    X = np.concatenate(Xs).astype("float32")
    Y = np.concatenate(Ys).astype("int64")
    D = np.concatenate(Ds).astype("int64")
    return X, Y, D, pi_d


def split_sources(X, Y, D, K, frac_fit=0.5, seed=0):
    """Split each SOURCE domain into a fit-half and a held-half; target stays whole.

    fit  : encoder + classifier training data
    held : disjoint source data used ONLY to fit/eval the CMI probes (held-out)
    tgt  : target domain (D==K), used for the accuracy generalization metric
    """
    rng = np.random.default_rng(1000 + seed)
    src = D < K
    fit_mask = np.zeros(len(D), dtype=bool)
    for d in range(K):
        idx = np.where(D == d)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * frac_fit)
        fit_mask[idx[:cut]] = True
    held_mask = src & (~fit_mask)
    tgt_mask = D == K

    def t(m, arr):
        return torch.tensor(arr[m])

    return dict(
        Xfit=t(fit_mask, X), yfit=t(fit_mask, Y), dfit=t(fit_mask, D),
        Xheld=t(held_mask, X), yheld=t(held_mask, Y), dheld=t(held_mask, D),
        Xtgt=t(tgt_mask, X), ytgt=t(tgt_mask, Y),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def per_domain_logprior(y, d, nd, nclass=2):
    """log pi_d(y) : empirical per-domain class log-prior (Laplace-smoothed)."""
    pi = torch.zeros(nclass, nd)
    for yy, dd in zip(y.tolist(), d.tolist()):
        pi[yy, dd] += 1.0
    pi = (pi + 1.0) / (pi.sum(0, keepdim=True) + nclass)   # normalise over y within d
    return torch.log(pi)                                   # [nclass, nd]


def gls_weights(y, d, nd, nclass=2):
    """Principled GLS per-domain label-correction weights w_d(y)=pi*(y)/pi_d(y).

    Reference prior pi* = uniform over classes (1/nclass). Per-sample weight is
    w = pi*(y_i) / pi_{d_i}(y_i). Normalised so mean weight = 1 (keeps loss scale).
    Returns a per-sample weight tensor aligned with (y,d).
    """
    pi = torch.zeros(nclass, nd)
    for yy, dd in zip(y.tolist(), d.tolist()):
        pi[yy, dd] += 1.0
    pi_d = (pi + 0.5) / (pi.sum(0, keepdim=True) + 0.5 * nclass)   # pi_d(y) per domain
    pistar = torch.full((nclass,), 1.0 / nclass)
    w = torch.empty(len(y))
    for i, (yy, dd) in enumerate(zip(y.tolist(), d.tolist())):
        w[i] = pistar[yy] / pi_d[yy, dd]
    w = w / w.mean()
    return w


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(s, lam_enc, lam_dec, label_correct, K, epochs=200, zc=16, seed=0):
    """Train encoder+classifier with CE + lam_enc*I(Z;D|Y) + lam_dec*I(Y;D|Z).

    If label_correct: apply GLS per-domain reweighting w_d(y)=pi*(y)/pi_d(y) to BOTH
    the CE term and the decoder cross-entropy terms so I~(Y;D)=0 (anchor A4).
    """
    torch.manual_seed(seed)
    X, y, d = s["Xfit"], s["yfit"], s["dfit"]
    nd = K
    enc = mlp(6, zc)
    clf = nn.Linear(zc, 2)
    qD = mlp(zc + 2, nd)        # q(D|Z,Y)
    hY = mlp(zc + nd, 2)        # h(Y|Z,D)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), 1e-3)
    optp = torch.optim.Adam(list(qD.parameters()) + list(hY.parameters()), 2e-3)

    log_piy = per_domain_logprior(y, d, nd)        # [2, nd]; log pi_d(y)
    if label_correct:
        wgts = gls_weights(y, d, nd)               # per-sample GLS weights
    else:
        wgts = torch.ones(len(y))

    yoh = F.one_hot(y, 2).float()
    doh = F.one_hot(d, nd).float()

    for ep in range(epochs):
        w = min(1.0, ep / 50.0)                    # ramp the regularisers
        # --- Step A: fit variational posteriors on detached Z ---
        z = enc(X).detach()
        for _ in range(3):
            la = (F.cross_entropy(qD(torch.cat([z, yoh], 1)), d)
                  + F.cross_entropy(hY(torch.cat([z, doh], 1)), y))
            optp.zero_grad(); la.backward(); optp.step()
        # --- Step B: encoder + classifier ---
        z = enc(X)
        logits = clf(z)
        ce = (F.cross_entropy(logits, y, reduction="none") * wgts).mean()
        # I(Z;D|Y) = E[ KL( q(D|z,y) || pi_d(y-marginal over d) ) ]  (variational lower est.)
        logq = F.log_softmax(qD(torch.cat([z, yoh], 1)), 1)
        # reference: marginal p(D|Y) implied by the per-domain priors -> use uniform-over-D
        # baseline log p(D|Y); we approximate log p(D|Y=y) by the empirical domain freq per y.
        Ienc = (logq.exp() * (logq - _logpD_given_Y(y, d, nd))).sum(1).mean()
        # I(Y;D|Z) = H(Y|Z) - H(Y|Z,D) ; reweighted if label_correct
        ce_yz = (F.cross_entropy(logits, y, reduction="none") * wgts).mean()        # H(Y|Z) proxy
        ce_yzd = (F.cross_entropy(hY(torch.cat([z, doh], 1)), y, reduction="none") * wgts).mean()
        Idec = ce_yz - ce_yzd
        loss = ce + lam_enc * w * Ienc + lam_dec * w * Idec
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, clf


def _logpD_given_Y(y, d, nd, nclass=2):
    """log p(D=d | Y=y) empirical, returned per-sample as [N, nd] reference for KL."""
    cnt = torch.zeros(nclass, nd)
    for yy, dd in zip(y.tolist(), d.tolist()):
        cnt[yy, dd] += 1.0
    p = (cnt + 1.0) / (cnt.sum(1, keepdim=True) + nd)     # p(D|Y), rows sum to 1
    logp = torch.log(p)                                   # [nclass, nd]
    return logp[y]                                        # [N, nd]


# ---------------------------------------------------------------------------
# Evaluation + HELD-OUT CMI measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def accuracy(enc, clf, X, y):
    return (clf(enc(X)).argmax(1) == y).float().mean().item() * 100.0


def measure_cmi_heldout(enc, s, K, probe_epochs=400, seed=0):
    """Measure I(Z;D|Y) and I(Y;D|Z) on a HELD-OUT split with fresh probes.

    Probes are TRAINED on the first half of the held-source data and EVALUATED on
    the disjoint second half -> genuine generalization (held-out) MI estimate.
    The encoder is FROZEN (no grad through it).
    """
    torch.manual_seed(2000 + seed)
    nd = K
    Xh, yh, dh = s["Xheld"], s["yheld"], s["dheld"]
    n = len(yh)
    perm = torch.randperm(n)
    ptr, pev = perm[: n // 2], perm[n // 2:]

    with torch.no_grad():
        Z = enc(Xh)
    zc = Z.shape[1]
    qD = mlp(zc + 2, nd)
    hY = mlp(zc + nd, 2)
    qY = mlp(zc, 2)
    op = torch.optim.Adam(list(qD.parameters()) + list(hY.parameters()) + list(qY.parameters()), 2e-3)

    yoh = F.one_hot(yh, 2).float()
    doh = F.one_hot(dh, nd).float()
    logpD_Y = _logpD_given_Y(yh, dh, nd)   # reference computed on held data

    Ztr, Zev = Z[ptr], Z[pev]
    for _ in range(probe_epochs):
        lq = F.cross_entropy(qD(torch.cat([Ztr, yoh[ptr]], 1)), dh[ptr])
        lh = F.cross_entropy(hY(torch.cat([Ztr, doh[ptr]], 1)), yh[ptr])
        ly = F.cross_entropy(qY(Ztr), yh[ptr])
        op.zero_grad(); (lq + lh + ly).backward(); op.step()

    with torch.no_grad():
        # I(Z;D|Y): KL( q(D|z,y) || p(D|y) ) averaged over the EVAL half
        logq = F.log_softmax(qD(torch.cat([Zev, yoh[pev]], 1)), 1)
        Ienc = (logq.exp() * (logq - logpD_Y[pev])).sum(1).mean().item()
        # I(Y;D|Z) = H(Y|Z) - H(Y|Z,D) on the EVAL half
        hyz = F.cross_entropy(qY(Zev), yh[pev]).item()
        hyzd = F.cross_entropy(hY(torch.cat([Zev, doh[pev]], 1)), yh[pev]).item()
        Idec = hyz - hyzd
    return max(Ienc, 0.0), max(Idec, 0.0)


# ---------------------------------------------------------------------------
# EXACT discrete MI tension sweep (closed form, numpy)
# ---------------------------------------------------------------------------
def _H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def exact_cmis(pZYD):
    """Given joint p(Z,Y,D) as a 3-D array, return (I(Z;D|Y), I(Y;D|Z), I(Z;D), I(Y;D))."""
    p = pZYD / pZYD.sum()
    pY = p.sum((0, 2))        # p(Y)
    pZ = p.sum((1, 2))        # p(Z)
    pD = p.sum((0, 1))        # p(D)
    pZD = p.sum(1)            # p(Z,D)
    pYD = p.sum(0)            # p(Y,D)
    pZY = p.sum(2)            # p(Z,Y)

    # I(Z;D|Y) = sum_y p(y) [ H(Z|y)+H(D|y)-H(Z,D|y) ]
    Izd_y = 0.0
    for yi in range(p.shape[1]):
        py = pY[yi]
        if py <= 0:
            continue
        pZy = p[:, yi, :].sum(1) / py     # p(Z|y)
        pDy = p[:, yi, :].sum(0) / py     # p(D|y)
        pZDy = p[:, yi, :] / py           # p(Z,D|y)
        Izd_y += py * (_H(pZy) + _H(pDy) - _H(pZDy))

    # I(Y;D|Z) = sum_z p(z) [ H(Y|z)+H(D|z)-H(Y,D|z) ]
    Iyd_z = 0.0
    for zi in range(p.shape[0]):
        pz = pZ[zi]
        if pz <= 0:
            continue
        pYz = p[zi, :, :].sum(1) / pz
        pDz = p[zi, :, :].sum(0) / pz
        pYDz = p[zi, :, :] / pz
        Iyd_z += pz * (_H(pYz) + _H(pDz) - _H(pYDz))

    Izd = _H(pZ) + _H(pD) - _H(pZD)       # I(Z;D)
    Iyd = _H(pY) + _H(pD) - _H(pYD)       # I(Y;D)
    return Izd_y, Iyd_z, Izd, Iyd


def build_joint(alpha, label_shift, nZ=2):
    """Construct p(Z,Y,D) for a 2-domain, binary-Y, binary/lowdim-Z toy.

    - Z is conditionally invariant given Y: p(Z|Y,D)=p(Z|Y), controlled by `alpha`
      (alpha = how cleanly Z encodes Y; alpha=1 -> Z=Y deterministically -> zero Bayes
      error; alpha<1 -> irreducible label noise so I(Z;D|Y)=0 can still leave I(Y;D|Z)).
    - Label shift: per-domain prior pi_d(Y) differs when label_shift>0.
    """
    D = 2
    Y = 2
    # per-domain priors
    if label_shift > 0:
        piY_d = np.array([[0.5 + label_shift, 0.5 - label_shift],
                          [0.5 - label_shift, 0.5 + label_shift]])  # [D, Y]
    else:
        piY_d = np.array([[0.5, 0.5], [0.5, 0.5]])
    pD = np.array([0.5, 0.5])
    # p(Z|Y): domain-invariant channel.  alpha controls cleanliness.
    pZ_Y = np.array([[alpha, 1 - alpha],
                     [1 - alpha, alpha]])          # [Y, Z]
    p = np.zeros((nZ, Y, D))
    for di in range(D):
        for yi in range(Y):
            for zi in range(nZ):
                p[zi, yi, di] = pD[di] * piY_d[di, yi] * pZ_Y[yi, zi]
    return p


def run_tension_exact():
    print("\n=== EXACT discrete MI tension sweep (closed form numpy) ===")
    print("Construction: Z conditionally invariant given Y  =>  I(Z;D|Y)=0 by design.")
    print("Identity checked:  I(Z;D|Y) - I(Y;D|Z)  ==  I(Z;D) - I(Y;D)\n")
    print(f"{'alpha':>6s} {'lblshift':>8s} {'I(Z;D|Y)':>9s} {'I(Y;D|Z)':>9s} "
          f"{'I(Z;D)':>8s} {'I(Y;D)':>8s} {'identity_resid':>15s}")
    for ls in [0.0, 0.15, 0.30]:
        for alpha in [0.70, 0.85, 0.95, 1.00]:
            p = build_joint(alpha, ls)
            izdy, iydz, izd, iyd = exact_cmis(p)
            resid = (izdy - iydz) - (izd - iyd)
            print(f"{alpha:6.2f} {ls:8.2f} {izdy:9.5f} {iydz:9.5f} "
                  f"{izd:8.5f} {iyd:8.5f} {resid:15.2e}")
    print("\nReading: with label shift (lblshift>0) and irreducible noise (alpha<1),")
    print("I(Z;D|Y)=0 forces I(Y;D|Z)=I(Y;D)-I(Z;D)>0  (anchor A2).  Only at alpha=1")
    print("(zero Bayes error) can both CMIs vanish together (anchor A3).")


# ---------------------------------------------------------------------------
# Main experiment matrix
# ---------------------------------------------------------------------------
def run_matrix(nseeds=5, K=4):
    METHODS = [
        ("erm",       0.0, 0.0, False),
        ("enc",       1.0, 0.0, False),
        ("dec",       0.0, 1.0, False),
        ("dual",      0.7, 0.7, False),
        ("dual+LC",   0.7, 0.7, True),
    ]
    DGPS = [
        ("covariate-only", dict(cov=1, con=0, labelshift=False)),
        ("concept-only",   dict(cov=0, con=1, labelshift=False)),
        ("all-three",      dict(cov=1, con=1, labelshift=True)),
    ]
    print("=" * 78)
    print("DUAL-CMI v2  --  held-out CMI measurement, strong concept arm, GLS label-correction")
    print(f"core_snr={CORE_SNR}  cov_snr={COV_SNR}  con_snr={CON_SNR}  "
          f"(concept dominates core on purpose)")
    print("=" * 78)
    print(f"{'DGP':16s} {'method':9s} {'tgtAcc':>7s} {'+-':>5s} "
          f"{'I(Z;D|Y)':>9s} {'I(Y;D|Z)':>9s}")
    print("-" * 78)
    for dname, kw in DGPS:
        for mname, le, ld, lc in METHODS:
            accs, ies, ids = [], [], []
            for sd in range(nseeds):
                X, Y, D, _ = gen(seed=sd, K=K, **kw)
                s = split_sources(X, Y, D, K, seed=sd)
                enc, clf = train(s, le, ld, lc, K=K, seed=sd)
                accs.append(accuracy(enc, clf, s["Xtgt"], s["ytgt"]))
                ie, idc = measure_cmi_heldout(enc, s, K, seed=sd)
                ies.append(ie); ids.append(idc)
            print(f"{dname:16s} {mname:9s} {np.mean(accs):6.1f} "
                  f"{np.std(accs):5.1f} {np.mean(ies):9.4f} {np.mean(ids):9.4f}")
        print("-" * 78)


def run_tension_learned(nseeds=5, K=4):
    """Learned-model tension: under label shift, sweeping lam_enc up (forcing
    I(Z;D|Y)->0) should RAISE the held-out I(Y;D|Z) unless label-correction is on."""
    print("\n=== LEARNED tension: forcing I(Z;D|Y)->0 raises I(Y;D|Z) under label shift ===")
    print("(covariate+label shift DGP; no decoder term, no LC -> the encoder-only")
    print(" pressure should re-route domain info into the decoder gap)")
    print(f"{'lam_enc':>8s} {'I(Z;D|Y)':>9s} {'I(Y;D|Z)':>9s} {'tgtAcc':>7s}")
    for le in [0.0, 0.5, 1.5, 4.0]:
        ies, ids, accs = [], [], []
        for sd in range(nseeds):
            X, Y, D, _ = gen(cov=1, con=0, labelshift=True, seed=sd, K=K)
            s = split_sources(X, Y, D, K, seed=sd)
            enc, clf = train(s, le, 0.0, False, K=K, seed=sd)
            ie, idc = measure_cmi_heldout(enc, s, K, seed=sd)
            ies.append(ie); ids.append(idc); accs.append(accuracy(enc, clf, s["Xtgt"], s["ytgt"]))
        print(f"{le:8.1f} {np.mean(ies):9.4f} {np.mean(ids):9.4f} {np.mean(accs):6.1f}")


if __name__ == "__main__":
    run_tension_exact()
    print()
    run_matrix(nseeds=5)
    run_tension_learned(nseeds=5)

#!/usr/bin/env python3
"""
verify_tension.py  --  Numerical verification for PILLAR 1 (Tension Theorem).

Run with:  /home/infres/yinwang/anaconda3/bin/python3 verify_tension.py

Verifies, by exact computation on discrete joint distributions p(Z,Y,D):

  (i)  IDENTITY A1:  I(Z;D|Y) - I(Y;D|Z)  ==  I(Z;D) - I(Y;D)
       over 3000 random joints; reports max abs error.

  (ii) LABEL-SHIFT SWEEP:  a conditionally-invariant encoder p(z|y) shared
       across domains, with per-domain label priors pi_d(y) whose spread
       grows with a knob s in [0,1].  Shows  I(Z;D|Y) = 0 (encoder is
       conditionally invariant) while the FORCED decoder leakage
       I(Y;D|Z) = I(Y;D) - I(Z;D) > 0 grows monotonically with label shift.

  (iii) GLS RESOLUTION (A4, sanity):  reweight each domain by
        w_d(y) = pi*(y)/pi_d(y) to a common reference prior pi*.  In the
        reweighted joint, I~(Y;D)=0 and BOTH CMIs collapse to ~0.

All entropies/MIs computed exactly from the joint table (no estimators),
in nats, with a numerically safe x*log(x) (0*log0 := 0).
"""

import numpy as np

np.random.seed(0)
EPS = 0.0  # we use safe xlogx; no smoothing needed for exact joints


# ----------------------------------------------------------------------
# Exact information-theoretic primitives on a 3-way joint table P[z,y,d].
# ----------------------------------------------------------------------
def _xlogx(p):
    """Elementwise p*log(p) with 0*log0 = 0."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    nz = p > 0
    out[nz] = p[nz] * np.log(p[nz])
    return out


def entropy(p):
    """H(.) of a (flattened) probability table, in nats."""
    return -_xlogx(p).sum()


def normalize(P):
    P = np.asarray(P, dtype=np.float64)
    s = P.sum()
    assert s > 0
    return P / s


def mi_pair(Pxy):
    """I(X;Y) from a 2D joint table P[x,y]."""
    Pxy = normalize(Pxy)
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    # I = sum P log( P / (Px Py) ) = H(X)+H(Y)-H(X,Y)
    return entropy(Px.ravel()) + entropy(Py.ravel()) - entropy(Pxy.ravel())


def cmi(P, a, b, c):
    """
    Conditional mutual information I(A;B|C) from a 3D joint P over axes
    (0,1,2) identified with variables (Z,Y,D).  a,b,c is a permutation of
    {0,1,2}.  Uses I(A;B|C) = H(A,C)+H(B,C)-H(C)-H(A,B,C).
    """
    P = normalize(P)
    H_ABC = entropy(P.ravel())
    H_C = entropy(P.sum(axis=tuple(x for x in range(3) if x != c)).ravel())
    # marginal over (A,C): sum out b
    H_AC = entropy(P.sum(axis=b).ravel())
    # marginal over (B,C): sum out a
    H_BC = entropy(P.sum(axis=a).ravel())
    return H_AC + H_BC - H_C - H_ABC


# Convenience wrappers (axis convention: 0=Z, 1=Y, 2=D)
def I_ZD_given_Y(P):
    return cmi(P, 0, 2, 1)        # A=Z, B=D, C=Y


def I_YD_given_Z(P):
    return cmi(P, 1, 2, 0)        # A=Y, B=D, C=Z


def I_ZD(P):
    return mi_pair(P.sum(axis=1))  # joint over (Z,D)


def I_YD(P):
    return mi_pair(P.sum(axis=0))  # joint over (Y,D)


def I_ZY(P):
    return mi_pair(P.sum(axis=2))  # joint over (Z,Y)


# ----------------------------------------------------------------------
# (i)  IDENTITY  A1  over random joints
# ----------------------------------------------------------------------
def verify_identity(n_trials=3000, max_card=5):
    rng = np.random.default_rng(12345)
    max_err = 0.0
    worst = None
    for _ in range(n_trials):
        nz = rng.integers(2, max_card + 1)
        ny = rng.integers(2, max_card + 1)
        nd = rng.integers(2, max_card + 1)
        # random joint; mix Dirichlet draw + occasional sparsity to stress xlog0
        P = rng.random((nz, ny, nd)) ** rng.integers(1, 4)
        if rng.random() < 0.3:
            P *= (rng.random((nz, ny, nd)) > 0.4)  # zero out some cells
        if P.sum() == 0:
            continue
        P = normalize(P)

        lhs = I_ZD_given_Y(P) - I_YD_given_Z(P)
        rhs = I_ZD(P) - I_YD(P)
        err = abs(lhs - rhs)
        if err > max_err:
            max_err = err
            worst = (nz, ny, nd)
    return max_err, worst, n_trials


# ----------------------------------------------------------------------
# (ii) LABEL-SHIFT SWEEP with a conditionally-invariant encoder.
#
# Construction:
#   - Binary label Y in {0,1}; encoder p(z|y) SHARED across domains
#     (=> conditional invariance => I(Z;D|Y)=0 exactly).
#   - K domains, each with its own label prior pi_d(y).
#   - knob s in [0,1] controls how far the per-domain priors spread from
#     the balanced prior (0.5,0.5).  s=0 -> no label shift; s->1 -> max.
# ----------------------------------------------------------------------
def build_sweep_joint(s, K=2, nz=4):
    """
    Returns joint P[z,y,d] (uniform domain prior p(d)=1/K).
    p(z|y) is a fixed, well-separated, domain-INVARIANT encoder.
    pi_d(y) spreads with s.
    """
    rng = np.random.default_rng(7)
    # Fixed domain-invariant encoder p(z|y): two distinct soft clusters.
    # Built once (deterministic given seed) and reused for every s and d.
    enc = np.zeros((nz, 2))
    base0 = np.array([0.55, 0.25, 0.15, 0.05])[:nz]
    base1 = base0[::-1].copy()
    enc[:, 0] = base0 / base0.sum()
    enc[:, 1] = base1 / base1.sum()

    # Per-domain label priors: domain d pushed toward class d%2 by amount s.
    pis = []
    for d in range(K):
        if d % 2 == 0:
            p1 = 0.5 - 0.49 * s
        else:
            p1 = 0.5 + 0.49 * s
        pis.append(np.array([1.0 - p1, p1]))  # [pi(y=0), pi(y=1)]

    P = np.zeros((nz, 2, K))
    pd = 1.0 / K
    for d in range(K):
        for y in range(2):
            P[:, y, d] = pd * pis[d][y] * enc[:, y]
    return normalize(P), enc, pis


def run_sweep(svals, K=2, nz=4):
    rows = []
    for s in svals:
        P, enc, pis = build_sweep_joint(s, K=K, nz=nz)
        izd_y = I_ZD_given_Y(P)
        iyd_z = I_YD_given_Z(P)
        iyd = I_YD(P)
        izd = I_ZD(P)
        # identity check on the constructed family:
        forced = iyd - izd  # predicted I(Y;D|Z) under I(Z;D|Y)=0
        rows.append(dict(s=s, I_ZD_given_Y=izd_y, I_YD_given_Z=iyd_z,
                         I_YD=iyd, I_ZD=izd, forced=forced))
    return rows


# ----------------------------------------------------------------------
# (iii) GLS RESOLUTION (A4):  reweight domain d by w_d(y)=pi*(y)/pi_d(y).
# ----------------------------------------------------------------------
def gls_reweight(P, pi_star=None):
    """
    Given P[z,y,d], reweight so each domain matches a common label prior
    pi_star(y).  Returns a renormalized joint Ptilde with I~(Y;D)=0.
    """
    nz, ny, nd = P.shape
    if pi_star is None:
        pi_star = np.full(ny, 1.0 / ny)
    pd = P.sum(axis=(0, 1))                  # p(d)
    pyd = P.sum(axis=0)                       # p(y,d), shape (ny,nd)
    Pt = np.zeros_like(P)
    for d in range(nd):
        if pd[d] == 0:
            continue
        pi_d = pyd[:, d] / pd[d]              # pi_d(y)
        for y in range(ny):
            if pi_d[y] == 0:
                continue
            w = pi_star[y] / pi_d[y]
            Pt[:, y, d] = P[:, y, d] * w
    return normalize(Pt)


# ----------------------------------------------------------------------
# A2 anchor: invariant encoder + per-domain label prior, I(Z;D|Y)=0 exactly,
# tuned into the I(Y;D) ~ 0.08 nats regime of the confirmed anchor.
# ----------------------------------------------------------------------
def build_anchor_example():
    """
    Reproduce the CONFIRMED anchor A2 exactly.
      2 domains, binary Y, binary Z, p(d)=1/2.
      Invariant symmetric encoder p(z=1|y=1)=p(z=0|y=0)=a  => I(Z;D|Y)=0.
      Per-domain label priors p(y=1|d) = 0.5 -/+ p1  (symmetric label shift).
    The two free knobs (a, p1) were solved by bisection so that
      I(Y;D) = 0.0823  and  I(Z;D) = 0.0340  (nats),
    forcing I(Y;D|Z) = I(Y;D) - I(Z;D) = 0.0483 -- matching the anchor.
    """
    a = 0.8240664125872859     # invariant-encoder separation (solved)
    p1 = 0.20002020602627796   # half-gap of per-domain label prior (solved)
    enc = np.array([[a, 1.0 - a],
                    [1.0 - a, a]])             # rows z, cols y; p(z|y)
    P = np.zeros((2, 2, 2))
    prior = {0: np.array([1.0 - (0.5 - p1), 0.5 - p1]),
             1: np.array([1.0 - (0.5 + p1), 0.5 + p1])}
    for d in (0, 1):
        for y in (0, 1):
            P[:, y, d] = 0.5 * prior[d][y] * enc[:, y]
    return normalize(P)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PILLAR 1 -- TENSION THEOREM : numerical verification")
    print("=" * 70)

    # ---- (i) identity ----
    max_err, worst, n = verify_identity(n_trials=3000, max_card=5)
    print("\n[i] IDENTITY  I(Z;D|Y) - I(Y;D|Z) = I(Z;D) - I(Y;D)")
    print(f"    random discrete joints tested : {n}")
    print(f"    cardinalities sampled         : |Z|,|Y|,|D| in [2,5]")
    print(f"    MAX ABS ERROR                 : {max_err:.3e}   "
          f"(worst shape z,y,d={worst})")

    # ---- (ii) label-shift sweep ----
    print("\n[ii] LABEL-SHIFT SWEEP  (encoder p(z|y) domain-invariant)")
    print("     s = label-shift strength;  I(Z;D|Y) is held at 0 by")
    print("     construction;  forced I(Y;D|Z) = I(Y;D) - I(Z;D) grows.\n")
    svals = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    rows = run_sweep(svals, K=2, nz=4)
    hdr = (f"   {'s':>5} | {'I(Z;D|Y)':>11} | {'I(Y;D|Z)':>11} | "
           f"{'I(Y;D)':>9} | {'I(Z;D)':>9} | {'I(Y;D)-I(Z;D)':>13}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for r in rows:
        print(f"   {r['s']:>5.2f} | {r['I_ZD_given_Y']:>11.3e} | "
              f"{r['I_YD_given_Z']:>11.6f} | {r['I_YD']:>9.6f} | "
              f"{r['I_ZD']:>9.6f} | {r['forced']:>13.6f}")

    # consistency: forced == I(Y;D|Z) on this family
    cons = max(abs(r['I_YD_given_Z'] - r['forced']) for r in rows)
    inv = max(abs(r['I_ZD_given_Y']) for r in rows)
    print(f"\n    max |I(Y;D|Z) - (I(Y;D)-I(Z;D))| over sweep : {cons:.3e}")
    print(f"    max |I(Z;D|Y)| over sweep (encoder invariance): {inv:.3e}")
    monotone = all(rows[i]['I_YD_given_Z'] <= rows[i + 1]['I_YD_given_Z'] + 1e-12
                   for i in range(len(rows) - 1))
    print(f"    forced I(Y;D|Z) monotone non-decreasing in s : {monotone}")

    # ---- (iii) GLS resolution on the s=0.95 (strong-shift) joint ----
    print("\n[iii] GLS RESOLUTION (A4): reweight w_d(y)=pi*(y)/pi_d(y),")
    print("      pi* = uniform.  Decouples the two CMIs (both -> 0).")
    P_shift, _, _ = build_sweep_joint(0.95, K=2, nz=4)
    print(f"      BEFORE  : I(Y;D)={I_YD(P_shift):.6f}  "
          f"I(Z;D|Y)={I_ZD_given_Y(P_shift):.3e}  "
          f"I(Y;D|Z)={I_YD_given_Z(P_shift):.6f}")
    Pt = gls_reweight(P_shift)
    print(f"      AFTER   : I(Y;D)={I_YD(Pt):.3e}  "
          f"I(Z;D|Y)={I_ZD_given_Y(Pt):.3e}  "
          f"I(Y;D|Z)={I_YD_given_Z(Pt):.3e}")
    # encoder p(z|y) preserved by reweighting (only priors change):
    print(f"      (I(Z;Y) before={I_ZY(P_shift):.6f}, "
          f"after={I_ZY(Pt):.6f}  -- predictive info on Y retained)")

    # ---- (iv) reproduce the A2 constructed anchor numbers ----
    print("\n[iv] A2 ANCHOR CHECK (constructed example: invariant p(z|y) + label shift)")
    print("     Search a 2-domain binary-label joint with a fixed invariant")
    print("     encoder so that I(Z;D|Y)=0 and I(Y;D)=I(Z;D)+I(Y;D|Z),")
    print("     I(Y;D) ~ 0.08 nats (the regime of the confirmed anchor).")
    P_anchor = build_anchor_example()
    print(f"     I(Z;D|Y) = {I_ZD_given_Y(P_anchor):.4e}")
    print(f"     I(Y;D)   = {I_YD(P_anchor):.4f}")
    print(f"     I(Z;D)   = {I_ZD(P_anchor):.4f}")
    print(f"     I(Y;D|Z) = {I_YD_given_Z(P_anchor):.4f}  "
          f"(= I(Y;D)-I(Z;D) = {I_YD(P_anchor)-I_ZD(P_anchor):.4f})")

    print("\nDONE.")

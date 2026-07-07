"""verify_resolution.py  --  PILLAR 2 numerical verification.

Builds the A2 "constructed fight" example EXACTLY (reproduces the verified anchors
I(Y;D)=0.0823, I(Z;D)=0.0340, I(Z;D|Y)=0, I(Y;D|Z)=0.0483), then applies the GLS
importance-reweighting w_d(y) = pi*(y)/pi_d(y) and recomputes ALL FOUR information
quantities under the reweighted joint p~(z,y,d).

Claims verified numerically (exact discrete MI, no estimation):
  (PRE)  fight is real:        I(Z;D|Y)=0, I(Y;D)>0  =>  I(Y;D|Z)=I(Y;D)-I(Z;D)>0
  (P2a)  reweighting kills label shift:     I~(Y;D)=0
  (P2a') identity in reweighted world:      I~(Z;D|Y) = I~(Y;D|Z) + I~(Z;D)
  (P2b)  invariant encoder now FREE:        I~(Z;D|Y)=0  AND  I~(Y;D|Z)=0  simultaneously
         (because with I~(Y;D)=0 and an invariant p(z|y), both sides of A1 vanish)

Run:  /home/infres/yinwang/anaconda3/bin/python3 verify_resolution.py
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ----------------------------------------------------------------------------
# Exact discrete information measures over a 3-axis joint P[z, y, d].
# Axis convention everywhere: 0 = Z, 1 = Y, 2 = D.
# ----------------------------------------------------------------------------
def _mi2(Pab):
    """Mutual information of a 2-D joint table Pab (nats)."""
    pa = Pab.sum(1, keepdims=True)
    pb = Pab.sum(0, keepdims=True)
    m = Pab > 0
    return float((Pab[m] * np.log(Pab[m] / (pa @ pb)[m])).sum())


def MI(P, a, b):
    """I(A;B): marginalize P over the third axis, then 2-D MI."""
    c = ({0, 1, 2} - {a, b}).pop()
    Pab = P.sum(axis=c)
    if a > b:                       # ensure axis order (a, b)
        Pab = Pab.T
    return _mi2(Pab)


def CMI(P, a, b, c):
    """I(A;B|C) = sum_c p(c) I(A;B | C=c)."""
    Q = np.transpose(P, (a, b, c))
    tot = 0.0
    for k in range(Q.shape[2]):
        Pc = Q[:, :, k]
        pc = Pc.sum()
        if pc > 0:
            tot += pc * _mi2(Pc / pc)
    return tot


def report(P, tag):
    Izd  = MI(P, 0, 2)          # I(Z;D)
    Iyd  = MI(P, 1, 2)          # I(Y;D)
    Izdy = CMI(P, 0, 2, 1)      # I(Z;D|Y)   (encoder leakage)
    Iydz = CMI(P, 1, 2, 0)      # I(Y;D|Z)   (decoder / concept term)
    print(f"  [{tag}]")
    print(f"    I(Z;D)     = {Izd:+.6f}")
    print(f"    I(Y;D)     = {Iyd:+.6f}")
    print(f"    I(Z;D|Y)   = {Izdy:+.6f}   (encoder leakage)")
    print(f"    I(Y;D|Z)   = {Iydz:+.6f}   (decoder / concept)")
    # A1 exact identity:  I(Z;D|Y) - I(Y;D|Z) == I(Z;D) - I(Y;D)
    lhs = Izdy - Iydz
    rhs = Izd - Iyd
    print(f"    A1 check:  I(Z;D|Y)-I(Y;D|Z) = {lhs:+.6f}  ==  I(Z;D)-I(Y;D) = {rhs:+.6f}"
          f"   |diff|={abs(lhs - rhs):.2e}")
    return dict(Izd=Izd, Iyd=Iyd, Izdy=Izdy, Iydz=Iydz)


# ----------------------------------------------------------------------------
# THE A2 CONSTRUCTED FIGHT EXAMPLE  (exact reconstruction of the anchors).
#
#   axes 0=Z,1=Y,2=D ; binary Y, binary D, p(d) uniform.
#   Label shift via per-domain prior  pi_d(y):  d=0 -> P(Y=1)=0.41,  d=1 -> P(Y=1)=0.80
#   Domain-INVARIANT class-conditional channel  p(z|y)  (shared by both domains):
#       p(z|y=0) = [0.84, 0.16]      p(z|y=1) = [0.18, 0.82]
#   => I(Z;D|Y)=0 BY CONSTRUCTION (p(z|y) does not depend on d).
#
#   p(z,y,d) = p(d) * pi_d(y) * p(z|y)
# ----------------------------------------------------------------------------
pd   = np.array([0.5, 0.5])                       # p(d)
pi_d = np.array([[1 - 0.41, 0.41],                # pi_d[d, y]  (rows = domain)
                 [1 - 0.80, 0.80]])
pzy  = np.array([[0.84, 0.18],                    # pzy[z, y]   (invariant across d)
                 [0.16, 0.82]])

def build_joint(pd, pi_d, pzy):
    """P[z,y,d] = p(d) * pi_d(y) * p(z|y)."""
    Z = pzy.shape[0]
    P = np.zeros((Z, 2, 2))
    for d in range(2):
        for y in range(2):
            P[:, y, d] = pd[d] * pi_d[d, y] * pzy[:, y]
    return P

P = build_joint(pd, pi_d, pzy)
assert abs(P.sum() - 1.0) < 1e-12

print("=" * 74)
print("STEP 0  --  ORIGINAL A2 'FIGHT' JOINT  p(z,y,d)  (reproduce the anchors)")
print("=" * 74)
orig = report(P, "original")
print()
print("  Anchor targets:  I(Y;D)=0.0823  I(Z;D)=0.0340  I(Z;D|Y)=0  I(Y;D|Z)=0.0483")
ok_anchor = (abs(orig['Iyd'] - 0.0823) < 1e-3 and abs(orig['Izd'] - 0.0340) < 1e-3
             and abs(orig['Izdy']) < 1e-9 and abs(orig['Iydz'] - 0.0483) < 1e-3)
print(f"  anchors reproduced: {ok_anchor}")
print(f"  TENSION present:  encoder-invariant (I(Z;D|Y)=0) FORCES decoder leak "
      f"I(Y;D|Z)={orig['Iydz']:.4f} = I(Y;D)-I(Z;D) = {orig['Iyd']-orig['Izd']:.4f}")
print()

# ----------------------------------------------------------------------------
# STEP 1  --  GLS IMPORTANCE REWEIGHTING.
#
#   Choose a common REFERENCE label prior pi*(y).  Here pi*(y) = average prior
#   (any fixed prior works; uniform also works -- we test both).
#   Per-domain weight:  w_d(y) = pi*(y) / pi_d(y).
#
#   Reweighted joint (renormalized):
#       p~(z,y,d)  ∝  w_d(y) * p(z,y,d)  =  p(d) * pi*(y) * p(z|y)
#   By construction the y-marginal within every domain becomes pi*(y),
#   identical across d  =>  Y ⟂ D under p~  =>  I~(Y;D)=0.
# ----------------------------------------------------------------------------
def reweight(P, pi_d, pd, pi_star):
    """Apply w_d(y)=pi_star(y)/pi_d(y) to P[z,y,d], renormalize per the design."""
    Pw = np.zeros_like(P)
    for d in range(2):
        for y in range(2):
            w = pi_star[y] / pi_d[d, y]
            Pw[:, y, d] = w * P[:, y, d]
    # Renormalize the whole joint (the per-domain mass is preserved when
    # pi_star is itself a valid prior; we renormalize for numerical safety).
    Pw /= Pw.sum()
    return Pw

pi_star_avg = (pd[:, None] * pi_d).sum(0)         # marginal/average label prior
pi_star_avg = pi_star_avg / pi_star_avg.sum()
Pw = reweight(P, pi_d, pd, pi_star_avg)

print("=" * 74)
print("STEP 1  --  REWEIGHTED JOINT  p~(z,y,d)   with  w_d(y)=pi*(y)/pi_d(y)")
print(f"            reference prior  pi*(y) = {pi_star_avg}")
print("=" * 74)
rew = report(Pw, "reweighted")
print()

# ----------------------------------------------------------------------------
# CHECKS for parts (a) and (b).
# ----------------------------------------------------------------------------
print("=" * 74)
print("VERDICT")
print("=" * 74)

c_iyd0   = abs(rew['Iyd']) < 1e-12
print(f"(P2a)  I~(Y;D) = {rew['Iyd']:.2e}  ->  label shift removed:           {c_iyd0}")

# Reweighted identity becomes  I~(Z;D|Y) = I~(Y;D|Z) + I~(Z;D)   (since I~(Y;D)=0)
lhs = rew['Izdy']
rhs = rew['Iydz'] + rew['Izd']
c_id = abs(lhs - rhs) < 1e-12
print(f"(P2a') I~(Z;D|Y) = I~(Y;D|Z)+I~(Z;D):  {lhs:.6f} = {rhs:.6f}            {c_id}")

c_enc0 = abs(rew['Izdy']) < 1e-12
c_dec0 = abs(rew['Iydz']) < 1e-12
print(f"(P2b)  invariant encoder: I~(Z;D|Y) = {rew['Izdy']:.2e}                {c_enc0}")
print(f"(P2b)  decoder FREED:     I~(Y;D|Z) = {rew['Iydz']:.2e}                {c_dec0}")
print(f"       => BOTH CMIs simultaneously 0 under reweighting:       {c_enc0 and c_dec0}")
print()
print(f"  BEFORE reweight:  I(Z;D|Y)={orig['Izdy']:.4f}  I(Y;D|Z)={orig['Iydz']:.4f}"
      f"   (tension: forced trade-off)")
print(f"  AFTER  reweight:  I~(Z;D|Y)={rew['Izdy']:.4f}  I~(Y;D|Z)={rew['Iydz']:.4f}"
      f"   (tension GONE)")

# ----------------------------------------------------------------------------
# ROBUSTNESS: the conclusion does not depend on the choice of pi*.
# Try uniform reference prior too -> I~(Y;D)=0 and both CMIs 0 just the same.
# ----------------------------------------------------------------------------
print()
print("-" * 74)
print("ROBUSTNESS: uniform reference prior pi*=[0.5,0.5]")
print("-" * 74)
Pw_u = reweight(P, pi_d, pd, np.array([0.5, 0.5]))
rew_u = report(Pw_u, "reweighted-uniform")
print(f"  I~(Y;D)={rew_u['Iyd']:.2e}  I~(Z;D|Y)={rew_u['Izdy']:.2e}  "
      f"I~(Y;D|Z)={rew_u['Iydz']:.2e}  (same verdict, pi*-independent)")

# ----------------------------------------------------------------------------
# CONTROL: if the encoder were NOT invariant, reweighting alone does NOT zero
# the decoder term -> shows the decoder/concept constraint is a SEPARATE need.
# Make p(z|y) depend on d (concept/conditional shift in feature space).
# ----------------------------------------------------------------------------
print()
print("-" * 74)
print("CONTROL: encoder NOT invariant (p(z|y) depends on d) + same reweighting")
print("  -> I~(Y;D)=0 but I~(Y;D|Z) stays >0 : decoder term is an INDEPENDENT")
print("     constraint (this is exactly our addition on top of GLS).")
print("-" * 74)
pzy_d0 = np.array([[0.84, 0.18], [0.16, 0.82]])   # domain 0 channel
pzy_d1 = np.array([[0.30, 0.70], [0.70, 0.30]])   # domain 1: concept-shifted channel
Pc = np.zeros((2, 2, 2))
for y in range(2):
    Pc[:, y, 0] = pd[0] * pi_star_avg[y] * pzy_d0[:, y]   # already reweighted: pi*(y)
    Pc[:, y, 1] = pd[1] * pi_star_avg[y] * pzy_d1[:, y]
Pc /= Pc.sum()
ctrl = report(Pc, "noninvariant+reweighted")
print(f"  I~(Y;D)={ctrl['Iyd']:.2e} (label shift still removed) BUT "
      f"I~(Y;D|Z)={ctrl['Iydz']:.4f} > 0 and I~(Z;D|Y)={ctrl['Izdy']:.4f} > 0")
print(f"  => GLS reweighting alone is NOT sufficient; the explicit decoder/concept")
print(f"     constraint I(Y;D|Z)->0 (our addition) is required.")

print()
print("=" * 74)
ALL = (ok_anchor and c_iyd0 and c_id and c_enc0 and c_dec0)
print(f"ALL PILLAR-2 CLAIMS VERIFIED: {ALL}")
print("=" * 74)

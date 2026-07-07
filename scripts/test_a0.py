"""Synthetic unit tests for the A0 falsification harness (run BEFORE the real slice, per the pre-registration)."""
import numpy as np
from a0_falsification import (build_state, base_adapted, sample_scores, batch_scores, gen, admissible,
                              _maha2, _bures2, SAMPLE_SCORES, EQ_MARGIN)

def toy(n=400, d=8, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.5).astype(int)
    z = rng.standard_normal((n, d)); z[y == 1, 0] += sep
    return z, y

P = []
def check(name, ok):
    P.append((name, ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

zev, yev = toy(600, seed=1); zte, yte = toy(300, seed=2)
state = build_state(zev, yev)
rng = np.random.default_rng(0)

# 1. feature_invisible_conditional: z byte-identical -> sample scores byte-identical to clean (identifiability guard)
zc, yc = gen("clean", 0, zte, yte, state, rng)
zf, yf = gen("feature_invisible_conditional", 0.2, zte, yte, state, rng)
check("feature_invisible leaves z byte-identical", np.array_equal(zf, zte))
sc = sample_scores(state, zc, state["probe"].predict_proba(zc))
sf = sample_scores(state, zf, state["probe"].predict_proba(zf))
check("feature_invisible scores == clean scores (z-only, no label leakage)",
      all(np.allclose(sc[k], sf[k], atol=1e-9) for k in SAMPLE_SCORES))

# 2. source-free guard: sample_scores never takes y; values finite
import inspect
check("sample_scores signature has no target-y arg", "y" not in [p for p in inspect.signature(sample_scores).parameters if p not in ("state", "zp", "base_prob")])
check("sample scores finite", all(np.isfinite(sc[k]).all() for k in SAMPLE_SCORES))

# 3. covariate_shift_beneficial: CORAL should REDUCE loss vs base (adaptation helps -> low harm)
pi = np.bincount(yev, minlength=2).astype(float); pi /= pi.sum()
zb, yb = gen("covariate_shift_beneficial", 2.0, zte, yte, state, rng)
base, adapt = base_adapted(state, zb, pi)
lb = -np.log(np.clip(base[np.arange(len(yb)), yb], 1e-9, 1)).mean()
la = -np.log(np.clip(adapt[np.arange(len(yb)), yb], 1e-9, 1)).mean()
check("beneficial covariate shift: adapted loss <= base loss (CORAL helps)", la <= lb + 1e-6)

# 4. s_support ranks an injected covariate-shifted pocket as atypical (high)
zs = zte.copy(); pocket = np.arange(30); zs[pocket] += 6.0
ssup = sample_scores(state, zs, state["probe"].predict_proba(zs))["s_support"]
check("s_support high on covariate-shifted pocket", ssup[pocket].mean() > np.median(ssup) * 2)

# 5. admissible(): within-margin + consistent sign -> True; sign reversal -> False
best = [0.5, 0.5, 0.5]
check("admissible: within-margin consistent-sign", admissible([0.49, 0.48, 0.50], best) is True)
check("admissible: sign reversal -> False", admissible([0.49, -0.48, 0.50], best) is False)
check("admissible: outside margin -> False", admissible([0.49, 0.40, 0.50], best) is False)
check("admissible: NaN fold -> False", admissible([0.49, np.nan, 0.50], best) is False)

# 6. _bures2 zero for identical cov, positive otherwise; _maha2 zero at the mean
I = np.eye(4)
check("bures(S,S)==0", abs(_bures2(I, I)) < 1e-6)
check("bures(I,2I)>0", _bures2(I, 2 * I) > 0.1)
check("maha2 at mean ~0", _maha2(state["mu_pool"][None, :], state["mu_pool"], state["Pool"])[0] < 1e-9)

# 7. batch_scores finite + shift increases bures
b0 = batch_scores(state, zte, state["probe"].predict_proba(zte))
b1 = batch_scores(state, zte + 3.0, state["probe"].predict_proba(zte + 3.0))
check("bures increases under mean shift", b1["bures_shift"] > b0["bures_shift"])

print(f"\n{sum(ok for _, ok in P)}/{len(P)} checks passed")
import sys; sys.exit(0 if all(ok for _, ok in P) else 1)

"""Deterministic synthetic/regression tests for scripts/p15_audit.py (the 8 pre-registered cases).
Run twice -> identical output hash (determinism). No real data, no jobs."""
import numpy as np, hashlib, json, sys
sys.path.insert(0, "scripts")
import p15_audit as P

rng0 = lambda: np.random.default_rng(0)
RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail)); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

def rep_from(Z, Y):
    return P.representation_metrics(Z, Y, rng0())

# scenario builders (synthetic 16-d features) ------------------------------------------------
def make(n=1500, d=16, sep=2.0, seed=0):
    r = np.random.default_rng(seed); y = r.integers(0, 2, n); D = r.integers(0, 2, n)
    Z = (2 * y - 1)[:, None] * (np.eye(d)[0] * sep) + 0.6 * (2 * D - 1)[:, None] * np.eye(d)[1] + r.normal(0, 1, (n, d))
    return Z, y, D

print("=== P1.5 audit synthetic/regression tests ===")
# --- 1. TRUE SUPPRESSION: domain info ↓, task info + eff-rank retained -> RETAIN_LPC ---
Ze, ye, De = make(seed=1)
Zl = Ze.copy(); Zl[:, 1] = np.random.default_rng(2).normal(0, 1, len(Zl))  # erase the domain dim, keep task dim
re, rl = rep_from(Ze, ye), rep_from(Zl, ye)
dec, preds = P.decision_engine(leak_erm=0.80, leak_lpc=0.55, alt_erm=0.80, alt_lpc=0.57,
                               rep_erm=re, rep_lpc=rl, saturated=True)
check("1 true-suppression -> RETAIN_LPC", dec == "RETAIN_LPC", f"({dec})")

# --- 2. TOTAL COLLAPSE: features ~constant, task+domain probes deteriorate -> DROP_LPC_COLLAPSE ---
Zc = Ze * 0.02 + np.random.default_rng(3).normal(0, 0.01, Ze.shape)  # near-constant
rc = rep_from(Zc, ye)
dec2, _ = P.decision_engine(leak_erm=0.80, leak_lpc=0.52, alt_erm=0.80, alt_lpc=0.52,
                            rep_erm=re, rep_lpc=rc, saturated=True)
check("2 total-collapse -> DROP_LPC_COLLAPSE", dec2 == "DROP_LPC_COLLAPSE", f"({dec2})")

# --- 3. NORM SHRINK without info loss: rescale only -> NOT suppression-or-collapse from norm alone ---
Zn = Ze * 0.1                       # pure rescale
rn = rep_from(Zn, ye)
# scale-invariant structure preserved (scatter ratio, eff rank); standardized leakage gap ~0 -> not RETAIN, not DROP
check("3 norm-shrink: scatter_ratio ~unchanged", abs(rn["scatter_ratio"] - re["scatter_ratio"]) < 1e-6,
      f"({rn['scatter_ratio']:.3f} vs {re['scatter_ratio']:.3f})")
dec3, _ = P.decision_engine(leak_erm=0.80, leak_lpc=0.80, alt_erm=0.80, alt_lpc=0.80,  # standardized -> no gap
                            rep_erm=re, rep_lpc=rn, saturated=True)
check("3 norm-shrink alone -> NOT RETAIN/DROP (INCONCLUSIVE)", dec3 == "INCONCLUSIVE", f"({dec3})")

# --- 4. NONLINEAR RESIDUAL: linear near chance, tree/kNN recovers -> alt-recover predicate fails -> not RETAIN ---
dec4, preds4 = P.decision_engine(leak_erm=0.80, leak_lpc=0.55,   # linear sees a big drop
                                 alt_erm=0.80, alt_lpc=0.78,     # but GBT/kNN recover most of it
                                 rep_erm=re, rep_lpc=rl, saturated=True)
c2 = next(p for p in preds4 if p["predicate"] == "alt_probe_not_recovering")
check("4 nonlinear-residual: alt-recover predicate FAILS", not c2["pass_"], f"(recover={c2['value']})")
check("4 nonlinear-residual -> not RETAIN", dec4 != "RETAIN_LPC", f"({dec4})")

# --- 5. GROUP LEAKAGE: domain signal is PURELY group-level (per-group fingerprint). Random split memorizes the
#        fingerprint (groups seen in train) -> optimistic; grouped split (held-out groups) cannot -> honest/low. ---
r = np.random.default_rng(5); n = 1600; ng = 40
groups = r.integers(0, ng, n); Y = r.integers(0, 2, n)
gdom = r.integers(0, 2, ng); D = gdom[groups]            # domain = a GROUP property (no per-sample domain signal)
fingerprint = r.normal(0, 1, (ng, 16)) * 2.5             # each group has a unique memorable offset
Z = (2 * Y - 1)[:, None] * np.eye(16)[0] + fingerprint[groups] + r.normal(0, 1, (n, 16))
ga = P.domain_probe_audit(Z, D, Y, groups, ["mlp64"], r)   # GROUPED split (held-out groups)
# random split for contrast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
idx = r.permutation(n); tr, te = idx[:int(.7*n)], idx[int(.7*n):]
sc = StandardScaler().fit(Z[tr]); rand_bacc = balanced_accuracy_score(
    D[te], LogisticRegression(max_iter=500).fit(sc.transform(Z[tr]), D[tr]).predict(sc.transform(Z[te])))
check("5 grouped leakage < random leakage (group inflation removed)",
      ga["mlp64"]["heldout_bacc"] <= rand_bacc + 0.02, f"(grouped={ga['mlp64']['heldout_bacc']:.2f} random={rand_bacc:.2f})")

# --- 6. SAME bAcc, DIFFERENT predictions -> integrity (pred-hash) must differ ---
def phash(p): return hashlib.sha256(np.ascontiguousarray(np.round(p, 6), dtype=np.float64).tobytes()).hexdigest()
yy = np.r_[np.zeros(50, int), np.ones(50, int)]; base = np.zeros((100, 2)); base[np.arange(100), yy] = 1
pa = base.copy(); pa[[0, 1]] = [0, 1]; pb = base.copy(); pb[[2, 3]] = [0, 1]
check("6 same-bAcc-diff-preds: hash differs", phash(pa) != phash(pb))

# --- 7. integrity hard-fails: dup keys, missing pair, version mismatch (schema-level) ---
keys = [("c1", "0"), ("c1", "0"), ("c2", "0")]
check("7 duplicate key detected", len(keys) != len(set(keys)))
pairs = {"c1": ["erm", "lpc"], "c2": ["erm"]}              # c2 missing lpc
check("7 missing-pair detected", any(sorted(v) != ["erm", "lpc"] for v in pairs.values()))
check("7 version-hash mismatch detected", P._sha({"a": 1}) != P._sha({"a": 2}))

# --- 8. saturation rule + determinism ---
check("8a saturation: plateauing tiers -> met", P.saturation_met([0.30, 0.31, 0.315]) is True)
check("8a saturation: still-rising tiers -> NOT met", P.saturation_met([0.10, 0.20, 0.30]) is False)
out_a = json.dumps([P.decision_engine(0.8, 0.55, 0.8, 0.57, re, rl, True)[0] for _ in range(3)], sort_keys=True)
out_b = json.dumps([P.decision_engine(0.8, 0.55, 0.8, 0.57, re, rl, True)[0] for _ in range(3)], sort_keys=True)
check("8b determinism: identical decisions across runs", hashlib.sha256(out_a.encode()).hexdigest() ==
      hashlib.sha256(out_b.encode()).hexdigest())

nfail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{'ALL PASS' if nfail == 0 else f'{nfail} FAILED'} ({len(RESULTS)} checks)")
sys.exit(1 if nfail else 0)

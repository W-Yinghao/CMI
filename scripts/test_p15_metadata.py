"""Synthetic tests for scripts/p15_metadata_replay.py — index reconstruction, sidecar schema, label-binding,
and the BLOCKED_ON_GROUP_METADATA path. No real data, no model, deterministic."""
import numpy as np, tempfile, os, sys
sys.path.insert(0, "scripts")
import p15_metadata_replay as M

R = []
def check(n, c, d=""):
    R.append((n, bool(c))); print(f"  [{'PASS' if c else 'FAIL'}] {n} {d}")

print("=== p15 metadata-replay synthetic tests ===")
# mock loader: 3 cohorts, subjects nested in cohorts, canonical order
rng = np.random.default_rng(0); N = 300
coh = np.array([f"ds{c}" for c in (rng.integers(0, 3, N))])
subj = rng.integers(0, 30, N); y = rng.integers(0, 2, N)

# 1. reconstruct_fold: te = exactly the held-out cohort rows; se/ev partition the rest by the seeded split
hold = "ds0"; fm = M.reconstruct_fold(y, subj, coh, hold, seed=0)
te_expected = np.where(coh == hold)[0]
check("1 te indices == held-out cohort rows", np.array_equal(fm["te"]["gidx"], te_expected))
check("1 te labels match y[te]", np.array_equal(fm["te"]["y"], y[te_expected]))
tr = np.where(coh != hold)[0]
check("1 se∪ev == sources, disjoint", set(fm["se"]["gidx"]) | set(fm["ev"]["gidx"]) == set(tr.tolist())
      and not (set(fm["se"]["gidx"]) & set(fm["ev"]["gidx"])))
check("1 se ~70% of sources", abs(len(fm["se"]["gidx"]) / len(tr) - 0.7) < 0.02)

# 2. determinism: same seed -> identical reconstruction; different seed -> different source split
fm2 = M.reconstruct_fold(y, subj, coh, hold, seed=0)
check("2 reconstruction deterministic (same seed)", np.array_equal(fm["se"]["gidx"], fm2["se"]["gidx"]))
fm3 = M.reconstruct_fold(y, subj, coh, hold, seed=1)
check("2 different seed -> different source split", not np.array_equal(fm["se"]["gidx"], fm3["se"]["gidx"]))

# 3. sidecar schema: all required fields present, target role only on z_te
rows = M.build_sidecar_rows("PD", hold, fm, seed=0)
check("3 sidecar has all required fields", all(set(M.SIDE_FIELDS) <= set(r) for r in rows))
check("3 z_te rows are domain_role=target", all(r["domain_role"] == "target" for r in rows if r["array_key"] == "z_te"))
check("3 cohort_id on z_te == held-out fold", all(r["cohort_id"] == hold for r in rows if r["array_key"] == "z_te"))

# 4. label pre-check: a CORRECT dump matches; a TAMPERED dump fails (BLOCKED_ON_GROUP_METADATA trigger)
with tempfile.TemporaryDirectory() as td:
    good = f"{td}/feat_PD_{hold}_erm_0.npz"
    np.savez(good, z_se=np.zeros((len(fm["se"]["gidx"]), 4)), y_se=fm["se"]["y"],
             z_ev=np.zeros((len(fm["ev"]["gidx"]), 4)), y_ev=fm["ev"]["y"],
             z_te=np.zeros((len(fm["te"]["gidx"]), 4)), y_te=fm["te"]["y"])
    pc = M.label_precheck(good, fm)
    check("4 correct dump: all labels match", all(v[1] for v in pc.values()))
    bad = f"{td}/bad.npz"; yb = fm["te"]["y"].copy(); yb[:5] = 1 - yb[:5]   # tamper 5 labels
    np.savez(bad, y_se=fm["se"]["y"], y_ev=fm["ev"]["y"], y_te=yb)
    pcb = M.label_precheck(bad, fm)
    check("4 tampered dump: te label mismatch detected", not pcb["te"][1])

# 5. feature_hash: identical arrays -> identical hash; perturbed -> different
a = rng.normal(0, 1, (50, 8))
check("5 feature_hash stable + sensitive", M.feature_hash(a) == M.feature_hash(a.copy())
      and M.feature_hash(a) != M.feature_hash(a + 1e-3))

# 6. global-namespaced IDs: same subject name across DIFFERENT cohorts -> DISTINCT group_id
fm_s = {"se": {"gidx": np.array([0, 1]), "y": np.array([0, 1]), "subj": np.array(["sub-001", "sub-001"]),
               "coh": np.array(["dsA", "dsB"])},
        "ev": {"gidx": np.array([2]), "y": np.array([0]), "subj": np.array(["sub-002"]), "coh": np.array(["dsA"])},
        "te": {"gidx": np.array([], int), "y": np.array([]), "subj": np.array([]), "coh": np.array([])}}
rows2 = M.build_sidecar_rows("X", "dsA", fm_s, 0); gids = {r["group_id"] for r in rows2}
check("6 same subject across cohorts -> distinct group_id", "dsA::sub-001" in gids and "dsB::sub-001" in gids)
check("6 sample_id namespaced (cohort::subj::rec::window)", rows2[0]["sample_id"].count("::") == 3)

# 7. binding-status engine (FEATURE_BOUND / REQUIRES_FROZEN_REDUMP / METADATA_BINDING_FAILED)
fz = {"frozen": {"select_pipeline": "full", "configs": ["erm:0", "lpc_prior:0.1", "lpc_prior:0.3"]}}
check("7 metadata-fail -> METADATA_BINDING_FAILED",
      M.assess_binding(fz, {"present": True, "deterministic": True}, False)[0] == M.METADATA_BINDING_FAILED)
gen_old = {"present": True, "deterministic": False, "select_pipeline": "insample"}
check("7 non-cosource (nondet/diff config) -> REQUIRES_FROZEN_REDUMP",
      M.assess_binding(fz, gen_old, True)[0] == M.REQUIRES_FROZEN_REDUMP)
gen_co = {"present": True, "deterministic": True, "select_pipeline": "full",
          "configs": ["erm:0", "lpc_prior:0.1", "lpc_prior:0.3"]}
check("7 cosource + both branches match -> FEATURE_BOUND",
      M.assess_binding(fz, gen_co, True, True, True)[0] == M.FEATURE_BOUND)
check("7 cosource + LPC feature mismatch -> REQUIRES_FROZEN_REDUMP",
      M.assess_binding(fz, gen_co, True, True, False)[0] == M.REQUIRES_FROZEN_REDUMP)

nfail = sum(1 for _, ok in R if not ok)
print(f"\n{'ALL PASS' if nfail == 0 else f'{nfail} FAILED'} ({len(R)} checks)")
sys.exit(1 if nfail else 0)

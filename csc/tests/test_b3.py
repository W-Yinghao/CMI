"""
CSC Route B3 sanity tests (paired minimal-information certifier). DEVELOPMENT, simulator-only;
NOT in the audited TEST_MODULES (Route B is a separate dev direction from the frozen A line). Runs
standalone:  python -m csc.tests.test_b3
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_geom
from csc.mininfo.paired_sim import make_paired_target
from csc.mininfo.paired_certifier import (certify_paired, CONCEPT_CONFIRMED, NO_CONCEPT_EVIDENCE,
                                          NEED_MORE_LABELS, INVALID_PAIR, UNIDENTIFIABLE)


def _target(kind, seed=0, missing_frac=0.0, n_subjects=30):
    cfg = SimConfig(seed=seed); geom = make_geom(cfg, np.random.default_rng(seed))
    return make_paired_target(kind, geom, cfg, n_subjects=n_subjects, missing_frac=missing_frac,
                              seed=10_000 + seed)


# 1 ---- m=0 (no labels) -> UNIDENTIFIABLE for EVERY kind (reproduces the impossibility boundary) ----
def test_m0_abstains():
    for kind in ("clean", "paired_covariate", "paired_concept", "paired_pure_conditional"):
        Z, Y, D, G, _ = _target(kind, seed=1)
        r = certify_paired(Z, Y, D, G, m=0, n_boot=10, seed=1)
        assert r["state"] == UNIDENTIFIABLE, f"m=0 must abstain on {kind}, got {r['state']}"
    print("OK m=0 -> UNIDENTIFIABLE for all kinds (Z-only triage cannot confirm)")


# 2 ---- no pair structure (all subjects single-condition) -> INVALID_PAIR_STRUCTURE -----------------
def test_invalid_pair_structure():
    Z, Y, D, G, _ = _target("paired_concept", seed=2, missing_frac=1.0)
    r = certify_paired(Z, Y, D, G, m=20, n_boot=10, seed=2)
    assert r["state"] == INVALID_PAIR, f"all-unpaired must be INVALID_PAIR, got {r['state']}"
    print("OK all-unpaired target -> INVALID_PAIR_STRUCTURE")


# 3 ---- paired_concept with enough labels -> CONCEPT_CONFIRMED --------------------------------------
def test_concept_confirmed():
    hits = 0
    for s in range(4):
        Z, Y, D, G, _ = _target("paired_concept", seed=s)
        r = certify_paired(Z, Y, D, G, m=20, alpha=0.05, decide_n=20, n_boot=120, seed=s)
        hits += int(r["state"] == CONCEPT_CONFIRMED)
    assert hits >= 3, f"paired_concept m=20 should confirm in >=3/4, got {hits}"
    print(f"OK paired_concept m=20 -> CONCEPT_CONFIRMED ({hits}/4)")


# 4 ---- paired_covariate (no concept) -> NOT CONCEPT_CONFIRMED (type-I control) ---------------------
def test_covariate_not_confirmed():
    bad = 0
    for s in range(4):
        Z, Y, D, G, _ = _target("paired_covariate", seed=s)
        r = certify_paired(Z, Y, D, G, m=20, alpha=0.05, decide_n=20, n_boot=120, seed=s)
        assert r["state"] in (NO_CONCEPT_EVIDENCE, NEED_MORE_LABELS), r["state"]
        bad += int(r["state"] == CONCEPT_CONFIRMED)
    assert bad == 0, f"paired_covariate must not CONCEPT_CONFIRM, got {bad}/4"
    print("OK paired_covariate m=20 -> not confirmed (0/4 false confirmations)")


if __name__ == "__main__":
    test_m0_abstains()
    test_invalid_pair_structure()
    test_concept_confirmed()
    test_covariate_not_confirmed()
    print("\nall CSC Route B3 sanity tests passed")

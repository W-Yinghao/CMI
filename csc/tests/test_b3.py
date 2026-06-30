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
        r = certify_paired(Z, Y, D, G, m=20, alpha=0.05, decide_n=20, h1_basis="pc", n_boot=120, seed=s)
        hits += int(r["state"] == CONCEPT_CONFIRMED)
    assert hits >= 3, f"paired_concept m=20 should confirm in >=3/4, got {hits}"
    print(f"OK paired_concept m=20 -> CONCEPT_CONFIRMED ({hits}/4) [pc controlled basis]")


# 4 ---- paired_covariate (no concept) -> NOT CONCEPT_CONFIRMED (type-I control) ---------------------
def test_covariate_not_confirmed():
    bad = 0
    for s in range(4):
        Z, Y, D, G, _ = _target("paired_covariate", seed=s)
        r = certify_paired(Z, Y, D, G, m=20, alpha=0.05, decide_n=20, h1_basis="pc", n_boot=120, seed=s)
        assert r["state"] in (NO_CONCEPT_EVIDENCE, NEED_MORE_LABELS), r["state"]
        bad += int(r["state"] == CONCEPT_CONFIRMED)
    assert bad == 0, f"paired_covariate must not CONCEPT_CONFIRM, got {bad}/4"
    print("OK paired_covariate m=20 -> not confirmed (0/4) [pc controlled basis; full_z REJECTED]")


# 5 ---- contract: observed T invariant to epoch duplication, for BOTH h1 bases (weighting) -----------
def test_epoch_duplication_invariance():
    from csc.mininfo.paired_conditional_test import paired_conditional_change_test
    Z, Y, D, G, _ = _target("paired_concept", seed=3, n_subjects=12)
    paired = [s for s in np.unique(G) if len(np.unique(D[G == s])) >= 2]
    pick = paired[:8]; mask = np.isin(G, pick)
    Zq, Yq, Dq, Gq = Z[mask], Y[mask], D[mask], G[mask]
    s0 = pick[0]; c0 = Dq[Gq == s0][0]; dup = (Gq == s0) & (Dq == c0)
    Z2 = np.concatenate([Zq, Zq[dup]]); Y2 = np.concatenate([Yq, Yq[dup]])
    D2 = np.concatenate([Dq, Dq[dup]]); G2 = np.concatenate([Gq, Gq[dup]])
    for basis in ("full_z", "pc"):
        T1 = paired_conditional_change_test(Zq, Yq, Dq, Gq, h1_basis=basis, n_boot=1, seed=0)["T"]
        T2 = paired_conditional_change_test(Z2, Y2, D2, G2, h1_basis=basis, n_boot=1, seed=0)["T"]
        assert abs(T1 - T2) < 1e-3, f"{basis}: T not epoch-dup invariant: {T1} vs {T2}"
    print("OK observed T invariant to epoch duplication for full_z AND pc [subject-condition weights]")


# 7 ---- B3-P2.2 guard: significant but m < min_confirm_pairs -> NEED_MORE_LABELS (would_confirm True) -
def test_min_confirm_pairs_guard():
    from csc.mininfo.paired_certifier import certify_paired
    Z, Y, D, G, _ = _target("paired_concept", seed=0, n_subjects=36)
    small = certify_paired(Z, Y, D, G, m=10, min_confirm_pairs=20, h1_basis="pc", n_boot=120, seed=0)
    big = certify_paired(Z, Y, D, G, m=20, min_confirm_pairs=20, h1_basis="pc", n_boot=120, seed=0)
    assert small["state"] == NEED_MORE_LABELS, small["state"]          # blocked despite significance
    assert small["would_confirm_without_min_pairs"], "expected raw significance at m=10"
    assert big["state"] == CONCEPT_CONFIRMED, big["state"]             # allowed at m>=20
    print("OK min_confirm_pairs=20 guard: m=10 significant -> NEED_MORE_LABELS; m=20 -> CONCEPT_CONFIRMED")


# 8 ---- B3-P2.2 R1: full_z interaction uses ALL d directions (not a low-rank truncation) -------------
def test_full_z_uses_all_directions():
    from csc.sim.shift_simulator import SimConfig
    from csc.mininfo.paired_conditional_test import paired_conditional_change_test, _resolve_C
    d = SimConfig().d
    Z, Y, D, G, _ = _target("paired_concept", seed=0, n_subjects=36)
    paired = [s for s in np.unique(G) if len(np.unique(D[G == s])) >= 2][:20]
    mask = np.isin(G, paired)
    t = paired_conditional_change_test(Z[mask], Y[mask], D[mask], G[mask], h1_basis="full_z",
                                       n_boot=1, seed=0)
    assert t["n_features_interaction"] == d, (t["n_features_interaction"], d)
    assert abs(t["C_used"] - 0.5 * 3 / d) < 1e-12, t["C_used"]
    assert _resolve_C("pc", d, 3, None)[1] == 3                       # pc baseline stays rank-3
    print(f"OK full_z interacts ALL {d} directions, C_full={t['C_used']:.4f} (=0.5*3/{d}); pc=rank-3")


# 6 ---- B3-P2.1 contract: a condition with <2 classes fails the validity gate (closed) --------------
def test_per_condition_class_coverage():
    from csc.mininfo.paired_conditional_test import paired_validity
    # subject 0/1 have both conditions; condition 1 carries only class 0 -> must be invalid
    G = np.array([0, 0, 1, 1, 0, 1]); D = np.array([0, 0, 0, 0, 1, 1]); Y = np.array([0, 1, 0, 1, 0, 0])
    ok, reason = paired_validity(Y, D, G, min_subjects=2)
    assert not ok and "condition" in reason, reason
    # give condition 1 both classes -> valid
    Y2 = np.array([0, 1, 0, 1, 0, 1])
    assert paired_validity(Y2, D, G, min_subjects=2)[0]
    print("OK per-condition class coverage fails closed (<2 classes in a condition -> invalid)")


# 9 ---- B3-P2.2 R1c: CENTERED +-0.5 coding fixes the full_z clean type-I that 0/1 coding caused --------
def test_centered_coding_fixes_full_z_clean():
    from csc.mininfo.paired_conditional_test import paired_conditional_change_test
    def clean_audit(seed):
        Z, Y, D, G, _ = _target("clean", seed=seed, n_subjects=36)
        pick = [s for s in np.unique(G) if len(np.unique(D[G == s])) >= 2][:20]
        m = np.isin(G, pick); return Z[m], Y[m], D[m], G[m]
    bad01 = cen = 0
    for s in range(4):
        Zq, Yq, Dq, Gq = clean_audit(1000 + s)
        bad01 += int(paired_conditional_change_test(Zq, Yq, Dq, Gq, h1_basis="full_z",
                     condition_coding="01", n_boot=120, seed=s)["p_value"] <= 0.05)
        cen += int(paired_conditional_change_test(Zq, Yq, Dq, Gq, h1_basis="full_z",
                   condition_coding="centered", n_boot=120, seed=s)["p_value"] <= 0.05)
    assert bad01 >= 3, f"expected the 0/1 trap to false-confirm clean (>=3/4), got {bad01}"
    assert cen <= 1, f"centered coding must control clean type-I (<=1/4), got {cen}"
    # the interaction code must be (near) zero-mean on a balanced audit under centered coding
    Zq, Yq, Dq, Gq = clean_audit(1000)
    wcm = paired_conditional_change_test(Zq, Yq, Dq, Gq, h1_basis="full_z", condition_coding="centered",
                                         n_boot=1, seed=0)["weighted_condition_mean_check"]
    assert abs(wcm) < 0.15, f"centered weighted condition mean should be ~0, got {wcm}"
    print(f"OK centered coding fixes full_z clean type-I: 0/1={bad01}/4 -> centered={cen}/4 (wcm={wcm:+.3f})")


# 10 -- B3-P2.2 R1c headline: centered coding lets the controlled pc basis RECOVER pure_conditional ----
def test_centered_pc_recovers_pure_conditional():
    # old 0/1 coding gave pc 0.00 on pure_conditional; centered coding recovers it (dev map: 0.75 @ m30).
    hits01 = hitc = 0
    for s in range(4):
        Z, Y, D, G, _ = _target("paired_pure_conditional", seed=s, n_subjects=36)
        hits01 += int(certify_paired(Z, Y, D, G, m=30, min_confirm_pairs=20, h1_basis="pc",
                      condition_coding="01", n_boot=200, seed=s)["state"] == CONCEPT_CONFIRMED)
        hitc += int(certify_paired(Z, Y, D, G, m=30, min_confirm_pairs=20, h1_basis="pc",
                    condition_coding="centered", n_boot=200, seed=s)["state"] == CONCEPT_CONFIRMED)
    assert hitc >= 2, f"centered pc should recover pure_conditional @ m=30 (>=2/4), got {hitc}"
    assert hitc > hits01, f"centered ({hitc}) must beat 0/1 ({hits01}) on pure_conditional"
    print(f"OK centered pc recovers pure_conditional @ m=30: 0/1={hits01}/4 -> centered={hitc}/4")


# 11 -- B3-P2.4 calibrated guards close the documented P2.3 leaks; power retained -----------------------
def test_p24_guards_and_power():
    from csc.mininfo.paired_calibrated import certify_paired_calibrated, INVALID_PAIR as INV
    def cert(kind, seed=0, m=24, **mt):
        cfg = SimConfig(seed=seed); geom = make_geom(cfg, np.random.default_rng(seed))
        Z, Y, D, G, _ = make_paired_target(kind, geom, cfg, n_subjects=36, seed=10_000 + seed, **mt)
        return certify_paired_calibrated(Z, Y, D, G, m=m, n_boot=80, seed=seed)
    # pair-integrity guard closes missing_pair; eligibility guard closes unequal_epochs_extreme
    assert cert("missing_pair")["state"] == INV, cert("missing_pair")["state"]
    assert cert("unequal_epochs_extreme")["state"] == INV, cert("unequal_epochs_extreme")["state"]
    # power retained on genuine concept; controls not confirmed
    conf = sum(cert("paired_concept", s)["state"] == CONCEPT_CONFIRMED for s in range(3))
    badr = sum(cert("random_label", s)["state"] == CONCEPT_CONFIRMED for s in range(3))
    assert conf >= 2, f"calibrated concept power too low ({conf}/3)"
    assert badr == 0, f"calibrated random_label false-confirmed ({badr}/3)"
    print(f"OK P2.4 guards: missing_pair & unequal_epochs -> INVALID; concept {conf}/3 confirmed; "
          f"random_label {badr}/3 (closed)")


if __name__ == "__main__":
    test_m0_abstains()
    test_invalid_pair_structure()
    test_concept_confirmed()
    test_covariate_not_confirmed()
    test_epoch_duplication_invariance()
    test_per_condition_class_coverage()
    test_min_confirm_pairs_guard()
    test_full_z_uses_all_directions()
    test_centered_coding_fixes_full_z_clean()
    test_centered_pc_recovers_pure_conditional()
    test_p24_guards_and_power()
    print("\nall CSC Route B3 sanity + contract tests passed")

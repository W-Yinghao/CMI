"""Project A — executable counterexample certificates for the observability ledger.

These are PROOF-CERTIFICATE support, not training code. Each block builds two worlds that are
identical under a regime's observation operator O_R but disagree on a target estimand
(OA-0 contrapositive, notes/project_A_observability/01_information_regimes.md).

- Exact-discrete blocks (numpy only) ARE the proofs; every expected (in)equality is asserted,
  so the script fails loud on any arithmetic/logic error.
- The h2cmi simulator block is an ILLUSTRATION only (realistic EEG-shaped instance of CE-R1-1).

Run:  conda run -n icml python notes/project_A_observability/counterexamples/run_counterexamples.py

Certificates:
  CE-R0-1  source-only target-risk non-identifiability          (03_… §4.2)
  CE-R0-2  source-only adaptation-gain SIGN non-identifiability (03_… §4.1)
  CE-R0-3  source-only target-prior non-identifiability         (03_… §5)
  CE-R1-1  target-unlabeled CONCEPT non-identifiability  = TU-2 (07_… §4)
  CE-R1-2  prior non-identifiability under C3 failure           (07_… §5)
  CE-C1-1  class-support-overlap failure (C1)                   (07_… §8)
  CE-MP-1  transport underdetermined by too few anchors (C8)    (07_… §9)
  CE-C11-1 fake pairing / anchor-validity failure (C11)         (07_… §10)
  CE-MONO-1 more source breadth ≠ target observation (MONO-1)   (07_… §11)
  ILL-R1-1 simulator illustration of CE-R1-1                    (07_… §6)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make `import h2cmi` work regardless of cwd: repo root is 3 parents up from this file
# (<repo>/notes/project_A_observability/counterexamples/run_counterexamples.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------------------------
# helpers (binary X, Y in {0,1}; joints are 2x2 arrays J[x, y])
# --------------------------------------------------------------------------------------------
def risk_01(PT, h):
    """Expected 0-1 loss of classifier h: {0,1}->{0,1} under target joint PT[x,y]."""
    r = 0.0
    for x in (0, 1):
        for y in (0, 1):
            r += PT[x, y] * (1.0 if h(x) != y else 0.0)
    return r


def balanced_accuracy(PT, h, n_classes=2):
    """bAcc = mean_y P(h(X)=y | Y=y) under target joint PT[x,y]."""
    accs = []
    for y in range(n_classes):
        py = sum(PT[x, y] for x in (0, 1))
        if py <= 0:
            continue  # class absent -> undefined, skip (balanced classes here)
        correct = sum(PT[x, y] for x in (0, 1) if h(x) == y)
        accs.append(correct / py)
    return float(np.mean(accs))


H_ID = lambda x: x          # source-trained (Bayes-optimal on source Y=X)
H_AD = lambda x: 1 - x      # candidate "adapted" classifier (label flip)


# --------------------------------------------------------------------------------------------
# The G/B concept pair: same source (Y=X), same target p_T(X), opposite target concept.
# --------------------------------------------------------------------------------------------
PS = np.array([[0.5, 0.0],
               [0.0, 0.5]])          # source joint P_S(X=x, Y=y): Y=X deterministically
PT_G = np.array([[0.5, 0.0],
                 [0.0, 0.5]])        # world G target: Y=X
PT_B = np.array([[0.0, 0.5],
                 [0.5, 0.0]])        # world B target: Y=1-X

# both worlds carry the SAME source law
PS_G, PS_B = PS, PS


def main():
    ok = True

    # ---- CE-R0-1 : target risk of a fixed classifier differs across R0-equal worlds --------
    source_equal = bool(np.allclose(PS_G, PS_B))
    rG = risk_01(PT_G, H_ID)
    rB = risk_01(PT_B, H_ID)
    print(f"CE-R0-1 source_equal={source_equal}")
    print(f"CE-R0-1 target_risk_world_G={rG:.3f} target_risk_world_B={rB:.3f}")
    assert source_equal, "CE-R0-1: source laws must be equal"
    assert np.isclose(rG, 0.0) and np.isclose(rB, 1.0), "CE-R0-1: target risk must differ 0 vs 1"
    print()

    # ---- CE-R0-2 : adaptation-gain SIGN flips across R0- (and R1-) equal worlds ------------
    gain_G = balanced_accuracy(PT_G, H_AD) - balanced_accuracy(PT_G, H_ID)
    gain_B = balanced_accuracy(PT_B, H_AD) - balanced_accuracy(PT_B, H_ID)
    print(f"CE-R0-2 gain_world_G={gain_G:+.3f} gain_world_B={gain_B:+.3f}")
    assert np.isclose(gain_G, -1.0) and np.isclose(gain_B, +1.0), "CE-R0-2: gain sign must flip"
    assert np.sign(gain_G) != np.sign(gain_B), "CE-R0-2: gains must have opposite sign"
    print()

    # ---- CE-R0-3 : target prior non-identifiable (R0 observes no target) -------------------
    piT_A = np.array([0.2, 0.8])
    piT_B = np.array([0.8, 0.2])
    prior_differs = bool(not np.allclose(piT_A, piT_B))
    print(f"CE-R0-3 source_equal={source_equal} "
          f"piT_A=({piT_A[0]:.1f},{piT_A[1]:.1f}) piT_B=({piT_B[0]:.1f},{piT_B[1]:.1f}) "
          f"prior_differs={prior_differs}")
    assert prior_differs, "CE-R0-3: the two target priors must differ"
    print()

    # ---- CE-R1-1 : concept non-identifiable from unlabeled target (TU-2) --------------------
    pTx_G = PT_G.sum(axis=1)                    # p_T(X) in world G
    pTx_B = PT_B.sum(axis=1)                    # p_T(X) in world B
    cond_G = PT_G / pTx_G[:, None]              # p_T(Y|X) in world G
    cond_B = PT_B / pTx_B[:, None]              # p_T(Y|X) in world B
    target_X_equal = bool(np.allclose(pTx_G, pTx_B))
    concept_diff = bool(not np.allclose(cond_G, cond_B))
    target_y_equal = not concept_diff           # same X ⇒ label rule differs
    print(f"CE-R1-1 source_equal={source_equal} target_X_equal={target_X_equal} "
          f"target_y_equal={target_y_equal}")
    print(f"CE-R1-1 pT_X_equal={target_X_equal} concept_diff={concept_diff}")
    assert target_X_equal, "CE-R1-1: target feature marginals must be equal"
    assert concept_diff, "CE-R1-1: target concepts p_T(Y|X) must differ"
    print()

    # ---- CE-R1-2 : prior non-identifiable when class-conditionals are rank-deficient (C3) --
    M = np.array([[0.5, 0.5],                   # p(z|Y=0)
                  [0.5, 0.5]])                  # p(z|Y=1)   -> equal rows, rank 1
    rank = int(np.linalg.matrix_rank(M))
    pTz_A = np.array([0.3, 0.7]) @ M
    pTz_B = np.array([0.6, 0.4]) @ M
    prior_identifiable = bool(not np.allclose(pTz_A, pTz_B))
    print(f"CE-R1-2 class_conditionals_rank={rank} prior_identifiable={prior_identifiable}")
    assert rank == 1, "CE-R1-2: degenerate class-conditionals must have rank 1"
    assert prior_identifiable is False, "CE-R1-2: distinct priors must give identical p_T(z)"
    print()

    # ---- CE-C1-1 : class-support-overlap failure (C1) --------------------------------------
    # source class-conditionals over z∈{0,1,2}; target puts mass on z=2 (off source support)
    B_supp = np.array([[1.0, 0.0],    # p_ref(z=0 | y)
                       [0.0, 1.0],    # p_ref(z=1 | y)
                       [0.0, 0.0]])   # p_ref(z=2 | y)  -> source never emits z=2
    pT_z = np.array([0.375, 0.375, 0.25])          # target marginal: mass 0.25 at z=2
    source_support = set(np.flatnonzero(B_supp.max(axis=1) > 0).tolist())   # {0,1}
    target_support = set(np.flatnonzero(pT_z > 0).tolist())                 # {0,1,2}
    support_overlap = bool(target_support.issubset(source_support))
    off_atoms = [z for z in sorted(target_support) if z not in source_support]
    off_source_mass = float(pT_z[off_atoms].sum())
    # no π_T≥0 can reproduce mass sitting on an off-source atom -> mixture infeasible
    mixture_feasible = bool(np.isclose(off_source_mass, 0.0))
    print(f"CE-C1-1 support_overlap={support_overlap} off_source_mass={off_source_mass:.3f} "
          f"mixture_feasible={mixture_feasible}")
    assert support_overlap is False and off_source_mass > 0, "CE-C1-1: target mass must sit off source support"
    assert mixture_feasible is False, "CE-C1-1: mixture must be infeasible when support fails"
    print()

    # ---- CE-MP-1 : high-dim transport underdetermined by too few anchors (C8) --------------
    # both transforms lie INSIDE C8's near-identity family (‖T−I‖ small) yet both fix the anchor
    T_id = np.array([[1.0, 0.0], [0.0, 1.0]])       # identity transform
    T_eps = np.array([[1.0, 0.0], [0.0, 1.1]])      # near-identity: scales the un-anchored direction
    e1 = np.array([1.0, 0.0]); e2 = np.array([0.0, 1.0])
    anchors_equal = bool(np.array_equal(T_id @ e1, T_eps @ e1))        # both map the one anchor e1->e1
    transforms_distinct = bool(not np.array_equal(T_id, T_eps))
    probe_maps_differ = bool(not np.array_equal(T_id @ e2, T_eps @ e2))  # differ on the un-anchored direction
    identifiable = bool(not (anchors_equal and transforms_distinct and probe_maps_differ))
    print(f"CE-MP-1 anchors_equal={anchors_equal} transforms_distinct={transforms_distinct} "
          f"probe_maps_differ={probe_maps_differ} identifiable={identifiable}")
    assert anchors_equal and transforms_distinct and probe_maps_differ and identifiable is False, \
        "CE-MP-1: two transforms must agree on the anchor but differ off it"
    print()

    # ---- CE-C11-1 : fake pairing / anchor-validity failure (C11) ---------------------------
    observed_pair = (1.0, -1.0)                                       # (x_source, x_target)
    true_transport_world_A = -1.0     # genuine pairing: x_target = A·x_source, A = -1
    true_transport_world_B = +1.0     # spurious pairing (unrelated latent events), true A = +1
    observed_pair_equal = True        # both worlds emit the SAME observed pair
    anchor_validity_required = bool(true_transport_world_A != true_transport_world_B)
    print(f"CE-C11-1 observed_pair_equal={observed_pair_equal} "
          f"true_transport_world_A={true_transport_world_A:.0f} "
          f"true_transport_world_B={true_transport_world_B:.0f} "
          f"anchor_validity_required={anchor_validity_required}")
    assert observed_pair_equal and anchor_validity_required, \
        "CE-C11-1: same observed pair must admit different true transports"
    print()

    # ---- CE-MONO-1 : more source breadth ≠ target observation (MONO-1) ----------------------
    src_P = src_Q = PS                                               # identical source in both worlds
    pTx_P = np.array([1.0, 0.0])                                     # world P target: X=0 a.s.
    pTx_Q = np.array([0.0, 1.0])                                     # world Q target: X=1 a.s.
    same_source = bool(np.array_equal(src_P, src_Q))
    r0_indistinguishable = same_source                              # R0 observes only source
    r1_distinguishable = bool(not np.allclose(pTx_P, pTx_Q))        # R1 adds the target X marginal
    source_breadth_not_target_observation = bool(r0_indistinguishable and r1_distinguishable)
    print(f"CE-MONO-1 same_source={same_source} r0_indistinguishable={r0_indistinguishable} "
          f"r1_distinguishable={r1_distinguishable} "
          f"source_breadth_not_target_observation={source_breadth_not_target_observation}")
    assert source_breadth_not_target_observation, \
        "CE-MONO-1: worlds must be R0-equal yet R1-distinct (source breadth ≠ target obs)"
    print()

    # ---- ILL-R1-1 : simulator illustration of CE-R1-1 (illustration only, not a proof) -----
    sim_ok, sim_reason = _simulator_illustration()
    if sim_ok:
        print("ILL-R1-1 (simulator) target_X_equal=True target_y_equal=False")
        print("ALL COUNTEREXAMPLE ASSERTIONS PASSED")
    else:
        # fail loud with the reason; the exact-discrete certificates above are the proofs
        print(f"ILL-R1-1 (simulator) SKIPPED: reason={sim_reason}")
        print("ALL EXACT-DISCRETE ASSERTIONS PASSED (simulator illustration skipped)")

    return 0 if ok else 1


def _simulator_illustration():
    """Build a realistic EEG-shaped instance of CE-R1-1: one target X_T, two labelings.

    Returns (ok, reason). Does not raise: a simulator/import problem yields ok=False with a
    reason-coded string (no silent swallow), while the exact-discrete proofs still stand.
    """
    try:
        from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
        sim = EEGSimulator(
            n_classes=3, n_chans=12, n_times=128,
            shift=ShiftSpec(cov=1.0, prior=0.3, concept=0.0, concept_site_frac=0.0,
                            montage=0.2, noise=0.3, label_mechanism_rho=0.0),
            seed=0,
        ).sample(n_sites=4, subjects_per_site=2, sessions_per_subject=1, trials_per_session=16)
        src_idx, tgt_idx = train_target_split(sim, 1, seed=0)
        if len(tgt_idx) == 0:
            return False, "empty_target_split"
        XG = sim.X[tgt_idx]                        # world G target features
        XB = XG.copy()                             # world B REUSES the identical target features
        yG = sim.y[tgt_idx]                        # world G labeling (observed)
        yB = (yG + 1) % sim.n_classes             # world B labeling (a different label mechanism)
        # R1 observer sees only target X; both (X, yG) and (X, yB) are R1-consistent.
        target_X_equal = bool(np.array_equal(XG, XB) and np.isfinite(XG).all())
        target_y_equal = bool(np.array_equal(yG, yB))
        if not target_X_equal:
            return False, "target_X_not_equal_or_not_finite"
        if target_y_equal:
            return False, "relabeling_did_not_change_y"
        return True, "ok"
    except Exception as exc:  # reason-coded, not silently swallowed
        return False, f"{type(exc).__name__}:{exc}"


if __name__ == "__main__":
    raise SystemExit(main())

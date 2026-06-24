"""
CSC-P1.3 frozen-path + calibration tests:

1. calibrate_tau_detect runs and returns a finite float (regression for the v0 positional-arg
   bug that routed alpha->target_size, n_block->block_ids_tr, seed->alpha).
2. run_frozen_protocol (the SINGLE certification path) gives the right states across the
   taxonomy -- this is what audits/confirmatory measure, not certify(...) directly.
3. the path is cluster-aware: passing source/target group_ids runs without error.
4. ProtocolConfig manifest hash is deterministic and tau_detect/tau_label appear as RULES.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_target
from csc.protocol import ProtocolConfig, run_frozen_protocol
from csc.calibration.lodo import calibrate_tau_detect
from csc.certificate import (
    analyze_source, FORBIDDEN, UNIDENTIFIABLE, COVARIATE_COMPATIBLE, CONCEPT_SUSPECT,
)

# n_boot >= ~20 is REQUIRED: concept_evidenced needs the residual-decoder p to reach <= alpha,
# and the bootstrap p has min 1/(n_boot+1). The residual-decoder gate is intrinsically
# budget-bound (n_boot=15 -> min p 0.0625 > 0.05 -> zero concept power by construction).
CFG = ProtocolConfig(n_boot=40, n_dir_boot=80, target_n_boot=50, tau_n_pseudotargets=80)


def test_calibrate_tau_detect_keyword_safe():
    src = make_source(SimConfig(seed=1), n_domains=8, concept_domains=3, seed=1)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=10, n_dir_boot=60, seed=1)
    tau = calibrate_tau_detect(src.Z, src.Y, src.D, sa.atlas, alpha=0.05, n_block=120, seed=1)
    assert np.isfinite(tau) and tau > 0, f"calibrate_tau_detect returned {tau}"
    print(f"OK calibrate_tau_detect keyword-safe -> tau_detect={tau:.3f}")


def test_frozen_path_taxonomy():
    expect = {"clean": UNIDENTIFIABLE, "covariate": COVARIATE_COMPATIBLE,
              "boundary_coupled": CONCEPT_SUSPECT, "label_shift": UNIDENTIFIABLE,
              "pure_conditional": UNIDENTIFIABLE}
    ok = {k: 0 for k in expect}
    must_abstain_forbidden = 0      # finite-sample false certs on clean/pure/label (STATISTICAL)
    full_forbidden = 0              # includes rare covariate crying-wolf (statistical, soft)
    n = 3
    for s in range(n):
        scfg = SimConfig(seed=s)
        src = make_source(scfg, n_domains=8, concept_domains=3, seed=s)
        for kind, want in expect.items():
            tb = make_target(kind, scfg, geom=src.geom, seed=100 + s)
            st = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, CFG,
                                     src_group_ids=src.group_ids, tgt_group_ids=tb.group_ids,
                                     tgt_condition_ids=np.zeros(len(tb.Z), int),
                                     seed=s)["certificate"].state
            ok[kind] += int(st == want)
            if st in FORBIDDEN[tb.truth]:
                full_forbidden += 1
                if kind in ("clean", "pure_conditional", "label_shift"):
                    must_abstain_forbidden += 1
    # CSC-P1.4.2 #7: false-certification of clean/pure/label is a FINITE-SAMPLE STATISTICAL
    # property (a chance clean marginal can exceed tau_detect + align with the atlas), NOT a
    # structural "never". It is controlled at level alpha by run_synthetic's exact-CP endpoint
    # over independent clusters; here we only smoke-bound gross regression. The STRUCTURAL
    # guarantee (byte-identical clean vs pure -> SAME output) is test_paired_clean_pure_*.
    assert must_abstain_forbidden <= 1, f"unidentifiable-shift false certs {must_abstain_forbidden} (smoke)"
    assert ok["clean"] >= n - 1 and ok["pure_conditional"] >= n - 1 and ok["label_shift"] >= n - 1
    assert ok["boundary_coupled"] >= 1, "frozen path shows no concept power at all"
    assert full_forbidden <= 2, f"gross forbidden-rate regression: {full_forbidden}/{n*len(expect)}"
    print(f"OK frozen-path: must-abstain false certs {must_abstain_forbidden} (statistical smoke); "
          f"concept {ok['boundary_coupled']}/{n}; covariate {ok['covariate']}/{n}; "
          f"full-suite forbidden {full_forbidden}")


def test_frozen_path_is_cluster_aware():
    scfg = SimConfig(seed=2)
    src = make_source(scfg, n_domains=8, concept_domains=3, seed=2)
    # use the REAL label-homogeneous subject ids (label_unit='subject'); artificial multi-label
    # 'subjects' now (correctly) fail the label_unit data-validation (CSC-P1.4.4 #1).
    tb = make_target("covariate", scfg, geom=src.geom, seed=200)
    out = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, CFG,
                              src_group_ids=src.group_ids, tgt_group_ids=tb.group_ids,
                              tgt_condition_ids=np.zeros(len(tb.Z), int), seed=2)
    assert out["certificate"].state in (COVARIATE_COMPATIBLE, UNIDENTIFIABLE)
    assert out["analysis"].detail["cluster_aware"] is True
    print(f"OK cluster-aware frozen path runs -> {out['certificate'].state}")


def test_manifest_hash_deterministic_and_full():
    a, b = ProtocolConfig(), ProtocolConfig()
    assert a.hash() == b.hash()
    assert len(a.hash()) == 64, "manifest hash must be the FULL sha256 (64 hex)"
    assert ProtocolConfig(consensus=0.9).hash() != a.hash()
    # the runtime root seed is NOT part of the method id (master_seed was removed from the
    # manifest); the rng ALGORITHM is, since it is a method choice.
    import dataclasses
    assert "master_seed" not in {f.name for f in dataclasses.fields(ProtocolConfig)}
    assert ProtocolConfig(rng_algorithm="x").hash() != a.hash()
    man = a.manifest()
    assert isinstance(man["tau_detect"], dict) and man["tau_detect"]["method"], "tau as RULE"
    print(f"OK manifest = FULL sha256 ({a.hash()[:12]}...); tau as RULE; rng in hash")


def test_fail_closed_and_validate():
    from csc.protocol import ProtocolError, execute_protocol
    src = make_source(SimConfig(seed=3), n_domains=6, concept_domains=2, seed=3)
    tb = make_target("covariate", SimConfig(seed=3), geom=src.geom, seed=300)
    # group_aware=True but NO group ids -> must FAIL CLOSED (not silently IID)
    try:
        execute_protocol(src.Z, src.Y, src.D, tb.Z, ProtocolConfig(group_aware=True), seed=3)
        raise AssertionError("expected ProtocolError for group_aware without ids")
    except ProtocolError:
        pass
    # unsupported config value -> validate() rejects
    for bad in (dict(quantile_convention="midpoint"), dict(analysis_unit="trial"),
                dict(oracle_eps_stable_ce=0.05, oracle_eps_concept_ce=0.03)):
        try:
            ProtocolConfig(**bad).validate()
            raise AssertionError(f"validate() should reject {bad}")
        except ProtocolError:
            pass
    print("OK manifest is EXECUTABLE: fail-closed on missing group ids; validate() rejects "
          "unsupported values")


if __name__ == "__main__":
    test_calibrate_tau_detect_keyword_safe()
    test_frozen_path_taxonomy()
    test_frozen_path_is_cluster_aware()
    test_manifest_hash_deterministic_and_full()
    test_fail_closed_and_validate()
    print("\nall protocol/calibration tests passed")

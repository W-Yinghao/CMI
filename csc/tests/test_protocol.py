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

CFG = ProtocolConfig(n_boot=15, n_dir_boot=80, target_n_boot=60, tau_n_pseudotargets=100)


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
    must_abstain_forbidden = 0      # STRUCTURAL guarantee -> must be exactly 0
    full_forbidden = 0              # includes rare covariate crying-wolf (statistical, soft)
    n = 4
    for s in range(n):
        scfg = SimConfig(seed=s)
        src = make_source(scfg, n_domains=8, concept_domains=3, seed=s)
        for kind, want in expect.items():
            tb = make_target(kind, scfg, geom=src.geom, seed=100 + s)
            st = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, CFG, seed=s)["certificate"].state
            ok[kind] += int(st == want)
            if st in FORBIDDEN[tb.truth]:
                full_forbidden += 1
                if kind in ("clean", "pure_conditional", "label_shift"):
                    must_abstain_forbidden += 1
    # HARD structural guarantees: unidentifiable shifts (clean/pure/label) are NEVER falsely
    # certified, and concept power exists. The full-suite forbidden rate (covariate crying-wolf)
    # is a STATISTICAL quantity controlled via run_synthetic's exact-CP endpoint, not a hard 0
    # over 4 seeds -- here we only guard against gross regression (<=1).
    assert must_abstain_forbidden == 0, f"unidentifiable shift FALSE-certified ({must_abstain_forbidden})"
    assert ok["clean"] == n and ok["pure_conditional"] == n and ok["label_shift"] == n
    assert ok["boundary_coupled"] >= 1, "frozen path shows no concept power at all"
    assert full_forbidden <= 1, f"gross forbidden-rate regression: {full_forbidden}/{4*len(expect)}"
    print(f"OK frozen-path: must-abstain 0 forbidden; concept {ok['boundary_coupled']}/{n}; "
          f"covariate {ok['covariate']}/{n}; full-suite forbidden {full_forbidden} (<=1 soft)")


def test_frozen_path_is_cluster_aware():
    scfg = SimConfig(seed=2)
    src = make_source(scfg, n_domains=8, concept_domains=3, seed=2)
    # synthetic subject ids: 5 subjects per domain
    rng = np.random.default_rng(2)
    src_groups = src.D * 100 + rng.integers(0, 5, len(src.D))
    tb = make_target("covariate", scfg, geom=src.geom, seed=200)
    tgt_groups = rng.integers(0, 6, len(tb.Z))
    out = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, CFG,
                              src_group_ids=src_groups, tgt_group_ids=tgt_groups, seed=2)
    assert out["certificate"].state in (COVARIATE_COMPATIBLE, UNIDENTIFIABLE)
    assert out["analysis"].detail["cluster_aware"] is True
    print(f"OK cluster-aware frozen path runs -> {out['certificate'].state}")


def test_manifest_hash_deterministic():
    a, b = ProtocolConfig(), ProtocolConfig()
    assert a.hash() == b.hash()
    assert ProtocolConfig(consensus=0.9).hash() != a.hash()
    man = a.manifest()
    assert isinstance(man["tau_detect"], dict) and man["tau_detect"]["method"], "tau as RULE"
    print(f"OK manifest hash deterministic ({a.hash()}); tau_detect recorded as a RULE")


if __name__ == "__main__":
    test_calibrate_tau_detect_keyword_safe()
    test_frozen_path_taxonomy()
    test_frozen_path_is_cluster_aware()
    test_manifest_hash_deterministic()
    print("\nall protocol/calibration tests passed")

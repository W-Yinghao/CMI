"""
CSC-P1.5 difficulty-envelope HARNESS tests (structure + one tiny cell). These validate the
harness mechanics ONLY -- they are NOT a sweep and select nothing. The envelope is DEVELOPMENT
scaffolding pending reviewer approval, so this module is intentionally NOT in the audited
TEST_MODULES gate yet; it runs standalone.
"""
import os
# deterministic single-threaded BLAS (matches the sbatch) so the parallel==serial test is exact:
# threaded BLAS has non-deterministic reduction order that could flip a borderline certificate.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.run_envelope import (
    EnvelopePoint, _materialize, default_grid, run_cell, run_envelope, _DEFAULT_LEVELS, KINDS,
)
from csc.protocol import ProtocolConfig


# 1 ---- each difficulty axis threads to the documented simulator knob -----------------------------
def test_axis_to_knob_mapping():
    p = EnvelopePoint()
    q = p.with_axis("concept_effect_size", 22.0).with_axis("target_subjects", 11)
    cfg, src_kw, tgt_kw = _materialize(q, src_seed=3)
    assert tgt_kw["concept_target_scale"] == 22.0, "concept_effect_size -> target concept scale"
    assert tgt_kw["subjects"] == 11, "target_subjects -> make_target subjects"
    # within_subject_corr / epochs / imbalance map onto SimConfig
    r = p.with_axis("within_subject_corr", 0.7).with_axis("prior_alpha", 0.5)
    cfg2, _, _ = _materialize(r, src_seed=3)
    assert cfg2.subject_tau == 0.7 and cfg2.prior_alpha == 0.5
    # mechanism_family offsets the geom seed (distinct latent geometry), clusters still independent
    cfg_f0, _, _ = _materialize(p.with_axis("mechanism_family", 0), src_seed=5)
    cfg_f1, _, _ = _materialize(p.with_axis("mechanism_family", 1), src_seed=5)
    assert cfg_f0.seed != cfg_f1.seed, "mechanism family must change the geom seed"
    # unknown axis fails closed
    try:
        p.with_axis("not_an_axis", 1)
        raise AssertionError("unknown axis must raise")
    except KeyError:
        pass
    print("OK each axis maps to its simulator knob; unknown axis fails closed")


# 2 ---- the default grid is a STAR design (baseline once + per-axis levels) ----------------------
def test_default_grid_is_star():
    cells = default_grid()
    labels = [c for c, _ in cells]
    assert labels[0] == "baseline" and labels.count("baseline") == 1
    base = EnvelopePoint()
    # every non-baseline cell differs from baseline in EXACTLY the one named axis
    for label, pt in cells[1:]:
        axis, _, val = label.partition("=")
        assert axis in _DEFAULT_LEVELS
        diffs = [f.name for f in pt.__dataclass_fields__.values()
                 if getattr(pt, f.name) != getattr(base, f.name)]
        assert diffs == [axis], f"cell {label} varies {diffs}, expected only [{axis}]"
    print(f"OK default grid is a star design: {len(cells)} cells, one axis varied each")


# 3 ---- run_cell is CLUSTER-denominated and returns the full operating-region block --------------
def test_run_cell_cluster_denominated():
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=40, target_n_boot=30, tau_n_pseudotargets=40, oracle_boot=10)
    cfg.validate()
    K = 2
    b = run_cell(EnvelopePoint(), cfg, n_clusters=K, base_seed=0)
    assert b["n_independent_clusters"] == K
    # every required reviewer metric is present
    for m in ("any_forbidden_full_suite", "any_forbidden_full_suite_cp_upper",
              "false_concept_on_synthetic_null", "false_concept_on_synthetic_null_cp_upper",
              "visible_concept_power", "visible_concept_power_cp_lower",
              "covariate_compatible_coverage", "abstention_rate_all_cells",
              "source_invalid_rate", "support_invalid_rate", "attribution_unassessable_rate",
              "attribution_unstable_rate", "atlas_availability", "gate_failure_decomposition",
              "robust_consensus_abstain", "residual_T_not_sig", "geometric_maxstat_not_sig"):
        assert m in b, f"missing metric {m}"
    # counts are bounded by the CLUSTER count (not the #targets), CP bounds are probabilities
    assert 0 <= b["any_forbidden_full_suite"] <= K
    assert 0.0 <= b["any_forbidden_full_suite_cp_upper"] <= 1.0
    assert 0.0 <= b["visible_concept_power_cp_lower"] <= 1.0
    # the gate-failure decomposition partitions ALL K visible clusters (FIRED + failure reasons)
    assert sum(b["gate_failure_decomposition"].values()) == K
    assert b["eigengap_axis_is_proxy"] is True
    print(f"OK run_cell cluster-denominated (K={K}): full metric block, decomp partitions clusters")


# 4 ---- the --phase full gate fails closed unless a PASSED, matching canary is referenced ---------
def test_full_gate_fails_closed():
    import json, tempfile
    from csc.run_envelope_p15 import verify_canary_ref, _GateError
    HEAD, MH = "abc1234", "manifest123"

    def art(**ovr):
        a = dict(phase="CANARY_ONLY_PASSED", code_commit=HEAD, protocol_manifest_hash=MH,
                 canary=dict(validator_ok=True, clusters_per_cell=2, base_seed=500_000))
        a.update(ovr); return a

    def write(a):
        p = tempfile.mktemp(suffix=".json"); json.dump(a, open(p, "w")); return p

    # a well-formed, matching canary passes the gate
    verify_canary_ref(write(art()), HEAD, MH, 2, 500_000)
    # every mismatch / missing-ref / not-passed / not-validated fails CLOSED
    bad = [
        (None, HEAD, MH, 2, 500_000),                                   # no ref
        (write(art(phase="ABORTED_AT_CANARY")), HEAD, MH, 2, 500_000),  # not passed
        (write(art(canary=dict(validator_ok=False, clusters_per_cell=2, base_seed=500_000))),
         HEAD, MH, 2, 500_000),                                          # validator not ok
        (write(art()), "OTHER", MH, 2, 500_000),                        # code-commit mismatch
        (write(art()), HEAD, "OTHER", 2, 500_000),                      # manifest mismatch
        (write(art()), HEAD, MH, 5, 500_000),                           # canary clusters mismatch
        (write(art()), HEAD, MH, 2, 999_999),                           # base-seed mismatch
    ]
    for argspec in bad:
        try:
            verify_canary_ref(*argspec)
            raise AssertionError(f"gate should have failed closed for {argspec[1:]}")
        except (_GateError, OSError, ValueError):
            pass
    print("OK --phase full gate: accepts a matching PASSED canary, fails closed on all 7 mismatches")


# 5 ---- joblib parallelism is BIT-IDENTICAL to serial (deterministic per-seed records) ----------
def test_parallel_matches_serial():
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=40, target_n_boot=30, tau_n_pseudotargets=40, oracle_boot=10)
    cfg.validate()
    cells = default_grid()[:2]                       # baseline + 1 axis cell
    serial = run_envelope(cells, cfg, n_clusters=2, base_seed=500_000, n_jobs=1)["grid"]
    par = run_envelope(cells, cfg, n_clusters=2, base_seed=500_000, n_jobs=2)["grid"]
    assert len(serial) == len(par) == 2
    for a, b in zip(serial, par):
        assert a == b, f"parallel != serial for cell {a.get('cell')}: differs"
    print("OK joblib parallel grid is BIT-IDENTICAL to serial (n_jobs changes wall-clock only)")


if __name__ == "__main__":
    test_axis_to_knob_mapping()
    test_default_grid_is_star()
    test_run_cell_cluster_denominated()
    test_full_gate_fails_closed()
    test_parallel_matches_serial()
    print("\nall CSC-P1.5 envelope-harness tests passed")

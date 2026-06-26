"""
CSC-P1.5 difficulty-envelope HARNESS tests (structure + one tiny cell). These validate the
harness mechanics ONLY -- they are NOT a sweep and select nothing. The envelope is DEVELOPMENT
scaffolding pending reviewer approval, so this module is intentionally NOT in the audited
TEST_MODULES gate yet; it runs standalone.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.run_envelope import (
    EnvelopePoint, _materialize, default_grid, run_cell, _DEFAULT_LEVELS, KINDS,
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


if __name__ == "__main__":
    test_axis_to_knob_mapping()
    test_default_grid_is_star()
    test_run_cell_cluster_denominated()
    print("\nall CSC-P1.5 envelope-harness tests passed")

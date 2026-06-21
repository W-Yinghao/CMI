"""B2a analyzer: the metadata_gated comparator must adapt+help on DIAG geometry, abstain on
non-DIAG, and the acceptance clauses + extra diagnostics must compute correctly."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from h2cmi.analyze_b2a import analyze, COMPARATORS


def _row(seed, site, ei, comp, op, dbacc, eff_geom, *, gate_pass=False, meta_op="identity",
         geom="NONE", prev="SAME", sel_bacc=0.6):
    return dict(data_seed=seed, target_site=site, episode=ei, comparator=comp, selected_op=op,
                adapted=(op != "identity"), dbacc_full=dbacc, selected_bacc=sel_bacc, eff_geom=eff_geom,
                geometry_compatibility=geom, prevalence_risk=prev, metadata_operator=meta_op,
                metadata_gate_pass=gate_pass)


def _episode(seed, site, ei, *, eff_geom, meta_op, gate_pass, geom, prev, win_op, win_dbacc):
    # oracle picks win_op; metadata_gated picks meta_op if gate_pass else identity
    mg_op = meta_op if gate_pass and meta_op != "identity" else "identity"
    mg_db = win_dbacc if mg_op == win_op else (0.0 if mg_op == "identity" else -0.01)
    return [
        _row(seed, site, ei, "identity", "identity", 0.0, eff_geom, geom=geom, prev=prev),
        _row(seed, site, ei, "always_pooled", "pooled_empirical_diag", win_dbacc if win_op == "pooled_empirical_diag" else -0.02, eff_geom, geom=geom, prev=prev),
        _row(seed, site, ei, "always_canonical", "gen_oneshot_diag", win_dbacc if win_op == "gen_oneshot_diag" else -0.02, eff_geom, geom=geom, prev=prev),
        _row(seed, site, ei, "n1_target_ranking", "identity", 0.0, eff_geom, geom=geom, prev=prev),
        _row(seed, site, ei, "metadata_gated", mg_op, mg_db, eff_geom, gate_pass=gate_pass, meta_op=meta_op, geom=geom, prev=prev,
             sel_bacc=0.6 + (win_dbacc if mg_op == win_op else 0.0)),
        _row(seed, site, ei, "metadata_oracle", win_op, win_dbacc, eff_geom, geom=geom, prev=prev, sel_bacc=0.6 + win_dbacc)]


def _grid():
    rows = []
    ei = 0
    for seed in (0, 1, 2):
        for site in (0, 1):
            # DIAG episodes: metadata picks pooled, gate passes, helps
            for _ in range(3):
                rows += _episode(seed, site, ei, eff_geom="DIAG_COMPATIBLE", meta_op="pooled_empirical_diag",
                                 gate_pass=True, geom="DIAG_COMPATIBLE", prev="SAME",
                                 win_op="pooled_empirical_diag", win_dbacc=0.06); ei += 1
            # NONE episodes: metadata says identity -> abstain
            for _ in range(2):
                rows += _episode(seed, site, ei, eff_geom="NONE", meta_op="identity", gate_pass=False,
                                 geom="NONE", prev="SAME", win_op="identity", win_dbacc=0.0); ei += 1
    return rows


def test_metadata_gated_passes_on_clean_grid():
    rep = analyze(_grid())
    mg = rep["by_comparator"]["metadata_gated"]
    assert mg["false_adaptation_rate"] == 0.0          # never adapts on NONE
    assert mg["coverage"] > 0.4 and mg["mean_dbacc_diag"] > 0.05
    assert mg["top1_oracle"] == 1.0 and mg["passes"]["ALL"]
    assert "B2A_PASS" in rep["decision"]


def test_extra_diagnostics_present():
    rep = analyze(_grid())
    mg = rep["by_comparator"]["metadata_gated"]
    for k in ("gate_positive_operator_regret", "metadata_op_veto_fail_rate",
              "missing_metadata_abstention_rate", "pooled_vs_cc_confusion"):
        assert k in mg
    assert set(COMPARATORS) <= set(rep["by_comparator"])


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"  {n} PASSED")
    print("test_analyze_b2a PASSED")

"""Stage-B tests for the Mechanism-Subspace Oracle AGGREGATOR (scripts/aggregate_mechanism_subspace_oracle.py),
amendment 03 schema. Covers _holm, _cluster_ci (bootstrap + exact sign-flip p), and _cell_specific SYMMETRIC
safe-vs-safe / unc-vs-unc against the shared-null-Haar primary control with fail-closed AMBIENT_ONLY fallback."""
import importlib.util
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("mech_agg", _REPO / "scripts/aggregate_mechanism_subspace_oracle.py")
AGG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AGG)


def test_holm_step_down_monotone_and_capped():
    adj = AGG._holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06]) and all(0.0 <= x <= 1.0 for x in adj)
    assert all(x <= 1.0 for x in AGG._holm([0.9, 0.95]))


def test_cluster_ci_bootstrap_and_exact_signflip():
    pos = AGG._cluster_ci([0.05, 0.06, 0.04, 0.05], n_boot=2000)
    neg = AGG._cluster_ci([-0.05, -0.06, -0.04], n_boot=2000)
    assert pos["lo"] > 0 and pos["signflip_p"] == 1.0 / (1 << 4)      # all-positive n=4 -> exact 1/16
    assert neg["hi"] < 0 and neg["signflip_p"] == 1.0                 # all-negative -> p=1
    assert AGG._cluster_ci([])["n"] == 0


def test_cell_specific_safe_vs_safe_and_unc_vs_unc():
    # PRIMARY shared-null-Haar control present -> symmetric contrasts against it (P0.1)
    row = dict(dU_informed_safe=0.03, dU_informed_unc=0.05, primary_control="SHARED_NULL_HAAR",
               shared_null_haar=[dict(dU_safe=0.01, dU_unc=0.02, selected_safe_gdis_capture=0.1, dictionary_gdis_capture=0.4, subspace_overlap=0.5),
                                 dict(dU_safe=0.00, dU_unc=0.01, selected_safe_gdis_capture=0.2, dictionary_gdis_capture=0.5, subspace_overlap=0.5)],
               ambient=[dict(dU_safe=0.05, dU_unc=0.06, selected_safe_gdis_capture=0.1, dictionary_gdis_capture=0.4, subspace_overlap=0.5)],
               selected_safe_gdis_capture=0.3, dictionary_gdis_capture=0.9)
    cs = AGG._cell_specific(row)
    assert cs["control"] == "SHARED_NULL_HAAR"
    assert np.isclose(cs["dU_safe_specific"], 0.03 - np.mean([0.01, 0.0]))     # safe informed vs safe RANDOM
    assert np.isclose(cs["dU_unc_specific"], 0.05 - np.mean([0.02, 0.01]))     # unc informed vs unc RANDOM
    # forbidden asymmetry never happens: safe uses control dU_safe (0.005), NOT ambient/unc


def test_cell_specific_falls_back_to_ambient_when_low_dof():
    # shared-null control degenerate/absent -> AMBIENT_ONLY, flagged (never silently called the primary control)
    row = dict(dU_informed_safe=0.03, dU_informed_unc=0.05, primary_control="SHARED_NULL_CONTROL_LOW_DOF",
               shared_null_haar=None, ambient=[dict(dU_safe=0.01, dU_unc=0.02, selected_safe_gdis_capture=0.1, dictionary_gdis_capture=0.4, subspace_overlap=0.5)],
               selected_safe_gdis_capture=0.3, dictionary_gdis_capture=0.9)
    cs = AGG._cell_specific(row)
    assert cs["control"] == "AMBIENT_ONLY"
    # no reference at all -> None (dropped from stats, not fabricated)
    assert AGG._cell_specific(dict(dU_informed_safe=0.03, primary_control="SHARED_NULL_CONTROL_LOW_DOF",
                                   shared_null_haar=None, ambient=None)) is None
    # non-finite informed -> None
    assert AGG._cell_specific(dict(dU_informed_safe=float("nan"), shared_null_haar=[dict(dU_safe=0.0, dU_unc=0.0)])) is None

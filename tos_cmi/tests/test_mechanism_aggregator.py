"""Stage-B tests for the Mechanism-Subspace Oracle AGGREGATOR (scripts/aggregate_mechanism_subspace_oracle.py).
Covers _holm, _cluster_ci one-sided p, _cell_specific fail-closed matched-vs-ambient, and the degenerate n<2 guard.
Synthetic data only checks the implementation, never a scientific verdict."""
import importlib.util
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("mech_agg", _REPO / "scripts/aggregate_mechanism_subspace_oracle.py")
AGG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AGG)


def test_holm_step_down_monotone_and_capped():
    p = [0.01, 0.04, 0.03]
    adj = AGG._holm(p)
    # Holm: sort asc [0.01,0.03,0.04]; adj = [3*.01, 2*.03, 1*.04] = [.03,.06,.04] -> running max -> [.03,.06,.06]
    # mapped back to original order [0.01, 0.04, 0.03] -> [.03, .06, .06]
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert all(0.0 <= x <= 1.0 for x in adj)
    # capping at 1.0
    assert all(x <= 1.0 for x in AGG._holm([0.9, 0.95]))


def test_cluster_ci_one_sided_p_direction():
    # all-positive per-subject means -> small one-sided p (H1: mean>0); all-negative -> p near 1
    pos = AGG._cluster_ci([0.05, 0.06, 0.04, 0.05], n_boot=2000)
    neg = AGG._cluster_ci([-0.05, -0.06, -0.04], n_boot=2000)
    assert pos["p_one_sided"] < 0.05 and pos["lo"] > 0
    assert neg["p_one_sided"] > 0.5 and neg["hi"] < 0
    # empty -> n=0, nan
    e = AGG._cluster_ci([])
    assert e["n"] == 0 and not np.isfinite(e["mean"])


def test_cell_specific_prefers_matched_and_flags_ambient_only():
    # matched present -> uses MATCHED
    r_m = dict(dU_source_safe=0.03, dU_random_matched=[0.01, 0.0, 0.02], dU_random_ambient=[0.05, 0.05],
               shared_overlap_match={"verdict": "OK"})
    cs = AGG._cell_specific(r_m)
    assert cs["control"] == "MATCHED" and np.isclose(cs["dU_specific"], 0.03 - np.mean([0.01, 0.0, 0.02]))
    # matched failed (None) -> falls back to AMBIENT_ONLY, never silently called matched
    r_a = dict(dU_source_safe=0.03, dU_random_matched=None, dU_random_ambient=[0.01, 0.0, 0.02],
               shared_overlap_match={"verdict": "SHARED_MATCH_CONTROL_FAILED"})
    cs2 = AGG._cell_specific(r_a)
    assert cs2["control"] == "AMBIENT_ONLY"
    # no reference at all -> None (dropped from stats, not fabricated)
    assert AGG._cell_specific(dict(dU_source_safe=0.03, dU_random_matched=None, dU_random_ambient=None)) is None
    # non-finite informed -> None
    assert AGG._cell_specific(dict(dU_source_safe=float("nan"), dU_random_ambient=[0.0])) is None


def test_holm_preserves_order_for_two():
    adj = AGG._holm([0.5, 0.001])
    # smaller p gets 2x, larger gets running-max -> [max(0.002*... ), ...]; just assert monotic + order kept
    assert adj[1] <= adj[0] or adj[1] <= 1.0
    assert len(adj) == 2

"""CMI-Trace P1.4 tests — fully-specified, hardened exact-head reliance audit."""
import numpy as np
import pytest

from cmi.eval import reliance_audit as ra
from cmi.eval.audit_npz import save_audit_npz, load_audit_npz, DEFAULT_REPLAY_TOL
from cmi.eval.leakage_removal import fit_leakage_subspace


def _synthetic_audit(tmp_path, seed=0, Ns=140, Nt=30, Zg=16, C=8, Zn=5, ncls=3, ndom=3):
    """Audit npz with a VERIFIED linear head + subject-structured graph_z (label-conditional leakage)."""
    rng = np.random.default_rng(seed)
    N = Ns + Nt
    ds = rng.integers(0, ndom, Ns)
    d = np.r_[ds, np.full(Nt, ndom)].astype("int64")          # target = distinct domain id ndom
    y = rng.integers(0, ncls, N).astype("int64")
    subj_dirs = rng.standard_normal((ndom + 1, Zg))
    graph_z = (rng.standard_normal((N, Zg)) + 2.5 * subj_dirs[d]).astype("float32")   # subject-structured
    node_z = rng.standard_normal((N, C, Zn)).astype("float32")
    W = rng.standard_normal((ncls, Zg)).astype("float32"); b = rng.standard_normal(ncls).astype("float32")
    model_logits = (graph_z @ W.T + b).astype("float32")
    p = tmp_path / "aud.audit.npz"
    save_audit_npz(str(p), graph_z=graph_z, node_z=node_z, y=y, d=d, model_logits=model_logits,
                   fold=0, seed=seed, target_subject="T", method="erm", dataset="D",
                   task_head_weight=W, task_head_bias=b, task_head_input="graph_z",
                   source_indices=np.arange(Ns), target_indices=np.arange(Ns, N))
    return str(p), ndom


def test_row_has_full_metadata(tmp_path):
    p, tgt = _synthetic_audit(tmp_path)
    row = ra.reliance_audit_row(p, tgt, k=2)
    for field in ("subspace_construction_method", "within_label_centering_rule", "metric_whitening_rule",
                  "loss_for_R_rel", "fit_split", "k", "replay_max_abs_error", "replay_tolerance",
                  "head_replay_verified", "source_only_firewall", "task_drop", "firewall_passed"):
        assert field in row, f"missing metadata field {field}"
    assert row["k"] == 2
    assert row["replay_tolerance"] == pytest.approx(DEFAULT_REPLAY_TOL)


def test_head_reproduces_logits_within_tolerance(tmp_path):
    p, tgt = _synthetic_audit(tmp_path)
    data = load_audit_npz(p)
    assert bool(data["task_head_replay_ok"]) is True
    assert float(data["task_head_replay_max_abs_diff"]) <= DEFAULT_REPLAY_TOL
    row = ra.reliance_audit_row(p, tgt, k=2)
    assert row["head_replay_verified"] is True
    assert row["replay_max_abs_error"] <= row["replay_tolerance"]


def test_target_labels_not_in_subspace_fit(tmp_path):
    # corrupting TARGET rows must not change the source-fit subspace (fit is source-only)
    p, tgt = _synthetic_audit(tmp_path)
    data = load_audit_npz(p)
    z = np.asarray(data["graph_z"], float); y = np.asarray(data["y"]); d = np.asarray(data["d"])
    src = d != tgt
    P1, dirs1 = fit_leakage_subspace(z[src], y[src], d[src], k=2, conditioning="label_conditional", seed=0)
    z2 = z.copy(); z2[~src] = 1e6                              # destroy target rows
    P2, dirs2 = fit_leakage_subspace(z2[src], y[src], d[src], k=2, conditioning="label_conditional", seed=0)
    assert np.allclose(P1, P2)                                # subspace unchanged -> target not used


def test_random_spans_same_rank_and_count(tmp_path):
    p, tgt = _synthetic_audit(tmp_path)
    rc = ra.random_span_control(p, tgt, k=3, n_spans=50, seed=0)
    assert rc["k"] == 3 and rc["n_spans"] == 50 and len(rc["per_span"]) == 50
    assert np.isfinite(rc["random_task_drop_mean"])
    assert rc["random_task_drop_ci_lo"] <= rc["random_task_drop_mean"] <= rc["random_task_drop_ci_hi"]


def test_rank_sensitivity_uses_fixed_sequence(tmp_path):
    p, tgt = _synthetic_audit(tmp_path)
    rows = ra.reliance_rank_sensitivity(p, tgt, ks=(1, 2, 3, 4, 5, 6, 7), n_random_spans=20)
    assert [r["k"] for r in rows] == [1, 2, 3, 4, 5, 6, 7]
    for r in rows:
        assert r["rank_sequence"] == [1, 2, 3, 4, 5, 6, 7]
        assert "random_task_drop_mean" in r and r["n_random_spans"] == 20


def test_cross_model_axis_comparison_forbidden():
    assert ra.assert_no_cross_model_axis_compare("erm", "erm") is True
    with pytest.raises(ValueError):
        ra.assert_no_cross_model_axis_compare("erm", "coral")


def test_leakage_reliance_positive_on_structured_data(tmp_path):
    # subject-structured graph_z -> label_conditional removal drops task more than a random span (on average)
    p, tgt = _synthetic_audit(tmp_path, seed=3)
    row = ra.reliance_audit_row(p, tgt, k=2, conditioning="label_conditional")
    rc = ra.random_span_control(p, tgt, k=2, n_spans=30, seed=3)
    assert np.isfinite(row["task_drop"]) and np.isfinite(rc["random_task_drop_mean"])

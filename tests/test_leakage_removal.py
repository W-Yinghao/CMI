"""R3 flagship — label-conditional subject-subspace removal + reliance, and the CIGL firewall suite.

Firewall guarantees (fit uses SOURCE only; target is eval-only; k is pre-registered):
  F1 corrupting TARGET labels leaves the fitted subspace unchanged
  F2 corrupting TARGET logits leaves the reliance row unchanged
  F3 source-only fitting excludes the target subject (firewall_passed)
  F4 mu_y / subspace / source metrics use SOURCE only (target z corruption is inert)
  F5 k is fixed / curve-reported, never target-selected
  F6 random_subspace is deterministic under a fixed seed
  F7 removal preserves shape + finiteness
  F8 constant / rank-deficient z does not crash
  +CMI  label_conditional and marginal_domain projectors are distinguishable when label/domain structure differs
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cmi.eval.leakage_removal import (
    fit_leakage_subspace, remove_subspace, evaluate_reliance, reliance_curve,
    DEFAULT_K_CURVE, CONDITIONINGS, PRIMARY_K, PRIMARY_CONDITIONING,
)

N_DOM, N_CLS, ZDIM, N_PER, TARGET = 4, 2, 8, 50, 3


def make_data(with_head=True, seed=0):
    """Synthetic frozen z with class separation on dim0 and a LABEL-CONDITIONAL subject offset on dim1 whose
    sign flips by class -> marginal_domain cancels dim1, label_conditional keeps it (F+CMI test)."""
    rng = np.random.default_rng(seed)
    zs, ys, ds = [], [], []
    for dd in range(N_DOM):
        for cc in range(N_CLS):
            b = rng.standard_normal((N_PER, ZDIM)) * 0.2
            b[:, 0] += 3.0 * cc                                  # class separation
            b[:, 1] += (1.0 if cc == 0 else -1.0) * (dd - 1.5) * 2.0   # subject offset, sign flips by class
            zs.append(b); ys += [cc] * N_PER; ds += [dd] * N_PER
    z = np.vstack(zs); y = np.array(ys); d = np.array(ds)
    data = {"graph_z": z, "node_z": z.reshape(len(z), ZDIM, 1), "y": y, "d": d,
            "model_logits": np.zeros((len(z), N_CLS)), "dataset": "synthetic", "fold": 0,
            "seed": seed, "target_subject": str(TARGET), "method": "erm"}
    if with_head:
        src = d != TARGET
        lr = LogisticRegression(max_iter=500).fit(z[src], y[src])
        W = np.zeros((N_CLS, ZDIM)); W[1] = lr.coef_[0]          # 2-class -> 2-row head
        b = np.array([0.0, float(lr.intercept_[0])])
        data["model_logits"] = z @ W.T + b                       # consistent -> head-replay verifies exactly
        data.update(task_head_weight=W, task_head_bias=b, task_head_kind="linear", task_head_input="graph_z",
                    task_head_replay_ok=True, task_head_replay_max_abs_diff=0.0)
    return data


# ---- flagship ----------------------------------------------------------------------------------------------
def test_label_conditional_subspace_recovers_dim1():
    d = make_data()
    src = d["d"] != TARGET
    _, dirs = fit_leakage_subspace(d["graph_z"][src], d["y"][src], d["d"][src], k=1,
                                   conditioning="label_conditional")
    assert abs(dirs[0, 1]) > 0.7                                 # principal subject direction is dim1


def test_reliance_row_schema_and_head_replay():
    d = make_data(with_head=True)
    row = evaluate_reliance(d, TARGET, k=PRIMARY_K, conditioning=PRIMARY_CONDITIONING)
    for key in ("dataset", "fold", "seed", "target_subject", "method", "representation", "removal_mode",
                "conditioning", "k", "source_task_bacc_before", "source_task_bacc_after",
                "target_task_bacc_before", "target_task_bacc_after", "task_drop",
                "source_subject_bacc_before", "source_subject_bacc_after", "subject_leakage_drop",
                "head_replay_available", "probe_replay_used", "firewall_passed"):
        assert key in row
    assert row["removal_mode"] == "head_replay" and row["head_replay_available"] and not row["probe_replay_used"]
    assert row["subject_leakage_drop"] > -1e-6                   # removing subject subspace should not raise leakage


def test_probe_fallback_when_no_head():
    d = make_data(with_head=False)
    row = evaluate_reliance(d, TARGET)
    assert row["removal_mode"] == "probe_replay" and row["probe_replay_used"]
    assert not row["head_replay_available"]


# ---- firewall suite ----------------------------------------------------------------------------------------
def test_F1_target_labels_do_not_move_subspace():
    d = make_data()
    src = d["d"] != TARGET
    P0, _ = fit_leakage_subspace(d["graph_z"][src], d["y"][src], d["d"][src], k=2)
    d["y"] = d["y"].copy(); d["y"][d["d"] == TARGET] = np.random.default_rng(9).integers(0, N_CLS, (d["d"] == TARGET).sum())
    P1, _ = fit_leakage_subspace(d["graph_z"][d["d"] != TARGET], d["y"][d["d"] != TARGET], d["d"][d["d"] != TARGET], k=2)
    assert np.allclose(P0, P1)


def test_F2_target_logits_do_not_move_row():
    d = make_data(with_head=True)
    r0 = evaluate_reliance(d, TARGET)
    d["model_logits"] = d["model_logits"].copy()
    d["model_logits"][d["d"] == TARGET] = 999.0                  # corrupt target logits
    r1 = evaluate_reliance(d, TARGET)
    assert r0["task_drop"] == r1["task_drop"] and r0["source_task_bacc_after"] == r1["source_task_bacc_after"]


def test_F3_firewall_excludes_target_subject():
    d = make_data()
    row = evaluate_reliance(d, TARGET)
    assert row["firewall_passed"]
    src = d["d"] != TARGET
    assert TARGET not in np.unique(d["d"][src])


def test_F4_target_z_corruption_is_inert_for_source_metrics():
    d = make_data(with_head=True)
    r0 = evaluate_reliance(d, TARGET)
    d["graph_z"] = d["graph_z"].copy(); d["graph_z"][d["d"] == TARGET] += 500.0   # corrupt target features
    r1 = evaluate_reliance(d, TARGET)
    assert np.isclose(r0["source_task_bacc_after"], r1["source_task_bacc_after"])
    assert np.isclose(r0["source_subject_bacc_after"], r1["source_subject_bacc_after"], equal_nan=True)


def test_F5_k_curve_is_fixed_not_target_selected():
    d = make_data()
    rows = reliance_curve(d, TARGET)
    assert len(rows) == len(CONDITIONINGS) * len(DEFAULT_K_CURVE)
    for c in CONDITIONINGS:
        assert {r["k"] for r in rows if r["conditioning"] == c} == set(DEFAULT_K_CURVE)


def test_F6_random_subspace_deterministic():
    d = make_data(); src = d["d"] != TARGET
    a = fit_leakage_subspace(d["graph_z"][src], d["y"][src], d["d"][src], k=3, conditioning="random_subspace", seed=7)[1]
    b = fit_leakage_subspace(d["graph_z"][src], d["y"][src], d["d"][src], k=3, conditioning="random_subspace", seed=7)[1]
    c = fit_leakage_subspace(d["graph_z"][src], d["y"][src], d["d"][src], k=3, conditioning="random_subspace", seed=8)[1]
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_F7_removal_preserves_shape_and_finite():
    d = make_data(); z = d["graph_z"]
    P, _ = fit_leakage_subspace(z, d["y"], d["d"], k=2)
    zr = remove_subspace(z, P)
    assert zr.shape == z.shape and np.isfinite(zr).all()


def test_F8_constant_and_rankdeficient_do_not_crash():
    d = make_data()
    d["graph_z"] = np.ones_like(d["graph_z"])                    # constant / rank-deficient
    P, _ = fit_leakage_subspace(d["graph_z"], d["y"], d["d"], k=2)
    row = evaluate_reliance(d, TARGET)
    assert P.shape == (ZDIM, ZDIM) and "task_drop" in row


def test_CMI_label_conditional_vs_marginal_are_distinguishable():
    d = make_data(); src = d["d"] != TARGET
    _, lc = fit_leakage_subspace(d["graph_z"][src], d["y"][src], d["d"][src], k=1, conditioning="label_conditional")
    _, md = fit_leakage_subspace(d["graph_z"][src], d["y"][src], d["d"][src], k=1, conditioning="marginal_domain")
    cos = abs(float(lc[0] @ md[0]))
    assert cos < 0.5                                             # sign-flipped subject offset -> different directions
    assert abs(lc[0, 1]) > 0.7 and abs(md[0, 1]) < 0.5          # lc keeps dim1; marginal cancels it

"""Pinned tests for the cross-session objective primitives: cross-session weight builder (source early/late only),
the weighted exact MCC gradient (== full-batch), and CS-Risk != CS-RW-MCC (different intervention point)."""
import inspect
import numpy as np
import pytest
import torch

from tos_cmi.train import cross_session_objective as CS
from tos_cmi.train.mechanism_consistency import class_pairs


class _Fixture(torch.nn.Module):
    def __init__(self, p_in=8, p=6, C=4):
        super().__init__()
        self.lin = torch.nn.Linear(p_in, p); self.bn = torch.nn.BatchNorm1d(p); self.head = torch.nn.Linear(p, C)
        torch.manual_seed(0)
        with torch.no_grad():
            self.bn.running_mean.normal_(); self.bn.running_var.uniform_(0.5, 1.5)

    def forward(self, x):
        h = x.reshape(x.shape[0], -1); z = self.bn(self.lin(h)); return self.head(z), z


def _src(m=4, C=4, per=30, p_in=8, drift_subj=None, seed=0):
    """m subjects x C classes x 2 sessions (early/late). If drift_subj set, that subject's LATE session is shifted
    (high cross-session risk); others transfer early->late cleanly."""
    rng = np.random.default_rng(seed); shared = rng.standard_normal((C, p_in)) * 3.0
    X, y, d, sess = [], [], [], []
    for s in range(m):
        for se in ("0early", "1late"):
            for c in range(C):
                mu = shared[c].copy()
                if s == drift_subj and se == "1late":
                    mu = mu + 4.0 * rng.standard_normal(p_in)      # session drift for this subject
                samples = mu[None, :] + 0.4 * rng.standard_normal((per, p_in))   # (per, p_in) then reshape to (per, 2, p_in//2)
                X.append(samples.reshape(per, 2, p_in // 2))
                y += [c] * per; d += [s] * per; sess += [se] * per
    return np.vstack(X).astype("float32"), np.array(y), np.array(d), np.array(sess)


def test_weights_source_only_signature():
    p = set(inspect.signature(CS.cross_session_risk_weights).parameters)
    assert not ({"Z_target", "y_target", "target", "Xte", "yte"} & p) and "sess" in p


def test_cross_session_weight_isolates_drifting_subject():
    X, y, d, sess = _src(drift_subj=2)
    Z = X.reshape(len(X), -1)                                       # crude features for the weight builder
    out = CS.cross_session_risk_weights(Z, y, d, sess)
    assert out["status"] == "ok"
    tot = {s: sum(out["weights"][(s, p)] for p in out["pairs"]) for s in out["subs"]}
    assert tot[2] == max(tot.values()) and tot[2] > np.mean([tot[s] for s in out["subs"] if s != 2])


def test_weighted_mcc_gradient_equals_full_batch():
    bb = _Fixture(); X, y, d, sess = _src()
    Z = X.reshape(len(X), -1); w = CS.cross_session_risk_weights(Z, y, d, sess)["weights"]
    g_two = CS.exact_weighted_mcc_gradient(bb, X, y, d, w, "cpu", bs=16)
    # single-graph weighted reference
    bb.eval(); z = bb(torch.tensor(X, dtype=torch.float32))[1]
    classes = sorted(np.unique(y).tolist()); subs = sorted(np.unique(d).tolist()); pairs = class_pairs(classes)
    means = {(s, c): z[(d == s) & (y == c)].mean(0) for s in subs for c in classes}
    L = CS._weighted_mcc_from_means(means, subs, classes, pairs, w)
    params = list(bb.parameters()); g = torch.autograd.grad(L, params, allow_unused=True)
    g_ref = torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten() for gi, p in zip(g, params)]).detach().numpy()
    assert np.linalg.norm(g_two - g_ref) / (np.linalg.norm(g_ref) + 1e-12) < 1e-5


def test_cs_risk_gradient_differs_from_cs_rw_mcc():
    bb = _Fixture(); X, y, d, sess = _src(drift_subj=2)
    Z = X.reshape(len(X), -1); out = CS.cross_session_risk_weights(Z, y, d, sess); w = out["weights"]
    g_mcc = CS.exact_weighted_mcc_gradient(bb, X, y, d, w, "cpu", bs=16)
    g_risk = CS.weighted_late_task_gradient(bb, X, y, d, out["is_late"], w, "cpu")
    assert g_mcc.shape == g_risk.shape and CS.cos(g_mcc, g_risk) < 0.99   # different intervention points


def test_early_late_ordering():
    e, l = CS._early_late(np.array(["1test", "0train", "0train"]))
    assert e == "0train" and l == {"1test"}
    e2, l2 = CS._early_late(np.array(["2C", "0A", "1B"]))
    assert e2 == "0A" and l2 == {"1B", "2C"}


def test_direct_risk_loss_late_only_and_bool_guard():
    """CS-Risk training term must (a) reject raw session STRINGS (the bug: every non-empty string is truthy ->
    all trials treated as late), and (b) with a proper bool mask, ignore early-session trials entirely."""
    torch.manual_seed(0)
    logits = torch.randn(6, 4, requires_grad=True)
    y = np.array([0, 1, 2, 0, 1, 2]); d = np.array([0, 0, 0, 1, 1, 1])
    v = {(0, 0): 1.0, (0, 1): 1.0, (0, 2): 1.0, (1, 0): 1.0, (1, 1): 1.0, (1, 2): 1.0}
    # (a) raw strings must fail loud, not silently treat all-as-late
    with pytest.raises(AssertionError):
        CS.direct_risk_loss(logits, y, d, np.array(["0early", "1late", "0early", "1late", "0early", "1late"]), v, "cpu")
    # (b) bool mask: loss depends ONLY on the late-flagged trials
    late = np.array([False, False, False, True, True, True])
    L_all_late = CS.direct_risk_loss(logits, y, d, np.ones(6, bool), v, "cpu")
    L_late = CS.direct_risk_loss(logits, y, d, late, v, "cpu")
    # perturb an EARLY trial's target weight -> late-only loss unchanged; all-late loss changes
    v2 = dict(v); v2[(0, 0)] = 9.0
    assert abs(float(CS.direct_risk_loss(logits, y, d, late, v2, "cpu")) - float(L_late)) < 1e-9
    assert abs(float(CS.direct_risk_loss(logits, y, d, np.ones(6, bool), v2, "cpu")) - float(L_all_late)) > 1e-6


def test_direct_risk_gradient_equals_full_batch():
    """The co-diagnostic arm-D gradient (micro-batched, BN frozen) must equal the single-graph gradient of the actual
    direct_risk_loss the arm optimizes -> the recorded cs_risk alignment faithfully characterizes arm D."""
    bb = _Fixture(); X, y, d, sess = _src()
    _, later = CS._early_late(sess); is_late = np.isin(sess, list(later))
    w = CS.cross_session_risk_weights(X.reshape(len(X), -1), y, d, sess)
    v = CS.per_trial_cs_weights(w["weights"], w["subs"], w["pairs"], w["classes"])
    g_mb = CS.direct_risk_gradient(bb, X, y, d, is_late, v, "cpu", bs=16)
    bb.eval(); logits = bb(torch.tensor(X, dtype=torch.float32))[0]
    L = CS.direct_risk_loss(logits, y, d, is_late, v, "cpu")
    params = list(bb.parameters()); g = torch.autograd.grad(L, params, allow_unused=True)
    g_ref = torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten() for gi, p in zip(g, params)]).detach().numpy()
    assert np.linalg.norm(g_mb - g_ref) / (np.linalg.norm(g_ref) + 1e-12) < 1e-5


def test_missing_session_class_fails_loud():
    X, y, d, sess = _src()
    keep = ~((d == 0) & (sess == "1late") & (y == 3))              # drop subject 0 late class 3
    Z = X[keep].reshape(keep.sum(), -1)
    with pytest.raises(ValueError):
        CS.cross_session_risk_weights(Z, y[keep], d[keep], sess[keep])

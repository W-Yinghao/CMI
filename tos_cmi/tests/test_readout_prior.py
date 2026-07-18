"""Pinned tests for Readout Prior Decomposition: fixed-tau objective gradient, parameter-level init-invariance
(H1==H1-W), source-only tau + gate, H2-beats-fair-ridge when the source prior helps, and aggregator routing."""
import inspect, sys
import numpy as np
import pytest
from tos_cmi.eval import readout_prior as RP
from tos_cmi.eval.readout_calibration import standardize, _std, fit_head


def _synth(seed=0, m=8, C=3, per=70, d=10, drift=0.7):
    rng = np.random.default_rng(seed); proto = rng.standard_normal((C, d)) * 0.9
    Zs, ys, ds, ss = [], [], [], []
    for s in range(m):
        sh = 0.5 * rng.standard_normal(d)
        for se in ("0e", "1l"):
            dr = drift * rng.standard_normal(d) if se == "1l" else 0
            for c in range(C):
                Zs.append(proto[c] + sh + dr + 0.9 * rng.standard_normal((per, d))); ys += [c] * per; ds += [s] * per; ss += [se] * per
    return np.vstack(Zs), np.array(ys), np.array(ds), np.array(ss), proto


def test_fixed_tau_gradient_correct():
    Zs, ys, ds, ss, _ = _synth(); mu, sd = standardize(Zs); Xs = _std(Zs, mu, sd); C = 3
    x = np.random.default_rng(0).standard_normal(C * Xs.shape[1] + C) * 0.3
    Wa, ba = np.zeros((C, Xs.shape[1])), np.zeros(C)
    f0, g = RP._ridge_map_obj(x, Xs[:60], ys[:60], C, Wa, ba, 1.0); eps = 1e-6
    for i in (0, 7, 25, C * Xs.shape[1], C * Xs.shape[1] + 1):
        xp = x.copy(); xp[i] += eps; fp, _ = RP._ridge_map_obj(xp, Xs[:60], ys[:60], C, Wa, ba, 1.0)
        assert abs((fp - f0) / eps - g[i]) / (abs(g[i]) + 1e-6) < 1e-4


def test_init_invariance_at_parameter_level():
    """H1 (zero init) and H1-W (source init) minimise the SAME convex objective -> identical parameters."""
    Zs, ys, ds, ss, _ = _synth(); mu, sd = standardize(Zs); Xs = _std(Zs, mu, sd); C = 3
    Ws, bs = fit_head(Xs, ys, C)
    W1, b1, a1 = RP.fit_ridge_map(Xs[:80], ys[:80], C, None, None, 10.0)
    W1w, b1w, a1w = RP.fit_ridge_map(Xs[:80], ys[:80], C, None, None, 10.0, init=np.concatenate([Ws.ravel(), bs]))
    assert a1["success"] and a1w["success"]
    assert np.linalg.norm(W1 - W1w) + np.linalg.norm(b1 - b1w) < 1e-3


def test_tau_and_gate_are_source_only():
    for fn in (RP.select_tau_budget_matched, RP.source_gate):
        p = set(inspect.signature(fn).parameters)
        assert not ({"Xq", "yq", "Zt", "y_target", "Xcal", "ycal", "target"} & p), f"{fn.__name__} sees target: {p}"


def test_source_prior_beats_fair_ridge_at_low_k():
    """When the source head transfers well (small drift), H2 (source-centered) beats H1 (fair zero-centered ridge) at
    k=1 because H1 shrinks to ~chance while H2 shrinks to the source head."""
    Zs, ys, ds, ss, proto = _synth(drift=0.4); C = 3; mu, sd = standardize(Zs); Xs = _std(Zs, mu, sd)
    Ws, bs = fit_head(Xs, ys, C)
    rng = np.random.default_rng(0)
    # target = protos + a small shift; cal (k=1/class) + query
    tsh = 0.4 * rng.standard_normal(Xs.shape[1] if False else 10)
    Xc = np.vstack([proto[c] + tsh + 0.9 * rng.standard_normal((1, 10)) for c in range(C)]); yc = np.arange(C)
    Xq = np.vstack([proto[c] + tsh + 0.9 * rng.standard_normal((40, 10)) for c in range(C)]); yq = np.repeat(np.arange(C), 40)
    Xcs, Xqs = _std(Xc, mu, sd), _std(Xq, mu, sd)
    from sklearn.metrics import balanced_accuracy_score
    W1, b1, _ = RP.fit_ridge_map(Xcs, yc, C, None, None, 10.0)          # H1 fair ridge, strong shrink at k=1
    Wm, bm, _ = RP.fit_ridge_map(Xcs, yc, C, Ws, bs, 10.0)              # H2 source-centered
    u1 = balanced_accuracy_score(yq, (Xqs @ W1.T + b1).argmax(1))
    u2 = balanced_accuracy_score(yq, (Xqs @ Wm.T + bm).argmax(1))
    assert u2 > u1                                                      # source prior beats chance-shrink at k=1


def test_aggregator_routing():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from scripts.aggregate_readout_prior_decomposition import _route
    def st(initbad=False, gnoharm=True, gpos=False, spec=False, pcf=False, hg=0.0, mfull=False, mfew=False, ku=None):
        return dict(k_center=None, k_util=ku, init_bad=initbad, harm_vs_frozen=False, gate_no_harm=gnoharm, gate_pos=gpos,
                    specific_any=spec, prior_center_full=pcf, headroom_gain=hg, map_full_pos=mfull, map_fewshot_pos=mfew)
    DEV = ["BNCI2014_001", "BNCI2015_001"]; EXT = ["Lee2019_MI", "BNCI2014_004"]; ALL = DEV + EXT
    # P-C: init not invariant
    assert _route({d: st(initbad=(d == "Lee2019_MI")) for d in ALL}, False)["verdict"].startswith("P-C")
    # P-A: clean prior center (beats fair ridge @Full WITH headroom) on a dev set AND external adaptation replicates
    pa = {"BNCI2014_001": st(pcf=True, hg=0.05, mfull=True), "BNCI2015_001": st(mfull=True),
          "Lee2019_MI": st(mfew=True, mfull=True), "BNCI2014_004": st()}
    assert _route(pa, False)["verdict"].startswith("P-A_SOURCE_HEAD_PRIOR")
    # HONEST PARTIAL (the actual result): dev adapt-vs-frozen real, NO prior_center_full anywhere, NO external adaptation
    partial = {"BNCI2014_001": st(mfull=True), "BNCI2015_001": st(mfull=True), "Lee2019_MI": st(), "BNCI2014_004": st()}
    assert _route(partial, False)["verdict"].startswith("P-A_PARTIAL_DEV_ONLY")
    # P-D: gate safe + positive, no dev adaptation
    pd = {d: st(gnoharm=True, gpos=True) for d in ALL}
    assert _route(pd, False)["verdict"].startswith("P-D")

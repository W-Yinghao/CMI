"""Pinned tests for the Information-Regime Ladder (Track B): firewall (query labels never enter SELECTION), balanced
draws, CE-based selection picks the CE-minimizing action, identity is always a no-harm candidate, and the aggregator
routing IL-A..E."""
import sys
import numpy as np
import pytest

from tos_cmi.eval import information_ladder as IL


def _synth(seed=0, m=4, C=4, per=40, D=10):
    """m source subjects x C classes; a target with a cal session and a query session. One class-contrast direction
    is subject-unstable (a deletable nuisance) so a subspace deletion can help."""
    rng = np.random.default_rng(seed)
    proto = rng.standard_normal((C, D)) * 2.0
    nuis = rng.standard_normal(D)                      # a nuisance axis
    Zs, ys, ds = [], [], []
    for s in range(m):
        shift = (s - m / 2) * 0.8 * nuis               # subject-specific nuisance offset
        for c in range(C):
            Zs.append(proto[c] + shift + 0.3 * rng.standard_normal((per, D))); ys += [c] * per; ds += [s] * per
    Zs = np.vstack(Zs); ys = np.array(ys); ds = np.array(ds)
    # target: cal + query sessions, same protos + its own nuisance shift
    tshift = 1.2 * nuis
    Zt, yt, st = [], [], []
    for sess in ("0cal", "1qry"):
        for c in range(C):
            Zt.append(proto[c] + tshift + 0.3 * rng.standard_normal((per, D))); yt += [c] * per; st += [sess] * per
    return Zs, ys, ds, np.vstack(Zt), np.array(yt), np.array(st)


def _prep(seed=0):
    from tos_cmi.eval import targetx_metric as TM
    Zs, ys, ds, Zt, yt, st = _synth(seed)
    W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W)
    cal = st == "0cal"; qry = st == "1qry"
    Xcal_w = TM.to_whitened(Zt[cal], W); ycal = yt[cal]
    Xq_w = TM.to_whitened(Zt[qry], W); yq = yt[qry]; sq = st[qry]
    d_white = Zs_w.mean(0) - Xcal_w.mean(0)
    B = TM.whitened_cond_basis(Zs_w, ys, ds, max_rank=IL.DICT_RANK)
    return Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white


def test_selection_is_invariant_to_query_labels():
    """FIREWALL: the SELECTED action of every regime depends only on source + cal; permuting the QUERY labels changes
    the utility lookup but NOT which action is selected."""
    Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white = _prep()
    recs_a, _ = IL.precompute_actions(Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white)
    rng = np.random.default_rng(1); yq_perm = rng.permutation(yq)
    recs_b, _ = IL.precompute_actions(Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq_perm, sq, d_white)
    classes = sorted(np.unique(ys).tolist())
    draws = [IL._balanced_draw(ycal, 2, np.random.default_rng(i)) for i in range(3)]
    for reg in IL.REGIMES:
        dd = draws if reg in IL.KSHOT else None
        _, sel_a = IL.select_and_utility(recs_a, reg, Xcal_w, ycal, classes, draws=dd)
        _, sel_b = IL.select_and_utility(recs_b, reg, Xcal_w, ycal, classes, draws=dd)
        assert sel_a == sel_b, f"{reg}: selection changed under query-label permutation (firewall breach)"


def test_balanced_draw_k_per_class():
    _, _, _, _, _, ycal, *_ = _prep()
    for k in (1, 2, 4):
        idx = IL._balanced_draw(ycal, k, np.random.default_rng(0))
        counts = np.bincount(ycal[idx])
        assert all(cc == k for cc in counts if cc > 0) and set(np.unique(ycal[idx])) == set(np.unique(ycal))


def test_ce_selection_prefers_lower_ce_action():
    """The CE selector must pick the action minimizing cal CE (not a random one)."""
    Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white = _prep()
    recs, _ = IL.precompute_actions(Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white)
    classes = sorted(np.unique(ys).tolist())
    ces = [IL._ce(r["head"], r["U"], Xcal_w, ycal, classes) for r in recs]
    _, sel = IL.select_and_utility(recs, "RF", Xcal_w, ycal, classes)
    j = [i for i, r in enumerate(recs) if r["S"] == sel][0]
    assert abs(ces[j] - min(ces)) < 1e-9


def test_identity_is_a_candidate_and_zero_gain():
    Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white = _prep()
    recs, _ = IL.precompute_actions(Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white)
    idr = [r for r in recs if r["S"] == []]
    assert len(idr) == 1 and abs(idr[0]["query_gain"]) < 1e-9 and idr[0]["g1"] == 0.0


def test_aggregator_routing():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from scripts.aggregate_information_ladder import _route
    def ci(lo, hi): return dict(lo=lo, hi=hi, mean=(lo + hi) / 2, signflip_p=0.01, n=9)
    def ds(dus, specs, headcase="D_neither"):
        d = {reg: {"dU": ci(*dus[reg]), "spec": ci(*specs[reg])} for reg in IL.REGIMES}
        d["_headcase"] = headcase; d["_rf_lcb"] = dus["RF"][0]; return d
    null = {r: (-0.01, 0.01) for r in IL.REGIMES}
    # IL-A: R2 dU and spec both LCB>0 on both datasets
    a = {**null, "R2": (0.01, 0.03), "R4": (0.01, 0.03), "RF": (0.01, 0.03)}
    aspec = {**null, "R2": (0.005, 0.02), "R4": (0.005, 0.02), "RF": (0.005, 0.02)}
    assert _route({"BNCI2014_001": ds(a, aspec), "BNCI2015_001": ds(a, aspec)})["verdict"].startswith("IL-A")
    # IL-B: R2 dU LCB>0 both but spec straddles 0 -> generic
    assert _route({"BNCI2014_001": ds(a, null), "BNCI2015_001": ds(a, null)})["verdict"].startswith("IL-B")
    # IL-D: RX dU>0, R0<=0 both
    dd = {**null, "RX": (0.01, 0.03)}
    assert _route({"BNCI2014_001": ds(dd, null), "BNCI2015_001": ds(dd, null)})["verdict"].startswith("IL-D")
    # IL-E: only RF>0
    ee = {**null, "RF": (0.01, 0.03)}
    assert _route({"BNCI2014_001": ds(ee, null), "BNCI2015_001": ds(ee, null)})["verdict"].startswith("IL-E")
    # IL-C: selection null but head-only positive
    assert _route({"BNCI2014_001": ds(null, null, "B_head_only"), "BNCI2015_001": ds(null, null, "B_head_only")})["verdict"].startswith("IL-C")
    # null
    assert _route({"BNCI2014_001": ds(null, null), "BNCI2015_001": ds(null, null)})["verdict"].startswith("IL_NULL")

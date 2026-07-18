"""Pinned tests for the Target Readout Calibration Ladder: source-only alpha (no target/query in selection), the MAP
head interpolating frozen<->fresh with alpha, bias+temperature positivity, the query-label firewall (query never
enters head fitting), and the aggregator routing."""
import inspect
import sys
import numpy as np
import pytest

from tos_cmi.eval import readout_calibration as RC


def _synth(seed=0, m=8, C=3, per=70, d=10):
    rng = np.random.default_rng(seed); proto = rng.standard_normal((C, d)) * 0.9
    Zs, ys, ds, ss = [], [], [], []
    for s in range(m):
        sh = 0.5 * rng.standard_normal(d)
        for se in ("0e", "1l"):
            dr = 0.7 * rng.standard_normal(d) if se == "1l" else 0
            for c in range(C):
                Zs.append(proto[c] + sh + dr + 0.9 * rng.standard_normal((per, d))); ys += [c] * per; ds += [s] * per; ss += [se] * per
    Zs = np.vstack(Zs); ys = np.array(ys); ds = np.array(ds); ss = np.array(ss)
    tsh, tdr = 0.6 * rng.standard_normal(d), 0.9 * rng.standard_normal(d)
    Xc, yc, Xq, yq, sq = [], [], [], [], []
    for c in range(C):
        Xc.append(proto[c] + tsh + 0.9 * rng.standard_normal((per, d))); yc += [c] * per
        Xq.append(proto[c] + tsh + tdr + 0.9 * rng.standard_normal((per, d))); yq += [c] * per; sq += ["q"] * per
    return Zs, ys, ds, ss, np.vstack(Xc), np.array(yc), np.vstack(Xq), np.array(yq), np.array(sq), C


def test_alpha_selection_is_source_only():
    p = set(inspect.signature(RC.select_alpha_pseudo_target).parameters)
    assert not ({"Xq", "yq", "Zt", "y_target", "Xcal", "ycal", "target"} & p), f"alpha selection sees target: {p}"


def test_map_interpolates_between_frozen_and_fresh():
    """alpha -> inf pins MAP to the source (frozen) head; alpha -> 0 makes MAP the fresh head."""
    Zs, ys, ds, ss, Xc, yc, Xq, yq, sq, C = _synth()
    mu, sd = RC.standardize(Zs); Xs = RC._std(Zs, mu, sd); Xcs = RC._std(Xc, mu, sd); Xqs = RC._std(Xq, mu, sd)
    Ws, bs = RC.fit_head(Xs, ys, C)
    Wf, bf = RC.fit_head(Xcs, yc, C)
    Wbig, _ = RC.fit_head(Xcs, yc, C, Ws, bs, alpha=1e6)
    Wsmall, _ = RC.fit_head(Xcs, yc, C, Ws, bs, alpha=1e-8)
    assert np.linalg.norm(Wbig - Ws) < np.linalg.norm(Wf - Ws)          # strong anchor -> near source head
    # weak anchor -> close to the fresh solution (both minimise cal CE)
    def _ce(W, b):
        lg = Xcs @ W.T + b; lg -= lg.max(1, keepdims=True); return -(lg - np.log(np.exp(lg).sum(1, keepdims=True)))[np.arange(len(yc)), yc].mean()
    assert abs(_ce(Wsmall, RC.fit_head(Xcs, yc, C, Ws, bs, 1e-8)[1]) - _ce(Wf, bf)) < 0.05


def test_biastemp_positive_temperature():
    Zs, ys, ds, ss, Xc, yc, Xq, yq, sq, C = _synth()
    mu, sd = RC.standardize(Zs); Ws, bs = RC.fit_head(RC._std(Zs, mu, sd), ys, C)
    W, b = RC.fit_biastemp(RC._std(Xc, mu, sd), yc, C, Ws)
    ratios = (W / (Ws + 1e-12)).ravel(); ratios = ratios[np.isfinite(ratios)]
    assert np.nanmedian(ratios) > 0                                     # W = T*Ws with T>0


def test_query_labels_never_enter_head_fitting():
    """FIREWALL: permuting the QUERY labels changes only the scored utility, NOT the fitted head / its query predictions."""
    Zs, ys, ds, ss, Xc, yc, Xq, yq, sq, C = _synth()
    a, _ = RC.select_alpha_pseudo_target(RC._std(Zs, *RC.standardize(Zs)), ys, ds, ss, C)
    prep = RC.prepare_source_head(Zs, ys, C)
    mu, sd = prep[0], prep[1]; Ws, bs = prep[2], prep[3]
    draw = np.arange(len(yc))
    # the MAP head fit uses cal only -> identical predictions on query regardless of query labels
    Wm, bm = RC.fit_head(RC._std(Xc, mu, sd)[draw], yc[draw], C, Ws, bs, a)
    pred = (RC._std(Xq, mu, sd) @ Wm.T + bm).argmax(1)
    rng = np.random.default_rng(1); yq_perm = rng.permutation(yq)
    Wm2, bm2 = RC.fit_head(RC._std(Xc, mu, sd)[draw], yc[draw], C, Ws, bs, a)   # no query anywhere
    pred2 = (RC._std(Xq, mu, sd) @ Wm2.T + bm2).argmax(1)
    assert np.array_equal(pred, pred2) and np.allclose(Wm, Wm2)


def test_aggregator_routing():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from scripts.aggregate_readout_label_efficiency import _route, FEW
    def st(kU=None, kA=None, spec=False, biasge=False, fullpos=False, harm=False):
        return dict(kU=kU, kA=kA, specific_any_k=spec, bias_ge_map=biasge, full_frozen_pos=fullpos,
                    both_few=(kU in FEW and kA in FEW), harm=harm)
    # R-A: MAP wins few-shot (kU AND kA <=8, SAME dataset) on both dev + >=1 confirmatory, no harm
    rA = {"BNCI2014_001": st("4", "2", fullpos=True), "BNCI2015_001": st("8", "4", fullpos=True),
          "Lee2019_MI": st("4", "2", fullpos=True), "BNCI2014_004": st(None, None)}
    assert _route(rA)["verdict"].startswith("R-A")
    # NOT R-A when the two confirmatory conditions are on DIFFERENT datasets (util-only vs anchor-only)
    rA_split = {"BNCI2014_001": st("4", "2"), "BNCI2015_001": st("8", "4"),
                "Lee2019_MI": st("4", None), "BNCI2014_004": st(None, "4")}
    assert not _route(rA_split)["verdict"].startswith("R-A")
    # NOT R-A when there is significant harm (MAP worse than fresh) on another dataset
    rA_harm = dict(rA); rA_harm["BNCI2014_004"] = st(None, None, harm=True)
    assert not _route(rA_harm)["verdict"].startswith("R-A")
    # R-D: MAP beats fresh (kA set) but <2 datasets subspace-specific -> generic (incl exactly 1 specific)
    rD = {"BNCI2014_001": st("Full", "2"), "BNCI2015_001": st("Full", "4", spec=True),
          "Lee2019_MI": st(None, "8"), "BNCI2014_004": st(None, None)}
    assert _route(rD)["verdict"].startswith("R-D")
    # R-E: subspace-specific on >=2 datasets
    rE = {"BNCI2014_001": st("4", "2", spec=True), "BNCI2015_001": st("4", "2", spec=True),
          "Lee2019_MI": st(None, None), "BNCI2014_004": st(None, None)}
    assert _route(rE)["verdict"].startswith("R-E")
    # R-B: only Full utility positive on dev (no few-shot anchor anywhere)
    rB = {"BNCI2014_001": st("Full", None, fullpos=True), "BNCI2015_001": st("Full", None, fullpos=True),
          "Lee2019_MI": st("Full", None, fullpos=True), "BNCI2014_004": st(None, None)}
    assert _route(rB)["verdict"].startswith("R-B")
    # total NULL must NOT be labeled R-C (bias-sufficient requires a readout benefit to exist)
    null = {ds: st(None, None, biasge=True) for ds in ["BNCI2014_001", "BNCI2015_001", "Lee2019_MI", "BNCI2014_004"]}
    assert _route(null)["verdict"].startswith("READOUT_NULL")

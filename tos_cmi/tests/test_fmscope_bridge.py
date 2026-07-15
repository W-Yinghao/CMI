"""CMI-Trace P0.5 tests -- FMScope protocol bridge (2x2: {ORACLE_GLOBAL_GEOMETRY, strict source-only}
x {subject-LEACE, same-rank random}), on synthetic gaussian dumps.

Three guarantees:
  1. FIREWALL: the strict source-only path cannot see target arrays before final scoring.
  2. The ORACLE (global) mode is labelled non-deployable / non-DG in EVERY emitted artifact.
  3. End-to-end 2x2 with both LEACE variants: all cells finite, structure present.
"""
import inspect
import json
import numpy as np
import pytest

from tos_cmi.eeg import fmscope_protocol_bridge as fb
from tos_cmi.eeg.erasure_baselines import leace_eraser
from tos_cmi.eeg.erasure_target_deploy import _task_metrics

VERDICTS = {"representation_geometry_difference", "transferable_subject_axis_benefit",
            "dimensionality_or_conditioning_effect",
            "target_cohort_conditioned_geometry_explains_discrepancy", "no_clear_pattern"}


def _dumps(n_folds=5, seeds=(0,), **kw):
    return [fb.make_synthetic_dump(target_subject=t, seed=s, **kw) for s in seeds for t in range(n_folds)]


# ------------------------------------------------------------------ FIREWALL
def test_firewall_source_only_fit_has_no_target_argument():
    """The strict source-only eraser fit signature must contain NO target/Zt parameter -> the eraser is
    provably a function of source rows only."""
    params = list(inspect.signature(fb.source_only_fit).parameters)
    assert "Zt" not in params and "y_target" not in params
    assert not any("target" in p.lower() for p in params)
    assert params[:3] == ["Zs", "ys", "subj_s"]


def test_firewall_source_fit_and_source_metrics_invariant_to_target_content():
    """Poison the target with a totally different distribution: the source-only eraser AND the source-side
    metrics (head is trained on transformed SOURCE only) must be BIT-IDENTICAL -- only target metrics move."""
    d = fb.make_synthetic_dump(target_subject=0, seed=3)
    Zs = np.asarray(d["Z_source"], np.float64); ys = np.asarray(d["y_source"], int)
    Zt = np.asarray(d["Z_target"], np.float64); yt = np.asarray(d["y_target"], int)
    subj_s = np.asarray(d["subject_source"]); n_cls = int(d["n_cls"])
    apply_so, k, aux = fb.source_only_fit(Zs, ys, subj_s, whiten="empirical")
    # eraser depends only on source -> refit is identical
    apply_so2, k2, _ = fb.source_only_fit(Zs, ys, subj_s, whiten="empirical")
    assert k == k2 and np.allclose(apply_so(Zs), apply_so2(Zs))
    Zt_poison = Zt * 1e6 + 999.0                                   # wildly different target
    rng = 7
    tb1, tn1, sb1, sn1 = _task_metrics(apply_so(Zs), ys, apply_so(Zt), yt, n_cls, np.random.default_rng(rng))
    tb2, tn2, sb2, sn2 = _task_metrics(apply_so(Zs), ys, apply_so(Zt_poison), yt, n_cls, np.random.default_rng(rng))
    # source-side metrics (a function of the source-trained head) are untouched by target content...
    assert sb1 == sb2 and sn1 == sn2
    # ...while the target-side metrics DO respond to the poisoned target (sanity that the poison mattered)
    assert (tb1 != tb2) or (tn1 != tn2)


def test_firewall_row_accounting_source_only_never_includes_target_rows():
    """Artifact-level firewall: source-only cells (B,D,full) fit on exactly len(source) rows; oracle cells
    (A,C) fit on len(source)+len(target)."""
    d = fb.make_synthetic_dump(target_subject=0, seed=0)
    n_src = len(d["Z_source"]); n_all = n_src + len(d["Z_target"])
    rows = fb.bridge_one_dump(d, n_random=3)
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    for c in ("B", "D", "full"):
        assert all(r["n_fit_rows"] == n_src for r in by_cell[c]), c
    for c in ("A", "C"):
        assert all(r["n_fit_rows"] == n_all for r in by_cell[c]), c


# ------------------------------------------------------------------ ORACLE non-deployable
def test_oracle_mode_labelled_nondeployable_in_every_row():
    d = fb.make_synthetic_dump(target_subject=0, seed=0)
    rows = fb.bridge_one_dump(d, n_random=3)
    oracle_rows = [r for r in rows if r["cell"] in ("A", "C")]
    assert oracle_rows
    for r in oracle_rows:
        assert r["fit_regime"] == fb.ORACLE == "ORACLE_GLOBAL_GEOMETRY"
        assert r["oracle_tag"] == "ORACLE_GLOBAL_GEOMETRY"
        assert r["oracle_global_geometry"] is True
        assert r["is_dg"] is False
        assert r["deployable"] is False
    # source-only + baseline rows are DG-eligible and NOT tagged oracle
    for r in [r for r in rows if r["cell"] in ("B", "D", "full")]:
        assert r["oracle_global_geometry"] is False
        assert r["oracle_tag"] == ""
        assert r["is_dg"] is True


def test_oracle_mode_labelled_nondeployable_in_summary_and_never_called_dg(tmp_path):
    dumps = _dumps(n_folds=4, seeds=(0,))
    summary = fb.run(dumps, str(tmp_path), "SYNTHETIC_SMOKE", n_random=3)
    assert summary["cells"]["A"]["deployable"] is False and summary["cells"]["A"]["is_dg"] is False
    assert summary["cells"]["C"]["deployable"] is False and summary["cells"]["C"]["is_dg"] is False
    assert summary["cells"]["A"]["oracle_tag"] == "ORACLE_GLOBAL_GEOMETRY"
    for bb, byv in summary["per_backbone"].items():
        for v in fb.VARIANTS:
            s = byv[v]
            assert s["oracle_mode_tag"] == "ORACLE_GLOBAL_GEOMETRY"
            assert s["verdict"] in VERDICTS
    # the emitted summary JSON must never label the global mode as DG
    blob = json.loads(open("%s/fmscope_bridge_summary.json" % tmp_path).read())
    txt = json.dumps(blob).lower()
    assert "oracle_global_geometry" in txt
    # every row artifact carries the oracle flag; oracle rows are non-DG
    for line in open("%s/fmscope_bridge_rows.jsonl" % tmp_path):
        r = json.loads(line)
        assert "oracle_global_geometry" in r
        if r["cell"] in ("A", "C"):
            assert r["is_dg"] is False and r["deployable"] is False


# ------------------------------------------------------------------ end-to-end 2x2, both LEACE variants
def test_end_to_end_2x2_both_variants_finite_and_structured(tmp_path):
    dumps = _dumps(n_folds=5, seeds=(0, 1))
    summary = fb.run(dumps, str(tmp_path), "SYNTHETIC_SMOKE", n_random=20)
    rows = [json.loads(l) for l in open("%s/fmscope_bridge_rows.jsonl" % tmp_path)]
    # all four cells + full present, for both LEACE variants
    variants_seen = set(r.get("leace_variant") for r in rows if r["cell"] != "full")
    assert variants_seen == set(fb.VARIANTS)
    for v in fb.VARIANTS:
        cells = set(r["cell"] for r in rows if r.get("leace_variant") == v)
        assert {"A", "B", "C", "D"} <= cells
    assert any(r["cell"] == "full" for r in rows)
    # every metric finite
    for r in rows:
        for k in ("tgt_bacc", "tgt_nll", "src_bacc", "src_nll"):
            assert np.isfinite(r[k]), (r["cell"], k)
    # 2x2 paired structure present + verdict computed
    for bb, byv in summary["per_backbone"].items():
        for v in fb.VARIANTS:
            paired = byv[v]["paired"]
            for name in ("A_vs_full", "B_vs_full", "A_vs_C", "B_vs_D"):
                assert name in paired and paired[name] is not None
                assert set(("delta_bacc", "lo", "hi", "positive", "deploy_ci_state")) <= set(paired[name])
            assert byv[v]["verdict"] in VERDICTS
        assert "lw_vs_empirical" in byv


def test_leace_rank_positive_and_removes_subject_on_source():
    """LEACE must remove >=1 rank and actually reduce SOURCE subject decode vs no-erasure (its defining
    property), confirming the subject-specific column is meaningful before the random control."""
    d = fb.make_synthetic_dump(target_subject=0, seed=0, subj_shift=2.0)
    rows = fb.bridge_one_dump(d, n_random=5)
    B = [r for r in rows if r["cell"] == "B" and r["leace_variant"] == "empirical"][0]
    full = [r for r in rows if r["cell"] == "full"][0]
    assert B["rank_removed"] >= 1
    assert B["subj_dec_after"] <= full["subj_dec_after"] + 1e-9


# ------------------------------------------------------------------ LEACE reimplementation parity
def test_empirical_leace_matches_repository_leace_eraser():
    """The empirical variant of `_leace_fit` must be numerically identical to the repository `leace_eraser`
    (rules out an implementation-only explanation of any discrepancy)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 10))
    subj = rng.integers(0, 4, size=120)
    oh = np.eye(4)[subj]
    apply_mine, k, _ = fb._leace_fit(X, oh, whiten="empirical")
    apply_repo = leace_eraser(X, oh)
    assert k >= 1
    assert np.allclose(apply_mine(X), apply_repo(X), atol=1e-8)


# ------------------------------------------------------------------ verdict contract logic
def test_verdict_contract_branches():
    v = fb.bridge_verdict
    # global positive, source null -> cohort-conditioned geometry
    assert v(global_pos=True, source_pos=False, global_rand_pos=False,
             global_above_random=True, source_above_random=False) == \
        "target_cohort_conditioned_geometry_explains_discrepancy"
    # global positive but random also positive and LEACE not above random -> conditioning/dimensionality
    assert v(global_pos=True, source_pos=False, global_rand_pos=True,
             global_above_random=False, source_above_random=False) == \
        "dimensionality_or_conditioning_effect"
    # global + source positive, both above random -> transferable subject axis
    assert v(global_pos=True, source_pos=True, global_rand_pos=False,
             global_above_random=True, source_above_random=True) == \
        "transferable_subject_axis_benefit"
    # frozen-FM positive but task-trained null -> representation geometry difference (rule 1, takes precedence)
    assert v(global_pos=True, source_pos=True, global_rand_pos=False,
             global_above_random=True, source_above_random=True,
             frozen_fm_pos=True, task_trained_null=True) == "representation_geometry_difference"
    # nothing positive -> no clear pattern
    assert v(False, False, False, False, False) == "no_clear_pattern"

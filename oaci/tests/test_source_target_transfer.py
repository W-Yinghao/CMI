"""C14b source->target instability diagnostics: anti-transfer flag, STI matches manual counts, correlation
robustness (constant/missing series), harm localization grouping."""
from __future__ import annotations

from oaci.falsification.transfer import (harm_localization, instability_metrics, pearson, transfer_correlations)


def _cell(t=1, te=0.1, lv=0, *, ssg, esg, dnll, dbacc=0.0, blow=False, fb=False):
    return {"target": t, "temp": te, "level": lv, "src_source_guard_nll": ssg, "erm_source_guard_nll": esg,
            "d_nll_vs_erm": dnll, "d_bacc_vs_erm": dbacc, "src_fallback_erm": fb, "target_nll_blowup": blow}


def test_antitransfer_flag_detects_source_improve_target_harm():
    cells = [_cell(ssg=1.0, esg=1.2, dnll=+0.5),        # source improved, target worse -> anti
             _cell(ssg=1.3, esg=1.2, dnll=+0.1)]        # source NOT improved -> not anti
    im = instability_metrics(cells)
    assert im["n_source_improved"] == 1 and im["n_anti_transfer"] == 1
    assert im["ATI_NLL"] == 0.5 and im["source_target_instability_score"] == 1.0   # 1 anti/2 active; 1 anti/1 improved


def test_source_target_instability_score_matches_manual_counts():
    cells = [_cell(ssg=1.0, esg=1.2, dnll=+0.5),   # improved + anti
             _cell(ssg=1.1, esg=1.2, dnll=+0.2),   # improved + anti
             _cell(ssg=1.05, esg=1.2, dnll=-0.1)]  # improved + transferred (target better)
    im = instability_metrics(cells)
    assert im["n_source_improved"] == 3 and im["n_anti_transfer"] == 2
    assert abs(im["source_target_instability_score"] - 2 / 3) < 1e-9      # STI = anti / source-improved
    assert abs(im["ATI_NLL"] - 2 / 3) < 1e-9                              # ATI = anti / active


def test_transfer_correlation_handles_constant_or_missing_series():
    assert pearson([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])["r"] is None        # constant x -> None (not a crash)
    assert pearson([1.0, None, 2.0], [None, 3.0, 4.0])["n"] == 1         # missing pairs dropped
    fb = [_cell(ssg=1.2, esg=1.2, dnll=0.0, fb=True)]                    # only a fallback cell -> no active
    im = instability_metrics(fb)
    assert im["n_active"] == 0 and im["ATI_NLL"] is None and im["source_target_instability_score"] is None


def test_harm_localization_groups_by_target_level_class():
    cells = [_cell(1, 0.1, 0, ssg=1.0, esg=1.2, dnll=+0.5, dbacc=-0.05, blow=True),
             _cell(1, 0.1, 1, ssg=1.2, esg=1.2, dnll=0.0, dbacc=0.0, fb=True),
             _cell(3, 0.3, 0, ssg=1.1, esg=1.2, dnll=+0.2, dbacc=-0.01, blow=False)]
    h = harm_localization(cells)
    assert set(h["by_target"]) == {"1", "3"} and set(h["by_level"]) == {"0", "1"} and set(h["by_temperature"]) == {"0.1", "0.3"}
    assert h["by_target"]["1"]["blowup"] == 1 and h["by_target"]["1"]["fallback"] == 1
    assert h["by_level"]["0"]["nll_harmed"] == 2 and len(h["per_cell"]) == 3


def test_transfer_correlations_source_nll_to_target_nll():
    # source improvement (negative Δsrc nll) paired with target harm (positive Δtgt nll) -> negative corr
    cells = [_cell(ssg=1.0, esg=1.3, dnll=+0.9), _cell(ssg=1.1, esg=1.3, dnll=+0.5),
             _cell(ssg=1.2, esg=1.3, dnll=+0.2)]
    tc = transfer_correlations(cells, {"audit_to_target_transfer": {}})
    r = tc["source_nll_to_target_nll"]["pearson"]["r"]
    assert r is not None and r < 0                                       # anti-transfer sign


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} source-target-transfer tests")


if __name__ == "__main__":
    _run_all()

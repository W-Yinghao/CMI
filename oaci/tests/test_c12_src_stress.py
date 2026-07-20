"""C12 SRC stress-replication aggregator: the gate verdict (continue / pivot / inconclusive), the blowup
flag, fallback + non-transfer detection, and canonical serialization. Synthetic per-config pilot bodies."""
from __future__ import annotations

from oaci.confirmatory.c12_src_stress import _cells, render_md, verdict
from oaci.artifacts.canonical_json import canonical_json_bytes


def _cfg(target, temp, *, src_tb, src_tn, src_sg_nll=1.15, fallback=False, feasible=True, guard_pass=5,
         erm_tb=0.50, erm_tn=1.20, erm_sg_nll=1.20, oaci_tb=0.46):
    def lvl():
        return {"ERM": {"target_worst_bacc": erm_tb, "target_worst_nll": erm_tn, "target_worst_ece": 0.1,
                        "source_guard_worst_nll": erm_sg_nll},
                "OACI": {"target_worst_bacc": oaci_tb, "target_worst_nll": erm_tn + 0.1},
                "SRC": {"target_worst_bacc": (erm_tb if fallback else src_tb),
                        "target_worst_nll": (erm_tn if fallback else src_tn), "target_worst_ece": 0.1,
                        "source_guard_worst_nll": (erm_sg_nll if fallback else src_sg_nll),
                        "fallback_erm": fallback, "risk_feasible": feasible, "n_guard_pass": (0 if fallback else guard_pass),
                        "K2_delta_target_worst_bacc": (0.0 if fallback else src_tb - erm_tb),
                        "K2_delta_target_worst_nll": (0.0 if fallback else src_tn - erm_tn)}}
    return {"target": target, "temp": temp, "body": {"target": target, "smooth_temperature": temp,
                                                     "levels": {"0": lvl(), "1": lvl()}}}


def test_blowup_triggers_pivot():
    # SRC target NLL 2.4 > uniform 1.386 -> blowup -> pivot (mirrors the real C11c target-001 result)
    cfgs = [_cfg(1, 0.1, src_tb=0.46, src_tn=2.40), _cfg(1, 0.3, src_tb=0.47, src_tn=1.90)]
    vd = verdict(_cells(cfgs))
    assert vd["verdict"] == "stop_SRC_pivot_measurement_only" and vd["n_target_nll_blowup"] >= 1
    assert any("blowup" in r for r in vd["pivot_reasons"])


def test_majority_fallback_triggers_pivot():
    cfgs = [_cfg(3, 0.1, src_tb=0.5, src_tn=1.2, fallback=True), _cfg(3, 0.3, src_tb=0.5, src_tn=1.2, fallback=True)]
    vd = verdict(_cells(cfgs))
    assert vd["verdict"] == "stop_SRC_pivot_measurement_only"
    assert any("fell back" in r for r in vd["pivot_reasons"])


def test_source_improved_not_transferred_triggers_pivot():
    # SRC improves source_guard NLL (1.10 < ERM 1.20) but target NLL worsens (1.30 > ERM 1.20), no blowup
    cfgs = [_cfg(5, 0.1, src_tb=0.50, src_tn=1.30, src_sg_nll=1.10),
            _cfg(5, 0.3, src_tb=0.50, src_tn=1.28, src_sg_nll=1.12)]
    vd = verdict(_cells(cfgs))
    assert vd["n_source_improved_not_transferred"] >= 1 and vd["verdict"] == "stop_SRC_pivot_measurement_only"


def test_clean_improvement_continues():
    # every active cell: bAcc up, NLL down, no blowup, feasible, >=2 gains -> continue
    cfgs = [_cfg(1, 0.1, src_tb=0.56, src_tn=1.05, src_sg_nll=1.10),
            _cfg(3, 0.1, src_tb=0.55, src_tn=1.08, src_sg_nll=1.12)]
    vd = verdict(_cells(cfgs))
    assert vd["verdict"] == "continue_SRC" and vd["n_target_nll_blowup"] == 0


def test_render_and_canonical_serializable():
    cfgs = [_cfg(1, 0.1, src_tb=0.46, src_tn=2.40), _cfg(3, 0.3, src_tb=0.55, src_tn=1.08)]
    rows = _cells(cfgs); vd = verdict(rows)
    md = render_md(rows, vd, cfgs)
    for s in ("## Table 1", "## Table 2", "## Table 3", "## Table 4", "VERDICT"):
        assert s in md
    assert canonical_json_bytes({"cells": rows, "verdict": vd})


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c12 stress tests")


if __name__ == "__main__":
    _run_all()

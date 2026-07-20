"""C16-B harm decomposition: logit-level metrics, calibration-vs-discrimination classification, class-boundary
rotation, SRC memorization index, canonical serialization. Synthetic logits + metric dicts (no real npz)."""
from __future__ import annotations

import numpy as np

from oaci.artifacts.canonical_json import canonical_json_bytes
from oaci.mechanism.harm_decomposition import _metrics, decompose


def test_metrics_on_perfect_vs_uniform_logits():
    y = np.array([0, 1, 2, 3] * 10)
    perfect = np.full((40, 4), -5.0); perfect[np.arange(40), y] = 5.0
    m = _metrics(perfect, y)
    assert m["bacc"] > 0.99 and m["nll"] < 0.05 and m["entropy"] < 0.1 and m["margin"] > 0.9
    uni = np.zeros((40, 4))
    u = _metrics(uni, y)
    assert abs(u["bacc"] - 0.25) < 1e-9 and abs(u["nll"] - np.log(4)) < 1e-6 and u["margin"] < 1e-6


def _m(bacc, nll, ece, *, conf_wrong=0.5, entropy=1.0, margin=0.3, logit_norm=3.0, mean_conf=0.6, recalls=None):
    r = recalls or {0: bacc, 1: bacc, 2: bacc, 3: bacc}
    return {"bacc": bacc, "nll": nll, "ece": ece, "entropy": entropy, "mean_conf": mean_conf,
            "conf_on_wrong": conf_wrong, "margin": margin, "logit_norm": logit_norm,
            "per_class_recall": r, "per_class_nll": {c: nll for c in range(4)},
            "confusion": np.eye(4) * bacc + (1 - bacc) / 3 * (1 - np.eye(4))}


def _rows(erm, oaci):
    return [{"seed": s, "target": t, "level": L, "ERM": erm, "OACI": oaci}
            for s in (0, 1, 2) for t in range(1, 10) for L in (0, 1)]


def test_calibration_harm_classification():
    # accuracy flat, NLL up, ECE up, overconfident on wrong -> calibration_harm
    d = decompose(_rows(_m(0.5, 1.2, 0.10, conf_wrong=0.5), _m(0.50, 1.45, 0.20, conf_wrong=0.65)))
    assert d["harm_type_tally"].get("calibration_harm", 0) == 54
    assert d["selected_checkpoint_verdict"] == "selected_oaci_calibration_harm"


def test_discrimination_harm_classification():
    d = decompose(_rows(_m(0.55, 1.2, 0.10), _m(0.45, 1.2, 0.10)))          # bAcc down, NLL flat
    assert d["harm_type_tally"].get("discrimination_harm", 0) == 54
    assert d["selected_checkpoint_verdict"] == "selected_oaci_discrimination_harm"


def test_calibration_improved_accuracy_flat():
    # softer, better-calibrated, accuracy flat -> the real OACI SELECTED verdict
    d = decompose(_rows(_m(0.50, 1.30, 0.12), _m(0.50, 1.15, 0.08, entropy=1.2, mean_conf=0.5)))
    assert d["selected_checkpoint_verdict"] == "selected_oaci_calibration_improved_accuracy_flat"
    assert d["aggregate_deltas"]["d_nll"] < 0


def test_class_boundary_rotation_detected():
    erm = _m(0.5, 1.2, 0.1, recalls={0: 0.4, 1: 0.6, 2: 0.4, 3: 0.6})
    oaci = _m(0.5, 1.2, 0.1, recalls={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5})       # 0,2 up; 1,3 down
    d = decompose(_rows(erm, oaci))
    assert d["class_boundary_rotation"] is True


def test_src_memorization_index_from_c12():
    c12 = {"cells": [{"target": 1, "temp": 0.1, "level": 0, "src_fallback_erm": False,
                      "src_source_guard_nll": 0.1, "erm_source_guard_nll": 1.2, "d_nll_vs_erm": +0.8},
                     {"target": 3, "temp": 0.1, "level": 1, "src_fallback_erm": True,   # fallback -> excluded
                      "src_source_guard_nll": 1.2, "erm_source_guard_nll": 1.2, "d_nll_vs_erm": 0.0}]}
    d = decompose(_rows(_m(0.5, 1.2, 0.1), _m(0.5, 1.2, 0.1)), c12=c12)
    sm = d["src_source_memorization"]
    assert sm["n_flagged"] == 1 and len(sm["per_cell"]) == 1          # only the non-fallback cell
    assert abs(sm["per_cell"][0]["memorization_index"] - (1.1 - (-0.8))) < 1e-9


def test_harm_decomposition_canonical_serializable():
    d = decompose(_rows(_m(0.5, 1.2, 0.1), _m(0.5, 1.15, 0.08)))
    assert canonical_json_bytes(d) and b'"per_subject"' in canonical_json_bytes(d)


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} harm-decomposition tests")


if __name__ == "__main__":
    _run_all()

"""C22 Estimand Transport Mechanism Audit. Frozen C19 config locked; epoch/order confound is the gating
override (reported before any rescue); normalization is diagnostic-only (not deployment); finite filtering;
no selector; deterministic taxonomy; within-target-exceeds-pooled decomposition. Synthetic score rows only."""
from __future__ import annotations

import numpy as np

from oaci.estimand_transport import (epoch_confound, estimand_decomposition, normalization_diagnostics,
                                     report, schema, score_loader, taxonomy)


def _rows(mode="in_regime", regimes=("S0_full_support", "S2_rare_cells", "S3_nonestimable_cells"),
          n_targets=6, n_per=10, offset=0.8, signal=1.0, epoch_signal=0.0, seed=0):
    rng = np.random.RandomState(seed); rows = []
    for regime in regimes:
        for t in range(1, n_targets + 1):
            toff = (t - n_targets / 2) * offset            # per-target score OFFSET (breaks pooling)
            for k in range(n_per):
                good = k % 2 == 0
                ep = 10 + k * 5
                score = toff + (signal if good else -signal) * 0.5 + rng.randn() * 0.3 + epoch_signal * ep
                r = {"mode": mode, "regime": regime, "seed": 0, "target": t, "level": 0,
                     "model_hash": f"{regime}{t}{k}", "score": score, "label": 1 if good else 0,
                     "epoch": ep, "order": k, "R_src": 0.8, "train_surrogate": 1.1}
                for f in schema.ROBUST_CORE:
                    r["feat__" + f] = toff + (0.3 if good else -0.3) + rng.randn() * 0.1
                rows.append(r)
    return rows


def test_frozen_config_locked():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_within_target_exceeds_pooled_when_offset_present():
    dec = estimand_decomposition.decompose(_rows())
    s = estimand_decomposition.summary(dec)
    # per-target offset destroys pooled AUC while within-target ranking survives
    assert s["within_exceeds_pooled_everywhere"] and s["mean_within_target_auc"] > s["mean_pooled_auc"]


def test_epoch_confound_flags_epoch_driven_signal():
    # score = epoch only (good/bad aligned with epoch) -> confounded
    rows = _rows(offset=0.0, signal=0.0, epoch_signal=0.05)
    # make label track epoch: relabel good = late epoch
    for r in rows:
        r["label"] = 1 if r["epoch"] > 35 else 0
        r["score"] = r["epoch"] + np.random.RandomState(hash(r["model_hash"]) % 99).randn() * 2
    ep = epoch_confound.epoch_confound(rows)
    assert ep["epoch_confounded"] is True


def test_epoch_reported_before_rescue_gates_taxonomy_to_T2():
    ep = {"epoch_confounded": True}
    t = taxonomy.transport_taxonomy(ep, {"mean_within_target_auc": 0.65}, {"per_mode": {}}, {}, {"offset_dominated_fraction": 0.0})
    assert t["primary_case"] == schema.T2       # epoch confound OVERRIDES everything


def test_taxonomy_T1_rank_signal_when_norm_recovers():
    ep = {"epoch_confounded": False}
    dsum = {"mean_within_target_auc": 0.64}
    norm = {"per_mode": {"cross_regime": {"target_normalization_recovers": True}}}
    t = taxonomy.transport_taxonomy(ep, dsum, norm, {}, {"offset_dominated_fraction": 0.6})
    assert t["primary_case"] == schema.T1 and schema.T4 in t["secondary"]


def test_taxonomy_T3_when_norm_does_not_recover():
    ep = {"epoch_confounded": False}
    t = taxonomy.transport_taxonomy(ep, {"mean_within_target_auc": 0.64},
                                    {"per_mode": {"cross_regime": {"target_normalization_recovers": False}}},
                                    {}, {"offset_dominated_fraction": 0.1})
    assert t["primary_case"] == schema.T3


def test_normalization_is_diagnostic_not_deployment():
    assert schema.NORMALIZATION_IS_DIAGNOSTIC
    nd = normalization_diagnostics.normalization_diagnostics(_rows())
    assert nd["is_diagnostic"] and "non-deployable" in nd["note"].lower()


def test_finite_filter_drops_none_nan_inf():
    assert score_loader._finite(0.5) and not score_loader._finite(None)
    assert not score_loader._finite(float("nan")) and not score_loader._finite(float("inf")) and not score_loader._finite(float("-inf"))


def test_report_forbids_selector_and_deployable_normalization():
    for bad in ("we built a selector", "deployable selector", "deployable normalization", "oaci is rescued",
                "external validation succeeded"):
        try:
            report._guard_forbidden("# C22\n\n" + bad + ".\n")
        except ValueError:
            continue
        raise AssertionError(f"forbidden claim not caught: {bad}")
    report._guard_forbidden("# C22\n\ntarget-wise normalization is a diagnostic mechanism test, not deployable.\n")


def test_normalization_recovers_pooled_under_pure_offset():
    # pure per-target offset + clean within signal -> target normalization should recover pooled AUC
    nd = normalization_diagnostics.normalization_diagnostics(_rows(offset=1.5, signal=1.2))
    inr = nd["per_mode"]["in_regime"]
    assert inr["target_normalization_recovers"] is True


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c22-estimand-transport tests")


if __name__ == "__main__":
    _run_all()

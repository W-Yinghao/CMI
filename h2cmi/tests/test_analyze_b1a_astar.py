"""A* analyzer: max-null/other-seed calibration must make N1 adapt to a clear shift, abstain on a
null, and the frozen decision rule (N1 vs N2 vs pivot) must fire correctly."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from h2cmi.analyze_b1a_astar import analyze, ACTIONS


def _nrow(seed, pair, ps, action, ev, cos=0.3, etn=1.0, dis=0.5):
    return dict(difficulty="standard", data_seed=seed, excluded_site_pair=pair, pseudo_target_site=ps,
                action=action, raw_evidence_score=ev, transform_direction_cosine=cos,
                transform_effect_to_noise_ratio=etn, crossfit_prediction_disagreement=dis)


def _nested(rng):
    rows = []
    for seed in (0, 1, 2):
        for pi, pair in enumerate(["0-1", "0-2", "1-3", "2-4"]):
            for ps in (int(pair[0]), int(pair[-1])):
                for a in ACTIONS:                                # null: low evidence + low B signals
                    rows.append(_nrow(seed, pair, ps, a, ev=float(rng.uniform(0, 1)),
                                      cos=float(rng.uniform(0.1, 0.4)), etn=float(rng.uniform(0.5, 1.5)),
                                      dis=float(rng.uniform(0.4, 0.7))))
    return rows


def _rrow(scen, seed, site, action, ev, bacc, cos=0.3, etn=1.0, dis=0.5, idb=0.60):
    return dict(difficulty="standard", scenario=scen, data_seed=seed, target_site=site, action=action,
                evidence_target=ev, bacc_uniform=bacc, grouped_oof_bacc=bacc,
                identity_bacc_uniform=idb, identity_grouped_oof_bacc=idb,
                transform_direction_cosine=cos, transform_effect_to_noise_ratio=etn,
                crossfit_prediction_disagreement=dis)


def _real():
    rows = []
    for seed in (0, 1, 2):
        for site in (0, 1):
            # cov: gen_oneshot is a clear, reproducible, high-evidence winner
            rows += [_rrow("cov", seed, site, "gen_oneshot_diag", ev=8.0, bacc=0.70, cos=0.95, etn=12.0, dis=0.05),
                     _rrow("cov", seed, site, "pooled_empirical_diag", ev=1.2, bacc=0.63, cos=0.4, etn=1.5, dis=0.3),
                     _rrow("cov", seed, site, "gen_iterative_diag", ev=1.0, bacc=0.62, cos=0.5, etn=2.0, dis=0.3)]
            # matched null: evidence at null level -> abstain
            rows += [_rrow("matched_domain_null", seed, site, a, ev=0.6, bacc=0.985, cos=0.3, etn=1.0, dis=0.5, idb=0.99)
                     for a in ACTIONS]
    return rows


def test_astar_adapts_on_shift_abstains_on_null():
    rng = np.random.default_rng(0)
    rep = analyze(_real(), _nested(rng))
    n1 = rep["by_router"]["N1"]["standard"]
    assert n1["false_adaptation_rate_null"] == 0.0           # abstain on the null
    assert n1["selection_frequency_shift"]["gen_oneshot_diag"] == 6   # all cov units -> winner
    assert n1["mean_dbacc_full_shift"] > 0.05 and n1["coverage"] > 0.4
    assert n1["top1_oracle_full"] == 1.0


def test_decision_rule_pivots_when_both_fail():
    # all evidence at null level -> nothing eligible -> no coverage -> both routers fail
    rng = np.random.default_rng(1)
    real = [r for r in _real()]
    for r in real:
        r["evidence_target"] = 0.5                            # kill the signal
    rep = analyze(real, _nested(rng))
    assert rep["by_router"]["N1"]["pass"]["ALL"] is False
    assert "pivot to B" in rep["decision"]["outcome"]


def test_n2_veto_is_subset_of_n1():
    rng = np.random.default_rng(2)
    rep = analyze(_real(), _nested(rng))
    n1cov = rep["by_router"]["N1"]["standard"]["selection_frequency_shift"]
    n2cov = rep["by_router"]["N2"]["standard"]["selection_frequency_shift"]
    # N2 can only abstain MORE than N1 (veto), never adapt where N1 abstains
    assert n2cov["identity"] >= n1cov["identity"]


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"  {n} PASSED")
    print("test_analyze_b1a_astar PASSED")

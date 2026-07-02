"""C7 K2: reproducible-gain decision across seeds/levels. Standalone + pytest-compatible."""
from __future__ import annotations

import random

from oaci.decision.k2_decision import (K2_ABSTAIN_ENDPOINT, K2_ABSTAIN_SEEDS, K2_REPRODUCIBLE, K2_STOP,
                                       k2_decision)

# manifest-derived spec (the function has NO code defaults; thresholds are always supplied)
K = dict(endpoints=["worst_domain_bacc", "worst_domain_nll"], min_seeds=3, level_policy="both_levels",
         margins={"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0})


def _u(seed, level, bacc=None, nll=None):
    return {"seed": seed, "level": level, "deltas": {"worst_domain_bacc": bacc, "worst_domain_nll": nll}}


def _grid(seeds, levels, bacc, nll):
    return [_u(s, L, bacc, nll) for s in seeds for L in levels]


def test_k2_abstains_with_single_seed():
    r = k2_decision(_grid([0], [0, 1], bacc=0.05, nll=-0.05), **K)
    assert r["k2_status"] == K2_ABSTAIN_SEEDS and not r["continue"] and r["n_seeds"] == 1


def test_k2_detects_reproducible_bacc_gain():
    r = k2_decision(_grid([0, 1, 2], [0, 1], bacc=0.02, nll=0.10), **K)     # bAcc up everywhere; NLL worse
    assert r["k2_status"] == K2_REPRODUCIBLE and r["continue"]
    assert r["reproduced_endpoints"] == ["worst_domain_bacc"]
    assert r["per_endpoint"]["worst_domain_bacc"]["reproducible"]
    assert not r["per_endpoint"]["worst_domain_nll"]["reproducible"]


def test_k2_detects_reproducible_nll_gain():
    r = k2_decision(_grid([0, 1, 2], [0, 1], bacc=-0.01, nll=-0.05), **K)   # NLL down everywhere; bAcc worse
    assert r["k2_status"] == K2_REPRODUCIBLE and r["reproduced_endpoints"] == ["worst_domain_nll"]


def test_k2_stops_on_mixed_seed_or_level_effects():
    units = _grid([0, 1, 2], [0, 1], bacc=0.02, nll=0.10)                   # bAcc gain everywhere...
    units[-1]["deltas"]["worst_domain_bacc"] = -0.01                        # ...except one (seed2,level1)
    r = k2_decision(units, **K)
    assert r["k2_status"] == K2_STOP and not r["continue"]
    assert r["per_endpoint"]["worst_domain_bacc"]["n_gain"] == 5            # 6 units, 1 fails


def test_k2_reports_missing_endpoint_cleanly():
    # one endpoint fully present + reproducible, the other entirely missing -> gain via the present one
    r = k2_decision(_grid([0, 1, 2], [0, 1], bacc=0.02, nll=None), **K)
    assert r["k2_status"] == K2_REPRODUCIBLE and not r["per_endpoint"]["worst_domain_nll"]["evaluable"]
    # BOTH endpoints missing -> abstain (cannot assess reproducibility)
    r2 = k2_decision(_grid([0, 1, 2], [0, 1], bacc=None, nll=None), **K)
    assert r2["k2_status"] == K2_ABSTAIN_ENDPOINT and not r2["continue"]


def test_k2_is_order_invariant_across_folds_and_seeds():
    units = _grid([0, 1, 2], [0, 1], bacc=0.02, nll=-0.03)
    a = k2_decision(units, **K)
    shuffled = list(units); random.Random(0).shuffle(shuffled)
    b = k2_decision(shuffled, **K)
    assert a["k2_status"] == b["k2_status"] and a["reproduced_endpoints"] == b["reproduced_endpoints"]
    assert a["per_endpoint"] == b["per_endpoint"] and a["seeds"] == b["seeds"]


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} decision-k2 tests")


if __name__ == "__main__":
    _run_all()

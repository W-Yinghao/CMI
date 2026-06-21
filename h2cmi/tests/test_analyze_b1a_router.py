"""The B1b-1 evidence router must apply the fixed rule (argmax positive evidence gain, else
identity), evaluate the held-out outcome of that choice, and never adapt on a null where no
action has positive evidence."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from h2cmi.analyze_b1a_router import route_units, router_report, DEPLOYABLE


def _row(diff, scen, seed, site, variant, evid, oof):
    return dict(difficulty=diff, scenario=scen, data_seed=seed, target_site=site, variant=variant,
                grouped_crossfit_evidence_gain=evid, grouped_oof_bacc=oof)


def _shift_unit(seed, site, *, win="gen_oneshot_diag"):
    # win action has the highest (positive) evidence AND the highest OOF bAcc
    rows = [_row("standard", "cov", seed, site, "identity", 0.0, 0.60)]
    for a, (e, b) in {"pooled_empirical_diag": (0.3, 0.64), "gen_oneshot_diag": (0.8, 0.70),
                      "gen_iterative_diag": (0.5, 0.66)}.items():
        if a != win:
            e, b = (0.1, 0.61)
        rows.append(_row("standard", "cov", seed, site, a, e, b))
    return rows


def _null_unit(seed, site):
    # nothing beats identity: all evidence <= 0
    rows = [_row("standard", "matched_domain_null", seed, site, "identity", 0.0, 0.99)]
    for a in ("pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag"):
        rows.append(_row("standard", "matched_domain_null", seed, site, a, -0.2, 0.98))
    return rows


def test_router_selects_max_positive_evidence():
    units = route_units(sum([_shift_unit(s, t) for s in (0, 1, 2) for t in (0, 1)], []))
    assert units and all(u["selected"] == "gen_oneshot_diag" for u in units)
    assert all(u["adapted"] and u["d_router"] > 0 and u["top1_agree"] for u in units)


def test_router_rolls_back_on_null():
    units = route_units(sum([_null_unit(s, t) for s in (0, 1, 2) for t in (0, 1)], []))
    assert all(u["selected"] == "identity" and not u["adapted"] for u in units)
    assert all(u["d_router"] == 0.0 for u in units)


def test_router_report_metrics():
    rows = sum([_shift_unit(s, t) for s in (0, 1, 2) for t in (0, 1)], [])
    rows += sum([_null_unit(s, t) for s in (0, 1, 2) for t in (0, 1)], [])
    rep = router_report(rows, n_boot=200)
    d = rep["by_difficulty"]["standard"]
    assert d["false_adaptation_rate_null"] == 0.0            # never adapt on the null
    assert d["adaptation_rate_shift"] == 1.0                 # always adapt on the shift
    assert d["harm_rate"] == 0.0                             # the winning action never hurts
    assert abs(d["delta_bacc_router_shift"]["mean"] - 0.10) < 1e-9   # 0.70 - 0.60
    assert d["top1_oracle_agreement"] == 1.0
    assert d["evidence_accuracy_spearman"] > 0.5            # evidence predicts accuracy gain
    assert list(DEPLOYABLE)[0] == "identity"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"  {n} PASSED")
    print("test_analyze_b1a_router PASSED")

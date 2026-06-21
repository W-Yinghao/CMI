"""The nested conjunction routers R0..R3: R3 must adapt only when A (null p), B (reproducibility)
and C (structure) all certify the action, and roll back to identity otherwise -- in particular it
must NOT adapt on a null where the raw evidence router (R0) does."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from h2cmi.analyze_b1a_routers import analyze, route_unit

SRC = dict(src_q05_cosine=0.3, src_q05_predstab=0.5, src_q05_etn=0.5, src_q95_anchor=0.10, src_q95_dsnd=0.10)


def _row(scen, seed, site, action, *, p, cos, dis, flip, occ, dsnd, dga, evid, bacc, idb=0.60):
    return dict(difficulty="standard", scenario=scen, data_seed=seed, target_site=site, action=action,
                n_actions=3, null_pvalue=p, transform_direction_cosine=cos,
                crossfit_prediction_disagreement=dis, anchor_flip_rate=flip, anchor_n=50,
                min_class_occupancy=occ, delta_snd=dsnd, delta_disc_gen_agreement=dga,
                evidence_target=evid, bacc_uniform=bacc, grouped_oof_bacc=bacc,
                identity_bacc_uniform=idb, identity_grouped_oof_bacc=idb, **SRC)


def _shift(seed, site):
    # gen_oneshot passes ALL R3 clauses; the others fail the p-test
    return [
        _row("cov", seed, site, "gen_oneshot_diag", p=0.001, cos=0.95, dis=0.05, flip=0.01, occ=0.30,
             dsnd=0.50, dga=0.10, evid=8.0, bacc=0.70),
        _row("cov", seed, site, "pooled_empirical_diag", p=0.5, cos=0.6, dis=0.2, flip=0.2, occ=0.25,
             dsnd=0.0, dga=0.0, evid=5.0, bacc=0.64),
        _row("cov", seed, site, "gen_iterative_diag", p=0.3, cos=0.95, dis=0.05, flip=0.01, occ=0.30,
             dsnd=0.40, dga=0.05, evid=9.0, bacc=0.66)]


def _null(seed, site):
    # nothing passes the p-test, but evidence is high (R0 would still adapt)
    return [_row("matched_domain_null", seed, site, a, p=0.5, cos=0.95, dis=0.05, flip=0.01, occ=0.30,
                 dsnd=0.5, dga=0.1, evid=6.0, bacc=0.985, idb=0.99) for a in
            ("pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag")]


def test_r3_selects_certified_action_and_respects_null():
    rows = sum([_shift(s, t) for s in (0, 1, 2) for t in (0, 1)], [])
    rows += sum([_null(s, t) for s in (0, 1, 2) for t in (0, 1)], [])
    rep = analyze(rows)
    r3 = rep["by_difficulty"]["standard"]["R3"]
    r0 = rep["by_difficulty"]["standard"]["R0"]
    assert r3["false_adaptation_rate_null"] == 0.0            # R3 never adapts on the null
    assert r0["false_adaptation_rate_null"] == 1.0            # R0 (raw evidence) always does
    assert abs(r3["mean_dbacc_full_shift"] - 0.10) < 1e-9     # 0.70 - 0.60 on cov
    assert r3["top1_oracle_full"] == 1.0 and r3["harm_rate_full"] == 0.0


def test_route_unit_rolls_back_when_nothing_eligible():
    vm = {r["action"]: r for r in _null(0, 0)}
    assert route_unit(vm, "R3") == "identity" and route_unit(vm, "R1") == "identity"
    vm2 = {r["action"]: r for r in _shift(0, 0)}
    assert route_unit(vm2, "R3") == "gen_oneshot_diag"


def test_nesting_monotone_eligibility():
    # an action eligible at R3 must be eligible at R2 and R1 (nested conjunction)
    vm = {r["action"]: r for r in _shift(0, 0)}
    from h2cmi.analyze_b1a_routers import _eligible
    g = vm["gen_oneshot_diag"]
    assert _eligible(g, 1) and _eligible(g, 2) and _eligible(g, 3)


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"  {n} PASSED")
    print("test_analyze_b1a_routers PASSED")

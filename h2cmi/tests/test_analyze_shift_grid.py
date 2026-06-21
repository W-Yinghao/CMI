"""Validate the Stage-A analysis: seed-clustered contrasts (5 contrasts + M3-max), the
renamed mass field, scenario aliasing, and the decision tree on planted failure modes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.analyze_shift_grid import analyze, canon

# planted per-scenario mean ΔbAcc per method + held-out gains, each hitting one branch.
PLAN = {
    "population_null":   dict(tta=0.016, oracle_prior=0.018, oracle_labels=0.029, oracle_supervised=0.029,
                              g_unsup=3.0, g_sup=3.0, adapted=True, expect="population_null_adapts"),
    "matched_domain_null": dict(tta=-0.03, oracle_prior=-0.02, oracle_labels=-0.01, oracle_supervised=-0.01,
                                g_unsup=0.0, g_sup=0.0, adapted=True, expect="rollback_loose"),
    "cov":               dict(tta=0.10, oracle_prior=0.10, oracle_labels=0.11, oracle_supervised=0.12,
                              g_unsup=2.0, g_sup=2.5, adapted=True, expect="unsup_helps", cmi_interaction=0.03),
    "prior":             dict(tta=0.00, oracle_prior=0.10, oracle_labels=0.10, oracle_supervised=0.12,
                              g_unsup=0.5, g_sup=2.0, adapted=True, expect="prior_bottleneck"),
    "resp":              dict(tta=0.00, oracle_prior=0.00, oracle_labels=0.10, oracle_supervised=0.12,
                              g_unsup=0.5, g_sup=2.0, adapted=True, expect="responsibilities_bottleneck"),
    "conditional_rotation": dict(tta=-0.05, oracle_prior=0.00, oracle_labels=0.00, oracle_supervised=0.00,
                                 g_unsup=0.0, g_sup=0.05, adapted=True, expect="family_insufficient"),
}
METHODS = ("identity", "tta", "oracle_prior", "oracle_labels", "oracle_supervised")


def _build(path):
    rows = []
    for scen, p in PLAN.items():
        for seed in (0, 1, 2):
            for site in (0, 1, 2):
                jit = 0.002 * (site - 1) + 0.001 * (seed - 1)
                for method in METHODS:
                    for cmi in ("off", "on"):
                        if method == "identity":
                            delta = 0.0
                        else:
                            delta = p[method] + jit
                            if method == "tta" and cmi == "on":
                                delta += p.get("cmi_interaction", 0.0)
                        rows.append(dict(
                            data_seed=seed, target_site=site, scenario=scen, method=method, cmi=cmi,
                            strict_bacc=0.70, adapted_bacc=0.70 + delta, delta_bacc=delta,
                            adapted=bool(p["adapted"]) if method != "identity" else False,
                            nll=0.5, brier=0.3, ece=0.1, prior_l1_error=0.1,
                            crossfit_evidence_gain=p["g_unsup"], crossfit_supervised_gain=p["g_sup"]))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_decision_tree_and_seed_cluster():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "grid.jsonl")
        _build(path)
        rep = analyze(path, n_boot=400, seed=0)
        assert rep["cluster"] == "seed"
        # decisions
        for scen, p in PLAN.items():
            got = rep["decisions"][scen]["code"]
            assert got == p["expect"], f"{scen}: expected {p['expect']}, got {got}"
        # seed-clustered interaction recovers the planted +0.03 CMI effect on cov
        cov_i = rep["contrasts"]["cov"]["interaction"]
        assert cov_i["n_seeds"] == 3
        assert abs(cov_i["seed_mean"] - 0.03) < 0.01, cov_i
        assert cov_i["bootstrap_mass_above_zero"] > 0.9
        # 3 seeds -> sign-flip two-sided p cannot beat 0.25 (screening is not confirmatory)
        assert cov_i["signflip_p_two_sided"] >= 0.25 - 1e-9
        # the renamed field exists; the old p-value-like name does not
        assert "bootstrap_mass_above_zero" in cov_i and "p_gt0" not in cov_i
        # five contrasts + m3_minus_max all present
        for name in ("strict_cmi", "tta_no_cmi", "tta_with_cmi", "cmi_under_tta",
                     "interaction", "m3_minus_max"):
            assert name in rep["contrasts"]["cov"], name
        # supervised flagged as transductive proxy (no OOF rows here)
        assert rep["oracle_table"]["cov"]["sup_is_oof"] is False


def test_scenario_alias():
    assert canon("no_shift") == "population_null"
    assert canon("concept") == "conditional_rotation"
    assert canon("cov_concept") == "cov_conditional_rotation"
    assert canon("cov") == "cov"


if __name__ == "__main__":
    test_decision_tree_and_seed_cluster()
    test_scenario_alias()
    print("test_analyze_shift_grid PASSED")

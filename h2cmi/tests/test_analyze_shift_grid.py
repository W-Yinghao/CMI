"""Validate the shift-grid analysis: the decision tree localises each planted failure mode,
and the paired interaction bootstrap recovers a planted CMI effect."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.analyze_shift_grid import analyze

# planted per-scenario mean ΔbAcc for each method, and held-out gains, designed to hit a
# specific decision-tree branch each.
PLAN = {
    "no_shift":    dict(tta=-0.03, oracle_prior=-0.02, oracle_labels=-0.02, oracle_supervised=-0.02,
                        g_unsup=3.0, g_sup=3.0, adapted=True, expect="rollback_loose"),
    "cov":         dict(tta=0.10, oracle_prior=0.10, oracle_labels=0.11, oracle_supervised=0.12,
                        g_unsup=2.0, g_sup=2.5, adapted=True, expect="unsup_helps"),
    "prior":       dict(tta=0.00, oracle_prior=0.10, oracle_labels=0.10, oracle_supervised=0.12,
                        g_unsup=0.5, g_sup=2.0, adapted=True, expect="prior_bottleneck"),
    "resp":        dict(tta=0.00, oracle_prior=0.00, oracle_labels=0.10, oracle_supervised=0.12,
                        g_unsup=0.5, g_sup=2.0, adapted=True, expect="responsibilities_bottleneck"),
    "concept":     dict(tta=-0.05, oracle_prior=0.00, oracle_labels=0.00, oracle_supervised=0.00,
                        g_unsup=0.0, g_sup=0.05, adapted=True, expect="family_insufficient"),
    "perp":        dict(tta=-0.04, oracle_prior=0.00, oracle_labels=0.00, oracle_supervised=0.00,
                        g_unsup=0.0, g_sup=1.5, adapted=True, expect="density_perp_boundary"),
}
METHODS = ("identity", "tta", "oracle_prior", "oracle_labels", "oracle_supervised")


def _build(path):
    rows = []
    for scen, p in PLAN.items():
        for seed in (0, 1):
            for site in (0, 1, 2):
                jit = 0.004 * (site - 1) + 0.003 * (seed - 0.5)   # tiny deterministic spread
                for method in METHODS:
                    for cmi in ("off", "on"):
                        if method == "identity":
                            delta = 0.0
                        else:
                            delta = p[method] + jit
                            # plant a small positive CMI effect on cov's TTA only
                            if scen == "cov" and method == "tta" and cmi == "on":
                                delta += 0.03
                        row = dict(data_seed=seed, target_site=site, scenario=scen,
                                   method=method, cmi=cmi, factorial_cell=None,
                                   strict_bacc=0.7, adapted_bacc=0.7 + delta, delta_bacc=delta,
                                   adapted=bool(p["adapted"]) if method != "identity" else False,
                                   crossfit_evidence_gain=p["g_unsup"],
                                   crossfit_supervised_gain=p["g_sup"])
                        rows.append(row)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def test_decision_tree_and_interaction():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "grid.jsonl")
        _build(path)
        rep = analyze(path, n_boot=500, seed=0)
        # each planted scenario maps to its intended diagnosis
        for scen, p in PLAN.items():
            got = rep["decisions"][scen]["code"]
            assert got == p["expect"], f"{scen}: expected {p['expect']}, got {got}"
        # paired interaction recovers the planted +0.03 CMI effect on cov, ~0 elsewhere
        cov = rep["interaction"]["cov"]["interaction"]
        assert cov["n"] == 6 and abs(cov["mean"] - 0.03) < 0.01 and cov["p_gt0"] > 0.9, cov
        concept = rep["interaction"]["concept"]["interaction"]
        assert abs(concept["mean"]) < 0.01, concept
        # harm rate present and sensible (concept TTA harmful)
        assert rep["interaction"]["concept"]["harm_rate_off"] == 1.0


if __name__ == "__main__":
    test_decision_tree_and_interaction()
    print("test_analyze_shift_grid PASSED")

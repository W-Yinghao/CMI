"""The frozen B1a analysis must compute the five contrasts (identity-cancelling differences of
two variants' per-unit metric, seed-clustered) and the hard-null safety panel correctly."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from h2cmi.analyze_b1a_grid import analyze, hard_null_safety, CONTRASTS

# known per-variant metrics: contrasts are metric[a]-metric[b]
BACC = {"identity": 0.50, "pooled_empirical_diag": 0.60, "gen_oneshot_diag": 0.70,
        "gen_iterative_diag": 0.66, "oracle_oneshot_diag": 0.72, "oracle_oneshot_lowrank": 0.74,
        "joint_iterative_diag": 0.63}
OOF = {"identity": 0.50, "pooled_empirical_diag": 0.58, "gen_oneshot_diag": 0.68,
       "gen_iterative_diag": 0.66, "oracle_oneshot_diag": 0.75, "oracle_oneshot_lowrank": 0.80,
       "joint_iterative_diag": 0.62}


def _rows(difficulty="standard", scenario="cov", seeds=(0, 1, 2), sites=(0, 1, 2, 3, 4)):
    rows = []
    for s in seeds:
        for t in sites:
            for v in BACC:
                rows.append(dict(difficulty=difficulty, scenario=scenario, data_seed=s, target_site=t,
                                 variant=v, bacc_uniform=BACC[v], grouped_oof_bacc=OOF[v]))
    return rows


def test_contrasts_match_known_differences():
    rep = analyze(_rows(), n_boot=200)
    cs = rep["contrasts"]["standard/cov"]
    # C_feedback (full) = bacc[gen_oneshot]-bacc[gen_iterative] = 0.70-0.66 = +0.04
    assert abs(cs["C_feedback"]["mean"] - 0.04) < 1e-9 and cs["C_feedback"]["meets_threshold"]
    # C_responsibility (oof) = 0.75-0.68 = +0.07
    assert abs(cs["C_responsibility"]["mean"] - 0.07) < 1e-9 and cs["C_responsibility"]["meets_threshold"]
    # C_class_cond full = 0.70-0.60 = +0.10 ; oof = 0.68-0.58 = +0.10
    assert abs(cs["C_class_cond_full"]["mean"] - 0.10) < 1e-9
    assert abs(cs["C_class_cond_oof"]["mean"] - 0.10) < 1e-9
    # C_family (oof) = 0.80-0.75 = +0.05
    assert abs(cs["C_family"]["mean"] - 0.05) < 1e-9 and cs["C_family"]["meets_threshold"]
    # C_prior_coupling (full) = bacc[gen_iterative]-bacc[joint] = 0.66-0.63 = +0.03
    assert abs(cs["C_prior_coupling"]["mean"] - 0.03) < 1e-9 and cs["C_prior_coupling"]["meets_threshold"]
    assert all(c["n_seeds"] == 3 for c in cs.values())


def test_subthreshold_contrast_flagged_false():
    # shrink the feedback gap below 0.02
    rows = _rows()
    for r in rows:
        if r["variant"] == "gen_iterative_diag":
            r["bacc_uniform"] = 0.695                          # gap now 0.005 < 0.02
    cs = analyze(rows, n_boot=200)["contrasts"]["standard/cov"]
    assert not cs["C_feedback"]["meets_threshold"]


def test_hard_null_safety_pass_and_fail():
    def hrows(delta, disagree, occ, nll, best):
        rows = []
        for s in (0, 1, 2):
            for t in (0, 1):
                for v in BACC:
                    rows.append(dict(difficulty="hard", scenario="matched_domain_null",
                                     data_seed=s, target_site=t, variant=v,
                                     bacc_uniform=BACC[v], grouped_oof_bacc=OOF[v],
                                     delta_bacc_uniform=(0.0 if v == "identity" else delta),
                                     prediction_disagreement=(0.0 if v == "identity" else disagree),
                                     final_class_occupancy=(0.33 if v == "identity" else occ),
                                     grouped_oof_nll=(0.9 if v == "identity" else nll),
                                     oracle_best_variant=best))
        return rows
    safe = hard_null_safety({v: [r for r in hrows(0.005, 0.01, 0.30, 0.9, "identity") if r["variant"] == v]
                             for v in BACC})
    assert safe["all_safe"] and safe["identity_in_oracle_best"]
    bad = hard_null_safety({v: [r for r in hrows(0.05, 0.10, 0.0, 1.5, "gen_oneshot_diag") if r["variant"] == v]
                            for v in BACC})
    assert not bad["all_safe"] and not bad["identity_in_oracle_best"]


def test_contrast_registry_is_the_five_questions():
    names = {c[0] for c in CONTRASTS}
    assert names == {"C_feedback", "C_responsibility", "C_class_cond_full", "C_class_cond_oof",
                     "C_family", "C_prior_coupling"}


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"  {n} PASSED")
    print("test_analyze_b1a_grid PASSED")

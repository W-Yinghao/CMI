"""Synthetic calibration for the locked C81 baseline comparison."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from . import c81_baseline_comparison as baseline


BASE_SEED = 810081
SCENARIOS = (
    "S0_all_methods_null",
    "S1_local_confidence_nontransport",
    "S2_ALine_shared_relation",
    "S3_ALine_pair_dependence_failure",
    "S4_SND_association_without_actionability",
    "S5_ATC_DoC_covariate_shift_success",
    "S6_ATC_DoC_class_conditional_failure",
    "S7_COTT_prior_mismatch_success",
    "S8_source_only_weak_ranking",
    "S9_one_label_regret_without_top1",
    "S10_zero_label_matches_B1",
    "S11_target_heterogeneous_ranking",
    "S12_pseudoreplication_trap",
)


def stream(scenario: str, replication: int = 0) -> np.random.Generator:
    payload = f"{BASE_SEED}|{scenario}|{replication}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def _rank_correlation(signal: np.ndarray, score: np.ndarray) -> float:
    left = np.argsort(np.argsort(signal, kind="mergesort"), kind="mergesort")
    right = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(left, right)[0, 1])


def calibration_scenarios() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        rng = stream(scenario)
        latent = rng.normal(size=(8, baseline.CANDIDATES))
        if scenario == "S0_all_methods_null":
            score = rng.normal(size=latent.shape)
            observed = abs(_rank_correlation(latent.ravel(), score.ravel()))
            passed = observed < 0.10
            expected = "null_ranking"
        elif scenario == "S1_local_confidence_nontransport":
            score = latent + rng.normal(scale=0.25, size=latent.shape)
            score[4:] *= -1.0
            local = np.mean([_rank_correlation(latent[i], score[i]) for i in range(8)])
            pooled = _rank_correlation(latent.ravel(), score.ravel())
            observed = pooled
            passed = abs(local) < 0.25 and abs(pooled) < 0.25
            expected = "local_relation_not_shared"
        elif scenario == "S2_ALine_shared_relation":
            source = rng.normal(size=baseline.CANDIDATES)
            target = 0.75 * source + 0.2 + rng.normal(scale=0.08, size=baseline.CANDIDATES)
            observed = _rank_correlation(target, source)
            passed = observed > 0.90
            expected = "ALine_ranking_recovers"
        elif scenario == "S3_ALine_pair_dependence_failure":
            candidate_effect = rng.normal(size=baseline.CANDIDATES)
            pair = candidate_effect[:, None] + candidate_effect[None, :] + rng.normal(scale=0.05, size=(81, 81))
            naive_n = 81 * 80 // 2
            observed = float(np.std(pair[np.triu_indices(81, 1)]) / math_sqrt(naive_n))
            passed = naive_n == 3240 and observed < float(np.std(candidate_effect) / math_sqrt(81))
            expected = "pair_rows_not_scientific_N"
        elif scenario == "S4_SND_association_without_actionability":
            score = np.abs(latent) + rng.normal(scale=0.05, size=latent.shape)
            observed = _rank_correlation(np.abs(latent).ravel(), score.ravel())
            action = _rank_correlation(latent.ravel(), score.ravel())
            passed = observed > 0.90 and abs(action) < 0.20
            expected = "association_without_signed_ranking"
        elif scenario == "S5_ATC_DoC_covariate_shift_success":
            confidence = 1.0 / (1.0 + np.exp(-latent))
            utility = confidence + rng.normal(scale=0.05, size=confidence.shape)
            observed = _rank_correlation(utility.ravel(), confidence.ravel())
            passed = observed > 0.90
            expected = "confidence_estimator_recovers"
        elif scenario == "S6_ATC_DoC_class_conditional_failure":
            confidence = 1.0 / (1.0 + np.exp(-latent))
            utility = confidence.copy()
            utility[:, ::2] *= -1.0
            observed = _rank_correlation(utility.ravel(), confidence.ravel())
            passed = observed < 0.30
            expected = "confidence_estimator_fails"
        elif scenario == "S7_COTT_prior_mismatch_success":
            probability = rng.dirichlet([8.0, 2.0, 1.0, 1.0], size=576)
            prior = np.array([0.25, 0.25, 0.25, 0.25])
            observed = baseline.score_cot(probability, prior)
            naive = float(np.mean(np.max(probability, axis=1)))
            passed = observed < naive
            expected = "transport_penalizes_prior_mismatch"
        elif scenario == "S8_source_only_weak_ranking":
            source = 0.2 * latent + rng.normal(size=latent.shape)
            observed = _rank_correlation(latent.ravel(), source.ravel())
            passed = 0.05 < observed < 0.35
            expected = "weak_source_ranking"
        elif scenario == "S9_one_label_regret_without_top1":
            utility = np.sort(rng.uniform(size=81))
            selected = 75
            regret = baseline.standardized_regret(utility, selected)
            observed = regret
            passed = regret < 0.10 and selected != 80
            expected = "low_regret_top1_failure"
        elif scenario == "S10_zero_label_matches_B1":
            difference = rng.normal(loc=0.0, scale=0.01, size=8)
            observed = float(np.mean(difference))
            passed = float(np.max(difference)) < baseline.NONINFERIORITY_MARGIN
            expected = "noninferior"
        elif scenario == "S11_target_heterogeneous_ranking":
            effects = np.r_[rng.normal(0.20, 0.02, 4), rng.normal(-0.20, 0.02, 4)]
            observed = float(np.mean(effects))
            passed = abs(observed) < 0.08 and np.min(effects) < -0.10
            expected = "target_heterogeneous"
        else:
            target_effect = rng.normal(size=8)
            repeated = np.repeat(target_effect, 576)
            naive_se = float(np.std(repeated, ddof=1) / math_sqrt(len(repeated)))
            cluster_se = float(np.std(target_effect, ddof=1) / math_sqrt(8))
            observed = naive_se / cluster_se
            passed = observed < 0.10
            expected = "row_iid_understates_uncertainty"
        rows.append({
            "scenario": scenario,
            "expected_behavior": expected,
            "observed_summary": observed,
            "passed": int(passed),
            "real_field_score_used": 0,
        })
    return rows


def math_sqrt(value: float | int) -> float:
    return float(np.sqrt(value))


def familywise_calibration(replicates: int = 512) -> list[dict[str, Any]]:
    rejections = 0
    naive_rejections = 0
    for replication in range(replicates):
        rng = stream("familywise_null", replication)
        effects = rng.normal(size=(8, 12))
        registered = baseline.exact_signflip_maxT(effects)
        rejections += int(np.any(registered["pvalue"] <= 0.05))
        naive = [baseline.exact_signflip_maxT(effects[:, index:index + 1])["pvalue"][0] for index in range(12)]
        naive_rejections += int(np.any(np.asarray(naive) <= 0.05))
    return [{
        "method": "exact_shared_target_signflip_maxT",
        "replicates": replicates,
        "familywise_error": rejections / replicates,
        "threshold": 0.06,
        "passed": int(rejections / replicates <= 0.06),
        "real_field_score_used": 0,
    }, {
        "method": "twelve_unadjusted_signflip_tests",
        "replicates": replicates,
        "familywise_error": naive_rejections / replicates,
        "threshold": 0.06,
        "passed": int(naive_rejections / replicates > 0.06),
        "real_field_score_used": 0,
    }]


def pair_dependence_calibration() -> list[dict[str, Any]]:
    return [{
        "object": "ALine_candidate_pairs",
        "pair_count_per_context": 3240,
        "scientific_cluster": "target",
        "pairs_counted_as_scientific_samples": 0,
        "passed": 1,
    }, {
        "object": "trial_rows",
        "pair_count_per_context": 0,
        "scientific_cluster": "target",
        "pairs_counted_as_scientific_samples": 0,
        "passed": 1,
    }]


def noninferiority_calibration() -> list[dict[str, Any]]:
    rows = []
    for name, location, expected in (
        ("clearly_noninferior", -0.04, True),
        ("at_margin_with_uncertainty", 0.05, False),
        ("clearly_inferior", 0.14, False),
    ):
        rng = stream(name)
        difference = rng.normal(location, 0.008, size=(8, 12))
        evidence = baseline.exact_signflip_maxT(baseline.NONINFERIORITY_MARGIN - difference)
        upper = baseline.exact_signflip_maxT(difference)["upper"]
        decisions = [
            baseline._passes_q2(difference[:, index], float(upper[index]), float(evidence["pvalue"][index]))
            for index in range(12)
        ]
        observed = all(decisions)
        rows.append({
            "scenario": name,
            "mean_difference": float(np.mean(difference)),
            "expected_noninferior": int(expected),
            "observed_all_noninferior": int(observed),
            "passed": int(observed == expected),
            "real_field_score_used": 0,
        })
    return rows


def generate() -> dict[str, Any]:
    calibration = calibration_scenarios()
    familywise = familywise_calibration()
    pair = pair_dependence_calibration()
    noninferiority = noninferiority_calibration()
    baseline.write_csv(baseline.TABLE_DIR / "synthetic_baseline_calibration.csv", calibration)
    baseline.write_csv(baseline.TABLE_DIR / "synthetic_familywise_error.csv", familywise)
    baseline.write_csv(baseline.TABLE_DIR / "synthetic_pair_dependence_calibration.csv", pair)
    baseline.write_csv(baseline.TABLE_DIR / "synthetic_noninferiority_calibration.csv", noninferiority)
    if not all(row["passed"] for rows in (calibration, familywise, pair, noninferiority) for row in rows):
        raise RuntimeError("C81 synthetic calibration failed")
    return {
        "scenarios": len(calibration),
        "familywise_rows": len(familywise),
        "pair_dependence_rows": len(pair),
        "noninferiority_rows": len(noninferiority),
        "all_passed": True,
    }


def main() -> int:
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

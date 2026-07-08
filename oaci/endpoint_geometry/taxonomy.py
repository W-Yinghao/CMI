"""C31 taxonomy — deterministic E1-E9 verdict from the endpoint-geometry evidence. Base rates / imbalance are
reported BEFORE any case is decided (hard gate #11). Diagnostic-only; every oracle endpoint is non-deployable and
no case licenses a selector."""
from __future__ import annotations

from . import schema


def _imbalance_flags(base):
    """Report endpoint imbalance / degeneracy before deciding anything (gate #11)."""
    flags = []
    for lab in ("accuracy_good", "calibration_good", "joint_good", "pareto_good"):
        r = base[lab]["rate"]
        if r is None:
            flags.append(f"{lab}: undefined (no finite labels)")
        elif r < 0.05 or r > 0.95:
            flags.append(f"{lab}: degenerate base rate {r:.3f} (AUC/overlap unstable)")
    return flags


def classify(base, overlap, source_rank, gauge, pareto) -> dict:
    cases, evid = [], {}

    # E1 — accuracy-calibration trade-off. Confirmed only if bAcc improvement is NEGATIVELY coupled with
    # calibration improvement AND that survives the epoch confound control.
    tradeoff = bool(overlap["tradeoff_confirmed"])
    evid["E1"] = {"tradeoff_confirmed": tradeoff, "mean_bacc_vs_calib_corr": overlap["mean_bacc_vs_calib_improve_corr"],
                  "coupling_survives_epoch_control": overlap["coupling_survives_epoch_control"],
                  "accuracy_good_calibration_bad_rate": overlap["conflict"]["accuracy_good_calibration_bad_rate"]}
    if tradeoff:
        cases.append(schema.E1)

    # E2 — joint-good absent/rare. E3 — joint-good exists but not deployably (pooled) source-observable.
    jr = base["joint_good"]["rate"] or 0.0
    joint_within = (source_rank["per_factor"]["score"]["joint_good"]["within_target_auc"] or 0.0)
    joint_pooled = (source_rank["per_factor"]["score"]["joint_good"]["pooled_auc"] or 0.0)
    joint_deployable = bool(abs(joint_pooled - 0.5) >= abs(schema.RANK_SIGNAL_MIN - 0.5))   # pooled = cross-target selectable
    evid["E2_E3"] = {"joint_good_rate": jr, "joint_within_target_auc": joint_within, "joint_pooled_auc": joint_pooled,
                     "joint_pooled_deployable": joint_deployable,
                     "joint_pareto_exists_fraction": pareto["joint_good_pareto_exists_fraction"]}
    if jr < 0.05:
        cases.append(schema.E2)
    elif not joint_deployable:
        cases.append(schema.E3)   # joint-good common + within-target-visible but pooled (deployable) transport broken

    # E4 vs E5 — is the source rank accuracy-specific or calibration-biased? RED-TEAM: "accuracy-specific" fires only
    # if the 9-target cluster-bootstrap acc-vs-calibration strength gap CI excludes 0 AND it is not purely
    # by-construction (the probe is trained on the accuracy label). At the primary margin neither fires:
    # endpoint-nonspecific-by-construction (only accuracy-vs-ECE is distinguishable).
    evid["E4_E5"] = {"score_accuracy_strength": source_rank["score_accuracy_strength"],
                     "score_calibration_strength": source_rank["score_calibration_strength"],
                     "score_joint_strength": source_rank["score_joint_strength"],
                     "ece_strength": source_rank["score_ece_strength"],
                     "accuracy_vs_calibration_gap_ci": source_rank["accuracy_vs_calibration_gap_ci"],
                     "accuracy_vs_ece_distinguishable": source_rank["accuracy_vs_ece_distinguishable"],
                     "accuracy_aligned_by_construction": source_rank["accuracy_aligned_by_construction"],
                     "verdict": ("E4_accuracy_specific" if source_rank["source_rank_accuracy_specific"] else
                                 "E5_calibration_biased" if source_rank["source_rank_calibration_biased"] else
                                 "endpoint_nonspecific_accuracy_aligned_by_construction")}
    if source_rank["source_rank_accuracy_specific"]:
        cases.append(schema.E4)
    elif source_rank["source_rank_calibration_biased"]:
        cases.append(schema.E5)

    # E6 vs E7 — is the gauge accuracy-specific or a general per-target offset?
    evid["E6_E7"] = {"metric_gauge_variance_fraction": gauge["metric_gauge_variance_fraction"],
                     "accuracy_gauge_gap": gauge["accuracy_gauge_gap"], "calibration_gauge_gap": gauge["calibration_gauge_gap"]}
    if gauge["gauge_accuracy_specific"]:
        cases.append(schema.E6)
    elif gauge["gauge_general_endpoint_offset"]:
        cases.append(schema.E7)

    # E8 — Pareto geometry explains the C16 barrier (accuracy-oracle systematically sacrifices calibration).
    # RED-TEAM: the headline OR-calibration fraction is definition-favorable; the strict both-metrics variant is
    # reported alongside, and the calibration-oracle symmetry is recorded (a real trade-off would wall at BOTH
    # oracles). E8 fires only if the accuracy-oracle is calibration-bad in a MAJORITY of trajectories.
    ao_bad = pareto["accuracy_oracle_calibration_bad_fraction"]
    ao_strict = pareto["accuracy_oracle_strict_calibration_bad_fraction"]
    e8 = bool(ao_bad is not None and ao_bad >= 0.5 and ao_strict is not None and ao_strict >= 0.5)
    evid["E8"] = {"accuracy_oracle_calibration_bad_fraction": ao_bad,
                  "accuracy_oracle_strict_calibration_bad_fraction": ao_strict,
                  "nll_oracle_accuracy_bad_fraction": pareto["nll_oracle_accuracy_bad_fraction"],
                  "ece_oracle_accuracy_bad_fraction": pareto["ece_oracle_accuracy_bad_fraction"],
                  "joint_good_pareto_exists_fraction": pareto["joint_good_pareto_exists_fraction"],
                  "mean_dominated_fraction": pareto["mean_dominated_fraction"]}
    if e8:
        cases.append(schema.E8)

    if not cases:
        cases.append(schema.E9)

    return {"cases": cases, "evidence": evid, "imbalance_flags": _imbalance_flags(base)}

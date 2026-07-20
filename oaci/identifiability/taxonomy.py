"""C17 — deterministic case taxonomy. Combines the univariate verdict, the multivariate leave-one-target-out
probe, and the axis decomposition into one of four pre-registered cases (I..IV). The multivariate LOTO probe
is the arbiter for weak-univariate cases: weak scalar signals only count as identifiability if a source-only
combination generalizes across held-out targets (beats the within-fold permutation)."""
from __future__ import annotations

from .schema import CASE_I, CASE_II, CASE_III, CASE_IV, CASE_INTERPRETATION


def case_taxonomy(univ, multi, axis) -> dict:
    strong = univ["n_strong_accuracy_signals"]
    weak = univ["n_weak_accuracy_signals"]
    nll = univ["n_signals_identify_nll"]
    beats = bool(multi["beats_permutation"])
    if strong >= 1:
        case = CASE_I                                     # a scalar source signal strongly ranks target accuracy
    elif beats:
        case = CASE_III                                   # combination recovers (weak) target-good info across targets
    elif nll >= 1:
        case = CASE_II                                    # calibration identifiable, accuracy not robustly
    elif weak >= 1:
        case = CASE_III                                   # weak scalar signals but no calibration axis -> weak III
    else:
        case = CASE_IV
    calib_bias = axis["source_signals_see_calibration_more_than_accuracy"]
    return {"case_label": case, "interpretation": CASE_INTERPRETATION[case],
            "inputs": {"univariate_verdict": univ["univariate_verdict"], "n_strong_accuracy_signals": strong,
                       "n_weak_accuracy_signals": weak, "n_signals_identify_nll": nll,
                       "oracle_signal_spearman_bacc": univ.get("oracle_signal_spearman_bacc"),
                       "max_abs_accuracy_spearman": univ.get("max_abs_accuracy_spearman"),
                       "accuracy_signal_families": univ.get("accuracy_signal_families"),
                       "multivariate_loto_auc": multi["loto_auc"], "multivariate_beats_permutation": beats,
                       "source_signals_calibration_biased": calib_bias},
            "next_science": {
                CASE_I: "pre-register a target-free competence detector (source signal exists, oracle used the wrong projection)",
                CASE_II: "study why competence lives on a source-invisible axis; target-free accuracy detection is NOT yet warranted",
                CASE_III: "a low-freedom source-only competence probe MAY be pre-registered; not deployable yet",
                CASE_IV: "target-good competence is invisible to the tested source evidence family; expand the source observable family or accept observability limit",
            }[case]}

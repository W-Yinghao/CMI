"""C17-D — calibration vs accuracy axis decomposition. Does source-side evidence mainly observe the
CALIBRATION axis (NLL/ECE) rather than the target-ACCURACY axis (bAcc)? Aggregates the univariate
within-fold-level correlations by signal axis and contrasts calibration-axis→target-NLL visibility against
accuracy-axis→target-bAcc visibility. This turns C16's `C3_calibration_not_discrimination` into a mechanism:
source signals track the calibration/softening axis; target accuracy lives on a source-fainter axis."""
from __future__ import annotations


def _mean_abs(xs):
    xs = [abs(x) for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def axis_decomposition(univ) -> dict:
    axes = {}
    for s, v in univ["per_signal"].items():
        ax = v["axis"]
        d = axes.setdefault(ax, {"signals": [], "rho_bacc": [], "rho_nll": []})
        d["signals"].append(s)
        d["rho_bacc"].append(v["mean_within_fold_spearman_bacc"])
        d["rho_nll"].append(v["mean_within_fold_spearman_nll"])
    by_axis = {ax: {"n": len(d["signals"]), "signals": d["signals"],
                    "mean_abs_rho_target_bacc": _mean_abs(d["rho_bacc"]),
                    "mean_abs_rho_target_nll": _mean_abs(d["rho_nll"])} for ax, d in axes.items()}
    calib_nll = by_axis.get("calibration", {}).get("mean_abs_rho_target_nll")
    acc_bacc = by_axis.get("accuracy", {}).get("mean_abs_rho_target_bacc")
    risk_bacc = by_axis.get("risk", {}).get("mean_abs_rho_target_bacc")
    calibration_more_visible = (calib_nll is not None and acc_bacc is not None and calib_nll > acc_bacc)
    return {"by_axis": by_axis, "calibration_axis_target_nll_visibility": calib_nll,
            "accuracy_axis_target_bacc_visibility": acc_bacc, "risk_axis_target_bacc_visibility": risk_bacc,
            "source_signals_see_calibration_more_than_accuracy": bool(calibration_more_visible),
            "note": ("If calibration-axis source signals track target NLL more than accuracy-axis source signals "
                     "track target bAcc, the source evidence is calibration-biased — explaining why the "
                     "source-audit oracle (an accuracy signal) failed while the selected OACI looked well-calibrated.")}

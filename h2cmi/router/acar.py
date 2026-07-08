"""Project B router — ACAR calibration-state / conformal-bound contract (Step-2C).

Action-Conditional Adaptation Risk (ACAR) state objects. This module:
  - NEVER trains an EEG model, runs a TTA, or fits a risk predictor;
  - ONLY consumes externally-supplied calibration records (true risks + PRE-COMPUTED risk
    predictions + action names) and turns them into split-conformal upper-risk bounds;
  - calibrates BOTH an absolute-error bound (eligibility-to-output) AND an adaptation-harm
    bound (allowed-to-adapt);
  - explicitly represents AVAILABLE / DEGENERATE / UNAVAILABLE per action per risk type.

Design note (v1, deliberate): ACAR does NOT fit the risk predictors. `pred_error` / `pred_harm`
are assumed already out-of-fold (or otherwise supplied by a future harness). This avoids using
the same pseudo-target records both to train and to calibrate a risk model — the leakage that
would invalidate the conformal guarantee. Step-2A established that source-only harm calibration
is frequently degenerate; ACAR therefore surfaces DEGENERATE/UNAVAILABLE rather than pretending
to produce a harm bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from h2cmi.router.actions import RouterAction, normalize_action
from h2cmi.router.features import CalibrationState, assess_acar_harm_calibration
from h2cmi.router.reasons import OACIReason, normalize_reasons


class ACARRiskType(str, Enum):
    ERROR = "error"    # absolute output risk, e.g. 1 - balanced_accuracy
    HARM = "harm"      # adaptation harm, e.g. bAcc_IDENTITY - bAcc_ACTION (>0 == action worse)


def _normalize_risk_type(rt: "str | ACARRiskType") -> ACARRiskType:
    if isinstance(rt, ACARRiskType):
        return rt
    if isinstance(rt, str):
        try:
            return ACARRiskType(rt)
        except ValueError:
            pass
        try:
            return ACARRiskType[rt]
        except KeyError:
            pass
    raise ValueError(f"unknown ACAR risk type: {rt!r}")


@dataclass(frozen=True)
class ACARConfig:
    alpha_error: float = 0.10
    alpha_harm: float = 0.10
    min_calibration_examples: int = 10
    harm_margin: float = 0.02
    risk_lower: float = 0.0
    risk_upper: float = 1.0
    finite_sample_strict: bool = True


# ------------------------------------------------------------------ conformal quantile
def conformal_quantile(
    scores: Sequence[float],
    *,
    alpha: float,
    finite_sample_strict: bool = True,
) -> "tuple[float | None, tuple[OACIReason, ...]]":
    """One-sided split-conformal upper residual quantile.

    Returns (qhat, reason_codes). qhat is None (with a reason) when the set is empty, non-finite,
    or too small for a finite-sample-valid order statistic under strict mode.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    arr = np.asarray(list(scores), dtype=float)
    n = int(arr.size)
    if n == 0:
        return None, (OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION,)
    if not bool(np.all(np.isfinite(arr))):
        return None, (OACIReason.OACI_DIAGNOSTIC_NONFINITE,)
    k = math.ceil((n + 1) * (1.0 - float(alpha)))
    if k > n:
        if finite_sample_strict:
            return None, (OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION,)
        k = n
    qhat = float(np.sort(arr)[k - 1])
    return qhat, ()


# ------------------------------------------------------------------ per-(action,risk) calibration
@dataclass(frozen=True)
class ACARRiskCalibration:
    action: RouterAction
    risk_type: ACARRiskType
    state: CalibrationState
    alpha: float
    n: int
    qhat: "float | None"
    residual_min: "float | None"
    residual_mean: "float | None"
    residual_max: "float | None"
    true_risk_min: "float | None"
    true_risk_mean: "float | None"
    true_risk_max: "float | None"
    reason_codes: tuple

    def upper_bound(
        self,
        predicted_risk: float,
        *,
        risk_lower: float = 0.0,
        risk_upper: float = 1.0,
    ) -> "float | None":
        """Calibrated upper risk bound = predicted_risk + qhat, clamped to [risk_lower, risk_upper].

        Returns None if this calibration is not AVAILABLE, qhat is None, or predicted_risk is
        non-finite (fail-loud: no bound rather than a fake one)."""
        if self.state != CalibrationState.AVAILABLE or self.qhat is None:
            return None
        try:
            pr = float(predicted_risk)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(pr):
            return None
        bound = pr + float(self.qhat)
        return float(min(max(bound, float(risk_lower)), float(risk_upper)))


def _stats(values) -> "tuple[float | None, float | None, float | None]":
    if values is None:
        return None, None, None
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return None, None, None
    return float(arr.min()), float(arr.mean()), float(arr.max())


def _make_cal(action, risk_type, state, alpha, n, qhat, residual, true_risk, reasons):
    rmin, rmean, rmax = _stats(residual)
    tmin, tmean, tmax = _stats(true_risk)
    return ACARRiskCalibration(
        action=action, risk_type=risk_type, state=state, alpha=float(alpha), n=int(n),
        qhat=(None if qhat is None else float(qhat)),
        residual_min=rmin, residual_mean=rmean, residual_max=rmax,
        true_risk_min=tmin, true_risk_mean=tmean, true_risk_max=tmax,
        reason_codes=normalize_reasons(reasons),
    )


def fit_risk_calibration(
    *,
    action: "str | RouterAction",
    risk_type: "ACARRiskType | str",
    true_risk: Sequence[float],
    predicted_risk: Sequence[float],
    config: ACARConfig,
    harm_gains: "Sequence[float] | None" = None,
) -> ACARRiskCalibration:
    """Calibrate a one-sided conformal upper bound for one (action, risk_type).

    ERROR risk is continuous — no class-diversity requirement. HARM risk additionally checks
    degeneracy (all-harm / all-non-harm) via harm_gains (preferred) or the sign of true_risk.
    Inputs are required finite; values outside [0,1] are allowed (the bound clamps later).
    """
    action = normalize_action(action)
    rt = _normalize_risk_type(risk_type)
    tr = np.asarray(list(true_risk), dtype=float)
    pr = np.asarray(list(predicted_risk), dtype=float)
    if tr.size != pr.size:
        raise ValueError(
            f"true_risk and predicted_risk length mismatch: {tr.size} vs {pr.size}")
    n = int(tr.size)
    alpha = config.alpha_error if rt == ACARRiskType.ERROR else config.alpha_harm

    # data integrity first
    if not (bool(np.all(np.isfinite(tr))) and bool(np.all(np.isfinite(pr)))):
        return _make_cal(action, rt, CalibrationState.UNAVAILABLE, alpha, n, None, None, None,
                         (OACIReason.OACI_DIAGNOSTIC_NONFINITE,))
    if n < config.min_calibration_examples:
        return _make_cal(action, rt, CalibrationState.UNAVAILABLE, alpha, n, None, None, tr,
                         (OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION,))

    # HARM: degeneracy detection (never pretend to calibrate a one-class harm set)
    if rt == ACARRiskType.HARM:
        if harm_gains is not None:
            summ = assess_acar_harm_calibration(
                harm_gains, min_examples=config.min_calibration_examples,
                harm_margin=config.harm_margin)
            if summ.state == CalibrationState.DEGENERATE:
                return _make_cal(action, rt, CalibrationState.DEGENERATE, alpha, n, None, None, tr,
                                 (OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE,))
            if summ.state == CalibrationState.UNAVAILABLE:
                return _make_cal(action, rt, CalibrationState.UNAVAILABLE, alpha, n, None, None, tr,
                                 summ.reason_codes)
            # AVAILABLE -> fall through to conformal fit
        else:
            harmed = tr >= config.harm_margin      # risk_harm >= margin == harmful
            if bool(harmed.all()) or bool((~harmed).all()):
                return _make_cal(action, rt, CalibrationState.DEGENERATE, alpha, n, None, None, tr,
                                 (OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE,))

    # one-sided conformal on non-negative residuals
    residual = np.maximum(0.0, tr - pr)
    qhat, qreasons = conformal_quantile(
        residual, alpha=alpha, finite_sample_strict=config.finite_sample_strict)
    if qhat is None:
        return _make_cal(action, rt, CalibrationState.UNAVAILABLE, alpha, n, None, residual, tr,
                         qreasons)
    return _make_cal(action, rt, CalibrationState.AVAILABLE, alpha, n, qhat, residual, tr,
                     (OACIReason.OACI_OK,))


# ------------------------------------------------------------------ per-action state
@dataclass(frozen=True)
class ACARActionState:
    action: RouterAction
    error: ACARRiskCalibration
    harm: "ACARRiskCalibration | None"
    reason_codes: tuple

    @property
    def error_available(self) -> bool:
        return self.error.state == CalibrationState.AVAILABLE

    @property
    def harm_available(self) -> bool:
        # IDENTITY has no adaptation harm by definition -> always "available" (== 0).
        if self.action == RouterAction.IDENTITY:
            return True
        return self.harm is not None and self.harm.state == CalibrationState.AVAILABLE

    def upper_bounds(
        self,
        *,
        predicted_error: float,
        predicted_harm: "float | None" = None,
        risk_lower: float = 0.0,
        risk_upper: float = 1.0,
    ) -> "dict[str, float | None]":
        out: "dict[str, float | None]" = {
            "error": self.error.upper_bound(predicted_error, risk_lower=risk_lower, risk_upper=risk_upper),
        }
        if self.action == RouterAction.IDENTITY:
            out["harm"] = 0.0
        elif self.harm is None or predicted_harm is None:
            out["harm"] = None
        else:
            out["harm"] = self.harm.upper_bound(predicted_harm, risk_lower=risk_lower, risk_upper=risk_upper)
        return out


# ------------------------------------------------------------------ full state
@dataclass(frozen=True)
class ACARState:
    actions: "dict[RouterAction, ACARActionState]"
    config: ACARConfig
    reason_codes: tuple

    def get(self, action: "str | RouterAction") -> ACARActionState:
        a = normalize_action(action)
        if a not in self.actions:
            raise KeyError(f"no ACAR state for action {a!r}")
        return self.actions[a]

    def available_actions_for_error(self) -> tuple:
        return tuple(a for a, st in self.actions.items() if st.error_available)

    def available_actions_for_harm(self) -> tuple:
        # IDENTITY is included by convention (harm not applicable == 0).
        return tuple(a for a, st in self.actions.items() if st.harm_available)


def _field(rec: Mapping[str, Any], key: str, action: RouterAction) -> float:
    if key not in rec:
        raise ValueError(f"calibration record for {action.value!r} missing required field {key!r}")
    return rec[key]


def fit_acar_state(
    records: Sequence[Mapping[str, Any]],
    *,
    config: "ACARConfig | None" = None,
    actions: "Sequence[str | RouterAction] | None" = None,
) -> ACARState:
    """Calibrate ACAR state from externally-supplied records.

    Record schema (per calibration example):
      action, true_error, pred_error, [true_harm, pred_harm for TTA actions], [gain optional].
    REFUSE is never calibrated (records or requested actions with REFUSE raise ValueError).
    No predictor is trained here — pred_* are assumed pre-computed / out-of-fold.
    """
    cfg = config if config is not None else ACARConfig()

    grouped: "dict[RouterAction, list]" = {}
    for rec in records:
        a = normalize_action(rec["action"])
        if a == RouterAction.REFUSE:
            raise ValueError("REFUSE must never be calibrated")
        grouped.setdefault(a, []).append(rec)

    if actions is None:
        action_list = sorted(grouped.keys(), key=lambda a: a.value)
    else:
        action_list = [normalize_action(a) for a in actions]
    if any(a == RouterAction.REFUSE for a in action_list):
        raise ValueError("REFUSE must never be calibrated")

    action_states: "dict[RouterAction, ACARActionState]" = {}
    all_reasons: list = []
    for action in action_list:
        recs = grouped.get(action, [])
        error_cal = fit_risk_calibration(
            action=action, risk_type=ACARRiskType.ERROR,
            true_risk=[_field(r, "true_error", action) for r in recs],
            predicted_risk=[_field(r, "pred_error", action) for r in recs],
            config=cfg,
        )
        if action == RouterAction.IDENTITY:
            harm_cal = None
        else:
            gains = [r.get("gain") for r in recs]
            gains = None if any(g is None for g in gains) else gains
            harm_cal = fit_risk_calibration(
                action=action, risk_type=ACARRiskType.HARM,
                true_risk=[_field(r, "true_harm", action) for r in recs],
                predicted_risk=[_field(r, "pred_harm", action) for r in recs],
                config=cfg, harm_gains=gains,
            )
        st_reasons = normalize_reasons(
            tuple(error_cal.reason_codes) + (tuple(harm_cal.reason_codes) if harm_cal else ()))
        action_states[action] = ACARActionState(action, error_cal, harm_cal, st_reasons)
        all_reasons.extend(st_reasons)

    return ACARState(actions=action_states, config=cfg, reason_codes=normalize_reasons(all_reasons))


if __name__ == "__main__":
    R = OACIReason
    cfg = ACARConfig()
    n = 12
    te = [0.30 + 0.02 * i for i in range(n)]        # true_error
    pe = [0.25 + 0.02 * i for i in range(n)]        # pred_error (residual ~0.05)
    gains_avail = [-0.10, -0.20, -0.05, -0.03, 0.05, 0.10, 0.0, 0.02, -0.15, 0.08, -0.04, 0.03]
    th = [-g for g in gains_avail]                   # true_harm == risk_harm == -gain
    ph = [x - 0.03 for x in th]                      # pred_harm
    gains_degen = [0.0] * n

    # 1. conformal quantile (non-strict) -> finite
    q, r = conformal_quantile([0.0, 0.1, 0.2, 0.3, 0.4], alpha=0.2, finite_sample_strict=False)
    assert q is not None and math.isfinite(q), (q, r)

    # 2. strict finite-sample insufficient
    q, r = conformal_quantile([0.0, 0.1, 0.2, 0.3], alpha=0.1, finite_sample_strict=True)
    assert q is None and R.OACI_ACAR_INSUFFICIENT_CALIBRATION in r
    # invalid alpha raises
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            conformal_quantile([0.1, 0.2], alpha=bad)
            raise AssertionError("bad alpha should raise")
        except ValueError:
            pass

    # 3. ERROR available + clamped upper bound
    ce = fit_risk_calibration(action="offline_tta", risk_type="error",
                              true_risk=te, predicted_risk=pe, config=cfg)
    assert ce.state is CalibrationState.AVAILABLE and ce.qhat is not None
    ub = ce.upper_bound(0.4, risk_lower=0.0, risk_upper=1.0)
    assert ub is not None and math.isfinite(ub) and 0.0 <= ub <= 1.0
    assert ce.upper_bound(float("nan")) is None
    # ERROR does NOT need class diversity: all-high errors still AVAILABLE
    ce_hi = fit_risk_calibration(action="identity", risk_type="error",
                                 true_risk=[0.9] * n, predicted_risk=[0.8] * n, config=cfg)
    assert ce_hi.state is CalibrationState.AVAILABLE

    # small n -> UNAVAILABLE + INSUFFICIENT
    ce_small = fit_risk_calibration(action="identity", risk_type="error",
                                    true_risk=te[:5], predicted_risk=pe[:5], config=cfg)
    assert ce_small.state is CalibrationState.UNAVAILABLE
    assert R.OACI_ACAR_INSUFFICIENT_CALIBRATION in ce_small.reason_codes

    # 4. HARM degenerate (all-zero gains)
    ch_d = fit_risk_calibration(action="offline_tta", risk_type="harm",
                                true_risk=th, predicted_risk=ph, config=cfg, harm_gains=gains_degen)
    assert ch_d.state is CalibrationState.DEGENERATE and ch_d.qhat is None
    assert R.OACI_ACAR_HARM_CALIBRATION_DEGENERATE in ch_d.reason_codes
    assert ch_d.upper_bound(0.1) is None
    # degeneracy also inferable from true_risk sign when gains absent
    ch_d2 = fit_risk_calibration(action="offline_tta", risk_type="harm",
                                 true_risk=[0.2] * n, predicted_risk=[0.1] * n, config=cfg)
    assert ch_d2.state is CalibrationState.DEGENERATE

    # 5. HARM available (mixed gains under margin)
    ch_a = fit_risk_calibration(action="offline_tta", risk_type="harm",
                                true_risk=th, predicted_risk=ph, config=cfg, harm_gains=gains_avail)
    assert ch_a.state is CalibrationState.AVAILABLE and ch_a.qhat is not None

    # 6. fit_acar_state with IDENTITY + OFFLINE_TTA
    records = [dict(action="identity", true_error=te[i], pred_error=pe[i]) for i in range(n)]
    records += [dict(action="offline_tta", true_error=te[i], pred_error=pe[i],
                     true_harm=th[i], pred_harm=ph[i], gain=gains_avail[i]) for i in range(n)]
    st = fit_acar_state(records, config=cfg)
    ident = st.get("identity")
    assert ident.error_available and ident.harm_available and ident.harm is None
    off = st.get(RouterAction.OFFLINE_TTA)
    assert off.error_available and off.harm_available
    ub_id = ident.upper_bounds(predicted_error=0.4)
    assert ub_id["harm"] == 0.0 and ub_id["error"] is not None
    ub_off = off.upper_bounds(predicted_error=0.4, predicted_harm=0.1)
    assert ub_off["error"] is not None and ub_off["harm"] is not None
    assert RouterAction.IDENTITY in st.available_actions_for_harm()
    assert RouterAction.IDENTITY in st.available_actions_for_error()

    # 7. length mismatch -> ValueError
    try:
        fit_risk_calibration(action="identity", risk_type="error",
                             true_risk=[0.1, 0.2], predicted_risk=[0.1], config=cfg)
        raise AssertionError("mismatch should raise")
    except ValueError:
        pass

    # 8. REFUSE record -> ValueError
    try:
        fit_acar_state([dict(action="refuse", true_error=0.1, pred_error=0.1)])
        raise AssertionError("REFUSE should raise")
    except ValueError:
        pass

    print("acar self-test passed")

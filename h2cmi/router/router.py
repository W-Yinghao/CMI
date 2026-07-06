"""Project B — RefusalFirstRouter policy layer (Step-2D).

Combines the Step-2B feature/reason contract and the Step-2C ACAR calibration contract into a
refusal-first routing decision. This layer:
  - does NOT train a model, run TTA, or touch the synthetic harness;
  - consumes per-action diagnostics (dict or RouterFeatureBundle), an optional ACARState, and
    optional externally-supplied risk predictions;
  - applies ACTION-SPECIFIC blockers: TTA-evidence / ACAR-harm failures block OFFLINE_TTA /
    ONLINE_TTA but NEVER an otherwise support-valid IDENTITY;
  - follows a safe-beneficial-then-identity selection policy (a beneficial TTA can win; otherwise
    a support-valid IDENTITY; otherwise REFUSE) — no least-interventional self-lock.

route_target(model, X_tgt, ...) is deliberately NOT implemented here; that belongs to the
Step-2E harness integration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from h2cmi.router.actions import (
    RouterAction,
    normalize_action,
    candidate_actions,
    action_priority,
)
from h2cmi.router.reasons import (
    OACIReason,
    TOS_CODES,
    TTA_CODES,
    DIAGNOSTIC_CODES,
    normalize_reasons,
)
from h2cmi.router.features import (
    RouterFeatureConfig,
    RouterFeatureBundle,
    build_router_features,
    CalibrationState,
)
from h2cmi.router.acar import ACARState


# ------------------------------------------------------------------ blocker sets
# Output blockers: block IDENTITY and every prediction action.
OUTPUT_BLOCKER_CODES = frozenset(
    set(TOS_CODES)            # all TOS/support failures
    | set(DIAGNOSTIC_CODES)  # missing / non-finite / internal
    | {
        OACIReason.OACI_PRIOR_DECOUPLING_FAILED,
        OACIReason.OACI_LEAKAGE_RESIDUAL_HIGH,
        OACIReason.OACI_ACAR_HIGH_ACTION_RISK,
    }
)

# TTA blockers: output blockers PLUS TTA-stability / gate-harm.
# (ACAR-harm unavailable/degenerate are added conditionally by the router — see tta_blockers.)
TTA_BLOCKER_CODES = frozenset(
    OUTPUT_BLOCKER_CODES
    | set(TTA_CODES)                       # unstable / negative-evidence / high-disagreement / identity-fallback
    | {OACIReason.OACI_GATE_HARM_RISK}
)


def output_blockers(reasons: Sequence[OACIReason]) -> tuple:
    return tuple(r for r in normalize_reasons(reasons) if r in OUTPUT_BLOCKER_CODES)


def tta_blockers(reasons: Sequence[OACIReason], *, block_on_acar_harm_unavailable: bool = False) -> tuple:
    blockset = set(TTA_BLOCKER_CODES)
    if block_on_acar_harm_unavailable:
        blockset |= {
            OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE,
            OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION,
        }
    return tuple(r for r in normalize_reasons(reasons) if r in blockset)


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class RouterConfig:
    feature_config: RouterFeatureConfig = field(default_factory=RouterFeatureConfig)

    error_budget: dict = field(default_factory=lambda: {
        "identity": 0.45, "offline_tta": 0.45, "online_tta": 0.45,
    })
    harm_budget: dict = field(default_factory=lambda: {
        "offline_tta": 0.02, "online_tta": 0.02,
    })
    min_expected_gain: float = 0.02

    # Project-B v1 posture: TOS/support is primary; ACAR-error is used when available.
    require_acar_error_for_output: bool = False
    # TTA is ACAR-harm-dependent unless explicitly allowed otherwise.
    require_acar_harm_for_tta: bool = True
    allow_tta_without_acar_harm: bool = False

    def error_budget_for(self, action: "str | RouterAction") -> float:
        a = normalize_action(action)
        if a.value not in self.error_budget:
            raise KeyError(f"no error budget for action {a.value!r}")
        return float(self.error_budget[a.value])

    def harm_budget_for(self, action: "str | RouterAction") -> float:
        a = normalize_action(action)
        if a.value not in self.harm_budget:
            raise KeyError(f"no harm budget for action {a.value!r}")
        return float(self.harm_budget[a.value])


# ------------------------------------------------------------------ risk predictions
def _opt_float(x: Any) -> "float | None":
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass(frozen=True)
class ActionRiskPrediction:
    predicted_error: "float | None" = None
    predicted_harm: "float | None" = None
    expected_gain: "float | None" = None


def normalize_risk_predictions(
    risk_predictions: "Mapping[str | RouterAction, Mapping[str, Any] | ActionRiskPrediction] | None"
) -> "dict[RouterAction, ActionRiskPrediction]":
    out: "dict[RouterAction, ActionRiskPrediction]" = {}
    if risk_predictions is None:
        return out
    for k, v in risk_predictions.items():
        a = normalize_action(k)
        if isinstance(v, ActionRiskPrediction):
            pe, ph, eg = v.predicted_error, v.predicted_harm, v.expected_gain
        elif isinstance(v, Mapping):
            pe, ph, eg = v.get("predicted_error"), v.get("predicted_harm"), v.get("expected_gain")
        else:
            raise ValueError(f"invalid risk prediction for {a.value!r}: {v!r}")
        pe, ph, eg = _opt_float(pe), _opt_float(ph), _opt_float(eg)
        if eg is None and ph is not None:            # expected_gain defaults to -predicted_harm
            eg = -ph
        out[a] = ActionRiskPrediction(predicted_error=pe, predicted_harm=ph, expected_gain=eg)
    return out


# ------------------------------------------------------------------ per-action evaluation / decision
@dataclass(frozen=True)
class ActionEvaluation:
    action: RouterAction
    admissible: bool
    reason_codes: tuple
    blocking_reason_codes: tuple
    upper_error: "float | None"
    upper_harm: "float | None"
    predicted_error: "float | None"
    predicted_harm: "float | None"
    expected_gain: "float | None"
    score: "float | None"


@dataclass(frozen=True)
class RouterDecision:
    action: RouterAction
    accepted: bool
    reason_codes: tuple
    diagnostics: dict
    action_scores: dict
    conformal_bounds: dict


class RefusalFirstRouter:
    def __init__(self, config: "RouterConfig | None" = None):
        self.config = config if config is not None else RouterConfig()

    # ---------------------------------------------------------- public entry
    def route_diagnostics(
        self,
        diagnostics_by_action: "Mapping[str | RouterAction, Mapping[str, Any] | RouterFeatureBundle]",
        *,
        mode: str = "offline",
        acar_state: "ACARState | None" = None,
        risk_predictions: "Mapping[str | RouterAction, Mapping[str, Any] | ActionRiskPrediction] | None" = None,
        acar_harm_gains: "Sequence[float] | None" = None,
    ) -> RouterDecision:
        cfg = self.config
        candidates = candidate_actions(mode)                     # raises ValueError on bad mode
        diag_map = self._normalize_action_mapping(diagnostics_by_action)
        risk_preds = normalize_risk_predictions(risk_predictions)

        evals: "dict[RouterAction, ActionEvaluation]" = {}
        bundles: "dict[RouterAction, RouterFeatureBundle | None]" = {}
        for action in candidates:
            ev, bundle = self._evaluate_action(
                action, diag_map.get(action), cfg, acar_state,
                risk_preds.get(action, ActionRiskPrediction()), acar_harm_gains)
            evals[action] = ev
            bundles[action] = bundle

        return self._select(candidates, evals, bundles, mode)

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _normalize_action_mapping(m) -> dict:
        out: dict = {}
        if m is None:
            return out
        for k, v in m.items():
            out[normalize_action(k)] = v
        return out

    def _evaluate_action(self, action, raw, cfg, acar_state, rp, acar_harm_gains):
        # missing diagnostics -> reason-coded, never silently reuse another action's
        if raw is None:
            ev = ActionEvaluation(
                action=action, admissible=False,
                reason_codes=(OACIReason.OACI_DIAGNOSTIC_MISSING,),
                blocking_reason_codes=(OACIReason.OACI_DIAGNOSTIC_MISSING,),
                upper_error=None, upper_harm=None, predicted_error=None,
                predicted_harm=None, expected_gain=None, score=None)
            return ev, None

        if isinstance(raw, RouterFeatureBundle):
            bundle = raw
        elif isinstance(raw, Mapping):
            bundle = build_router_features(raw, config=cfg.feature_config, acar_harm_gains=acar_harm_gains)
        else:
            ev = ActionEvaluation(
                action=action, admissible=False,
                reason_codes=(OACIReason.OACI_INTERNAL_ERROR,),
                blocking_reason_codes=(OACIReason.OACI_INTERNAL_ERROR,),
                upper_error=None, upper_harm=None, predicted_error=None,
                predicted_harm=None, expected_gain=None, score=None)
            return ev, None

        is_identity = action == RouterAction.IDENTITY
        is_tta = action.is_tta

        reasons: list = []
        # feature-bundle reasons; for IDENTITY drop ACAR-HARM codes (harm N/A for identity)
        for r in bundle.reason_codes:
            if is_identity and r in (OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE,
                                     OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION):
                continue
            reasons.append(r)

        predicted_error = rp.predicted_error
        predicted_harm = rp.predicted_harm
        expected_gain = rp.expected_gain

        # ACAR bounds from the (external) calibration state
        acar_action = None
        if acar_state is not None:
            try:
                acar_action = acar_state.get(action)
            except KeyError:
                acar_action = None

        upper_error = None
        upper_harm = 0.0 if is_identity else None
        if acar_action is not None:
            b = acar_action.upper_bounds(
                predicted_error=(predicted_error if predicted_error is not None else float("nan")),
                predicted_harm=predicted_harm, risk_lower=0.0, risk_upper=1.0)
            upper_error = b["error"]
            upper_harm = b["harm"]

        # --- error axis (eligibility to output) ---
        if upper_error is not None and upper_error > cfg.error_budget_for(action):
            reasons.append(OACIReason.OACI_ACAR_HIGH_ACTION_RISK)
        if cfg.require_acar_error_for_output and upper_error is None:
            reasons.append(OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION)

        # --- harm axis (allowed to adapt) — TTA only; identity harm == 0 by convention ---
        if is_tta:
            harm_available = acar_action is not None and acar_action.harm_available
            if not harm_available:
                reasons.append(self._harm_unavailable_reason(acar_action, bundle))
            if upper_harm is not None and upper_harm > cfg.harm_budget_for(action):
                reasons.append(OACIReason.OACI_ACAR_HIGH_ACTION_RISK)
            # expected-gain justification
            if expected_gain is None:
                if cfg.require_acar_harm_for_tta:
                    reasons.append(OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION)
            elif expected_gain < cfg.min_expected_gain:
                reasons.append(OACIReason.OACI_ACAR_HIGH_ACTION_RISK)

        reason_tuple = normalize_reasons(reasons)

        # action-specific blockers
        if is_identity:
            blockers = output_blockers(reason_tuple)
        else:
            block_acar = cfg.require_acar_harm_for_tta and not cfg.allow_tta_without_acar_harm
            blockers = tta_blockers(reason_tuple, block_on_acar_harm_unavailable=block_acar)
        admissible = len(blockers) == 0

        # audit score
        if is_identity:
            score = (-upper_error) if upper_error is not None else 0.0
        elif expected_gain is None:
            score = None
        else:
            score = expected_gain - max(0.0, upper_harm if upper_harm is not None else 0.0)

        ev = ActionEvaluation(
            action=action, admissible=admissible, reason_codes=reason_tuple,
            blocking_reason_codes=tuple(blockers), upper_error=upper_error, upper_harm=upper_harm,
            predicted_error=predicted_error, predicted_harm=predicted_harm,
            expected_gain=expected_gain, score=score)
        return ev, bundle

    @staticmethod
    def _harm_unavailable_reason(acar_action, bundle) -> OACIReason:
        if acar_action is not None and acar_action.harm is not None:
            hr = acar_action.harm.reason_codes
            if OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE in hr:
                return OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE
            if OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION in hr:
                return OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION
        if bundle is not None and bundle.acar_harm_calibration.state == CalibrationState.DEGENERATE:
            return OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE
        return OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION

    def _select(self, candidates, evals, bundles, mode) -> RouterDecision:
        tta_admissible = [a for a in candidates if a.is_tta and evals[a].admissible]
        any_tta_candidate = any(a.is_tta for a in candidates)

        if tta_admissible:
            def _key(a):
                eg = evals[a].expected_gain
                return (eg if eg is not None else float("-inf"), action_priority(a))
            chosen = max(tta_admissible, key=_key)
            action = chosen
            accepted = True
            decision_reasons = list(evals[chosen].reason_codes)
        elif RouterAction.IDENTITY in candidates and evals[RouterAction.IDENTITY].admissible:
            action = RouterAction.IDENTITY
            accepted = True
            decision_reasons = list(evals[RouterAction.IDENTITY].reason_codes)
            if any_tta_candidate:
                decision_reasons.append(OACIReason.OACI_TTA_IDENTITY_FALLBACK)
        else:
            action = RouterAction.REFUSE
            accepted = False
            decision_reasons = []
            for a in candidates:
                decision_reasons.extend(evals[a].blocking_reason_codes)
            decision_reasons.append(OACIReason.OACI_CONF_EMPTY_ACTION_SET)

        reason_codes = normalize_reasons(decision_reasons)

        action_scores: dict = {}
        conformal_bounds: dict = {}
        for a in candidates:
            ev = evals[a]
            bundle = bundles[a]
            action_scores[a.value] = dict(
                admissible=ev.admissible,
                reason_codes=[r.value for r in ev.reason_codes],
                blocking_reason_codes=[r.value for r in ev.blocking_reason_codes],
                predicted_error=ev.predicted_error,
                predicted_harm=ev.predicted_harm,
                expected_gain=ev.expected_gain,
                upper_error=ev.upper_error,
                upper_harm=ev.upper_harm,
                score=ev.score,
                tos_pass_feature_summary=(bool(bundle.tos_pass) if bundle is not None else None),
                prior_shift_only=(bool(bundle.prior_shift_only) if bundle is not None else None),
                cmi_residual_available=(bool(bundle.cmi_residual_available) if bundle is not None else None),
                acar_harm_calibration_state=(bundle.acar_harm_calibration.state.value
                                             if bundle is not None else None),
            )
            conformal_bounds[a.value] = {"error": ev.upper_error, "harm": ev.upper_harm}

        diagnostics = dict(
            mode=mode,
            candidates=[a.value for a in candidates],
            selected_action=action.value,
            accepted=bool(accepted),
            reason_codes=[r.value for r in reason_codes],
        )
        return RouterDecision(action=action, accepted=bool(accepted), reason_codes=reason_codes,
                              diagnostics=diagnostics, action_scores=action_scores,
                              conformal_bounds=conformal_bounds)


if __name__ == "__main__":
    from h2cmi.router.acar import ACARConfig, fit_acar_state
    R = OACIReason

    def clean(**over):
        d = dict(n_target=200.0, ess=48.0, delta_density_nll=0.10, transform_norm=2.0,
                 condition_number=1.5, prior_shift=0.05, pred_disagreement=0.08, ood_score=1.0,
                 cmi_residual=0.0, density_nll_source_prior=3.0, density_nll_target_prior=2.9,
                 min_class_responsibility=0.30)
        d.update(over)
        return d

    router = RefusalFirstRouter()

    # Case 1 — TTA negative evidence blocks TTA, not identity
    dec = router.route_diagnostics({"identity": clean(delta_density_nll=0.0),
                                    "offline_tta": clean(delta_density_nll=-0.2)}, mode="offline")
    assert dec.action == RouterAction.IDENTITY and dec.accepted
    assert R.OACI_TTA_NEGATIVE_EVIDENCE.value in dec.action_scores["offline_tta"]["reason_codes"]
    assert R.OACI_TTA_NEGATIVE_EVIDENCE not in dec.reason_codes                  # identity NOT blocked by it
    assert R.OACI_TTA_IDENTITY_FALLBACK in dec.reason_codes

    # Case 2 — low ESS refuses all
    dec = router.route_diagnostics({"identity": clean(ess=3.0),
                                    "offline_tta": clean(ess=3.0)}, mode="offline")
    assert dec.action == RouterAction.REFUSE and not dec.accepted
    assert R.OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE in dec.reason_codes
    assert R.OACI_CONF_EMPTY_ACTION_SET in dec.reason_codes

    # Case 3 — prior-shift-only identity accepted
    ps = clean(prior_shift=0.6, density_nll_source_prior=5.0, density_nll_target_prior=2.0)
    dec = router.route_diagnostics({"identity": ps, "offline_tta": ps}, mode="offline")
    assert dec.action == RouterAction.IDENTITY and dec.accepted
    assert R.OACI_PRIOR_SHIFT_ONLY_INFO in dec.reason_codes

    # Case 4 — ACAR harm degenerate blocks TTA, identity accepted
    dec = router.route_diagnostics({"identity": clean(), "offline_tta": clean()},
                                   mode="offline", acar_harm_gains=[0.0] * 20)
    assert dec.action == RouterAction.IDENTITY and dec.accepted
    assert R.OACI_ACAR_HARM_CALIBRATION_DEGENERATE.value in \
        dec.action_scores["offline_tta"]["reason_codes"]
    assert not dec.action_scores["offline_tta"]["admissible"]
    assert R.OACI_ACAR_HARM_CALIBRATION_DEGENERATE not in dec.reason_codes        # not an identity blocker

    # Case 5 — safe-beneficial ACAR -> OFFLINE_TTA selected
    n = 12
    te = [0.30 + 0.02 * i for i in range(n)]
    pe = [0.25 + 0.02 * i for i in range(n)]
    gains = [-0.10, -0.20, -0.05, -0.03, 0.05, 0.10, 0.0, 0.02, -0.15, 0.08, -0.04, 0.03]
    th = [-g for g in gains]
    ph = [x - 0.03 for x in th]
    records = [dict(action="identity", true_error=te[i], pred_error=pe[i]) for i in range(n)]
    records += [dict(action="offline_tta", true_error=te[i], pred_error=pe[i],
                     true_harm=th[i], pred_harm=ph[i], gain=gains[i]) for i in range(n)]
    st = fit_acar_state(records, config=ACARConfig())
    rp = {"identity": dict(predicted_error=0.35),
          "offline_tta": dict(predicted_error=0.30, predicted_harm=-0.10)}
    dec = router.route_diagnostics({"identity": clean(), "offline_tta": clean()},
                                   mode="offline", acar_state=st, risk_predictions=rp,
                                   acar_harm_gains=gains)
    assert dec.action == RouterAction.OFFLINE_TTA and dec.accepted, dec.diagnostics["reason_codes"]
    assert dec.conformal_bounds["offline_tta"]["error"] is not None
    assert dec.conformal_bounds["offline_tta"]["harm"] is not None

    # Case 6 — ACAR high error blocks output when required
    cfg6 = RouterConfig(require_acar_error_for_output=True)
    r6 = RefusalFirstRouter(cfg6)
    st6 = fit_acar_state([dict(action="identity", true_error=te[i], pred_error=pe[i]) for i in range(n)],
                         config=ACARConfig())
    dec = r6.route_diagnostics({"identity": clean()}, mode="identity", acar_state=st6,
                               risk_predictions={"identity": dict(predicted_error=0.9)})
    assert not dec.action_scores["identity"]["admissible"]
    assert R.OACI_ACAR_HIGH_ACTION_RISK.value in dec.action_scores["identity"]["reason_codes"]
    assert dec.action == RouterAction.REFUSE

    # Case 7 — invalid mode raises
    try:
        router.route_diagnostics({"identity": clean()}, mode="bad")
        raise AssertionError("bad mode should raise")
    except ValueError:
        pass

    print("router self-test passed")

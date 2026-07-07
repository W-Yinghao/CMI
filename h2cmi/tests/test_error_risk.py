"""Project B-Next (S2B) — metadata/unit test for the identity error-risk router integration.

No real EEG data. Verifies that a cross-fitted ErrorRiskFit plugs into the EXISTING
RefusalFirstRouter via require_acar_error_for_output + risk_predictions, that a high upper error blocks
IDENTITY while a low one allows it, and that an unavailable fit fabricates no ACARState.
"""
from __future__ import annotations

import numpy as np

from h2cmi.router.error_risk import (
    ErrorRiskConfig, fit_error_risk_crossfit, predict_error_risk, make_identity_error_acar_state,
)
from h2cmi.router.features import CalibrationState
from h2cmi.router.router import RefusalFirstRouter, RouterConfig
from h2cmi.router.actions import RouterAction
from h2cmi.router.reasons import OACIReason

FEATS = ("f_signal", "f_noise")


def _rows(n_groups, per, seed=0, err_scale=1.0):
    r = np.random.RandomState(seed)
    rows = []
    for g in range(n_groups):
        base = 0.15 + 0.6 * (g / max(1, n_groups - 1))
        for _ in range(per):
            sig = base + r.normal(0, 0.03)
            err = float(np.clip(err_scale * sig + r.normal(0, 0.02), 0, 1))
            rows.append(dict(cf_group=f"g{g}", identity_error=err,
                             f_signal=sig, f_noise=r.normal(0, 1.0)))
    return rows


def _identity_diag(**over):
    d = dict(n_target=200.0, ess=48.0, delta_density_nll=0.0, transform_norm=0.0,
             condition_number=1.0, prior_shift=0.05, pred_disagreement=0.0, ood_score=1.0,
             density_nll_source_prior=3.0, density_nll_target_prior=2.9, min_class_responsibility=0.30)
    d.update(over)
    return d


def main() -> None:
    R = OACIReason

    # 1-2. synthetic rows with 3+ groups -> AVAILABLE
    fit = fit_error_risk_crossfit(_rows(6, 4, seed=1), feature_names=FEATS, group_key="cf_group")
    assert fit.state is CalibrationState.AVAILABLE and fit.qhat is not None, fit.reason_codes

    # 3. ACARState from fit is accepted by RefusalFirstRouter with require_acar_error_for_output=True
    st = make_identity_error_acar_state(fit)
    assert st is not None
    router = RefusalFirstRouter(RouterConfig(require_acar_error_for_output=True))

    # 4. HIGH predicted/upper error blocks IDENTITY
    hi_pred = float(predict_error_risk(fit, [dict(f_signal=0.95, f_noise=0.0)])[0])
    dec_hi = router.route_diagnostics({"identity": _identity_diag()}, mode="identity",
                                      acar_state=st, risk_predictions={"identity": dict(predicted_error=hi_pred)})
    assert not dec_hi.action_scores["identity"]["admissible"], dec_hi.action_scores["identity"]
    assert R.OACI_ACAR_HIGH_ACTION_RISK.value in dec_hi.action_scores["identity"]["reason_codes"]
    assert dec_hi.action == RouterAction.REFUSE

    # 5. LOW predicted/upper error allows IDENTITY
    lo_pred = float(predict_error_risk(fit, [dict(f_signal=0.12, f_noise=0.0)])[0])
    dec_lo = router.route_diagnostics({"identity": _identity_diag()}, mode="identity",
                                      acar_state=st, risk_predictions={"identity": dict(predicted_error=lo_pred)})
    assert dec_lo.action == RouterAction.IDENTITY and dec_lo.accepted, dec_lo.diagnostics["reason_codes"]
    ub = dec_lo.conformal_bounds["identity"]["error"]
    assert ub is not None and ub <= router.config.error_budget_for("identity")

    # 6. unavailable fit -> no ACARState (no fabrication)
    fit_u = fit_error_risk_crossfit(_rows(2, 6, seed=2), feature_names=FEATS, group_key="cf_group")
    assert fit_u.state is CalibrationState.UNAVAILABLE
    assert make_identity_error_acar_state(fit_u) is None

    print("error_risk router-integration test passed:",
          f"qhat={fit.qhat:.3f} hi_pred={hi_pred:.3f} lo_pred={lo_pred:.3f}")


if __name__ == "__main__":
    main()

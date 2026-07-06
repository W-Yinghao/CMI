"""Project A — audited evaluation bridge.

Turns `h2cmi/eval/harness.py`-style outputs (`run_three_settings`: strict_dg / offline_tta /
online_tta, plus cross-fitted leakage) into Project A `Claim`s and a machine-checked
`ObservabilityReport`. The Step-6 discipline it enforces:

  * real EEG benchmark labels are allowed for EVALUATION, but they are NOT part of the R0/R1
    adaptation observation operator (06 §2) — so every target metric is emitted as
    ORACLE / evaluation-only (`oracle=True` → reportable, identifiable=False);
  * an offline-TTA target prior is admitted only under TU-1 (C1∧C2∧C3) — otherwise rejected;
  * leakage is emitted as a diagnostic (never a target-risk guarantee).

Nothing here trains a model or modifies the harness — it is a wrapper over the harness dicts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from .audit import build_report
from .schema import Claim, ContractID, Estimand, ObservabilityReport, Regime

_SOURCE_OBS = ("X_s", "Y_s", "D_s")               # R0 adaptation observation
_R1_OBS = ("X_s", "Y_s", "D_s", "X_T")            # R1 adds target-UNLABELED data
# oracle metrics ALSO consume held-out target labels for EVALUATION — declaring them keeps the
# integrity check a structural backstop (dropping oracle=True then hard-rejects the claim).
_SOURCE_OBS_ORACLE = _SOURCE_OBS + ("heldout_target_labels",)
_R1_OBS_ORACLE = _R1_OBS + ("heldout_target_labels",)


_PRIOR_KEYS = ("pi_T_hat", "target_prior", "per_domain_pi_T", "prior_estimates")


def _payload(d: Dict, keys) -> Optional[Dict]:
    """A JSON-ish sub-dict of the present keys (None if none present) — evaluation evidence only."""
    sub = {k: d[k] for k in keys if k in d}
    return sub or None


def _has_prior_payload(results: Dict) -> bool:
    """True iff the harness output actually carries a target-prior estimate."""
    return any(k in results for k in _PRIOR_KEYS)


def claims_for_strict_dg(metrics: Dict, *, has_oracle_target_labels: bool = True) -> List[Claim]:
    """strict-DG: source-only ADAPTATION (R0). Target bAcc is oracle/evaluation-only."""
    if not has_oracle_target_labels:
        return []                                 # no target labels -> no target metric exists
    claims: List[Claim] = []
    for key, name in (("balanced_acc", "strict_dg.target_bacc"),
                      ("worst_domain_bacc", "strict_dg.worst_domain_bacc")):
        if key in metrics:
            claims.append(Claim(name, Regime.R0, Estimand.BALANCED_ACCURACY,
                                observed=_SOURCE_OBS_ORACLE, estimator="strict-DG (source-trained)",
                                oracle=True, metric_payload={key: metrics[key]}))
    return claims


def claims_for_offline_tta(results: Dict, *, has_oracle_target_labels: bool = True,
                           prior_contracts: Optional[Set[ContractID]] = None,
                           prior_conclusion: bool = True) -> List[Claim]:
    """offline transductive TTA (R1, uses target X unlabeled). The measured adaptation gain is
    oracle/evaluation-only; a target-prior claim is emitted ONLY when the harness actually
    produced a prior estimate, and is identifiable only under TU-1 (C1∧C2∧C3).

    `prior_conclusion=False` marks the prior claim as a flagged demonstration (not a finalised
    conclusion) — for a pilot that reports the estimate WITHOUT asserting it is identified, so a
    (correctly) rejected undeclared-contract prior does not trip the forbidden-claim guard."""
    claims: List[Claim] = []
    if has_oracle_target_labels and ("delta_adapt" in results or "adapt" in results):
        gain = _payload(results, ("delta_adapt", "gain_bootstrap", "selective_risk",
                                  "per_domain_gain"))
        claims.append(Claim("offline_tta.adaptation_gain", Regime.R1, Estimand.TARGET_GAIN,
                            observed=_R1_OBS_ORACLE, estimator="offline TTA (EM transform + prior)",
                            oracle=True, metric_payload=gain))
    # a target-prior claim is admissible ONLY if a prior estimate exists; without the declared
    # C1∧C2∧C3 it is (correctly) rejected as an overclaim.
    if _has_prior_payload(results):
        claims.append(Claim("offline_tta.target_prior", Regime.R1, Estimand.TARGET_PRIOR,
                            observed=_R1_OBS, contracts=frozenset(prior_contracts or set()),
                            estimator="offline TTA EM prior", conclusion=prior_conclusion,
                            metric_payload=_payload(results, _PRIOR_KEYS)))
    return claims


def claims_for_online_tta(results: Dict, *, has_oracle_target_labels: bool = True) -> List[Claim]:
    """online streaming TTA (R1). Target bAcc is oracle/evaluation-only."""
    if not (has_oracle_target_labels and "balanced_acc" in results):
        return []
    return [Claim("online_tta.target_bacc", Regime.R1, Estimand.BALANCED_ACCURACY,
                  observed=_R1_OBS_ORACLE, estimator="online TTA (EMA prior)", oracle=True,
                  metric_payload={"balanced_acc": results["balanced_acc"]})]


def claims_for_leakage(leakage: Dict, regime: Regime = Regime.R0) -> List[Claim]:
    """cross-fitted signed leakage per DAG factor -> LEAKAGE diagnostics (never a risk guarantee)."""
    return [Claim(f"leakage.{factor}", regime, Estimand.LEAKAGE, observed=("Z_s",),
                  estimator="cross-fitted signed CMI + within-(Y,Pa) permutation null",
                  metric_payload=(leakage[factor] if isinstance(leakage[factor], dict)
                                  else {"value": leakage[factor]}))
            for factor in leakage]


def build_audited_eval_report(title: str, *, strict_dg: Optional[Dict] = None,
                              offline_tta: Optional[Dict] = None,
                              online_tta: Optional[Dict] = None,
                              leakage: Optional[Dict] = None,
                              prior_contracts: Optional[Set[ContractID]] = None,
                              prior_conclusion: bool = True,
                              has_oracle_target_labels: bool = True) -> ObservabilityReport:
    """Assemble a full audited evaluation report from harness-style output dicts."""
    claims: List[Claim] = []
    if strict_dg is not None:
        claims += claims_for_strict_dg(strict_dg, has_oracle_target_labels=has_oracle_target_labels)
    if offline_tta is not None:
        claims += claims_for_offline_tta(offline_tta,
                                         has_oracle_target_labels=has_oracle_target_labels,
                                         prior_contracts=prior_contracts,
                                         prior_conclusion=prior_conclusion)
    if online_tta is not None:
        claims += claims_for_online_tta(online_tta,
                                        has_oracle_target_labels=has_oracle_target_labels)
    if leakage is not None:
        claims += claims_for_leakage(leakage)
    return build_report(title, claims)

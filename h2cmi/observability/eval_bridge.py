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
                                oracle=True))
    return claims


def claims_for_offline_tta(results: Dict, *, has_oracle_target_labels: bool = True,
                           prior_contracts: Optional[Set[ContractID]] = None) -> List[Claim]:
    """offline transductive TTA (R1, uses target X unlabeled). The measured adaptation gain is
    oracle/evaluation-only; the EM's estimated target prior is admitted only under TU-1."""
    claims: List[Claim] = []
    if has_oracle_target_labels and ("delta_adapt" in results or "adapt" in results):
        claims.append(Claim("offline_tta.adaptation_gain", Regime.R1, Estimand.TARGET_GAIN,
                            observed=_R1_OBS_ORACLE, estimator="offline TTA (EM transform + prior)",
                            oracle=True))
    # the EM also estimates a target prior pi_T -> identifiable ONLY under TU-1 (C1∧C2∧C3);
    # without the declared contracts this claim is (correctly) rejected as an overclaim.
    claims.append(Claim("offline_tta.target_prior", Regime.R1, Estimand.TARGET_PRIOR,
                        observed=_R1_OBS, contracts=frozenset(prior_contracts or set()),
                        estimator="offline TTA EM prior"))
    return claims


def claims_for_online_tta(results: Dict, *, has_oracle_target_labels: bool = True) -> List[Claim]:
    """online streaming TTA (R1). Target bAcc is oracle/evaluation-only."""
    if not (has_oracle_target_labels and "balanced_acc" in results):
        return []
    return [Claim("online_tta.target_bacc", Regime.R1, Estimand.BALANCED_ACCURACY,
                  observed=_R1_OBS_ORACLE, estimator="online TTA (EMA prior)", oracle=True)]


def claims_for_leakage(leakage: Dict, regime: Regime = Regime.R0) -> List[Claim]:
    """cross-fitted signed leakage per DAG factor -> LEAKAGE diagnostics (never a risk guarantee)."""
    return [Claim(f"leakage.{factor}", regime, Estimand.LEAKAGE, observed=("Z_s",),
                  estimator="cross-fitted signed CMI + within-(Y,Pa) permutation null")
            for factor in leakage]


def build_audited_eval_report(title: str, *, strict_dg: Optional[Dict] = None,
                              offline_tta: Optional[Dict] = None,
                              online_tta: Optional[Dict] = None,
                              leakage: Optional[Dict] = None,
                              prior_contracts: Optional[Set[ContractID]] = None,
                              has_oracle_target_labels: bool = True) -> ObservabilityReport:
    """Assemble a full audited evaluation report from harness-style output dicts."""
    claims: List[Claim] = []
    if strict_dg is not None:
        claims += claims_for_strict_dg(strict_dg, has_oracle_target_labels=has_oracle_target_labels)
    if offline_tta is not None:
        claims += claims_for_offline_tta(offline_tta,
                                         has_oracle_target_labels=has_oracle_target_labels,
                                         prior_contracts=prior_contracts)
    if online_tta is not None:
        claims += claims_for_online_tta(online_tta,
                                        has_oracle_target_labels=has_oracle_target_labels)
    if leakage is not None:
        claims += claims_for_leakage(leakage)
    return build_report(title, claims)

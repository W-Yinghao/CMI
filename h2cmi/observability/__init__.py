"""Project A — executable observability audit layer.

Machine-checks the claim ledger of `notes/project_A_observability/08_experimental_protocol.md`
against the OACI identifiability rules (`06_…`), the contract taxonomy (`02_…`), and the
counterexample certificates (`07_…`). No experiment is admitted unless its (regime, contracts,
estimator) license its estimand under `OA-0`.

Quick use:
    from h2cmi.observability import Claim, Regime, Estimand, ContractID, check_claim_allowed
    v = check_claim_allowed(Claim("t", Regime.R1, Estimand.TARGET_PRIOR,
                                  contracts={ContractID.C1, ContractID.C2, ContractID.C3}))
    assert v.allowed and v.theorem == "TU-1"
"""
from __future__ import annotations

from .audit import (assert_forbidden_claims_not_made, attach_failure_certificates, build_report,
                    check_claim_allowed, split_checkable_uncheckable_contracts)
from .registry import CONTRACTS, FORBIDDEN_CLAIMS, THEOREMS, check_monotone_checkability
from .report import (REQUIRED_CLAIM_FIELDS, report_to_dict, write_observability_report_json,
                     write_observability_report_md)
from .schema import (Claim, ContractID, Estimand, ForbiddenClaimViolation, ObservabilityReport,
                     Regime, Verdict)

__all__ = [
    "Regime", "ContractID", "Estimand", "Claim", "Verdict", "ObservabilityReport",
    "ForbiddenClaimViolation",
    "check_claim_allowed", "build_report", "split_checkable_uncheckable_contracts",
    "attach_failure_certificates", "assert_forbidden_claims_not_made",
    "CONTRACTS", "THEOREMS", "FORBIDDEN_CLAIMS", "check_monotone_checkability",
    "report_to_dict", "write_observability_report_json", "write_observability_report_md",
    "REQUIRED_CLAIM_FIELDS",
]

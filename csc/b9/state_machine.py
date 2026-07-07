"""B9 state machine (development-only, NO scientific claim). Contract-FIRST: the Z/T-blind validator decides
CONTRACT_INVALID_OR_OUT_OF_ESTIMAND / INSUFFICIENT_LABELS_OR_SUPPORT BEFORE any p-value is computed. Only a VALID contract
reaches the exact randomization null. States are disjoint and fail-closed. Never emits NO_CONCEPT."""
import numpy as np
from csc.b9 import randomization_table as RT
from csc.b9 import exact_randomization_null as EN

STATES = ("B9_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE", "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",
          "INSUFFICIENT_LABELS_OR_SUPPORT", "SAMPLER_INVALID")


def b9_certify(Z, Y_design, C, subject, microblock, table, seed=0, n_boot=200, alpha=EN.ALPHA,
               min_support=RT.DEFAULT_MIN_SUPPORT_STRATA, natural_prevalence_requested=False, **null_kw):
    """Full B9 decision. Returns a dict with b9_state + contract diagnostics + (if the contract is valid) the exact-null
    test fields. CONTRACT-FIRST: the validator (Z/T-blind) runs before the test; an invalid/insufficient contract NEVER
    reaches the p-value. Fail-closed."""
    out = dict(b9_state=None, contract_valid=False, invalid_reasons=[], ran_test=False,
               observed_T=float("nan"), p_meanT=1.0, p_stud=1.0, n_eligible=0, n_exact_invalid=0)
    # --- CONTRACT VALIDATOR (Z/T-BLIND, BEFORE any test) ---
    refuse_state, cdiag = RT.check_contract(C, Y_design, subject, microblock, table,
                                            min_support=min_support, natural_prevalence_requested=natural_prevalence_requested)
    out.update(invalid_reasons=list(cdiag.get("invalid_reasons", [])),
               **{f"contract_{k}": v for k, v in cdiag.items() if k != "invalid_reasons"})
    if refuse_state is not None:
        out["b9_state"] = refuse_state  # CONTRACT_INVALID_OR_OUT_OF_ESTIMAND or INSUFFICIENT_LABELS_OR_SUPPORT
        return out
    out["contract_valid"] = True
    # --- EXACT RANDOMIZATION NULL (only on a VALID contract) ---
    r = EN.exact_null_test(Z, Y_design, C, subject, microblock, seed=seed, n_boot=n_boot, alpha=alpha, **null_kw)
    out.update(ran_test=bool(r.get("ran_test")), observed_T=float(r.get("observed_T", float("nan"))),
               observed_Tz=float(r.get("observed_Tz", float("nan"))) if r.get("observed_Tz") is not None else float("nan"),
               p_meanT=float(r.get("p_meanT", 1.0)), p_stud=float(r.get("p_stud", 1.0)),
               exact_null_mean_T=float(r.get("exact_null_mean_T", float("nan"))),
               exact_null_sd_T=float(r.get("exact_null_sd_T", float("nan"))),
               n_exact_invalid=int(r.get("n_exact_invalid", 0)), n_eligible=int(r.get("n_eligible", 0)))
    if not r.get("ran_test"):
        # contract valid but the cross-fit could not run (too few eligible / degenerate) -> not invalid, insufficient
        out["b9_state"] = "INSUFFICIENT_LABELS_OR_SUPPORT"; return out
    if not r.get("estimable"):
        out["b9_state"] = "SAMPLER_INVALID"; return out
    alert = bool(r.get("estimable") and r.get("p_meanT", 1.0) <= alpha and r.get("p_stud", 1.0) <= alpha and r.get("size_ok"))
    out["b9_state"] = "B9_CONCEPT_ALERT" if alert else "NO_ACTIONABLE_CONCEPT_EVIDENCE"
    return out

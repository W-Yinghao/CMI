"""B9.2 acquisition-stage adherence + attrition validator (development-only). Real acquisition records a SUBSET of the
pinned assignment table (attrition/dropout), so adherence must JOIN executed rows to the pinned table on trial_id (the
row-order-independent key the B9.0 round-2 red-team recommended for B9.1), NOT positional/full-length match. Whole-subject
or trial dropout that PRESERVES within-(subject,microblock,Y_design) balance -> REDUCED SUPPORT (valid); dropout that BREAKS
balance / induces a prior shift -> CONTRACT_INVALID_OR_OUT_OF_ESTIMAND. Reuses the byte-frozen B9.0 primitives + exact null;
does NOT modify the frozen B9.0 module. Z/T-blind, contract-first."""
import numpy as np
from csc.b9 import randomization_table as RT
from csc.b9 import exact_randomization_null as EN
from csc.mininfo import paired_calibrated as PC


def check_contract_acquisition(exec_rows, pinned_table, min_support=RT.DEFAULT_MIN_SUPPORT_STRATA, bal_tol=0, prior_tol=0.05):
    """Z/T-BLIND acquisition validator. exec_rows: dict(subject, microblock, trial_id, C, Y_design) = the RECORDED trials
    (subset of the pinned table). pinned_table: full make_assignment_table output + 'trial_id'. Returns (state_or_None, diag)
    -- None = valid. Uses ONLY (C, Y_design, subject, microblock, trial_id, table) -- NO Z, NO T, NO p-values."""
    reasons = []
    if pinned_table is None or "trial_id" not in pinned_table or pinned_table.get("manifest", {}).get("table_hash") is None:
        return "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND", dict(invalid_reasons=["missing_or_invalid_assignment_table"],
                                                           attrition_fraction=float("nan"), adherence=float("nan"),
                                                           n_support_strata=0, max_cxy_imbalance=-1, prior_shift=float("nan"))
    man = pinned_table["manifest"]
    if not (man.get("generated_before_recording") is True and man.get("Y_design_pre_assignment") is True):
        reasons.append("table_not_pre_recording_or_ydesign_post_hoc")
    if RT.table_hash(pinned_table["C"], pinned_table["Y_design"], pinned_table["subject"], pinned_table["microblock"]) != man.get("table_hash"):
        reasons.append("missing_or_invalid_assignment_table")
    # index the pinned table by trial_id
    tid = np.asarray(pinned_table["trial_id"])
    pin = {int(tid[i]): (int(pinned_table["C"][i]), int(pinned_table["Y_design"][i]),
                         int(pinned_table["subject"][i]), int(pinned_table["microblock"][i])) for i in range(len(tid))}
    eC, eYd, eS, eMB, eTID = (np.asarray(exec_rows["C"]), np.asarray(exec_rows["Y_design"]),
                              np.asarray(exec_rows["subject"]), np.asarray(exec_rows["microblock"]), np.asarray(exec_rows["trial_id"]))
    # ADHERENCE via trial_id JOIN: every recorded trial must match its pinned row exactly
    matched = 0; unknown = 0
    for i in range(len(eTID)):
        key = int(eTID[i])
        if key not in pin:
            unknown += 1; continue
        matched += (pin[key] == (int(eC[i]), int(eYd[i]), int(eS[i]), int(eMB[i])))
    adherence = float(matched / len(eTID)) if len(eTID) else 0.0
    if unknown > 0 or adherence < 1.0:
        reasons.append("executed_deviates_from_registered_table")
    attrition = float(1.0 - len(eTID) / len(tid)) if len(tid) else float("nan")
    # RECORDED-SUBSET balance / support / prior (attrition that breaks these invalidates; else it's reduced support)
    imb = RT._max_cxy_imbalance(eC, eYd, eS, eMB)
    if imb > bal_tol:
        reasons.append("cxy_design_imbalance")
    psh = RT._prior_shift(eC, eYd)
    if psh > prior_tol:
        reasons.append("attrition_or_noncompliance_prior_shift")
    nsup, locked_all = RT._support_strata(eC, eYd, eS, eMB)
    if locked_all:
        reasons.append("condition_not_randomized_or_locked")
    diag = dict(invalid_reasons=sorted(set(reasons)), attrition_fraction=float(attrition), adherence=float(adherence),
                n_support_strata=int(nsup), max_cxy_imbalance=int(imb), prior_shift=float(psh), n_recorded=int(len(eTID)))
    if reasons:
        return "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND", diag
    if nsup < min_support:
        return "INSUFFICIENT_LABELS_OR_SUPPORT", diag
    return None, diag


def b9_2_certify(Z, exec_rows, pinned_table, seed=0, n_boot=200, alpha=EN.ALPHA, min_confirm_pairs=20, **null_kw):
    """Contract-FIRST acquisition certify: trial_id-join adherence + attrition validator BEFORE any test; only a valid
    recorded subset reaches the byte-reused B9.0 exact randomization null. Z is aligned to exec_rows order. Fail-closed."""
    out = dict(b9_state=None, contract_valid=False, invalid_reasons=[], ran_test=False, observed_T=float("nan"),
               p_meanT=1.0, p_stud=1.0, n_eligible=0, attrition_fraction=float("nan"), adherence=float("nan"))
    state, cdiag = check_contract_acquisition(exec_rows, pinned_table)
    out.update(invalid_reasons=list(cdiag.get("invalid_reasons", [])),
               attrition_fraction=float(cdiag.get("attrition_fraction", float("nan"))),
               adherence=float(cdiag.get("adherence", float("nan"))),
               **{f"contract_{k}": v for k, v in cdiag.items() if k not in ("invalid_reasons", "attrition_fraction", "adherence")})
    if state is not None:
        out["b9_state"] = state; return out
    out["contract_valid"] = True
    r = EN.exact_null_test(Z, exec_rows["Y_design"], exec_rows["C"], exec_rows["subject"], exec_rows["microblock"],
                           seed=seed, n_boot=n_boot, alpha=alpha, min_confirm_pairs=min_confirm_pairs, **null_kw)
    out.update(ran_test=bool(r.get("ran_test")), observed_T=float(r.get("observed_T", float("nan"))),
               observed_Tz=float(r.get("observed_Tz", float("nan"))) if r.get("observed_Tz") is not None else float("nan"),
               p_meanT=float(r.get("p_meanT", 1.0)), p_stud=float(r.get("p_stud", 1.0)),
               n_exact_invalid=int(r.get("n_exact_invalid", 0)), n_eligible=int(r.get("n_eligible", 0)))
    if not r.get("ran_test"):
        out["b9_state"] = "INSUFFICIENT_LABELS_OR_SUPPORT"; return out
    if not r.get("estimable"):
        out["b9_state"] = "SAMPLER_INVALID"; return out
    alert = bool(r.get("estimable") and r.get("p_meanT", 1.0) <= alpha and r.get("p_stud", 1.0) <= alpha and r.get("size_ok"))
    out["b9_state"] = "B9_CONCEPT_ALERT" if alert else "NO_ACTIONABLE_CONCEPT_EVIDENCE"
    return out

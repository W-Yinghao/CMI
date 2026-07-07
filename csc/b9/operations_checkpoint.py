"""B9.2 operations-only CHECKPOINT (development-only). At an early 6-10 subject checkpoint the operator may inspect ONLY
logistics; the scientific endpoints (B9_ALERT, exact-null p-values, observed_T, T_z, injected-positive results,
condition-specific trends) are STRUCTURALLY BLINDED. checkpoint_report computes contract validity + support + sampler
FEASIBILITY (can the null run?) WITHOUT ever computing a p-value or an alert -- it never calls the exact-null p-value loop.
Use it to decide logistics-only pause/continue; NEVER to change the protocol based on signal."""
import numpy as np
from csc.b9 import randomization_table as RT
from csc.b9 import acquisition_adherence as AA   # trial_id-join validator (Z/T-blind; used for adherence on PARTIAL cohorts)
from csc.mininfo import paired_calibrated as PC

# fields the checkpoint is ALLOWED to expose; anything p-value/alert/T is FORBIDDEN and never produced here
ALLOWED_FIELDS = ("n_enrolled_subjects", "n_analyzable_subjects", "assignment_adherence", "attrition_fraction",
                  "per_cell_valid_min", "per_cell_valid_median", "n_support_strata", "sampler_feasible",
                  "n_eligible_subjects", "contract_state", "contract_invalid_reasons")
FORBIDDEN_FIELDS = ("b9_state_alert", "p_meanT", "p_stud", "observed_T", "observed_Tz", "exact_null_mean_T",
                    "injected_positive_result", "condition_trend")


def checkpoint_report(exec_rows, table, quality=None, min_epochs=PC.MIN_EPOCHS_PER_CONDITION, n_folds=PC.N_FOLDS):
    """OPERATIONS-ONLY report. exec_rows = dict(subject, microblock, trial_id, C, Y_design) of the RECORDED (possibly
    partial 6-10 subject) cohort. Returns ONLY logistics + contract/sampler FEASIBILITY. Does NOT compute or return any
    p-value, observed_T, T_z, alert, or injected-positive result (those keys are never created). Z is NOT used."""
    C, Y_design, subject, microblock = (np.asarray(exec_rows["C"]), np.asarray(exec_rows["Y_design"]),
                                        np.asarray(exec_rows["subject"]), np.asarray(exec_rows["microblock"]))
    q = quality or {}
    # contract validity + adherence via the trial_id-JOIN validator (Z/T-blind, p-free) -- informative on PARTIAL cohorts
    # (the frozen positional RT.check_contract would read NaN / always-INVALID on a partial enrollment)
    state, cdiag = AA.check_contract_acquisition(exec_rows, table)
    n_enrolled = int(len(np.unique(subject)))
    # analyzable = subjects meeting the PREDECLARED support criterion (both conditions >= min_epochs); NOT p-based
    elig = PC.eligible_complete_pairs(C, subject, min_epochs)
    n_analyzable = int(len(elig))
    adherence = float(cdiag.get("adherence", float("nan")))
    attrition = float(cdiag.get("attrition_fraction", float("nan")))
    cells = []
    for s in np.unique(subject):
        for mb in np.unique(microblock[subject == s]):
            for c in (RT.LO, RT.HI):
                for y in (0, 1):
                    cells.append(int(((subject == s) & (microblock == mb) & (C == c) & (Y_design == y)).sum()))
    per_min = int(min(cells)) if cells else 0
    per_med = float(np.median(cells)) if cells else 0.0
    # sampler FEASIBILITY: can the exact null RUN? (enough eligible + fold prep not degenerate) -- WITHOUT computing p
    sampler_feasible = False
    if n_analyzable >= n_folds * 2 and len(np.unique(Y_design)) >= 2 and len(np.unique(C)) == 2:
        m = np.isin(subject, elig)
        folds, _ = PC._make_folds(elig, n_folds, 0)
        # _prep_folds needs a Z; feasibility of the design matrix is Z-shape-agnostic, so use a zero Z placeholder (NO signal)
        Zdummy = np.zeros((int(m.sum()), 4))
        prep = PC._prep_folds(Zdummy, C[m], subject[m], folds, "centered", 3, 0.5)
        sampler_feasible = prep is not None
    report = dict(n_enrolled_subjects=n_enrolled, n_analyzable_subjects=n_analyzable,
                  assignment_adherence=adherence, attrition_fraction=attrition,
                  per_cell_valid_min=per_min, per_cell_valid_median=per_med,
                  n_support_strata=int(cdiag.get("n_support_strata", 0)), sampler_feasible=bool(sampler_feasible),
                  n_eligible_subjects=n_analyzable, contract_state=("VALID" if state is None else str(state)),
                  contract_invalid_reasons=list(cdiag.get("invalid_reasons", [])),
                  BLIND_NOTE="operations-only: scientific endpoints (alert / p-values / observed_T / T_z / injected-positive) are NOT computed here")
    assert not (set(report) & set(FORBIDDEN_FIELDS)), "checkpoint leaked a forbidden scientific endpoint"
    return report

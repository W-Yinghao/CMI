"""B9 prospective randomized-audit assignment table + Z/T-BLIND contract validator (development-only, NO scientific claim).

The assignment table is generated + hash-pinned BEFORE recording. It fixes, per (subject, microblock), a balanced set of
(condition C, design_class Y_design) cells -- the recommended quadruplet {(0,0),(0,1),(1,0),(1,1)} x R in randomized order.
Y_design is a PRE-ASSIGNMENT cue (NOT an observed/generated post-hoc label). The validator checks whether an EXECUTED
dataset adheres to a valid contract, using ONLY (C, Y_design, subject, microblock, table) -- it NEVER reads Z or T.
"""
import hashlib
import numpy as np

LO, HI = 0, 1
DEFAULT_MIN_SUPPORT_STRATA = 8
INVALID_REASONS = ("missing_or_invalid_assignment_table", "table_not_pre_recording_or_ydesign_post_hoc",
                   "executed_deviates_from_registered_table", "cxy_design_imbalance",
                   "condition_not_randomized_or_locked", "session_confounding",
                   "attrition_or_noncompliance_prior_shift", "insufficient_randomization_support",
                   "natural_prevalence_out_of_estimand")


def make_assignment_table(subjects, n_microblocks, R, seed):
    """Generate a valid PRE-RECORDING assignment table: per (subject, microblock), R randomized-order quadruplets of
    (C, Y_design). Returns arrays (subject, microblock, C, Y_design) + a manifest dict (generated_before_recording=True,
    seed, table_hash). C x Y_design is EXACTLY balanced within each (subject, microblock) by construction."""
    rng = np.random.default_rng(seed)
    quad = np.array([[LO, 0], [LO, 1], [HI, 0], [HI, 1]], int)   # (C, Y_design)
    S, MB, C, Yd = [], [], [], []
    for s in subjects:
        for mb in range(n_microblocks):
            cells = np.repeat(quad, R, axis=0)                    # R of each of the 4 cells
            order = rng.permutation(len(cells))                   # randomized order within the microblock
            cells = cells[order]
            S += [int(s)] * len(cells); MB += [int(mb)] * len(cells)
            C += list(cells[:, 0]); Yd += list(cells[:, 1])
    S, MB, C, Yd = map(lambda a: np.asarray(a, int), (S, MB, C, Yd))
    h = table_hash(C, Yd, S, MB)
    manifest = dict(schema="b9_assignment_contract/v0", generated_before_recording=True, Y_design_pre_assignment=True,
                    C_randomized_within_block=True, seed=int(seed), unit="subject x microblock x design_class",
                    R=int(R), n_microblocks=int(n_microblocks), n_subjects=len(subjects), table_hash=h)
    return dict(subject=S, microblock=MB, C=C, Y_design=Yd, manifest=manifest)


def table_hash(C, Y_design, subject, microblock):
    """Order-invariant hash of the per-(subject,microblock,Y_design) C-composition (HI/LO counts)."""
    C, Y_design, subject, microblock = map(np.asarray, (C, Y_design, subject, microblock))
    rows = []
    for s in sorted(np.unique(subject)):
        for mb in sorted(np.unique(microblock[subject == s])):
            for y in sorted(np.unique(Y_design[(subject == s) & (microblock == mb)])):
                m = (subject == s) & (microblock == mb) & (Y_design == y)
                rows.append((int(s), int(mb), int(y), int((C[m] == HI).sum()), int((C[m] == LO).sum())))
    return hashlib.sha1(str(sorted(rows)).encode()).hexdigest()[:16]


def _support_strata(C, Y_design, subject, microblock):
    """# of (subject, microblock, Y_design) strata that contain BOTH conditions (randomization support)."""
    C, Y_design, subject, microblock = map(np.asarray, (C, Y_design, subject, microblock))
    n = 0; locked_all = True
    for s in np.unique(subject):
        for mb in np.unique(microblock[subject == s]):
            for y in np.unique(Y_design[(subject == s) & (microblock == mb)]):
                m = (subject == s) & (microblock == mb) & (Y_design == y)
                if (C[m] == HI).sum() >= 1 and (C[m] == LO).sum() >= 1:
                    n += 1; locked_all = False
    return n, locked_all


def _max_cxy_imbalance(C, Y_design, subject, microblock):
    """Max |#(C=HI) - #(C=LO)| over (subject, microblock, Y_design) strata that CONTAIN BOTH conditions -- 0 for an
    exactly balanced contract. Single-condition strata are left to the condition-lock/support check so the reason codes
    stay disjoint (a locked stratum is 'condition_not_randomized_or_locked', not spuriously 'cxy_design_imbalance')."""
    C, Y_design, subject, microblock = map(np.asarray, (C, Y_design, subject, microblock))
    worst = 0
    for s in np.unique(subject):
        for mb in np.unique(microblock[subject == s]):
            for y in np.unique(Y_design[(subject == s) & (microblock == mb)]):
                m = (subject == s) & (microblock == mb) & (Y_design == y)
                nhi = int((C[m] == HI).sum()); nlo = int((C[m] == LO).sum())
                if nhi >= 1 and nlo >= 1:                      # only strata with BOTH conditions count as "imbalanced"
                    worst = max(worst, abs(nhi - nlo))
    return int(worst)


def _prior_shift(C, Y_design):
    """|P(Y_design=1 | C=HI) - P(Y_design=1 | C=LO)| -- a prior shift means the contract balance was NOT met (attrition/
    noncompliance). Under a valid balanced contract this is ~0."""
    C, Y_design = np.asarray(C), np.asarray(Y_design)
    if (C == HI).sum() == 0 or (C == LO).sum() == 0:
        return 1.0
    return float(abs(Y_design[C == HI].mean() - Y_design[C == LO].mean()))


def check_contract(C, Y_design, subject, microblock, table, min_support=DEFAULT_MIN_SUPPORT_STRATA,
                   bal_tol=0, prior_tol=0.05, natural_prevalence_requested=False):
    """Z/T-BLIND contract validator (PREDECLARED, before any test). Returns (state_or_None, diag). Returns None (valid)
    only if ALL hard checks pass; else the refuse-before-p state string + reason. Uses ONLY (C, Y_design, subject,
    microblock, table) -- NO Z, NO T, NO p-values.
      H0 estimand: natural-prevalence request -> OUT_OF_ESTIMAND.
      H1 a valid assignment table exists + its hash matches the pin.
      H1b PROVENANCE ATTESTATION: manifest.generated_before_recording AND manifest.Y_design_pre_assignment must be True
          (the B9-vs-B8 differentiator). Data-level provenance (was Y_design really pre-assignment, was the table really
          generated before recording) is INHERENTLY UNVERIFIABLE from (C, Y_design, Z); this boolean manifest attestation
          is the enforceable floor, and the real guarantee is the B9.1 acquisition protocol.
      H2 ADHERENCE: the EXECUTED full pre-registered tuple (C, Y_design, subject, microblock) must FOLLOW the registered
          table row-for-row (equal length + elementwise match) -- binds BOTH randomized factors (an analyst may relabel
          neither C NOR Y_design; the null holds Y_design fixed). Positional row-alignment; B9.1 joins on a per-trial id.
      H3 C x Y_design balance within (subject,microblock) <= bal_tol (both-condition strata only).
      H4 no prior shift P(Y_design|C) (<= prior_tol) -- attrition/noncompliance invalidates.
      H5 condition randomized (not locked to whole (subject,microblock)/session) AND randomization support >= min_support.
    Distinguishes CONTRACT_INVALID_OR_OUT_OF_ESTIMAND (structural) from INSUFFICIENT_LABELS_OR_SUPPORT (valid but too small)."""
    reasons = []
    if natural_prevalence_requested:
        reasons.append("natural_prevalence_out_of_estimand")
    # H1 table exists + hash integrity
    if table is None or "C" not in table or table.get("manifest", {}).get("table_hash") is None:
        return "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND", dict(invalid_reasons=sorted(set(reasons + ["missing_or_invalid_assignment_table"])),
                                                           n_support_strata=0, max_cxy_imbalance=-1, prior_shift=float("nan"), adherence=float("nan"))
    man = table.get("manifest", {})
    tC = np.asarray(table["C"]); pinned = man.get("table_hash")
    hash_ok = (table_hash(tC, table["Y_design"], table["subject"], table["microblock"]) == pinned)
    if not hash_ok:
        reasons.append("missing_or_invalid_assignment_table")
    # H1b PROVENANCE ATTESTATION (the B9-vs-B8 differentiator; enforceable floor for what is otherwise unverifiable)
    if not (man.get("generated_before_recording") is True and man.get("Y_design_pre_assignment") is True):
        reasons.append("table_not_pre_recording_or_ydesign_post_hoc")
    # H2 ADHERENCE: the executed FULL pre-registered tuple (C, Y_design, subject, microblock) must FOLLOW the registered
    # table row-for-row (anti-p-hacking on BOTH randomized factors -- an analyst may post-hoc relabel neither C NOR Y_design;
    # the exact null holds Y_design fixed, so a Y_design relabel is the same class of hole as a C relabel). ROW-ORDER
    # CONVENTION: the executed inventory and the pinned table MUST be delivered in the same (subject, microblock, trial)
    # row order (positional alignment); B9.1 should instead join on a registered per-trial id. Aligned by index here.
    C = np.asarray(C); Y_design = np.asarray(Y_design); subject = np.asarray(subject); microblock = np.asarray(microblock)
    tYd, tS, tMB = np.asarray(table["Y_design"]), np.asarray(table["subject"]), np.asarray(table["microblock"])
    aligned = (len(C) == len(tC) == len(tYd) == len(tS) == len(tMB))
    full_match = bool(aligned and np.all(C == tC) and np.all(Y_design == tYd)
                      and np.all(subject == tS) and np.all(microblock == tMB))
    adherence = float(np.mean((C == tC) & (Y_design == tYd) & (subject == tS) & (microblock == tMB))) if aligned else 0.0
    if not full_match:
        reasons.append("executed_deviates_from_registered_table")
    # H3 balance
    imb = _max_cxy_imbalance(C, Y_design, subject, microblock)
    if imb > bal_tol:
        reasons.append("cxy_design_imbalance")
    # H4 prior shift (attrition/noncompliance)
    psh = _prior_shift(C, Y_design)
    if psh > prior_tol:
        reasons.append("attrition_or_noncompliance_prior_shift")
    # H5 randomization support + condition-lock / session confound
    nsup, locked_all = _support_strata(C, Y_design, subject, microblock)
    # session confound (most specific): C is constant within each SUBJECT -> a subject/session label, no within-subject
    # randomization. condition-lock (weaker): C varies within subject but is constant within each (subject,microblock,Y)
    # stratum -> a per-microblock label, no within-block randomization support anywhere.
    subj_lock = all(len(np.unique(C[subject == s])) < 2 for s in np.unique(subject))
    if subj_lock:
        reasons.append("session_confounding")
    elif locked_all:
        reasons.append("condition_not_randomized_or_locked")
    diag = dict(invalid_reasons=sorted(set(reasons)), n_support_strata=int(nsup), max_cxy_imbalance=int(imb),
                prior_shift=float(psh), adherence=float(adherence), hash_ok=bool(hash_ok))
    if reasons:
        return "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND", diag
    if nsup < min_support:
        # structurally valid (adhered, balanced, randomized) but too little randomization support -> not invalid, just insufficient
        return "INSUFFICIENT_LABELS_OR_SUPPORT", diag
    return None, diag

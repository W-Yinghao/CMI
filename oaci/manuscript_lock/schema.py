"""C21 — Estimand Boundary + Manuscript Lock. Freezes the C14->C20 science chain into a final paper object.
NO new experiments. The core lesson is an ESTIMAND BOUNDARY: C19's weak robust-core competence signal is real
IN-REGIME, but C20 shows it does not transport as a stable cross-regime POOLED estimand -- within-target
ranking information persists (~0.64) while the pooled cross-target/cross-regime scoring does not. The probe is
DIAGNOSTIC-ONLY; no detector / no selector claim is permitted.

Every ledger claim carries an evidence pointer (committed report + commit + key number) so the lock is
machine-auditable. The within-target ~0.64 is FUTURE-WORK / mechanism ONLY -- it may NEVER retroactively
replace C20's pre-registered pooled estimand.
"""
from __future__ import annotations

DIAGNOSTIC_ONLY = True

PAPER_TITLE = ("When Source-Side Signals Do Not Transfer: Falsification, Observability, and Estimand "
               "Boundaries in EEG Domain Generalization")

# ---- estimand reconciliation (the crux) ------------------------------------------------------------
ESTIMAND_RECONCILIATION = {
    "c19_pooled_loto_auc_in_regime": 0.561,       # C19: passes pre-registered in-regime criteria (p=0.005, margin>=0.03)
    "c20_pooled_loto_auc_cross_regime_range": [0.500, 0.536],   # C20: near chance; external validation NOT established
    "within_target_mean_auc_range": [0.632, 0.652],             # persists across ALL held-out regimes (C20 robustness)
    "c19_per_target_auc_range": [0.57, 0.73],
    "resolution": ("The probe contains within-target RANKING information, but its score is not calibrated as a "
                   "stable pooled cross-target / cross-regime competence estimand. Competence information exists; "
                   "the estimand does not transport."),
    "why_c19_stands": ("C19's pooled 0.561 is CONSERVATIVE relative to its per-target ~0.64 (pooling misaligns "
                       "cross-target offsets and lowers the AUC), so C19's in-regime positive is if anything "
                       "under-stated by pooling -- it stands."),
    "forbidden_reading": ("C20 did NOT succeed under a better estimand. within-target ~0.64 is a mechanism / "
                          "future-work observation, NOT a retroactive replacement of the pre-registered pooled estimand."),
}

# ---- claim ledger: id, text, category, evidence (report + commit + key), key_number -----------------
# categories: established | diagnostic_only | not_established | future_work
CLAIMS = (
    # ---- established ----
    {"id": "E1", "category": "established", "commit": "8046929/7b38bee",
     "report": "C14_DG_FALSIFICATION_BATTERY / C8 / C12",
     "text": "The tested source-only DG control objectives (OACI-control, SRC-control) do NOT transfer to "
             "reproducible target worst-domain improvements under strict target isolation.",
     "key": "C8 K1/K2 stop; C14 verdict falsified; C12 SRC anti-transfer"},
    {"id": "E2", "category": "established", "commit": "38206d6",
     "report": "C16_TARGET_ORACLE_CEILING",
     "text": "Target-accuracy-good OACI checkpoints EXIST but are invisible to simple source-audit selectors "
             "(source-side OBSERVABILITY failure), with a separate calibration barrier.",
     "key": "6/6 seed-levels reproducible bAcc gain via non-deployable target oracle; source-audit oracle fails"},
    {"id": "E3", "category": "established", "commit": "0eedfee",
     "report": "C16_HARM_DECOMPOSITION",
     "text": "Selected OACI is calibration-improved / accuracy-flat (class-boundary rotation); SRC anti-transfer "
             "is source memorization.",
     "key": "mean ΔNLL -0.074, ΔbAcc -0.002; SRC memorization index +1.965 (6/6 flagged)"},
    {"id": "E4", "category": "established", "commit": "a8af8c6",
     "report": "C17_SOURCE_SIGNAL_IDENTIFIABILITY",
     "text": "No strong scalar source signal identifies target-good checkpoints; a WEAK multivariate source-only "
             "signal exists; class-boundary rotation is source-mirrored; source signals are calibration-biased.",
     "key": "LOTO probe AUC 0.6023 beats perm p=0.008; best scalar |ρ|<=0.236; boundary corr +0.547"},
    {"id": "E5", "category": "established", "commit": "8046929",
     "report": "C18_CONTROLLED_SUPPORT_MISMATCH_STRESS",
     "text": "The weak multivariate signal SURVIVES cell-present support stress and collapses only under cell "
             "DELETION, and there because the worst-domain accuracy ENDPOINT becomes non-estimable (estimator-"
             "level), not because the signal vanished; leakage and class-boundary mirror are support-robust.",
     "key": "S2 0.603 / S3 0.562 beat perm; S4/S6/S7 bAcc->NaN; leakage estimability 1.0"},
    {"id": "E6", "category": "established", "commit": "0eebae5",
     "report": "C19_SOURCE_ONLY_COMPETENCE_PROBE",
     "text": "A PRE-REGISTERED low-freedom robust-core probe (deletion-robust source observables) recovers weak "
             "IN-REGIME competence information (config hash frozen before the run).",
     "key": "robust-core LOTO 0.561 beats perm p=0.005 margin>=0.03 on S0/S2/S3; per-target 0.57-0.73"},
    {"id": "E7", "category": "established", "commit": "7b38bee",
     "report": "C20_FROZEN_PROBE_NEW_REGIME_VALIDATION",
     "text": "Frozen cross-regime external/new-regime validation is NOT established: the C19 signal is largely "
             "REGIME-LOCAL. Apparent held-out passes do not survive robust interpretation.",
     "key": "cross-regime pooled AUC 0.50-0.54; Holm 2/4->0/4; Simpson-decoupled; S7 control > S6 treatment"},
    # ---- diagnostic-only ----
    {"id": "D1", "category": "diagnostic_only", "commit": "0eebae5/7b38bee", "report": "C19/C20",
     "text": "The C19/C20 competence probe is DIAGNOSTIC-ONLY: it emits no selector and makes no deployment claim.",
     "key": "no_selector_artifact gate = True in C19 and C20"},
    {"id": "D2", "category": "diagnostic_only", "commit": "38206d6", "report": "C16/C17",
     "text": "Target-oracle labels and source-signal identifiability analyses are diagnostic-only, joined post hoc.",
     "key": "diagnostic_only_non_deployable=True throughout"},
    # ---- not established ----
    {"id": "N1", "category": "not_established", "commit": "7b38bee", "report": "C19/C20",
     "text": "No deployable target-free selector / competence detector is established (diagnostic-only; no-selector gate).",
     "key": "no-selector gate; C20 external validation not established"},
    {"id": "N2", "category": "not_established", "commit": "7b38bee", "report": "C20",
     "text": "External / new-regime dataset generalization of the probe is NOT established.",
     "key": "cross-regime pooled near chance; Holm 0/4"},
    {"id": "N3", "category": "not_established", "commit": "a8af8c6", "report": "C17/C18",
     "text": "Support mismatch as a NATURAL cause in BNCI2014_001 is NOT demonstrated (masks are controlled).",
     "key": "controlled masks only; BNCI001 not shown naturally support-mismatched"},
    {"id": "N4", "category": "not_established", "commit": "8046929", "report": "C14/C18",
     "text": "We do NOT make blanket over-claims (all-DG-failure, EEG-transfer-impossibility, or support-aware-"
             "invariance worthlessness); the falsification is protocol-scoped (single dataset / single backbone).",
     "key": "single dataset/backbone; falsification is protocol-scoped"},
    # ---- future work ----
    {"id": "F1", "category": "future_work", "commit": "7b38bee", "report": "C20",
     "text": "Score CALIBRATION / a target-free competence ESTIMAND that transports cross-regime: within-target "
             "ranking appears stable (~0.64) and suggests a future score-calibration question.",
     "key": "within-target mean AUC ~0.63-0.65 stable; pooled does not transport"},
    {"id": "F2", "category": "future_work", "commit": "89b8a6a", "report": "C20_EXTERNAL_DATASET_PROTOCOL",
     "text": "External-dataset validation on a genuinely new cohort, via a pre-registered protocol (C20-B) -> C21+ "
             "execution ONLY after approval. BNCI2014_004 remains BARRED.",
     "key": "protocol only; no execution; BNCI2014_004 BARRED_pending_explicit_approval"},
    {"id": "F3", "category": "future_work", "commit": "a8af8c6", "report": "C17/SCPS",
     "text": "Real support-mismatched clinical EEG validation (hierarchical D=(cohort,subject)) is future work.",
     "key": "PD/SCZ cohorts not provisioned; clinical loader offline-only"},
)

CATEGORIES = ("established", "diagnostic_only", "not_established", "future_work")

# ---- C14->C20 evidence chain (for the paper spine) --------------------------------------------------
EVIDENCE_CHAIN = (
    ("C14", "OACI-control / SRC-control closed by falsification battery", "verdict falsified"),
    ("C16", "target-accuracy-good checkpoints exist but source-unobservable; calibration barrier; SRC memorization",
     "6/6 oracle gain; source-audit oracle fails"),
    ("C17", "no strong scalar source signal; weak multivariate signal; boundary source-mirrored; calibration-biased",
     "LOTO 0.6023 p=0.008; boundary corr +0.547"),
    ("C18", "weak signal survives cell-present stress; deletion collapse = accuracy-endpoint non-estimability",
     "S2/S3 beat perm; leakage/boundary robust"),
    ("C19", "pre-registered robust-core probe recovers weak IN-REGIME competence", "pooled 0.561 p=0.005"),
    ("C20", "frozen cross-regime validation NOT established; signal largely regime-local",
     "pooled 0.50-0.54; Holm 0/4; Simpson"),
)

# ---- reviewer objection matrix -------------------------------------------------------------------
REVIEWER_OBJECTIONS = (
    ("Isn't this just a negative result?",
     "No -- it localizes the failure (observability + estimand boundary) and recovers a real in-regime weak "
     "positive (C19). The contribution is a falsification + observability framework."),
    ("C19's positive could be noise / a fishing result.",
     "Config frozen + committed BEFORE the run (hash 664007686afb520f); beat 200/200 permutations; holds vs "
     "strict chance; endpoint-augmented adds nothing."),
    ("The C20 S6/S7 'passes' show partial generalization.",
     "No -- Holm 2/4->0/4, permutation p-floor non-discriminative, 95% CI includes chance, S7 is the random "
     "control exceeding the S6 treatment, and the split is severity-local + Simpson-confounded."),
    ("The within-target 0.64 means the probe works.",
     "It is within-target RANKING, NOT the pre-registered pooled estimand; regime-nonspecific; reported as "
     "future-work calibration question, not a success."),
    ("A metric bug undermines trust.",
     "The availability-metric bug was caught GATE-FIRST before any claim and fixed (fail-loud); reason-code-"
     "feature-loss discipline. Same for C18 v1 leakage bug."),
    ("Why not just try BNCI2014_004?",
     "External validation needs a pre-registered protocol + approval (C20-B), not an ad-hoc second dataset; "
     "BNCI2014_004 stays barred."),
)

# ---- forbidden claims (guarded in every generated file) ------------------------------------------
FORBIDDEN_CLAIM_SUBSTRINGS = (
    "deployable selector", "deployable target-free selector", "target-free selector", "we built a selector",
    "we found a selector", "detector is validated", "competence detector is validated", "oaci is rescued",
    "oaci rescue", "external validation succeeded", "external validation is established", "all dg fails",
    "eeg transfer is impossible", "support-aware invariance is useless", "c20 succeeded",
    "generalization is established", "production selector",
)

CANONICAL_CONCLUSION = (
    "Under strict target isolation, the tested source-side DG control objectives do not transfer to reproducible "
    "target worst-domain improvements. However, the failure is not a simple absence of target-good states: OACI "
    "trajectories contain target-accuracy-good checkpoints that are invisible to simple source-audit selectors. "
    "Source-only competence information is weakly present in multivariate, deletion-robust observables and can be "
    "recovered by a pre-registered low-freedom diagnostic probe in-regime. Yet this signal does not establish "
    "stable cross-regime external validation: held-out support-stress regimes reveal that within-target ranking "
    "information does not transport as a pooled cross-target competence estimand. The contribution is therefore a "
    "falsification and observability framework, not a deployable target-free selector.")

# ---- manuscript outline v2 -------------------------------------------------------------------------
PAPER_SECTIONS = (
    ("01_introduction", "strict-DG EEG; the measurement->control gap; contribution = falsification + observability + estimand boundary"),
    ("02_problem_setting", "target isolation; support graph; overlap-aware leakage; K1/K2; estimands"),
    ("03_falsification_battery", "C14: OACI/SRC control objectives falsified under protocol (C8/C12/C14)"),
    ("04_mechanism_results", "C16: target-good checkpoints exist but source-unobservable; calibration barrier; SRC memorization"),
    ("05_identifiability_and_probe", "C17 weak multivariate identifiability; C18 support-stress + endpoint-estimability; C19 pre-registered in-regime weak positive"),
    ("06_external_boundary", "C20 frozen cross-regime validation not established; estimand boundary (within-target vs pooled)"),
    ("07_discussion_limitations", "single dataset/backbone; diagnostic-only; future work = calibration estimand + external protocol"),
)

# ACAR_FROZEN_v4.md — v4 (CURB) candidate freeze **(DRAFT — §4 AUDITED + CLI IMPLEMENTED; NOT YET TAGGED)**

```
STATE         : DRAFT — binding ONLY when committed AND tagged `acar-v4-protocol` after sign-off
LINEAGE       : v2 MEASUREMENT_ONLY (9b2f0c1) · v3 DEV_STOP (817b04f/9f4e83f) · v4 DEV candidate (e4c4e91, EXPLORATORY)
EXTERNAL ARM  : NOT RUN — authorized only AFTER tag + sign-off
LOCKBOX       : SEALED / NOT CONSUMED
§4 HELD-OUT LIST : AUDITED + FILLED (metadata-only, notes/ACAR_V4_LOCKBOX_AUDIT.md) — admissible: (zenodo14808296,SCZ),
                   (ds007526,PD), both single-site; ASZED provisional; ds007020 excluded
EXTERNAL CLI  : IMPLEMENTED — acar/v4/run_external_armb.py (+ acar/v4/external_adapter.py); synthetic/preflight guards
                   acar/v4/tests/test_external_armb.py (11 green); committed WITH this protocol (binding execution path).
REMAINING BLOCKERS TO TAG : (1) clean-process guard run of all v4 suites + clean worktree; (2) sign-off; THEN tag
                   `acar-v4-protocol` on that commit; the single external read happens only after via the CLI.
DATE          : 2026-06-29
```

Freezes the single v4 candidate from `notes/ACAR_V4_CANDIDATE_SELECTION.md` (DEV exploration #001,
`results/acar_v4_dev_exploration_001/`). The protocol text is auditable on its own; the binding execution path is the
committed CLI (`acar/v4/run_external_armb.py`), frozen together with this document under the `acar-v4-protocol` tag.
v4 never edits the frozen v2/v3 commits or tags.

## 1. Fixed substrate (inherited, unchanged)
Estimand `ΔR_a(B) = R_B(f_a) − R_B(f_0)` (NLL, paired, label-free at deployment; ΔR<0 = good). Actions `[identity,
matched_coral, spdim, t3a]`. Calibration/eval unit = the subject cluster `cohort_id::subject_id`. B=32, MIN_BATCH=8
(fallback retained, forced identity, in the denominator). Substrate = the v3 single-execution path (`acar/v3` +
`acar/v4/real_adapter.py`), reused unchanged.

## 2. The frozen candidate — ONE of everything
```
score_family   : shift_margin     harm_a(B) = benefit_a(B) = +features_v2[B, a, 1]  (= +d_margin; label-free, no ΔR)
policy_family  : benefit_ranked   π(B) = identity if min_a benefit_a(B) > τ else argmin_a benefit_a(B);
                                   ties in argmin broken by NON_IDENTITY order (matched_coral ≺ spdim ≺ t3a);
                                   fallback batches forced identity. direct_selective is the bit-for-bit numerically
                                   identical alias (gate = −min_a harm, decreasing-λ) — recorded NON-PRIMARY.
calibration    : finite-grid Learn-Then-Test (NO monotone-CRC theorem assumed)
                 method = ttest (one-sided H0: E[loss] ≥ budget) · correction = holm · alpha = 0.10
                 aggressiveness = increasing_lambda · select the MOST AGGRESSIVE passing λ
                 PASS only if EVERY EVAL fold (DEV) / the single CAL stratum (external) certifies a passing λ
weighting      : subject-macro (subject-equal); coverage/red/harm are weighted means; fallback in the denominator
provenance     : score_family_registry_sha256 = fe5a1f58986f7af1e8cb9db797ae9f08b46bfd749fc22ef8dbc8619005bc774e
                 DEV manifest_sha256 = 8f5ccb288c7ca93857acd593ff6ec31bb4965c522a20d24b289ab9800bb970da
                 DEV v4_oof_records_sha256 = 7c7bcd51de874533cd75f9ec2ba64690930cd3bcbbb868bda7bcbce0a4909768
```

### 2a. Two DISTINCT harm metrics (do not conflate)
```
L_harm_all(s)      = mean_{B∈B(s)} 1[ π(B)≠identity ∧ ΔR_{π(B)}(B) > 0 ]      # ALL of s's batches in the denominator
                     This is the LTT-CONTROLLED loss. Budget = 0.10. (= hierarchy.policy_subject_risk loss=harm_indicator)
harm_among_adapted = P( ΔR_{π(B)}(B) > 0 | π(B) ≠ identity )                   # DESCRIPTIVE only; NOT the LTT budget
```
DEV #001 (the safe candidate): subject-macro `L_harm_all` ≈ PD 0.03 / SCZ 0.05 (well inside the 0.10 budget — it is what
LTT controlled on CAL); `harm_among_adapted` = PD 0.154 / SCZ 0.205 (the descriptive "of the batches we adapt, this
fraction were harmful"). The DEV result note's "harm" column is `harm_among_adapted`. Both are reported; only `L_harm_all`
is the risk-control object.

### 2b. λ grid formula (auditable; matches acar.v4.develop._grid_for_family for benefit_ranked)
```
per calibration stratum (see §3a), over that stratum's CAL batches (label-free; NO ΔR):
  stat(B) = min_a benefit_a(B) = min_a (+d_margin_a(B))                    # benefit_ranked's ranking statistic
  lo = min_B stat(B) ; hi = max_B stat(B)                                  # must be finite (records validated finite)
  if not (hi > lo): grid is empty → stratum NOT_EVALUABLE
  grid = numpy.unique(numpy.linspace(lo, hi, 12))                          # 12 points, dedup; if <2 → NOT_EVALUABLE
  aggressiveness = increasing_lambda (larger τ ⇒ larger adopt set ⇒ more coverage)
λ* = most aggressive τ in grid whose Holm-adjusted one-sided ttest p ≤ alpha for E[L_harm_all] ≤ 0.10 on CAL subjects.
```

## 3. External Arm-B endpoint (run ONLY after tag + audited §4 + sign-off)

### 3a. Calibration stratum + site-local split (executable)
```
stratum            = (site, disease); for a multi-acquisition-unit site (e.g. ASZED) the DEFAULT stratum is
                     (acquisition_unit, disease) → a site-local claim; pooling units is allowed ONLY as an explicit
                     mixture-exchangeability claim, declared per site in §4.
within each stratum: subject-disjoint CAL/EVAL split, subject-hash seed = 0, CAL fraction = 0.40 (subjects),
                     min_CAL_subjects = 20, min_EVAL_subjects = 20. Subjects are the unit (cohort_id::subject_id).
fallback-only subjects: retained in EVAL (forced identity, in the denominator); never in CAL selection.
coverage rule      : every EVAL subject scored OOF under the stratum's single CAL-calibrated λ*; subject-macro metrics.
NOT_EVALUABLE      : a stratum with < min_CAL/EVAL subjects, an empty/degenerate λ grid, or LTT NOT_EVALUABLE is reported
                     NOT_EVALUABLE (flagged, NEVER silently dropped) and counts as neither pass nor fail.
```

### 3b. Comparators (apples-to-apples; computed on the SAME stratum split)
```
v2_replay (external) : the bit-for-bit v2 recipe (acar.regressor.ActionRegressor, seed 0; HGB≥40 / Ridge≥8 / constant)
                       on the SAME stratum CAL/EVAL split, SAME EVAL subjects/batches, SAME fallback denominator, SAME
                       v2 11-D feature schema, SAME subject-macro red — i.e. the v3 run_c0 path on the external stratum.
best_fixed (external): best single fixed non-identity action red on the same EVAL (descriptive utility floor).
```

### 3c. The SINGLE confirmatory endpoint (criterion A — chosen)
Per evaluable stratum, **V4_EXTERNAL_CONFIRMED(stratum)** requires ALL of:
```
(precond) CAL LTT certifies λ* for L_harm_all at budget 0.10 (the risk-control claim);
(safety)  EVAL subject-macro L_harm_all ≤ 0.10                 # criterion A: the safety claim is checked on EVAL too
(utility) EVAL subject-macro red > 0  AND  red > v2_replay(external)
(cover)   EVAL adaptation coverage ≥ 0.15
report also: harm_among_adapted (descriptive), best_fixed (descriptive), the Direction-C ceilings.
```
(We adopt criterion **A** — EVAL `L_harm_all ≤ 0.10` is REQUIRED, not just CAL-certified — to support a "safe router"
external claim. `harm_among_adapted` is reported but is NOT a pass/fail gate.)

### 3d. Multi-site / multi-stratum taxonomy
```
Holm scope  : Holm correction is applied ONLY INSIDE each stratum's finite λ-grid LTT selection (the within-grid
              multiple test over the 12 λ candidates). ACROSS strata, disease confirmation is DETERMINISTIC — there is
              NO cross-stratum p-value adjustment, because the per-stratum EVAL criteria (§3c: L_harm_all ≤ 0.10,
              red > 0, red > v2_replay, coverage ≥ 0.15) are threshold endpoints, not hypothesis tests.
per stratum : V4_EXTERNAL_CONFIRMED | V4_EXTERNAL_NEGATIVE | NOT_EVALUABLE
per disease : "externally confirmed" iff ≥1 evaluable stratum is CONFIRMED AND no evaluable stratum is NEGATIVE. A
              disease with exactly one evaluable stratum is reported SINGLE-SITE confirmatory (explicitly, NOT
              "replicated").
overall     : V4_EXTERNAL_CONFIRMED iff BOTH diseases are externally confirmed (per above);
              else V4_EXTERNAL_NEGATIVE; a killed/partial run = OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT.
PD single-site contingency: if only one admissible PD site exists, PD can only be SINGLE-SITE confirmatory — state this
              as a limitation; it is NOT cross-site replication.
no NOT_EVALUABLE stratum is silently dropped (each is listed with its reason).
```

## 4. External held-out cohort list — AUDITED (metadata-only; see notes/ACAR_V4_LOCKBOX_AUDIT.md)
Primary-source re-verified 2026-06-29 (metadata only — no modeling read). **Admissible strata (this protocol):**

```
stratum (zenodo14808296, SCZ)   site = Zenodo 14808296 (single setup, Semmelweis); 38 SCZ + 39 HC; 64ch/1000Hz; resting
                                eyes-closed; raw; CC-BY-4.0. split: subject-hash seed 0, CAL frac 0.40, min 20/20 →
                                CAL≈31 / EVAL≈46 (EVALUABLE). Per-subject diagnosis from the clinical .xlsx (metadata,
                                read at Arm-B label time). DISJOINT from DEV SCZ (different repo/institution).
stratum (ds007526, PD)          site = OpenNeuro ds007526 v1.0.2 (Tel Aviv Sourasky); 116 PD + 28 HC (participants.tsv
                                `group`); resting-RUNS ONLY (exclude walking); raw BIDS 1.10.0; CC0. split: seed 0, CAL
                                frac 0.40, min 20/20 → CAL≈58 / EVAL≈86 (EVALUABLE; runner preflight confirms ≥1 PD AND
                                ≥1 HC in BOTH CAL and EVAL under seed 0). DISJOINT from DEV PD (different
                                accession/institution; UCSD ds002778 etc.).
```
**Both diseases are SINGLE-SITE** at present → external Arm B is single-site-per-disease confirmatory (NOT replication);
this limitation is stated in the result. **PROVISIONAL (NOT admitted):** Zenodo 14178398 (ASZED, SCZ 2nd site) — pending
data-integrity clearance + per-acquisition-unit (Contec 200 Hz / BrainMaster 256 Hz) metadata; admitting it later is a
separately-dated protocol amendment. **EXCLUDED:** OpenNeuro ds007020 (mortality-only; no HC-vs-PD label).

Runner preflight (metadata-only, before any signal read): re-confirm ds007526 channel/Fs + resting-run selection, the
Zenodo id↔diagnosis mapping, and no re-released/derived subject overlap with the seven DEV cohorts; any stratum failing
the split-feasibility or label checks is reported NOT_EVALUABLE (never silently dropped).

## 5. Execution discipline + leakage firewall (CLI = acar/v4/run_external_armb.py)
The unique external Arm-B CLI mirrors v3's `run_dev_binding` preflight (FAIL-CLOSED, no bypass): manifest schema
(admissible §4 strata only — ASZED/ds007020/DEV rejected), output-dir absent + atomic `os.mkdir` claim, HEAD ==
protocol commit, tag `acar-v4-protocol` → HEAD, clean worktree, per-dump SHA-256, env-lock runtime; THEN a single
confirmatory pass (`evaluate_stratum` per stratum → `external_taxonomy`) → `results/acar_v4_external_001/`
(manifest.json + RESULT.json sentinel; allow_nan=False). No threshold/seed/loss/registry/grid change after the read
starts; a killed/partial run is OPERATIONALLY_ABORTED (cleanup the claimed dir).

**Leakage firewall (binding).** External diagnosis labels enter ONLY (a) CAL λ* selection and (b) EVAL endpoint
scoring. They MUST NOT enter f_0 / source-state fitting (the source state is the DEV-frozen artifact; external labels
never refit it), adapter execution, the label-free `features_v2`, or the action choice before scoring. EVAL labels never
affect λ*; CAL labels affect λ* but never the `score_family` outputs. The v2-replay comparator (§3b) fits/calibrates on
subject-disjoint C0_FIT/C0_CAL drawn from CAL only — never EVAL.

## 6. Status
DRAFT. Binding only when (i) committed (§4 audited + CLI included — done), (ii) all v4 suites green on a clean worktree,
(iii) tagged `acar-v4-protocol`, (iv) signed off. v4's authoritative status is EXPLORATORY_CANDIDATE (Evidence Ledger
A6) until a `V4_EXTERNAL_CONFIRMED` exists. Never edit `9b2f0c1`, `817b04f`, `9f4e83f`, or any v3 result.

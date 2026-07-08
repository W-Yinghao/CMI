CEDAR_01 Acceptance Criteria

Status: pre-registered gate. These criteria decide whether a real frozen-latent
shadow audit is worth taking to PM review for possible P1 authorization. Passing
CEDAR_01 does not itself authorize P1, P2, deployment, or target-generalization
claims.

Hard acceptance criteria

All criteria below must pass:

1. Red-team pass with zero warnings.

2. Grouped conditional leakage is non-null:
   the selected candidate's pre-mask leakage advantage must exceed its
   permutation null, and the null must not approach the selected effect.

3. Selected leakage reduction beats matched random control in at least two
   dataset x backbone cells:
   selected leakage drop fraction must be at least 3x the matched random drop
   fraction, or selected-minus-random absolute margin must be at least 0.15.

4. Source task non-inferiority:
   source balanced-accuracy drop must be <= 0.01. If source bAcc is near ceiling,
   CE / NLL must also be reported to guard against confidence collapse.

5. R3 does not worsen:
   R3 task_drop must not increase. A leakage drop that increases task reliance
   is treated as deletion of task structure, not nuisance surgery.

6. No collapse:
   effective rank, entropy, and label balance guards must remain in a basic
   non-collapse range. Leakage reduction through representation collapse is a
   known failure mode and is not accepted.

7. Source-only decision hygiene:
   target metrics may appear only under diagnostic target fields. They must not
   appear in `decision_reasons`, `utility`, `rank_key`, `tie_break`,
   `candidate_accept`, or any selected-candidate field used for ranking.

Required stop outcomes

- If leakage drops but source or R3 worsens: STOP and write CEDAR_01 negative.
- If leakage drop does not beat random control: STOP, do not enter P1.
- If source/R3 are stable but target bAcc does not improve: P1 may be requested
  only as privacy/leakage surgery, not as generalization.
- If target bAcc improves but source-side criteria fail: reject the outcome.

Minimum report fields

The CEDAR_01 readout must report:

- dataset and backbone for every cell
- feature dump path and schema validation
- grouped split metadata
- candidate universe and skipped-k dimensionality reasons
- selected candidate signature
- selected leakage drop fraction
- matched random drop fraction
- selected-minus-random margin and ratio
- source bAcc / CE / NLL before and after
- R3 before and after
- effective rank / entropy / label-balance guards
- target diagnostics, if present, under `diagnostic_target_metrics` only
- red-team JSON path and zero-warning status

Interpretation limits

Accepted phrasing:

CEDAR_01 finds source-side, grouped, real-frozen-latent evidence for
domain-heavy / task-light units that survive random-control, source-risk, R3,
and collapse guards.

Forbidden phrasing:

- CEDAR works on EEG.
- CEDAR improves generalization.
- CEDAR is deployable.
- CEDAR is a safety gate.
- CEDAR justifies P1 without PM review.

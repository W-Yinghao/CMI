# B2B_SOURCE_POWER_FAIL — terminal conclusion for source-calibrated eligibility (pre-frozen)

Source-only futility checkpoint (no target evaluation launched; ~45 trainings saved). Route-
conditioned SINGLE-action gate, frozen evidence score, other-seed empirical null at FPR <= 0.10:

    route   null-FPR   cross-fit retention   ROC ceiling @FPR0.10
    pooled  0.083      0.153                 0.157
    cc      0.117      0.177                 0.159
    aggregate cross-fit retention = 0.165  (< 0.25)  -> B2B_SOURCE_POWER_FAIL

Decisive point: the null FPR is on target (calibration correct) and the multi-action max-null
correction is removed, yet the ROC CEILING -- the maximum achievable retention at FPR <= 0.10 -- is
~0.15-0.16. So no threshold/calibration choice can reach 0.25: the FROZEN change-of-variable evidence
score lacks the discriminative POWER to separate the small gain-based geometry shifts from the true
null at a 10% false-adaptation rate. This is not multi-action conservatism, not magnitude-transfer,
not regime mismatch -- it is a score-power limit.

Per the pre-registered termination rule: target evaluation is NOT launched, and no further target-
only statistic (normalized evidence, combined stability, new score) is attempted. Outcome = C.

## Where the program lands (immutable)
- Mechanism (solid): the joint EM's harm is the PRIOR M-STEP; geometry-only / fixed-prior beats it
  on the cov-family (B1a, CIs exclude 0).
- Metadata ROUTING is validated: the two-axis rule table is already safe WITHOUT a gate (B2a
  metadata-ungated false-adaptation 0.00) and abstains correctly on UNKNOWN/UNSUPPORTED.
- The OPEN problem is purely the ELIGIBILITY gate's power: unlabeled, source-calibrated evidence
  cannot detect small acquisition-geometry shifts at an acceptable false-adaptation rate. Detecting
  WHETHER to adapt -- not WHICH operator -- is the binding limitation.

Tags: h2cmi-b1a-astar-terminal (A* fail) ; h2cmi-b2a-canonical-freeze ; h2cmi-b2a-terminal (B2a fail)
; this commit (B2B source-power fail).

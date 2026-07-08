# Project B Merge Readiness Checklist

## Code boundaries
- [x] No forbidden h2cmi/cmi files modified in Step-4.
- [x] Router core has unit tests.
- [x] Router harness has smoke test.
- [x] Real EEG bridge has metadata-only test.

## Label safety
- [x] Target labels are post-hoc only.
- [x] Support thresholds are source-only.
- [x] No target-label threshold tuning.

## Statistical posture
- [x] ACAR-harm degeneracy is explicit.
- [x] No fake harm bound is emitted.
- [x] TTA blockers do not mark IDENTITY unsafe.
- [x] Prior-shift-only is audit-only.

## Evidence
- [x] Synthetic R2/HF3/H-OOD frozen.
- [x] Real BNCI2014_004 bridge smoke complete.
- [x] Bounded real benchmark complete.
- [x] Claim boundary updated.

## Limitations
- [x] No full MOABB benchmark claim.
- [x] No concept-shift solution claim.
- [x] No guaranteed TTA improvement claim.
- [x] No real beneficial-TTA recovery claim.

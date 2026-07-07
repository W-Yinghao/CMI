# FSR_34 — Final Claim Ledger Patch (Phase 6A)

**Project FSR.** Record of the Phase-6A patches applied to the claim ledger (`FSR_05_CLAIM_LEDGER.md`) so the
manuscript's `Table~\ref{tab:claims}` (`paper/fsr/tables/table1_claim_ledger.tex`) reflects the final,
frozen claim status. The ledger is the manuscript-facing source of truth; the paper table mirrors it.

## Patches applied
- **C7 → SUPERSEDED (Phase 4B).** "Branch-local leakage/reliance is missing" was correct at the frozen-artifact
  stage but is **superseded** by the Phase-4B ERM refit, which produced a direct real-EEG branch-local L1–L6
  audit. RQ4 is **no longer blocked** in the manuscript. (Paper section `06_branch_blocked` → replaced by
  `06_branch_natural_verification`.)
- **+C16 — Natural branch-local subject leakage is not automatically harmful (READY).** Phase 4B
  `NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`: spatial is max-leaky + load-bearing, but subject-subspace removal hurts
  the target → task-entangled, refuse the harmful-shortcut label. Forbidden: "spatial/subject leakage is
  harmful", "erase to improve DG".
- **+C17 — Repair scope is first-moment-specific (READY_WITH_CAVEAT).** 4F first-moment constant-offset
  repairable (construction-matched: 73% mechanical identity, fails leave-one-dataset-out); 4G second-moment
  stochastic not repairable at the operating point (even oracle-directed). Forbidden: "repairs general shortcuts",
  "second-moment unconditionally unrepairable", "FSR solves shortcut repair".

## Final ledger (C1–C17) — statuses
| id | status | claim |
|---|---|---|
| C1 | READY_WITH_CAVEAT | Measured leakage magnitude does not certify reliance. |
| C2 | READY_WITH_CAVEAT | Task-head alignment is closer to reliance than raw leakage. |
| C3 | READY | Subject signal is erasable. |
| C4 | READY | Erasure strength does not certify target benefit (0/40). |
| C5 | READY_WITH_CAVEAT | Random-k falsifies non-specific NLL movement. |
| C6 | READY | Spatial branch is load-bearing. |
| C7 | SUPERSEDED (Phase 4B) | Branch-local leakage/reliance was missing in frozen artifacts. |
| C8 | READY | CMI-control remains closed. |
| C9 | SUPPORT_ONLY | TTA-Control is positive but non-CMI. |
| C10 | READY | FSR is an audit framework, not a new DG method. |
| C11 | READY | FSR detects, localizes, and attributes controlled injected harmful shortcuts. |
| C12 | READY_WITH_CAVEAT | E4 repairs controlled first-moment constant-offset shortcuts. |
| C13 | FORBIDDEN | E4 repairs general or natural shortcuts. |
| C14 | NOT_ESTABLISHED (4G none) | E4b repairs controlled second-moment shortcuts. |
| C15 | NOT_READY (GPU_PAUSED) | PC2 learned-reliance repair works. |
| C16 | READY | Natural branch-local subject leakage is not automatically harmful. |
| C17 | READY_WITH_CAVEAT | Repair scope is first-moment-specific. |

**Manuscript-eligible (state as findings):** C1–C6, C8, C10, C11, C16 (READY / READY_WITH_CAVEAT), C12/C17 with
mandatory caveats, C9 as support/context. **Forbidden as claims:** C13. **Not in results (future work only):**
C14, C15.

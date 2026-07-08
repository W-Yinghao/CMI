# Project B: Refusal-First Safe EEG Adaptation Router

## Summary
Adds a **refusal-first deployment router** on top of the frozen H2-CMI model. For an unlabelled EEG
target it chooses `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and emits OACI reason codes, using
only source data; target labels are post-hoc only. This is **not** a new TTA optimizer.

## Motivation
TTA can help or harm an EEG decoder and source data alone cannot certify which. Rather than always
adapting, Project B refuses or falls back to a support-valid identity when adaptation cannot be shown
safe, and records why.

## Main changes
- `h2cmi/router/*` — reasons, actions, features, ACAR calibration state, refusal-first policy.
- `h2cmi/eval/router_harness.py` — source-only integration harness (support threshold + pseudo harm).
- `h2cmi/data/real_eeg_bridge.py` — label-safe MOABB (BNCI2014_004) subject->session bridge + LOSO.
- `h2cmi/tests/*` — router smoke test + real-bridge metadata test.
- `scripts/project_b_*.py` — synthetic sweep/eval, Step-2G report, real bridge/benchmark, Step-3B/3D
  paper packaging, Step-4 packaging.
- `notes/PROJECT_B_*.md` — design, protocols, synthetic report, real results, paper draft, claim
  boundary, reviewer checklist, branch summary, PR description, reproducibility, manifest, checklist.

## Safety and label-safety
Target labels are used **only post-hoc**. Support thresholds are **source-only**; there is **no
target-label threshold tuning**. ACAR-harm degeneracy is explicit and blocks TTA (no fabricated harm
bound). TTA blockers never mark IDENTITY unsafe. Prior-shift-only is audit-only, never a refusal.

## Evidence
- **Synthetic (frozen, Step-2G):** R2 nested support fixes all-refuse; HF3/H-OOD show TTA harm + limits.
- **Real bridge smoke (Step-3A):** BNCI2014_004 targets 1-2, raw TTA harmful, identity accepted.
- **Bounded real benchmark (Step-3C):** BNCI2014_004, 4 targets, subject+session, both support modes.
  Raw offline TTA harmful (mean d_bAcc -0.140); OFFLINE_TTA never
  selected; SUPPORT_MISMATCH-driven refusal; LOW_ESS inactive; nested support inert.

## Tests run
Smoke / contract tests run now (fast/medium): `router.reasons/actions/features/acar/router`,
`eval.router_harness`, `tests.test_router_smoke`, `tests.test_real_eeg_bridge` — all pass. The
synthetic sweeps, the real bridge, and the bounded benchmark are **frozen experiment outputs generated
earlier** (see `reproducibility.md`); this PR does **not** re-run those expensive experiments.

## Claim boundary
Claimable 7 / NOT-claimable 8; see
`notes/PROJECT_B_CLAIM_BOUNDARY.md`. We do not claim accuracy improvement over identity, beneficial-TTA
recovery on real EEG, a concept-shift solution, guaranteed TTA improvement, or a full benchmark.

## Known limitations
Source-only ACAR-harm frequently degenerate; support cannot detect concept-shift accuracy loss; real
run in a harmful-TTA regime; real evidence is bounded (bridge smoke + bounded expansion), not a full
benchmark.

## Reviewer notes
Start at `notes/PROJECT_B_BRANCH_SUMMARY.md`, then `PROJECT_B_CLAIM_BOUNDARY.md` and
`PROJECT_B_REVIEWER_CHECKLIST.md`. Reproduction levels in `PROJECT_B_REPRODUCIBILITY.md`.

## Follow-up work
Step-3E real-dataset expansion (BNCI2014_001 / Lee2019_MI, GPU, session-level, no target-label tuning)
on a separately scoped expansion branch.

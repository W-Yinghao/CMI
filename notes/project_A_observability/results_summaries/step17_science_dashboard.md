# Step 17 — Science Dashboard (estimand-consistent harm control)

Scope: estimand-consistent harm control (accuracy vs balanced-accuracy); not SOTA.

## Key metrics

- real runs: **54** · accuracy benefit-rate **0.1481** · bAcc benefit-rate **0.1481** · sign-agreement **1.0**
- runs accuracy-benefit∧bAcc-harm **0** · bAcc-benefit∧accuracy-harm **0**
- estimand relationship: **identical_on_grid** · identical on grid **True** · all targets class-balanced **True** · max |acc−bAcc gain| **0.0**
- accuracy policy controls bAcc: **False** · no overall best across estimands: **True**
- best accuracy-gain policy **plugin_sign** (k **full**, minimal-label control **False**) · best bAcc-gain policy **plugin_sign** (k **full**, minimal-label control **False**, requires C13 **True**)
- estimand-consistency warning: On this grid the two estimands COINCIDE (all targets class-balanced -> accuracy-gain == balanced-accuracy-gain per run); the Step-16 gap was a THRESHOLD artifact, not an estimand divergence. An accuracy-gain policy is still never reported as a balanced-accuracy-gain control.
- claim boundary ok **True**

## What we learned

1. On this grid all 54 target sets are class-balanced (True); accuracy-gain == balanced-accuracy-gain per run (max |diff| = 0.0). The Step-16 0.1481-vs-0.0926 gap was a THRESHOLD artifact — accuracy benefit thresholded at eps=0 vs bAcc at eps=0.005. At a shared threshold the rates coincide: eps=0 -> 0.1481 (acc) / 0.1481 (bAcc); eps=0.005 -> 0.0926 / 0.0926.
2. On this grid the two estimands COINCIDE (all targets class-balanced -> accuracy-gain == balanced-accuracy-gain per run); the Step-16 gap was a THRESHOLD artifact, not an estimand divergence. An accuracy-gain policy is still never reported as a balanced-accuracy-gain control.
3. The Step 15/16 negative PERSISTS under bAcc-consistent control: no minimal-label policy meets harm<=0.10 at coverage>=0.05 for EITHER estimand (only k=full / oracle-equivalent budgets do; best-k accuracy full, bAcc full) — both estimands COINCIDE here (class-balanced targets), so this is estimand-invariant.
4. Frontiers are kept strictly separate by estimand and sampling; there is no overall best policy across estimands; the class-balanced balanced-accuracy frontier requires contract C13.

## What remains unknown

1. Whether the accuracy/bAcc gap widens on more class-imbalanced clinical EEG.
2. Whether a class-balanced (C13) acquisition protocol is feasible in real BCI calibration.
3. Whether a bAcc-consistent policy could control harm at higher coverage with active sampling.

> Accuracy-gain and balanced-accuracy-gain are distinct target functionals; a policy licensed for one is never reported as controlling the other; class-balanced bAcc-gain estimation requires contract C13; k>0 slices are R2 under a sampling contract, NOT R1 identifiability. No SOTA.

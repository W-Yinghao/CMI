# FSR_32 — Experiment Freeze (Phase 6A)

**Project FSR.** The experiment phase is **CLOSED** for this manuscript (PM decision, 2026-07-07). This file is
the authoritative frozen inventory: every experiment, its binding verdict, and its frozen artifacts. **No new
experiments** are run for the paper. What remains is manuscript freeze + rewrite (FSR_33/34/35 + `paper/fsr/`).

## Frozen experiments (all adversarially verified; branch `project/functional-shortcut-reliance` docs + code
`project/fsr-rq4-refit`)
| phase | question | binding verdict | key docs |
|---|---|---|---|
| Audit (Step 2B) | measurable ≠ reliance? | leakage wrong-signed; alignment closer (not validated) | FSR_04/05 |
| TOS | erasable ≠ beneficial? | `benefit_claimable = 0/40` | (tos_cmi) / C4 |
| **4B** | natural branch-local shortcut? | **`NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`** (spatial task-entangled) | FSR_16/17 |
| **PC1-S** | can FSR detect a *known* injected shortcut? | **detect+localize+attribute PASS; erasure ≠ repair** | FSR_19/20 |
| **4D** | counterfactual-head repair? | **`none`** (ties random control; harm under-powered) | FSR_21/22 |
| **4E** | token-centering repair? | **`none`** (E4 ties task-destructive ERASE gate) | FSR_24/25 |
| **4F** | corrected first-moment repair? | **`strong_within_controlled_first_moment_scope`** (abs +0.033 bAcc; 73% mechanical identity; fails LODO) | FSR_26/27 |
| **4G** | second-moment repair? | **`none`** (E4b ≤ random; even oracle sub-DELTA; genuinely-weak) | FSR_29/30 |
| PC2-E4 | learned-reliance repair (GPU)? | **PAUSED / not eligible** (design only) | FSR_23/28/31 |

## The frozen scientific closure
- **Verification/attribution succeed** (PC1: detect+localize+attribute a known injected shortcut).
- **Natural spatial subject leakage is task-entangled, not verified harmful** (4B; C16).
- **Repair scope boundary** (C17): first-moment **deterministic** offset repairable (4F, construction-matched);
  second-moment **stochastic** perturbation **not** repairable at the operating point, even oracle-directed (4G).
- **The measurement→control gap persists at the intervention layer**, now mechanistically bounded.

## Frozen commits (heads at freeze)
`project/fsr-rq4-refit` @ `d4e7c37` (code + result CSV/JSON); `project/functional-shortcut-reliance` @ `5007870`
(docs + results). Every verdict JSON is the machine source of truth; every doc claim is evidence-grounded.

## NOT done (frozen out of this manuscript)
```
No PC2 GPU run.            No Lee2019_MI (or any 3rd-dataset) preset for this paper.
No new FBCSP-LGG refit.    No new repair primitive.       No CMI / fbdualpc.
No hyperparameter rescue.  No target-label selection.     No larger dataset sweep.
```
Deferred to **future infrastructure backlog** (not critical path): add ≥1 preset-ready dataset + a repair
primitive beyond first-moment before any learned-reliance PC2 (FSR_31 go/no-go).

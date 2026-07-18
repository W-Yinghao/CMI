# Risk-Weighted MCC — RESULT: source-LOSO risk is trainable + specifically targeted, but does NOT transfer to DG (Result B)

Real EEG, full LOSO. 63 bundles × 3 arms = 189/189 (A=ERM / B=true-RW / C=weight-permuted from one hash-verified
warm-up; λ_RW=1.0; weights frozen at warm-up). All weight_status=ok (no no-op bundles). Engineering all passed
(RW0 weights healthy + seed-stable, RW1 checks green). All compute via SLURM. Manuscript FROZEN.

## Result
| dataset | ΔU_RW−ERM (B−A) | p | **ΔU_RW−WPerm (B−C)** DECISIVE | p | dR_src (B−A) | **dR_src specific (B−C)** | src drop |
|---|---|---|---|---|---|---|---|
| BNCI2014 | −0.0014 [−0.0041] | 0.84 | **−0.0012 [−0.0057]** | 0.69 | −0.00085 | **−0.00109** | +0.0016 |
| BNCI2015 | −0.0004 [−0.0071] | 0.54 | **−0.0019 [−0.0087]** | 0.66 | −0.00092 | **−0.00104** | +0.0019 |

## Reading = B (source-risk trainable + specific, but DG-inert)
- **The weighting works on its own target**: true RW-MCC reduced the source-LOSO excess risk (dR_src B−A ≈ −0.0009)
  AND reduced it MORE than the weight-permuted control (dR_src specific B−C ≈ −0.0011, consistent sign on BOTH
  datasets) — so the *correct* source-risk assignment specifically constrains the right subjects' contrasts (the
  permuted control slightly RAISED source risk, ≈ +0.0002). The mechanism is trainable and the weights are specific.
- **But it does NOT transfer to DG**: ΔU_RW−ERM ≈ 0 (p 0.84/0.54) and — the contract's decisive gate — ΔU_RW−WPerm
  (B−C) ≈ 0 (p 0.69/0.66). Getting the right source subjects weighted gives no future-subject accuracy advantage
  over a permuted assignment. No source damage (drop < 0.002), no no-op bundles.
- → **Result B: source meta-generalization failure ≠ future-target failure.** Cross-SUBJECT source-LOSO instability
  is trainable and specifically targetable, but it is not the axis that governs the held-out subject's accuracy.

## Honest caveats (magnitude + accumulated pattern)
- Everything here is TINY (~0.001 scale): the source-risk reduction (−0.001) and every DG number (±0.002). The
  20-epoch λ=1 continuation barely moved anything; the "specific" B−C source-risk effect is directionally consistent
  across datasets but small.
- **This completes a consistent pattern across the whole mechanism-consistency-training line** (frozen: notes/
  MECHANISM_CONSISTENCY_MCC_RESULT.md, MCC_LAMBDA1_RESULT.md, MCC_ESTIMATOR_AUDIT_RESULT.md, and this): global MCC
  (λ=0.25, λ=1.0) and now risk-weighted MCC all shape a REAL, controllable source-side geometry/consistency signal,
  and none of it produces DG utility on the held-out subject. M1-P (disagreement magnitude ≠ future harm), λ1
  (geometry decoupled from DG, corr≈−0.05), the estimator audit (K=4 not variance-limited), and RW (correct
  source-risk weighting DG-null) all point the same way: **source-side task-mechanism consistency — however
  shaped/weighted — is not the DG lever on BNCI2014/2015.**

## Disposition (PM routing B) + honest flag
Per the routing, Result B → next round builds weights from source-only CROSS-SESSION (early→later session)
instability rather than cross-subject LOSO — testing whether a within-subject temporal shift is a better proxy for
the deployment shift than cross-subject disagreement. HOWEVER, given the accumulated null across the entire line, I
flag for the PROJECT OWNER: this cross-session pivot is worth ONE bounded test, but if source-side consistency
shaping remains DG-inert, the honest conclusion may be that the DG bottleneck on these datasets is not a
source-shapeable mechanism-consistency object at all (consistent with the earlier target-X / erasure-oracle
findings that the beneficial direction is target-hindsight-only / source-unobservable). That is an OWNER-level
decision on whether to continue the consistency-training line or pivot; I do not stop it unilaterally.

HELD: EMA/prototype, cross-session RW (pending owner go), M2, learned projector, TTE, CMI, manuscript. Scientific
line ACTIVE; the source-LOSO-weighted hypothesis is terminated (trainable but DG-inert).

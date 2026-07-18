# Information-Regime Ladder — IL-C (verified with caveats) [Track B]

**Branch/worktree** `CMI_AAAI_infoladder` @ post-verify. **Manuscript FROZEN.** Only the owner stops/redirects a line.
Turned `TARGET_HINDSIGHT_ONLY` into a sample-complexity question. 63 frozen ERM EEGNet cells (BNCI2014 9 subj +
BNCI2015 12 subj × seed0-2), NO re-inference. Selection-only ladder R0/RX/R1/R2/R4/RF, informed B_cond vs matched-rank
random, session-macro query gain; target labels only SELECT (never retrain encoder/head). Head-only calibration
secondary. Inference unit = target subject (seeds→subject first), subject-cluster bootstrap + exact sign-flip.

## Verdict = IL-C_HEAD_CALIBRATION_NOT_SUBSPACE (adversarially verified: IL-C_WITH_CAVEATS, all 3 skeptics confirm)

### 1. Selection-only is NON-IDENTIFIABLE at every information level (both datasets)
| regime | BNCI2014 dU | BNCI2015 dU | 2014 specificity | 2015 specificity |
|---|---|---|---|---|
| R0 source-LOSO | −0.0021 | −0.0005 | −0.0016 | −0.0005 |
| RX unlabeled X | **−0.0213** | **−0.0114** | −0.0049 | −0.0163 |
| R1 1-shot | −0.0025 | −0.0001 | +0.0026 | −0.0105 |
| R2 2-shot | −0.0027 | +0.0040 | −0.0015 | −0.0099 |
| R4 4-shot | +0.0061 | +0.0094 | +0.0031 | −0.0106 |
| RF all cal | +0.0120 [−0.0030] | +0.0208 [−0.0042] | −0.0019 | −0.0138 |

**k\* = None, subspace-specific threshold = None.** No regime's query gain has LCB95>0, and the informed B_cond
dictionary is never reliably better than matched-rank random (specificity LCB<0 everywhere; NEGATIVE throughout on
BNCI2015). RF is a near-miss on the null side (dU +0.012/+0.021, signflip p≈0.08, uncorrected over 6 regimes) — treating
it as positive would be selection optimism; **do not re-run expecting RF to cross.** RX (unlabeled-X G1 selection)
actively HURTS (−0.021/−0.011) — the unlabeled observability mis-ranks (echoes the target-X closure).

### 2. The useful deletion EXISTS but is HINDSIGHT-ONLY
Crossfit target-oracle (selects on QUERY labels) beats its own rank-matched random on both datasets:
+0.0111 [lcb +0.0030] vs random −0.0133 (BNCI2014); +0.0319 [lcb +0.0095] vs random −0.0084 (BNCI2015). A subject-
subspace deletion that helps the unseen session EXISTS — but it is recoverable only with target **query** labels
(hindsight), not from any amount of **calibration** information (0 → unlabeled X → 1/2/4 labels → all cal). (Caveat:
the oracle is doubly favored — hindsight labels AND a larger greedy rank≤8 budget vs selection subsets≤3 — so the
clean existence basis is the paired oracle-minus-its-own-rank-matched-random contrast, not the raw RF/oracle ratio,
which is dataset-inconsistent.)

### 3. What target labels DO help = the P(Y|Z) READOUT, and only at FULL calibration, and GENERICALLY
Head-only calibration (fresh linear head on k cal labels, encoder frozen), query bAcc minus the frozen source readout:
| regime | native−source 2014 | native−source 2015 | selected−source (informed-DELETED) 2014 | 2015 |
|---|---|---|---|---|
| R1 | −0.100 | −0.110 | −0.099 | −0.103 |
| R2 | −0.067 | −0.050 | −0.067 | −0.040 |
| R4 | −0.030 | −0.002 | −0.032 | −0.003 |
| RF | **+0.125 [+0.096]** | **+0.135 [+0.069]** | **+0.110 [+0.083]** | **+0.113 [+0.049]** |

A fresh readout beats the frozen source head ONLY with the FULL calibration session; at few-shot (1/2/4 labels) it is
WORSE than the frozen source head on both datasets. And the benefit is **generic** — the informed-subspace-DELETED
refit (`selected−source`) still beats source at RF (+0.110/+0.113 ≈ native−source), so deleting the contested rank-8
subspace leaves the readout benefit essentially intact; the subspace deletion never adds (selected−native negative).
**IL-C means: the DG bottleneck is the target-specific readout that needs MANY calibration labels — not a deletable
subspace, and not near-label-free.**

## Answer to the PM's sample-complexity question
The useful subspace is NOT identifiable at ANY tested target-information level (source-only, unlabeled target X, or
1/2/4/all labeled cal trials-per-class); it is accessible only in query-label hindsight. Few-shot labels do NOT
resolve its identifiability; the labels that help (many of them) are best spent fitting a new P(Y|Z) readout, whose
benefit is generic to the representation, not tied to any subject subspace.

## Owner options (report-then-wait; nothing run unilaterally; no CLOSED without a 3rd dataset — this line has
## historically reversed sign between the two datasets)
1. **Few-shot readout adaptation** — the confirmed lever is a FULL-label P(Y|Z) refit; the open question is the
   label-efficiency of the head (close the gap between few-shot-worse-than-source and full-cal +0.13). This is
   ordinary supervised calibration, not subspace surgery.
2. **Third-dataset confirm** before any status change, per line discipline.
3. **A dedicated random-subspace-deleted head column** to make "generic, not subspace-specific" airtight (currently
   evidenced via the informed-deleted `selected−source`, which already beats source).
4. Transductive / hindsight selector only as a NEGATIVE-boundary probe (selection-null holds even at full cal, so a
   positive identifiable-selection line is unlikely — frame as characterizing the measurement→control gap).
5. Stop the erasure/selection sub-line and keep only the readout-label-efficiency question.

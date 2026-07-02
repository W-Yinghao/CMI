# CIGL_49 appendix — BNCI2014_001 CSP fold-difficulty map (P5-A, CPU-only)

Classical CSP+LDA on all 9 LOSO folds (same moabb data as F0: tmin0.5/tmax3.5/resample128). Cross-subject
= train on 8 source subjects, test on the held-out subject. Within-subject = 5-fold CV on the held-out
subject (a decodability ceiling). Chance = 0.25. Table: `results/fblgg_f0/BNCI2014_CSP_ALL_FOLDS.csv`.
This is NOT a GPU gate and NOT a test of FBLGG — only a difficulty map to pick future pilot folds.

| fold | subj | cross-subj CSP | within-subj CSP | cross-decodable (>0.40) |
|---|---|---|---|---|
| 0 | 1 | 0.483 | 0.717 | yes (F0 fold) |
| 1 | 2 | 0.248 | 0.521 | no (F0 fold — hard) |
| 2 | 3 | 0.571 | 0.821 | yes |
| 3 | 4 | 0.342 | 0.472 | no |
| 4 | 5 | 0.255 | 0.390 | no |
| 5 | 6 | 0.285 | 0.396 | no |
| 6 | 7 | 0.325 | 0.703 | no (within OK, no transfer) |
| 7 | 8 | 0.599 | 0.817 | yes (best) |
| 8 | 9 | 0.444 | 0.689 | yes |
| **mean** | | **0.395** | **0.614** | 4 / 9 |

## Answers to the P5-A questions

1. **How many target subjects are CSP cross-subj > 0.40?** **4 of 9** — subjects **1, 3, 8, 9**
   (0.483, 0.571, 0.599, 0.444). CSP mean 0.395 already exceeds FBLGG (0.296) and DGCNN (0.342).
2. **Is subj 2 (F0 fold 1) a hard target?** **Yes** — cross-subj 0.248 (chance) and within-subj only
   0.521. It is the 2nd-worst cross-subject fold; several subjects (4/5/6) are also near chance
   cross-subject even for CSP, so BNCI2014 has a genuinely hard tail.
3. **Are the current F0 folds pessimistic?** **Partially.** F0 used subj 1 (decodable, 0.483) + subj 2
   (hard, 0.248); the hard fold drags the 2-fold mean down. A fold set weighted toward CSP-decodable
   subjects would give a less pessimistic — and more architecture-diagnostic — read.
4. **Which folds for the next GPU pilot?** Use **CSP-decodable folds as the architecture sanity set** so
   that a good backbone *must* improve there, and keep one hard fold as a reference:
   - decodable set (recommended): **subj 8 (0.599) + subj 3 (0.571)** — highest cross-subj & within-subj;
     a competent backbone should clear the cross-subject bar there.
   - keep **subj 1 (F0 fold 0)** for continuity with F0/DGCNN comparisons.
   - keep **subj 2** as the hard-case reference (do not judge the backbone by it alone).
   Concrete recommendation for the FBCSP-LGG F0 run-spec: `target_indices` covering subj 1 + subj 8
   (indices 0 and 7) — one decodable-with-F0-continuity, one strongly-decodable — plus subj 2 (index 1)
   only if a hard reference is wanted. Final fold choice is the PI's; this appendix just supplies the map.

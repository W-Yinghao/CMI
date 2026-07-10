# S2P_17 - Route B FACED Final Results

**Status:** confirmatory through 1000 h; H2000 is excluded pending completion and a new frozen-checkpoint audit.

## Scope

This report closes the currently valid Route B FACED frozen-probe analysis:

- CBraMod-only Route B pretraining on the pinned 33-channel TUEG substrate.
- Completed checkpoints at H in {200, 500, 1000} h, seeds {0, 1}.
- FACED native 32-channel input, 10 s, 200 Hz, 9 classes.
- Fixed subject split: train 1-80, validation 81-100, test 101-123.
- Frozen encoder; source-only PCA and logistic head; no fine-tuning.
- Primary endpoint: test Cohen's Kappa. Secondary: balanced accuracy and weighted F1.
- Target labels enter final scoring and subject-cluster bootstrap scoring only.

The final verifier used 5000 paired bootstrap replicates over the 23 FACED test subjects. Training seeds were
averaged within each bootstrap replicate. It exactly reproduced all eight committed through-1000 objects: six
pretrained checkpoints, random initialization, and the released CBraMod reference.

## Provenance correction

The earlier D2-2 H2000 audit is not load-bearing. Job 891546 completed at 03:27, while both H2000 training jobs were
still running and subsequently overwrote `best.pth` at 03:54 and 04:44. The original audited checkpoint SHA was not
recorded and the intermediate files are not recoverable. Final job 891560 correctly failed when the current
H2000_s0 checkpoint differed from the committed metrics by 0.0216.

At detection, H2000_s0/s1 were still active SLURM jobs 890151_6/7. They must finish, produce completion markers, and
have checkpoint SHAs pinned before any new H2000 downstream audit. Until then, these claims are withdrawn:

- H2000 sustains the floor crossing.
- H2000 reaches the released frozen-reference band.
- H1000 to H2000 is a positive budget step.

The through-1000 results are unaffected. Full details are in
`results/s2p_route_b_33ch_b1_faced/faced_h2000_provenance_incident.json`.

## Confirmatory results

All confidence intervals below are paired target-subject cluster-bootstrap intervals. L1 is the source-side mean
pairwise subject balanced accuracy and remains descriptive because its raw pair-level bootstrap was not rerun here.
L5 is the Kappa drop from subject-subspace erasure minus the drop from a source-val-energy-matched random erasure.

| Checkpoint | Kappa mean [95% CI] | bAcc mean [95% CI] | Delta Kappa vs random | L1 | Task gate | L5 subject-minus-null |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.0285 [0.0081, 0.0485] | 0.1371 [0.1189, 0.1550] | - | 0.7806 | reference | - |
| 200 h | 0.0423 [0.0264, 0.0589] | 0.1489 [0.1349, 0.1636] | +0.0137 | 0.9791 | 2/2 | -0.0026 |
| 500 h | 0.0724 [0.0552, 0.0911] | 0.1747 [0.1598, 0.1914] | +0.0438 | 0.9926 | 2/2 | -0.0166 |
| 1000 h | 0.0673 [0.0457, 0.0893] | 0.1703 [0.1515, 0.1894] | +0.0388 | 0.9885 | 2/2 | -0.0077 |
| Released | 0.0755 [0.0537, 0.0991] | 0.1763 [0.1565, 0.1978] | +0.0469 | 0.9763 | reference | - |

### Floor comparisons

Paired budget-mean deltas against random initialization:

| Budget | Delta Kappa [95% CI] | Delta bAcc [95% CI] | Point clears +0.02 rule |
|---:|---:|---:|---:|
| 200 h | +0.0137 [-0.0045, +0.0320] | +0.0117 [-0.0048, +0.0283] | no |
| 500 h | +0.0438 [+0.0148, +0.0718] | +0.0376 [+0.0115, +0.0628] | yes |
| 1000 h | +0.0388 [+0.0110, +0.0651] | +0.0331 [+0.0084, +0.0563] | yes |

Thus 200 h remains at the floor. The first sampled budget with an above-random paired CI is 500 h, and both 500 h
and 1000 h pass the predeclared point rule of random +0.02. The CI lower bounds establish an above-random effect,
but do not themselves exceed +0.02; the +0.02 statement remains the protocol's point-estimate margin.

### Released reference

The H500 mean differs from released by -0.0031 Kappa [-0.0222, +0.0154] and -0.0016 bAcc
[-0.0191, +0.0156]. H1000 differs by -0.0081 Kappa [-0.0339, +0.0137] and -0.0060 bAcc
[-0.0293, +0.0138]. These comparisons support "within the released frozen-reference band" descriptively. They
do not establish equivalence, superiority, or reproduction of released pretraining.

## Budget response

Across the three valid budget means, the descriptive linear response is positive:

- Kappa: +0.0114 per log2 hour, 95% CI [+0.0036, +0.0185].
- bAcc: +0.0098 per log2 hour, 95% CI [+0.0033, +0.0157].

This is not a scaling law. The observed means are non-monotone because H500 is above H1000. Leave-one-budget-out
slopes change sign when H200 is omitted, and the three-point quadratic sensitivity is concave. The defensible term
is **budget-dependent emergence** or **pretraining-budget response**, not monotonic scaling or an optimal budget.

## Task-gated mechanism

The frozen task gate is source-val Kappa >= 0.05636. All six pretrained cells pass, so L4/L5/L6 are interpretable
under the fixed frozen linear head.

- L1 rises from random 0.7806 to 0.9791 at 200 h and remains 0.9885-0.9926 at 500-1000 h. Subject identity is
  nearly saturated before FACED transfer clears the random floor.
- L4 task-head energy in the rank-5 subject subspace is small: budget means are 0.00579, 0.00224, and 0.00066 at
  200, 500, and 1000 h.
- L6 subject-erasure Kappa consequences are near zero. Budget means are -0.00245, -0.00202, and +0.00056.
- No pretrained cell's L5 subject intervention exceeds the variance-matched null after Holm correction across the
  six cells. The budget-mean subject-minus-null contrasts are -0.0026, -0.0166, and -0.0077.

The final L5 control is stricter than the historical FACED rank-matched random control. Each random orthobasis is
partially erased until its source-validation removed energy exactly matches the subject intervention; maximum match
error is below 4e-16. The null construction uses source-validation features without labels and never uses target
labels for fitting or selection.

The supported mechanism statement is:

> As pretraining budget increases from 200 h to 500-1000 h, FACED task information becomes linearly accessible
> while subject identity remains nearly perfectly separable. Under the frozen source-only head, erasing the measured
> subject subspace does not have a larger task effect than an equal-energy random intervention.

This is evidence against measured subject-subspace reliance under this probe. It is not proof that all subject
information is harmless, that CBraMod is subject-invariant, or that no nonlinear task mechanism uses subject cues.

## Locked conclusion

The load-bearing Route B result currently ends at 1000 h:

> Subject-identifiable structure appears by 200 h and rapidly approaches a ceiling. Frozen FACED transfer is still
> at the random floor at 200 h, but is above random by the sampled 500-1000 h budgets. Transfer emerges without a
> reduction in subject separability, and the measured subject subspace is not a task lever under the task-gated
> frozen linear intervention.

No new pretraining, H4000, CodeBrain run, or fine-tuning is part of this result.

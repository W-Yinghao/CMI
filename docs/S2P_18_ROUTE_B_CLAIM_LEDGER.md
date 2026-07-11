# S2P_18 - Route B Claim Ledger

**Locked against:** `results/s2p_route_b_33ch_b1_faced/faced_final_verification.json`,
`faced_h2000_provenance_incident.json`, and
`results/s2p_route_b_h2000_immutable_closure/h2000_immutable_checkpoint_manifest.json`.

## Load-bearing claims

| ID | Status | Allowed wording | Evidence boundary |
|---|---|---|---|
| RB-C1 | SUPPORTED | The through-2000 FACED frozen-probe results reproduce exactly under the final verifier. | 10/10 random, released, and H200/H500/H1000/H2000 objects; max aggregate metric difference 0; H2000 SHA stable before/after inference. |
| RB-C2 | SUPPORTED | Frozen FACED transfer is at the random floor at 200 h and above random by the sampled 500-2000 h budgets. | Paired test-subject CIs for H500/H1000/H2000 minus random exclude 0; all three point estimates exceed random +0.02. |
| RB-C3 | DESCRIPTIVE | H500/H1000/H2000 reach the released frozen-reference band. | Budget-minus-released CIs include 0; one released checkpoint, unmatched provenance. No equivalence claim. |
| RB-C4 | SUPPORTED | Subject identity becomes strongly linearly separable before FACED-transferable structure emerges. | L1 0.979 at 200 h while H200 transfer remains at floor; L1 stays about 0.99 through H2000. |
| RB-C5 | SUPPORTED | Transfer emergence does not coincide with reduced subject separability. | H500-H2000 transfer clears random while L1 remains near ceiling. This is coexistence, not causal mediation. |
| RB-C6 | SUPPORTED, PROBE-BOUNDED | Under the frozen source-only head, the measured subject-subspace intervention does not exceed an equal-energy random intervention. | 8/8 cells pass task gate; variance matching error <4e-16; no Holm-significant L5 cell; L6 remains small. |
| RB-C7 | DESCRIPTIVE | The valid budgets show a positive overall pretraining-budget response. | Global log-budget slope CI is positive, but H500 > H1000/H2000 and leave-one-budget-out signs are unstable. |
| RB-C8 | SUPPORTED | Immutable H2000 sustains the above-random floor crossing and lies in the released frozen-reference band. | SHA-pinned epochs 48/49; paired deltas vs random exclude 0; differences vs released include 0. This is not superiority or reproduction. |

## Scientific wording

This section constrains future scientific communication; it does not authorize manuscript drafting.

Preferred headline:

> Subject identity precedes transferable structure in EEG foundation pretraining.

Preferred result statement:

> In a CBraMod-only 33-channel budget calibration, subject identity is already nearly perfectly separable at
> 200 h, while frozen FACED transfer first appears above the random baseline at the sampled 500 h budget and remains
> above baseline through 2000 h.
> Transfer emerges without subject invariance, and the measured subject subspace is not functionally privileged
> over an equal-energy random intervention under the task-gated frozen linear head.

Required scope suffix:

> This is a frozen-probe, CBraMod-only Route B result on FACED, not a full-fine-tuning scaling-law reproduction and
> not a CodeBrain-compatible 19-channel experiment.

## Forbidden claims

- More pretraining data monotonically improves performance.
- Route B establishes a scaling law or an optimal budget.
- H500 is optimal.
- H2000 outperforms, reproduces, or is equivalent to released CBraMod.
- H1000 to H2000 is a statistically established positive budget step.
- Route B outperforms, reproduces, or is equivalent to released CBraMod.
- Subject leakage decreases with scale.
- Subject identity is harmless.
- CBraMod becomes subject-invariant or channel-invariant.
- The measured L5 null proves no nonlinear or unmeasured subject reliance exists.
- Subject diversity was isolated.
- Route B is CodeBrain-compatible or reproduces CodeBrain's 19-channel/full-fine-tuning scaling analysis.
- H4000 was tested.

## Interpretation rules

Use these terms:

- budget-dependent emergence
- pretraining-budget response
- floor crossing
- linear accessibility
- frozen-reference band
- measured subject-subspace reliance

Avoid these terms:

- scaling law
- convergence to released
- invariance
- harmless leakage
- causal effect of subject diversity

Negative L5 subject-minus-null values mean only that this subject-subspace erasure was no more damaging than an
equal-energy random erasure. They do not mean subject identity helps the task and do not license a general null over
all subject information.

## H2000 closure status

The six unlock conditions are complete: jobs 890151_6/7 reached 50 epochs, strict reload passed, epochs 48/49 were
selected by pretrain-val loss, immutable SHA256 payloads were created, job 892861 reran FACED, and job 892882
reproduced all ten final objects under `--scope full`. The historical mutable D2-2 result remains invalid and must
not be cited in place of the immutable result.

No resume, rerun, H4000 job, manuscript work, or resource-policy change is authorized by this ledger.

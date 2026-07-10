# S2P_18 - Route B Claim Ledger

**Locked against:** `results/s2p_route_b_33ch_b1_faced/faced_final_verification.json` and
`faced_h2000_provenance_incident.json`.

## Load-bearing claims

| ID | Status | Allowed wording | Evidence boundary |
|---|---|---|---|
| RB-C1 | SUPPORTED | The through-1000 FACED frozen-probe results reproduce exactly under the final verifier. | 8/8 random, released, and H200/H500/H1000 objects; max aggregate metric difference 0. |
| RB-C2 | SUPPORTED | Frozen FACED transfer is at the random floor at 200 h and above random by the sampled 500-1000 h budgets. | Paired test-subject CIs for H500/H1000 minus random exclude 0; both point estimates exceed random +0.02. |
| RB-C3 | DESCRIPTIVE | H500/H1000 reach the released frozen-reference band. | Budget-minus-released CIs include 0; one released checkpoint, unmatched provenance. No equivalence claim. |
| RB-C4 | SUPPORTED | Subject identity becomes strongly linearly separable before FACED-transferable structure emerges. | L1 0.979 at 200 h while H200 transfer remains at floor; L1 stays about 0.99 at H500/H1000. |
| RB-C5 | SUPPORTED | Transfer emergence does not coincide with reduced subject separability. | H500/H1000 transfer clears random while L1 remains near ceiling. This is coexistence, not causal mediation. |
| RB-C6 | SUPPORTED, PROBE-BOUNDED | Under the frozen source-only head, the measured subject-subspace intervention does not exceed an equal-energy random intervention. | 6/6 cells pass task gate; variance matching error <4e-16; no Holm-significant L5 cell; L6 near zero. |
| RB-C7 | DESCRIPTIVE | The valid budgets show a positive overall pretraining-budget response. | Global log-budget slope CI is positive, but H500 > H1000 and leave-one-budget-out signs are unstable. |
| RB-C8 | WITHDRAWN/PENDING | No H2000 scientific claim is currently allowed. | D2-2 evaluated mutable in-flight checkpoints; original SHA absent; jobs 890151_6/7 still training at incident detection. |

## Manuscript wording

Preferred headline:

> Subject identity precedes transferable structure in EEG foundation pretraining.

Preferred result statement:

> In a CBraMod-only 33-channel budget calibration, subject identity is already nearly perfectly separable at
> 200 h, while frozen FACED transfer first appears above the random baseline at the sampled 500-1000 h budgets.
> Transfer emerges without subject invariance, and the measured subject subspace is not functionally privileged
> over an equal-energy random intervention under the task-gated frozen linear head.

Required scope suffix:

> This is a frozen-probe, CBraMod-only Route B result on FACED, not a full-fine-tuning scaling-law reproduction and
> not a CodeBrain-compatible 19-channel experiment.

## Forbidden claims

- More pretraining data monotonically improves performance.
- Route B establishes a scaling law or an optimal budget.
- H500 is optimal.
- H2000 sustains the floor crossing or reaches released level, until a completed SHA-pinned checkpoint is audited.
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

## H2000 unlock conditions

H2000 may re-enter the claim ledger only after all of the following:

1. Jobs 890151_6 and 890151_7 leave the queue normally or their infrastructure status is resolved.
2. Both runs reach the protocol endpoint and write `run_summary.json` with checkpoint reload success.
3. Final `best.pth` SHA256 values are pinned before downstream inference.
4. The B1 pretrain gate passes for both runs.
5. FACED D2-2 is rerun against those immutable checkpoints.
6. The final verifier runs with `--scope full` and reproduces the new H2000 metrics exactly.

No resume, rerun, H4000 job, or resource-policy change is authorized by this ledger.

CEDAR_01 — Real-EEG Frozen-Latent Shadow Audit Protocol

Status: protocol only. CEDAR_00 is frozen as a scaffold / red-team contract.
CEDAR_01 is not P1 channel pruning and does not authorize any deployable mask,
checkpoint, selector, or target-generalization claim.

Objective

Answer the minimum next question on real frozen EEG representations:

Can source-side, grouped conditional leakage probes identify domain-heavy /
task-light latent units whose in-memory deletion reduces conditional domain
leakage more than matched random subspace deletion, without source task, R3, or
basic representation-collapse failures?

Scope

- Phase: real EEG frozen-latent shadow audit.
- Allowed datasets: BNCI2014_001 first, then BNCI2015_001 only if existing
  frozen feature dumps are already available.
- Allowed backbones: EEGNetMini and EEGConformerMini only.
- Training: forbidden. This phase consumes existing frozen feature dumps only.
- Selection regime: source-only shadow selection.
- Target labels: quarantined diagnostic-only, never used in candidate selection,
  ranking, tie-breaks, acceptance, or decision reasons.
- Output: compact readout artifacts only; no deployable mask artifact.

Required feature dump schema

Each real feature dump must contain:

- `z` or `Z` or `features`: source frozen latent features `[n, d]`
- `y` or `labels`: source task labels
- `domain` or `d` or `domains`: source subject/session/site labels
- `groups` or `recording` or `session`: grouped split labels

Optional diagnostic-only fields:

- `z_target` or `Z_target` or `target_z`
- `y_target` or `target_y`

Grouped split policy

Grouped cross-fit is mandatory for real CEDAR_01. Missing, singleton, or invalid
groups must hard-fail. There is no random-window fallback, because window
autocorrelation can make leakage probes look valid when they are not.

Candidate universe

The candidate universe is pre-registered and fixed:

- `drop_top_1`
- `drop_top_2`
- `drop_top_4`
- matched random subspace control for each available k

If a latent dimensionality cannot support a requested k, that k is skipped only
by a deterministic dimensionality rule, not by observed performance. Optional
bottom-leakage negative controls may be reported separately, but cannot alter
selection.

Deterministic source-side ranking

Among ACCEPT candidates, selection is determined only by stable source-side
fields:

1. larger utility
2. smaller k
3. lower source bAcc drop
4. larger selected-minus-random leakage drop fraction margin
5. larger leakage drop fraction
6. lexical candidate name

Target diagnostics, file order, dict insertion order, and any target-derived
field are forbidden tie-breakers.

Required red-team checks before any report

1. Target perturbation invariance:
   replacing target metrics with NaN/random values must leave the selected
   candidate signature byte-identical.

2. Candidate completeness:
   every candidate must contain random-control fields, grouped split metadata,
   permutation-null fields, source utility deltas, and explicit decision state.
   No missing-field candidate may be silently dropped.

3. Tie-break determinism:
   reported selection must match the predefined source-side rank key exactly.

Allowed outputs

```text
cedar_eeg/reports/CEDAR_01_REAL_FROZEN_LATENT_READOUT.md
results/cedar/p0_real_shadow/<run_id>/report.json
results/cedar/p0_real_shadow/<run_id>/red_team.json
results/cedar/p0_real_shadow/<run_id>/candidate_table.csv
```

Forbidden outputs

```text
*.pt checkpoint
deployable_mask.json
selected_mask.npz
any model artifact marked deployable=true
```

If an implementation must persist a mask-like diagnostic for audit replay, the
filename must include `DIAGNOSTIC_ONLY_NON_DEPLOYABLE` and its manifest must
state:

```json
{
  "deployable": false,
  "target_label_role": "quarantined_diagnostic_only",
  "mask_selection_regime": "source_only_shadow"
}
```

Current gate state

- CEDAR_00 scaffold: accepted as contract only.
- Synthetic P0 witness: accepted as smoke only.
- CEDAR_01 real EEG frozen-latent: approved to run only under this protocol.
- P1 channel pruning: blocked until CEDAR_01 readout passes PM review.
- P2 TTA preconditioner: blocked.
- Generalization / safety claim: forbidden.

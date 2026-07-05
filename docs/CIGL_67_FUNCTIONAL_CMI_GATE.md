# CIGL_67 — Functional CMI gate (branch `project/cigl-functional-cmi`)

```
New method phase, off the frozen evidence branch project/cigl-r123-scaffold @ 1339592. Directly acts on CIGL_66
Outcome A: CIGL reduces measured leakage but the RESIDUAL subject subspace becomes concentrated + task-head-
aligned, so R3 reliance does not fall. Functional CMI penalizes the intersection of the subject subspace with the
task path. Goal: reduce reliance (R3 task_drop) + head/subject alignment while retaining task on real EEG.
```

## Estimand shift
Old CIGL ≈ controls representation-wide `I(Z;D|Y)`. CIGL_66 showed what matters is the *task-relevant* part:
`I(P_task Z; D|Y)` — subject-predictive directions that overlap the task-head row space. The two variants penalize
that overlap directly, on top of the existing CIGL graph/node terms (λg=λn=0.010).

## Two primary variants (`cmi/train/trainer.py`, `FCIGL_METHODS`)
Both use the DGCNN adapter + the graphcmi graph/node CMI terms; both add a **source-only, periodically-updated,
detached** label-conditional subject-subspace projector on `graph_z` (k=2 primary, k=4 secondary; refreshed every
`fcigl_update_every` epochs after warmup, from a forward pass over the training set = all source; **no target
labels**).
- **fcigl_align** (η): `L_align = ‖P_S Wₜᵀ‖²_F / ‖Wₜᵀ‖²_F`, S = top-k subject subspace, Wₜ = live task-head weight.
  Pushes the head row-space OUT of the subject subspace. Loss `+ η·warm·L_align`.
- **fcigl_removal_aug** (α): `+ α·warm·CE(head((I−SᵀS) graph_z), y)` — the classifier must still work after the
  subject subspace is removed (training-time analogue of the R3 removal test).

Loss (both): `CE + λg·CIGL_graph + λn·CIGL_node + [η·L_align | α·CE_removed]`.

## Pre-registered small grid (NOT a blind sweep)
`η ∈ {0.01, 0.05}`, `α ∈ {0.5, 1.0}`; `λg=λn=0.010` fixed. Four variants:
`fcigl_align_eta0.01`, `fcigl_align_eta0.05`, `fcigl_removal_aug_a0.5`, `fcigl_removal_aug_a1.0`.

## Seed0 gate (this phase)
`scripts/run_cigl_functional_gate.py` — 4 variants × BNCI2014_001 (9) + BNCI2015_001 (12) = **84 GPU runs**, full
LOSO, seed 0, same adapter / firewall / audit+R3+head-replay export as the R2 gate. **Reference comparators (ERM,
CIGL, DANN, cond-DANN, CDAN) are NOT rerun** — compared against the frozen CIGL_65 tables. (ERM/CIGL are rerun
only if training code touches their shared semantics — it does not: for non-fcigl methods the trainer path is
byte-inert, confirmed by 221-test regression.)

## Seed0 primary metrics (post-hoc from the audit npz + frozen tables)
`target_bacc`, measured `graph_kl/node_kl`, **R3 task_drop k2**, **task_head_alignment_k2**. Compared to old CIGL,
ERM, and CDAN (near-peer). Read (PM-frozen):
- **strongest:** target↑ over CIGL AND R3 task_drop↓ AND alignment↓ AND leakage below ERM.
- **functional pass:** target≈CIGL AND R3 task_drop↓ AND alignment↓ AND leakage controlled.
- **bounded/diagnostic:** alignment↓ but no R3/target gain.
- **fail:** target drops materially, or R3 task_drop doesn't fall, or alignment stays high, or leakage rebounds to ERM.

## seeds 1/2 (only if seed0 signal)
Expand top 1–2 variants if a variant meets `target_bacc ≥ CIGL−0.005 AND R3 task_drop k2 ≤ CIGL−0.01 AND
alignment_k2 < CIGL`, or clear target improvement with no leakage rebound. 2015 gives the cleaner reliance readout
(above chance); 2a near-chance limits interpretation — report both, don't cherry-pick.

## Firewall / integrity (same as R2 gate)
Strict source-only; projector + subspace fit on training data only; target eval-only; same DGCNN adapter for all;
head-replay export + verify per fold. Stop-and-report on: NaN, replay not reproducing logits, indices
unreconstructable, any target-label use in a fit, backbone-path divergence.

## Engineering tests (`tests/test_fcigl.py`, 7 CPU pass)
methods registered + finite train; **fcigl_align measurably REDUCES head/subject alignment vs η=0** (behavioral);
removal_aug trains + logs; projector source-only (no target arg) + deterministic; fails closed on non-graph
backbone / missing head; `I−SᵀS` is a valid subject-removing projector. Synthetic = engineering only.

## Explicitly NOT done
No old-CIGL λ sweep, no baseline zoo, no DANN/CDAN variants, no P10, no architecture search, no
CDAN+CIGL+functional stacking, no external data (Cho2017/HGD), no writing push.
```
```

# STAR_00 S2P Dependency Ledger

## Git authority

- Repository: `W-Yinghao/CMI`
- Dependency branch: `origin/project/s2p-subject-scaling`
- Required commit: `a9134eb5eb7f8486a5e1ee41831823dab39381ed`
- STAR branch: `project/star-task-anchor`
- Isolation: independent worktree; no merge, rebase, or modification of S2P is authorized.

The start-time remote ref resolved exactly to the required commit before the STAR branch/worktree was created. A newer ref is not an acceptable substitute. If the shared remote-tracking ref advances later, the preflight records the drift and requires that the STAR merge-base remain exactly the frozen SHA; STAR never rebases onto or consumes the newer ref.

## Read authority

The following files were reviewed before implementation:

- `docs/S2P_16_ROUTE_B_TRAINING_PROTOCOL.md`
- `docs/S2P_17_ROUTE_B_FINAL_RESULTS.md`
- `docs/S2P_18_ROUTE_B_CLAIM_LEDGER.md`
- `docs/S2P_19_NEXT_STAGE_SCIENTIFIC_EXPLORATION_PLAN.md`
- `docs/S2P_20_H2000_IMMUTABLE_CLOSURE_PROTOCOL.md`

Implementation provenance was located in:

- `s2p/scripts/run_frontier_cbramod.py`
- `s2p/scripts/tueg_subject_loader.py`
- `s2p/scripts/route_b_33ch_loader.py`
- `s2p/scripts/route_b_train_cbramod.py`
- `s2p/scripts/route_b_faced_downstream_audit.py`
- `s2p/scripts/route_b_faced_final_verification.py`
- `s2p/scripts/route_b_h2000_immutable_closure.py`
- `results/s2p_route_b_33ch_b1_faced/faced_checkpoint_manifest.csv`
- `results/s2p_route_b_33ch_b1_faced/faced_dataset_manifest.csv`
- `results/s2p_route_b_33ch_b1_faced/faced_channel_order_manifest.json`
- `results/s2p_route_b_33ch_b1_faced/faced_final_verification.json`
- `results/s2p_route_b_h2000_immutable_closure/h2000_immutable_checkpoint_manifest.json`
- `results/s2p_route_b_33ch_contract/route_b_b1_training_tasks.csv`

## Imported contracts

STAR may read or import stable Route B utilities without changing them. Imported behavior is limited to the native CBraMod architecture/objective semantics, the pinned H200 Route B TUEG pool, checkpoint state structure, FACED native32 normalization/feature extraction, the 1–80/81–100/101–123 split, and the frozen-probe/mechanism verifier definitions.

S2P source files and scientific artifacts are immutable dependencies. H2CMI, OACI, project-A observability, CEDAR, TALOS, and TTA closeout artifacts are outside STAR ownership.

## Checkpoint roles

| Object | STAR role |
|---|---|
| H200_s0, H200_s1 | Only permitted STAR starting checkpoints |
| H500_s0/s1 | Frozen descriptive reference only |
| H1000_s0/s1 | Frozen descriptive reference only |
| Immutable H2000_s0/s1 | Frozen descriptive reference only |
| Released | Single descriptive frozen reference only |
| Random config | Frozen reference only |

H500/H1000/H2000/released/random cannot train STAR, select variants, tune the schedule, select a checkpoint, or establish equivalence/reproduction/superiority.

The generated `results/star/star00a_preflight/dependency_manifest.json` hashes this authority set and records protected-path results. `checkpoint_inventory.json` independently records payload SHA256, strict reload, route/channel contracts, and training provenance.

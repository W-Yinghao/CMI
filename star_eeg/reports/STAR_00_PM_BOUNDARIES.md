# STAR_00 PM Boundaries

## Permanent project exclusions

STAR is not S2P Phase B replacement, S2P Phase C, H2CMI/qxu AAAI, OACI/TPAMI, CMI training, a subject-invariance or erasure method, CEDAR/TALOS/TTA continuation, CutClean/pruning/sparsity, CSP initialization, a safety/harm router, or source-free deployment.

The active STAR path may not add conditional leakage penalties, subject/domain adversaries, parameter/subspace surgery, pruning or trainable masking, low-rank/LoRA adaptation, target-unlabeled adaptation, target-entropy objectives, target-label selection, H4000, CodeBrain, a new downstream dataset, manuscript/abstract/figure work, or any hyperparameter/layer/label-budget/anchor-ratio sweep.

## STAR_00A mutation boundary

Only `star_eeg/` and `results/star/star00a_preflight/` may be added. No `.pth`, `.pt`, `.npz`, raw EEG, real feature dump, or large log may be committed.

Do not modify:

- `docs/S2P_*.md`
- `results/s2p_*`
- any S2P scientific protocol or behavior
- `h2cmi/`
- `oaci/`
- `notes/project_A_observability/`
- CEDAR, TALOS, or TTA closeout artifacts
- pre-existing untracked noise

Stable S2P utilities may be imported read-only. STAR cannot merge/rebase S2P, force-delete a branch/worktree, force-push, or open a PR without a new PM instruction. The current PM instruction permits only the bounded STAR_00C smoke and, after all 00C machine gates and a SHA-bound approval lock pass, one six-cell STAR_01A array plus its `afterok` immutable closure. It does not permit target scoring.

## No rescue after scientific failure

Failure does not authorize changing anchor fraction, learning rate, label budget, layer scope, head, checkpoint, or pretraining budget. It does not authorize adding a penalty/adversary or moving to H500/H1000/H2000. Any extension requires a new explicit PM gate.

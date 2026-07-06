# CIGL_70 — Source-only CMI reliance-control: CLOSURE (systematic negative)

```
Freezes the source-only CMI reliance-control ROUTE as failed and closed. This is NOT a closure of the CMI
project — only of the source-only reliance-control line. CMI audit and partial proxy control are retained as
durable contributions. The next scientifically justified step is a DIFFERENT information regime
(target-unlabeled CMI adaptation), not any further source-only sweep.
```

## Verdict
```
Source-only CMI reliance-control route: FAILED and CLOSED.
```
Four control routes — static-DGCNN **CIGL** (global leakage), **FCIGL** (task-head alignment), **dCIGL**
(direct counterfactual), and source-episodic **MetaCMI** on EEGNetMini / EEGConformerMini — each measured and/or
reduced a CMI proxy, but **none** stably converted that into **functional reliance reduction** or **target
generalization improvement** on the held-out subject.

This is not an early-stopping misjudgment: the key routes were taken through **full-LOSO × seeds 0/1/2**
method-level confirmation (CIGL_65/67/68), and the MetaCMI attribution was decisive already at the **full-LOSO
seed0** screening tier (MetaCMI ≤ MetaCE on target in every cell). We keep the standing rule: tiny = engineering
only; full-LOSO seed0 = scientific screening; full-LOSO × seeds 0/1/2 = method-level judgment.

## The four routes (see `results/cigl_source_only_closure/route_summary.csv`)
| route | CMI object / proxy | positive part | failed part | decision |
|---|---|---|---|---|
| **CIGL** (DGCNN) | global I(Z_g;D\|Y)+node posterior-KL proxy, λ=0.010 | leakage ↓ ~40–65%, task retained, non-dominated Pareto, FDR-sig | **R3 reliance NOT ↓** (measurement→control gap) | NEGATIVE (CIGL_65 d3593c8) |
| **FCIGL** (DGCNN) | task-head row-space energy in subject subspace, η 0.01/0.05 | alignment ↓, task retained | **R3 reliance NOT ↓** | NEGATIVE (CIGL_67 d02c400) |
| **dCIGL** (DGCNN) | direct SymKL(logits,logits_rem)+γ·CE(rem,y), β0.1 | seed0 reliance-↓ signal | **seeds 0/1/2 not reproduced** | NEGATIVE unstable seed0 (CIGL_68) |
| **MetaCMI** (Mini) | source-episodic β·SymKL(h(z_mh),h((I−SᵀS)z_mh)), β0.1 | tiny extra leakage-proxy ↓ vs MetaCE | **MetaCMI ≤ MetaCE target (4/4); R3 ↓ within noise** | FAIL (CIGL_69C 8068191) |

## Frozen scientific finding
```
CMI audit works.
CMI proxy control works partially.
CMI source-only reliance control fails robustly.
```
More completely:
> In this strict source-only EEG setting, label-conditional leakage can be measured and reduced as a proxy, but
> neither global leakage control, task-head-alignment control, direct counterfactual consistency, nor
> source-episodic MetaCMI reliably reduces functional reliance or improves generalization. Source subjects do
> not carry enough information to stably control the held-out subject's functional nuisance structure.

This is a **strong negative result, not void work.** The durable contribution is the audit machinery
(permutation-significant label-conditional leakage; verified classifier-level R3 head-replay; valid
random-subspace control) and the demonstration that measured leakage is *reducible* without task collapse — the
gap is specifically the **measurement/proxy → functional-control** step, and it holds across BOTH the graph and
the source-episodic formulations.

## Retained vs closed
- **Retained (durable):** CIGL audit + partial proxy control.
- **Closed (negative):** CIGL / FCIGL / dCIGL / MetaCMI as a *stronger source-only method*.

## Frozen prohibitions (no further source-only CMI work)
No more β/η/α/k/λ sweeps; no MetaCMI seeds 1/2; no ConformerFull MetaCMI; no new source-only backbone for CMI
reliance control; no source-only CMI schedule sweep; no static-DGCNN CIGL/FCIGL/dCIGL revival.

## Next (distinct information regime — separate branch/eval/boundary)
`project/cita-target-unlabeled-cmi` — **target-unlabeled CMI adaptation**. New question: *if the held-out
subject's functional nuisance structure cannot be inferred from source subjects, can unlabeled target X make CMI
control effective?* Success there must be stated as **"target-unlabeled CMI adaptation works under a different
information regime"**, never "CIGL source-only works". Target X allowed at adaptation; **target y forbidden**
(training, adaptation, model selection, pseudo-label selection, early stopping, hyperparameter selection). See
`docs/CITA_01_TARGET_UNLABELED_PROTOCOL.md`.
```

# Phase 3.0 — Backbone generality frozen pilot (BNCI2014_001/2a, second backbone = EEGNet)

**Question.** Is Phase 2's TSMNet negative (subject leakage high-dim/redundant, NOT low-rank removable;
global LPC de-domains only via collapse) specific to the SPD/LogEig representation, or general? Same
dataset/protocol (2a, LOSO, domain=subject), frozen-feature pilot, second backbone = **EEGNet**
(braindecode EEGNetv4, conv, penultimate latent **z_dim=16**; vs TSMNet LogEig/SPD **z_dim=210**).
All reuse Phase 2 tooling with `--backbone EEGNet` (boundaries honored: no TOS training, no PCGrad, no
task_protect/power-floor change, no λ-tuning beyond the pre-registered sweep, no target-based selection).

## Headline (adversarially verified — wf_cb3b4958-8e2, 4 skeptics + synthesis)
Low-rank removability of conditional subject leakage **is genuinely backbone-dependent** (survives the
fraction-confound attack), but the result is more nuanced than a positive method:
1. It is **PARTIAL** — linear leakage ~67% removed, a substantial **nonlinear residual persists**.
2. It is best framed as **latent-compactness / capacity-mediated, NOT "SPD vs conv" type** (type and
   dimensionality are perfectly collinear in this 2-backbone/1-dataset design — an unresolved confound).
3. Removing the leakage yields **NO domain-generalization benefit** (target accuracy flat-to-worse) —
   "removable but useless." This **reinforces the measurement→control gap**, it is not a DG method.

## Stage A — geometry / low-rank deletion (ERM, 9 folds × 3 seeds)
Ablation: delete the score-Fisher low-rank V_D (M-oblique projector), probe subject + task on the
complement RZ=(I−P_N)Z with linear AND MLP heads; random-k removal (`Rrand`, same k) is the control.

| backbone | subj Z→RZ (linear) | subj Z→RZ (MLP) | random-k (lin/MLP) | task Z→RZ | nDcand / z_dim |
|---|---|---|---|---|---|
| **EEGNet** | 0.82→**0.35** | 0.88→**0.54** | 0.73 / 0.81 | 0.64→0.64 | 5.3 / 16 |
| **TSMNet** | 1.00→0.93 | 1.00→0.96 | 1.00 / 1.00 | 0.75→0.75 | 3.2 / 210 |

- **Genuine selectivity (fraction confound REFUTED):** random-k deletion at the *same* k barely dents
  EEGNet subject (0.73/0.81), while V_D crushes it (0.35/0.54) — selective gap +0.38 (lin)/+0.27 (MLP),
  selectivity 0.55/0.35. On TSMNet V_D ≈ random (gap +0.07/+0.04, selectivity 0.08/0.04). EEGNet
  selectivity is ~8–13× TSMNet's. Deleting 33% of EEGNet at random does NOT reproduce it.
- **Partial / linear (residual honesty):** EEGNet RZ_mlp=0.54 sits +0.42 above chance 0.125 (t=31.5,
  p~3e-22) — a large nonlinear subject residual SURVIVES low-rank deletion. So "linearly reducible
  (~67%), nonlinear residual persists," not unqualified "removable."
- **Capacity, not type (the key reframe):** with 8 source subjects under LOSO, LDA caps at 7 Fisher
  directions for BOTH backbones. Deleting the FULL 7-dim Fisher subspace: TSMNet subject stays 0.92–0.98
  (redundantly re-encoded across its 210 dims), EEGNet drops to 0.51–0.53. TSMNet is even MORE compact in
  Fisher terms (90% decode at k≈3 vs EEGNet k≈5) — so it is not "info spread over many components"; it is
  **redundant re-encoding enabled by high latent capacity**. Type↔dim are collinear (SPD=210, conv=16);
  a clean type claim would need a high-dim conv or low-dim SPD latent. Flag dim as a confound.
- Leakage is **subject-identity** (0.88) ≫ session (0.63 ≈ chance) on EEGNet too.

## Stage B — global LPC λ-sweep (raw_lpc, 9 folds × 3 seeds, 108 runs)
| λ | src | **tgt (LOSO)** | task_dec | subj_dec | feat_norm |
|---|---|---|---|---|---|
| 0 | 0.73 | 0.43 | 0.65 | 0.89 | 0.60 |
| 0.3 | 0.69 | 0.41 | 0.63 | 0.40 | 1.15 |
| 1 | 0.62 | 0.39 | 0.54 | 0.27 | 0.62 |
| 3 | 0.56 | 0.39 | 0.31 | 0.19 | 0.76 |

- **No collapse (contrast with TSMNet):** src degrades *gradually* (0.73→0.56, never to chance 0.25);
  feat_norm never →0 (0.60–1.15). EEGNet does NOT undergo the TSMNet objective-scaling collapse. Global
  LPC genuinely reduces subject leakage (0.89→0.19, monotone) without destroying the representation.
- **But NO DG benefit (removal-benefit REFUTED):** target bAcc is flat-to-declining (0.43→0.39),
  statistically WORSE than λ=0 at every λ (paired-t p≤0.012); corr(leak reduction, target gain) =
  −0.14 (n.s.). Source bAcc + task decode also degrade. Removing leakage destroys useful signal without
  a generalization payoff.

## TSMNet vs EEGNet — summary
| property | TSMNet (LogEig/SPD, 210-d) | EEGNet (conv, 16-d) |
|---|---|---|
| subject leakage strength | 0.997 | 0.88 |
| low-rank deletion effect | dents only (≈ random) | linear removed (~67%), MLP partial (~45%), ≫ random |
| selectivity (informed vs random) | 0.04–0.08 | 0.35–0.55 |
| raw global LPC at high λ | feature-norm collapse to origin | no collapse, gradual |
| collapse-free LPC removes leakage? | no (only via collapse) | yes (0.89→0.19) |
| removing leakage helps target DG? | n/a (collapses) | **no** (flat-to-worse) |
| TOS gate intent | abstain (not removable) | accept (removable subspace exists) |

## Verdict (pre-registered branches)
A **hybrid of branch 2 (partial replication) + branch 3 (representation-dependent removability)**, with
the decisive honest twist that removability is **capacity-mediated and DG-useless**:
- The TSMNet collapse + "not low-rank removable" is **representation/capacity-specific**, NOT universal —
  on a compact conv latent the leakage is low-rank (linearly) removable and LPC does not collapse.
- BUT removing the leakage buys **no generalization** (target flat-to-worse), so this is **not a positive
  DG method** — it deepens the measurement→control gap: leakage is measurable, sometimes removable, yet
  controlling it still does not improve DG on 2a.

## One line for the paper
> Whether conditional domain (subject) leakage is low-rank removable is representation-dependent and
> capacity-mediated: on a compact 16-d EEGNet latent it sits in an identifiable low-rank subspace whose
> informed deletion (vs same-k random) selectively removes the linear component (~67%, leaving a
> substantial nonlinear residual), whereas on a high-dimensional 210-d TSMNet latent subject identity is
> redundantly re-encoded and is not low-rank removable — yet on the removable EEGNet, removing the leakage
> yields no domain-generalization benefit (target accuracy flat-to-worse, uncorrelated with leakage
> reduction), reinforcing the measurement-to-control gap.

## Caveats / scope
(1) Removability is PARTIAL/linear; nonlinear residual persists (RZ_mlp=0.54 ≫ chance). (2) Latent
dimensionality is confounded with representation type (SPD=210, conv=16) — cannot attribute to type; a
clean test needs a high-dim conv or low-dim SPD latent. (3) Single dataset (2a), 2 backbones, LOSO with
8 source subjects (caps LDA/Fisher at 7 directions). (4) The DG-null is on the frozen-feature pilot — it
shows leakage *removal per se* does not buy DG, not that end-to-end training cannot. (5) Task preserved on
both backbones (no task-destruction artifact). (6) TSMNet "not removable" robust within the LDA cap
(k=7 Fisher deletion still leaves 0.92–0.98).

## Next options
- **Write up** (Phase 2 + Phase 3): the consolidated story is measurement→control gap, now with a
  representation/capacity-mediated removability axis and a removable-but-useless result. (Recommended.)
- **Break the dim↔type confound** (only if a cleaner generality claim is wanted): a high-dim conv latent
  or a low-dim SPD latent on 2a — one more frozen pilot.

Artifacts: `results/.../BNCI2014_001_EEGNet_LOSO/{ablation,adversarial}_report_seed*.json`,
`results/.../lpc_collapse_curves/EEGNet/raw_lpc_sub*_seed*.{json,npz}`.

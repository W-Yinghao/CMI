# Fork 2 (source-rich environment discovery) — FINAL VERDICT

**Branch `science-source-rich-v1`. Status: CLOSED — PARTIAL SUCCESS. No further positive-expansion, no
seed-robustness, no environment-audit expansion, no tuning.** This document is the closeout; the phase records are
`SOURCE_RICH_ENV_DESIGN.md`, `SOURCE_RICH_PHASE0_META.md`, `SOURCE_RICH_PHASE1A_VERDICT.md` (Lee dev),
Phase 1B (Cho confirm, commit 69ec85c), `SOURCE_RICH_PHASE1C_VERDICT.md` (TSMNet, commit 8174483).

Provenance held throughout: `D_nuis = z` is the KNOWN INJECTED nuisance, NOT EEG subject identity; environments
E0–E5 are separate from `D_nuis`; params were frozen on Lee development before Cho confirmation and unchanged for
the TSMNet backbone check; target labels were audit-only (`selection_uses_target: false`); this is SEMI-SYNTHETIC,
not a real-EEG erasure target-gain result.

---

## 3.1 Scientific question

```
Can better source environments make deployment-relevant benefit source-visible,
allowing source-only accept without target labels?
```
(Motivation: V2 established a source-only acceptance CEILING — when the target-beneficial shift is source-INVISIBLE,
a strict source-only gate must refuse. Proposition 2 says the ceiling can be crossed WITHOUT target labels IF the
shift is represented among source ENVIRONMENTS. Fork 2 is the finite constructive test of that "if".)

## 3.2 Positive result

```
In semi-synthetic source-rich World A, EEGNet Lee→Cho confirms:
E_oracle accepts safe target-beneficial erasers;
E2 covariance_cluster recovers the oracle acceptance source-only;
random partitions do not reproduce it.
```
Concretely (EEGNet, frozen frac 0.4/0.3/0.3, thresholds safety≤0.02 / benefit>0.01, k=8, seed0, first5):
Lee E_oracle 6 accepts / E2 covariance_cluster 5 accepts; Cho E_oracle 6 / E2 covariance_cluster 6; margin /
augmentation / random do not recover. This is a clean **source-only positive witness** for Proposition 2's
constructive side: when the target regime is REPRESENTED among source regimes, a leave-one-ENVIRONMENT-out benefit
(over a source-only DISCOVERED covariance environment) safely licenses accept, while leave-one-SUBJECT-out (E0)
misses it. First source-only positive in the project. (Caveat carried into 3.4.)

## 3.3 High-dimensional limitation

```
TSMNet does not cleanly support the same source-rich discovery:
E_oracle is fragile/near-threshold;
E2/E4/E5 do not recover;
covariance-regime signal collapses in the high-dimensional latent.
```
Concretely (TSMNet z_dim=210, m=42, same frozen params; commit 8174483; adversarially verified wf_650c9d1f-aec,
high confidence, integrity clean 0 fail/skip): Lee E_oracle accepts a subset (LEACE benefit_lcb +0.029..+0.031,
safe, target-good) = Case B, but E2 covariance benefit_lcb goes NEGATIVE (−0.015) and max discovered-env benefit_lcb
over everything is +0.00014 ≪ +0.010; Cho E_oracle is a SAFE target-beneficial THRESHOLD near-miss (benefit_lcb
+0.0065..+0.0074 < +0.010 → abstain, yet safe td_ucb ≤0.015 and genuine target +0.033..+0.041, LCB +0.021..+0.028,
all 5 folds positive, bootstrap-stable). E2/E4/E5/random = 0 accepts on both. The covariance↔regime alignment
COLLAPSES at high dim: subject-level AMI(covariance_cluster, true regime) = **+0.365 EEGNet vs −0.011 TSMNet**
(chance). The gate did NOT misfire — **0 harmful, 0 discovered-env accepts** on both TSMNet datasets; the problem is
DISCOVERABILITY of the source-rich construction, not gate over-acceptance. The EEGNet covariance discovery success
partly relied on the low-dimensional latent and the construction (the m-dim injected nuisance is a large fraction of
16 dims; at 252 augmented dims it is diluted and the covariance-feature cap excludes the tail nuisance block).

## 3.4 Final interpretation

```
Proposition 2 is constructively supported but not generally solved:
source-only acceptance is possible when deployment-relevant shift is source-visible
and discoverable, but discovering such environments is representation-dependent.
```
Preferred one-line statement of the Fork 2 result:
> **Source-rich sufficiency is real but conditional: when the right source environments are known or discoverable,
> source-only accept is possible; however, environment discovery itself is fragile and representation-dependent.**

Do NOT write:
> ~~source-rich environments generally solve source-only acceptance.~~

## 3.5 Next implication

```
Because source-only discovery is fragile, the next scientific frontier is target-information:
how much target information is needed to safely cross the source-only ceiling?
```
This motivates Fork 1 (branch `science-target-info-v1`, design-locked in `TARGET_INFORMATION_FRONTIER_DESIGN.md`):
source-only accept is possible under source-rich environments, but discovering those environments is fragile, so
target information is the natural next variable to study.

---

## Fork 2 final status (locked)

```
Fork 2 result:
  PARTIAL SUCCESS.

  It demonstrates the constructive side of Proposition 2:
  if the target-relevant shift is represented among source environments,
  source-only accept can be safe.

  But it also shows environment discovery is not robust:
  EEGNet covariance clustering works in the constructed setting;
  TSMNet does not.
```

## Mandatory caveats (keep all)
```
semi-synthetic ; construction-favorable ; D_nuis known ; 1 seed / 5 folds ;
EEGNet source-rich positive is a low-dim constructive existence proof, NOT proof real EEG shifts are
covariance-discoverable ; TSMNet shows the discovery side does not scale to deep latents ;
this is NOT a real-EEG erasure target-gain result.
```
The uploaded PDF is an early 2a-only snapshot (still carries the old single-dataset limitation and earlier result
structure); Fork 2 results are NOT to be force-fit into that old draft. See [[tos-cmi-method-deepen-v2]].

# Source-rich Phase 1C (TSMNet backbone check) --- VERDICT

## Provenance (required)
- `D_nuis = z` is the KNOWN INJECTED nuisance label, NOT the EEG subject identity. Environments E0-E5 are separate
  from `D_nuis`.
- Parameters are FROZEN from Lee development / Cho confirmation (EEGNet): frac 0.4/0.3/0.3, alpha {0.5,1.0,2.0},
  thresholds safety<=0.02 & benefit>0.01, cluster k=8, environment definitions unchanged. **No retuning.**
- Cho2017 was already the EEGNet confirmation dataset; the TSMNet run is a BACKBONE ROBUSTNESS check, NOT a second
  round of tuning.
- nuisance dim scaled by z_dim, same rule as V2: `m = _nuisance_m(z_dim=210, "fraction_of_z_dim", 0.20, 4) = 42`.
  TSMNet base latent = 210; augmented latent = 210 + 42 = 252.
- Scope: Lee2019_MI + Cho2017, TSMNet only, seed 0, first 5 folds, source-rich World A only.
- Jobs 887160 (Lee) / 887161 (Cho), 720 tasks each, CPU SLURM, env `icml`.

## Integrity (verified, adversarial workflow wf_650c9d1f-aec, high confidence)
n_fail=0, n_skip=0 on both; 720 tasks -> 144 (env,interv,alpha) cells each with exactly 5 folds; all benefit_mean
finite; n_env_mean sane (subject 53/51, oracle 3, cov/margin/aug 8, random 3). The loky "worker stopped" warning
dropped ZERO tasks. **The negatives are genuine, not silent task loss.**

## Result: source-rich positive does NOT cleanly generalize to TSMNet (partial / backbone-specific)

### Lee2019_MI x TSMNet = Case B (oracle accepts with KNOWN env; source-only discovery FAILS)
- E_oracle (leave-one-REGIME-out, DIAGNOSTIC/privileged): 5 ACCEPT = 3 target-good (LEACE a0.5/1.0/2.0,
  benefit_lcb +0.029/+0.030/+0.031, safe td_ucb +0.011..+0.012, target +0.029 LCB +0.011..+0.012) + 2 benign-
  boundary (RLACE a0.5/1.0, eps_coverage slack), 0 HARMFUL. **Prop 2 with KNOWN environments HOLDS on Lee-TSMNet.**
- E2 covariance_cluster (source-only DISCOVERED): benefit_lcb NEGATIVE (-0.015/-0.015/-0.013) for the SAME LEACE
  that the oracle accepted. Max discovered-env benefit_lcb over ALL discovered envs/interventions = +0.00014
  (<< +0.010 gate). E4 margin / E5 augmentation / random: 0 accepts. **Source-only discovery robustly fails.**

### Cho2017 x TSMNet = auto-label Case C (0 accepts), but PRECISELY a SAFE target-beneficial THRESHOLD near-miss
- E_oracle LEACE: benefit_lcb +0.0065/+0.0073/+0.0074 (benefit_mean +0.0096/+0.0108/+0.0108) sits ~0.0026 BELOW
  the +0.010 gate -> ABSTAIN. But the intervention is SAFE (td_ucb <=0.0151) and GENUINELY target-beneficial:
  target ΔbAcc +0.041/+0.035/+0.033, LCB +0.028/+0.022/+0.021 (>+0.01), every one of 5 folds positive.
- The abstain is bootstrap-STABLE below 0.010 (20 reseeds, std ~1e-4, never crosses). Mechanism: within each fold
  two oracle-env LOEO benefits are strongly positive and the third is consistently negative (~-0.018..-0.020),
  a poor pseudo-target that pins the source proxy mean at the threshold. This is the source-only LOEO estimator
  being CONSERVATIVE at high dim, NOT an absence of benefit.
- **The report's auto-label "Case C: world construction failed; redesign world-gen" is OVERRULED here: the world
  produced a real safe benefit; retuning is forbidden and unwarranted. It is a correct-but-conservative abstain.**

### Gate safety INTACT at 210-dim (verified)
0 HARMFUL accepts (ACCEPT & target hi < -0.01) and 0 discovered-env accepts on BOTH TSMNet datasets. random
reproduces nothing. The gate never misfired at high dim.

## Why discovery fails on TSMNet: the EEGNet covariance "recovery" was a LOW-DIM ARTIFACT
Subject-level AMI(covariance_cluster labels, true regime), one Lee fold per backbone, averaged over alphas
(oracle sanity AMI = 1.000 both backbones):
- EEGNet: AMI **+0.365**, purity 0.73 (>> 0.396 aligned-fraction floor) -> covariance clusters DO align with regime.
- TSMNet: AMI **-0.011** (chance), purity 0.52 (~floor) -> covariance clusters are orthogonal to regime.
At z_dim=16 the m=4 injected nuisance is a large fraction of the space and organizes the per-subject covariance
features; at z_dim=210 the m=42 block is diluted (and the covariance-feature cap `triu[:200]` falls entirely on
the real-EEG dims 0-209, excluding the tail nuisance block 210-251), so covariance clustering tracks real subject
structure orthogonal to the randomly-assigned regime. The EEGNet E2 recovery does not persist at deep latents.
(Raw AMI from a single-fold KMeans run; corroborated indirectly via code + the downstream gate consequence.)

## EEGNet -> TSMNet contrast (accepts, excl. DIAGNOSTIC)
```
           EEGNet oracle | EEGNet cov  ||  TSMNet oracle | TSMNet cov
Lee2019       6          |   5         ||     5          |   0
Cho2017       6          |   6         ||     0          |   0
```
Source-only covariance discovery recovered the oracle accept on EEGNet (5-6 accepts) but NEVER on TSMNet (0/0).

## Honest overall verdict
The source-rich positive is essentially EEGNet-specific under this construction:
1. Even the PRIVILEGED E_oracle (Prop 2 with known environments) is FRAGILE at high dim -- clean on Lee-TSMNet,
   a safe real-benefit near-miss on Cho-TSMNet.
2. Source-only environment DISCOVERY (E2 covariance) does NOT transfer to TSMNet at all; the EEGNet recovery was
   (at least partly) a low-dim artifact.
3. Gate SAFETY is fully intact at high dim (0 harmful, 0 false/discovered accepts).
This is consistent with the V2 TSMNet World A caveat (high-dim latents dilute the constructed nuisance).

## Mandatory caveats (keep all)
```
semi-synthetic ; construction-favorable ; D_nuis known ; 1 seed / 5 folds ;
EEGNet source-rich positive is a low-dim constructive existence proof, NOT proof real EEG shifts are
covariance-discoverable ; TSMNet shows the discovery side does not scale to deep latents.
```
Do NOT write: "source-rich works across backbones" / "covariance discovery scales to deep latents" /
"Cho-TSMNet is a negative/world failure" / "the gate under-protected on TSMNet".

## Decision (PM): seed robustness / environment audit / stop Fork 2 positive-expansion
The environment-discovery audit is effectively already answered (E2 = low-dim artifact, mechanism understood);
further discovered-env search risks fishing (forbidden). Recommendation carried to the PM report; hold for go.

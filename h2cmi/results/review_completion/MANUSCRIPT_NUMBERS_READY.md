# Manuscript Numbers Ready

> **CANONICAL W1 STATUS:** The only current W1 source is
> `FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md` at P10. All contiguous-split W1
> numbers are superseded. No old-split W1 value may be used as a current
> result.

This file is the writer-facing digest for the review-completion package. Values are balanced accuracy unless otherwise stated. Intervals are 95% percentile bootstrap CIs unless the table explicitly says no CI was computed.

## Canonical Repaired-Split MI/W1 Four-Branch Decomposition

Status: repaired-split confirmatory P7 result, frozen by P10. Source artifacts:
`w1_repaired_h2cmi_results.csv`, `w1_repaired_h2cmi_four_branch_ci.csv`, and
`FINAL_REPAIRED_W1_EVIDENCE_FREEZE.json`. Source seeds are averaged within
target subject before the subject-cluster bootstrap.

| quantity | value | 95% CI | n | wording boundary |
|---|---:|---:|---:|---|
| BA(I, Unif) | 0.6736216 | [0.6565223, 0.6912903] | 115 | canonical identity/uniform branch mean |
| BA(I, pi_J) | 0.6639614 | [0.6460370, 0.6822751] | 115 | canonical identity/joint-prior branch mean |
| BA(T_J, Unif) | 0.6814493 | [0.6634068, 0.6995276] | 115 | canonical joint-geometry/uniform branch mean |
| BA(T_J, pi_J) | 0.6770274 | [0.6588374, 0.6950935] | 115 | canonical full-joint branch mean |
| G | +0.0078277 | [+0.0026504, +0.0132705] | 115 | small positive geometry component on the repaired split |
| P | -0.0096602 | [-0.0136878, -0.0057326] | 115 | negative decision-prior component under balanced accuracy |
| I_int | +0.0052383 | [+0.0012029, +0.0093253] | 115 | positive interaction component |
| full joint delta | +0.0034058 | [-0.0021352, +0.0091774] | 115 | CI spans zero; do not claim full-joint improvement |

## Canonical Repaired-Split MI Heterogeneity

Status: repaired-split confirmatory P7 heterogeneity analysis. Source artifact:
`w1_repaired_h2cmi_dataset_heterogeneity_ci.csv`. The repaired result is small
and is not driven by the old contiguous Cho2017 evaluation blocks.

| summary | G | P | I_int | full joint delta | wording boundary |
|---|---:|---:|---:|---:|---|
| BNCI2014_001, n=9 | +0.0128601 [-0.0041152, +0.0339506] | -0.0087449 [-0.0200617, +0.0015432] | +0.0066872 [-0.0056584, +0.0200617] | +0.0108025 [-0.0072016, +0.0339506] | small dataset; all listed CIs overlap zero |
| Cho2017, n=52 | +0.0067521 [-0.0004594, +0.0139105] | -0.0108761 [-0.0152674, -0.0064525] | +0.0073504 [+0.0022115, +0.0125219] | +0.0032265 [-0.0030350, +0.0096052] | repaired balanced evaluation blocks; no legacy single-class effect |
| Lee2019_MI, n=54 | +0.0080247 [-0.0001235, +0.0167901] | -0.0086420 [-0.0161728, -0.0014815] | +0.0029630 [-0.0038272, +0.0096296] | +0.0023457 [-0.0072840, +0.0122222] | small geometry component; negative prior component |
| subject-weighted, n=115 | +0.0078277 [+0.0026504, +0.0132705] | -0.0096602 [-0.0136878, -0.0057326] | +0.0052383 [+0.0012029, +0.0093253] | +0.0034058 [-0.0021352, +0.0091774] | canonical aggregate |
| dataset-macro, n=3 | +0.0092123 [+0.0024340, +0.0172130] | -0.0094210 [-0.0140882, -0.0049499] | +0.0056669 [+0.0007107, +0.0110245] | +0.0054582 [-0.0017396, +0.0142792] | dataset-equal diagnostic; full-joint CI spans zero |

## Official SPDIM Repaired-Split Three-Seed Baseline

Status: complete official repaired-split same-split baseline from P9. Source
artifacts: `spdim_w1_repaired_three_seed_results.csv`,
`spdim_w1_repaired_three_seed_method_ci.csv`, and
`spdim_w1_repaired_three_seed_contrast_ci.csv`.

| method | subject-weighted bAcc | 95% CI |
|---|---:|---:|
| source-only TSMNet | 0.5419807 | [0.5334460, 0.5508986] |
| RCT | 0.6471643 | [0.6304244, 0.6637672] |
| SPDIM geodesic | 0.6444235 | [0.6277308, 0.6610388] |
| SPDIM bias | 0.6431530 | [0.6264633, 0.6599215] |

Paired subject-weighted contrasts:

| contrast | estimate | 95% CI |
|---|---:|---:|
| RCT - source-only | +0.1051836 | [+0.0918437, +0.1189907] |
| SPDIM geodesic - source-only | +0.1024428 | [+0.0894991, +0.1161096] |
| SPDIM bias - source-only | +0.1011723 | [+0.0882946, +0.1148618] |
| SPDIM geodesic - RCT | -0.0027407 | [-0.0046506, -0.0008260] |
| SPDIM bias - RCT | -0.0040113 | [-0.0072577, -0.0007150] |

The official SPDIM baseline is complete. RCT improves over source-only;
SPDIM geodesic and bias do not improve over RCT. This is not an equivalence or
noninferiority result. H2CMI and SPDIM are same-split full-pipeline results,
not an adapter-only controlled comparison.

## Superseded Legacy Contiguous-Split Diagnostic

The old W1 analysis reported `G = +0.0604` and Cho2017 `G = +0.1227`. These
values came from the quarantined contiguous split, where Cho2017 evaluation
blocks were single-class. They are retained only as historical diagnostics and
must not be cited as current or confirmatory W1 results. Historical source
artifacts are `four_branch_complete_ci.csv/json` and
`mi_dataset_heterogeneity_complete_ci.csv`.

## Sleep Four-Branch Decomposition

Status: current Sleep/W2 confirmatory reanalysis from frozen raw rows. Source
artifacts: `four_branch_complete_ci.csv` and `four_branch_complete_ci.json`.
Bootstrap cluster: sleep subject/night pair.

| panel | quantity | value | 95% CI | n | wording boundary |
|---|---|---:|---:|---:|---|
| Sleep_W2_primary | identity_uniform | 0.6572 | [0.6341, 0.6784] | 75 | branch mean |
| Sleep_W2_primary | identity_joint_prior | 0.5134 | [0.4890, 0.5372] | 75 | branch mean |
| Sleep_W2_primary | joint_geometry_uniform | 0.6371 | [0.6113, 0.6613] | 75 | branch mean |
| Sleep_W2_primary | joint_geometry_joint_prior | 0.5521 | [0.5266, 0.5764] | 75 | branch mean |
| Sleep_W2_primary | G | -0.0201 | [-0.0407, +0.0010] | 75 | geometry does not explain the main sleep effect |
| Sleep_W2_primary | P | -0.1439 | [-0.1593, -0.1285] | 75 | decision-prior harm under balanced accuracy |
| Sleep_W2_primary | I_int | +0.0588 | [+0.0425, +0.0758] | 75 | interaction offsets part of prior harm |
| Sleep_W2_primary | G+P+I_int | -0.1052 | [-0.1250, -0.0851] | 75 | primary sleep total delta |
| Sleep_W2_secondary | G | -0.0231 | [-0.0440, -0.0026] | 75 | secondary protocol check |
| Sleep_W2_secondary | P | -0.1300 | [-0.1495, -0.1103] | 75 | secondary protocol check |
| Sleep_W2_secondary | I_int | +0.0238 | [+0.0031, +0.0449] | 75 | secondary protocol check |
| Sleep_W2_secondary | G+P+I_int | -0.1293 | [-0.1524, -0.1052] | 75 | secondary protocol check |

## Sleep Replay And Per-Stage Recall

Status: accepted deterministic replay diagnostic. Source artifacts: `sleep_replay_hash_audit.md`, `sleep_branch_confusion_matrices.json`, `sleep_per_stage_recall.csv`. No CI is attached to per-stage recall deltas; they are supplement diagnostics, not the main claim.

Replay gate: 75 primary units, decomposition residual `1.85e-17`; terminal comparison accepted (`G -0.020007 -> -0.020125`, `P -0.143848 -> -0.143875`).

Primary per-stage recall deltas vs `identity_uniform`:

| branch | W | N1 | N2 | N3 | REM | wording boundary |
|---|---:|---:|---:|---:|---:|---|
| identity_uniform | 0.9244 (+0.0000) | 0.2881 (+0.0000) | 0.8854 (+0.0000) | 0.5342 (+0.0000) | 0.6386 (+0.0000) | anchor |
| identity_joint_prior | 0.9522 (+0.0278) | 0.0065 (-0.2816) | 0.8171 (-0.0683) | 0.4833 (-0.0509) | 0.3073 (-0.3313) | prior harm is stage-specific under balanced accuracy |
| joint_geometry_uniform | 0.9245 (+0.0001) | 0.2127 (-0.0754) | 0.4988 (-0.3866) | 0.9622 (+0.4280) | 0.6397 (+0.0011) | geometry reshuffles stage recalls |
| joint_geometry_joint_prior | 0.9586 (+0.0342) | 0.0083 (-0.2798) | 0.5549 (-0.3305) | 0.8823 (+0.3481) | 0.4080 (-0.2306) | diagnostic only |
| fixed_iterative_geometry_uniform | 0.9042 (-0.0202) | 0.2720 (-0.0161) | 0.5126 (-0.3728) | 0.9616 (+0.4274) | 0.6787 (+0.0401) | diagnostic only |
| fixed_reference_oneshot_uniform | 0.9070 (-0.0174) | 0.3162 (+0.0281) | 0.6288 (-0.2566) | 0.9324 (+0.3982) | 0.7324 (+0.0938) | diagnostic only |
| pooled_uniform | 0.8612 (-0.0632) | 0.2600 (-0.0281) | 0.7736 (-0.1118) | 0.7981 (+0.2639) | 0.6386 (+0.0000) | diagnostic only |
| latent_im_diag_uniform | 0.9356 (+0.0112) | 0.0837 (-0.2044) | 0.5914 (-0.2940) | 0.9415 (+0.4073) | 0.6818 (+0.0432) | internal comparator, not SPDIM |
| source_recolored_ea | 0.9221 (-0.0023) | 0.2875 (-0.0006) | 0.8839 (-0.0015) | 0.5190 (-0.0152) | 0.6364 (-0.0022) | sensor-space diagnostic |

## V2P Corrected Reanalysis

Status: corrected reanalysis from frozen Wave0 raw artifacts. Source artifacts: `v2p_corrected_unit_key_audit.md`, `v2p_corrected_grid_summary.csv`, `v2p_corrected_method_summary.csv`, `v2p_corrected_paired_contrasts.csv`, `v2p_corrected_cluster_bootstrap.json`.

Executed q-grid: `{0.1,...,0.9}`. Corrected unit key: `(dataset, pair, subject, target_session, source_seed, method)`; repeated BNCI2014_004 transitions remain distinct. Bootstrap cluster: `(dataset, subject)`, 72 clusters. Wording boundary: displacement is not utility; oracle-label diagnostic is not deployable.

Endpoint and center utility values:

| method | q | bAcc [CI] | ord-acc unif [CI] | ord-acc oracle-q [CI] | status |
|---|---:|---:|---:|---:|---|
| pooled | 0.1 | 0.5779 [0.5592, 0.5974] | 0.5733 [0.5453, 0.6016] | 0.8488 [0.8262, 0.8690] | corrected |
| pooled | 0.5 | 0.5853 [0.5642, 0.6081] | 0.5853 [0.5642, 0.6081] | 0.5853 [0.5642, 0.6081] | corrected |
| pooled | 0.9 | 0.5698 [0.5528, 0.5876] | 0.4620 [0.4338, 0.4908] | 0.8649 [0.8530, 0.8758] | corrected |
| fixed_reference_oneshot | 0.1 | 0.5197 [0.5103, 0.5305] | 0.7082 [0.6498, 0.7626] | 0.7307 [0.6760, 0.7812] | corrected |
| fixed_reference_oneshot | 0.5 | 0.5245 [0.5134, 0.5378] | 0.5245 [0.5134, 0.5378] | 0.5245 [0.5134, 0.5378] | corrected |
| fixed_reference_oneshot | 0.9 | 0.5188 [0.5104, 0.5286] | 0.5151 [0.4419, 0.5881] | 0.5484 [0.4760, 0.6199] | corrected |
| fixed_iterative | 0.1 | 0.5040 [0.5004, 0.5086] | 0.7141 [0.6561, 0.7687] | 0.7139 [0.6558, 0.7685] | corrected |
| fixed_iterative | 0.5 | 0.5027 [0.5000, 0.5068] | 0.5027 [0.5000, 0.5068] | 0.5027 [0.5000, 0.5068] | corrected |
| fixed_iterative | 0.9 | 0.5028 [0.5004, 0.5065] | 0.4980 [0.4217, 0.5734] | 0.4979 [0.4215, 0.5732] | corrected |
| joint | 0.1 | 0.5034 [0.5003, 0.5075] | 0.7154 [0.6574, 0.7697] | 0.7152 [0.6571, 0.7695] | corrected |
| joint | 0.5 | 0.5027 [0.5000, 0.5068] | 0.5027 [0.5000, 0.5068] | 0.5027 [0.5000, 0.5068] | corrected |
| joint | 0.9 | 0.5023 [0.5003, 0.5054] | 0.4980 [0.4216, 0.5733] | 0.4978 [0.4214, 0.5732] | corrected |
| oracle_label_conditional | 0.1 | 0.5062 [0.5016, 0.5119] | 0.9010 [0.9003, 0.9020] | 0.9007 [0.9001, 0.9014] | oracle diagnostic |
| oracle_label_conditional | 0.5 | 0.5471 [0.5292, 0.5668] | 0.5471 [0.5292, 0.5668] | 0.5471 [0.5292, 0.5668] | oracle diagnostic |
| oracle_label_conditional | 0.9 | 0.5052 [0.5014, 0.5103] | 0.9007 [0.8999, 0.9017] | 0.9004 [0.9001, 0.9009] | oracle diagnostic |

Displacement from `q=0.5`:

| method | q | embed disp [CI] | translation disp [CI] | logscale disp [CI] |
|---|---:|---:|---:|---:|
| pooled | 0.1 | 0.0803 [0.0556, 0.1096] | 0.2996 [0.2615, 0.3432] | 0.3964 [0.3569, 0.4444] |
| pooled | 0.5 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| pooled | 0.9 | 0.0807 [0.0550, 0.1117] | 0.2852 [0.2482, 0.3271] | 0.3650 [0.3315, 0.4031] |
| fixed_reference_oneshot | 0.1 | 0.3792 [0.2867, 0.4832] | 0.3577 [0.2717, 0.4530] | 0.0574 [0.0426, 0.0743] |
| fixed_reference_oneshot | 0.5 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| fixed_reference_oneshot | 0.9 | 0.4694 [0.3607, 0.5840] | 0.4508 [0.3464, 0.5627] | 0.0655 [0.0484, 0.0859] |
| fixed_iterative | 0.1 | 0.6979 [0.4711, 0.9514] | 0.6513 [0.4430, 0.8849] | 0.0954 [0.0723, 0.1209] |
| fixed_iterative | 0.5 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| fixed_iterative | 0.9 | 0.8446 [0.6040, 1.1047] | 0.7954 [0.5706, 1.0391] | 0.1027 [0.0789, 0.1290] |
| joint | 0.1 | 0.7050 [0.4776, 0.9598] | 0.6580 [0.4490, 0.8909] | 0.0958 [0.0726, 0.1214] |
| joint | 0.5 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| joint | 0.9 | 0.8447 [0.6041, 1.1053] | 0.7955 [0.5702, 1.0389] | 0.1022 [0.0786, 0.1285] |
| oracle_label_conditional | 0.1 | 2.5362 [2.4547, 2.6200] | 2.4079 [2.3267, 2.4911] | 0.2135 [0.1921, 0.2368] |
| oracle_label_conditional | 0.5 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| oracle_label_conditional | 0.9 | 1.8154 [1.7468, 1.8908] | 1.8475 [1.7772, 1.9243] | 0.2141 [0.1970, 0.2338] |

## Geometry Capacity

Status: bounded operator-family stress. Existing stress is confirmatory from frozen W1 artifacts; off-diagonal stress is an additive exploratory/supplemental GPU run. Sources: `geometry_capacity_existing_ci.csv`, `geometry_capacity_offdiagonal_results.csv`, `offdiag_completion_audit.md`, raw `results/h2cmi/review_completion_offdiag/*.jsonl`.

Wording boundary: say "bounded operator-family stress"; do not say "diagonal geometry is adequate" or "universal montage robustness." Montage-layout / cross-montage remapping remains untested.

Existing frozen stress, cluster `(dataset,subject)`, n=72:

| perturbation | best full-cov minus best diagonal [CI] | identity drop [CI] | best diagonal operator | best full/sensor operator | neg-change rates best diagonal at 0/-0.01/-0.02 | neg-change rates best full at 0/-0.01/-0.02 |
|---|---:|---:|---|---|---:|---:|
| none | -0.0033 [-0.0081, +0.0013] | anchor | pooled_uniform | coral_latent | 0.5278 / 0.3611 / 0.2222 | 0.5278 / 0.3889 / 0.2500 |
| reref | -0.0063 [-0.0115, -0.0011] | +0.0375 | pooled_uniform | coral_latent | 0.3611 / 0.2917 / 0.1389 | 0.3333 / 0.2778 / 0.2222 |
| gain | -0.0036 [-0.0097, +0.0024] | +0.0209 | latent_im_diag_uniform | source_recolored_ea | 0.3194 / 0.2222 / 0.1389 | 0.3194 / 0.2639 / 0.1944 |
| dropout | -0.0001 [-0.0068, +0.0064] | +0.0280 | pooled_uniform | source_recolored_ea | 0.2639 / 0.1806 / 0.0972 | 0.2500 / 0.1528 / 0.1389 |

Off-diagonal stress, cluster `(dataset,pair,subject)`, n=90:

| perturbation | best full-cov minus best diagonal [CI] | identity BA | best diagonal operator [BA CI] | best full/sensor operator [BA CI] | neg-change rates best diagonal at 0/-0.01/-0.02 | neg-change rates best full at 0/-0.01/-0.02 |
|---|---:|---:|---|---|---:|---:|
| rotation | +0.0000 [-0.0066, +0.0071] | 0.5044 | pooled_uniform 0.5038 [0.4942, 0.5135] | source_recolored_ea 0.5032 [0.4934, 0.5132] | 0.5778 / 0.4444 / 0.3444 | 0.5222 / 0.4222 / 0.3333 |
| mixing | -0.0043 [-0.0090, +0.0002] | 0.5780 | latent_im_diag_uniform 0.5851 [0.5611, 0.6101] | coral_latent 0.5843 [0.5606, 0.6083] | 0.4111 / 0.3333 / 0.2333 | 0.4556 / 0.2444 / 0.1222 |
| strong_reref | -0.0046 [-0.0100, +0.0004] | 0.5489 | latent_im_diag_uniform 0.5679 [0.5500, 0.5875] | coral_latent 0.5632 [0.5438, 0.5836] | 0.3333 / 0.2333 / 0.1556 | 0.3667 / 0.3111 / 0.2222 |
| block_mixing | +0.0009 [-0.0054, +0.0069] | 0.5025 | pooled_uniform 0.5051 [0.4952, 0.5155] | source_recolored_ea 0.5100 [0.5006, 0.5199] | 0.5111 / 0.3889 / 0.2556 | 0.4778 / 0.4111 / 0.3222 |

## Encoder / Backbone

Status: implementation audit from code and bundle sidecars. Sources: `encoder_backbone_details.md`, `encoder_backbone_details.json`, `h2cmi/config.py`, `h2cmi/models/encoder.py`, `h2cmi/data/real_eeg.py`, `h2cmi/data/sleep_eeg.py`, `h2cmi/tta/class_conditional.py`, bundle sidecars under `results/h2cmi/*_bundles`.

Experiment text-ready description: MI uses MOABB binary motor-imagery tensors with deterministic preprocessing and per-trial channel z-scoring; Sleep uses Sleep-EDF Sleep-Cassette 30s epochs over W/N1/N2/N3/REM with per-epoch z-scoring. The frozen source encoder is `H2Encoder`, combining a temporal EEGNet-like branch, SPD covariance/log-tangent branch, graph/electrode-set branch, fused MLP, split `z_c/z_n`, and optional near-identity canonicalizer. The latent defaults are `z_c_dim=32`, `z_n_dim=16`, `fuse_hidden=128`, `temporal_filters=8`, `spd_rank=8`, `graph_hidden=16`, and bands `[4,8]`, `[8,13]`, `[13,30]`, `[30,45]` Hz. Class-conditionals use a Student-t mixture density with one component/class by default, low-rank+diagonal covariance rank 4, df 8, and eig_floor 1e-2. TTA geometry uses a diagonal affine map `z = exp(a)*u + b` with EM-style prior/transform fitting.

Missing/unrecovered field: exact per-run H2CMI optimizer state was not decoded.
The official SPDIM repaired-split configuration and runtime provenance are now
recorded in the P9 protocol, result summary, and shard audits; official
pretrained demo weights were not used.

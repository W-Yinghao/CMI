# C8 — BNCI2014_001 LOSO seeds [0, 1, 2] (native K1/K2)

> **BNCI2014-001 minimum-seed K1/K2 run. Seeds [0,1,2] meet the configured K2 minimum. This is not yet the full 5-seed manifest sweep.**

## run

- dataset: **BNCI2014-001** · targets: **9** · seeds: **[0, 1, 2]** · fold-runs: **27** · levels: **[0, 1]**
- methods: `ERM, OACI, global_lpc, uniform` · bootstrap: **full** · K1 permutations: **2000** · staged execution: **true**

## verification

- artifacts_complete: **27/27** · deep_verified: **True** · target_fit_ids_empty: **True**
- decision_payloads_present: **27/27 × 2 levels** (k1.json + k1.npz + k2.json each)
- provenance_groups: **2** · approved_provenance_exception: **True**

## K1 — held-out audit permutation null (pre-registered PER-FOLD; sweep line is descriptive)

- level 0: detected **4/27**, stop 23/27, other 0 · Δ mean -0.0379, median -0.0297, min -0.2068, max +0.0257
- level 1: detected **7/27**, stop 20/27, other 0 · Δ mean -0.0193, median -0.0070, min -0.1145, max +0.0404

- **overall**: n_tests 54, detected (uncorrected) 11, stop 43, fraction_detected +0.204
- Δ overall: mean -0.0286, median -0.0158, min -0.2068, max +0.0404
- **multiplicity control** (α=0.05, 54 one-sided tests): Bonferroni survivors **0** (thr +0.00093), BH-FDR survivors **0**
- **K1 sweep status (descriptive): `stop_no_detectable_heldout_leakage_reduction`** — K1 is pre-registered PER-FOLD; no sweep-level K1 threshold is frozen. This sweep line is DESCRIPTIVE (multiplicity-corrected via BH-FDR), not a frozen go/no-go. The frozen sweep decision is K2.

| seed | target | level | K1 | Δ (OACI−ERM) | p_lower | p_two |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 1 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0144 | 0.3663168415792104 | 0.7411294352823589 |
| 0 | 2 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0115 | 0.1794102948525737 | 0.4057971014492754 |
| 0 | 3 | 0 | stop_no_detectable_heldout_leakage_reduction | +0.0024 | 0.5902048975512244 | 0.8120939530234883 |
| 0 | 4 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0117 | 0.1294352823588206 | 0.2813593203398301 |
| 0 | 5 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0499 | 0.1359320339830085 | 0.2653673163418291 |
| 0 | 6 | 0 | stop_no_detectable_heldout_leakage_reduction | +0.0100 | 0.6781609195402298 | 0.6531734132933533 |
| 0 | 7 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0324 | 0.19290354822588707 | 0.39280359820089955 |
| 0 | 8 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0735 | 0.05997001499250375 | 0.11544227886056972 |
| 0 | 9 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0194 | 0.335832083958021 | 0.6756621689155422 |
| 1 | 1 | 0 | leakage_reduction_detected | -0.1684 | 0.031484257871064465 | 0.06996501749125437 |
| 1 | 2 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0455 | 0.09095452273863068 | 0.1934032983508246 |
| 1 | 3 | 0 | stop_no_detectable_heldout_leakage_reduction | +0.0055 | 0.6416791604197901 | 0.6941529235382309 |
| 1 | 4 | 0 | stop_no_detectable_heldout_leakage_reduction | +0.0174 | 0.9345327336331835 | 0.12493753123438281 |
| 1 | 5 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0160 | 0.25337331334332835 | 0.49225387306346824 |
| 1 | 6 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0483 | 0.15692153923038482 | 0.29485257371314344 |
| 1 | 7 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0322 | 0.19940029985007496 | 0.4172913543228386 |
| 1 | 8 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0297 | 0.1619190404797601 | 0.3188405797101449 |
| 1 | 9 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0456 | 0.17791104447776113 | 0.3498250874562719 |
| 2 | 1 | 0 | stop_no_detectable_heldout_leakage_reduction | +0.0257 | 0.7241379310344828 | 0.5282358820589705 |
| 2 | 2 | 0 | leakage_reduction_detected | -0.0470 | 0.02498750624687656 | 0.051974012993503245 |
| 2 | 3 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0235 | 0.22338830584707647 | 0.4767616191904048 |
| 2 | 4 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0138 | 0.08395802098950525 | 0.15892053973013492 |
| 2 | 5 | 0 | leakage_reduction_detected | -0.2068 | 0.0024987506246876563 | 0.0034982508745627187 |
| 2 | 6 | 0 | leakage_reduction_detected | -0.1117 | 0.0074962518740629685 | 0.012993503248375811 |
| 2 | 7 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0424 | 0.1089455272363818 | 0.22038980509745126 |
| 2 | 8 | 0 | stop_no_detectable_heldout_leakage_reduction | +0.0112 | 0.7231384307846077 | 0.5677161419290355 |
| 2 | 9 | 0 | stop_no_detectable_heldout_leakage_reduction | -0.0509 | 0.21139430284857572 | 0.38830584707646176 |
| 0 | 1 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0229 | 0.670664667666167 | 0.6436781609195402 |
| 0 | 2 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0000 | 0.49525237381309345 | 0.9975012493753124 |
| 0 | 3 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0079 | 0.7706146926536732 | 0.4522738630684658 |
| 0 | 4 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0023 | 0.5987006496751625 | 0.824087956021989 |
| 0 | 5 | 1 | leakage_reduction_detected | -0.1145 | 0.005497251374312844 | 0.011494252873563218 |
| 0 | 6 | 1 | leakage_reduction_detected | -0.0519 | 0.046476761619190406 | 0.0944527736131934 |
| 0 | 7 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0032 | 0.4427786106946527 | 0.8725637181409296 |
| 0 | 8 | 1 | leakage_reduction_detected | -0.0604 | 0.04897551224387806 | 0.09295352323838081 |
| 0 | 9 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0130 | 0.408295852073963 | 0.8260869565217391 |
| 1 | 1 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0274 | 0.27886056971514245 | 0.5767116441779111 |
| 1 | 2 | 1 | leakage_reduction_detected | -0.0290 | 0.030484757621189407 | 0.06296851574212893 |
| 1 | 3 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0070 | 0.271864067966017 | 0.5582208895552224 |
| 1 | 4 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0094 | 0.84007996001999 | 0.31284357821089454 |
| 1 | 5 | 1 | leakage_reduction_detected | -0.0942 | 0.012993503248375811 | 0.03248375812093953 |
| 1 | 6 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0287 | 0.9415292353823088 | 0.12343828085957022 |
| 1 | 7 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0156 | 0.2933533233383308 | 0.5827086456771614 |
| 1 | 8 | 1 | leakage_reduction_detected | -0.0930 | 0.022988505747126436 | 0.05147426286856572 |
| 1 | 9 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0153 | 0.3538230884557721 | 0.7081459270364817 |
| 2 | 1 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0036 | 0.5857071464267866 | 0.8680659670164917 |
| 2 | 2 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0021 | 0.5702148925537232 | 0.8365817091454273 |
| 2 | 3 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0014 | 0.5362318840579711 | 0.9380309845077461 |
| 2 | 4 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0227 | 0.11344327836081959 | 0.223888055972014 |
| 2 | 5 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0433 | 0.18840579710144928 | 0.3733133433283358 |
| 2 | 6 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0124 | 0.6881559220389805 | 0.6331834082958521 |
| 2 | 7 | 1 | leakage_reduction_detected | -0.0590 | 0.047476261869065464 | 0.09245377311344327 |
| 2 | 8 | 1 | stop_no_detectable_heldout_leakage_reduction | -0.0018 | 0.46526736631684157 | 0.9335332333833083 |
| 2 | 9 | 1 | stop_no_detectable_heldout_leakage_reduction | +0.0404 | 0.8750624687656172 | 0.25537231384307846 |

## K2 — reproducible worst-held-out-target gain across seeds (FROZEN sweep go/no-go)

- **`stop_no_reproducible_gain`** · available_seeds = 3 · required_min_seeds = 3 · level_policy = both_levels

- worst_domain_bacc: Δ mean -0.0049, median -0.0052, worst-fold -0.0260 · improved 2/6, harmed 4/6
- worst_domain_nll: Δ mean -0.1067, median -0.1190, worst-fold +0.3205 · improved 4/6, harmed 2/6

| seed | level | Δ worst bAcc | Δ worst NLL |
|---:|---:|---:|---:|
| 0 | 0 | +0.0017 | -0.2622 |
| 0 | 1 | -0.0191 | +0.1122 |
| 1 | 0 | +0.0243 | -0.1320 |
| 1 | 1 | -0.0017 | +0.3205 |
| 2 | 0 | -0.0260 | -0.1059 |
| 2 | 1 | -0.0087 | -0.5728 |

## provenance transition

- accepted: **True** (2 groups) — execution-only two-commit split during live sweep (staged_b.sh cpus 16->32 + controller --leakage-jobs); parallelism is in no scientific hash
- provenance_hashes: ['1b8310930b1b', '39b1287a81d0'] · commits: ['7931091', 'a1a09b8']
- affected_folds: {39b1287a81d0: 11, 1b8310930b1b: 16}
- probe_config_hash (constant): `6d81484fc585` · artifact schema `oaci-artifact-v1` · decision schema `oaci-artifact-v1`
- science-hash policy: artifact pure-science identities remain authoritative; execution parallelism / git tree are not inputs to any fold science hash

## decision hierarchy & verdict

- **K1 (per-fold gate; multiplicity-corrected sweep summary): `stop_no_detectable_heldout_leakage_reduction`**
- **K2 (frozen sweep go/no-go): `stop_no_reproducible_gain`**

> **VERDICT: pause.** K1 shows no multiplicity-surviving held-out leakage reduction and/or K2 shows no reproducible gain. Per pre-registration: do NOT run seeds [3,4]; do NOT add BNCI2014_004. Write this up as a negative BNCI001 minimum-seed result.
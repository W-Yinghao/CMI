# V2 Stage-2 (scoped) --- plan

Per V2_STAGE1B_VERDICT.md: accept the honest asymmetric result; scope Stage 2 to (i) the EEGNet World-A clean
ceiling and (ii) B/C refusal robustness on both backbones, across `source_subject_counts`. **No World-A
redesign; gate thresholds FROZEN; TSMNet World A skipped** (Stage-1B verdict carried forward).

## Scope (config `v2_stage2_scoped.yaml`, hash b8e24e34fc84)
```
World A : EEGNet only              (clean ceiling robustness)
World B : EEGNet + TSMNet          (unsafe-erasure refusal robustness)
World C : EEGNet + TSMNet          (removable-but-useless refusal robustness)
datasets: Lee2019_MI, Cho2017      seeds: 0,1,2      folds: 15
source_subject_counts: 8, 16, 32, all      alpha_grid: 0.25,0.5,1.0,2.0      n_pseudo: 8
interventions: all 10 (incl oracle diagnostic)      thresholds: safety 0.02 / benefit 0.01 (FROZEN)
```

## Task count / runtime (dry-run verified)
72,000 tasks, composition (world,backbone): A-EEGNet 14,400 ; B-EEGNet 14,400 ; B-TSMNet 14,400 ;
C-EEGNet 14,400 ; C-TSMNet 14,400. **No TSMNet World A.**
Estimated ~13.5 h on a single 128-core node (TSMNet B/C 28,800 heavy tasks dominate; EEGNet 43,200 light).
Optional reductions if faster wall-clock is wanted (not applied unless the PM asks): B/C alpha grid -> {0.5,1.0}
(B/C are alpha-insensitive) ~halves it; or a 5-shard job array by (world,backbone) for ~3-4 h wall.

## Stage-2 stop conditions (6 runtime in stop_conditions() + 1 pre-run structural gate; HALT+report)
1 our gate any false ACCEPT ; 2 target labels used outside audit -- PRE-RUN structural gate target_leak_structural_check() -> TARGET_LEAK_STRUCTURAL_PASS or halt ; 3 EEGNet World-A clean
positives disappear at any n_source ; 4 random-k reproduces EEGNet World-A oracle/deployable gain ; 5 World-B
unsafe accept > 0 ; 6 World-C principled ACCEPT > 0 ; 7 TSMNet B/C degeneracy > 20%.

## Pass conditions
* World A / EEGNet: `clean_worldA_positive` cells exist at each n_source; principled ACCEPT = 0; source-LOSO
  benefit LCB <= +0.01 for target-beneficial cells; oracle target-informed selector true-accepts some; random-k
  does not reproduce. Robustness = holds as n_source grows 8 -> all.
* World B: unsafe accept = 0 (task-entangled erasures rejected) at each n_source, both backbones.
* World C: principled ACCEPT = 0; high-domain-gain-useless cells exist; naive controllers false-accept them.

## Outputs -> tos_cmi/results/method_deepen/v2_stage2/
v2_stage2_scoped_{rows.csv, summary.json, manifest.json, report.md, ceiling_scatter.png,
naive_controller_table.csv}; plus a per-n_source World-A/EEGNet clean-ceiling table and per-n_source B/C
refusal table (printed by the driver).

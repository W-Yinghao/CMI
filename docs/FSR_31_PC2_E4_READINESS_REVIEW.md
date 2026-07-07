# FSR_31 — PC2-E4 Readiness Review (design/inventory only; NO GPU)

**Project FSR — PC2-E4 feasibility.** Not a PC2 run and not a GPU authorization. This checks whether the PC2-E4
learned-reliance experiment (FSR_28) is even *runnable*, and pins the concrete blockers. Artifacts:
`results/fsr_pc2_e4_preflight/` (`dataset_inventory.csv`, `stress_feasibility.csv`, `rho_dose_response_plan.yaml`,
`gpu_budget_table.csv`, `pc2_e4_go_nogo.json`).

## Bottom line
**PC2 GPU is NOT eligible** (`pc2_gpu_eligible = false`). Two hard blockers:
1. **Only 2 datasets are preset-ready.** FBCSP-LGG has channel + `central_strip_v1` presets for **BNCI2014_001**
   and **BNCI2015_001** only (verified in `cmi/run_loso.py` `_infer_ch_names` and `cmi/models/fb_lgg_dualcmi.py`
   `central_strip_v1`). The 4F/4G lesson makes **leave-one-dataset-out the binding robustness axis**, which needs
   **≥3** datasets. A 3rd requires adding a channel-preset + `central_strip_v1` group + MOABB loader for one
   candidate — a bounded CPU engineering task, **not** a GPU run.
2. **Phase 4G E4b scope not yet established** (`≥ partial` required before PC2).

## The 7 readiness questions (answers in `pc2_e4_go_nogo.json`)
1. **≥3 compatible datasets?** **No today.** 2 preset-ready; candidates that fit the 10-20 EEG-MI + central-strip
   setup and have enough subjects/channels: **Cho2017** (52 subj, 64 ch), **Lee2019_MI** (54 subj, 62 ch;
   already used elsewhere in the project), **PhysionetMI** (109 subj, 64 ch; in the PROCESSED datalake),
   Weibo2014, Schirrmeister2017. Marginal: BNCI2014_004 (3 channels → degenerate central-strip graph),
   Zhou2016 (4 subjects).
2. **Same FBCSP-LGG refit + checkpoint dump per dataset?** Yes for the 2 preset-ready (identical Phase-4B runner
   path). A 3rd needs the preset + loader added first.
3. **Induce subject↔class reliance holding global P(y)?** Yes — skew `P(y|subject=d)` toward a spurious `c_d` at
   stress `ρ`, with complementary `c_d` assignment so the mixture reproduces `P(y)` (coupling in the joint, not
   imbalance).
4. **Shuffled-stress control?** Yes — shuffle the `subject→c_d` mapping at matched marginal to rule out
   class-imbalance.
5. **ρ ∈ {0, 0.5, 0.8} feasible?** Yes on both preset-ready datasets (enough per-class trials to subsample).
6. **GPU budget?** ~25–60 GPU-h for 2 datasets (FSR_23); ~35–90 GPU-h for ≥3 (scales with dataset count);
   V100/A40, per-fold checkpointing.
7. **If Phase 4G fails, does PC2 auto-pause?** **Yes** — PC2 stays PAUSED (not auto-run); return for PM review.

## Go / no-go (frozen)
```
PC2_GPU_ELIGIBLE  only if  ALL of:
  Phase 4G E4b >= partial            [PENDING — 4G not established]
  >= 3 preset-ready datasets         [FALSE — 2 today; add one candidate preset+loader]
  stress feasibility pass            [TRUE]
  PM budget approval                 [PENDING]
```
All four must hold; currently two fail/pending → **not eligible**. Even a 4G pass does not unlock PC2 GPU until
the ≥3-dataset prerequisite is met and the PM approves budget.

## Recommended prerequisite (if the PM wants to move toward PC2)
Add a channel-preset + `central_strip_v1` group + MOABB loader for **one** of {Cho2017, Lee2019_MI, PhysionetMI}
(CPU task; Lee2019_MI is lowest-friction since it is already used in the project). Then the leave-one-dataset-out
gate becomes meaningful for both Phase 4G (second-moment) re-confirmation and PC2 (learned reliance). **This
document commits no GPU and no such preset work — it is the readiness assessment the PM requested.**

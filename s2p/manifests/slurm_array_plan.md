# S2P SLURM array plan (Phase 9A design; array built in 9B-0)

One array task per `run_id` from `pretrain_subset_design.csv`. Stages chained by dependency:
```
stage A pretrain      : sbatch --array over run_ids (GPU A40/A100); resume from last ckpt; ckpt every epoch;
                        log pretrain-val loss on held-out pretrain subjects each epoch.
stage B feature_dump  : depends on A; frozen encoder -> downstream F1 embeddings (SHU-MI/PhysioNetMI/BNCI).
stage C downstream    : depends on B; FSR L1/L4/L5/L6 audit (reuse cb_cbm_8b_f1_audit-style), source-only selection.
stage D aggregate     : depends on all C; clustered-CI slopes vs log(N); claim-safe verdict.
```
Concurrency capped (min(16, matrix)); CUDA determinism pinned per run seed (or tolerance disclosed).
run_id key = model_id_corpus_condition_N_H0_seed_stage. P0 smoke first (2 runs), then P1 pilot (6), then P2 (post-review).

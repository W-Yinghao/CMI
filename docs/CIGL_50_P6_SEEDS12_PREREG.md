# CIGL_50 P6 — seeds 1/2 confirmation PRE-REGISTRATION (frozen before results)

Committed **before** any seed-1/2 result JSON exists. PI overrode the null-screen recommendation to obtain a
multi-seed confirmation of the single least-indefensible config while architecture work proceeds in parallel.
The point of pre-registration is to prevent a post-hoc full-mean rescue of a config whose seed0 gain was a
chance-fold artifact.

## Scope (frozen)

- **Configs (exactly 2):** `erm:0` (floor-matched control) and `fbdualpc:0.000:0.000:0.003:0.000:0.100:50`
  (spatialCMI λspatial=0.003). NO dec-only, NO all-CMI, NO dec_scale=100 — seed0 disqualified them
  (all-CMI regresses full-2a; dec100 lead is one-subject-driven; dec-only carries no CMI term).
- **Datasets / folds:** BNCI2014_001 t0–t8 (9) + BNCI2015_001 t0–t11 (12) = 21 folds.
- **Seeds:** 1 and 2 (seed0 already in hand → 3-seed picture after this run).
- **Everything else identical to seed0:** `--fusion_floor 0.05 --epochs 300 --bs 64 --warmup 40 --n_inner 2
  --source_val_early_stop --backbone FBCSPLGGGraph`, worktree `6afe0cc`, env eeg2025, CUDA fail-closed.
- **Jobs:** 42 array tasks (21 folds × 2 seeds), each runs both configs.
  Out → `results/p6_fbdualpc_seeds12/${DS}_t${TIDX}_seed${SEED}.json`.

## PRIMARY endpoint (pre-committed — this decides the config's fate)

**CSP-decodable BNCI2014_001 subset {1,3,8,9}, spatialCMI − erm, 3-seed mean (seeds 0/1/2).**

- **REJECT** (seed0 artifact confirmed) if the 3-seed CSP-decodable Δ stays materially negative
  (≤ −0.010, i.e. the seed0 −0.016 holds). The full-2a gain is then declared a chance-fold artifact and
  spatialCMI dies; next step is architecture, not more seeds.
- **SURVIVE** only if the 3-seed CSP-decodable Δ is ≥ 0 (non-negative; ideally within-noise-positive) **AND**
  the 2015 3-seed mean Δ stays ≥ 0. Only then is spatialCMI a live method candidate worth a wider sweep.
- **AMBIGUOUS** (−0.010 < Δ < 0): treat as not-passing; report honestly, no promotion without a further
  pre-registered test.

## SECONDARY (reported, NOT decisive)

Full-2a mean, 2015 mean, worst-fold, gate means/entropy, dec%/reg_spatial_gls/loss_spatial finiteness —
all reported per-seed + 3-seed mean ± across-seed spread + paired-fold sign/t. Per seed0 these full-mean
metrics are chance-band-driven, so they cannot by themselves rescue a CSP-decodable regression.

## Analysis plan (frozen)

Report per-seed and 3-seed: (a) CSP-decodable {1,3,8,9} mean Δ [PRIMARY]; (b) full-2a mean Δ;
(c) 2015 mean Δ; (d) per-subject Δ on the decodable set; (e) across-seed SD as the noise yardstick;
(f) guard finiteness. Verdict applies the PRIMARY rule above verbatim. PI-gated thereafter.

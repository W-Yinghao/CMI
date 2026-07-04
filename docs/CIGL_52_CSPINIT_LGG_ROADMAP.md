# CIGL_52 — P8 CSP-initialized / CSP-teacher spatial branch (non-GPU scaffold)

Branch `project/cspinit-lgg-spatial` off the clean FBCSP-LGG base `5c26e6d`. PI-gated main line after the P6
null (spatial-CMI) and P7a FAIL (cov_tangent underfit).

## 0. Evidence chain

1. FBCSP-LGG ERM is the current best backbone (full-LOSO refs: 2a 0.349 > DGCNN 0.311 > FBLGG 0.287; 2015
   0.608 > FBLGG 0.593 > DGCNN 0.588).
2. The **logvar** spatial branch is load-bearing on 2a (zero_spatial ablation).
3. P7a **cov_tangent** did NOT help — it underfit the near-rank-deficient 2a covariances (CSP-decodable
   Δ = −0.123). Killed.
4. Classical CSP still beats the neural spatial branch on several subjects.

→ P8 does not change the feature FORM (keeps logvar). It makes the neural spatial filters START from
source-estimated CSP filters, and adds source-only auxiliary supervision so the spatial branch is not left to
be pulled around indirectly by the fused gate.

## 1. P8-A — source-CSP-initialized spatial filters

`FBCSPLGGGraph(spatial_init="random" | "source_csp")`, default `random` (= P6 path unchanged). `source_csp`:
- Fit CSP on **source-train only** (`cmi/models/csp_init.py: source_csp_filters`). **FIREWALL:** the LOSO caller
  removes the held-out target subject; the source-val subject is removed by the trainer before the fit
  (`train_model` calls `init_spatial_from_csp` right after the source-val split). Target labels never touched.
- 2a (n_cls>2): **one-vs-rest** multiclass CSP (each class vs rest), m filters per class.
- 2015 (n_cls=2): **binary** CSP (class-0 end + class-1 end), m per end.
- Filters written into each band's per-temporal-filter K spatial slots (top-K by eigenvalue discriminability,
  broadband/shared across bands v1); extra slots keep random init; **then trainable, not frozen**.
- Per-fold metadata: `csp_fit_subjects, csp_excluded_target, csp_excluded_source_val, csp_n_filters_used,
  csp_n_filters_pool, csp_rank, csp_cov_shrinkage, csp_m_per_contrast, csp_classes_present`.

## 2. P8-B — spatial auxiliary head

Source-only auxiliary classifier over `spatial_z`: `L = L_fused + η·CE(aux(spatial_z), y)`, first version
**η=0.2** only (no graph/temporal aux yet). Goal: keep the spatial branch task-usable rather than starved by
the fused gate. Diagnostics: `loss_spatial_aux`, `spatial_aux_source_val_bacc` (source-val), and
`spatial_aux_target_bacc` (target — **evaluation only, NEVER selection**), plus `gate_spatial_mean`,
`zero_spatial_target_bacc`.

## 3. P8-C — fusion floor

Keep `fusion_floor=0.05` (no no-floor — P7a showed no-floor confounds feature verdict with gate starvation;
P6/P7a are floor-matched).

## 4. Contract / scope

- `spatial_z_dim` unchanged; the aux head is a new submodule (trained by the main optimizer only when
  η>0). `spatial_init="random"` + `η=0` is byte-identical to P6. 5-tuple contract, ablations, firewall intact.
- CLI: `--spatial_init {random,source_csp}`, `--spatial_aux_weight`. NO CMI, no penalty, no projector.

## 5. Tests (CPU, `tests/test_cspinit_spatial.py`, 12 pass) + full regression

random default byte-identical + 5-tuple; source_csp filters finite/unit-norm; one-vs-rest (2a) & binary (2015)
init changes filters; **firewall** (source-val + target excluded from CSP fit); CSP fit deterministic
source-only; aux loss finite + head trains (off→no aux key); all ablations both inits; end-to-end determinism;
central_strip datasets; CLI flags.

## 6. GPU gate — full-LOSO seed0, floor=0.05 (PI: no 2-fold)

CPU smoke first (runability + metadata only, 2a+2015 × {random,source_csp} × {aux 0.0, 0.2}). Then:

```text
Backbone FBCSPLGGGraph, config erm:0, --source_val_early_stop, --fusion_floor 0.05, seed 0,
epochs 300 bs 64 warmup 40 n_inner 2, BNCI2014_001 (9) + BNCI2015_001 (12)
Variants (backbone/run args, NOT method configs):
  A. --spatial_init random     --spatial_aux_weight 0.0   # baseline
  B. --spatial_init source_csp --spatial_aux_weight 0.0   # isolates CSP init
  C. --spatial_init random     --spatial_aux_weight 0.2   # isolates aux supervision
  D. --spatial_init source_csp --spatial_aux_weight 0.2   # combined candidate
21 folds x 4 variants = 84 jobs, max concurrency 8 (do NOT crowd running jobs)
out results/p8_cspinit_s0/${DATASET}_t${TIDX}_${VARIANT}_seed0.json
```

**PASS (PI-pre-committed):**
- PRIMARY: BNCI2014 CSP-decodable `{1,3,8,9}` mean, `best P8 variant − (random,aux0) baseline ≥ +0.02`.
- SECONDARY: 2a full non-negative; 2015 full ≥ −0.01; `zero_spatial` still load-bearing;
  `spatial_aux_target_bacc` improves or does not collapse; `source_bacc` no train-memorization with source-val/
  target failure; cov/csp metadata finite.
- If P8 improves only 2015 but hurts 2a decodable → reject (same as P7a).
- If P8 improves 2a decodable but 2015 drops slightly → keep as live candidate, do not promote until multi-seed.
- If seed0 passes → run seeds 1/2 for the best variant only.

## 7. Frozen / not now
P7b (TaskNullProjector), decodability-adaptive gating, PCMI-TIF training, conditioning-first covariance
(P7a follow-up) — all frozen. Multi-window FBCSP, graph auxiliary head, EEGNet/Shallow/Deep/Conformer sidecar
baselines = later, not before P8.

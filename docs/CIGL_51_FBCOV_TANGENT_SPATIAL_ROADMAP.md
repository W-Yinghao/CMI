# CIGL_51 — P7a FBCov-Tangent spatial branch (ERM-first, non-GPU scaffold)

Branch `project/fbcov-tangent-spatial` off P6 tip `5c26e6d`. PI-gated successor to the P6 null screen.

## 0. Why (from the P6 verified null screen, `6afe0cc`)

P6 `fbdualpc`/spatial-CMI was a clean **negative screen**: every CMI/decoder config regressed on the
CSP-decodable BNCI2014 subset `{1,3,8,9}` (−0.013…−0.033) while the small full-mean gains came only from
the near-chance subjects `{2,4,5,6,7}`. Root cause, **verified in code** (`_FBCSPBand.forward`): the spatial
branch takes learned per-band spatial filters then **variance over time only** — the diagonal after a point
spatial projection. Layering a marginal-reference CMI penalty on that thin, near-noise, class-and-subject
entangled 32-d feature has one cheap minimizer — shrink it — the project-wide *leakage-reduction-via-collapse*.

**Honest framing (not "structurally impossible"):** a learned spatial filter + log-variance *is* parametric
CSP and *can* represent the 4-class contrast in principle. The claim is **empirical underfitting** — SGD fits
the point-filter CSP poorly on decodable subjects with limited, subject-entangled data (`gate_spatial`
starved to 0.27 in P6). Whether the richer covariance-tangent feature actually helps is what the ERM-first
gate settles — asserted by no one, tested cheaply.

## 1. What P7a changes (ONLY the ERM spatial feature)

`FBCSPLGGGraph(spatial_mode="logvar"|"cov_tangent")`, default `"logvar"` = **byte/state-dict identical to P6**
(verified: identical `state_dict` keys + identical forward under a matched seed). `"cov_tangent"` replaces the
per-band log-variance with a covariance-tangent feature exposing the full second-order geometry to the linear
`proj`:

```text
after the per-band temporal conv + BN -> h [B, n_filt, C, T']   (n_filt learned sub-bands)
per (sample, sub-band):
  center over time; S = h_c h_c^T / (T'-1)          # C x C channel covariance
  S = S / trace(S)                                  # trace-normalize -> trace 1
  S = (1-a) S + a I / C   (a = cov_shrinkage 0.05)  # SPD, min eig >= a/C
  logS = Q diag(log clamp(l, eps)) Q^T              # SPD tangent map (eps = cov_eps 1e-4)
  feat = vech(logS)  (off-diagonal * sqrt2)         # tangent isometry, C(C+1)/2 dims
concat over sub-bands and kernels -> Linear -> spatial_z [B, spatial_z_dim]   # dim UNCHANGED (32)
```

`spatial_z_dim` is unchanged, so **`trainer.py` needs no changes**. The grouped spatial conv is still built
(so logvar stays state-dict identical) but is unused (no gradient) under cov_tangent.

**Gradient safety (`_SPDLogm`):** the generic `torch.linalg.eigh` backward has a `1/(l_i−l_j)` term that
NaNs on (near-)degenerate eigenvalues — realistic for low-rank band covariances whose small eigenvalues pin
to the shrinkage floor. P7a uses a custom Daleckii–Krein / Loewner backward whose off-diagonal
`(log l_i − log l_j)/(l_i − l_j)` has a finite `1/l_i` limit at degeneracy. Tested on a rank-1 (all-channels-
identical) input: finite forward **and** backward.

## 2. NOT in P7a (held / killed by the PI)

- **P7b TaskNullProjector** — held behind P7a passing the ERM full-LOSO gate below.
- **decodability-adaptive gating** — killed (it is the failed global-LPC shrink penalty, dosed per-subject).
- **PCMI-TIF** — monitor only, never a training term.
- **FDR / fusion reroute** — downstream pivot, not the first fix.
No CMI, no penalty, no projector, no new gating in this patch. P7a is a pure ERM-backbone feature swap.

## 3. Diagnostics (recorded per fold, via `cov_summary`)

`spatial_mode, cov_shrinkage, cov_eps, cov_eig_min_mean, cov_eig_min_p05, cov_log_feature_norm_mean,
cov_log_feature_norm_p95` (new) plus the existing `gate_{graph,temporal,spatial}_mean`, `gate_entropy_mean`,
`ablate_zero_{graph,temporal,spatial}_target_bacc`, `ablate_permute_nodes_target_bacc`, `source_bacc`,
`best_source_val_bacc`, `final_val_source_bacc`. `cov_eig_min` is post-shrinkage (floor `a/C`); values near
that floor flag a near-singular raw covariance. A runaway `cov_log_feature_norm_p95` flags a log/eigen blow-up.
**Memorization guard:** if `cov_tangent` drives `source_bacc → ≈1.0` while source-val/target do not rise, that
is subject-covariance memorization — NOT a pass.

## 4. Tests (CPU, `tests/test_fbcov_tangent_spatial.py`, 11 pass)

logvar default byte/state-dict-identical + minimal cov_summary; cov_tangent 5-tuple; finite forward **and**
backward through eigh; no-NaN on a rank-1 near-singular input (gradient-safety); zero_spatial + all ablations;
gate_summary sums to 1; central_strip_v1 resolves for 2a (22→253) and 2015 (13→91) with C-agnostic covariance;
tiny ERM writes the cov diagnostics; CLI accepts `--spatial_mode/--cov_shrinkage/--cov_eps`; vech Frobenius
isometry + `_spd_logm` round-trip. Plus full regression suite.

## 5. GPU gate — full-LOSO seed0 (PI: NOT a cheap subset gate)

After CPU tests + CPU smoke pass:

```text
Backbone FBCSPLGGGraph, config erm:0, --source_val_early_stop, seed 0, epochs 300 bs 64 warmup 40 n_inner 2
Spatial modes: logvar  AND  cov_tangent      (same branch/runner/CLI -> apples-to-apples, no historical mix)
Datasets: BNCI2014_001 all 9 folds + BNCI2015_001 all 12 folds
21 folds x 2 modes = 42 jobs, max concurrency 8
out results/p7a_cov_tangent_s0/${DATASET}_t${TIDX}_${SPATIAL_MODE}_seed0.json
```

**PASS (PI-pre-committed):**
- PRIMARY: BNCI2014 CSP-decodable `{1,3,8,9}` mean, `cov_tangent − logvar ≥ +0.02`.
- SECONDARY: 2a full-mean Δ non-negative; 2015 full-mean Δ `≥ −0.01` (prefer non-neg); `source_bacc` not ≈1.0
  with flat source-val/target; `zero_spatial` still load-bearing; cov diagnostics finite (no eig/log blow-up).
- If `cov_tangent` only lifts the chance-band subjects and lowers `{1,3,8,9}` → **negative, treat like P6.**
- 2a-decodable up but 2015 down 1–2pp → consider a dataset-sensitive branch, not an immediate general SOTA;
  decide from the full table.

## 6. Frozen

No GPU until CPU passes; then the 42-job seed0 screening above only. No P7b, no CMI, no seeds 1/2 for P7a
until the seed0 gate passes. Seed0 is screening; a method verdict still needs full-LOSO × multiple seeds.

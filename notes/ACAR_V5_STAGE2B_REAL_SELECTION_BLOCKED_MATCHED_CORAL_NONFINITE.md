# ACAR V5 — Stage-2B Real DEV Selection: PRE-FLIGHT BLOCKED (matched_coral non-finite)

```
status:                  STAGE2B_PREFLIGHT_BLOCKED  (full run NOT submitted; caught in the readiness/timing probe)
intended_run_id:         acar-v5-stage2b-a5c44c3-r1
implementation_base_sha: a5c44c31607b0771e5367d6eb84bcaced7c71b37
runtime:                 git worktree pinned to a5c44c3 (worktree_HEAD verified == a5c44c3)
stage1b_run_id:          acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256: 2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha: 4278435975a72b1127803dd2cffab420c083e430

selected_candidate_id:   null           (selection engine never ran to completion)
selection_report_written: no
```

The real Stage-2B run was authorized and pinned to `a5c44c3`. Before submitting the multi-hour full run, a readiness/timing
probe (evaluate ONE candidate on ONE disease, real labels + real action seam) was run under SLURM. It FAILED at the action seam.
Per the "if the run fails, stop and report — no patch/rerun/skip/tune" instruction, the full run was **not** submitted.

## First failure (SLURM job 883377, `--time`, nodecpu at HEAD==a5c44c3)

```
first failing module:    stage2_real_action_provider.validated_real_action -> stage2_action_provider_validation.validate_action_output
first failing candidate:  V5-P1-001
first failing disease:    PD
first failing split:      CAL  (inside stage2_policy_eval.evaluate_candidate_disease)
first failing action:     matched_coral
exception:                Stage2ActionValidationError: "matched_coral: p_a has non-finite entries"
```

The validated real-action seam fail-closed correctly: matched_coral returned a p_a with non-finite entries, so
`validate_action_output` raised rather than letting a NaN propagate into NLL / ΔR / the gates.

## Characterization (label-free diagnostics)

- **First-batch-only diagnostic** (job 883379): ran identity/matched_coral/spdim/t3a on the FIRST 32-window batch of every subject
  with ≥32 windows across all 10 selection refs — **2070 batches, ZERO non-finite for every action**.
- **All-full-batches diagnostic** (job 883382): ran the same four actions on **10,770** full 32-window batches (all subjects, all
  32-window chunks, all 10 refs) — **ZERO non-finite for every action**, including matched_coral.
- Yet the `--time` probe hit a non-finite on a batch that is **inside** that same 10,770-batch space.

Same deterministic code + same input space → both finite (diagnostic) and non-finite (`--time`) across different runs/nodes ⇒ the
failure is a **borderline numerical overflow that is floating-point-sensitive across nodes**, not a fixed bad batch and not
randomness.

## Mechanism

`matched_coral` = `acar.actions.act_matched_coral` → `cmi.eval.source_state.pmct_predict_serialized(state, z_tgt, ref="pooled",
tmap="wc", em_iters=3)`. There is **no `random`/`randn`/`seed`** in `pmct_predict_serialized` (deterministic). It computes the
CORAL transport, whose core is the **inverse matrix square-root of the target-batch covariance**:
`bar_Sig_T = _shrink_cov(np.cov(z_tgt), rho, eps)` then `Wt_inv = _sqrtm(bar_Sig_T, eps, inv=True)` (source_state.py:78-79).

On the v5 substrate the embedding is **256-dim**, but a batch is **32 windows** → `np.cov(z_tgt)` has rank ≤ 31 in 256-D, i.e.
**massively rank-deficient**. After shrinkage (adapter `rho=0.1`, `eps=1e-3`) the smallest eigenvalues are ~`rho·trace/d + eps`
(tiny), so their inverse-square-roots are enormous → the transported `z_post` sits at the **edge of float64 overflow**. Whether a
given batch tips to `Inf` depends on eigen-rounding, which differs across BLAS/nodes → the observed intermittency. `Inf` z_post →
`f₀(z_post)` softmax → `NaN` p_a.

Contributing, all frozen/pinned: (a) 256-dim substrate; (b) 32-window batch (`STAGE2_BATCH_SIZE`); (c) the v5→old source-state
adapter's shrinkage `rho=0.1` in `stage2_action_records.SourceLDA.old_state`.

## Impact on the real run

The full selection runs matched_coral ~10⁵ times (22 candidates × 2 diseases × 5 folds × subjects × batches × recomputed per
candidate). With even a rare per-batch non-finite, the all-or-none run **will** hit one → the validated seam raises → the run
fails; and any "success" would be **node-dependent / non-reproducible**. This is exactly the 256-D-vs-small-batch CORAL
ill-conditioning the earlier reuse recon flagged.

## No forbidden actions taken

No full selection run submitted; no candidate selected; no thresholds/CAL cert/EVAL utility/G1–G6 computed; no report written. The
probe read real DEV labels (inside policy evaluation) for the subjects processed before the failure, but produced no scores and no
selection. S1 seeds 20260712/20260713 were never opened; no external/ASZED/lockbox; no substrate rebuild; no policy change. The
production code executed was byte-identical to `a5c44c3` (worktree HEAD verified).

## This is a method/numerics decision (requires separate authorization)

The fix is NOT a run tuning — it is a reviewed change to how CORAL is conditioned on the v5 256-D substrate, e.g. (candidate
directions, for the user to choose, each a new dated authorization + review): stronger shrinkage (`rho`) / larger `eps` in the
adapter or the CORAL target covariance; a dimensionality reduction (e.g. PCA to the batch rank) before CORAL; a larger batch size
so the target covariance is better-conditioned; or dropping matched_coral / P-family rows that depend on it. Choosing any of these
is a protocol/method change; it is not applied here.

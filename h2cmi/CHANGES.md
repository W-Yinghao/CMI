# H²-CMI — correctness fixes (second code review)

The first cut wired every component together but had several implementation bugs that would
change experimental conclusions. This pass fixes the correctness-critical ones, shrinks the
default to a trustworthy minimal core, and adds fast unit tests with null/power checks.
Deeper redesigns the review itself stages for later are deferred (and disabled by default).

## Fixed

| id | issue | fix | test |
|---|---|---|---|
| **P0-1** | source subset trained against the FULL-dataset DAG → critic outputs over target-only levels; reference entropy & critic CE on different supports; target cardinality leaks into "strict DG" | `compact_domain_labels()` builds a source-only DAG with contiguous levels; called on every fold in `run_synthetic`/`run_ablation` | `test_source_dag_remap` |
| **P0-3** | low-rank TTA `A=I+UVᵀ` with `U=V=0` → zero gradient, transform frozen at identity | seed both `U,V` small-random (A starts ≈ identity, gradient ≠ 0) | `test_tta::lowrank` |
| **P0-4** | TTA `nll_after` omitted the `+log|det A|` Jacobian → `delta_density_nll` inconsistent with the EM objective; `adapted=True` even when evidence worsened (no rollback); `prior_kl` was an unused `max()` arg | change-of-variable evidence everywhere; **cross-fitted** held-out evidence gate with identity rollback; `prior_kl` renamed `prior_anchor_strength` — a Dirichlet pseudo-count anchor that penalises `H(π_S,π_T)=KL(π_S‖π_T)+const` (reverse direction), not a forward `KL(π_T‖π_S)` | `test_tta::change_of_vars`, `::rollback` |
| **P0-2 (online)** | "online TTA" never updated the transform (only the prior) but was reported as transform adaptation | honest label `online_prior_only`; transform stays identity; causal online transform deferred | — |
| **CMI** | `grl` option would double-flip the penalty sign; critics got (wasted) gradient in Step B | removed `grl`; critics **frozen** (`requires_grad=False`) during the encoder step (envelope-theorem profile gradient); per-epoch critic-CE diagnostic; `critic_inner` raised in the core | `test_cmi_gradient_sign` |
| **P0-7** | leakage cross-fit used a random trial split (autocorrelation inflates leakage) and a permutation null that did **not** refit the critic | **stratified grouped** split (site→subject, subject→session, session→temporal-half) so each measured level is in both folds via different recordings; **refit-under-permutation** null | `test_cmi_null_and_power`, `test_leakage_group_split` |

## Deferred (disabled by default in `core_config`; `--full` to enable)

These work mechanically but have an optimisation-direction or evaluation-protocol issue the
review flags; they are off until each is validated on its own.

* **P0-5 safety gate** — `train_safety_gate` is not yet a true nested inner-LOSO (the
  pseudo-target participated in the source model's training, and gate train == gate eval).
  A correct version must retrain the source model excluding each pseudo-target and do
  leave-one-pseudo-target-out gate prediction.
* **P0-6 disentanglement** — the label-leakage term is a min-min surrogate (the probe and
  encoder share one optimiser, so the probe lowers the penalty by becoming a worse
  classifier). Needs alternating Step A (train probe) / Step B (freeze probe, update
  encoder), with the probe conditioned on `D` and reference `H(Y|D)` (not uniform).
* **SSL** — `z_c`→raw-EEG reconstruction encourages `z_c` to keep nuisance (fights CMI); the
  mask is applied to the output, not the encoder input. Restructure: consistency/VICReg on
  `z_c`; reconstruction on `z_n`/`(z_c,z_n)`; mask the input.
* **canonicaliser** — a global linear map after an arbitrary fusion layer is not
  identifiable (absorbable).
* **reference alignment** — silently inactive when domain-class cells < 4; should use a
  domain-class-balanced sampler / EMA prototypes and a leave-one-domain-out reference.

## Known limitations still open (not yet addressed)

* CMI critics output over all (source) global levels; **parent-local** critic outputs
  (predict the local child index given the parent) would be easier to fit for
  subject/session. The compaction (P0-1) reduces cardinality but not to parent-local.
* Encoder temporal/SPD branches use `len(bands)` but not the actual `(lo,hi)` band edges as
  filters (only the graph FFT branch uses the bands); BiMap weights are not kept on the
  Stiefel manifold after init; the graph branch has no electrode coordinates / channel
  mask, so different montages are not yet handled natively (graph branch off in core).
* The simulator's per-trial channel z-score cancels pure channel-gain montage shift; the
  `montage` knob is mostly dropout/noise. Shift mechanisms also share one RNG stream, so
  toggling one knob perturbs the others (not yet a clean paired/orthogonal design).

## Recommended next experiments (review §6)

0. module-correctness unit tests (done here) → 1. validate `p_φ(z|y)` alone →
2. CMI null/power on constructed latents → 3. the **CMI × TTA 2×2** across shift mechanisms
(no-shift / cov / prior / concept / montage / noise) with leave-one-site-out × 10 seeds →
4. hierarchy (flat-joint vs site/subject/session) → 5. TTA structure ablation →
6. truly-nested safety gate → then BNCI2014_001/004 pilots.
**Do not run the `--full` 10-seed ablation yet.**

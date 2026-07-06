# Project B: Refusal-First Safe EEG Adaptation

> Status: **DESIGN (Step 1)** — no core code changed. Branch `project-b-refusal-router` (off `main`).
> This document is the pre-registration surface for the Project B router. Section 10 is the Step-2 build order.
> Every API symbol referenced below was verified against the **local checked-out** `h2cmi/` source
> (not the GitHub mirror) on 2026-07-06. Where the originating plan's assumptions diverged from the
> real code, the correction is called out inline and consolidated in §2.4.

---

## 1. Goal and non-goals

### 1.1 Goal
Add a **deployment-time decision layer** on top of a frozen, source-trained H²-CMI model that, for an
*unlabelled* target batch or stream, chooses one of a small set of adaptation actions **or refuses**, and
emits an auditable reason for the choice. The layer is:

- **Refusal-first**: the default action is `REFUSE`; a non-refusal action must clear explicit support,
  prior-decoupling, and calibrated-risk gates before it is allowed.
- **Action-conditional**: each action carries its *own* calibrated upper risk bound, not a single marginal
  "is adaptation safe?" verdict.
- **Support-aware**: refusal is driven by a vector of support/OOD diagnostics, not one scalar.
- **Prior-decoupled**: a shift in the label prior `π_T ≠ π_S` alone is *never* sufficient to refuse; it is
  recorded as audit information and separated from covariate/support/concept shift.
- **Zero target labels**: the router consumes target `X` only. No target `y`, no target-label calibration.

> **Positioning (revised after the Step-2A substrate study, §9.2).** Project B's primary contribution is
> **NOT** a source-only learned harm regressor — Step-2A showed that harm on the deployment target is real but
> the source-side harm-calibration signal is degenerate (non-nested subject gate: `pseudo_gain ≡ 0`) and
> non-transferable (nested source-site LODO: `corr(target,nested)=+0.06`). The load-bearing safeguard is
> therefore **TOS/support-aware refusal (§4)** plus **prior-decoupled diagnostics (§7)** and **OACI reason
> codes (§5)**. **ACAR-style action risk (§6) is a *conditional* module**, active only when its calibration
> set is non-degenerate; otherwise the router emits `OACI_ACAR_HARM_CALIBRATION_DEGENERATE` and falls through
> to support/refusal logic. The learned `SafetyGate` is a **baseline / negative control**, not a foundation.
> The honest thesis: *when harm is not source-identifiable, do not pretend to predict it — refuse on support
> grounds and say why.*

### 1.2 Non-goals
- Not a new EEG encoder. The encoder, density head, and classifier are **frozen** after source training.
- Not a retraining/fine-tuning method. Actions only estimate an affine transform `(A,b)` on the *frozen
  embedding* and a target prior `π_T` (see §3.3). No gradient touches encoder or classifier weights.
- Not a target-label-informed selector. Any statistic that needs target `y` is out of scope for Project B.
- Not a replacement for the existing `SafetyGate`. Project B *subsumes and generalises* it (the learned
  harm gate becomes one feature-source and one baseline; see §6.4).

### 1.3 Success criterion (frozen at Step 2, not here)
Project B is a **positive result** only if, on held-out target domains, router-selected actions achieve
**selective harm avoidance without destroying benefit**: on the synthetic substrate first
(recoverable-shift world → accept & gain; concept-shift world → refuse/identity & avoid harm), then on the
same real-EEG protocols the parent line uses. The exact numeric go/no-go is pre-registered at the top of
Step 2, not in this design draft. Consistent with the project's standing evidence posture, the honest null
hypothesis is that a source-only router **cannot** certify deployment benefit (non-identifiability); Project B
must beat that null on the substrate before any real-EEG claim.

---

## 2. Existing repo substrate

Project B is built **entirely on the isolated `h2cmi/` package**, not on the AAAI `cmi/` main line. The
`cmi/` package (MOABB loaders, LPC-CMI regularizer, LOSO runner) is referenced only for vocabulary and for a
future real-EEG bridge; **no `cmi/` file is modified**.

### 2.1 What already exists and is reused verbatim (verified signatures)

| Concern | Real local symbol | Verified signature / return |
|---|---|---|
| Frozen embedding | `h2cmi.train.trainer.H2Model.embed` | `embed(self, X, device="cpu", bs=256) -> np.ndarray [n, z_c_dim]` (`@torch.no_grad`) |
| Source/blended posterior | `H2Model.predict_proba` | `predict_proba(self, X, device="cpu", prior=None, mode="blend", bs=256) -> np.ndarray [n, C]` |
| Reference prior | `h2cmi.train.trainer.reference_prior` **(module-level fn, _not_ a model method)** | `reference_prior(y, n_classes, mode) -> np.ndarray [C]`; `mode="source_marginal"` else uniform |
| Source training | `train_h2` | `train_h2(X, y, domains, dag, cfg, align_factor="site", verbose=False) -> (model, hcmi, dual, history)` |
| Class-cond density | `h2cmi.density.student_t_mixture.ClassConditionalDensity` | `log_prob_all(z)->[B,C]`, `log_prob(z,y)->[B]`, `class_posterior(z, log_prior)->[B,C]` |
| TTA action | `h2cmi.tta.class_conditional.ClassConditionalTTA` | `__init__(density, source_prior, cfg, n_classes, device)`; `fit(U, pseudo_labels=None)->TTAResult`; `fit_online(batches)->TTAResult` |
| TTA result | `TTAResult` | `dataclass(transform: Transform, pi_T: np.ndarray, adapted: bool, diagnostics: dict)` |
| Learned harm gate | `h2cmi.gate.safety_gate.SafetyGate` | `fit(feats[M,8], gains[M])`, `predict_harm_prob(g)->float`, `should_adapt(g, evidence)->bool` |
| Gate feature vector | `gate_features(diag)->np.ndarray[8]` and `GATE_FEATURE_KEYS` | 8 fixed keys, see §4.1 |
| Three-setting eval | `h2cmi.eval.harness.run_three_settings` | returns `{strict_dg, gate_info, offline_tta, online_tta}` |
| Offline-TTA eval (gate-optional) | `evaluate_offline_tta(model, X, y, domain, cfg, source_prior, gate=None, device)` | selective panel; see §2.3 |
| Gate training | `train_safety_gate(model, X, y, site, cfg, pseudo_unit_levels, source_prior, device)->(SafetyGate, info)` | pseudo-target LODO harness |
| Metrics | `h2cmi.eval.metrics.metric_panel(prob, y, domain)` | `{balanced_acc, macro_f1, nll, brier, ece, worst_domain_bacc, domain_cvar25, per_domain_bacc}` |
| Cluster bootstrap | `cluster_bootstrap_ci(per_domain_delta, n_boot, alpha, seed)` | `{mean, lo, hi, p_gt0, n}` |
| Signed conditional leakage | `h2cmi.eval.leakage.crossfit_conditional_leakage(Z, y, domains, dag, n_classes, ..., n_perm=0)` | per-factor `{I_hat, ce, h_ref, cond_dom_acc, budget, [null_q95, null_mean, excess]}` |
| Hierarchical CMI | `h2cmi.cmi.hierarchical.HierarchicalCMI.estimate(z, y, lev, pk, grl=None)` | `{factor: I_hat_tensor}` (signed, not truncated) |
| Domain DAG | `h2cmi.domains.dag.DomainDAG` / `DomainFactor` / `DomainLabels` | factors carry `handling ∈ {invariant, random_effect, conditional, label_mechanism}`, `budget`, `determines_label` |
| Synthetic driver | `h2cmi.run_synthetic.main` | report keys `{config, strict_dg, offline_tta, online_tta, gate_info, leakage}` |

### 2.2 Config surface (verified `h2cmi.config`)
Ten dataclasses; the two Project B cares about most:

- `TTAConfig`: `transform, lowrank, trust_region, trust_region_b, logdet_weight, prior_kl, dirichlet,
  em_iters, em_lr, online_ema, min_target(=16), min_effective_classes(=2)`.
- `GateConfig`: `enabled, model, harm_delta(=0.0), risk_threshold(=0.5), min_evidence(=0.0)`.
- `H2Config` aggregates all ten + `n_classes`; `H2Config.small()` shrinks epochs/critic/EM for `--fast`.

Project B adds a **new** `RouterConfig` (see §8), leaving `TTAConfig`/`GateConfig` untouched.

### 2.3 The existing selective-risk contract (what `evaluate_offline_tta` already returns)
`evaluate_offline_tta(..., gate=None)` returns, per target set:
`identity`, `adapt`, `selective` (three full `metric_panel`s), plus `delta_adapt`, `delta_selective`
(paired `panel_delta`s), `per_domain_gain`, `gain_bootstrap` (`cluster_bootstrap_ci`), `gate_decisions`,
and a nested `selective_risk = {coverage, avoided_harm, missed_benefit, selective_gain}`. When a fitted
`gate` is passed, the per-domain selective prediction is `p_adapt if do_adapt else p_identity`, where
`do_adapt = res.adapted and gate.should_adapt(g, res.diagnostics["delta_density_nll"])`.
**Project B's router harness (§9) extends this contract to per-action, conformal, reason-coded output** — it
does not overwrite it.

### 2.4 Substrate reality vs. originating-plan assumptions (corrections that shape the API)
These are the assumptions in the launching plan that the local source **contradicts or refines**; Step-2 code
must follow the *real* column.

1. **`reference_prior` is a module-level function**, not `H2Model.reference_prior()`. Router calls
   `trainer.reference_prior(y_src, C, "source_marginal")` to obtain `π_S`.
2. **TTA runs on frozen *embeddings*, not raw X, and freezes the *density* params (encoder/classifier are
   absent from the TTA module).** So the `OFFLINE_TTA` action must first compute
   `U = torch.tensor(model.embed(X_tgt))` and pass `U` into `ClassConditionalTTA.fit(U)`. `π_T` is updated in
   **closed form** in the EM M-step, Dirichlet/KL-anchored to `π_S`; only `(A,b)` move by Adam. This means
   the substrate is *already* prior-anchored — a fact the prior-decoupling design (§7) leans on.
3. **There is no dedicated "safety-gated offline TTA" evaluator.** Gating is an optional `gate=` parameter on
   `evaluate_offline_tta`. Project B's router harness is the new orchestrator (§9).
4. **`SafetyGate` contains no LODO loop.** It fits one classifier on a precomputed `(M,8)` feature matrix and
   `gains = bAcc_adapt − bAcc_identity`. The leave-one-source-domain-out construction lives in
   `harness.train_safety_gate`. ACAR (§6) reuses *that* pseudo-target harness to generate its calibration set.
   The gate estimator is configurable (`logistic` default, else `GradientBoostingClassifier`).
5. **`cmi_residual` is inert in the TTA diagnostics** — it is hard-coded to `0.0` in every code path of
   `ClassConditionalTTA`, and no `cmi_residual` field exists in `leakage.py`/`hierarchical.py`. So the 6th
   `GATE_FEATURE_KEYS` slot currently carries no signal. Project B must **populate it** from
   `HierarchicalCMI.estimate(...)` (signed per-factor `I_hat`) or `crossfit_conditional_leakage(...)`'s
   `excess`-over-null, or explicitly document it as a zero placeholder (§4.2). This is a real, previously
   silent, dead feature.
6. **The label mechanism `p(Ỹ|Y*,D)` lives in `h2cmi.label.site_mechanism`, not in `dag.py`.** `dag.py` only
   carries a `determines_label` flag and the `label_mechanism` handling category; `label_mechanism` factors
   are excluded from the encoder CMI penalty. Prior-decoupling (§7) must not conflate label-mechanism factors
   with covariate factors.
7. **The density head exposes no `nll`/`score` method.** Support/OOD scores come from `log_prob_all(z)` and
   `log_prob(z, y)`; a marginal density NLL is `−mean(logsumexp_c[log_prob_all(z) + log π_c])`. §4.2 specifies
   the two prior-conditioned NLLs from this primitive.
8. **`crossfit_conditional_leakage` has no p-value and `n_perm` defaults to 0.** Significance is `excess =
   max(I_hat − null_q95, 0)`; the router must pass `n_perm > 0` to get a null at all.

---

## 3. Action space

### 3.1 Enum (v1 — fixed at four)
```
REFUSE        # emit no decode; return OACI reason codes + diagnostics only
IDENTITY      # source-only / strict-DG prediction, no target adaptation
OFFLINE_TTA   # batch transductive TTA: ClassConditionalTTA.fit(U) on frozen target embeddings
ONLINE_TTA    # streaming prior/affine update: ClassConditionalTTA.fit_online(batches), predict-before-update
```
`REQUEST_CALIBRATION` is intentionally excluded (the project is calibration-free / zero target labels). A
deployment UI can map `REFUSE + OACI_TOS_*` onto "request calibration" downstream.

### 3.2 Action semantics on the real substrate
- `IDENTITY`: `proba = model.predict_proba(X_tgt, prior=π_S, mode="blend")` (equivalently `evaluate_strict_dg`
  with `prior=π_S`). This is the safe fallback every gate degrades to.
- `OFFLINE_TTA`: `U = torch.tensor(model.embed(X_tgt)); res = ClassConditionalTTA(density, π_S, cfg.tta,
  C).fit(U)`; predict via the harness `_predict_transform(model, U, res.transform, res.pi_T)`. If
  `res.adapted is False` (e.g. `N < cfg.tta.min_target` → `reason="too_few_target"`), this action *is already*
  identity — the router treats a non-adapting `OFFLINE_TTA` as `IDENTITY` and records the TTA reason.
- `ONLINE_TTA`: `ClassConditionalTTA.fit_online(batches)`; each batch is predicted under the running
  EMA transform/prior **before** its update (no peeking — matches `evaluate_online_tta`).

### 3.3 Invariant enforced by construction
Every non-`REFUSE` action only ever touches `(A, b, π_T)`. The encoder, density, and classifier weights are
frozen. `REFUSE` produces no class prediction at all.

---

## 4. TOS refusal checks

TOS = **T**arget-support / **O**verlap **S**anity. Not a single OOD scalar — a vector, thresholded per
component, with each failing component mapped to an OACI reason (§5).

### 4.1 Base support vector (already produced by the substrate)
The 8 `GATE_FEATURE_KEYS`, computed by `ClassConditionalTTA` diagnostics + `gate_features`:
```
delta_density_nll   transform_norm    condition_number   prior_shift
pred_disagreement   cmi_residual      ood_score          ess
```
Plus, from the identity/adapt diagnostics: `nll_before`, `nll_after`, `n_target` (= `U.shape[0]`),
`min_class_responsibility` (min over EM class responsibilities, added by the router).

### 4.2 Two new prior-decoupled support features (router-computed)
Computed directly from `ClassConditionalDensity.log_prob_all`:
```
density_nll_under_source_prior = -mean_z logsumexp_c[ log_prob_all(z)[c] + log π_S[c] ]
density_nll_under_target_prior = -mean_z logsumexp_c[ log_prob_all(z)[c] + log π̂_T[c] ]
support_gap = density_nll_under_source_prior - density_nll_under_target_prior
```
`π̂_T` is the TTA-estimated target prior (`TTAResult.pi_T`). **Decoupling rule:** if source-prior NLL is bad
but target-prior NLL is fine (`support_gap` large positive), the degradation is *prior-only* → **do not
refuse**; emit `OACI_PRIOR_SHIFT_ONLY_INFO`. Only when target-prior NLL is *also* bad is it genuine support
mismatch → `OACI_TOS_DENSITY_OOD` / `OACI_TOS_SUPPORT_MISMATCH`.

### 4.3 `cmi_residual` remediation (from §2.4-5) — labels-permitting only
Because the substrate hard-codes `cmi_residual = 0.0` it carries no signal until populated — but population is
**only legitimate when true labels are available**. Pseudo-label leakage is explicitly **not** a trusted
refusal signal: `I(Z;D|Ŷ)` conditioned on the model's own pseudo-labels is not the target estimand
`I(Z;D|Y)`; it can be masked by the model's own errors, or worse, report the model's own bias as leakage.
Two disjoint modes:

**Calibration / evaluation mode** (source pseudo-targets, or any split where true `y` + `domains` exist):
`leakage_excess_true` = worst-factor `excess` of
`crossfit_conditional_leakage(Z=embed(X), y=y_true, domains, dag, C, n_perm>0)`. This MAY populate
`cmi_residual` for source-pseudo-target calibration and for evaluation reports. `n_perm` **must be `> 0`** (the
substrate defaults it to `0`, i.e. no null → no `excess`).

**Deployment route mode** (`route_target`, target `y` unavailable):
Do **not** use pseudo-label leakage as a trusted `cmi_residual`. Set `cmi_residual = 0.0`, emit
`OACI_LEAKAGE_RESIDUAL_UNAVAILABLE` (audit-only, never a refusal), and set
`diagnostics["cmi_residual_available"] = False`. An optional `pseudo_cmi_residual` MAY be logged separately
for audit, but it is **not** a refusal trigger in v1. `HierarchicalCMI.estimate`'s signed `I_hat` (no null
correction) is a fallback/ablation source only — never the v1 primary. Never a silent zero.

### 4.4 Thresholds (structure only; numbers frozen at Step 2)
Each check is `feature (>|<) τ`. Defaults live in `RouterConfig` (§8); a `NULL_*` canary config disables all
gates to confirm the harness is not self-fulfilling. Thresholds are set on **source pseudo-targets only**
(§6), never on the real target.

---

## 5. OACI reason codes

OACI = **O**verlap-**A**ware **C**onditional **I**nvariance reason enum. Auditable, one-per-failure. Refusal
reasons vs. audit-only info are explicitly separated.

```
# --- pass ---
OACI_OK

# --- TOS / support refusals ---
OACI_TOS_TOO_FEW_TARGET               # n_target < min_target  (mirrors TTA "too_few_target")
OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE    # ess below floor
OACI_TOS_SINGLE_CLASS_IDENTIFIABILITY # effective target classes < min_effective_classes
OACI_TOS_DENSITY_OOD                  # density NLL under BOTH priors bad
OACI_TOS_SUPPORT_MISMATCH             # support_gap + condition_number jointly out of range

# --- TTA-stability refusals ---
OACI_TTA_UNSTABLE_TRANSFORM           # transform_norm / condition_number over trust region
OACI_TTA_NEGATIVE_EVIDENCE            # delta_density_nll < min_evidence (adaptation worsens density fit)
OACI_TTA_HIGH_PRED_DISAGREEMENT       # pred_disagreement over threshold

# --- prior (decoupled) ---
OACI_PRIOR_SHIFT_ONLY_INFO            # AUDIT-ONLY: prior moved, support fine -> NOT a refusal
OACI_PRIOR_DECOUPLING_FAILED         # could not separate prior from support (both degrade together)

# --- ACAR (action-conditional conformal) ---
OACI_ACAR_HIGH_ACTION_RISK            # calibrated upper harm bound for action a > harm_budget[a]
OACI_ACAR_INSUFFICIENT_CALIBRATION    # too few pseudo-targets to calibrate action a
OACI_ACAR_HARM_CALIBRATION_DEGENERATE # source pseudo-target gains carry NO learnable harm signal
                                      #   (all-zero / single-class), NOT an error and NOT too-few-samples
                                      #   -> ACAR-harm unavailable; fall through to TOS/support refusal
OACI_CONF_EMPTY_ACTION_SET            # every non-refuse action failed -> REFUSE

# --- legacy gate / leakage / audit ---
OACI_GATE_HARM_RISK                   # learned SafetyGate predicts harm (baseline signal)
OACI_LEAKAGE_RESIDUAL_HIGH            # populated cmi_residual over budget
OACI_LEAKAGE_RESIDUAL_UNAVAILABLE     # AUDIT-ONLY: cmi_residual could not be computed (see 4.3)
OACI_INTERNAL_ERROR                   # reason-coded failure, never a silent None
```
Rules: `OACI_PRIOR_SHIFT_ONLY_INFO` and `OACI_LEAKAGE_RESIDUAL_UNAVAILABLE` are **audit-only** and never by
themselves cause `REFUSE`. Every code path that drops a feature or aborts must emit a code — no silent `None`,
no silent zero (project standing lesson: reason-code every feature loss, fail loud).

---

## 6. ACAR: action-conditional conformal adaptation risk

### 6.1 Estimands
For each action `a ∈ {IDENTITY, OFFLINE_TTA, ONLINE_TTA}` and each **source pseudo-target** domain `u`
(labels available because `u` is carved from source):
```
risk_error[a,u] = 1 - balanced_acc[a,u]
risk_harm[a,u]  = balanced_acc[IDENTITY,u] - balanced_acc[a,u]     # >0 means the action HURT vs identity
```
`risk_harm` is the action-conditional generalisation of the existing gate's scalar `gain = bAcc_adapt −
bAcc_identity` (note the sign flip: harm = −gain).

### 6.2 Pseudo-target construction (reuse the real harness)
Reuse `harness.train_safety_gate`'s machinery: partition source by `pseudo_unit_levels` (subject/session),
hold each unit out as a pseudo-target, run each action on it, and record `(features[a,u], risk_harm[a,u])`.
`features[a,u] = concat(gate_features(diag[a,u]), [support_gap, density_nll_under_target_prior, ...])`. A
pseudo-unit with `mask.sum() < cfg.tta.min_target` is skipped (matches the harness).

### 6.3 Per-action conformal calibration — DUAL bounds (error AND harm)
Controlling harm alone is insufficient: `IDENTITY` has `risk_harm ≡ 0` by construction, so a harm-only router
makes IDENTITY *tautologically* safe and can never answer "is this target even eligible to be predicted?".
So **both** estimands of §6.1 are calibrated. Per action `a`, fit two regressors and two split-conformal
quantiles (ridge/GBT):
```
r̂_error_a = f_error_a(features)                 r̂_harm_a = f_harm_a(features)
score_error[a,u] = max(0, risk_error[a,u] - r̂_error_a(features[a,u]))
score_harm[a,u]  = max(0, risk_harm[a,u]  - r̂_harm_a(features[a,u]))
q_error_a = Quantile_{1-α_a}({score_error[a,u]})    q_harm_a = Quantile_{1-α_a}({score_harm[a,u]})
upper_error_a(x) = r̂_error_a(features[a,x]) + q_error_a    # bounds ABSOLUTE risk -> "eligible to output?"
upper_harm_a(x)  = r̂_harm_a(features[a,x])  + q_harm_a     # bounds harm vs identity -> "allowed to adapt?"
```
Admissibility (IDENTITY is exempt from the harm bound, which is ~0 by construction):
```
IDENTITY:                TOS pass  AND  upper_error_identity <= error_budget["identity"]
OFFLINE_TTA / ONLINE_TTA: TOS pass  AND  TTA-stability pass
                          AND upper_error_action <= error_budget[action]
                          AND upper_harm_action  <= harm_budget[action]
```
`risk_harm` governs *whether adaptation is allowed*; `risk_error` governs *whether the target is eligible to
be predicted at all*. They are never conflated. If `#calib(a) < min_calib` → drop `a` with
`OACI_ACAR_INSUFFICIENT_CALIBRATION`. If no prediction action is admissible → `REFUSE` with
`OACI_CONF_EMPTY_ACTION_SET`. Selection among admissible actions follows the safe-beneficial policy of §6.6
(NOT a naive least-interventional pick).

### 6.4 Relationship to `SafetyGate`
The learned `SafetyGate.predict_harm_prob` becomes **one feature and one baseline**, not the decision. ACAR's
value-add over the raw gate is (i) per-action bounds instead of one marginal verdict, (ii) a finite-sample
conformal *upper* bound instead of a point probability, (iii) explicit `INSUFFICIENT_CALIBRATION` abstention.
The ablation "ACAR vs. bare `SafetyGate`" is a required Step-2 comparison.

### 6.5 Honest caveat on the guarantee
The conformal guarantee is **only over the source pseudo-target distribution**. It transfers to the real
target **iff** pseudo-targets are exchangeable with the target — which is exactly the source-only
non-identifiability the parent project has repeatedly hit. ACAR therefore claims a *calibrated source-side
harm bound*, not a target-side guarantee, and the evaluation (§9) reports the gap between the two explicitly.
The action-conditional-conformal framing follows recent conformal risk-averse decision work
(*"Conformal Risk-Averse Decision Making with Action Conditional Guarantee"*, arXiv:2606.05551, ICML 2026 —
**citation verified 2026-07-06**). That work (i) builds action-conditional conformal prediction sets, (ii)
uses them as a proxy for the feasible decision space of a risk-averse (action-conditional VaR) decision maker,
and (iii) gives a finite-sample pinball-loss algorithm — strengthening Kiyani et al. (2025)'s *marginal-only*
guarantee to a *per-action* one. ACAR (§6.3) is the EEG-deployment instantiation: our `q_a` quantiles and
`upper_error_a` / `upper_harm_a` bounds are the pinball/split-conformal analogue applied to the two estimands
of §6.1.

### 6.6 Action-selection policy (safe-beneficial, NOT least-interventional)
A naive "pick the least-interventional admissible action" self-locks to `IDENTITY` whenever IDENTITY is
admissible, so `OFFLINE_TTA` could never be chosen even when it clearly helps. The router therefore uses a
safe-beneficial policy:
```
1. default REFUSE.
2. run TOS / support / prior-decoupled checks (§4, §7).
3. build the admissible prediction-action set (§6.3).
4. if NO prediction action is admissible          -> REFUSE (OACI_CONF_EMPTY_ACTION_SET).
5. if a TTA action satisfies ALL of:
       upper_error_action    <= error_budget[action]
       upper_harm_action     <= harm_budget[action]
       predicted_gain_action >= min_expected_gain
       delta_density_nll      >= min_delta_density_nll
   -> select that TTA action (prefer OFFLINE_TTA over ONLINE_TTA on ties).
6. else if IDENTITY admissible                     -> IDENTITY.
7. else                                            -> REFUSE.
```
v1 defines `predicted_gain_action = − r̂_harm_action` (expected reduction in harm vs identity; a proper
conformal *lower* bound on gain is deferred to a later version). `min_expected_gain` (default `0.02`) lives in
`RouterConfig`. This keeps the router refusal-first **without** letting it collapse to a permanent IDENTITY
policy — `risk_harm` still guards against harmful adaptation, but a beneficial TTA action can win.

---

## 7. Prior-decoupled protocol

### 7.1 Principle
A change in the label prior `π_T ≠ π_S` is a *first-class, benign* quantity, not evidence of OOD. The
substrate already anchors `π̂_T` to `π_S` via a Dirichlet/KL term in the EM M-step (§2.4-2), so the router's
job is to **not double-count** prior movement as support failure.

### 7.2 Decoupling test (from §4.2)
```
if density_nll_under_target_prior is GOOD and density_nll_under_source_prior is BAD:
        -> prior-only movement; action allowed on support grounds; emit OACI_PRIOR_SHIFT_ONLY_INFO
elif both BAD:
        -> genuine support mismatch; refuse with OACI_TOS_DENSITY_OOD / OACI_TOS_SUPPORT_MISMATCH
elif decoupling estimate is itself unstable (e.g. π̂_T degenerate, single-class):
        -> OACI_PRIOR_DECOUPLING_FAILED
```
`prior_shift` (an existing diagnostic) is fed to ACAR as a **feature** but is **removed from the refusal
trigger set** — it can only influence the decision through a *calibrated* action-conditional bound, never as
a raw threshold.

### 7.3 Label-mechanism separation
Factors with `handling == "label_mechanism"` (or `determines_label == True`) are excluded from support/leakage
refusal, mirroring `DomainFactor.penalised` (which is `True` only for `invariant`/`random_effect`). They are
handled — if at all — through `h2cmi.label.site_mechanism`, not through TOS.

---

## 8. Router API draft

*(Draft only — no implementation in Step 1. Signatures chosen to match the real substrate: `model` is an
`H2Model`; source calibration takes `pseudo_unit_levels` like `train_safety_gate`; `source_prior` is a numpy
vector from `trainer.reference_prior`.)*

```python
from dataclasses import dataclass, field
from enum import Enum


class RouterAction(Enum):
    REFUSE = "refuse"
    IDENTITY = "identity"
    OFFLINE_TTA = "offline_tta"
    ONLINE_TTA = "online_tta"


class OACIReason(Enum):
    OK = "oaci_ok"
    TOS_TOO_FEW_TARGET = "oaci_tos_too_few_target"
    TOS_LOW_EFFECTIVE_SAMPLE_SIZE = "oaci_tos_low_effective_sample_size"
    TOS_SINGLE_CLASS_IDENTIFIABILITY = "oaci_tos_single_class_identifiability"
    TOS_DENSITY_OOD = "oaci_tos_density_ood"
    TOS_SUPPORT_MISMATCH = "oaci_tos_support_mismatch"
    TTA_UNSTABLE_TRANSFORM = "oaci_tta_unstable_transform"
    TTA_NEGATIVE_EVIDENCE = "oaci_tta_negative_evidence"
    TTA_HIGH_PRED_DISAGREEMENT = "oaci_tta_high_pred_disagreement"
    PRIOR_SHIFT_ONLY_INFO = "oaci_prior_shift_only_info"          # audit-only
    PRIOR_DECOUPLING_FAILED = "oaci_prior_decoupling_failed"
    ACAR_HIGH_ACTION_RISK = "oaci_acar_high_action_risk"
    ACAR_INSUFFICIENT_CALIBRATION = "oaci_acar_insufficient_calibration"
    CONF_EMPTY_ACTION_SET = "oaci_conf_empty_action_set"
    GATE_HARM_RISK = "oaci_gate_harm_risk"
    LEAKAGE_RESIDUAL_HIGH = "oaci_leakage_residual_high"
    LEAKAGE_RESIDUAL_UNAVAILABLE = "oaci_leakage_residual_unavailable"  # audit-only
    INTERNAL_ERROR = "oaci_internal_error"


@dataclass
class RouterConfig:
    # TOS thresholds
    min_target: int = 16                 # inherits cfg.tta.min_target
    min_effective_classes: int = 2
    ess_floor: float = 8.0
    max_condition_number: float = 1e3
    max_transform_norm: float = 4.0
    min_delta_density_nll: float = 0.0   # inherits GateConfig.min_evidence
    max_pred_disagreement: float = 0.5
    density_ood_nll_z: float = 3.0       # z-score vs source pseudo-target NLL
    # ACAR
    alpha: dict = field(default_factory=lambda: {"identity": 0.10,
                                                 "offline_tta": 0.10,
                                                 "online_tta": 0.10})
    harm_budget: dict = field(default_factory=lambda: {"identity": 0.0,
                                                       "offline_tta": 0.02,
                                                       "online_tta": 0.02})
    error_budget: dict = field(default_factory=lambda: {"identity": 0.45,   # require upper-bounded bAcc >= 0.55
                                                        "offline_tta": 0.45,
                                                        "online_tta": 0.45})
    min_expected_gain: float = 0.02      # safe-beneficial policy (SS6.6): a TTA action must beat this
    min_calib: int = 8
    leakage_n_perm: int = 200            # crossfit_conditional_leakage null (MUST be > 0)
    # meta
    refusal_first: bool = True
    disable_all_gates: bool = False      # NULL_* canary


@dataclass
class RouterDecision:
    action: RouterAction
    accepted: bool
    reason_codes: list          # list[OACIReason]
    diagnostics: dict           # merged TTA/gate/prior/leakage diagnostics (incl. audit-only info)
    action_scores: dict         # {action_name: {r_hat_error, r_hat_harm, predicted_gain}}
    conformal_bounds: dict      # {action_name: {upper_error, upper_harm}}


class RefusalFirstRouter:
    def __init__(self, cfg: RouterConfig): ...

    def fit_source_calibration(self, model, X_src, y_src, domains_src, dag,
                               pseudo_unit_levels, source_prior):
        """Build (features[a,u], risk_harm[a,u]) over source pseudo-targets via the
        train_safety_gate harness; fit per-action regressors f_a and conformal quantiles q_a;
        fit the baseline SafetyGate. No target data touched."""
        ...

    def route_target(self, model, X_tgt, domain_tgt, source_prior, mode: str = "offline"):
        """Return a RouterDecision for one unlabelled target batch/stream. Default REFUSE.
        Applies the safe-beneficial selection policy (SS6.6): promote to a TTA action only when its
        conformal error+harm bounds and the expected-gain gate all pass, else IDENTITY if eligible, else
        REFUSE. Never uses pseudo-label leakage as a trusted cmi_residual (SS4.3). mode in {offline, online}."""
        ...
```

---

## 9. Evaluation protocol

### 9.1 Report table (prior-decoupled; refused samples never folded into plain bAcc)
```
strict_dg_identity            # metric_panel under IDENTITY
offline_tta_raw               # metric_panel under always-adapt (no router)
offline_tta_router_selected   # metric_panel over ACCEPTED domains only
online_tta_raw
online_tta_router_selected
refusal_rate                  # fraction of target domains routed to REFUSE
coverage_by_action            # {action: fraction of domains}
accepted_bAcc                 # bAcc on accepted domains only
harm_rate_by_action           # fraction where risk_harm > 0
avoided_harm / missed_benefit / selective_gain   # reuse selective_risk contract (SS2.3)
oaci_reason_histogram         # count per OACIReason
prior_shift_only_cases        # count of OACI_PRIOR_SHIFT_ONLY_INFO
support_mismatch_cases        # count of OACI_TOS_SUPPORT_MISMATCH
acar_bound_vs_realized        # per action: mean(upper_harm_a) vs realized risk_harm on real target (the gap)
```
Deltas use `panel_delta` + `cluster_bootstrap_ci` (domains as clusters), exactly as the existing harness.

### 9.2 Worlds (synthetic substrate, FULL training — NOT `--fast`; established in Step-2A)
Step-2A (`scripts/project_b_sweep_synthetic.py`, 30 epochs) established these worlds and **two distinct
harmful regimes that must never be merged into one metric**:
- **Recoverable world = R2** (`cov1.2/prior0.4/montage0.2/concept0`): raw OFFLINE_TTA reliably HELPS
  (d_bAcc +0.071, p_gt0 0.95, 3/3 seeds). Router should **accept** — refusing here is a false refusal.
- **Harmful-CAL (H-CAL)** — the *attempted* source-calibratable harmful world (`concept_frac≈0.5` → ~3
  concept sites, target hits one, ~2 remain in source, `source_concept_count=2`). **Step-2A RESULT: NOT
  achievable with the current gate design (Case B).** Across all 30 HFRAC runs the target is often reliably
  harmed (HF3 concept=1.2: 4/5 seeds negative, harmful seeds p_gt0=0.00) but **`pseudo_gain ≡ 0` in every
  run** (`pseudo_gain_min=max=0`, `2class=0/30`, `gate_auroc` NaN 30/30) — even at `source_concept_count=2`.
  See the confirmed caveat below.
- **Harmful-OOD (H-OOD) stress = seed-32 @ `concept_frac=0.17`** (lone concept site held out as target →
  `source_concept_count=0`): harm is real (d_bAcc −0.11..−0.29 across H configs) but **not source-
  identifiable** (`gate_auroc` NaN — source stays concept-free). Used ONLY to stress **TOS/support-aware
  refusal** (§4); **never** as ACAR calibration evidence. This is the substrate face of the source-only
  non-identifiability barrier.
- **NULL canary** (`disable_all_gates=True`): accepted-bAcc collapses to raw-TTA (harness not self-fulfilling).

> **CONFIRMED finding (Step-2A HFRAC, 30 runs, commit pending):** the ACAR/`SafetyGate` source-only harm
> signal is **structurally unavailable by gate design**, not by shift strength. `train_safety_gate` measures
> TTA gain on source subjects the model was *trained on* (in-distribution); on those, TTA is **inert**
> (`pseudo_gain ≡ 0` — min=mean=max=0 in all 30 runs, so 0/30 two-class, `gate_auroc` NaN 30/30), even when
> `source_concept_count=2`. Harm manifests only on the truly-held-out **OOD** target (HF3: reliable target
> harm, p_gt0=0.00). **Consequence for Project B:** a purely source-calibrated harm regressor (ACAR §6.3,
> `SafetyGate` §6.4) has a **degenerate all-zero calibration set** and cannot be fit — this is the source-only
> non-identifiability barrier, made concrete and reproducible at the substrate. A fix requires calibrating on
> genuinely-OOD pseudo-targets, i.e. a **nested-LODO gate** that retrains the encoder excluding each
> pseudo-domain (expensive; and whether even that transfers is the same open source-only question). This is a
> load-bearing negative result and must gate the Step-2B/ACAR design decision.

> **Nested source-SITE LODO probe (Step-2A-NL, HF3, same 5 seeds, 25 runs — bounds the claim).** To avoid
> overclaiming, we asked whether calibrating on genuinely held-out source *sites* (nested LODO: retrain on 4
> sites excluding {target, pseudo-site}, evaluate TTA on the OOD held-out source site) yields a
> non-degenerate signal where the non-nested subject gate saw `pseudo_gain ≡ 0`. **Result: POSITIVE by the
> pre-registered rule but not usefully so** — 3/5 seeds show a 2-class pseudo-gain (vs 0/5 for the subject
> gate), so the degeneracy is a property of the *non-nested* protocol, NOT proof that every source-only
> protocol is impossible. **However the signal is (a) borderline** (exactly 3/5, 3/5 thresholds), **(b) driven
> by cross-site variance, not concept shift** (concept pseudo-sites mean d +0.027 vs non-concept +0.037), and
> **(c) NON-TRANSFERABLE to the deployment target**: `corr(target_delta_bacc, nested_pseudo_gain_mean) = +0.06`
> — the seed with the *strongest* target harm (seed8, −0.153, p=0.00) has **0/5** nested pseudo-harm, while
> seed4 (−0.137) has 4/5. Worse, where nested harm is real (seed4/u3, −0.183) it coincides with extreme TOS
> flags (`transform_norm≈20`, `ess≈5`) that support-aware refusal (§4) catches anyway.
> **Net:** a source-only ACAR-harm regressor is *not literally impossible* but is degenerate under the cheap
> protocol and non-predictive of target harm under the nested protocol. **Project B v1 therefore treats
> ACAR-harm as a conditional module (emit `OACI_ACAR_HARM_CALIBRATION_DEGENERATE` when the calibration set is
> degenerate) and makes TOS/support-aware refusal (§4) the load-bearing safeguard.** Nested calibration is
> recorded as a possible *future* expensive optional mode, not a v1 dependency; the Step-2B scope is
> unchanged (`reasons`/`actions`/`features` first).

> **Baseline reality check (2026-07-06, `--fast` → 4 epochs).** The two Step-1 baseline runs do **not** yet
> realise the intended recoverable-vs-harmful contrast, and Step 2 must fix the world config before using it
> as a go/no-go substrate:
> - `--concept 0.0` ("recoverable"): source model is under-trained (strict-DG bAcc **0.457**, barely above
>   the 0.333 chance floor), and raw offline-TTA **hurts** (`delta_adapt.d_balanced_acc = −0.174`,
>   `gain_bootstrap.mean = −0.186`, `p_gt0 = 0`). The existing gate **correctly refused all** domains
>   (`selective_risk.coverage = 0.0`, `avoided_harm = 0.186`, selective delta = 0). So the *gate behaves*,
>   but the world is not a "TTA-helps" world.
> - `--concept 0.6 --concept_frac 0.5` ("harmful"): strict-DG bAcc **0.667**, raw offline-TTA is **neutral**
>   (`delta_adapt.d_balanced_acc = 0.0`), gate adapts all (`coverage = 1.0`), no harm, no gain.
> - Net: under `--fast` the regimes are effectively *inverted/flattened* — the concept-0.0 world is where TTA
>   hurts, and neither world shows TTA *helping*. **Step-2 action:** drop `--fast` (it force-sets 4 epochs and
>   overrides `--epochs`), train to convergence, enlarge the sim, and tune `--cov/--prior/--montage` to
>   produce a genuine "TTA-helps" world before freezing the go/no-go. Also note `gate_info.harm_metrics.auroc
>   = NaN` in the recoverable run: all 12 pseudo-targets were single-harm-class, exactly the
>   `OACI_ACAR_INSUFFICIENT_CALIBRATION` / single-harm-class degeneracy the router must handle (§6.3, §5).
> - Confirmed live: `ClassConditionalTTA` diagnostics carry all 8 gate keys **with `cmi_residual = 0.0`**
>   (the dead-feature finding of §2.4-5), and `delta_density_nll` can be **negative** under adaptation
>   (self-test: nll 7.61 → 8.32), i.e. `OACI_TTA_NEGATIVE_EVIDENCE` is a live, reachable trigger.

### 9.3 Real-EEG bridge (later, not Step 2)
Same protocol on the parent line's real datasets (Lee2019 cross-session, BNCI cross-subject), reusing
`cmi/` loaders through a thin adapter that emits the `(X, y, DomainLabels, DomainDAG)` the h2cmi harness
expects. Gated behind a positive synthetic result.

### 9.4 Guardrails
- All thresholds/regressors/quantiles fit on **source pseudo-targets only**.
- Config hash of `RouterConfig` frozen and logged before any target evaluation (pre-registration, per the
  project's C19 lesson).
- Every dropped feature / aborted path emits an OACI code; NaN sentinels never silently become 0.

---

## 10. Step-2 implementation plan

Build order (each step is additive; no edits to `tta/class_conditional.py`, `gate/safety_gate.py`, or
`eval/harness.py` until §10.6 — wrappers only):

1. **`h2cmi/router/reasons.py`** — `OACIReason` enum + `REFUSAL_REASONS` / `AUDIT_ONLY_REASONS` sets. Pure,
   no deps. Unit-tested for exhaustiveness.
2. **`h2cmi/router/actions.py`** — `RouterAction` enum + thin action executors that call
   `model.embed`/`predict_proba`, `ClassConditionalTTA.fit`/`fit_online`, returning `(proba, TTAResult)`.
3. **`h2cmi/router/features.py`** — assemble the router feature vector: wrap `gate_features`, add
   `support_gap` + two prior-conditioned NLLs (§4.2), and populate `cmi_residual` from
   `crossfit_conditional_leakage`/`HierarchicalCMI.estimate` (§4.3). Fail-loud on missing keys.
4. **`h2cmi/router/acar.py`** — pseudo-target harness (reuse `train_safety_gate` partitioning), per-action
   regressors, split-conformal quantiles, admissibility test.
5. **`h2cmi/router/router.py`** — `RouterConfig`, `RouterDecision`, `RefusalFirstRouter`
   (`fit_source_calibration` + `route_target`), refusal-first control flow, OACI emission.
6. **`h2cmi/eval/router_harness.py`** — `evaluate_router(...)` producing the §9.1 table; internally *calls*
   (does not modify) `evaluate_offline_tta`/`evaluate_online_tta`.
7. **`h2cmi/tests/test_router_smoke.py`** — mirror `tests/test_smoke.py`: build simulator, source/target
   split, train, fit source calibration, route, assert finite metrics + that concept-world refusal_rate >
   recoverable-world refusal_rate.
8. **Pre-registration**: freeze `RouterConfig` hash + go/no-go in `notes/PROJECT_B_FROZEN.md` **before** any
   target evaluation.

**Guard for Step 2:** ship steps 1–3 + a failing/So-far test first, get review, then 4–7. Do not touch the
three core substrate files.

---

### Appendix A — verified discrepancy ledger (for Step-2 correctness)
| # | Plan assumption | Local reality | Impact on router |
|---|---|---|---|
| 1 | `H2Model.reference_prior()` | module-level `trainer.reference_prior(y, C, mode)` | call the fn, not a method |
| 2 | TTA on `X`, freezes encoder | TTA on `embed(X)`, freezes **density**; `π_T` closed-form Dirichlet-anchored | action must embed first; prior already anchored |
| 3 | harness has safety-gated TTA fn | `gate=` param on `evaluate_offline_tta` | router harness is the new orchestrator |
| 4 | `SafetyGate` does LODO internally | LODO lives in `train_safety_gate`; gate just fits `(M,8)` | ACAR reuses `train_safety_gate` partitioning |
| 5 | `cmi_residual` is a real signal | hard-coded `0.0` everywhere | router must populate it or emit `LEAKAGE_RESIDUAL_UNAVAILABLE` |
| 6 | label mechanism in `dag.py` | in `h2cmi.label.site_mechanism`; dag only flags it | keep label-mech factors out of TOS |
| 7 | `density.nll`/`score` | only `log_prob`/`log_prob_all`; NLL via logsumexp | compute the two prior-NLLs explicitly |
| 8 | leakage null on by default | `n_perm=0` default, no p-value, `excess` vs `null_q95` | pass `n_perm>0`; use `excess` |

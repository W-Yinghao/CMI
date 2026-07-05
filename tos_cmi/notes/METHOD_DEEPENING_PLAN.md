# Method-deepening phase (branch `method-deepen-v1`, from `tos-cmi-multidataset-v1`)

Paper is FROZEN at the multi-dataset snapshot. This phase does NOT edit the paper; it deepens the method from
"erasure has no target gain" to **"a verifiable source-only decision system: refuse useless/harmful erasure,
accept only with source-OOD benefit evidence."** All results go under `tos_cmi/results/method_deepen/`.

## Priority order
1. **Track B — source-OOD benefit gate (HIGHEST).** Can source-only evidence (no held-out target) reject the
   useless/harmful erasures we saw on real EEG? Datasets: Lee2019, Cho2017 (50+ subjects → inner
   leave-one-source-subject-out has power); both backbones. HGD secondary.
2. Task-preserving / conditional erasure variants (class-conditional LEACE, oblique task-preserving).
3. V2 semi-synthetic certificate on real latents (worlds A/B/C: accept / reject-safety / reject-no-benefit).
4. Failure-mechanism audit (leakage↓ vs target Δ; task–subject entanglement; nonlinear residual; random/compression).

## NOT doing
Track E (no positive frozen signal); full big-N LPC sweep; full big-N capacity factorial; more datasets.

## Track B gate (FIXED thresholds, pre-registered before touching target)
Three layers, kept separate:
- **Safety gate.** Within-source (stratified) task-bAcc drop from erasing. REJECT if drop **UCB > 0.02**.
  (Catches Lee/Cho-EEGNet LEACE/RLACE driving task → chance.)
- **Benefit gate.** Source leave-one-source-subject-out pseudo-target ΔbAcc (erased − full). ACCEPT only if
  **LCB > +0.01** (and safe).
- **Domain-gain.** Diagnostic ONLY (confirms subject info was removed); never sufficient for accept — leakage
  reduction is not benefit.
- Action: REJECT if unsafe; ACCEPT if safe ∧ benefit-LCB > +0.01; else ABSTAIN.
- **Target labels: post-hoc audit only** (false-accept, harm-prevented) — never in the gate decision.

## Track B acceptance
1. Lee/Cho-EEGNet LEACE/RLACE harms rejected by the safety gate.
2. Any method with actual target ΔbAcc upper CI < +0.01 is NOT accepted.
3. false-accept count ≈ 0.
4. All-real-EEG reject/abstain is an acceptable (good) outcome.
5. Any ACCEPT must have source-LOSO LCB > +0.01 and no source task harm.

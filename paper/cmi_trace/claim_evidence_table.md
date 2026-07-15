# CMI-Trace — claim / evidence ledger (P0/P1 revision)

Authoritative mapping from every quantitative claim to its generating **script**, **raw artifact**,
**aggregation**, **seeds/folds**, **CI method**, and **allowed / forbidden wording**. `STATUS` is one of
`CONFIRMED` (real artifact exists), `PENDING` (code + launched job; awaiting GPU-hours), `WEAKENED`,
`OVERTURNED`. No claim may be upgraded from PENDING without the named artifact.

Base manuscript source: `paper/cigl_latex/` (CIGL LaTeX; the CMI-Trace revision extends it — see
`paper/cmi_trace/MANUSCRIPT_INTEGRATION.md`). Repo SHA + config hash for every result row are stamped by the
runners' firewall metadata (`config_sha256=002e9241…` for `configs/cmi_trace_p0p1.yaml`).

Statistical unit = held-out subject / outer fold. Main uncertainty = **paired fold/subject-cluster 95%
bootstrap** (seeds within a fold travel together). Seed SD is descriptive only.

---

## P0.1 — same-backbone objective comparison (CORAL / C-CORAL / IRMv1 / V-REx completed)

| # | Claim | Script | Raw artifact | Aggregation | Seeds×Folds | CI | Status | Allowed / Forbidden wording |
|---|-------|--------|--------------|-------------|-------------|----|--------|------------------------------|
| 1.1 | CORAL, conditional-CORAL, IRMv1, V-REx run on the SAME DGCNN adapter + source-only LOSO as ERM/encoder-CMI/adversarial | `scripts/run_cmi_trace_objective_comparison.py` (methods via `_train_eval`, `OBJECTIVE_METHODS`) | `results/cmi_trace_p0p1/objective_comparison/<ds>/*.json` + `.audit.npz` | `raw_rows.jsonl` | 3×(9,12) | — | PENDING (jobs 896396 BNCI2014_001, 896397 BNCI2015_001) | ALLOWED: "same backbone, same source-only protocol". FORBIDDEN: leaderboard-across-architectures |
| 1.2 | Nested-selected encoder-CMI (`cigl_nested`) chosen by SOURCE-ONLY nested leave-one-source-domain val, no target labels | same runner (`_inner_source_select`) | `selected_hparams.csv` | `aggregate_cmi_trace_objective.py` | 3×(9,12) | — | PENDING | ALLOWED: "source-only nested selection". FORBIDDEN: any target-tuned λ |
| 1.3 | conditional-CORAL ignores a pure label-prior shift; detects a within-class domain moment gap | `cmi/methods/dg_penalties.py::label_coral` | unit tests | `tests/test_cmi_trace_dg_penalties.py` | — | — | CONFIRMED (10/10 tests) | ALLOWED: "class-conditional moment alignment" |
| 1.4 | Identical initialization across methods (seed-before-build, cloned init) | `_train_eval` (`torch.manual_seed` before `build_graph_task_backbone`) | test | `tests/test_cmi_trace_dg_penalties.py::test_identical_initialization_seed_before_build` | — | — | CONFIRMED | — |

## P0.2 — unified objective→effect audit + fold-cluster inference

| # | Claim | Script | Raw artifact | Aggregation | CI | Status | Wording |
|---|-------|--------|--------------|-------------|----|--------|---------|
| 2.1 | Every trained model receives the SAME audit: moment gaps, graph/node CMI + null, per-domain risk var, IRMv1 diag, feature norm/top-sv/eff-rank, exact-head reliance R_rel(k=2), same-rank random control | `cmi/eval/objective_effect_report.py` (artifact-driven, reads `.audit.npz`) | `raw_rows.jsonl` | `objective_effect_summary.csv` | cluster 95% | PENDING (jobs running); code CONFIRMED (10/10 tests + real BNCI2014_001 fold0 smoke) | ALLOWED: "same audit for every objective" |
| 2.2 | Main intervals are paired fold/subject-cluster CIs (not seed SD) | `objective_effect_report.cluster_bootstrap_ci` | — | `cluster_intervals.csv`, `paired_deltas.csv` | cluster 95% | CONFIRMED (method + tests) | FORBIDDEN: reporting seed SD as the main interval |
| 2.3 | ΔI_enc < 0 does NOT imply ΔR_rel < 0 (measurement→control gap) | audit joins CMI + reliance per model | `paired_deltas.csv` (graph_kl vs R_rel_k2) | cluster CI | PENDING (needs real runs) | ALLOWED: two explicit non-implications (see MANUSCRIPT_INTEGRATION) |

## P0.4 — deployment CI semantics

| # | Claim | Script | Artifact | Status | Wording |
|---|-------|--------|----------|--------|---------|
| 4.1 | `confirmed_practical_benefit` iff lower_95(ΔbAcc)>+0.01; `practical_gain_ruled_out` iff UPPER_95<+0.01; else inconclusive | `tos_cmi/eeg/deployment_ci.py` | `tos_cmi/tests/test_deployment_ci.py` | CONFIRMED (8/8) | FORBIDDEN: using the LOWER bound to support "no practical gain" |
| 4.2 | A +0.01 gain is "ruled out" in the main statement ONLY when upper CI<+0.01 for EVERY cell/method | `deployment_ci.practical_gain_ruled_out_everywhere`; `erasure_target_deploy.aggregate` | `erasure_target_deploy_summary.json` (deploy_ci_state) | CONFIRMED (semantics); deployment numbers PENDING frozen dumps | ALLOWED: "a +0.01 gain is ruled out (upper CI<+0.01)" |

## P0.5 — FMScope bridge (oracle vs source-only × subject vs random)

| # | Claim | Script | Artifact | Status | Wording |
|---|-------|--------|----------|--------|---------|
| 5.1 | 2×2 {ORACLE_GLOBAL_GEOMETRY, strict source-only} × {subject erasure, same-rank random} on the SAME frozen features | `tos_cmi/eeg/fmscope_protocol_bridge.py` | `results/cmi_trace_p0p1/fmscope_bridge/` | CONFIRMED code (9/9 tests, synthetic only — real frozen dumps ABSENT/pruned) | FORBIDDEN: calling the oracle mode DG/deployable |
| 5.2 | The oracle (cohort-conditioned) mode is labeled non-deployable in every artifact (`ORACLE_GLOBAL_GEOMETRY`, is_dg=False) | same | firewall + oracle tests | CONFIRMED | ALLOWED: "oracle diagnostic only" |
| 5.3 | Related work: Lin et al. (arXiv:2606.06647) studies marginal subject dominance/LEACE/spectral carrier/task-axis; Tai (arXiv:2606.09189) is privacy/attribute + cross-encoder transfer — NOT both "marginal subject-variance audits" | manuscript | `MANUSCRIPT_INTEGRATION.md` §related-work | CONFIRMED (prose) | FORBIDDEN: "both are marginal subject-variance audits" |

## P1.1 — same CMI ruler on TOS/erased latents (cross-fitted)

| # | Claim | Script | Artifact | Status | Wording |
|---|-------|--------|----------|--------|---------|
| 1.1a | The SAME posterior-KL ruler on full & erased (TOS_VD/LEACE/RLACE/INLP/random-k) with 3-way eraser-fit/posterior-train/posterior-eval cross-fitting | `cmi/eval/conditional_subject_leakage.py`; `scripts/run_cmi_trace_tos_cmi_bridge.py` | `results/cmi_trace_p0p1/tos_cmi_bridge/` | CONFIRMED code (5/5 tests; synthetic demo: full kl 0.257/res 0.98 → TOS_VD kl 0.023/res 0.50). Real EEGNet/TSMNet PENDING frozen dumps | ALLOWED: "cross-fitted neural posterior plug-in estimate"; FORBIDDEN: "exact CMI"/"calibrated bits" |

## P1.2 — synthetic numerical ground-truth CMI

| # | Claim | Script | Artifact | Status | Wording |
|---|-------|--------|----------|--------|---------|
| 1.2a | Monte-Carlo TRUTH I(Z;D|Y) from exact DGP densities + reported MC SE; ranking/calibration/MAE vs neural ruler and kNN across the 39 settings | `synthetic/true_cmi.py` (+ sanity_check/validate_proxy) | `results/cmi_trace_p0p1/synthetic_truth/` | PENDING (delegated agent) | FORBIDDEN: calling either estimator "unbiased" |

## P1.3 — multi-capacity probe + valid familywise null

| # | Claim | Script | Artifact | Status | Wording |
|---|-------|--------|----------|--------|---------|
| 1.3a | Capacity family linear/mlp_small(primary)/mlp_large each reported separately; familywise MAX null repeats max-over-capacities on every permutation (valid p); primary = mlp_small | `cmi/eval/multicapacity_probe.py` | `tests/test_multicapacity_probe.py` | CONFIRMED (6/6 + graph_leakage backcompat 13/13) | ALLOWED: "familywise maximum as underfitting guard"; FORBIDDEN: reporting a max-over-capacities without the matching max null |

## P1.4 — hardened exact-head reliance

| # | Claim | Script | Artifact | Status | Wording |
|---|-------|--------|----------|--------|---------|
| 1.4a | Every reliance row carries full metadata (subspace, centering, metric/whitening, k, fit split, loss, replay max-abs-err + tol, firewall); ≥50 same-rank random spans; rank sensitivity k=1..7; primary k=2 | `cmi/eval/reliance_audit.py` | `tests/test_reliance_audit.py`; `results/cmi_trace_p0p1/reliance_sensitivity/` | CONFIRMED (7/7). Real-run sensitivity PENDING GPU audit npz | ALLOWED: "exact-head (head-replay) reliance"; FORBIDDEN: comparing raw subspace axes across models |

## P1.5 — terminology & claims (global)

| Rule | Where enforced | Status |
|------|----------------|--------|
| "cross-fitted neural posterior plug-in estimate" (not "Barber–Agakov-style plug-in") | manuscript wording (MANUSCRIPT_INTEGRATION) | PENDING LaTeX apply |
| Separate symbols `L_CMI_train` (in-loop) vs `I_hat_audit` (held-out) | manuscript notation | PENDING LaTeX apply |
| State finite-critic underfitting can UNDER-report leakage | already present in CIGL §3/§7; reinforce | CONFIRMED (present) |
| Keep raw nats + add normalized `I_hat/H_hat(D|Y)` secondary | `conditional_subject_leakage.normalized_leakage` | CONFIRMED (code) |
| Never "exact CMI"/"unbiased"/"universal upper bound"/"calibrated bits" | grep-gate over LaTeX | CONFIRMED (CIGL already disclaims; add gate) |
| Do NOT claim subject erasure generally fails — utility is representation/protocol/task-dependent | manuscript discussion | PENDING LaTeX apply |
| Do NOT claim TOS improves DG (geometry diagnostic / guarded intervention w/ identity fallback) | manuscript | PENDING LaTeX apply |
| Decoder-side CMI OUT of core contribution/experiments this revision | scope | CONFIRMED (not added) |

---

## REAL RESULT — BNCI2014_001 complete (job 896396, 8 methods × 27 cells, fold-cluster 95% CI)

Directly evidences the measurement→control gap across the full objective family:

| Method | target bAcc Δ vs ERM | graph-CMI Δ vs ERM | R_rel(k=2) Δ vs ERM | R_rel random control |
|--------|----------------------|--------------------|--------------------|---------------------|
| CORAL | +0.005 [+0.000,+0.009] | −0.130 [−0.136,−0.124] | +0.003 [−0.001,+0.007] | −0.001 |
| C-CORAL | +0.004 [−0.001,+0.009] | −0.120 [−0.126,−0.115] | +0.002 [−0.003,+0.007] | −0.003 |
| IRMv1 | +0.011 [+0.004,+0.019] | −0.036 [−0.039,−0.032] | +0.004 [−0.003,+0.010] | +0.000 |
| V-REx | −0.002 [−0.007,+0.005] | −0.075 [−0.085,−0.066] | −0.002 [−0.005,+0.002] | −0.001 |
| cond-DANN | +0.003 [−0.015,+0.023] | −0.953 [−1.001,−0.905] | +0.028 [+0.013,+0.047] | +0.018 (elevated!) |
| Enc-CMI (0.010) | +0.008 [+0.001,+0.016] | −0.571 [−0.623,−0.525] | +0.015 [+0.002,+0.029] | +0.001 |
| Enc-CMI (nested) | +0.012 [+0.004,+0.021] | −0.619 [−0.766,−0.458] | +0.034 [+0.011,+0.057] | +0.001 |

- **ALL objectives reduce measured encoder-CMI** (every graph-CMI Δ CI excludes 0).
- **None reduces exact-head reliance**; R_rel(k=2) RISES for the strongest reducers (Enc-CMI/cond-DANN, CIs
  exclude 0 on the positive side). Enc-CMI's rise is above its random control (subject-subspace-specific);
  cond-DANN's random control is also elevated (partly a scale/dimensionality effect — see moment-gap caveat).
- **Target gains modest**; NONE clears the +0.01 confirmed-benefit lower-CI threshold (best lower bound
  +0.004, Enc-CMI-nested). Deployment verdict per P0.4: no CONFIRMED practical benefit.
- CAVEAT: raw moment gaps are scale-sensitive; all 27 cond-DANN cells show inflated gaps (adversarial GRL
  inflates feature norm — read alongside the feature_norm column). A scale-normalized moment gap is future work.
- STATUS: BNCI2014_001 CONFIRMED (`objective_effect_summary.csv`, `paired_deltas.csv`, `objective_table.tex`,
  `fig3_cmi_reliance_bacc.png`). BNCI2015_001 PENDING (job 896397 running); re-aggregate on completion.

## Non-implications the paper must state explicitly (replaces the old causal chain)

The old figure chain `CMI reduction ⇒ removability ⇒ predictor use ⇒ target gain` is REPLACED by two
non-implications:

1. `ΔÎ_enc < 0` does **not** imply `ΔR_rel < 0` (lower measured leakage ≠ lower original-head reliance).
2. successful source-side erasure does **not** imply `Δ target bAcc > 0` (removability ≠ transfer gain).

Causal wording softened: "because the residual becomes…" → "accompanied by a more concentrated and
task-aligned residual"; "the mechanism is…" → "the results are consistent with…" / "a parsimonious
explanation is…".

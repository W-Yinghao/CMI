# Target-X observability audit — PRE-REG AMENDMENT 04 (F2.1b final protocol hardening)

Transparent amendment fixing the 7 P0 bugs + the central conceptual gap PM flagged on `a4c6f97e`: the source-
whitened metric + task-contested anchor now actually enter the code. Branch `agent/cmi-trace-targetx-
observability`. Manuscript FROZEN; no full F2.1 run in F2.1b (ends at a 2-subject/dataset engineering smoke).

## D0 — the pivot enters Track A code (was only in the plan)
All geometry is defined in the SOURCE Ledoit-Wolf-WHITENED metric `Z̃ = A_s(Z − μ_s)`, `A_s = Σ_s^{-1/2}`
(`tos_cmi/eval/targetx_metric.py`). Deletions are an affine map back to raw coords `T_U(z) = μ_s + A_s^{-1}
(I − UᵀU) A_s(z − μ_s)`, so the task interface and the whitening-only baseline stay independent. The PRIMARY
basis is the **task-CONTESTED** cond span `orth(B_cond · P_row(W̃_c))` (row space of the whitened class-centered
head — EEGNet fresh logistic; DGCNN stored head `W̃_c = W_c A_s^{-1}`). The **free** (head-null) cond span is a
sanitation control, NOT selectable. Full `B_cond` kept as diagnostic. G1 and ambient random projectors live in
the whitened metric. Every row records `metric`, `basis_label` (cond_contested / cond_full / ambient_whitened /
baseline), `whitening_hash`, `cov_condition_number`, and the contested/free/full ranks.

## D1 (P0.1) — random control matched to the SELECTED rank
Fold summary reports `delta_random_selected_rank` = mean utility of ONLY the random controls of the selected
rank (0 if identity), plus `delta_random_rank1/2/3` for diagnostics. Gate 2's paired random comparison uses the
selected-rank control, never the all-rank mixture.

## D2 (P0.2) — source-greedy comparator in smoke AND full (parity)
`build_actions` always creates the source-greedy prefixes (eligible) + a standalone comparator (non-selectable)
over the SAME contested span, rank≤3, same source-task-safety gate (policy-symmetric to the primary family).
Smoke only shrinks subjects/random/epochs; no load-bearing comparator is dropped.

## D3 (P0.3) — a single SHARED, HASHED selection rule
The selector is one function `g1_select` driving off the frozen `M.RULE` (hash `M.rule_hash()`). The Gate-5
runner re-runs THAT rule on the eraser-fit basis and records `same_rule_implementation = (rule_hash == audit
rule_hash)`. The hardcoded `rule_matches = True` is removed. Action index / principal angle / rank match are
diagnostics only; the protocol requirement is same-rule, not same-action.

## D4 (P0.4) — Gate 4 confirms harm with the UPPER bound
Unsafe iff one dataset has `LCB95(Δ_TX) > 0` while another has `UCB95(Δ_TX) < −0.005`. A merely-negative LCB
(cannot rule out harm) does NOT trip Gate 4. Gate engine receives per-dataset {lo, hi}.

## D5 (P0.5) — Gate 5 is a FULL posterior-KL with a training-only null
`_posterior_kl` computes `E_{z,y}[ Σ_d q(d|z,y) log(q(d|z,y)/p̂(d|y)) ]` over the full predicted distribution
(not just the true class). The permutation null permutes D within Y on the posterior-TRAINING trials only,
refits the critic, and keeps the posterior-eval support fixed. Critics: linear logistic + MLP-small + MLP-large.

## D6 (P0.6) — tie-aware, seed-clustered observability
Spearman uses tie-aware ranks (`scipy.stats.rankdata` average). Hierarchy: per (subject, seed, rank) Spearman
→ rank-macro → seed-average within subject → subject-cluster bootstrap of the median. `observability_by_rank`
reports subject-cluster CI per rank, not a pooled coefficient.

## D7 (P0.7) — Gate 3 = constrained-hindsight paired inequality (no ratio bootstrap)
Report a **constrained** hindsight oracle (same contested span, rank≤3, source-safe) and an **unconstrained**
ceiling. Gate 3 passes iff `LCB95(Δ_hindsight_constrained) > 0` AND `LCB95(Δ_TX − 0.25·Δ_hindsight_constrained)
≥ 0`. The plain recovery ratio is descriptive only; per-fold ratios with small denominators are never
bootstrapped.

## D8 — frozen config + provenance
`configs/cmi_trace_targetx_observability.yaml` (its SHA-256 stamped into every row) + `feature_dump_manifest.csv`
(npz_sha256, latent_dim, n_source/cal/query, sessions). Config covers whitening, contested/full, thresholds,
random draws/quantile, session policy, bootstrap, oracle defs, Gate-5 critics + permutation.

## D9 — scope unchanged
Synthetic/smoke = engineering only; method GO/NO-GO needs full LOSO × 3 seeds × both datasets, cluster CI, full
controls. Full F2.1, learned oblique selector, TTE, new backbones, manuscript all HELD/FROZEN.

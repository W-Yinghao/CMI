# FSR_01 — Evidence Ledger

**Project FSR: Functional Shortcut Reliance in EEG Representations** — Step 1 evidence freeze.

Branch: `project/functional-shortcut-reliance` (cut from `project/cita-target-unlabeled-cmi` @ `c889730`).
Companion: `docs/FSR_00_PROJECT_SPEC.md` (definitions), `results/fsr_artifact_index/artifact_index.csv` (machine-readable map).

**Provenance discipline.** Every number below is quoted from a frozen on-disk artifact and carries a branch + SHA + path. Numbers were read via local `git show <sha>:<path>` / on-disk parse, never from GitHub raw URLs (which are CDN-cached and can mislead). Anything referenced but not on disk is labelled **MISSING**, never inferred. Two headline CIGL correlations and the TOS erasure/deployment numbers were additionally re-verified by hand against the raw CSV/JSON.

**How to read a row.** Each route reports where it lands on the six-level FSR ladder (L1 detectability · L2 reducibility · L3 erasability · L4 task coupling · L5 functional reliance · L6 target consequence), a collapse/task-safety verdict, the frozen decision, and the allowed vs forbidden claim boundary. A claim may only be stated as a *relationship between levels* — never inferring L5/L6 from L1/L2.

---

## 0. Frozen premises (boundary conditions — NOT pending hypotheses)

These are settled repository verdicts. FSR builds *on top of* them and may not re-open them as "maybe λ was too small."

| Premise | Verdict code | Scope | Source | SHA / branch |
|---|---|---|---|---|
| **Source-only CMI reliance-control is closed** | `SOURCE_ONLY_CMI_RELIANCE_CONTROL_FAILED_AND_CLOSED` | closes CIGL, FCIGL, dCIGL, MetaCMI as *control* methods | `docs/CIGL_70_SOURCE_ONLY_CMI_CLOSURE.md`, `results/cigl_source_only_closure/{MANIFEST,negative_gates,route_summary}` | branch of record `project/metacmi-eegnet-conformer` |
| **CMI control closed in BOTH regimes** | `CMI_CONTROL_CLOSED_BOTH_REGIMES` | source-only + target-unlabeled; audit + TTA retained | `docs/CMI_SYNTHESIS_01.md`, `results/cmi_synthesis/` | on `project/cita-target-unlabeled-cmi` @ `c889730` |
| **Target-unlabeled active-λ CMI does not beat TTA-Control** | CITA_03 "FAIR FAILURE" | CMI term active (17.8–29.3% of loss) yet adds ~0 over TTA on target & R3 | `docs/CITA_03_ACTIVE_LAMBDA_READOUT.md`, `results/cita/gate_lambda1/` | `project/cita-target-unlabeled-cmi` |

**What is RETAINED (durable, explicitly NOT closed):**
- `cmi_audit` — **RETAINED**: label-conditional domain leakage is measurable, permutation-significant, head-replay-verifiable.
- `cmi_proxy_control_partial` — **RETAINED_PARTIAL**: measured leakage is reducible without task collapse (CIGL ~40–65%; MetaCMI/CITA reduce feature-KL slightly more than their non-CMI baselines).
- `TTA-Control` — a **separate, non-CMI positive** (target-unlabeled adaptation improves target +0.035…+0.093; must never be reported as evidence CMI control works).

The three frozen sentences (CMI_SYNTHESIS_01): *"CMI audit works. CMI proxy control works partially. CMI source-only reliance control fails robustly."*

---

## 1. Route summary table (one line each)

| Route | Regime | L1 measure | L2/L3 proxy/erase | L5 reliance ↓? | L6 target ↑? | task-safe | Decision |
|---|---|---|---|---|---|---|---|
| **CIGL** | source-only DG | ✓ sig | ✓ leakage ↓40–65% | ✗ (+0.025 sig, *wrong way*) | ✗ (+0.001 ns) | ✓ | measurement→control gap (frozen) |
| **FCIGL** | source-only DG | ✓ | alignment ↓ sig (KL traded ↑) | ✗ (+0.001 ns) | ✗ (+0.003 ns) | ✓ | method-level NEGATIVE |
| **dCIGL** | source-only DG | ✓ | obj acts (KL traded ↑) | ✗ (−0.007 ns) | ✗ (+0.003 ns) | ✓ (no flatten) | method-level NEGATIVE (seed0-unstable) |
| **MetaCMI** | source-only episodic | ✓ | featKL ↓ ~2–4% (still sig) | ✗ (−0.001…−0.004 vs MetaCE, within noise) | ✗ (≤0 vs MetaCE) | ✓ | seed0-screening FAIL |
| **CITA_lambda_1** | target-unlabeled | ✓ (active 17.8–29.3%) | n/a | ✗ (≈0 vs TTA) | ✗ (≈0 vs TTA; TTA itself +0.037…+0.093) | ✓ | fair failure — terminal close |
| **CITA_lambda_0.010** | target-unlabeled | ✓ (near-inert) | n/a | ✗ (≈0 vs TTA) | ✗ (≈0 vs TTA) | ✓ | TTA-only pass (inert-λ baseline) |
| **TTA_Control** (non-CMI) | target-unlabeled | — | — | ✗ (R3 not ↓) | ✓ (+0.037…+0.093 all 4) | ✓ | genuine non-CMI positive (walled off) |
| **TOS_mean_scatter** (VD) | source-only frozen | ✓ | dents (TSMNet) / partial (EEGNet) | — | ✗ (ΔbAcc ≈0) | ✓ (is the guard) | diagnostic, not deployable eraser |
| **TOS_LEACE** | source-only frozen | ✓ | linear→chance; nonlinear residual | — | ✗ (≈0; **HARMS −0.15…−0.19 on binary EEGNet**) | ✗ (binary) | best linear eraser, no DG gain |
| **TOS_INLP** | source-only frozen | ✓ | subject→chance | — | ✗ (**task collapse**, −0.06…−0.16) | ✗ (over-erase) | over-erasure negative control |
| **TOS_RLACE** | source-only frozen | ✓ | partial | — | ✗ (≈0; HARMS binary EEGNet) | ✗ (binary) | no DG gain |
| **TOS_random_k** | source-only frozen | control | does NOT remove | — | ✗ (ΔNLL −0.034 = LEACE's, but nothing erased) | n/a | falsifier control |
| **TOS_refusal_gate** | source-only | ✓ | refuse→identity | — | — | ✓ (refuses unsafe) | 0 unsafe-accepts (floor) vs 6; default-on NOT certified |
| **TOS_global_LPC_collapse** | source-only (in-loss) | ✓ | global penalty | — | ✗ (collapse / harm) | ✗ | TSMNet collapses, EEGNet target −0.04 (p≤0.001) |
| **TOS_task_preserving_erasure** | source-only (in-loss) | ✓ | collapse-free penalty | — | ✗ (removes nothing) | ✓ | collapse-free ⇒ subject decode = ERM |
| **TOS_capacity_factorial** | source-only frozen | ✓ | — | — | — | — | removability capacity-mediated, arch-type refuted at high d_z |
| **FBCSP_LGG_branch_ablation** | source-only LOSO | fused-Z only | — | — | — (branch diagnostic) | ✓ | spatial branch = load-bearing |
| **FBCSP_LGG_gate_summary** | source-only LOSO | — | spatial-CMI | — | ✗ (fragile survive, effectively null) | ✓ | not promoted |

Legend: ✓ = holds / present · ✗ = does not hold · "—" = not measured for this route.

---

## 2. CMI-control cluster (source-only DG + target-unlabeled) — the measurement→control gap

### 2.1 CIGL — global label-conditional leakage penalty (static-DGCNN)
- **Setup.** strict source-only LOSO, BNCI2014_001 (2a, 4-cls, 9 folds) + BNCI2015_001 (2-cls, 12 folds), seeds {0,1,2}; backbone `dgcnn_forward_graph_adapter`; representation graph_z + node_z; CIGL λ = {graph 0.010, node 0.010}. 315 runs, 0 NaN, firewall passed all folds.
- **L1 Detectability.** Label-conditional posterior-KL is permutation-significant on **126/126** rows (n_perm=1000, `num_exceed=0/1000`, exact_p=1/1001, BH-FDR q≤0.001). Observed/null KL ratio 6.6–18. CIGL *reduces but never eliminates* — reduced CIGL leakage stays FDR-significant.
- **L2 Reducibility.** CIGL−ERM graph_kl **−0.684 [−0.791, −0.547] sig↓**; node_kl **−0.259 [−0.312, −0.181] sig↓** (fractional: 2a graph 0.574 / 2015 graph 0.767; node 0.20 / 0.30). Graph reduced *more* than node.
- **L4 Task coupling (the mechanism, CIGL_66).** CIGL shrinks the subject subspace but the **residual becomes more task-head-aligned and more concentrated**: `task_head_alignment_k2` ERM→CIGL 2a 0.0044→0.0321 (7×), 2015 0.0497→**0.4396** (9×); top2_energy_fraction 2015 0.67→0.87; effective_rank 2015 11.24→8.22.
  **Correlation directions (pooled, n=126, Spearman, CI excludes 0):**
  - `task_head_alignment_k2` → R3 task_drop_k2 = **+0.338 [+0.168, +0.504]** — POSITIVE (more alignment ⇒ more reliance).
  - `graph_kl` → R3 task_drop_k2 = **−0.342 [−0.507, −0.166]** — NEGATIVE (more measured leakage ⇒ *less* reliance).
  - Per-dataset caveat: on 2a both are significant (align +0.467; graph_kl −0.486); on **2015 neither is significant** (align +0.103 ns; graph_kl −0.196 ns). Magnitudes comparable (±0.34) — the claim is that alignment points in the mechanistically-correct *direction* and the KL proxy points the wrong way, not that alignment dominates.
- **L5 Functional reliance.** CIGL−ERM R3 task_drop k2 = **+0.025 [+0.009, +0.051] sig↑** (removing the subject subspace costs CIGL's task *at least as much as* ERM's — reliance NOT reduced); weakens at k8 (+0.017, per-dataset ns). Head-replay exact: erm 63/63, cigl 63/63 classifier-level (max|Δ|≈3e-8).
- **L6 Target consequence.** CIGL−ERM target_bacc = **+0.001 [−0.010, +0.011] ns**; CIGL non-dominated on the task/leakage Pareto frontier on **both** datasets, all seeds. NLL/ECE/worst-subject: **MISSING** (bAcc-only).
- **Collapse/task-safety.** PASS — target CI includes 0 (not accuracy-for-leakage), 0 NaN/crash, firewall (source-only fit, target eval-only) all folds.
- **Decision.** **measurement→control gap** — CMI is useful & controllable as a *proxy*, but measured-leakage control does not guarantee functional-reliance control; mechanism = CIGL removes task-*orthogonal* leakage while task-*entangled* subject directions persist and concentrate. Frozen CIGL_65.
- **Allowed:** CIGL stably reduces measured leakage; retains task; non-dominated CMI-specific control point; measured-leakage reduction does NOT reduce classifier reliance. **Forbidden:** "better decoder", "shortcut remover", "eliminates leakage", and the wording-guarded "CIGL relies *more* on subject leakage" (correct: the reshaped subspace is *not less* load-bearing).
- **Artifacts:** `results/cigl_r123/final/{bootstrap_ci,multiseed_pareto,r3_reliance,head_replay_check,r1_hardened_nperm1000,gap_correlations,gap_diagnostic_summary,gap_graph_node_mismatch}.csv/.yaml` — all present. SHA **d3593c8**, branch `project/cigl-r123-scaffold`.

### 2.2 FCIGL — functional CMI (task-head alignment penalty)
- **Setup.** Same DGCNN source-only LOSO, seeds {0,1,2}; penalize task-head row-space energy inside the label-conditional subject subspace; η∈{0.01,0.05}; projector k=2 source-train-only. Comparators = frozen CIGL_65.
- **L2/L4.** The alignment scalar reduces reliably: `task_head_alignment_k2` FCIGL−CIGL = **−0.117 [−0.219, −0.010] sig↓** (2015 −0.194 sig). But measured KL was *traded up* to buy it: graph_kl **+0.166 sig↑**, node_kl **+0.157 sig↑** (still below ERM, no rebound).
- **L5.** R3 task_drop k2 FCIGL−CIGL = **+0.001 [−0.014, +0.019] ns** — reliance NOT reduced. (seed0 2015 screen 0.082→0.063 did not replicate.)
- **L6.** target_bacc FCIGL−CIGL = **+0.003 [−0.003, +0.009] ns**.
- **Collapse/task-safety.** PASS — random-subspace task_drop ≈0, head_replay_ok 126/126, no leakage rebound.
- **Decision.** **method-level NEGATIVE** — "alignment-controllable, reliance-neutral." Controlling the alignment *scalar* is not sufficient for reliance control.
- **Allowed:** FCIGL reduces task-head alignment with residual subject subspace while retaining task, but does NOT reduce functional reliance. **Forbidden:** "FCIGL reduces reliance / is a better decoder / proves old CIGL wrong."
- **Artifacts:** `results/cigl_functional/final/{functional_multiseed_metrics,functional_multiseed_r3,functional_multiseed_alignment,functional_vs_oldcigl_bootstrap_ci}.csv` — present. SHA **d02c400**, branch `project/cigl-functional-cmi`.

### 2.3 dCIGL — direct-reliance CMI (training-time counterfactual SymKL)
- **Setup.** Same DGCNN source-only LOSO, seeds {0,1,2}; `CE + λg·graph + λn·node + β·SymKL(logits, logits_removed) + γ·CE(logits_removed,y)`, β0.1 (γ0.5 fixed).
- **L2/L3.** Objective genuinely acts, but measured KL traded up: graph_kl **+0.251 sig↑**, node_kl **+0.169 sig↑** (below ERM).
- **L5.** R3 task_drop k2 dcigl−CIGL = **−0.007 [−0.024, +0.006] ns** — reliance NOT reduced. Crucially the negative is *clean of the flattening confound*: predictions became **more** confident, not flatter (mean_margin **+0.252 sig↑**, prediction_entropy **−0.044 sig↓**). (seed0 2015 screen 0.082→0.050 was inflated by CIGL's high-seed0 baseline.)
- **L6.** target_bacc dcigl−CIGL = **+0.003 [−0.004, +0.010] ns**.
- **Collapse/task-safety.** PASS — random control ≈0; R3 did not fall via logit-flattening.
- **Decision.** **method-level NEGATIVE (unstable seed0)** — establishes the *three-level* measurement→reliance gap: measured leakage / alignment proxy / direct reliance objective all fail. "STOP the CMI method-development line."
- **Allowed:** the direct-reliance objective produced only an unstable seed0 screen; does not reduce reliance at method level; predictions became more confident, not flatter. **Forbidden:** "dcigl reduces reliance / fixes CIGL."
- **Artifacts:** `results/cigl_direct_reliance/final/{direct_reliance_metrics,direct_reliance_r3,direct_reliance_logit_diagnostics,direct_reliance_bootstrap_ci}.csv` — present. Branch `project/cigl-direct-reliance-cmi` (**e619713**); doc CIGL_68.

### 2.4 MetaCMI — source-episodic meta-learned reliance control
- **Setup.** strict source-only source-episodic, 2a+2015, backbones **EEGNetMini + EEGConformerMini** (internal minimal transformer — NOT official braindecode), representation feature_z; `MetaCE + warm·β·SymKL(h(z), h((I−SᵀS)z))`, projector S fit on meta-train SOURCE subjects only, β0.1, k2. **seed0 only** (84 runs).
- **L1.** feature_z label-conditional leakage significant, perm_p<0.05 all folds (naming guard: "graph_z"/graph_kl here = generic feature_z, node_z is a dummy — no structural claim).
- **L2/L3.** featKL Δ vs ERM: MetaCMI −0.031…−0.044, more than MetaCE (as designed) but only ~2–4%; leakage stays significant.
- **L5.** R3 k2 MetaCMI−MetaCE = **−0.003 / −0.001 / −0.004 / −0.002** across the four dataset×backbone cells — 10–20× within the fold-sd (0.05–0.07). MetaCMI−ERM is MIXED (one cell +0.005). *(Provenance: these R3 deltas are from the CIGL_69C readout table; the `metacmi_phase2/` per-fold JSONs store only featKL / target_bacc / source_bacc / head_replay — R3 is not re-derivable from that path alone.)*
- **L6.** target_bacc MetaCMI−MetaCE **≤0 in all 4 cells** (−0.001…−0.006); MetaCMI−ERM positive only 1/4.
- **Collapse/task-safety.** PASS — all 8 stop-conditions clean (exact head-replay max|Δ|≤9e-6, random control ≈0, projector firewall true on 84/84).
- **Decision.** **seed0-screening FAIL** — no CMI-specific signal; "do NOT request seeds 1/2." (Weaker frozen status than FCIGL/dCIGL, which have seeds {0,1,2}.) Backbone context: ConformerMini/Full sit at parity with EEGNetMini regardless of capacity — capacity is not the limiter.
- **Allowed:** source-episodic MetaCMI shows no seed0 CMI-specific signal; the term adds nothing beyond episodic CE. **Forbidden:** "CMI works on Conformer" (Mini ≠ official); any Conformer-family verdict from Mini alone.
- **Artifacts:** `results/cigl/metacmi_phase2/{BNCI2014_001,BNCI2015_001}/` (per-fold JSONs + `*_seed0_metacmi_rows.json`); ERM comparators `results/cigl/metacmi_gate*/`. SHA **8068191** (doc CIGL_69C), branch `project/metacmi-eegnet-conformer`.

### 2.5 CITA_lambda_1 — target-unlabeled offline-transductive, active-λ probe
- **Setup.** offline transductive, 2a+2015, EEGNet + Conformer, feature_z, seed0. Three methods (ERM, TTA-Control, CITA-CMI) all adapt the identical source-ERM M0 (matched attribution). **Firewall: target X allowed at adaptation; target y forbidden for training/adaptation/selection/early-stopping — used only for the final metric.** → `target_labels_used_for_fit = AUDIT_ONLY`.
- **Activation gate (closes the "λ too small" loophole).** cond-domain term = **1.8% / 17.8% / 4.1% / 29.3%** of the adaptation loss across the four cells — **active** (vs ~0.1–0.2%, near-inert, at λ=0.010).
- **L6.** TTA−ERM = **+0.043 / +0.037 / +0.093 / +0.075** (positive all four — the project's first robust generalization gain, non-CMI). **CITA(λ1)−TTA = −0.000 / −0.002 / +0.000 / −0.003** — the active CMI term adds nothing, marginally worse on the most-active cell. NLL/ECE: **MISSING** (bAcc-only).
- **L5.** R3 CITA(λ1)−TTA ≈ 0 everywhere (largest −0.006 on 2015·conformer, within fold-sd).
- **Collapse/task-safety.** PASS — entropy non-degenerate, label-balance KL small; CITA's adaptation barely differs from TTA even at 29% cond-domain loss.
- **Decision.** **"FAIR FAILURE"** — CMI term active, task retained, no collapse, random control valid, but CITA does not beat TTA on target (≈0) nor R3 (≈0). **Terminal: close CMI method development.**
- **Allowed:** CMI is robust as an audit tool; target-unlabeled adaptation (TTA) improves target in all 4 cells; the CMI *control* objective is closed across both regimes. **Forbidden:** attributing the +0.037…+0.093 gain to CMI (it is entirely TTA); any CMI/CITA target or reliance win; further λ/β/k sweeps.
- **Artifacts:** `results/cita/gate_lambda1/{BNCI2014_001,BNCI2015_001}/` (per-fold JSONs + `*_seed0_cita_rows.json`) — present. Branch `project/cita-target-unlabeled-cmi` @ **c889730** (artifacts stamp commit e0f647d, phase CITA_01).

### 2.6 CITA_lambda_0.010 — the pre-registered (near-inert) baseline
- **Setup.** Identical to §2.5 but at the pre-registered λ=0.010; 2a+2015, EEGNet+Conformer, seed0. `target_labels_used_for_fit = AUDIT_ONLY` (same firewall; `adapt()` has no target-label param).
- **Activation.** cond-domain penalty is **near-inert** — ≈0.1–0.2% of the loss (final_cond_domain 0.03–0.22 × 0.010) — so CITA is numerically ~identical to TTA.
- **L6.** TTA−ERM = **+0.046 / +0.035 / +0.093 / +0.077** (positive all four); **CITA−TTA = −0.001 / +0.002 / +0.000 / −0.001** (≈0).
- **L5.** R3 CITA−TTA ≈ 0 in every cell (TTA−ERM R3 is mixed/slightly positive — adaptation *increases* reliance).
- **Decision.** **TTA-only pass** — TTA robustly improves target, CITA-CMI adds nothing; this seed0 verdict is scoped to "at λ=0.010 the CMI term adds nothing," which the §2.5 λ=1.0 active probe then closed decisively.
- **Allowed:** at λ=0.010 the CMI term is near-inert and adds nothing over TTA. **Forbidden:** treating λ=0.010's null as "λ was simply too small" (the λ=1.0 active probe refutes that) or attributing the TTA gain to CMI.
- **Artifacts:** `results/cita/gate/{BNCI2014_001,BNCI2015_001}/` — present. Branch `project/cita-target-unlabeled-cmi` @ c889730 (doc CITA_02).

### 2.7 TTA_Control_non_CMI — the separate, non-CMI positive
- **What it is.** Target-unlabeled adaptation objective `CE(source_replay) + τ·H(p_target) + μ·KL(mean p_target ‖ source_prior)` — a confidence/label-balance adaptation on target X (no CMI term).
- **L6.** TTA−ERM = **+0.037…+0.093** across all four dataset×backbone cells (2a +0.037…+0.046; 2015 +0.077…+0.093), positive at both λ=0.010 and λ=1.0 runs, trading a little source accuracy, **no entropy/label collapse** — the project's **first robust generalization gain**.
- **L5.** TTA−ERM R3 is mixed/slightly positive: TTA improves accuracy **without reducing** subject-subspace reliance (relevant to RQ2 — target benefit is decoupled from reliance reduction).
- **Decision.** a genuine positive, but **non-CMI**; retained as a possible standalone line (stronger objectives, more datasets, seeds {0,1,2}), explicitly framed as non-CMI.
- **Allowed:** target-unlabeled adaptation improves target performance in all four cells. **Forbidden:** reporting it as evidence that CMI control works; "CITA−TTA ≈ 0 at both λ" is the wall.
- **target_labels_used_for_fit = AUDIT_ONLY** (adapts on target X; target y eval-only).
- **Artifacts:** `results/cita/gate/` (λ=0.010 `*_tta_control_*`) + `results/cita/gate_lambda1/` (λ=1.0 `*_tta_control_*`). Branch `project/cita-target-unlabeled-cmi` @ c889730.

---

## 3. TOS erasure cluster (source-only frozen-feature erasability + refusal)

**Shared setup.** Post-hoc erasure on frozen latents, BNCI2014_001 (2a) LOSO, backbones **TSMNet (z=210, SPD)** + **EEGNet (z=16, conv)**, seeds {0,1,2}; multi-dataset extension Lee2019_MI / Cho2017 / HGD / 2b (129 subjects beyond 2a). Erasure fit on source frozen features; deployed to target; **target y used only for the deployment metric** → `target_labels_used_for_fit = NO`. 2a chance: subject 0.125 (8 source subjects), task 0.25 (4-class). Headline (README/CLAIMS_LEDGER C9/C10) = **measurement-to-control gap**; TOS's contribution is reframed as *measurement + certification/refusal + the gap*, explicitly **NOT the eraser** ("the score-Fisher projection is not the contribution").

### 3.1 TOS_mean_scatter (EEG route label `TOS_VD` — the score-Fisher mean/scatter family)
- **L3 erasability (subject decode).** TSMNet 0.997→**lin 0.917 / MLP 0.958** (barely dents; leakage high-dim/redundant). EEGNet 0.819→**lin 0.358 / MLP 0.545** (partial, *selective*: same-k random only 0.734/0.808 → selectivity 0.35–0.55 vs TSMNet 0.04–0.08). Linear ~67% removed on EEGNet with large nonlinear residual (0.545 ≫ chance 0.125).
- **L4.** removed V_D is label-light by construction; task preserved (TSMNet 0.746→0.746 lin; EEGNet 0.639→0.637 lin).
- **L6.** ΔbAcc TSMNet **−0.0003 [−0.0021, +0.0014]**, EEGNet **−0.00045 [−0.0023, +0.0015]** (ΔNLL slightly worse both). Multi-dataset all ≈0. `improves_target=false`.
- **Task-safety.** PASS — this route *is* the guard (score-Fisher label-light selection / task-risk UCB), task preserved on every cell.
- **Decision.** supported diagnostic, **NOT a deployable eraser** — dents (TSMNet) / partially removes (EEGNet) with zero target benefit; LEACE dominates it on subject removal at equal task cost.
- **Note.** The pure first-moment mean-scatter baseline (`mean-scatter-v2`) is **synthetic-only** and no-ops on the covariance-only counterexample (`tests/test_limits.py`) — MISSING as an EEG deletion number.
- **Artifacts:** `tos_cmi/{fisher,subspace,score_fisher}.py`, `tos_cmi/eeg/erasure_baselines.py`, `results/tos_cmi_eeg_frozen/BNCI2014_001_{TSMNet,EEGNet}_LOSO/erasure_report.json`, `.../erasure_target_deploy/erasure_target_deploy_summary.json`. Branch `tos` @ **1c65d79**.

### 3.2 TOS_LEACE (closed-form linear concept erasure)
- **L3.** Drives **linear** subject to chance on both backbones (TSMNet & EEGNet lin → **0.115** = chance) but **nonlinear MLP residual persists** (TSMNet MLP **0.740**, EEGNet MLP **0.393** ≫ chance). Best linear eraser.
- **L4.** 4-class task preserved (TSMNet 0.746→0.743, EEGNet 0.639→0.606). **But on binary Lee/Cho-EEGNet task → chance** (0.79→0.50, 0.77→0.50) — subject/task entangled in the compact 16-d binary latent.
- **L6.** 2a TSMNet ΔbAcc **+0.0007 [−0.0038, +0.0050]**, ΔNLL −0.031 — **but same-k random matches it (−0.034)** → non-specific. (The raw JSON `improves_target` flag reads `true` for TSMNet LEACE/RLACE — but it is NLL-driven and `random_k` trips the same flag *without removing subject*, so it is non-specific regularization, not a bAcc gain; every EEGNet principled eraser and `TOS_VD` on both backbones read `false`.) 2a EEGNet **−0.011 [−0.021, −0.002]** (worsens). **Lee2019 EEGNet −0.185 [−0.220, −0.152], Cho2017 EEGNet −0.150 [−0.185, −0.118] — actively HARMS** (task→chance). TSMNet cells all ≈0.
- **Task-safety.** No guard; passes on 4-class (2a/HGD), **FAILS on binary Lee/Cho-EEGNet**.
- **Decision.** supported — best linear eraser, but no target-bAcc gain in any valid cell (every principled-eraser upper 95% CI < +0.01; C12 CONFIRM 9/9) and harmful without a task-risk certificate.
- **Allowed:** LEACE removes linear subject to chance, leaves nonlinear residual, no target gain, can harm on entangled latents. **Forbidden:** "LEACE removes subject leakage" (residual remains) / "improves DG."
- **Artifacts:** `tos_cmi/eeg/erasure_baselines.py` (`leace_eraser`), `erasure_report.json`, `.../{Lee2019_MI,Cho2017}/erasure_target_deploy_summary.json`, `notes/REAL_EEG_VALIDATION.md`. Branch `tos` @ 1c65d79.

### 3.3 TOS_INLP (iterative nullspace projection)
- **L3.** subject → chance (TSMNet lin 0.141 / MLP 0.464; EEGNet lin & MLP 0.115 = chance).
- **L4.** **OVER-ERASURE — task destroyed**: TSMNet task 0.746→0.550; EEGNet task 0.639→**0.246** (= chance). Not task-selective.
- **L6.** 2a TSMNet ΔbAcc **−0.062 [−0.095, −0.033]**; 2a EEGNet **−0.160 [−0.236, −0.085]** (**tgt & src bAcc → 0.25**). Always collapses source task.
- **Task-safety.** FAIL on every cell.
- **Decision.** supported as a documented over-erasure failure — negative control; excluded from the principled-eraser deployment verdict.
- **Allowed:** INLP drives subject to chance but destroys task; a negative control, not a usable eraser. **Forbidden:** presenting INLP subject removal as success.
- **Artifacts:** `erasure_baselines.py` (`inlp_eraser`), `erasure_report.json`, `erasure_target_deploy_summary.json`. Branch `tos` @ 1c65d79.

### 3.4 TOS_RLACE (relaxed linear adversarial concept erasure)
- **L3.** Partial: TSMNet lin 0.856 / MLP 0.944 (weak on 210-d SPD); EEGNet lin 0.293 / MLP 0.487 (between LEACE and TOS_VD).
- **L4.** 4-class task preserved (TSMNet 0.746→0.739, EEGNet 0.639→0.611); binary Lee/Cho-EEGNet → chance (like LEACE).
- **L6.** 2a TSMNet **−0.0040 [−0.0062, −0.0016]**, 2a EEGNet **−0.012 [−0.024, −0.002]** (both worsen). **Lee2019 EEGNet −0.185, Cho2017 EEGNet −0.150 — HARMS.** TSMNet cells ≈0.
- **Task-safety.** No guard; passes 4-class, fails binary Lee/Cho-EEGNet.
- **Decision.** supported — principled eraser (C12 CONFIRM), no target gain, harmful on entangled binary latents.
- **Allowed:** RLACE partially removes subject, preserves 4-class task, no target gain, harms binary EEGNet. **Forbidden:** "RLACE gives DG benefit."
- **Artifacts:** `erasure_baselines.py` (`rlace_eraser`, self-contained), `erasure_report.json`, `erasure_target_deploy_summary.json`. Branch `tos` @ 1c65d79.

### 3.5 TOS_random_k (random-k subspace removal — the falsifier control)
- **L3.** By design does NOT remove subject (TSMNet 0.997/0.997 unchanged; EEGNet lin 0.734 / MLP 0.808, barely dented at same k) — confirms V_D/LEACE selectivity is genuine.
- **L6 (decisive).** 2a TSMNet ΔbAcc −0.0001, **ΔNLL −0.034 — matches LEACE's −0.031** while `subj_dec_after` = **0.998 (NOT erased)** → the LEACE NLL movement is **non-specific regularization, not domain removal**.
- **Decision.** supported as the falsifier control — closes the measurement-to-control loop.
- **Allowed:** same-k random removal reproduces the erasers' small NLL drop without removing subject → non-specific. **Forbidden:** treating the random-k NLL drop as a method effect.
- **Artifacts:** `erasure_baselines.py` (`random_k`), `erasure_report.json`, `erasure_target_deploy_summary.json`. Branch `tos` @ 1c65d79.

### 3.6 TOS_refusal_gate (source-only certify/refuse → identity when no safe subspace)
- **(A) Synthetic certification** (`notes/PHASE131_CERTIFICATION.md`). With the power floor ON, over the pre-registered explaining-away grid: **UNSAFE_ACCEPT = 0** (SAFE_REJECT 36 / UNSAFE_REJECT 12 / BAYES_AMBIGUOUS 2) **vs 6 unsafe-accepts without the floor**. But **default-on selective deletion is NOT certified** at the pre-registered operating point (independent-seed cert straddles the LCB≥0.80 bar → `power_ok=False`; oracle density-ratio robustly clears → residual gap is *estimator inefficiency*, not sample size). Defaults frozen `task_protect=False`, `task_power_floor=False`; the only deleting config is exact-fingerprint `certified_synthetic_experimental`, else identity.
- **(B) EEG frozen-pilot gate** (`notes/PHASE2_EEG_FROZEN_PILOT.md`, 2a TSMNet LOSO). ACCEPT **5/9** vs TASK_RISK_UCB refuse **4/9** (seed0; 3-seed range 5–7/9) — but the accepts are **VACUOUS**: `certified_accept=False` and deleting the accepted V_D does **not** remove domain (subject 1.00→0.96 ≈ random-k). At high λ (collapsed model) **DOMAIN_GATE_CLOSED 9/9** (gate correctly refuses). Under LOSO domain==subject, group-aware folds can't cover all subjects → FOLD_COVERAGE_FAILURE.
- **Safety win.** Refusal-when-entangled is exactly what protects against the LEACE/RLACE binary-EEGNet harm (−0.15…−0.19): erasing without a task-risk certificate can hurt, and the gate refuses to.
- **Decision.** a measurement-to-control certification/refusal framework — refuses where geometry-only / weak learned gates unsafe-accept (0 vs 6); intentionally conservative; **default-on NOT certified**.
- **Allowed:** the gate refuses unsafe deletion (0 unsafe-accepts under the floor vs 6 without); conservative at moderate n. **Forbidden:** "certified task-protected default-on deleter" / "the gate enables safe deletion on EEG."
- **Artifacts:** `tos_cmi/score_fisher.py`, `tos_cmi/eval/{power_certificate,bayes_oracle}.py`, `tos_cmi/subspace.py`, `notes/{PHASE131_CERTIFICATION,PHASE13_DIAGNOSIS,PHASE2_EEG_FROZEN_PILOT}.md`. Branch `tos` @ 1c65d79.

### 3.7 TOS_global_LPC_collapse — why global erasure *in the loss* is unsafe (CLAIMS C5, Fig 5D)
- **TSMNet (SPD).** Training with a global `I(Z;D|Y)` penalty (LPC) **collapses the representation**: feature-norm 1.09→0.00, top-1 singular value → 0.001, penalty → 0, CE → ln(4) (chance). It is objective-scaling collapse, **not** gradient explosion (adversarially verified, `wf_c2880caf`).
- **EEGNet (conv).** raw-LPC sweep (λ∈{0,0.3,1,3}) does **not** collapse but **harms task**: subject 0.89→0.17, mean LOSO target **0.43→0.39 (paired-t p≤0.001 at λ≥1)**, corr(leak,target) −0.14.
- **Decision.** global conditional-invariance-in-the-loss is either collapse-prone (TSMNet) or target-harming (EEGNet) — motivating TOS's *selective + refuse* design; it is a boundary result, not a method.
- **Allowed:** global LPC collapses TSMNet / harms EEGNet target. **Forbidden:** any "global CMI-in-loss improves DG."
- **Artifacts:** `results/tos_cmi_eeg_frozen/lpc_collapse_curves/{TSMNet,EEGNet}/`. Branch `tos` @ 1c65d79.

### 3.8 TOS_task_preserving_erasure — collapse is fixable, but the fix removes nothing (CLAIMS C6)
- Collapse-free variants (`lpc_warm_ramp`, `lpc_scale_invariant`, flag-gated, default-off) **avoid** the TSMNet collapse (warm_ramp λ=1&3 collapse 0/9), **but** the collapse-free model's subject decode ≈ **0.997 = ERM** — i.e. once you prevent collapse, the penalty removes *no* subject information. The apparent "removal" in the collapsing runs was the collapse itself.
- **Decision.** a task-preserving conditional-invariance penalty removes nothing measurable — the L2/L3 "reduction" of the collapsing variant was an artifact of representation collapse.
- **Allowed:** collapse-free conditional-invariance training leaves subject decodability at ERM. **Forbidden:** citing the collapsing-run leakage drop as removal.
- **Artifacts:** `results/tos_cmi_eeg_frozen/lpc_collapse_curves/TSMNet/variant_compare.json`. Branch `tos` @ 1c65d79.

### 3.9 TOS_capacity_factorial — is removability architecture or capacity? (Track C, 3-seed)
- The linear-erasure **nonlinear residual rises monotonically with latent dim** in BOTH architectures: OLS log(d_z) = **+0.089 [0.086, 0.092]**. Matched-dimension comparison removes ~68% of the raw TSMNet(0.74)-vs-EEGNet(0.39) residual gap → **capacity is the dominant axis**; but a residual architecture×dim **interaction +0.058 [0.051, 0.063]** persists and at matched d_z=210 the SPD backbone retains **+0.11 [+0.094, +0.125]** more recoverable subject id.
- **Decision.** "removability is architecture-*type* driven" is **REFUTED at high d_z** — it is *largely capacity-mediated* with a residual SPD×dim interaction. (Caveat: single dataset 2a, LDA cap 8 subj.)
- **Allowed:** low-rank subject removability is largely capacity-mediated, with a residual SPD-geometry interaction at high d_z. **Forbidden:** "SPD architecture per se makes subject leakage more removable" (matched-dim refutes the pure-type claim).
- **Artifacts:** `results/tos_cmi_eeg_frozen/factorial/factorial_multiseed.json`. Branch `tos` @ 1c65d79.

---

## 4. FBCSP-LGG branch cluster (branch-locality; source-only LOSO)

**Backbone.** `FBCSPLGGGraph` = filter-bank CSP + learnable-graph-Gaussian, 3-way `gate3` fusion over **graph / temporal / spatial** branches. **There is no separate "node branch"** (`permute_nodes` is a node-permutation *null*, not a branch-zeroing) — correcting the project brief's graph/node/edge framing. Datasets 2a (9 folds) + 2015 (12 folds), seeds {0,1,2}, full-LOSO ERM.

### 4.1 FBCSP_LGG_branch_ablation — which branch is load-bearing
- **Spatial branch is load-bearing.** `zero_spatial` is the biggest ablation drop on both datasets: **2a 0.349→0.275 (−7.4pp)**, **2015 0.608→0.520 (−8.8pp)** (toward chance); gate weight `gate_spatial` highest on both (0.489 / 0.572). Graph branch is **neutral/slightly harmful** (`zero_graph` 2a 0.349→0.366, +1.7pp; gate_graph starved 0.239–0.279); temporal near-neutral. `permute_nodes` near-neutral (2a −0.2pp).
- **Same-fold subj1 comparison.** FBCSP-LGG **0.474** ≈ CSP cross-subj **0.483** (spatial branch closes the CSP gap), vs FBLGG **0.306** (+16.8pp) and DGCNN 0.403. On CSP-decodable subjects {1,3,8,9} FBCSP mean 0.430; CSP-hard {2,4,5,6,7} 0.284 (≈chance) drags the full-LOSO 2a mean (0.349) down.
- **Ladder.** L4/branch-load evidence (ablation + gate weights). L1 leakage here is **fused-Z only** (in-training GLS posterior-KL, e.g. 2015 leakage_kl 0.645) — **no per-branch leakage probe**.
- **Decision.** spatial branch is load-bearing, robust across seeds/datasets.
- **Allowed:** the spatial (CSP-style) branch is load-bearing; on the decodable fold FBCSP-LGG closes the CSP gap FBLGG could not. **Forbidden:** any per-branch *leakage* claim (no per-branch probe exists).
- **Artifacts:** `results/fbcsp_lgg_f0_full_s012/{FBCSP_F0_SUMMARY.md,FBCSP_F0_AGGREGATE.csv,FBCSP_F0_SEED_TABLE.csv}`. Branch `project/fbcsp-lgg-spatial-cmi-fusion` @ **39c245a** (internal provenance `project/fbcsp-lgg-dualcmi-scaffold @ add0a76`).

### 4.2 FBCSP_LGG_gate_summary — spatial-CMI P6 go/no-go
- **Question.** does aligning the CMI penalty to the load-bearing spatial branch (new `fbdualpc` spatial-CMI) + un-starving the graph branch (fusion_floor) beat FBCSP-LGG ERM?
- **seed0 screen.** NULL/NEGATIVE — full-mean "pass" is mechanism-falsified; gains live in the chance band and *reverse on every CSP-decodable subject* (spatialCMI Δ=−0.016, all-CMI Δ=−0.033 on {1,3,8,9}); all deltas within single-seed noise.
- **3-seed confirmation** (PI override, one least-indefensible config). Pre-registered primary = 2a CSP-decodable Δ: seeds −0.016 / −0.007 / **+0.048** → **mean +0.0082 (sd 0.0285)**; 2015 full **+0.0057 (sd 0.0064)**. **SURVIVE by the letter** (both ≥0) **but fragile**: SD 3.5× the mean, 2/3 seeds negative on the decisive subset, and the +0.048 on seed2 is a **collapsed-ERM-baseline artifact** (seed2 ERM s8/s9 cratered to ≈chance) — spatialCMI's *absolute* decodable accuracy never beats ERM's best (0.4683).
- **Decision (PI, 2026-07-05).** "fragile survive; not promoted; effectively null for resource allocation; no further seed/λ expansion."
- **Allowed:** the spatial-CMI regularizer honored the pre-registered SURVIVE rule. **Forbidden:** that `fbdualpc` spatial-CMI is a working control/method (effectively null, baseline-collapse-driven, 2/3 seeds negative).
- **Artifacts:** `docs/CIGL_50_FBCSP_LGG_SPATIAL_CMI_ROADMAP.md`, `docs/CIGL_50_P6_SCREENING_REPORT.md`, `results/p6_fbdualpc_seeds12/{SEEDS12_REPORT.md,SEEDS12_AGG.json}`. Branch `project/fbcsp-lgg-spatial-cmi-fusion` @ 39c245a.

---

## 5. Repository-wide corroborating evidence (branch-locality · observability · deployment · information-contract)

These clusters do not add CMI-control routes; they corroborate the FSR thesis from other angles and set boundary conditions. Each carries a `status_kind` so protocol/design/simulator work is never read as a scientific-efficacy result (red-flags §9–§11 of `FSR_00`).

| Route | branch @ sha | status_kind | one-line decision |
|---|---|---|---|
| FBCSP_LGG_bottleneck_analysis | fblgg-2a-bottleneck @ 787fcc7 | SCIENTIFIC_RESULT | graph/temporal branches lack CSP-style spatial-spectral features (subj1 CSP 0.483 ≫ FBLGG 0.306) |
| FBCSP_LGG_graph_starvation | fblgg-2a-bottleneck @ 787fcc7 | SCIENTIFIC_RESULT | no transferable 4-class graph signal (2a ablations ≈0); graph load-bearing only on binary 2015 |
| CIGL_35_blueprint | fblgg-2a-bottleneck @ 787fcc7 | BACKGROUND | claim-boundary contract for the DGCNN CIGL leakage line (~40–65% reduction, "partial not elimination") |
| P6_spatial_CMI_scaffold | fbcsp-lgg-dualcmi-scaffold @ eb47bd0 | DESIGN_ONLY | roadmap only, F1/CMI/GPU frozen — **NOT a real-EEG CMI result** (red-flag 11) |
| OACI_selection_leakage_not_target | oaci @ afc8f50 | CLOSED_NEGATIVE | selection leakage ↓54/54 but does not transfer to audit leakage or target |
| OACI_source_audit_oracle_failure | oaci @ afc8f50 | CLOSED_NEGATIVE | even a source-audit **oracle** selector finds no reproducible target gain |
| OACI_multivariate_weak_identifiability | oaci @ afc8f50 | CLOSED_NEGATIVE | no scalar source signal; diagnostic LOTO probe weakly beats perm (AUC 0.602, p=0.008) |
| OACI_endpoint_estimability_limit | oaci @ afc8f50 | CLOSED_NEGATIVE | abstention precedes signal loss; worst-domain bAcc→NaN under cell deletion (C18) |
| ACAR_paired_action_risk_design | acar @ d287635 | DESIGN_ONLY | leak-proof action-conditional ΔR_a(B) estimand + conformal router (design, not efficacy) |
| ACAR_v5_protocol_substrate_success | acar @ d287635 | PROTOCOL_ONLY | hash-bound Stage-1B substrate + Stage-2B authorization (engineering success, not efficacy) |
| ACAR_stage2b_dev_stop | acar @ d287635 | CLOSED_NEGATIVE | Stage-2B RAN → pre-registered **DEV_STOP**, 0/22 eligible, router REFUTED on DEV |
| CSC_Z_only_unidentifiable | csc @ 72085b7 | CLOSED_NEGATIVE | Z-only concept shift provably unidentifiable (Prop 1); frozen detector FAILED both endpoints |
| CSC_dual_witness_candidate | csc @ 72085b7 | PENDING | B6+B7 dual-witness redesign; B7.1 confirmatory is a committed-but-UNRUN protocol |
| CSC_information_contract_boundary | csc @ 72085b7 | DESIGN_ONLY | unlabeled marginal insufficient; fitted-null under-dispersed ~7–10× — the publishable boundary |
| LPC_CMI_legacy_boundary | exp/lpc-cmi @ 050d3a4 | CLOSED_NEGATIVE | LPC survives only as measurement; deployment/calibration/accuracy DROPPED; TUAB retracted |
| PriorDecoupled_four_branch_protocol | exp/h2cmi-wave0 @ 60db118 | BACKGROUND | exact geometry-vs-decision-prior decomposition; harm is the prior, not geometry |
| PriorDecoupled_geometry_vs_prevalence | exp/h2cmi-wave0 @ 60db118 | BACKGROUND | prevalence is a poor bAcc decision prior even when known; metric-mismatch dominates |

### 5A. FBCSP-LGG bottleneck & blueprint (branch-locality precursor)

**5.1 FBCSP_LGG_bottleneck_analysis** — *SCIENTIFIC_RESULT; `target_labels_used_for_fit = NO`.* CPU-only diagnostic on the exact F0 folds. On the decodable BNCI2014 (4-class) fold, classical CSP+LDA transfers cross-subject far better than the graph backbones: **subj1 CSP 0.483 vs DGCNN 0.403 vs FBLGG 0.306** (within-subject CSP ceiling 0.717); on binary 2015 CSP 0.629 ≈ FBLGG 0.627 ≫ DGCNN 0.575. BNCI2014 **FAILS** the FBLGG ERM gate (0.296 < DGCNN 0.342 ≈ chance), BNCI2015 **PASSES** (+5.2pp). Diagnosis: the graph/temporal branches omit CSP-style discriminative spatial-spectral filtering — a *feature-extraction bottleneck*, not a data/split/early-stop/grouping artifact (all ruled out; failure diffuse across all 4 classes, no single-class collapse). **Allowed:** the FBLGG backbone fails 4-class cross-subject MI because it does not extract transferable spatial-spectral features; a CSP-style spatial branch closes the gap on CSP-decodable subjects. **Forbidden:** any SOTA/general-4-class-solution claim (full-LOSO 2a mean only 0.349, dragged by ~5/9 CSP-hard subjects); any CMI/leakage claim (none run here). **Artifacts:** `docs/CIGL_48_BNCI2014_BOTTLENECK_ANALYSIS.md`, `results/fblgg_f0/{F0_CSP_BASELINE.json,F0_SUMMARY.md,F0_DIAGNOSTICS.md}` on `project/fblgg-2a-bottleneck-analysis` @ 787fcc7.

**5.2 FBCSP_LGG_graph_starvation** — *SCIENTIFIC_RESULT; `NO`.* On 4-class 2a the graph branch carries **no transferable signal** (ablation deltas zero_graph +0.023, zero_temporal +0.012, permute_nodes +0.020 — all ≈0); on binary 2015 it is clearly load-bearing (zero_graph +0.085, permute_nodes +0.114). After the P5 CSP-style spatial branch is added, the 3-way fusion gate collapses toward spatial (gate_spatial 0.489/0.572) and **starves** the graph branch (neutral/slightly harmful on both; 2015 regresses −1.9pp). **Forbidden:** reading the near-zero 2a ablations as robustness (they mean there is little signal to remove); claiming the graph branch adds 4-class value or that the fusion is balanced. **Artifacts:** `CIGL_48 §P4-D`, `results/fblgg_f0/F0_DIAGNOSTICS.md` @ 787fcc7; FBCSP gate table on `project/fbcsp-lgg-dualcmi-scaffold` @ eb47bd0.

**5.3 CIGL_35_blueprint** — *BACKGROUND; `NO`.* The claim-boundary contract for the **separate DGCNN CIGL leakage-audit line** (not the FBCSP accuracy track). Underlying measurement = posterior-KL plug-in proxy vs retrained within-label permutation null (graph KL ~8× / node ~15× perm mean). Fixed λ_graph=λ_node=0.010 reduces measured leakage **~40–65%** (2a folds −35..−58%, 2015 −43..−77%) without harming source-task accuracy, but leakage still clears the null in every fold (dented, not erased) — and CIGL_65 (R2 gate) shows this does not reduce R3 reliance (the same measurement→control gap). **Allowed (verbatim-safe):** "a fixed graph/node CMI regularizer partially and reproducibly reduces leakage without harming source-task accuracy… partial (~40–65%), not elimination; posterior-KL proxy, not unbiased CMI." **Forbidden:** SOTA/leaderboard; "removes/eliminates"; "unbiased CMI"; edge-CMI; "λ-robust". **Artifacts:** `docs/{CIGL_35_PAPER_BLUEPRINT,CIGL_32_METHOD_FRAMING_AND_CLAIMS,CIGL_33_EVIDENCE_INDEX}.md` @ 787fcc7.

**5.4 P6_spatial_CMI_scaffold** — *DESIGN_ONLY; `NO`.* The FBCSP-LGG-DualCMI / spatial-CMI is a **non-GPU roadmap scaffold** (CIGL_49): F1/graphcmi/graphdualpc/λ-sweep/GPU all frozen, never run. **Nothing was measured.** **Forbidden (red-flag 11):** any FBCSP-LGG / spatial-CMI leakage-audit or leakage-reduction *result* — it is a scaffold. **Artifacts:** `docs/CIGL_49_FBCSP_LGG_DUALCMI_ROADMAP.md` on `project/fbcsp-lgg-dualcmi-scaffold` @ eb47bd0.

### 5B. OACI — source-side observability failure (Overlap-Aware Risk-Feasible Conditional Invariance)

Strict source-only DG selector on BNCI2014_001 real-EEG LOSO; the control line is a **closed negative** — support-aware leakage survives only as a measurement/falsification instrument.

**5.5 OACI_selection_leakage_not_target** — *CLOSED_NEGATIVE; `NO`.* OACI reduces selection-time extractable leakage on **54/54** folds (Δ mean −0.3261), **but** this does not transfer: held-out audit leakage Δ mean **+0.0076** (reduced only 25/54), corr(Δselection, Δaudit) = +0.004/+0.091 (≈0 → selection optimism / criterion overfit); target worst-domain bAcc Δ mean **−0.0024** (harmed 28/54), and corr(Δaudit, Δtarget worst bAcc) = −0.064/−0.222 (near-zero/wrong-sign, n=54). **Allowed:** OACI provably drives down *measured selection-time* leakage on 100% of folds. **Forbidden:** any claim it reduces true/held-out leakage or that leakage reduction improves target accuracy/calibration (orthogonal, sometimes wrong-sign). **Artifacts:** `oaci/reports/C10_OACI_FAILURE_DIAGNOSTICS.{md,json}` @ afc8f50.

**5.6 OACI_source_audit_oracle_failure** — *CLOSED_NEGATIVE; `NO`.* Epoch-level counterfactual selector replay is exact (216/216, argmax flips 0, max|Δlogit|=1.8e-15). All six selectors return `stop_no_reproducible_gain` — including **S5, a source-audit oracle** (reads the held-out source_audit split, chooses ERM 3/54, `oracle_reproducible=False`). **Allowed:** no source-selectable (even oracle-selectable) checkpoint reproducibly improves worst-target DG — the negative is oracle-robust. **Forbidden:** claiming a better source-only selector/tuning could rescue OACI. **Artifacts:** same C10 + `C8_BNCI001_LOSO_SEEDS012_K1K2.md` @ afc8f50.

**5.7 OACI_multivariate_weak_identifiability** — *CLOSED_NEGATIVE; `target_labels_used_for_fit = AUDIT_ONLY`.* No strong scalar source signal (0/12 strong-accuracy; oracle scalar within-fold ρ(target bAcc)=+0.12), but a **diagnostic-only** multivariate leave-one-target-out competence probe weakly beats permutation: **LOTO AUC 0.602 vs perm mean 0.537, p=0.0083** (n=621, 12 features). The probe trains on post-hoc target labels purely to test information content — non-deployable. **Allowed:** source observables carry weak, permutation-significant multivariate information about target competence (motivating a future low-freedom probe). **Forbidden:** claiming a deployable source-only selector exists. **Artifacts:** `oaci/reports/C17_SOURCE_SIGNAL_IDENTIFIABILITY.{md,json}`, `C17_IDENTIFIABILITY_CASE_TAXONOMY.md` @ afc8f50.

**5.8 OACI_endpoint_estimability_limit** — *CLOSED_NEGATIVE; `NO`.* Controlled support-mismatch stress (C18): the weak source signal **survives cell-present stress** but collapses only under cell *deletion* via worst-domain bAcc→NaN — an estimator-level (endpoint non-estimability) limit, not signal loss; leakage estimability precedes abstention. **Allowed:** the abstention/estimability boundary is a mechanism finding. **Forbidden:** reading the NaN-collapse as evidence the signal never existed. **Artifacts:** `oaci/reports/C18_CONTROLLED_SUPPORT_MISMATCH_STRESS.{md,json}` @ afc8f50 (exact stress numbers in the C18 report).

### 5C. ACAR — action-conditional deployment successor (Direction 2)

Leak-proof successor to the closed A0 gate line. Estimand = paired incremental risk ΔR_a(B)=R_B(f_a)−R_B(f_0).

**5.9 ACAR_paired_action_risk_design** — *DESIGN_ONLY; `NO`.* A fully-implemented, leak-audited estimand (does acting beat not-acting on this batch) from 7 label-free pre→post features, routed by a subject-clustered conformal upper bound, with 8 hard leakage guards (label-permutation bit-identity, whole-batch aggregation, no class-conditional deletion, serialized source state); distinct from shift-detection / absolute-accuracy by construction. **Forbidden:** claiming the *design* predicts/controls negative transfer or closes the measurement→control gap (proven by neither the design nor the downstream run). **Artifacts:** `acar/{README.md,features.py,risk.py,conformal.py}`, `notes/ACAR_FROZEN{,_v2}.md`, `acar/tests/` @ d287635.

**5.10 ACAR_v5_protocol_substrate_success** — *PROTOCOL_ONLY; `AUDIT_ONLY`.* v5 built a hash-bound external-compatible Stage-1B DEV substrate (**30/30 refs admitted**, registry_sha256 recomputed & matched, SLURM 881227 rc=0) plus a fail-closed Stage-2B authorization gate (binds protocol tag + target-SHA + 10 selection refs + 22 frozen candidates + 3 forbid flags). This is **engineering/protocol recovery success only** (fixes v3 coverage collapse, v4 substrate incompatibility) — explicitly NOT efficacy. **Forbidden:** reading substrate/authorization success as method efficacy; claiming external/held-out/ASZED validation (lockbox sealed). **Artifacts:** `notes/ACAR_V5_STAGE1B_REALRUN_RESULT_*.md`, `acar/v5/{stage2b_authorization,protocol,stage2_gates}.py` @ d287635 (provenance notes + code git-tracked; package bytes live on the cluster, not committed).

**5.11 ACAR_stage2b_dev_stop** — *CLOSED_NEGATIVE; `AUDIT_ONLY`.* **Correction to the PM's "pending" assumption:** a real Stage-2B DEV selection RAN at this SHA (SLURM 885395, rc=0, 8.18h) and emitted a **pre-registered DEV_STOP** — **0/22 candidates eligible**, `selected_candidate_id=null`, router **REFUTED on DEV**. Coverage is fine (G1 PD 13/22, SCZ 19/22 — candidates do adapt), so the failure is harm-control, not starvation: cert_pass 0/22 both diseases; G4 harm-among-adapted UCB 0.61–0.87 ≫ 0.30 on all 42 cells; EVAL red −12.12…+0.01; the no-adapt identity/source-state LDA f_0 dominates. **Forbidden:** writing any gate as passed, any candidate selected, any efficacy achieved, or Stage-2B as pending/incomplete — it ran to a clean rc=0 DEV_STOP. **Artifacts:** `notes/ACAR_V5_STAGE2B_REAL_SELECTION_RESULT_*.md`, `notes/ACAR_V5_STAGE2B_CLOSEOUT.md`, `acar/v5/stage2_gates.py` @ d287635 (outcome committed as verified-provenance notes; the raw SLURM 885395 report lives on the cluster — no machine-readable summary JSON is committed).

### 5D. CSC — partial-identification concept-shift certificates (information-contract boundary)

*Provenance flag: `csc` local `72085b7` is 1 commit ahead of `origin/csc` `7f64a49` (the unpushed B7.1 protocol commit). All "real EEG" here = Lee2019 SM16 **semi-synthetic** — not clinical.*

**5.12 CSC_Z_only_unidentifiable** — *CLOSED_NEGATIVE; `NO`.* Z-only concept shift is provably **unidentifiable** (Proposition 1: for any target marginal Q_Z there exist joints with identical Q_Z but different P(Y|Z)) → a three-state abstention certificate (COVARIATE_COMPATIBLE / CONCEPT_SUSPECT / UNIDENTIFIABLE). The frozen synthetic confirmatory (dee8958) **FAILED both endpoints**: false-certification forbidden 1/65, CP-UB 0.0709 > α=0.05; power fired 28/65, conditional 0.431 < 0.50 bar (vs dev 0.83). Direction A is a frozen NEGATIVE — only the impossibility theorem + abstention boundary survive. **Forbidden:** claiming the Z-only certificate controls false-certification at α on unseen shifts or has power on real concept shift; calling it "precise CMI" or a "safety gate". **Artifacts:** `csc/results/confirmatory.json`, `csc/THEORY.md §1`, `csc/PREREGISTRATION.md` @ 72085b7.

**5.13 CSC_dual_witness_candidate** — *PENDING; `NO`.* B6 condition-randomization (randomize the covariate C|Z,S, not the label) + B7 dual-witness (old_B3 ∧ B6_plain). B6 fixes the strong-covariate false-confirm **mechanism** (18/28→0/0) but opens a prior-shift taxonomy gap (NULL_label 0→25/50); B7.0 passes a **development-only** screen (4 residual dual false-confirms) and **B7.1 confirmatory is a committed-but-UNRUN protocol** (the unpushed commit; no result rows). Foundation B3 passed synthetic confirmatory (0595f64) but FAILED real-feature v2 (NULL_cov 15/100). **Forbidden:** claiming a validated concept certifier, a universal type-I guarantee, deployability, or any B7.1 outcome (no results exist). **Artifacts:** `csc/results/b7_stage0_dual_witness/`, `csc/results/b7_stage1_full_replay/b7_stage1_protocol.json` (protocol only), `csc/results/b6_condition_randomization/` @ 72085b7.

**5.14 CSC_information_contract_boundary** — *DESIGN_ONLY; `NO`.* The surviving positive: the unlabeled marginal is provably insufficient to separate concept vs covariate shift, and empirically the fitted-h0 null is **under-dispersed + mis-centered ~7–10×** under a strong session covariate — an oracle-confirmed *null* defect (the statistic itself is a typical draw from the true null). No fix repairs it: B4 estimable-null CLOSED, B5.0 random & B5.1 SSL features leave it intact, and router R1 only **masks** it in the weak-covariate regime (session_auc≈0.52, ~17% power) while under strong covariate it is a **SOUND FAIL** (auc0.81→7.3% breach, auc0.94→26% breach). **Allowed:** unlabeled-marginal insufficiency is proven and empirically demonstrated; feature richness is not the lever — this partial-identification boundary is the publishable result. **Forbidden:** claiming any working unlabeled-marginal concept-shift controller, or that features/router recover control; pooling NULL kinds; any clinical claim. **Artifacts:** `csc/THEORY.md §1`, `csc/results/router_stage1_validation/{scaleup/README.md,README.md}`, `csc/results/b5_features/{b5_0_random_encoder,b5_1_ssl_encoder}/` @ 72085b7.

### 5E. LPC-CMI legacy boundary

**5.15 LPC_CMI_legacy_boundary** — *CLOSED_NEGATIVE; `target_labels_used_for_fit = YES_FORBIDDEN` (scoped to one retracted sub-line).* Boundary check only — **do not restart**. The LPC-CMI / CITA / source-free CMI-safety-gate line is closed: **only the measurement survives** (Claim 1 — LPC reduces *extractable conditional domain information*, multi-probe, perm-null≈0, beats CDANN; binding rename: not "precise CMI"/"I(Z;D|Y)"). Everything else dropped: deployment regularizer DROPPED (`DROP_LPC_COLLAPSE` at every λ — leakage drop entangled with representation collapse); harm-gate Claims 3/4/6 → DIAGNOSTIC_ONLY (density/CMI anti-aligned with harm; A0-PILOT retained NLL worse than random); accuracy Claim 7 → CITA-no-LPC ≡ plain matched-CORAL (|Δ|=0), ≈ SPDIM (58.4 vs 58.6); calibration Claim 9 → temperature side-effect (a single oracle-T on ERM beats LPC NLL on **123/130**); **batch-rollback RETRACTED** as a target-label-leakage artifact (g_unc ρ→−0.40 when scored label-blind — this is the `YES_FORBIDDEN` scope); **TUAB retracted** as a clean lockbox (13 result files predate the lockbox doc). **Allowed:** "LPC reduces extractable conditional domain information" (measurement) and "source-free adaptation diagnostics are not deployment controllers" (the measurement→control gap). **Forbidden:** "precise CMI"; "safety gate"; LPC as deployment regularizer/calibration; CITA as a distinct positive; batch-rollback via uncertainty; TUAB as a lockbox. **Artifacts:** `CLOSEOUT.md`, `notes/EVIDENCE_LEDGER.md`, `results/p15_audit_deployment/P1.5_CLOSED.json`, `results/calibration_deconfound/summary.json`, `results/a0_prime_r/*/a0primer_summary.json` on `exp/lpc-cmi` @ 050d3a4.

### 5F. Prior-decoupled TTA / H²-CMI — **BACKGROUND only** (NOT FSR leakage/reliance evidence)

Supports the *measurement-then-decompose* philosophy behind the measurement→control gap; must never be entered as leakage `I(Z;D|Y)` or reliance evidence. No h²cmi manuscript exists yet (staged inserts only).

**5.16 PriorDecoupled_four_branch_protocol** — *BACKGROUND; `NO`.* An exact algebraic decomposition {I,T_J}×{Unif,π_J} splitting a single unlabeled joint-TTA delta into geometry (G) + fit-prior (P) + interaction (per-unit residual ≤1.85e-17). On W2 sleep (75 subj) the harm is almost entirely the decision prior: **P = −0.1439 [−0.159, −0.128]** (dominates) while geometry **G = −0.0201 [−0.041, +0.001]** is NS; on W1 motor-imagery (115 LOSO) it inverts (G = +0.0604 sig, P NS) → the joint effect is prevalence-dependent. **Forbidden:** entering as leakage/reliance; claiming a working unlabeled harm-controller. **Artifacts:** `h2cmi/{run_w1_mi,run_w2_sleep}.py`, `h2cmi/results/{W1_W2_RESULTS.md,wave0_w2.report.json}` in `CMI_AAAI_qxu` (`exp/h2cmi-wave0-mechanism` @ 60db118).

**5.17 PriorDecoupled_geometry_vs_prevalence** — *BACKGROUND; `AUDIT_ONLY`.* A three-part split of the decision-prior harm (metric-prior mismatch + adapt→eval transfer + fit-prior deviation): metric-prior mismatch **−0.16** dominates and is protocol-invariant, transfer is NS, and estimation deviation partially *offsets* the harm — i.e. even the oracle evaluation prevalence is a poor balanced-accuracy decision prior. `ρ_E`/`ρ_A` are oracle diagnostics (eval labels), non-deployable. **Forbidden:** presenting as leakage/reliance; claiming a deployable prevalence controller. **Artifacts:** `h2cmi/run_prior_decomp.py`, `h2cmi/results/{W0.3_RESULTS.md,wave0_priordecomp.report.json}` @ 60db118.

---

## 6. Preliminary cross-route reads (hypotheses for Phase 2 — NOT yet claims)

These are pattern observations to be tested statistically in Phase 2 (CPU-only). They are **not** endorsed claims yet.

- **RQ1 (does measured leakage predict reliance?)** — CIGL_66 says *no, wrong sign*: pooled `graph_kl → R3` Spearman **−0.34** (excludes 0). Leakage magnitude is anti-correlated with functional reliance. (Caveat: significant on 2a only; 2015 ns.)
- **RQ2 (does erasure strength predict target benefit?)** — *no*: LEACE drives linear subject to chance yet target ΔbAcc ≈0 (2a) or **−0.15…−0.19** (binary EEGNet); every principled eraser's upper 95% CI < +0.01 (C12 9/9). Erasure strength and target benefit are decoupled.
- **RQ3 (does task-head alignment beat leakage as a reliance predictor?)** — *directionally yes*: `align_k2 → R3` Spearman **+0.34** (correct sign) vs `graph_kl → R3` **−0.34** (wrong sign). Comparable magnitude; alignment is in the mechanistically-correct direction. Not yet a validated estimator (2015 ns).
- **RQ4 (does branch-locality change the meaning of leakage?)** — the spatial branch is load-bearing while graph/temporal are near-neutral/starved (§4.1, §5.2), and the bottleneck analysis (§5.1) independently shows the graph/temporal branches carry *no* transferable 4-class signal (ablations ≈0) — so leakage measured in graph/node branches likely indexes *non-load-bearing* structure. **BLOCKED on a missing metric**: no frozen per-branch embeddings or per-branch leakage probe exist (see §7).
- **Cross-cutting corroboration (observability / deployment / information-contract).** The measurement≠reliance/consequence pattern recurs beyond the CMI and TOS lines: **OACI** (§5.5–5.8) — reducing *selection-time* leakage on 54/54 folds does not transfer to audit leakage or target, and even a source-audit *oracle* cannot find a reproducible target gain (source-side observability failure); **LPC legacy** (§5.15) — "source-free adaptation diagnostics are not deployment controllers," the same measurement→control gap, with a target-label-leakage sub-line caught and retracted; **CSC** (§5.12, §5.14) — the unlabeled marginal is provably insufficient to certify concept vs covariate shift (an *information-contract* boundary), the deployment-relevant analogue of "measurable ≠ identifiable"; **ACAR** (§5.11) — on a clean external-compatible substrate, no policy in the frozen candidate universe is safe/beneficial (DEV_STOP), so even an action-conditional deployment successor confirms the control side is where these lines fail. None of these is a positive method; together they widen "measurable ≠ relied-upon" into "measurable ≠ controllable/identifiable."

---

## 7. Missing-metric summary

| missing_metric | needed_for | candidate_source | blocking_level | proposed_resolution |
|---|---|---|---|---|
| Target **NLL / ECE** for all CMI-cluster routes (CIGL/FCIGL/dCIGL/MetaCMI/CITA) | L6 consequence beyond bAcc; calibration angle of RQ2 | preds saved per run? (`save-probabilities` policy) — needs check | MEDIUM | CPU_recompute from saved `.preds.npz` if present, else defer |
| Per-route **random-k multi-dataset** numbers (TOS) | strengthen the "non-specific NLL" falsifier beyond 2a | `erasure_target_deploy/{Lee2019_MI,Cho2017,...}` | LOW | CPU_recompute (frozen latents on disk) |
| **ECE** anywhere in TOS deployment | RQ2 calibration | frozen latents + saved probs | LOW | defer (bAcc+NLL suffice for Step 2) |
| **Frozen per-branch embeddings** `spatial_z / graph_z / node_z` (FBCSP-LGG) | RQ4 branch-local leakage audit | none on disk (0 binary artifacts on branch) | HIGH | small_frozen_probe (Phase 3/4, only if Phase 2 justifies) — re-inference to dump `last_spatial_z` per fold |
| **Per-branch leakage probe AUC** (FBCSP-LGG) | RQ4 — is load-bearing-branch leakage different from non-load-bearing? | none (only fused-Z in-training GLS KL) | HIGH | small_frozen_probe (couples to the embedding dump) |
| **MetaCMI seeds 1/2** + bootstrap CI | method-level (vs seed0-screening) status parity with FCIGL/dCIGL | deliberately not run (no-signal rule) | LOW | defer (frozen premise says do not run) |
| **R3 logit-SymKL / CE-delta** for CIGL (L5 beyond task_drop) | finer L5 granularity for RQ1 | may live in `cigl_direct_reliance/`,`cigl_functional/` R3 CSVs (not cited by CIGL_62/65/66) | LOW | reuse if schema-compatible, else defer |
| **Worst-subject** breakdown (CIGL cluster) | L6 worst-case consequence | fold-level rows in `r3_reliance.csv` / pareto CSV | LOW | CPU_recompute from per-fold rows |
| Pure first-moment **mean-scatter EEG deletion** number (TOS) | isolate mean-scatter vs score-Fisher | synthetic-only currently | LOW | defer (score-Fisher `TOS_VD` is the EEG instantiation) |
| **ACAR Stage-2B machine-readable result** (summary JSON/manifest) | re-deriving the 22-row DEV_STOP gate table from committed data | notes only committed; raw report on cluster (SLURM 885395) | MEDIUM | reuse (numbers verified in notes) — commit a summary JSON if ACAR is cited quantitatively |
| **OACI C18 exact stress numbers** (endpoint-estimability) | quantifying §5.8 beyond the qualitative finding | `oaci/reports/C18_*.json` (on disk, not parsed here) | LOW | CPU_recompute/reuse (JSON present) |
| **CSC B7.1 dual-witness confirmatory** result | closing the CSC PENDING route (§5.13) | committed protocol JSON, UNRUN | LOW | defer (CSC-internal, out of FSR Phase-2 scope) |
| **Real-clinical (PD/SCZ) external validation** — ACAR + CSC | external validity of the deployment/boundary findings | lockbox sealed, never run | LOW | defer (not an FSR deliverable) |

**Blocking-level HIGH items are exactly the two RQ4 branch-local artifacts** — both absent, both requiring a small frozen-probe re-inference (Phase 3/4, GPU, only if Phase 2 justifies). Everything else is CPU-recomputable from frozen artifacts or deferrable.

---

## 8. What Step 1 establishes

Every required route is placed on the ladder with an artifact path + SHA, a decision, and an allowed/forbidden boundary: the CMI-control cluster (CIGL, FCIGL, dCIGL, MetaCMI, CITA_lambda_1, CITA_lambda_0.010, TTA_Control), the TOS erasure cluster (mean_scatter, LEACE, INLP, RLACE, random_k, refusal_gate, global_LPC_collapse, task_preserving_erasure, capacity_factorial), the FBCSP branch cluster (branch_ablation, gate_summary, bottleneck_analysis, graph_starvation, CIGL_35 blueprint, P6 scaffold), and the repository-wide corroborators (OACI ×4, ACAR ×3, CSC ×3, LPC legacy, prior-decoupled TTA ×2 background). Every number was adversarially re-verified against its raw artifact (all confirmed within rounding; all claim-boundaries clean); one PM assumption was corrected by the evidence — **ACAR Stage-2B is a completed DEV_STOP, not pending**.

The load-bearing empirical fact for the FSR thesis is already frozen and in-hand: **measured leakage magnitude is anti-correlated with functional reliance (`graph_kl→R3` = −0.34), while task-head alignment is positively correlated (`align_k2→R3` = +0.34)** — i.e. *measurable ≠ relied-upon* — and the same measurement≠consequence gap recurs across erasure (§3, target benefit decoupled from leakage removal), source-side observability (OACI), the information contract (CSC), and deployment (ACAR). Phase 2 (CPU-only) tests whether the leakage↔reliance↔alignment pattern is a statistically robust cross-route regularity.

**Ladder-coverage note (Phase-1 gate).** The Phase-1 gate requires each route on ≥3 of {L1, L2, L5, L6}. The CMI-control cluster clears this cleanly. Pure diagnostics do not, by their nature: FBCSP branch-ablation (L1-fused + L4), the refusal gate (L1 + refuse), and the TOS erasure operators (L3/L4/L6, no L5) each populate 2 canonical slots — this is a true property, and for FBCSP it coincides exactly with the one HIGH missing metric (no per-branch leakage probe on disk, §7). It is recorded here rather than papered over; Phase 1 must either accept these as partial-coverage corroborators or supplement them (the FBCSP frozen-probe run) before the ≥3-level gate can pass for them.

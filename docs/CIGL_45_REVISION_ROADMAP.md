# CIGL_45 — Revision Roadmap (CIGL-2): from "proxy + regularizer" to a rigorous EEG graph leakage-control framework

> **Purpose.** The reviewer judged the current v0.6 draft *honest and well-bounded* but **too narrow / evidentially thin** for a strong methods paper — currently a "measurement + partial regularization" workshop-level result. This roadmap turns the review into a **gated, GPU-approval-controlled program** that upgrades the evidence from "a proxy can be reduced vs ERM" to "a reliable, transferable, comparable, diagnosable leakage-control framework."
>
> **Protocol (unchanged).** Every experiment is **reviewer-GPU-gated**. Each step is: build non-GPU scaffolding + tests → push to GitHub → reviewer reviews → **explicit GPU approval** → run → results doc → next gate. **No GPU runs without approval. Fixed-λ=0.010 pre-registered confirmation stays as an honesty anchor. Do not touch CITA/DualPC/Tri-CMI. No fabrication; honest-null branches are pre-committed.**
>
> Grounded against the actual `project/cigl` code on 2026-07-01 (infrastructure verified to exist; see §Grounding).

---

## 0. Grounding (what already exists — verified, big reuse)

| Component | File(s) | Status |
|---|---|---|
| Posterior-KL audit + within-label permutation null + bootstrap CI | `cmi/eval/graph_leakage.py`, `probe_splits.py`, `graph_map_stability.py` | ✅ exists; `n_perm` default 20–50, seeds 3, no FDR |
| **Multi-probe suite** (linear, mlp_s/mlp_l, rf, hgbm, hsic, knn_cmi) | `cmi/eval/leakage_audit.py` | ✅ exists **but NOT wired into the phase-3A audit** (only 1-layer MLP runs today) |
| **DG baseline methods** (CORAL, MMD, IRM, V-REx, ChSIC, SCLDGN, DANN, CDANN, GroupDRO) | `cmi/methods/dg_penalties.py` + trainer dispatch in `cmi/train/trainer.py` + `cmi/run_loso.py` | ✅ exists (generic DG infra — **not** CITA/DualPC/Tri-CMI); needs wiring to the DGCNN adapter + phase-3A audit |
| DGCNN static-adjacency backbone (`edge_logits=None`) | `cmi/models/graph_task_backbones.py` | ✅ the task-capable CIGL backbone |
| Dynamic-adjacency + edge-CMI (GraphCMINet, `EdgePosterior`) | `cmi/models/gnn.py`, `cmi/methods/graph_regularizers.py` | ✅ exists but overfits (the negative result); basis for constrained-edge phase |
| Dual-λ config skeleton (declared, unused) | `h2cmi/config.py` (`dual_lr`, `lambda_init`, `lambda_max`, `warmup`) | ✅ fields exist; not yet implemented in trainer |
| Source-only LOSO + selection + firewall | `cmi/run_loso.py`, `cmi/run_lambda_select.py`, `run_cigl_phase3a_*` | ✅ exists |
| Datasets: MI (2a, 2b, PhysionetMI, HGD, …) + session-domain protocol | `cmi/data/moabb_data.py`, `cmi/paths.py` | ✅ PhysionetMI present in datalake; `subject_session` domain + `leave_one_session` implemented |

**Implication:** much of the review's ask is *integration + hardening + new analysis*, not greenfield. The biggest cost is **GPU compute** (more seeds, more permutations, a baseline suite), not new algorithms.

---

## 1. The five revision phases (R1–R5) + rewrite

Numbered **R1–R5** to avoid clashing with the manuscript-polish phases (4A–4J). Each phase = non-GPU scaffolding (pushed for review) → gated GPU runs → results doc.

### R1 — Evidence hardening + multi-probe stress audit  *(priority 1)*
**Review:** §three-A1, drawbacks 5,6. **Goal:** make "leakage exists and CIGL reduces it" statistically unimpeachable and probe-family-robust.
- **Build (non-GPU):** `cmi/eval/evidence_hardening.py` (hierarchical bootstrap dataset→fold→seed; Benjamini-Hochberg FDR; exact perm p `(1+#≥)/(B+1)`), `probe_calibration.py` (ECE + temperature scaling), `multiprobe_audit.py` (wire the existing 7-probe suite over frozen features + agreement/envelope), probe early-stopping leak check + strict train/val/audit-test isolation. **Tests for each.**
- **GPU runs (gated):** (2a) fold-0 hardened pilot `n_perm=1000`, seeds 3, ERM + `graph_node_010`; (2b) **graph-only / node-only / graph+node ablation** (the reviewer's required decomposition); (2c) **λ-curve** {0,0.001,0.003,0.01,0.03,0.1}; (3a) multi-fold × **10-seed** confirmation `n_perm=1000` with FDR + hierarchical CI; (3b/4a) multi-probe + calibration post-hoc on frozen features (CPU).
- **GPU estimate:** ~150 (2a) + ~75 (2b) + ~150 (2c) + ~900 (3a) ≈ **1,150–1,300 V100-hrs** (phase it; 3a is the big one and can drop to 5 seeds ≈ 450 hrs if budget-constrained). Non-GPU tiers ~1 day CPU.
- **Success:** `graph_node_010` still clears the **BH-FDR-corrected** null on ≥2/3 seeds (pilot) and the primary folds (confirmation); **≥5/7 probe families** agree leakage exists; ranges hold under 10 seeds. **Stop:** if the hardened null is *not* cleared → the effect was under-powered; report honestly and re-scope.

### R2 — Baseline suite on the same backbone + Pareto frontier  *(priority 1 — highest methodological value)*
**Review:** §three-A2/A3, drawbacks 1,2. **Goal:** prove CIGL beats (or Pareto-dominates) existing conditional/adversarial/alignment regularizers — not just ERM.
- **Build (non-GPU):** wire the existing `dg_penalties.py` methods to the **DGCNN static adapter + phase-3A source-only audit**; add two small heads — `NodeAdversaryDiscriminator` (NodeDAT, ~30 lines in `graph_regularizers.py`) and `dg_eegdg` (EEG-DG marginal+conditional, ~40 lines in `dg_penalties.py`); unified metric set (source/target bAcc, graph/node KL, stress-probe leakage, top-k subject retrieval, reduction, target-change, Pareto). **Additive only** (new configs/flags; do not alter shared method semantics). Smoke test all methods on CPU.
- **Baselines:** ERM, marginal DANN `q(D|Z)`, conditional DANN `q(D|Z,Y)`, CDAN `q(D|Z⊗Ŷ)`, CORAL/label-CORAL, MMD/label-MMD, IRM/VREx, GroupDRO, NodeDAT, EEG-DG, **CIGL (g / n / g+n)**.
- **GPU runs (gated):** BS-2 pilot (1 fold × 1 seed × {ERM, DANN, CDANN}) to de-risk; then BS-3 full suite (~11 methods × 9 folds × 3 seeds).
- **GPU estimate:** BS-2 ~12–15 hrs; BS-3 ~**1,000–1,200 V100-hrs** (staggerable). Post-hoc audit/Pareto = CPU.
- **Success:** CIGL sits on the **leakage-vs-task Pareto frontier** (lower leakage at equal task, or higher task at equal leakage) vs the conditional/adversarial/alignment baselines. **Stop / honest branch:** if a plain conditional-DANN matches CIGL, the contribution narrows to the *audit protocol + node-level analysis* (still publishable, but reframed) — report it.

### R3 — Mechanism & reliance (does the classifier *use* the leakage?)  *(priority 1 — the review's "most valuable single experiment")*
**Review:** §nine/ten, drawbacks 3,7,8. **Goal:** connect *decodability* → *classifier reliance* → *generalization relevance*.
- **Build (non-GPU / light):** save per-fold node/edge leakage maps + frozen features as `.audit.npz` sidecars (currently only means are saved); `cmi/eval/{leakage_removal, node_masking, spatial_correlation, topomaps}.py`.
- **Experiments (mostly forward-pass / post-hoc; light GPU):**
  1. **Leaking-subspace removal** (flagship): project the source-fit subject-predictive subspace out of `Z`; measure task-bAcc drop, ERM vs CIGL. Large drop for ERM + small for CIGL ⇒ CIGL reduces *shortcut reliance*, not just decodability.
  2. **Top-leak node masking** curves (top-leak vs random vs top-task-saliency).
  3. **Task-gradient ↔ leakage-map** spatial correlation.
  4. **Node-leakage topomap** before/after + cross-fold/seed/dataset stability (extends the corr≈0.945 result into a figure).
  5. **Leakage-reduction ↔ target-bAcc** correlation (the second-level DG-relevance test the reviewer wants).
- **GPU estimate:** **~20–30 V100-hrs** (forward passes + light gradients); most analysis is CPU.
- **Success:** ERM reliance-drop ≫ CIGL reliance-drop; leakage-reduction correlates with target-bAcc (ρ>0.3). **Pre-committed honest branch (R3-pivot):** if reliance is null / decoupled from target, **reframe CIGL as a representation-privacy / leakage *audit* (+ variance-reduction regularizer), not shortcut mitigation** — and say so plainly. This is the single most decision-relevant phase for the paper's framing.

### R4 — Constrained CIGL (method upgrade)  *(priority 2)*
**Review:** §four, fixed-λ drawback. **Goal:** turn the hand-set λ into a source-only **constrained** framework.
- **Build (non-GPU):** `cmi/train/constrained_trainer.py` — primal-dual Lagrangian `min CE s.t. R_g≤τ_g, R_n≤τ_n` with `λ_j ← [λ_j+η(R_j−τ_j)]_+` (cap + warm-up + task-retention monitor); `threshold_selection.py` (τ from perm-null `μ+cσ` or reduction target `α·R_ERM`, source-only); `run_constrained_phase4_selection.py` (nested inner-LOSO λ selection, target locked); Pareto reporting; **firewall test** (mask target → identical selection). Reuse the clean Step-A/Step-B loop + the `h2cmi/config.py` dual fields.
- **GPU runs (gated):** nested selection (~128 jobs) + main training/eval on the selected config, **alongside** the fixed-λ=0.010 anchor.
- **GPU estimate:** **~13–20 V100-hrs**.
- **Success:** constrained CIGL matches/beats fixed-λ on the Pareto frontier with **no target leakage** (firewall test byte-identical). Keep the pre-registered fixed-λ result reported side-by-side.

### R5 — Edge / dynamic-graph + data expansion  *(priority 5 — highest-risk, most ambitious)*
**Review:** §five, drawbacks 4,9,10 + data. **Goal:** convert "edge-CMI out of scope" into a real edge-level audit, and widen datasets.
- **Build/experiments:** constrained dynamic adjacency `A(x)=A₀+ΔA(x)` (Frobenius/L1/low-rank/smoothness + anatomical prior), gated behind task-capability + no-overfit checks; **edge-CMI audit** `R_e` once `A(x)` is task-capable; **dynamic-edge mechanism ablations** (A(x)→subject acc, →label acc, within-label shuffle, subject-mean substitution, stop-grad/frozen/random A) to causally probe the fingerprint hypothesis; static-adjacency interpretability (learned `A₀` vs electrode distance / motor network; is CIGL still effective with a channel-MLP instead of graph-conv?); **+PhysionetMI (and ≥1 more MI) + a cross-session `(subject,session)` domain**.
- **GPU estimate:** **~400–600 V100-hrs** (this is a second paper's worth; do last, or defer).
- **Honest flags:** edge collapse under stacked penalties; PhysionetMI is binary L/R (class-subset curation needed); session folds are fewer/noisier; montage/channel mismatches. Pre-commit to reporting nulls (constraints collapse `A(x)` / edge-CMI inert / MLP ≥ graph-conv / dataset infeasible).

### R6 — Manuscript v1.0 rewrite  *(after R1–R3 at minimum)*
Rewrite around the stronger evidence: retitle toward **probe-audited, label-conditional graph/node leakage reduction on task-capable static EEG graph networks**; expand Related Work (conditional-invariance, DANN/CDAN, RGNN-NodeDAT/EEG-DG, classifier-based CMI); results order = backbone selection → leakage exists (multi-probe) → CIGL vs ERM → **CIGL vs baselines + Pareto** → **reliance tests** → constrained CIGL → negatives/edge → limitations. Keep every honest-null outcome visible.

---

## 2. Recommended execution order (minimal-viable fast path first)

The review's own "minimum revision" + "most valuable experiment" imply this order:

1. **R1 scaffolding** (non-GPU): stats hardening + multi-probe wiring + λ/ablation runners. *(no GPU; push for review now)*
2. **R2 scaffolding** (non-GPU): baseline wiring on DGCNN + NodeDAT/EEG-DG heads + unified metrics. *(no GPU)*
3. **R3 scaffolding** (non-GPU): artifact saving + reliance/masking/topomap modules. *(no GPU)*
4. **GPU gate #1 (small, high-value):** R2 baseline **pilot** (1 fold: ERM/DANN/CDANN) + R3 **leaking-subspace-removal flagship** (forward-pass) + R1 fold-0 hardened pilot. ~**Small** (≈ tens of GPU-hrs) — buys the biggest credibility jump.
5. **GPU gate #2 (large):** R2 full baseline suite + R1 10-seed n_perm=1000 confirmation. ~**Large** (≈ 2,000 GPU-hrs combined) — the核心 evidence.
6. **GPU gate #3:** R4 constrained CIGL (small).
7. **R6 rewrite** once R1–R3 land; **R5** last (or defer to a follow-up paper).

This front-loads the three things the reviewer said the paper needs — **baselines, a reliance test, and hardened statistics** — while keeping big compute behind explicit gates.

---

## 3. GPU budget summary (for your planning)

| Phase | GPU (V100-hr, est.) | Notes |
|---|---|---|
| R1 hardening | 1,150–1,300 (or ~450 at 5 seeds) | 3a multi-fold×10-seed×n_perm=1000 dominates |
| R2 baselines | 1,000–1,200 (+15 pilot) | 11 methods × 9 folds × 3 seeds |
| R3 mechanism | 20–30 | mostly forward-pass / CPU |
| R4 constrained | 13–20 | small |
| R5 edge+data | 400–600 | highest-risk; defer |
| **Total (R1–R4)** | **~2,200–2,550** | **R5 roughly doubles it** |

These are *planning estimates* (agent-derived), to be firmed up per SLURM partition after the pilots. Everything is submittable in staggered arrays; nothing runs without your approval.

---

## 4. Gate protocol for this program (how each step reaches you)

- Each phase's **non-GPU scaffolding + tests** is implemented on its own `project/cigl-rN-*` branch and **pushed** — you review the code diff + tests on GitHub (no compute spent).
- I then **request GPU approval** with the exact run spec (partitions `A100,V100,V100-32GB,A40`, default QOS, no `--qos`/`--time`, fail-closed on no-CUDA, no silent seed/fold/perm reduction).
- After an approved run: a `docs/CIGL_NN_*` results doc + updated tables/figures, pushed for your decision on the next gate.
- **Honest-null branches are pre-committed** (R2 baseline-parity, R3 reliance-null pivot, R5 collapse/inert) and will be reported without spin.

---

## 5. Immediate next step (pending your gate)

**Recommend: authorize R1 + R2 + R3 *scaffolding* (all non-GPU, code+tests only).** That builds the hardened-stats modules, wires the existing 7-probe suite and the DG-baseline suite onto the DGCNN backbone, and adds the reliance/artifact-saving modules — all reviewable on GitHub with **zero compute** — and positions us for a small, high-value first GPU gate (baseline pilot + leaking-subspace-removal flagship).

I will **not** submit any GPU job until you approve a specific run spec. Tell me which phase(s)/scaffolding to start, and whether the fast-path order above is right.

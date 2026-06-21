# AAAI protocol/code audit — the pre-submission fix batch (started 2026-06-20)

Status snapshot at start of this batch (all pushed to `github.com/W-Yinghao/CMI`):
- **Headline (5-seed, nested no-oracle):** SCZ ERM 51.9±1.0 → CITA-nested 55.1±1.1 = **+3.1±1.3** (all 5 seeds +,
  1.4…5.1); PD 58.3±0.8 → 61.0±0.9 = **+2.7±1.0** (all 5 seeds +, 1.7…4.7). ± = seed std; **cohort-clustered
  bootstrap CI (2–4 cohorts) will be wider** — that is the CI to report.
- **Baselines (protocol-matched, 2-seed):** SPDIM **ties/slightly beats** CITA (SCZ 54.2/55.4 vs 54.0/55.8; PD
  63.6/62.3 vs 62.2/60.8); T3A ≈ native. ⟹ CITA's gain is over **ERM**, not over the strongest baseline →
  positioning is **methodological/diagnostic** (closed-form + CMI screen + lower leakage + safety gate).
- **Theory blockers fixed (commit fbc5939):** P0-2 (plug-in estimator, NOT an upper bound), P0-3 (DPI equality is
  `D⊥Y|Z`, not `Y=f(Z)`), P0-5 (GLS reference `p_d_ref=p_d`). P0-4 already handled.
- **CMI gate is load-bearing** in the confident-but-wrong regime (catches errors confidence rejection reverses on);
  score-direction wording corrected (reverse-ranking, not "blind").
- **TUAB sealed** (`notes/TUAB_LOCKBOX.md`) until this batch + freezing are done.

## The protocol/code-table batch (reviewer items), in priority order
**P1 — could move a HEADLINE number (do first):**
1. **Recording/session-grouped leakage split.** Current leakage probe uses a *random trial/window* split, which
   EEG within-recording autocorrelation lets the probe exploit → inflates measured `I(Z;D|Y)` and label
   separability. Fix: split source pool by **subject/recording** (whole recordings to train vs eval). Rerun the
   leakage numbers; compare random vs grouped. *If grouped shrinks the leakage gap, the leakage claim must be
   restated.* ← **STARTED HERE.**

**P2 — correctness (may shift numbers within noise):**
2. **Seed before backbone build.** `train_model` sets seed *after* `build_backbone` → methods may not share init;
   order-dependent. Fix: set py/np/cuda seed before build; clone one initial `state_dict` across methods.
3. **Double class-balancing.** classbal sampler + inverse-class CE weight + GLS reweight can stack. Each method
   must declare its *effective* training distribution; GLS methods use the raw sampler (or re-derive weights
   under the sampler-induced distribution).
4. **`drop_last=True`** drops rare domain×class cells (CMI is sensitive). Turn off / domain-class-aware batching.
5. **Multi-capacity leakage probes.** One MLP capacity → low leakage may be probe underfitting. Report
   linear / 2-layer MLP / strong MLP / kNN-or-HSIC; take the max detectable leakage.
6. **Signed MI/residual estimates + CIs.** Stop truncating negative estimates to 0 (upward bias); keep signed +
   CI in the statistics, truncate only in visualization.

**P3 — protocol completeness:**
7. Full-pipeline nested selection (encoder + alignment + gate selected together, not encoder-only then alignment).
8. Unified preprocessing manifest (immutable dataset hash); separate confirmatory table from exploratory results.
9. P0-5 rerun for any *marginal/dual-CMI* numbers the paper reports (headline CITA unaffected).
10. Offline / mini-batch / online-streaming TTA reported separately; never one "DG accuracy" header.

**Then:** freeze all configs → unseal TUAB once (class-spanning target batches per the lockbox).

## Decisions / findings log (appended as the batch runs)
- (start) Beginning P1.1 recording-grouped leakage split: add `--leakage_split {random,grouped}` to
  `run_scps_crossdataset`; grouped assigns whole subjects to the probe-train vs probe-eval split.

- (P2.2 ✅) seed py/np/torch/cuda BEFORE build_backbone (`_seed_all`) in config loop + nested `_train_on` — all methods/folds now share initialization (paired comparison, order-independent).
- (P2.4 ✅) `drop_last` now only when the tail batch would be size 1 (BN-safe); otherwise keep it so rare domain×class cells survive.
- NOTE: P2.2 changes initialization → the final frozen confirmatory run will re-derive the +3 numbers (robust across seeds, expected within noise). r11leak (P1.1) launched on OLD code is unaffected (it compares splits, not init).

- (P2.3 ✅) raw sampler now forced for ALL GLS methods (dualpc* AND dualc/dual+reweight) — no double-balancing. Headline (lpc_prior/erm) uses no GLS → unaffected.
- (P2.5 ✅) `leakage_probe(cap=...)` linear|mlp|strong + `--leakage_multicap` → report MAX detectable leakage (anti-underfit). r11mc (grouped split + multicap) launched.
- (P2.6) signed-MI + CIs handled at HARVEST time (report signed values + CI, truncate only in plots) — applied in the r11leak/r11mc harvesters.

## P1.1 RESULT — recording-grouped vs random leakage split (r11leak, seed 0)
| cond | ERM leakKL random→grouped | lpc0.3 random→grouped | ratio random→grouped |
|---|---|---|---|
| PD  | 0.205 → 0.225 | 0.021 → 0.039 | 10.0× → **5.7×** |
| SCZ | 0.456 → 0.415 | 0.063 → 0.057 | 7.2× → **7.3×** |
**Verdict: leakage claim SURVIVES the honest grouped split.** ERM leakage ~unchanged (within ~10%) ⟹ within-recording
autocorrelation is NOT a material confounder for cross-site (D=cohort) results; bAcc identical (split only touches the
probe). The cut holds at **~6–7×**. HONEST SCOPING: the "10–100×" headline is the within-dataset *subject*-level number;
**cross-site *cohort* leakage cut is ~6–7×** — restate accordingly. (Awaiting r11mc multi-capacity: does a STRONG probe
detect more lpc leakage = underfit? If strong≈mlp, the low-leakage claim is robust to probe capacity too.)

## COHORT-CLUSTERED CI (the honest headline statistics) — GPU-free from 5-seed
| | per-cohort gains | mean | cohort-clustered 95% CI | sig? |
|---|---|---|---|---|
| SCZ (4 cohorts) | +0.2,+1.0,+5.9,+5.3 | +3.1 | **[+0.6, +5.6]** | ✅ excludes 0 |
| PD (3 cohorts) | +7.5,+1.0,−0.5 | +2.7 | **[−0.5, +7.5]** | ❌ includes 0 |
**HONEST HEADLINE RESTATEMENT:** seed-level ±1 was over-optimistic. At the cohort-clustered level (the correct
unit), **SCZ +3.1 is significant; PD +2.7 is NOT** (driven by 1 cohort, 3 cohorts = no power). Gain is
cohort-HETEROGENEOUS (some +5-7, some flat/neg). ⟹ (a) report cohort-clustered CIs + worst-cohort, not seed std;
(b) PD needs MORE COHORTS (OpenNeuro same-disease, see cmi-scps-crossdataset memory) to establish significance;
(c) combined with SPDIM-parity, the accuracy contribution is "consistent but cohort-underpowered on PD" — the
paper must say this plainly. This is the single most important honesty fix for the AAAI submission.

## P2.5 RESULT — multi-capacity leakage (grouped split, r11mc, seed 0)
| | linear | mlp | strong | cut erm/lpc0.3 (strong) |
|---|---|---|---|---|
| PD  erm/lpc0.3 | 0.018/0.002 | 0.183/0.037 | 0.369/0.120 | **3.1×** |
| SCZ erm/lpc0.3 | 0.035/0.001 | 0.432/0.059 | 0.690/0.193 | **3.6×** |
**VERDICT: the mlp probe was UNDER-fitting — a strong probe finds 2–3× MORE leakage; linear finds ~none (leakage is
nonlinear).** The honest, anti-underfit leakage cut (strong probe + grouped split) is **~3–3.6×** on cross-site
cohorts. The lpc-vs-erm RATIO is preserved across capacities (lpc genuinely reduces leakage) but the MAGNITUDE
deflates: 10–100× (subject-level, weak probe, random split) → 5–7× (cohort, mlp) → **~3–3.6× (cohort, strong probe)**.
⟹ the paper MUST report the capacity sweep (or `maxcap`) as the headline leakage, NOT the single-mlp number.

## NET honest AAAI picture (after the full protocol audit)
- **Accuracy:** SCZ +3.1 [+0.6,+5.6] sig; PD +2.7 [−0.5,+7.5] NOT sig (underpowered); **ties SPDIM**.
- **Leakage:** **~3–3.6× honest cut** (strong probe), not 10–100×.
- **CMI gate:** load-bearing in the confident-but-wrong regime (the genuinely novel, defensible piece).
⟹ contribution is **methodological/diagnostic** (closed-form, source-free, abstention-safe CMI screen), NOT an
accuracy-or-leakage-magnitude SOTA. Both headline magnitudes deflate under rigorous measurement — must be stated.

## P1.4 RESULT — hierarchical variance decomposition (the PD diagnosis)
SCZ (4 coh, ICC=0.44 MIXED): per-cohort +0.2/+1.0/+5.9/+5.3; LOCO mu stays +2.2..+4.1 (robust); already sig.
**PD (3 coh, ICC=0.78 COHORT-HETEROGENEITY-DOMINATED):** per-cohort ds002778 +7.5 / ds003490 +1.0 / ds004584 −0.5;
**LOCO drop ds002778 → mu +0.27** (entire effect = 1 cohort); **power sim: +5 same-dist cohorts only P(lower>0)≈0.59.**
⟹ **PD is 情况 B (heterogeneity), NOT 情况 A (few cohorts). Adding cohorts will NOT reliably fix it — must EXPLAIN
the heterogeneity first** (per the decision rule). SCZ adding cohorts → tightens (情况 A-ish).

## P1.4b — PD heterogeneity is STRUCTURAL + the gate-harm hypothesis
Per-cohort: alignment helps ∝ (covariate-shift μgap × target-separability); HURTS on imbalanced/low-separability
ds004584 (sep 63.3, imbal 0.66, gain −0.5). ds002778 wins (+7.5): worst base 52.3 + big shift 0.64 + separable 71.4.
**Gate-harm check (does the gate predict the −0.5 harm?):**
| cohort | gain | g_unc(current gate) | imbal | tgt_typ P(target|z) |
|---|---|---|---|---|
| ds002778 | +7.5 | 1.00 | 0.03 | 0.63 |
| ds003490 | +1.0 | 1.00 | 0.00 | 0.41 |
| ds004584 | −0.5 | 1.00 | 0.32 | **0.85** |
⟹ **the current reliability gate (g_unc) is BLIND to adaptation harm (1.00 everywhere); but the domain-density
signal P(target|z) (CMI-family I(D;Z)) flags the harmful cohort (0.85), as does class imbalance.** The SAME
density signal was load-bearing in the confident-but-wrong concept study (AUROC 0.87–1.00). **GATE-REDESIGN
HYPOTHESIS (for P2):** harm-gate = f(domain-density P(target|z), class-imbalance/separability), NOT the entropy/
uncertainty α-gate. Test in the P2 gate stress suite (pre-screening vs post-abstention, harm as the target).

## P1.5 RESULT — representation-collapse audit (genuine suppression vs collapse)
Stage means (grouped CI over cohorts): pre-align(erm) task_bAcc 66.5[62,71] effRank 10.1 | post-align 66.4[63,70]
effRank 10.5 | **post-LPC 59.2[53,65] effRank 9.0**.
- ✅ ALIGNMENT preserves representation (task_bAcc & effRank unchanged) — no collapse from the test-time transform.
- ✅ Domain leakage drops at ALL probe capacities (r11mc linear/mlp/strong) — not weak-probe hiding.
- ⚠️ LPC reduces leakage PARTLY via a representation cost: task-separability −7 (66.5→59.2), effRank −1.1. BUT
  disproportionate: domain leakage −67% (~3×) vs task-structure −11% ⟹ **MOSTLY genuine suppression, NOT free** —
  an honest leakage–utility tradeoff. (Caveat: task_bAcc is the LINEAR-probe upper bound; the model's own head
  may recover some — lpc native bAcc ≈ erm. One outlier fold SCZ_ds004367 pre-align degenerate.)
- TODO capacity-saturation: add GBT/kNN (different inductive bias) probe to confirm the domain-leakage plateau.
**Honest leakage conclusion (final):** lpc genuinely suppresses conditional domain leakage (~3× under strong probe,
disproportionate to the ~11% task cost), measured under grouped split + multi-capacity probe — NOT 10–100×, and
NOT free, but real.

## P0.2 + P1.6 dry-run results (pipeline tested, NOT frozen)
- **Stability-report pipeline TESTED on PD r11fp** (encoder-only vs full-pipeline selection, 2 seeds): pick
  agreement 67%; full-pipeline MORE STABLE (entropy 0.45 vs 1.01) + LOWER regret vs oracle-best (0.79 vs 1.95)
  ⟹ full-pipeline is the better protocol-correct selector. DECISION DEFERRED until SCZ r11fp lands (harvest both).
- **Gate-suite dry run:** harm-label (base-correct→adapted-wrong) mechanics OK; negative-control verified — the
  domain-density gate correctly FAILS on pure_conditional (AUROC 0.52). FIX logged: pure_conditional must relabel
  margin-UNCORRELATED samples (boundary-rotation variant lets MSP catch low-margin → artifact).
- **Falsification slice ADDED to GATE_STRESS_SUITE.md §A0** (P2 step 1): compare α-gate / P(target|z) / covariate /
  separability / shift×sep / CMI against BOTH sample-harm AND cohort-loss; unify into ONE gate only if same score
  wins both on held-out cohorts+generators, else TWO-LEVEL controller (batch eligibility + sample abstention).

## r12 plan — continuous decomposition, two-layer freeze, gate status (per directive 2026-06-21)
**Effect decomposition (NOT a binary):** Δ_init = g(enc,current)−g(enc,old); Δ_protocol = g(full,current)−
g(enc,current); total = Δ_init+Δ_protocol. r12enc landing between 1.5 and 3 ⟹ BOTH contribute; don't force-attribute.
Pre-registered practical-equivalence band **EPS_EQ=0.5 bAcc** (set BEFORE results). Harvester: `scripts/r12_decomp.py`
(per-cohort direction, LOCO, config transitions, selection entropy, inner→outer regret, CITA-abs & ERM-abs separately).
**FAIL-FAST equivalence:** r12enc & r11fp share code/seed/init ⟹ their per-CONFIG outer bAccs must be IDENTICAL per
fold (only selection differs); mismatch ⟹ run-path inequivalence (batch order/checkpoint/init), not protocol. No 4th
cell (old-init+full) — old code is out of deployment; only needed for init×protocol interaction if 3 groups conflict.
**Version-mixing FIX:** collapse audit had erm dumps PRE-P2.2 (feat_dump, Jun-20) vs lpc POST-P2.2 (feat_dump_lpc).
Re-dumping matched erm+lpc from ONE current-code run → `results/feat_dump_v2/` (r12feat); redo P1.5 on it.

**TWO-LAYER FREEZE:**
- **Freeze A (after r12enc):** core adaptation + accuracy pipeline — current P2.2 init, source checkpoints +
  serialized state, outer/inner splits, candidate grid, FULL-PIPELINE selection objective, target batch
  construction/order, seeds, eval code + immutable output schema. Freezes the PROTOCOL/estimation PROCESS, NOT the
  headline — accept whatever gain (1.2/1.8/3.0). Adopt current-code full-pipeline regardless of magnitude.
- **Freeze B (after P1.5 + falsification slice):** full deployment system (LPC utility/collapse boundary + gate
  control form decided) → then formal P2 suite → then TUAB (last, once).

**STRICT ORDER after r12enc:** (1) equivalence checks + 3-way decomposition; (2) selection-stability report;
(3) Freeze A manifest; (4) close P1.5 on version-matched dumps (capacity-saturation, stage-wise leakage,
separability, eff-rank, var/norm, utility cost); (5) minimal gate falsification slice; (6) decide single-scalar vs
two-level controller; (7) full 8-generator stress suite + adapter matrix.

**GATE STATUS (precise):** strongest UNIFIED CANDIDATE signal — unification NOT established. Evidence: flags danger
in high-margin confident-wrong generator; domain-density ↔ harmful PD cohort. Falsification (predict REAL harm
`1[base-correct, adapted-wrong]` + batch ΔL=L_adapted−L_base, NOT just 'shift happened') decides:
| result | system |
|---|---|
| same frozen direction+aggregation predicts BOTH sample-harm & batch/cohort-harm on held-out | single deployable gate |
| density predicts batch-level harm, CMI predicts sample-level danger | two-level controller (eligibility + abstention) |
| effect depends on specific generator/cohort, unstable held-out | diagnostic only, NOT a unified controller |
`pure_conditional` negative control EXPECTED to fail (defines the label-free density identifiability boundary).

## r12enc FAIL-FAST CAUGHT A REAL PROBLEM (freeze BLOCKED, exit 2) — LPC nondeterminism
Per-config |r12enc − r11fp| outer bAcc: **erm:0 = 0.00 (BIT-IDENTICAL)**; lpc0.1 max 0.47-0.69; **lpc0.3 max 3.96**
(SCZ). Same GPU (V100-PCIE-16GB). ⟹ NOT hardware, NOT general nondeterminism, NOT a protocol effect per se: it's
**LPC-SPECIFIC CUDA nondeterminism** — the CMI critic/dual-alternation has a non-deterministic CUDA op, so the
outer LPC training is COUPLED to the GPU state left by the (different) inner CV. ERM (plain CE) is deterministic.
**IMPLICATION:** LPC is not run-to-run reproducible (~0.8–4 bAcc) → clean protocol comparison impossible AND the
headline (LPC-selected) carries hidden nondeterminism on top of seed variance. **FIX:** enable deterministic
training (torch.use_deterministic_algorithms + cudnn.deterministic + CUBLAS_WORKSPACE_CONFIG), re-run r12enc+r11fp
deterministically, re-run scripts/r12_decomp.py — only then attribute Δ_init/Δ_protocol and freeze. The fail-fast
did exactly its job: it stopped a confounded attribution.

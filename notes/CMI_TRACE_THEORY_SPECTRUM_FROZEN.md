# CMI-Trace Theory-Spectrum — pre-registration (FROZEN before aggregate)

Three theorem-verifying experiments that give the manuscript's theory teeth on real EEG. **This doc is frozen
before any aggregate is viewed.** Probe results (single-unit QC) may be inspected before freezing the fleet;
the interpretation grid below is written before seeing any aggregate.

- **Base:** `agent/cmi-trace-erasure-oracle` @ `11a80e0e` (origin head; pinned by handoff). Working branch
  `agent/cmi-trace-theory-spectrum` (isolated worktree `CMI_AAAI_theoryspectrum`).
- **Firewall (all three):** strict source-only fit; target trials eval-only; subject/domain = outer unit;
  no query labels touch any subspace/eraser/probe/λ fit; fail-loud on incomplete cells; stamp git SHA +
  config/artifact hashes on every row.
- **Inference unit:** held-out subject (outer LOSO fold); seeds grouped within a fold; subject-cluster
  bootstrap 95% CI over folds (n_boot=10000 for the fleet; 2000 for probes).

---

## 0. CORRECTED reuse/feasibility matrix (the handoff's premise, re-verified against disk)

The handoff said "reuse the confirmed checkpoints." I verified what is actually banked. **All three
experiments are CPU frozen-latent analyses over already-banked artifacts — no GPU retrain is required.**
This is materially lighter than "make-or-break GPU run" framing; the correction is disclosed here.

| Exp | Data | Backbone / d_z | Banked artifact | Head | Compute |
|-----|------|----------------|-----------------|------|---------|
| E1 | BNCI2014_001 (9 folds), BNCI2015_001 (12 folds) | DGCNN `graph_z` (64) | **126 `.audit.npz`** (ERM+CIGL × seeds{0,1,2} × folds) in `CMI_AAAI_cmitrace/results/cmi_trace_p0p1/objective_comparison/{ds}/audit/` | **exact, verified** (replay max\|Δ\|≈1e-6) | CPU |
| E2 | BNCI2014_001 (2a) | TSMNet (210) | 27 dumps `BNCI2014_001_TSMNet_LOSO/subX_erm_lam0_seedY.npz` (`CMI_AAAI_tos/tos_cmi/results/tos_cmi_eeg_frozen/`) | **exact, recovered+verified** (lstsq replay max\|Δ\|≈7e-7) | CPU |
| E2 | BNCI2014_001 (2a) | EEGNet (16) | 27 dumps `BNCI2014_001_EEGNet_LOSO/...` | **NOT exactly recoverable** from dump (Z→logits nonlinear; class-centered lstsq argmax agreement 0.78) → probe-head, labeled | CPU |
| E3 | synthetic spurious-task DGP | linear ridge readout | `tos_cmi/data/spurious_task_dgp.py` (ported from `agent/cmi-trace-dg-oracle`) | fitted linear readout on source Z | CPU |

**Provenance validated for reuse (E1):** `config_sha256 = 002e924...dcd032e` matches the frozen
`configs/cmi_trace_p0p1.yaml` in this worktree byte-for-byte; sidecars carry
`task_head_replay_ok=True`, `source/target/source_val_indices`, `method/dataset/seed/fold/target_subject`.
Reuse proceeds only on this match; a `ProvenanceError` (config hash / replay flag / index mismatch) STOPS
the unit and is reported, never silently relaxed.

### Three disclosed discrepancies vs the handoff
1. **`sim/make_spectrum.py` does not exist** in any branch/commit of the repo. The handoff cites it as an
   already-run synthetic confirmation (corr(τ,Δλ)=+.90, top-dir reliance 0→.058, eff-rank 4.80→3.65). It
   cannot be used to cross-check the new module. E1 is validated instead against the frozen P0.2 audit
   numbers (`ΔRel(k=2)`, eff-rank, energy, alignment) that the banked sidecars feed, plus a self-authored
   synthetic sanity DGP inside the E1 unit test.
2. **E2 backbone-head availability is inverted vs the handoff's worry.** The handoff flagged TSMNet as
   "BLOCKED by torchaudio ABI." In fact TSMNet-2a dumps are banked **and** its head recovers exactly;
   **EEGNet(16)** is the one whose banked dump lacks an exactly-recoverable linear head. E2's exact-head
   clause is therefore clean on TSMNet(210); on EEGNet(16) it is reported with a source-fit **probe head**
   (explicitly labeled representation-reliance, not exact-head) unless the true EEGNet head is re-exported
   (small optional GPU dump-regen, env `icml`; **HELD pending owner**).
3. **Whitened vs raw metric.** The existing `leakage_removal.fit_leakage_subspace` /
   `reliance_audit` machinery uses the **raw** Euclidean subject subspace. E1's Theorem-2 spec requires the
   **whitened** metric (pooled within-class Σ_W^{-1/2}) with directions ordered by subject energy λ.
   `subject_spectrum.py` is a NEW whitened-metric module; it does not modify the frozen raw audit.

---

## E1 — subject-information × exact-head-use spectrum (Theorem 2; the make-or-break)

**Claim.** Amount-only CMI control (CIGL) strips the highest-λ, task-orthogonal (low-τ) subject directions
first, so total CMI and subject effective-rank fall, the top subject direction rotates toward a
task-bearing one, and exact-head reliance on the top direction rises.

**Module:** `cmi/eval/subject_spectrum.py`. Per banked sidecar (ERM and CIGL, matched cell):
1. On SOURCE rows (`d != target_domain`, cross-checked against stored `source_indices`): whiten `graph_z`
   by pooled within-class covariance `Σ_W^{-1/2}` (ledoit-wolf shrinkage; `+εI` floor). In whitened space
   form the between-subject-within-label scatter `S_B = Σ_{y,d} n_{y,d}(\bar z̃_{y,d}-\bar z̃_y)(...)^T`;
   eigendecompose → whitened directions `ũ_1..ũ_r`, eigenvalues (subject-score energy) `λ^{energy}_1≥…`.
2. Per direction `j`:
   - **1-D projection** `t_j = Z̃ ũ_j` (whitened subject coordinate, [N,1]).
   - **`λ_j`** = null-calibrated conditional subject information of `t_j`: run
     `conditional_subject_leakage.flat_conditional_cmi` on `t_j` with a `three_way_support_split`
     (eraser-disjoint) posterior-train/eval split and the fully-retrained within-label permutation null.
     Report `excess_over_null` (primary) and `posterior_kl_nats`.
   - **`τ_j`** = exact-head reliance = `CE(h(Z_removed), Y) − CE(h(Z), Y)` where
     `Z_removed = (I − u_j u_j^T) Z`, `u_j = normalize(Σ_W^{-1/2} ũ_j)` (the RAW-space unit direction whose
     whitened readout is `t_j`), `h` = stored verified head via `replay_head`. Also report bAcc drop.
     Control = same-rank-1 random direction (≥50 seeded spans), matched to the raw metric.
3. Match ERM↔CIGL directions by subspace principal angles (Jonker-Volgenant on |cos| between whitened
   direction sets); pair the SCALARS only (never compare raw axes across models —
   `assert_no_cross_model_axis_compare` discipline); `Δλ_j = λ_j^{CIGL} − λ_j^{ERM}`.

**Primary endpoints (subject-cluster bootstrap 95% CI over folds, seeds grouped):**
- `corr(τ_j^{ERM}, Δλ_j)` — **predict > 0**.
- top-subject-direction (j=1) exact-head reliance, ERM vs CIGL — **predict rises**.
- subject effective rank (participation ratio of λ^{energy}) ERM vs CIGL — **predict falls**.
- top-2 energy concentration and top-direction head-alignment — **predict rise**.

**Output:** `results/spectrum/{dataset}_{seed}_fold{f}.json` (per-direction λ,τ,energy for ERM/CIGL/random),
aggregate `results/spectrum/summary.json`. Fills manuscript **Fig. 2 Panel B** + the `corr(τ,Δλ)` sentence.

**Falsifier (report either way):** if the `corr(τ_j,Δλ_j)` CI includes 0, OR top-direction reliance does
not rise → the leakage–reliance reversal is real but **not** via the predicted spectral mechanism;
Theorem 2's EEG interpretation is down-scoped. This is the one experiment that can refute the mechanism.

---

## E2 — linear removability threshold `r_D` + head-geometry overlaps (Theorem 1)

**Claim.** Complete linear removal needs eraser rank `k ≥ r_D` (rank of the whitened conditional subject
span `S_D`); exact-head safety holds iff `S_D ⊆ ker(WΣ^{1/2})`.

**Module:** `cmi/eval/rank_threshold.py`. Per backbone (TSMNet d_z=210 exact-head; EEGNet d_z=16
probe-head) per fold on frozen `Z_source`:
1. Whitened conditional subject span `S_D`; **`r_D`** = # eigenvalues of `S_B` above an energy threshold
   AND above a within-label subject-permutation floor (both predeclared: energy retains 99%, floor =
   95th-pct permuted eigenvalue).
2. Stored/recovered linear head `W` (fail-closed replay); **`dim(S_D ∩ ker(WΣ^{1/2}))`** and
   **`dim(S_D ∩ row(WΣ^{1/2}))`** via principal angles (angle < 5° counts as intersection).
3. For eraser rank `k=1..r_D` (informed top-k `S_D` projector; same-rank random control): residual
   linear + MLP subject decodability (bAcc), and **exact-head logit change** `‖h(Z)−h((I−P_k)Z)‖`
   (predict ≈0 iff `S_D ⊆ ker(WΣ^{1/2})`).

**Endpoints:** `r_D` per backbone; the two intersection dims; empirical `k*` where residual subject bAcc
reaches chance (predict `k* ≥ r_D`, and `< r_D` leaves residual decodability); logit change removing `S_D`.
**Output:** `results/rank_threshold/{backbone}.json`. Replaces the capacity/latent-dim proxy wording.

**Falsifier:** complete linear removal at `k<r_D`, or nonzero exact-head logit change when
`S_D⊆ker(WΣ^{1/2})` within tolerance → shared-covariance scope violated where it matters; report.

---

## E3 — `K*_subj` on beneficial vs legitimate-use worlds (Proposition 2)

**Claim (exact, squared loss).** `Gain⋆(T)=E[δ_T²](1−K*)`, `K*=2E[r_Tδ_T]/E[δ_T²]`; removal helps iff
`K*<1`. Beneficial (removable) bias → `K*<1`; legitimate-use signal → `K*>1`. CMI amount does not enter the
sign.

**Module:** `cmi/eval/kstar_worlds.py` over `spurious_task_dgp.make_spurious_task_dgp`. Two deployment
worlds share source data, target-X, and candidate subspace `T` (removes the `Z_spur` coordinates):
(i) **beneficial** — target spurious sign flipped (shortcut breaks at deployment);
(ii) **legitimate-use** — target sign preserved (relation holds at deployment). Fit head `h` on source Z
(ridge, squared loss); `h_T` = readout on `(I−P_T)Z`; on each world's deployment set compute
`δ_T=h(Z)−h_T(Z)`, `r_T=Y−h_T(Z)`, `K*=2E[r_Tδ_T]/E[δ_T²]`, `Gain⋆=E[δ_T²](1−K*)`, and the direct
`Gain_direct = R(h)−R(h_T)`.

**Endpoints:** `(E[δ_T²], E[r_Tδ_T], K*, Gain⋆)` per world. **Predict:** beneficial `K*<1, Gain⋆>0`;
legitimate-use `K*>1, Gain⋆<0`. **Output:** `results/kstar/worlds.json`. Upgrades the bias-positive control
from "not always negative" to a Prop-2 endpoint.

**Falsifier (identity is exact → a mismatch is a bug, fix before reporting):**
`|Gain⋆ − Gain_direct| > 1e-8` OR `sign(1−K*) ≠ sign(Gain⋆)` → leakage/estimation bug. QC gate must be
green before any world separation is reported.

---

## Pre-committed interpretation grid (written before any aggregate)

| Exp | Outcome | Reading |
|-----|---------|---------|
| E1 | `corr(τ,Δλ)` CI>0 AND top-dir reliance↑ AND eff-rank↓ | Thm-2 spectral mechanism CONFIRMED on EEG; fills Fig 2B |
| E1 | reversal present (reliance↑/eff-rank↓) but `corr(τ,Δλ)` CI∋0 | reversal REAL but not via predicted spectrum → down-scope Thm-2's EEG claim |
| E1 | no reversal (reliance not↑) | mechanism not present on this EEG; report negative, do not fit Fig 2B |
| E2 | `k*≥r_D` AND logit-change≈0 with `S_D⊆ker(WΣ^{1/2})` | Thm-1 rank threshold + exact-head safety CONFIRMED (per backbone) |
| E2 | complete removal at `k<r_D` OR logit-change≠0 in-kernel | shared-cov scope violated; report scope limit |
| E3 | beneficial `K*<1,Gain⋆>0` AND legit `K*>1,Gain⋆<0`, identity green | Prop-2 VERIFIED; worlds separate on sign(1−K*) |
| E3 | worlds do not separate | Prop-2 not demonstrated here; check DGP strengths, report |

## QC sentinels (probe-gate before any fleet aggregate)
- E1: head replay verified (`replay_ok`), whitening PSD (`Σ_W` eigenvalues > 0 post-shrinkage), firewall
  split == stored indices, permutation null non-degenerate, random-span control finite.
- E2: recovered-head replay ≤ tol (TSMNet) / labeled probe-head (EEGNet); `r_D≥1`; residual decodability
  monotone-ish in k; logit-change computed in class-centered head space.
- E3: `|Gain⋆−Gain_direct| ≤ 1e-8` (exact-identity gate) on BOTH worlds; DGP block indices consistent.

## Compute / provenance
- CPU via `sbatch --partition=CPU` (no login-node aggregation); per-cell append + skip-if-done; address
  cells by real (dataset,method,seed,fold) id, never bench index; coverage guard = proper unique count.
- Two-commit: code+result-rows first, interpretation second. Frozen artifacts read-only. Push only on
  explicit instruction.

---

## APPENDUM (2026-07-22) — adversarial-review fixes, frozen before any aggregate viewed

An independent adversarial verification of the three modules (confirmed the core math of E1 raw-direction
correspondence, λ null reuse, E2 kernel/rowspace, E3 exact identity) drove the following corrections. Only
E1/E2/E3 **probe** QC has been seen; no aggregate has been computed.

1. **E1 firewall now fail-loud.** `subject_spectrum` derives `target_domain` ONLY from `target_indices` and
   RAISES if they are absent (previously fell back to `target_subject`, a human label that can collide with a
   source subject's remapped id and silently fold target rows into the fit). The fleet trusts no cell without
   `target_indices`.
2. **E1 direction matching (supersedes §E1 step 3 wording).** ERM↔CIGL directions are matched by principal
   angle in the **common raw graph_z ambient** (the only coordinate system shared by two separately-trained
   models; each model's whitened basis is model-relative and NOT cross-comparable), i.e. on
   `u_raw = Σ_W^{-1/2} ũ`, via **optimal (Hungarian) assignment**. A second **energy-rank pairing** is
   reported as a robustness variant that assumes no geometric correspondence. Only the SCALAR endpoints (λ,τ)
   are compared across models; raw axis values are never compared. The primary `corr(τ,Δλ)` endpoint must
   hold under BOTH pairings to be called confirmed.
3. **E2 `W_tilde` identity clarified.** `Z̃ W̃ᵀ` equals the class-centered logits up to the per-class bias
   `b_c`; every E2 endpoint uses bias-invariant logit DIFFERENCES, so results are unaffected (docstring
   corrected).
4. **E2 dim-count caveat.** The 5° `dim(S_D∩ker)` / `dim(S_D∩row)` counts are a COARSE descriptor and can
   read 0 despite heavy reliance; the theorem's exact-head-safety endpoint is carried by the continuous
   **logit-change magnitude** (`logit_change_remove_SD_relative`) and the reported **min principal angle**,
   not the dim counts.
5. **Reason-coded failures.** Singular LDA in the E2 sweep → chance (not silent skip); E1 aggregate REFUSES
   any cell with `firewall_ok=False`.

### Watch item for E2 interpretation (from the TSMNet probe, single fold — NOT an aggregate)
On one TSMNet fold, complete linear removal reached chance at `k*=17 < r_D=22` — the FALSIFIER direction for
Theorem 1's `k≥r_D`. This may reflect the permutation floor over-counting weak subject directions in `r_D`
vs. the soft `k*` chance threshold. The aggregate over all 27 dumps (with the pre-registered r_D and k*
definitions) decides; if `k*<r_D` holds robustly, report it as an honest scope refinement, not a pass.

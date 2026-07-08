# S2P_07 — P1 Design Red-Team Record (Phase P1; pre-launch)

**Project S2P — P1.** Design red-team of S2P_06 (agent a7ea257d) + the pre-launch feasibility check. **Verdict:
BLOCKERS_PRESENT — do not launch.** The single fixed-H0 line is **not identifiable** for the diversity claim. This
record + the required revision; launch stays held (`p1_launch_go_nogo.json launch_p1=false`) pending PM choice of
the BL-1 fix.

## BLOCKERS
- **BL-1 — NOT IDENTIFIABLE (the core defect).** At fixed H0=100h, exposure `e = 100/N` is **deterministic** ⇒
  `corr(log N, log e) = −1` (perfect collinearity). The design matrix `[log N, log e]` is rank-deficient, so the
  mixed model **cannot** estimate `β_N` and `β_e` separately — the "exposure covariate" (S2P_06) is **void**. There is
  exactly **one** estimable slope `d(outcome)/d(log N) = ∂diversity − ∂exposure` (**fused**). The primary
  "subject-diversity scaling" claim is **not estimable** from a single fixed-H0 line. (FSR-8C needed fixed-vs-growing
  for the same reason; S2P_06 dropped the second axis.)
  **FIX (PM chooses one):**
  - **(a) Two fixed-H0 lines** (H0 ∈ {100, 200}h): matched-**exposure**-different-N cells (e.g. N=32@100h ≡ 3.125h ≡
    N=64@200h) identify **diversity**; matched-N-different-H0 cells identify **exposure**. Minimal grid that breaks
    collinearity. ~**2× compute**.
  - **(b) Constant-per-subject-cap growing-hours arm** (exposure FIXED, N grows, total grows) = direct
    diversity-at-constant-exposure. Adds a second axis; total-hours grows (disclose).
  - **(c) Reframe/rename** to "**fixed-budget subject-vs-depth tradeoff frontier**" (a legitimate engineering
    question), **delete** the covariate-separation claim, and **strike every "diversity" attribution** from the
    primary claim. No new compute; no diversity claim.
- **BL-2 — under-power + selection bias.** 100h from scratch (~12k windows) for a 12-layer transformer is small; if
  encoders are weak, most downstream cells fail the 0.58 gate (FSR had PhysioNet/BNCI ~0.4–0.5 even on *released*
  encoders), and if gate-pass is N-dependent the slope is **survivorship-biased**.
  **FIX:** (i) **positive-control floor** — the from-scratch encoder must beat a **random-init frozen** CBraMod at the
  max-data cell; else declare **under-powered → STOP** (no "null" claim); (ii) **per-cell convergence gate** (pretrain-
  val loss plateau / common relative-improvement criterion, not fixed epochs); (iii) treat **gate-pass as an analyzed
  outcome** and report every slope **with and without** gated cells (no hard censoring).
- **BL-3 — thresholds not pre-registered** (researcher DoF): STOP-Gini, MDE, seed-SD tolerance all have **no numbers**.
  **FIX:** freeze numeric **MDE** (bAcc-slope units from cluster variance), **Gini ceiling**, **seed-SD tolerance**
  before launch.

## MAJORS
- **MJ-1 — feasibility on the WRONG corpus + POPULATION confound.** `fixed_hours_subset_feasibility.csv` (go/no-go
  "all feasible") is on the **full 33-ch** corpus (13,446). P1 uses **19-common** (6,535). On 19-common, N=32 needs
  ≥3.125h ⇒ **~80 eligible subjects = extreme long-recording clinical (epilepsy-monitoring) patients**, while N=2000
  draws from ~6,500 general subjects ⇒ the N axis confounds diversity, exposure **and clinical population**. Cannot
  nest all N in a common ≥3.125h pool (only ~80 qualify ⇒ can't reach N=128) — that impossibility itself proves the
  single line fuses count/exposure/population. **FIX:** feasibility on 19-common (done in `p1_feasibility_by_cell`);
  report per-cell population drift; restrict to a common eligibility pool (caps max N) **or** disclose population as
  a named confound.
- **MJ-2 — pretrain-val not a fixed common set** ⇒ per-cell val difficulty varies with N ⇒ best-val-loss checkpoints
  on non-comparable scales. **FIX:** one **fixed external pretrain-val set** (constant subjects+exposure, disjoint
  from every training pool); report its loss per cell as a target-free outcome.
- **MJ-3 — no primary outcome / multiplicity control** (3 datasets × 6 metrics). **FIX:** one primary (SHU-MI
  target-bAcc slope on log N), rest secondary/exploratory, **Holm** across the pre-registered family.
- **MJ-4 — sparse-exposure redundancy undisclosed.** N=2000 ⇒ ~6 likely-**contiguous** 30 s windows/subject
  (near-duplicates) ⇒ effective within-subject diversity ≪ 6. **FIX:** pin windowing/overlap policy (constant across
  cells); report contiguity + an **effective (decorrelated) windows-per-subject**.
- **MJ-5 — 3 seeds conflate subset-draw × init** (N=32 subset draws overlap ~40%). **FIX:** factor **subset-seed ×
  init-seed**; ≥5 seeds at N∈{32,2000}.

## MINORS
- **MN-1** L1: pin a **dynamic-range check** (reference-cell L1 off both 0.5 floor and ~0.95 ceiling), **fixed probe +
  PCA rank** across cells, and a **3-way subject-disjoint split** within each downstream dataset. **MN-2** state
  nested vs independent N draws (prefer **nested**). **MN-3** 5 endpoint-dominated points ⇒ require **leave-one-N-out**
  slope-robustness.

## Feasibility bug (pre-launch check, independent of the agent)
The loader enforced the per-subject budget at **whole-recording** granularity ⇒ at high N (cap < one recording) each
subject contributes ≥1 full recording ⇒ **actual total hours balloons** (N=32:116h … N=2000:597h ≠ 100h) ⇒ the
"fixed-hours" axis was silently confounded with total hours. **FIX: window-level budget truncation** (cap_windows =
round(cap·120); truncate within recording). Must land before any run.

## Disposition
**Launch held.** BL-1 requires a PM decision (add a 2nd identifying axis ≈ 2× compute, or reframe away from
"diversity"). BL-2/BL-3/MJ-1..5 + the loader window-budget fix must be resolved and the numeric thresholds frozen
before `p1_launch_go_nogo.launch_p1 = true`. This is the same identifiability lesson as FSR-8C, now caught **before**
any GPU spend.

---

## PM RESOLUTION (2026-07-08) + verification

**PM chose BL-1 fix = TWO fixed-H0 lines (100 h + 200 h), a lean crossed grid** (not the reframe, not a
constant-exposure arm as primary). Rationale (PM): a single fused slope cannot be read as a mechanism — subject
count and per-subject exposure must be separated, isomorphic to the Prior-Decoupled TTA four-branch split. S2P_06 was
rewritten to v2. Resolution of each item, **verified on the real 19-common corpus** (not the 33-ch feasibility table):

- **BL-1 RESOLVED (identifiable).** Grid `100h:{128,512,1024}` + `200h:{256,512,1024,2048}`. Design matrix
  `[1, log N, log e]` is **rank 3/3**, `corr(log N, log e) = −0.918` (was −1). Matched-**exposure** pairs
  (e=0.781/0.195/0.098) identify **diversity**; matched-**N** pairs (N=512/1024) identify **exposure**.
  (`p1_identifiability_matrix.csv`, `p1_exposure_crosswalk.csv`, `p1_feasibility_by_cell_v2.csv`.)
- **MJ-1 RESOLVED + escalated to a disclosed STRUCTURAL confound.** N=32 removed from primary. Real-corpus
  feasibility done on 19-common. Population drift **quantified** (`p1_population_balance_diagnostics.csv`): deep-exposure
  pool (e=0.781) = **713** long-recording clinical subjects (median 4 rec, 11.9% single-rec) vs sparse pools ≈**6,500**
  general (median 1 rec, ~67% single-rec). In this corpus exposure **range** is structurally entangled with clinical
  **population** ⇒ **diversity contrasts (within-pool) are primary/clean; the exposure axis is reported with population
  as a named confound**, never as a clean causal exposure effect. Common eligibility pools + **nested** sampling pin
  each contrast to one pool (`p1_common_eligibility_pools.csv`).
- **BL-2 RESOLVED (pre-registered).** Positive-control floor (must beat random-init frozen source-val +0.02),
  per-cell convergence gate (pretrain-val loss ↓ ≥20%), gate-pass reported as an outcome (slopes with & without
  gated cells), under-powered→STOP (no null claim).
- **BL-3 RESOLVED (frozen thresholds in S2P_06):** total-hours ±1%, per-subject-window ±1, Gini ≤0.02; MDE
  target-bAcc +0.02 / L1 −0.03 / L5 ±0.01; seed-SD ≤0.03, ≥2/3 sign, ≤70% single-seed; conv ≥20%.
- **MJ-2..5 RESOLVED:** fixed common external pretrain-val; one primary + Holm; effective (nested) windows-per-subject
  + windowing policy pinned; subset-seed × init-seed factorial at the two extreme cells (+ seeds {3,4}).
- **Loader window-budget bug FIXED + promoted to a hard launch gate** (total-hours ±1% per cell).

**Compute:** 25 CBraMod pretraining runs (21 primary × seeds{0,1,2} + 4 extreme seed-ext) ≈ 2× v1 (PM-accepted);
optional high-N diagonal (+6) only if clean. **Status: launch still HELD** pending (i) a fresh design red-team of
this revised two-H0 protocol and (ii) explicit PM go after the launch-condition checklist. This record + S2P_06 v2
+ the four new CSVs constitute the revised pre-registration.

---

## v2 RED-TEAM (agent a744a1d5, 2026-07-08) — BLOCKERS_PRESENT (algebra sound, but code≠design + crossed slope unsound)
The two-budget algebra restores identifiability (rank 3/3), and the three matched-exposure diversity **pairs** are
each within-pool clean under the loader. But three blockers make the **primary decomposition claim** uninterpretable
as written, and my go/no-go **over-claimed** launch conditions (corrected below).
- **BL-4 — LOADER ⇄ PROTOCOL GAP (I over-claimed).** `tueg_subject_loader.build_subset` has **no** `pool_min_h`,
  **no** nested `parent_subjects`, **no** fixed common external val, **no** `subset_seed`≠`init_seed`: it does
  `eligible=subj_hours>=cap` + independent `rng.choice` per cell + a per-cell 15% val split. So the claimed nested /
  common-pool / fixed-common-val / factorial-seed machinery **does not exist in code**. Launch conditions 2 & 8 were
  wrongly PASS. FIX = rewrite the loader; re-run the balance/feasibility check against the *real* loader before any
  condition flips to met.
- **BL-5 — DUAL-POOL INCONSISTENCY (crossed regression internally inconsistent).** `100h/N512` must be within-pool
  6485 (diversity, nested in `200h/N1024`) AND from the ≥0.391h common pool 2231 (exposure, vs `200h/N512`) — one
  trained frozen encoder cannot satisfy both. "Restrict in analysis, not retraining" is **impossible** (pretraining
  population is baked in; no TUEG subjects exist at analysis time, only SHU-MI eval). FIX = **PM decision** (retrain
  exposure arms from a common pool ≈ +4 runs, OR delete the exposure coefficient + crossed regression, keep only the
  3 within-pool diversity pairs).
- **BL-6 — POOLED SLOPE POPULATION-CONFOUNDED.** The 3 diversity pairs sit on **different** populations
  (713 clinical / 6485 / 6516 general); pooling them into one `logN` slope reintroduces MJ-1. FIX = primary estimand
  = the 3 within-pool pairwise diversity **differences reported SEPARATELY** with population labels; the pooled
  `outcome~logN+log e` is **NOT** primary (descriptive only, with a population covariate if reported).
- **MJ-6 — identifiable ≠ powered.** VIF=1/(1−0.918²)=**6.36** ⇒ slope SE ×2.52; two budgets only log2 apart;
  MDE +0.02 < seed-SD 0.03 ⇒ leave-one-N-out likely flips a slope. FIX = **PM decision** (add off-diagonal cells /
  3rd budget to cut VIF, OR pre-declare slopes descriptive + leave-one-N-out sign-stability as the go/no-go).
- **MJ-7 — `log e` triple-confounded** (population + within-subject window redundancy [94 contiguous windows/subj at
  e=0.781 vs 12 at e=0.098] + depth). FIX = compute effective decorrelated windows/subj; relabel as
  "budget-depth (population+redundancy confounded)", not per-subject-exposure.
- **MJ-8 — eligibility on summed hours, not floored windows.** `subj_hours>=cap` (sum) but windows floored per
  recording ⇒ a multi-recording clinical subject can fall short of `cap_windows` ⇒ `max−min>1` / Gini>0.02 fails the
  **hard balance gate**, exactly in the e=0.781 clinical cells. FIX = eligibility on `Σ(n_timepoints//6000) ≥ cap_windows`.
- **MN (v2):** stale v1 keys in go/no-go json (fix); per-window z-score may floor L1 (verify dynamic-range check);
  single transfer target (SHU-MI); name the ONE primary statistic (no post-hoc selection under Holm).

**Disposition v2:** launch HELD. **PM decisions required: BL-5 (exposure axis: retrain-common-pool vs drop-exposure-
claim) and MJ-6 (power: add cells vs descriptive slopes).** BL-4/BL-6/MJ-7/MJ-8/MN I will fix in the loader rewrite +
analysis spec once the PM sets scope (the loader shape depends on the two decisions, so implement once, correctly).

---

## PM RESOLUTION v2 (2026-07-08) → P1 = matched-exposure subject-scaling pilot (S2P_06 v3)
**BL-5 = (ii) DROP the exposure coefficient + crossed regression; keep the 3 within-pool matched-exposure NESTED
pairs. MJ-6 = (ii) descriptive slopes + leave-one-pair sign-stability; NO extra off-diagonal cells.** PM rationale:
don't pay compute for a confounded exposure coefficient (population × window-redundancy × depth); narrow P1 to a
clean, interpretable matched-exposure subject-scaling question. Resolution of every open item, **all verified on the
real 19-common corpus**:
- **BL-5 RESOLVED** — no exposure coefficient, no crossed regression (`p1_primary_statistic_spec.json`
  `exposure_coefficient_estimated=false`, `crossed_regression=false`).
- **BL-6 RESOLVED** — primary = 3 within-pool Δ_pair reported **separately** (each population-matched within pair),
  aggregated by **unweighted mean**; no pooled causal diversity slope (`fixed_budget_diversity_claim_allowed=false`).
- **MJ-6 RESOLVED** — descriptive slope only; leave-one-pair-out sign-stability + seed-SD≤0.03 + ≥2/3 sign + ≤70%
  single-seed are the pre-registered stability gate.
- **BL-4 RESOLVED (loader rewritten + VERIFIED)** — `build_matched_exposure_pair`: floored-window common eligibility
  pools, fixed per-contrast subset-seed-invariant pretrain-val (n_val=64, disjoint), nested low⊂high, exact
  per-subject window cap, subset_seed≠init_seed. Balance verified across all 18 cells + val × 3 seeds
  (`p1_loader_balance_verification.csv`): per-subject window max−min=**0**, Gini=**0.0**, total within **0.0002%** of
  quantized budget, nested **True**, disjoint **True**.
- **MJ-8 RESOLVED** — eligibility on floored available windows (`Σ n_timepoints//6000 ≥ cap_windows`), not summed hours.
- **MJ-7 RESOLVED** — exposure axis dropped; window-redundancy (94/23/12 windows/subj) reported per pair, matched
  within pair; population diagnostics per pair.
- **MN RESOLVED** — go/no-go de-staled; single primary statistic named; L1 dynamic-range + z-score-floor check
  pre-registered.
**Window-quantization note:** per-subject budget = cap_windows·30 s; nominal-vs-quantized e gap (−0.27/+1.87/−2.40%
for A/B/C) is a per-pair constant shared by both cells ⇒ cancels in the within-pair contrast.

**Grid (18 runs):** A(100h/N128 vs 200h/N256, e≈0.78) · B(100h/N512 vs 200h/N1024, e≈0.20) ·
C(100h/N1024 vs 200h/N2048, e≈0.10), seeds{0,1,2}. Exposure-contrast cell (200h/N512) and N=32 NOT in P1; high-N
diagonal deferred (not launched). **Status: launch HELD** pending (i) a re-red-team of this narrowed protocol showing
no BLOCKER and (ii) explicit PM go on the launch checklist.

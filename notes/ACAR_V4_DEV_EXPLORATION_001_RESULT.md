# ACAR v4 — DEV EXPLORATION #001: RESULT = `V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE`

```
STATUS    : NON-BINDING / POST-V3 DEV_STOP / OLD-SEVEN DEV ONLY (exploratory / model-selection)
VERDICT   : V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE
LOCKBOX   : NOT CONSUMED      EXTERNAL ARM : NOT APPROACHED      FREEZE : NOT WRITTEN
DATE      : 2026-06-29
```

First real V4 Phase-1 exploratory run on the seven old DEV cohorts, per `notes/ACAR_V4_DEV_EXPLORATION_RUN_PLAN.md`.
SLURM job `875699` on `nodecpu05` (env `eeg2025`; kernel `Linux-6.12.0-211.7.3.el10_2`); **exit 0**, elapsed **139 s**;
code at `e9760e6` (clean worktree). `manifest_sha256 = 8f5ccb288c7ca93857acd593ff6ec31bb4965c522a20d24b289ab9800bb970da`;
`v4_oof_records_sha256 = 7c7bcd51…`; `score_family_registry_sha256 = fe5a1f58…` (matches the run plan).
Artifacts: `results/acar_v4_dev_exploration_001/` (manifest.json 425 KB + RESULT.json + run.sbatch + console.log + slurm out).

## Faithfulness check (the adapter reuses v3 correctly)
The v2-replay comparator (`run_c0.red_router`) gives **PD 0.0449 + SCZ 0.1521 → macro 0.0985**, which **exactly matches
v3's C0 macro red (0.0985)** from `DEV_STOP.json`. EVAL pools: PD 230 subjects / 406 batches (44 fallback), SCZ 225 / 450
(0 fallback). This confirms `acar/v4/real_adapter.py` derives ΔR + the bit-for-bit v2 features and the C0 comparator from
the v3 single-execution substrate faithfully.

## Headline (exploratory)
**Control-first calibration unlocks usable adaptation coverage that v2/v3 could not.** Where v2/v3 collapsed to ~1 %
coverage (all-action conformal upper bound, `q ≫ |ΔR|`), V4 — calibrating the *executed* policy's subject risk directly
on a finite λ grid (LTT) — achieves **16–86 % coverage with positive deployed NLL reduction beating the v2-replay
comparator on BOTH diseases, out-of-fold**. **14 of 90** both-disease configs pass the pre-registered G0–G6.

### Passing configs (disease-macro red; both diseases pass G0–G6)
| macro red | score family | policy | loss | PD cov/red/harm | SCZ cov/red/harm |
|-----------|--------------|--------|------|------------------|-------------------|
| 0.419 | shift_margin | benefit_ranked / direct_selective | mean | 0.86 / 0.486 / 0.46 | 0.54 / 0.352 / 0.33 |
| 0.228 | shift_margin | benefit_ranked / direct_selective | positive | 0.28 / 0.161 / 0.24 | 0.29 / 0.295 / 0.23 |
| 0.205 | shift_margin | safe_set | positive | 0.24 / 0.133 / 0.22 | 0.25 / 0.277 / 0.25 |
| 0.158 | shift_margin | benefit_ranked / direct_selective | harm_indicator | 0.20 / 0.116 / 0.15 | 0.25 / 0.201 / 0.21 |
| 0.151 | n_eff_neg | safe_set / benefit_ranked / direct_selective | positive | 0.33 / 0.172 / 0.31 | 0.37 / 0.130 / 0.27 |
| 0.145 | n_eff_neg | safe_set / benefit_ranked / direct_selective | mean | 0.56 / 0.141 / 0.44 | 0.54 / 0.149 / 0.38 |
| 0.131 | shift_margin | safe_set | harm_indicator | 0.18 / 0.111 / 0.13 | 0.16 / 0.151 / 0.23 |

Comparators (G3 primary = v2_replay; macro v2_replay = 0.0985; best_fixed PD 0.123 / SCZ 0.060). All passing configs
beat the v2-replay macro. The **safest** passing family is `shift_margin + benefit_ranked/direct_selective +
harm_indicator` (max disease harm 0.21, macro red 0.158); the **highest-utility** is `shift_margin + benefit_ranked +
mean` (macro red 0.419, but harm up to 0.46 — it adapts aggressively).

### Direction-C decomposition (per disease)
| disease | true-oracle ceiling | score-union ceiling | c0 v2-replay | c0 best-fixed |
|---------|--------------------|--------------------|--------------|---------------|
| PD | 0.981 | 0.506 | 0.045 | 0.123 |
| SCZ | 1.055 | 0.816 | 0.152 | 0.060 |

Label-free observables capture ~52 % (PD) / ~77 % (SCZ) of the oracle benefit — a real but non-vacuous information gap;
the calibrated policies then deliver a further-reduced fraction (the calibration gap).

## Caveats (binding for any write-up — do NOT overclaim)
1. **DEV / model-selection only.** 14 of 90 both-disease configs pass; selecting the best among 90 is post-hoc model
   selection on DEV — a clear **selection-bias** risk. This is a **candidate for POSSIBLE freeze**, NOT a confirmed or
   external result. Confirmation requires a NEW frozen V4 protocol (one candidate) + held-out / external data.
2. **Two DISTINCT harm metrics — do not conflate** (see `ACAR_FROZEN_v4.md` §2a). The "harm" column in the tables above
   is **`harm_among_adapted`** = P(ΔR>0 | adapted) (descriptive: of the batches we adapt, what fraction were harmful);
   it is 0.15–0.46. The **LTT-CONTROLLED** loss is **`L_harm_all`** = subject-mean over ALL batches of
   1[adapted ∧ ΔR>0], budget 0.10 — for the SAFE candidate its EVAL value is only ≈ PD 0.03 / SCZ 0.05 (= coverage ×
   harm_among_adapted), well inside budget. So "control" here means the *all-batch* harmful-adaptation rate is held low
   (≈3–5 %); it is NOT a claim that few of the *adapted* batches are harmful. The aggressive `mean`-loss configs adapt up
   to 86 % with harm_among_adapted ~0.46; net red is positive because beneficial batches outweigh harmful ones.
3. **Not external validation.** The old seven cohorts are development data. No lockbox consumed; external Arm B not
   approached; `ACAR_FROZEN_v4.md` not written.
4. **Provenance.** Exact OOF coverage enforced (every subject & batch EVAL once); subject-macro weighting; fold-local
   CAL→EVAL (λ* from CAL only); the run is reproducible from the pinned registry/config/record digests.

## Mechanism (why V4 > v2/v3)
v3's failure was the *control object*: an all-action simultaneous conformal upper bound forced `q ≫ |ΔR|`, so the router
abstained (~1 % coverage). V4 calibrates the **deployed policy's subject risk** directly (finite-grid LTT) — the harm
signal that v2/v3 only *measured* now becomes *usable* coverage. This is the pre-registered control-first hypothesis
(`notes/ACAR_V4_DESIGN_DRAFT.md` §4) holding on real DEV data.

## Next step (per the run plan; GATED — awaiting decision)
A candidate is found, so the next step is to **DRAFT `ACAR_FROZEN_v4.md`**, freeze **ONE** candidate family (e.g. the
safest `shift_margin + harm_indicator`, or the high-utility `shift_margin + mean` — a decision), and tag a new V4
protocol. Whether any external / held-out data may then be consumed is a SEPARATE later decision. **NOT** external Arm B
now. No threshold/seed/loss/registry change to chase a better number (that would be post-hoc).

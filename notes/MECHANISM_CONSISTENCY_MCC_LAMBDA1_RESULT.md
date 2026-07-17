# MCC λ=1.0 lever — RESULT: specific mechanism, λ-INERT, DG-null; estimator CAUSE UNDIAGNOSED (present a fork)

Real EEG, full LOSO. λ=1.0 rerun (results/cmi_trace_mcc_lambda1) vs the λ=0.25 round (results/cmi_trace_mcc):
63/63 bundles, 189/189 arms, all 3 arms/bundle. **Warm-up hashes 63/63 IDENTICAL to the λ=0.25 round → λ is the
SOLE difference.** full_encoder_trainable 63/63 (2228/2228 params). This note reflects a 3-agent adversarial panel
(wf_ced0df6f) that reproduced every number and CORRECTED the first-pass label + next-step (both over-claimed a
cause). Manuscript FROZEN; graded, not a stop.

## Observation (solid, correctly measured — freeze this)
| dataset | n | amp_BC = ΔB−C(λ1 − λ0.25) [95% LCB] p | dir B−A vs ERM @λ1 [LCB] | dU_spec1 = U(B₁)−U(C₁) [95%CI] p | grad_ratio@λ1 | src drop |
|---|---|---|---|---|---|---|
| BNCI2014_001 | 9 | −0.00012 [−0.00085] p=0.617 | +0.00091 [+0.00004] | +0.0009 [−0.0037] p=0.402 | 0.519 | +0.0013 |
| BNCI2015_001 | 12 | +0.00056 [−0.00043] p=0.189 | −0.00032 [−0.00078] | −0.0003 [−0.0036] p=0.560 | 0.826 | +0.0029 |

- The true-vs-shuffle mechanism separation **persists and is significant at BOTH λ** (B−C ≈ +0.0013; p=0.018 /
  0.001) — the mechanism is real, specific, controllable (strict E stays falsified).
- **It did NOT amplify with 4× λ**: amp_BC not significant on either dataset even one-sided (p=0.617/0.189); no
  alternative metric (|B−C| growth p=0.465/0.086, relative Δ, pooled ratio 0.909/1.422) reaches p<0.05; datasets
  DISAGREE in sign of amplification. It stays ~10× below the 0.01 DG-relevant scale.
- **Geometry vs ERM at λ=1.0 is still ~0 / inconsistent** (dir B−A +0.0009 barely-LCB>0 on 2014, −0.0003 on 2015).
- **DG utility is NULL**: dU_spec1 straddles zero, p=0.40/0.56. Raising λ 4× did not move DG.
- **No damage / no collapse** (source drop <0.003; eff rank / contrast norm healthy).

## Label (honest, panel-corrected): SPECIFIC_BUT_LAMBDA_INERT — estimator cause UNDIAGNOSED
NOT `MINIBATCH_estimator_limited` (my first pass) and NOT `GLOBAL_..._amplifies`:
- amp_BC=NS **rules OUT** "λ amplifies geometry" → GLOBAL is unsupported.
- But amp_BC=NS does **NOT establish** an estimator cause. Three panel-verified reasons the "minibatch estimator is
  noise-limited → EMA fixes it" diagnosis is NOT earned:
  1. **grad_ratio is definitional in λ** (`grad_ratio = lam_t·‖∇L_MCC‖/‖∇L_CE‖`, run_mcc_arms.py) — so "0.13→0.69"
     is guaranteed by 4× λ, not an independent "the update got stronger and still failed." The λ0.25 grad_ratio was
     never stored; "0.13" is back-inferred ÷4. (The absolute λ=1.0 grad_ratio ≈ 0.69 does show MCC is a substantial,
     applied fraction of the update — that part stands.)
  2. **Noise-domination is UNMEASURED**: the gradient diagnostic logs only the FIRST batch/epoch, so batch-to-batch
     variance — the entire premise of "EMA fixes a high-variance estimator" — is absent from the data; and grad_cos
     = cos(∇CE, ∇MCC) is POSITIVE in 100% of 63 bundles (mean 0.38–0.49), i.e. a task-redundant gradient, NOT a
     cancellation/noise-dominated one.
  3. **Geometry is decoupled from DG**: corr(dir_BC, dU_shuf) ≈ −0.05 across subjects — the geometry axis the
     mechanism DOES control does not predict the true-vs-shuffle DG utility. So even a cleaner/larger geometry effect
     has no demonstrated DG payoff.

## Disposition — present a FORK; gate a full round behind ONE cheap discriminator (no committed EMA)
DG stayed null under the strongest allowed single lever, and the geometry axis is decoupled from DG — so no DG claim
is on the table and no full GPU round is justified on faith. Options for the PM (a decision, not mine to pick):
1. **Estimator change** (source-only EMA / memory-bank subject-class prototypes) — motivated ONLY if the estimator
   is actually variance-limited, which is currently unmeasured.
2. **Risk-weighted MCC** (weight consistency by source-only predictive instability) — motivated if the geometry
   axis is simply the wrong target (consistent with corr≈0 to DG).
3. **Drop the global-consistency axis** — motivated if it is both un-amplifiable AND DG-decoupled.

**Recommended cheap discriminator BEFORE any full round** (panel consensus): a **full-batch / very-large-batch
source-consistency MCC** (the low-variance-estimator limit) + **batch-to-batch gradient-variance logging** (log
every batch, not just the first). Decision rule: EMA/prototypes earn a full round ONLY if the low-variance limit
**(a) scales B−C toward the 0.01 scale AND (b) that geometry then tracks DG**; if the low-variance limit also fails
to scale B−C → the mechanism saturates → risk-weighted or drop; if it scales B−C but geometry still doesn't track
DG → risk-weighted. This directly tests the estimator hypothesis instead of assuming it.

HELD: M2 selector, learned projector, TTE, CMI, EMA-estimator round, risk-weighted round — pending the PM's fork
choice and the cheap discriminator. Manuscript FROZEN. Scientific line ACTIVE.

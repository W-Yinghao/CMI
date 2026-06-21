# DualPC: AAAI candidate objective for joint `P(z)` and `P(y|Z)` control

Date: 2026-06-16

## Motivation

The old `dual` objective co-minimizes

```text
I(Z;D|Y) + I(Y;D|Z)
```

This is mathematically meaningful, but under label shift the two terms fight:

```text
I(Z;D|Y) - I(Y;D|Z) = I(Z;D) - I(Y;D).
```

For an AAAI-ready method, the algorithm should match the intended deployment story more directly:

- optimize the representation distribution `P(z)` so the feature space is domain-stable;
- optimize the predictor distribution `P(y|Z)` so the decision boundary is domain-stable;
- remove label-prior effects first, otherwise `P(z)` alignment can erase labels and raw `I(Y;D|Z)` reads label-prior/calibration artifacts.

## New method: `dualpc`

Config:

```bash
--configs dualpc:<lambda_pz>:<gamma_py>
```

Training objective after the CPU ablation pass:

```text
L = CE(Y|Z)
    + lambda * I_w(Z;D|Y)
    + gamma  * [ JS_w(h_full(Y|Z,D), h0(Y|Z,D)) - tau ]_+
```

where `w_i = pi*(y_i) / pi_{d_i}(y_i)` is the GLS label-shift weight and `pi*` is uniform. The task CE is
not automatically GLS-weighted; if that is desired, use the explicit `--label_correct` flag. CPU synthetic
tests showed automatic GLS task CE hurts target balanced accuracy in the all-three DGP.

Term meanings:

- `CE(Y|Z)`: trains the classifier on the empirical source risk, preserving the current stable behavior.
- `I_w(Z;D|Y)`: weighted posterior-KL estimate using `q(D|Z,Y)` against the post-GLS domain marginal. Because GLS makes `Y` independent of `D` under the reference measure, forcing `I_w(Z;D|Y)->0` implies alignment of the induced reference mixture
  `P_w(Z|D)=sum_y pi*(y)P(Z|Y=y,D)`. This is the factorized `P(z)` control and avoids the label-erasure failure mode of direct marginal `I_w(Z;D)`.
- `JS_w(h_full,h0)`: prediction-distribution consistency between the full domain decoder and the
  intercept-only decoder `h0(Y|Z,D)=u(Z)+b_D`. This is the training loss for the `P(y|Z)` side. The older
  CE residual `R_res,w(Y;D|Z)=CE(h0)-CE(h_full)` is retained as a diagnostic/evaluation CMI probe, but CPU
  concept-gate tests showed it is a weaker training loss: it can raise held-out residual leakage while
  improving target accuracy.

Implementation:

- `cmi/train/trainer.py`: method `dualpc`, included in `ALL_METHODS`.
- `dualpc` internally uses the raw source sampler because the CMI terms already apply explicit GLS weights;
  adding class/domain-balanced resampling would impose a second implicit prior.
- `dualpc_marginal` is retained as the direct marginal `I_w(Z;D)` ablation.
- Decoder-gate defaults are method-specific: `dualpc` / `dualpc_marginal` default to `tau=0.0` so the JS
  `P(Y|Z)` term is active by default; `dualc` keeps `tau=0.02` for the CE-residual Route-C gate. An explicit
  `--dec_margin` still overrides the method-specific default.
- `cmi/methods/regularizers.py`: added `log_pd_ref` and weighted marginal KL.
- `cmi/run_loso.py`: parses `dualpc:a:b` and `dualpc_marginal:a:b`.
- `cmi/run_cross_dataset.py`: parses `dualpc:a:b` and records in-loop diagnostics for MI Protocol C.
- `cmi/run_scps_crossdataset.py`: uses the same generic config parser and passes through to `train_model`.
- Result JSONs now include `inloop_dec`, `inloop_dec_loss`, and `train_sampler` per fold; summaries include
  mean `inloop_dec` and `inloop_dec_loss`.
- Step-A `q(D|Z,Y)` accuracy is now recorded for `dual/dualc/dualpc/dualpc_marginal`, not only for older
  CMI methods. This matters because decoder CMI quality depends on the auxiliary posterior actually fitting.
- `run_loso.py` and `run_scps_crossdataset.py` can optionally report source-only permutation-null calibrated
  residual decoder diagnostics via `--decoder_null_perms N`. The permutation shuffles domain labels within
  each class, preserving label-domain counts while breaking domain-dependent decision-boundary structure.
- `cmi/eval/metrics.py` now includes `marginal_leakage_probe`, a frozen-backbone `q(D|Z)` diagnostic for
  direct `P(z)` leakage. Runners record raw `marginal_leakage_kl` and GLS-weighted
  `marginal_leakage_kl_rw`, plus domain-accuracy advantage fields.
- `leakage_probe(..., reweight=True)` now reports GLS-weighted conditional leakage `leakage_kl_rw`,
  matching DualPC's training-side `I_w(Z;D|Y)` reference. Raw `leakage_kl` is retained as a diagnostic.
- `decoder_leakage_probe` now reports both the held-out CE residual diagnostic (`decoder_cmi_res`) and the
  held-out JS consistency diagnostic (`decoder_js_res`), each with raw and GLS-weighted runner summaries.
- Decoder probes now carry validity metadata: `decoder_valid` is true only when the source probe has at
  least two decoder domains and every decoder domain spans at least two classes. JSON summaries include
  `decoder_valid_frac`, `decoder_valid_n`, `decoder_min_domain_classes`, and `decoder_single_class_frac`.
  Paper-facing decoder tables should use the `*_valid_mean` fields, e.g. `decoder_js_res_valid_mean`, so
  invalid source-domain splits appear as JSON `null` instead of contaminating the aggregate.
- `cmi/run_lambda_select.py` supports source-only selection for `dualpc`:
  `--method dualpc --lams 0 0.05 0.1 --gammas 0.05 0.1`. The ERM candidate (`0,0`) is always present.
  It also supports `--select_rule guarded_probe`, which chooses among source-val-competitive candidates by
  a source-only probe penalty over GLS conditional leakage, GLS marginal `P(z)` leakage, and held-out JS
  `P(y|Z)` consistency. Raw conditional leakage and the CE residual `P(y|Z)` probe are still recorded as
  diagnostics. If a decoder
  source split is invalid, the selector records `selection_probe_valid=false` and does not use the JS term
  as a valid tie-breaker.
  `--dec_margins` adds source-only selection of the decoder gate, avoiding target-tuned fixed margins.
  `--final_probe_epochs` optionally records the same source-only GLS/JS probes after retraining the selected
  and fixed-ERM models on all source data, so readiness tables can distinguish selection-time candidate
  probes from final-model probes.
- `run_loso.py`, `run_cross_dataset.py`, `run_lambda_select.py`, and `run_scps_crossdataset.py` support
  `--device cpu|cuda|auto`.
- `analysis/dualpc_readiness.py` summarizes synthetic, runner, and source-selection JSONs into one
  DualPC readiness table. It checks current `I_w(Z;D|Y)` (`cond_kl_rw` / `leakage_kl_rw`), GLS marginal
  `P(z)` leakage, JS `P(Y|Z)` consistency, decoder validity, and source-only selector terms.
  It also expands input globs internally and reports `selector_final` rows when final retrain probes are
  present. When a runner JSON contains ERM/LPC baselines, it emits `runner_compare` rows with DualPC-vs-baseline
  deltas for accuracy, GLS conditional leakage, GLS marginal `P(z)` leakage, and JS `P(Y|Z)`. Synthetic
  source-selector smoke files are treated as selector path checks, not as method gates.
- `analysis/dualpc_paper_summary.py` is the cross-run aggregate for paper-profile JSONs. It groups runner
  outputs across files/seeds into method tables and DualPC-vs-ERM/LPC comparison rows, and separately
  aggregates source-selector final probes, selected-config histograms, and target bAcc vs fixed ERM.
- `analysis/dualpc_decision_gate.py` consumes the final readiness and paper-summary JSONs and emits the
  conservative mechanical decision `HEADLINE_READY`, `NEEDS_REVIEW`, `NOT_READY`, or `PENDING`. The paper
  protocol gate requires PASS synthetic evidence for `null_prior`, `concept`, and `all_three`, requires
  all four formal comparison tasks to have both ERM and `lpc_prior` baselines, and requires final-probe
  selector summaries for both LOSO datasets.
- `scripts/dualpc_protocol.py` generates the paper/smoke command protocol without submitting jobs. It
  covers synthetic gates, LOSO instrumentation/null-safety runs, source-only guarded selection, SCPS
  cohort-domain runs, the final readiness summary command, the aggregate paper-summary command, and the
  headline decision gate. The selector commands enable
  `--final_probe_epochs` in both smoke and paper profiles. The generated command list starts with
  `scripts/dualpc_regression_checks.py`, a fast CPU-only guard for DualPC defaults, GLS weighting,
  all-ones weighted-loss equivalence, decoder validity, readiness/paper-summary parsing, protocol/SLURM
  command fields, preflight status parsing, and a tiny end-to-end `train_model(..., method="dualpc")` path.
- `scripts/dualpc_slurm_plan.py` writes a non-submitting SLURM package for the paper protocol:
  `scripts/dualpc_paper_tasks.tsv`, `scripts/dualpc_paper_array.slurm`, and
  `scripts/dualpc_paper_post.slurm`. `scripts/dualpc_paper_status.py` is the matching read-only preflight
  and completion checker.

## Difference from existing variants

| method | representation side | predictor side | label correction | intended role |
|---|---|---|---|---|
| `lpc_prior` | `I(Z;D|Y)` | none | `pi_y(D)` target | stable main baseline |
| `dual` | `I(Z;D|Y)` | raw `I(Y;D|Z)` | optional / partial | tension ablation |
| `dualc` | `I(Z;D|Y)` | gated CE residual `R_res` | GLS probes | concept diagnostic / CE-residual ablation |
| `dualpc` | **factorized `I_w(Z;D|Y)` -> reference `P(z)`** | **gated JS consistency to h0** | **GLS probes + KL** | candidate main algorithm for joint `P(z)` and `P(y|Z)` |
| `dualpc_marginal` | direct `I_w(Z;D)` | gated JS consistency to h0 | GLS probes + KL | negative/control ablation |

## Submission gates

Before making `dualpc` a headline method, it needs three checks:

1. CPU smoke: finite loss/probs on synthetic tensors and tiny LogCov/LOSO path.
2. Null safety: on MUMTAZ and cross-site SCZ/PD, `dualpc` should match `lpc_prior` accuracy and not raise residual decoder CMI.
3. Positive control: on injected concept shift or PD medication-state subject response, the JS consistency
   loss should control the residual decoder diagnostic without collapsing `P(z)` alignment.

If these fail, keep `dualpc` as an ablation and make `lpc_prior + Route-C diagnostic` the paper's main claim.

## Paper Protocol

The reproducible command manifest is generated, not hand-written:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/dualpc_protocol.py \
  --profile paper --device cuda --seeds 0 1 2 \
  --out-dir results/dualpc_protocol_paper \
  --loso-datasets BNCI2014_001 MUMTAZ \
  --scps-conditions PD SCZ \
  --write scripts/dualpc_paper_cmds.sh
```

Do not run this while GPUs are full. The script only writes commands. The generated protocol includes:

- fast CPU-only DualPC regression checks;
- synthetic current-code gate over `null_prior`, `concept`, and `all_three`;
- LOSO runs with `erm`, `lpc_prior`, `dualc`, `dualpc`, and `dualpc_marginal`;
- source-only guarded selection with fixed candidate grid `lambda in {0,0.05,0.1}`, `gamma in {0.05,0.1}`;
- SCPS cohort-domain runs for PD and SCZ with `--domain cohort --dec_domain cohort`;
- decoder null calibration with `--decoder_null_perms 20 --decoder_null_quantile 0.95`;
- final `analysis/dualpc_readiness.py results/dualpc_protocol_paper/*.json` summary;
- final `analysis/dualpc_paper_summary.py results/dualpc_protocol_paper/*.json` aggregate across files/seeds;
- final `analysis/dualpc_decision_gate.py` headline/ablation decision JSON.

For local path checks without GPU:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/dualpc_protocol.py \
  --profile smoke --device cpu --seeds 0 \
  --out-dir results/dualpc_protocol_smoke \
  --loso-datasets BNCI2014_001 --scps-conditions PD
```

This prints a CPU-only command list and does not submit SLURM.

Current non-submitting SLURM package:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/dualpc_slurm_plan.py \
  --profile paper --device cuda --seeds 0 1 2 \
  --out-dir results/dualpc_protocol_paper \
  --loso-datasets BNCI2014_001 MUMTAZ --scps-conditions PD SCZ \
  --task-file scripts/dualpc_paper_tasks.tsv \
  --array-script scripts/dualpc_paper_array.slurm \
  --post-script scripts/dualpc_paper_post.slurm \
  --max-parallel 2

/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/dualpc_paper_status.py \
  --task-file scripts/dualpc_paper_tasks.tsv \
  --array-script scripts/dualpc_paper_array.slurm \
  --post-script scripts/dualpc_paper_post.slurm
```

The current package has 20 array tasks (`0-19%2`): one regression guard, one synthetic gate, and 18
GPU tasks over seeds 0/1/2 for LOSO, source-only selection, and SCPS. The post-stage decision command uses
`--min-comparison-tasks 4 --min-selector-tasks 2 --required-baselines erm lpc_prior --required-synthetic-groups null_prior concept all_three`.
Preflight reports all plan checks PASS, with one READY no-output regression task, 19 PENDING task JSONs, and three PENDING post
JSONs. The preflight policy checks also PASS: synthetic tasks include `all_three`, `concept`, and
`null_prior`; runner tasks cover `loso:BNCI2014_001`, `loso:MUMTAZ`, `scps:PD`, and `scps:SCZ`; every
runner task includes `erm` and `lpc_prior`; selector tasks cover both LOSO datasets. It has
not been submitted. When GPU quota is available, submit with:

```bash
jid=$(sbatch --parsable scripts/dualpc_paper_array.slurm)
sbatch --dependency=afterok:$jid scripts/dualpc_paper_post.slurm
```

## Current CPU evidence

Quick production-trainer synthetic smoke:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
  --quick --out results/dualpc_synthetic_quick.json
```

This completed on CPU. It proves the end-to-end path, JSON logging, weighted `P(z)` probe, and residual
`P(y|Z)` probe work. The first pass used the now-demoted direct marginal objective plus automatic GLS task
CE; it was a useful negative result:

| DGP | method | target bAcc | `I_w(Z;D)` | residual `P(y|Z)` |
|---|---:|---:|---:|---:|
| covariate+label | ERM | 78.0 | 0.100 | 0.053 |
| covariate+label | LPC | **83.7** | 0.100 | 0.060 |
| covariate+label | DualC | 83.5 | 0.099 | 0.060 |
| covariate+label | DualPC | 81.4 | 0.109 | 0.058 |
| all-three | ERM | 61.5 | 0.080 | 0.048 |
| all-three | LPC | **62.8** | 0.079 | 0.047 |
| all-three | DualC | 62.1 | 0.079 | 0.047 |
| all-three | DualPC | 61.2 | **0.078** | 0.050 |

Reading: direct marginal `P(z)` plus automatic GLS CE is runnable but not a headline method.

After this, `dualpc` was revised to the factorized/no-auto-GLS-CE objective above, and the direct marginal
variant was renamed `dualpc_marginal`. A slightly longer CPU all-three check:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
  --dgps all_three --n 250 --seeds 2 --epochs 30 --probe_epochs 80 --warmup 10 --bs 128 \
  --out results/dualpc_factorized_noce_allthree_quick.json
```

| method | target bAcc | `I_w(Z;D)` | residual `P(y|Z)` |
|---|---:|---:|---:|
| ERM | 72.4 | 0.147 | 0.042 |
| LPC | **75.3** | **0.117** | 0.045 |
| DualC | 74.9 | 0.119 | 0.044 |
| DualPC factorized | 74.9 | 0.119 | 0.044 |
| DualPC marginal | 73.1 | 0.121 | 0.044 |

Current reading: `dualpc` should be treated as the paper-facing version of the stable Route-C objective,
with the theorem phrased through GLS factorization of reference `P(z)`. `dualpc_marginal` and automatic GLS
task CE are negative ablations, not main methods. The next gate is real-data null safety against
`lpc_prior/dualc`, not more tuning of the direct marginal penalty.

After the concept/null gate below, the `dualpc` decoder-side training loss was updated from the CE residual
`CE(h0)-CE(h_full)` to a gated JS consistency loss between `h_full(Y|Z,D)` and `h0(Y|Z,D)`. The CE residual
is still reported as the `P(y|Z)` diagnostic.

CPU gate for JS-consistency `dualpc`:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
  --n 300 --seeds 2 --epochs 40 --probe_epochs 100 --warmup 10 --bs 128 --n_inner 2 \
  --lam 0.1 --gamma 1.0 --dec_margin 0.0 --dgps null_prior concept \
  --out results/dualpc_gate_js_concept_null_quick.json
```

Key means:

| DGP | method | target bAcc | `I_w(Z;D)` | residual `P(y|Z)` | reading |
|---|---:|---:|---:|---:|---|
| null-prior | ERM | 76.7 | 0.0356 | 0.0040 | baseline |
| null-prior | DualPC-JS | 76.7 | **0.0337** | **0.0036** | null-safe, no target loss |
| concept | ERM | 52.7 | 0.1298 | 0.0829 | positive-control residual present |
| concept | DualC CE-res | **63.1** | 0.1559 | 0.1723 | target improves but both probes worsen |
| concept | DualPC-JS | 53.5 | **0.1221** | **0.0801** | both probes improve without hurting target |

Reading: the JS consistency loss is the safer paper-facing `P(y|Z)` optimizer. The old CE residual route can
be used as a positive-control/diagnostic ablation, but it is not clean enough to claim simultaneous
`P(z)` and `P(y|Z)` optimization.

CPU all-three smoke after the JS update:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
  --quick --dgps all_three --out results/dualpc_js_allthree_quick.json
```

This remains path-healthy: DualPC-JS target bAcc `54.6` vs ERM `53.9`, `I_w(Z;D)` `0.098` vs `0.101`;
residual `P(y|Z)` rises by `0.0045`, which stays within the current `gate_py_not_raised` tolerance.

Tiny real-data CPU/LogCov smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/infres/yinwang/anaconda3/envs/icml/bin/python -m cmi.run_loso \
  --dataset BNCI2014_001 --backbone LogCov \
  --configs erm:0 lpc_prior:0.1 dualpc:0.1:0.05 dualpc_marginal:0.1:0.05 \
  --epochs 2 --warmup 1 --n_inner 1 --bs 32 --max_subjects 3 --resample 64 \
  --device cpu --out results/dualpc_losologcov_smoke.json
```

This completed without GPU/SLURM and writes a probability sidecar. It is a path check, not a performance
claim:

| method | target bAcc | leakage KL | in-loop reg | in-loop dec | qDomAcc | sampler |
|---|---:|---:|---:|---:|---:|---|
| ERM | 37.2 | 0.651 | 0.000 | 0.000 | 0.0 | classbal |
| LPC | 39.5 | 0.652 | 0.262 | 0.000 | 87.5 | classbal |
| DualPC | 38.5 | 0.654 | 0.256 | -0.008 | 89.1 | raw |
| DualPC marginal | 38.5 | 0.653 | 0.233 | -0.008 | 89.2 | raw |

Reading: the real-data CPU path is healthy and `dualpc` diagnostics are no longer zeroed. This smoke was run
before the JS decoder-loss update, so the bAcc numbers are too short-trained and not used as method-quality
evidence.

After the JS update, two 2-subject/1-epoch CPU path smokes completed:

- `results/dualpc_js_loso_smoke.json` records both `inloop_dec` (CE residual diagnostic, `0.0024`) and
  `inloop_dec_loss` (JS training loss, `1.0e-4`).
- `results/dualpc_validity_loso_smoke.json` verifies the held-out JS fields and validity metadata:
  `decoder_js_res=2.56e-4`, `decoder_js_res_rw=2.51e-4`, `inloop_dec_loss=1.0e-4`,
  `decoder_valid_frac=0.0`, and `decoder_js_res_valid_mean=null` because each 2-subject LOSO source fold
  has only one source domain.

As with the earlier 2-subject smoke, this should only be read as a path check; the important improvement is
that the degenerate decoder probe is now explicitly marked invalid.

Non-degenerate `P(z)` probe smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/infres/yinwang/anaconda3/envs/icml/bin/python -m cmi.run_loso \
  --dataset BNCI2014_001 --backbone LogCov --configs dualpc:0.1:0.05 \
  --epochs 1 --warmup 1 --n_inner 1 --bs 32 --max_subjects 3 --resample 64 \
  --device cpu --out results/dualpc_pz_probe_3subj_smoke.json
```

This completed on CPU and checks that the true-data runners now log both conditional leakage and direct
`P(z)` leakage. Summary: target bAcc `39.6±11.1`, conditional KL `0.642`, raw marginal `P(z)` KL `0.629`,
GLS-weighted marginal `P(z)` KL `0.630`, Step-A `q(D|Z,Y)` accuracy `70.4%`. It is still a 1-epoch smoke,
so the numbers should only be used to validate instrumentation.

Non-degenerate decoder-valid CPU smoke after the method-specific DualPC gate default:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/infres/yinwang/anaconda3/envs/icml/bin/python -m cmi.run_loso \
  --dataset BNCI2014_001 --backbone LogCov --configs dualpc:0.1:0.05 \
  --epochs 1 --warmup 1 --n_inner 1 --bs 32 --max_subjects 3 --resample 64 \
  --device cpu --out results/dualpc_tau_valid_loso3_smoke.json
```

This completed on CPU with `train_dec_margin=0.0`, `decoder_valid_frac=1.0`, `decoder_valid_n=3`,
`decoder_js_res_valid_mean=0.00199`, `decoder_js_res_rw_valid_mean=0.00188`, and `inloop_dec_loss=4.59e-4`.
It is still only a 1-epoch instrumentation check, but it verifies that the default `dualpc` run now keeps
the JS `P(Y|Z)` side active and reports valid-only decoder summaries on real data.

Decoder residual null-calibration smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/infres/yinwang/anaconda3/envs/icml/bin/python -m cmi.run_loso \
  --dataset BNCI2014_001 --backbone LogCov --configs dualpc:0.1:0.05 \
  --epochs 1 --warmup 1 --n_inner 1 --bs 32 --max_subjects 2 --resample 64 \
  --device cpu --decoder_null_perms 1 --decoder_null_quantile 0.8 \
  --out results/dualpc_decoder_null_smoke.json
```

This completed on CPU and produced summary fields:

- `decoder_cmi_res_null_q`
- `decoder_cmi_res_excess`
- `decoder_cmi_res_rw_null_q`
- `decoder_cmi_res_rw_excess`
- `decoder_js_res_null_q`
- `decoder_js_res_excess`
- `decoder_js_res_rw_null_q`
- `decoder_js_res_rw_excess`

For full experiments, use more permutations, e.g. `--decoder_null_perms 20 --decoder_null_quantile 0.95`.
Do not interpret the 1-permutation smoke values statistically.

Source-only guarded selector smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/infres/yinwang/anaconda3/envs/icml/bin/python -m cmi.run_lambda_select \
  --dataset BNCI2014_001 --backbone LogCov --method dualpc \
  --lams 0 0.05 --gammas 0.05 --dec_margins 0.0 0.02 \
  --select_rule guarded_probe --select_tolerance 0.05 --select_probe_epochs 2 \
  --epochs 1 --select_epochs 1 --warmup 1 --n_inner 1 --bs 64 \
  --max_subjects 4 --resample 32 --device cpu \
  --out results/dualpc_lamsel_js_guarded_smoke.json
```

This completed on CPU before the method-specific default-label cleanup. It selected `dualpc:0.05:0.05` on
three folds and `dualpc:0.05:0.05:tau=0` on one fold. Under the current label convention, default
`dualpc` tau is `0.0` and is omitted from the config label; non-default tau is explicit, e.g.
`dualpc:0.05:0.05:tau=0.02`. The JSON records every candidate's `source_val_bacc`, `dec_margin`,
`select_cond_kl`, `select_cond_kl_rw`, `select_pz_kl_rw`, `select_py_js_rw`, `select_py_res_rw`, and
`selector_penalty`. The
1-epoch result is only a path check; the important point is that `lambda/gamma/tau` can now be selected
without target labels while explicitly scoring both sides of the DualPC objective.

Source-only synthetic selector alignment smoke:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
  --n 120 --seeds 1 --epochs 3 --probe_epochs 3 --select_epochs 2 --select_probe_epochs 2 \
  --warmup 1 --bs 64 --dgps concept --source_select --select_methods dualpc \
  --select_rule guarded_probe --select_lams 0 0.05 --select_gammas 0.05 \
  --out results/dualpc_synthetic_select_aligned_smoke.json
```

This completed on CPU and verifies that the synthetic harness now matches the production selector. The
current aligned selector uses GLS conditional leakage (`select_cond_kl_rw`), GLS marginal `P(z)` leakage
(`select_pz_kl_rw`), and JS `P(y|Z)` consistency (`select_py_js_rw`). A follow-up CPU smoke after this
alignment is stored at `results/dualpc_condrw_selector_smoke.json`; candidate records include
`select_cond_kl`, `select_cond_kl_rw`, `select_pz_kl_rw`, `select_py_js_rw`, `select_py_res_rw`,
`selection_probe_valid`, and `selector_penalty`. This is a logic check, not a performance claim.

Production guarded selector smoke after the GLS-conditional alignment:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/infres/yinwang/anaconda3/envs/icml/bin/python -m cmi.run_lambda_select \
  --dataset BNCI2014_001 --backbone LogCov --method dualpc \
  --lams 0 0.05 --gammas 0.05 --select_rule guarded_probe --select_tolerance 0.05 \
  --select_probe_epochs 1 --final_probe_epochs 1 --epochs 1 --select_epochs 1 \
  --warmup 1 --n_inner 1 --bs 64 --max_subjects 4 --resample 64 --device cpu \
  --out results/dualpc_finalprobe_lamsel_smoke.json
```

This completed on CPU. All selected candidate probes and all final retrain probes were valid. The JSON records
`final_selected_cond_kl_rw`, `final_selected_pz_kl_rw`, `final_selected_py_js_rw`, matching `final_erm_*`
fields, and target bAcc for the selected and ERM retrain models. `analysis/dualpc_readiness.py
results/dualpc_finalprobe_lamsel_smoke.json` reports `PASS=8`, `WARN=0`, `FAIL=0`, including four
`selector_final` rows. This validates the production source-only selector and final-model probe path; the
1-epoch numbers are not performance evidence.

Current GLS-conditional synthetic gate:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/dualpc_validation.py \
  --n 240 --seeds 2 --epochs 25 --probe_epochs 50 --warmup 8 --bs 128 --n_inner 2 \
  --lam 0.1 --gamma 1.0 --dgps null_prior concept \
  --out results/dualpc_condrw_gate_js_concept_null_quick.json
```

This completed on CPU and includes `cond_kl_rw`, so it supersedes older synthetic gates for the current
selector/probe semantics. DualPC passes the current synthetic readiness gates: in the concept DGP, target
bAcc improves by `+0.046`, `cond_kl_rw` decreases by `-0.0046`, GLS `P(z)` KL decreases by `-0.0024`, and
JS `P(Y|Z)` decreases by `-0.00045`; in null-prior, target is unchanged and probe increases stay below the
null-safety tolerance while `P(z)` and JS improve slightly. These are still CPU quick results, not final
multi-seed paper tables.

Current runner/readiness smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/infres/yinwang/anaconda3/envs/icml/bin/python -m cmi.run_loso \
  --dataset BNCI2014_001 --backbone LogCov --configs dualpc:0.1:0.05 \
  --epochs 1 --warmup 1 --n_inner 1 --bs 32 --max_subjects 3 --resample 64 \
  --device cpu --out results/dualpc_condrw_loso3_smoke.json

/home/infres/yinwang/anaconda3/envs/icml/bin/python analysis/dualpc_readiness.py \
  results/dualpc_condrw_gate_js_concept_null_quick.json \
  results/dualpc_condrw_loso3_smoke.json \
  results/dualpc_finalprobe_lamsel_smoke.json \
  --out-json results/dualpc_readiness_current.json
```

This completed on CPU. The LOSO smoke records `leakage_kl_rw=0.632`, `marginal_leakage_kl_rw=0.620`,
`decoder_js_res_rw_valid_mean=0.00194`, and `decoder_valid_n=3`, proving the current runner emits all
paper-facing DualPC probes. The current readiness bundle reports `PASS=15`, `WARN=2` (ERM baseline rows),
`FAIL=0`; it now includes final selected/ERM source-only probes in the selector evidence.
`analysis/dualpc_paper_summary.py` on the same bundle writes `results/dualpc_paper_summary_current.json`;
it reports selector-final aggregate `PASS=1`, `WARN=0`, `FAIL=0`. The runner comparison table is empty for
this smoke because `results/dualpc_condrw_loso3_smoke.json` only contains `dualpc`; formal protocol runner
JSONs include ERM/LPC baselines and will populate DualPC-vs-baseline deltas.

Fast regression checks:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/dualpc_regression_checks.py
```

This is CPU-only and data-free. It currently reports `ALL PASS (11 checks)` over the DualPC tau defaults,
GLS label-shift weights, all-ones weighting equivalence for the auxiliary losses, decoder-validity summaries,
readiness parsing (including `runner_compare` rows), paper-summary parsing, headline decision-gate behavior,
protocol command generation, SLURM package generation, preflight status parsing, and a tiny
`train_model(..., method="dualpc")` run that verifies raw sampler forcing, `tau=0`, and JS-side
`inloop_dec_loss` logging. It is a regression guard, not performance evidence.

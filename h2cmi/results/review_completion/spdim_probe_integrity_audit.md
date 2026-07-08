# SPDIM Probe Integrity Audit

Status: PASS for bounded BNCI2014_001 expansion.

This is a CPU/artifact audit of the existing P4 official-SPDIM feasibility
probe. No GPU job was launched for this audit.

## Inputs Re-opened

- protocol: `h2cmi/results/review_completion/spdim_probe_protocol.md`
- results: `h2cmi/results/review_completion/spdim_probe_results.csv`
- audit: `h2cmi/results/review_completion/spdim_probe_audit.md`
- command log: `h2cmi/results/review_completion/COMMAND_LOG.md`
- stdout: `h2cmi/results/review_completion/slurm/logs/spdim-probe-888850.out`
- stderr: `h2cmi/results/review_completion/slurm/logs/spdim-probe-888850.err`
- runner: `h2cmi/run_spdim_probe.py`
- external official code: `/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39`

## Provenance And External Code

- runner commit recorded by the probe: `54e855b18f765e6e6f043df146a261266383733e`
- external official SPDIM SHA: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- external import policy: `PYTHONPATH` only; no `spdnets` or SPDIM source tree exists under `h2cmi/`.
- pretrained weights: not used. The runner has no `torch.load` / pretrained-model path, and the protocol explicitly forbids the official BNCI2015_001 13-channel weights.
- third-party vendoring: not detected.

## Exact Splits

Dataset `BNCI2014_001` loaded as `X=(2592, 22, 500)`, subjects `1..9`, sessions `0,1`.

| target | source subjects | source n | source labels | target session | adapt n | adapt labels, audit only | eval n | eval labels | source idx sha256 | adapt idx sha256 | eval idx sha256 |
|---:|---|---:|---|---:|---:|---|---:|---|---|---|---|
| 1 | 2 3 4 5 6 7 8 9 | 2304 | [1152, 1152] | 0 | 72 | [36, 36] | 72 | [36, 36] | `e465e339076ddda8a860f1cf45b08ae0f29c0edb6d28ce5a64253851137c54f0` | `a795e5bc03ed5f1f953e4539af97da663030b8db4faa31de506ad5cdd98478a0` | `bd9a2c8fbf053af0e055841916ee360363cf260d55f96945c7db0b7a4c89e968` |
| 9 | 1 2 3 4 5 6 7 8 | 2304 | [1152, 1152] | 0 | 72 | [36, 36] | 72 | [36, 36] | `f74bb71ff9113b5d5601f4f9b6ab0846db25b819402a53cdb2738628b3a9b928` | `aed590dfda5daa6583dd5d3b464b7647b1b347b119357bd1ccd65cfb987dc2a7` | `c6b7acaf45ac2d38863c0e83fbae97f79d679f0cd069d98c5d809ef6693e0b21` |

The audit-only adaptation label counts above were computed after the run from
the frozen H2CMI arrays. They were not available to the adaptation code path.

## Target-label Leakage Check

No target-label leakage was detected.

- Source training uses all non-target subject labels only.
- Source validation uses source labels only.
- Target subject selection is deterministic metadata-only: first and last sorted subject IDs.
- Target adaptation dataset is constructed with dummy zero labels.
- `domainadapt_finetune`, `get_information_maximization_geodesic`, and `get_information_maximization_bias` receive the target adaptation loader; the official IM routines optimize prediction entropy/diversity and do not use true target labels.
- Target labels enter only in the evaluation dataset used by `_predict` / `_metrics` after each source-only/refit/adaptation mode has completed.
- No target-label-based target subsampling, early stopping, model selection, or subject filtering was used.

## Accuracy Equality

Ordinary accuracy equals balanced accuracy in all 8 rows because both evaluation
sets are exactly balanced: 72 evaluation trials per target subject, with 36
class-0 and 36 class-1 trials. With equal class counts,

```text
ordinary accuracy = (36*r0 + 36*r1) / 72 = (r0 + r1) / 2 = balanced accuracy
```

where `r0` and `r1` are the two class recalls.

## Identical Values

The repeated `0.8055555555555556` value is the exact fraction `58/72`, not a
default. Evidence:

- Source-only rows differ from adaptation rows (`target 1: 0.500`, `target 9: 0.833`), so the runner is not emitting a constant fallback metric.
- Failure rows would have `status=failed` and a failure reason; all 8 rows are `status=ok`.
- Each method has measured nonzero `eval_seconds`; adaptation methods have nonzero `adapt_seconds`.
- The source model hashes differ across target folds, matching distinct LOSO source sets.
- Slurm stdout shows source training epochs for both targets (`epoch 0`, `10`, `19`).
- The metrics are produced only by `_metrics(y_true, y_pred)` after `_predict(...)` runs on the evaluation loader.

Target 1: RCT, SPDIM-geodesic, and SPDIM-bias all report `0.8055555555555556`.
The geodesic optimizer found `parameter_t=1.009957832338944`, close to the RCT
identity parameter; the bias path also left the hard argmax decisions unchanged
relative to RCT. The identical bAcc/acc/macro-F1 values are therefore consistent
with identical hard predictions after the RCT refit.

Target 9: RCT and SPDIM-geodesic both report `0.8055555555555556`, but their
macro-F1 values differ (`0.804953560371517` vs `0.8054054054054054`) and the
geodesic optimizer found `parameter_t=0.977700700282926`. The equal bAcc/acc is
therefore an equal-accuracy outcome on the balanced 72-trial eval set, not a
complete row clone. SPDIM-bias differs (`0.8194444444444444`).

## Prediction And Logit Hashes

The P4 probe runner did not persist prediction tables, logits, prediction
hashes, or logits hashes. Consequently, no prediction/logit hash is available
for the existing 8 rows. This is not an integrity failure for P4 because hashes
were not part of the approved feasibility-probe output contract. It is a
requirement for P5, and the bounded BNCI2014_001 expansion must persist
`prediction_hash` and `logits_hash` for every method/subject row.

## Verdict

- `probe_pass`: true
- `target_label_leakage_detected`: false
- `fallback_prediction_detected`: false
- `split_mismatch_detected`: false
- `pretrained_weight_detected`: false
- `identical_result_explained`: true
- `approve_bnci001_expansion`: true

P5 is approved only for BNCI2014_001, source seed 0, W1-style LOSO, all 9
BNCI2014_001 target subjects, and the four methods already used in P4:
source-only TSMNet, RCT, SPDIM geodesic, and SPDIM bias.

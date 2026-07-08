# SPDIM BNCI001 Provenance Reconciliation

Status: FAIL for confirmatory/full-run escalation. The BNCI2014-001 seed-0
SPDIM expansion remains usable only as an exploratory bounded run.

No GPU job was launched for this reconciliation.

## Inputs

- launch_head: `a749ba953b7f625cf713ab6673a569264c38af6a`
- result_commit: `6ebcb91feab99cfc057e69a166b64c506fea1d05`
- recorded_runner_diff_sha256_at_launch:
  `870ca4e40c417a0fbd80ee63e9833e3cc22bb727388fa350f7eb21d748e9ca82`
- recorded source:
  `h2cmi/results/review_completion/spdim_bnci001_summary.json`

## Reconstructed Diff Check

Command:

```bash
git diff a749ba953b7f625cf713ab6673a569264c38af6a..6ebcb91 -- h2cmi/run_spdim_probe.py | sha256sum
```

Observed:

```text
251bd1c67b38adb777c7e9851e6f7a70c1007f0603fa3bd0b5dcdb8b0a2609da  -
```

Verdict: mismatch.

The result commit's runner diff is not byte-identical to the dirty runner diff
recorded at Slurm launch. The visible post-launch difference is that the final
commit also changes the CSV writer to `lineterminator="\n"` so the committed
P5 result CSV is LF-normalized and passes `git show --check`. That line was not
part of the launch-time runner diff recorded by the Slurm stdout hash.

Therefore the launch-time dirty runner state cannot be certified as exactly
reconstructed by `6ebcb91`.

## Result Scope

The BNCI2014-001 result artifact is still parseable and internally complete:

- results_csv: `h2cmi/results/review_completion/spdim_bnci001_results.csv`
- results_csv_sha256:
  `b0ccaaa05c00ca9209224a728d39bbdc71b17c7989c28673257fb89886e43a7e`
- observed_rows: `36`
- expected_rows: `36`
- row_status: `36/36 ok`

However, because the provenance reconciliation hash failed, this run is marked
`exploratory_only` and is not accepted as a confirmatory template for a full W1
SPDIM sweep.

## Stop Rule Applied

Per P5.1 instructions:

- result digest was not created;
- full W1 SPDIM protocol draft was not created;
- no new GPU job was launched;
- no Cho2017, Lee2019-MI, extra seed, geometry stress, orthogonal-score, or TeX
  work was performed.

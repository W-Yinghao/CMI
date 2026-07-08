# Run Provenance

## `pwd`

```
/home/infres/yinwang/CMI_AAAI_qxu
```
## `git rev-parse --show-toplevel`

```
/home/infres/yinwang/CMI_AAAI_qxu
```
## `git branch --show-current`

```
exp/h2cmi-wave0-mechanism
```
## `git rev-parse HEAD`

```
283832710c93d56cca45b27682286e64e37a4034
```
## `git status --short`

```
?? results/h2cmi/
```
## `python --version`

```
Python 3.9.13
```

Note: raw Wave0/W1 artifacts are gitignored under `results/h2cmi/`; this audit package is additive.

## Final Commit Reconciliation

- `analysis_base_commit=283832710c93d56cca45b27682286e64e37a4034`
- `artifact_commit=29a219596a976c1300443dfcd4e890d6010db7e9`
- `hygiene_digest_commit=483ff8c01c95d3083864f0f0c8b8243d271bdb75`
- `current_remote_head=483ff8c01c95d3083864f0f0c8b8243d271bdb75`
- `artifact_branch=exp/h2cmi-wave0-mechanism`
- `remote_tracking_branch=origin/exp/h2cmi-wave0-mechanism`
- `responsibility_branch_head=09e92499ecec9d245e12d92f2c3b355e8e1b93d1`

Interpretation: the analysis began from HEAD `2838327`; the generated review-completion package and off-diagonal stress artifacts were committed afterward as `29a2195`; the hygiene and writer-facing digest pass was committed afterward as `483ff8c`.

## Raw Artifact Inputs

- Existing frozen W1/W2/V2P artifacts under `results/h2cmi/`, including `p0_w1_*`, `p0_w2_*`, `wave0_v2p/`, `wave1_geom/`, and report JSON files.
- Existing committed result reports under `h2cmi/results/`, including `wave1_geom.report.json` and Wave0 checksum manifests.
- New off-diagonal raw rows under `results/h2cmi/review_completion_offdiag/*.jsonl`.

## Generated Artifact Outputs

- Review-completion package: `h2cmi/results/review_completion/`
- Manuscript digest: `h2cmi/results/review_completion/MANUSCRIPT_NUMBERS_READY.md`
- Validation gate report: `h2cmi/results/review_completion/VALIDATION_STATUS.md`
- Branch reconciliation: `h2cmi/results/review_completion/BRANCH_RECONCILIATION.md`

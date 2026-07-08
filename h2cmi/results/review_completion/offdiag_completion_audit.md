# Off-Diagonal Geometry Completion Audit

Checked after SLURM jobs left the queue.

## Job status

`squeue` no longer lists `888742`, `888743`, or `888784`; the off-diagonal
array and post-analysis jobs have completed from the scheduler queue's
perspective.

`sacct` was attempted but the cluster accounting database refused the
connection:

```text
sacct: error: Problem talking to the database: Connection refused
```

Therefore final accounting is verified from `squeue`, job logs, artifact
existence, parse checks, row counts, and checksums rather than from `sacct`.

## Raw artifact validation

| file | rows | bad JSON rows | units | seeds | perturbation rows |
|---|---:|---:|---:|---|---|
| `results/h2cmi/review_completion_offdiag/w1offdiag_bnci001.jsonl` | 108 | 0 | 9 | 0,1,2 | 27 each |
| `results/h2cmi/review_completion_offdiag/w1offdiag_bnci004_01.jsonl` | 108 | 0 | 9 | 0,1,2 | 27 each |
| `results/h2cmi/review_completion_offdiag/w1offdiag_bnci004_23.jsonl` | 108 | 0 | 9 | 0,1,2 | 27 each |
| `results/h2cmi/review_completion_offdiag/w1offdiag_bnci004_24.jsonl` | 108 | 0 | 9 | 0,1,2 | 27 each |
| `results/h2cmi/review_completion_offdiag/w1offdiag_lee2019.jsonl` | 648 | 0 | 54 | 0,1,2 | 162 each |

Total post-analysis input rows: 1,080. Total unit x perturbation cells after
seed averaging: 360.

## Post-analysis artifact

`h2cmi/results/review_completion/geometry_capacity_offdiagonal_results.csv`
exists and parses as CSV with 128 rows. Bootstrap cluster is
`(dataset,pair,subject)` and source seeds are averaged within unit before
bootstrap.

## SHA-256

```text
ae0ab83404613ecbe9a4827cbc1533e48a6b52197b834272fa9a936c031d1262  results/h2cmi/review_completion_offdiag/w1offdiag_bnci001.jsonl
79a5350fee6e0969963de2ee70cff30969930457b85eef659fa07eeff0856e16  results/h2cmi/review_completion_offdiag/w1offdiag_bnci004_01.jsonl
b630b7015b377b5921a98398f262a8209d40a313363c07433ca5846671ba1487  results/h2cmi/review_completion_offdiag/w1offdiag_bnci004_23.jsonl
ddcc0a0c23301f3d68e6f4a5724b8d1a8854e7e5faa0dbd9b02ef30ee3df95d3  results/h2cmi/review_completion_offdiag/w1offdiag_bnci004_24.jsonl
90ee02f63dd8129d443170e3ec122d844cf28a08762f45f9523794c6921cdc7c  results/h2cmi/review_completion_offdiag/w1offdiag_lee2019.jsonl
3026234a35d09d57df317e18e06adbc79e8120fa084ae28b5d5cc57c4faac267  h2cmi/results/review_completion/geometry_capacity_offdiagonal_results.csv
```

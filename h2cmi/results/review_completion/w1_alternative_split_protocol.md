# W1 Alternative Split Protocol

- status: PASS
- no GPU work; no model rerun.
- recommended_split_family: `class_stratified_half`
- recommended_split_all_datasets_pass: `True`
- recommended_split_no_single_class_eval: `True`

## Candidate Families

| split family | BNCI2014_001 | Cho2017 | Lee2019_MI | all datasets pass | labels used for split construction |
|---|---|---|---|---|---|
| class_stratified_half | `True` | `True` | `True` | `True` | `True` |
| interleaved_odd_even_trial | `True` | `True` | `True` | `True` | `False` |
| session_block_aware_stratified | `True` | `True` | `True` | `True` | `True` |
| leave_one_run_out | `True` | `False` | `False` | `False` | `False` |

## Recommended Replacement Split

`class_stratified_half` is recommended because it passes all target subjects in BNCI2014_001, Cho2017, and Lee2019_MI and makes evaluation balanced-accuracy meaningful for every target. It uses target labels only to construct and freeze the benchmark split before any model run. Target labels remain unavailable to adaptation algorithms at run time.

Implementation rule: within each target subject's earliest W1 session, sort trials by the frozen loader order, split each MI class into first-half adaptation and second-half evaluation, then concatenate the two class-specific halves. Adaptation and evaluation trial IDs are disjoint.

The existing runners are not compatible as-is because they call `contiguous_split`; a rerun would require a split-function change and PM approval.

## Validation Gates For Any Future Run

- `n_adapt > 0` and `n_eval > 0` for every target.
- evaluation contains both MI classes for every target.
- adaptation contains both MI classes for every target.
- adaptation/evaluation trial IDs are disjoint.
- split is frozen before model execution.
- target labels are never provided to adaptation operators at run time.

## Red Team Review

- The protocol designs a replacement only; it does not approve or launch a replacement run.
- It explicitly discloses label use for benchmark split construction.
- It blocks old `contiguous_split` reuse for confirmatory W1.

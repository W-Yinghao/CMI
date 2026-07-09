# Command Log

- Inspected worktree/branch/provenance.
- Verified committed checksum manifests: `review_p0.sha256`, `w03.sha256`, `w04.sha256`, `w02_w05.sha256`, `w1g.sha256`.
- Read `h2cmi/AGENT_ONBOARDING.md`, `h2cmi/results/WAVE0_INDEX.md`, `h2cmi/results/WAVE0_EVIDENCE_PACKET.md`, and Wave0/W1 report JSON/CSV files.
- Generated this package from existing raw JSONL/report artifacts; no new training/adaptation experiment was launched.
- Added and submitted review-completion off-diagonal W1.geometry stress jobs via SLURM:
  `h2cmi/results/review_completion/slurm/w1offdiag_probe.slurm`,
  `h2cmi/results/review_completion/slurm/w1offdiag_array.slurm`, and
  `h2cmi/results/review_completion/slurm/w1offdiag_post.slurm`.
  Submitted IDs on 2026-07-08: probe `888741`, dependent array `888742`,
  dependent post-analysis `888743`.
- Added CPU watcher `888784` with script
  `h2cmi/results/review_completion/slurm/w1offdiag_watch.slurm`; it writes
  `h2cmi/results/review_completion/offdiag_watch_status.md`.
- Per PM gate, performed P0-P3 artifact hygiene only: branch reconciliation,
  provenance final-commit clarification, status/blocker consistency fixes,
  manuscript-number digest, and validation gate report. No GPU was launched.
- Per PM P0.5, performed CPU-only provenance/wording patch and validation:
  stale wording search returned no matches; review-completion CSV/JSON artifacts
  parsed successfully; required provenance fields were present. Committed and
  pushed `5fbc030` with message `Record hygiene head and tighten review-completion wording`.
- Per PM P4, prepared bounded official-SPDIM feasibility protocol using external
  `fightlesliefigt/SPDIM` checkout at
  `1b0de0ccd4c48a4ff28f087b866a0b671b029c39` under
  `/home/infres/yinwang/.cache/h2cmi_external/`, without vendoring third-party
  code. Frozen target subjects are BNCI2014_001 subjects `1` and `9`.
- CPU dry-run gate for P4 passed before GPU submission:
  `python -m h2cmi.run_spdim_probe --dry-run --device cpu --subjects 1`
  with the external SPDIM checkout on `PYTHONPATH`.
- Submitted bounded P4 GPU job `888850` via
  `h2cmi/results/review_completion/slurm/spdim_probe.slurm`.
  The job ran from protocol commit `54e855b18f765e6e6f043df146a261266383733e`
  and external SPDIM SHA `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`.
- P4 output validation passed: `spdim_probe_results.csv` contains 8 ok rows
  (targets `1` and `9`; modes `source_only`, `rct_refit`,
  `spdim_geodesic`, `spdim_bias`), ordinary accuracy and balanced accuracy are
  populated for every row, and CSV SHA-256 is
  `2637cb87095ddd188153516f8a9567ec83fcc71fffccf2e7146955fb6789ca58`.
  `spdim_probe_audit.md` reports PASS. Per `SLURM_MONITORING_POLICY.md`,
  `sacct` is not used on this server; completion was accepted by final
  `squeue` absence, empty stderr, existing stdout, artifact parse checks,
  expected row count, and checksum validation.
- Per PM P4.1, performed a CPU/artifact integrity audit of the existing SPDIM
  feasibility probe and wrote
  `h2cmi/results/review_completion/spdim_probe_integrity_audit.md` plus
  `h2cmi/results/review_completion/spdim_probe_integrity_audit.json`. The gate
  passed: no target-label leakage, fallback prediction, split mismatch,
  pretrained-weight use, or third-party vendoring was detected; the repeated
  `0.8055555555555556` values were explained as real 58/72 evaluation outcomes.
- Per PM P5, ran only the bounded BNCI2014_001 expansion: seed `0`, all 9
  BNCI2014_001 target subjects, and methods `source_only_tsmnet`, `rct`,
  `spdim_geodesic`, `spdim_bias`. Submitted Slurm job `888854` with
  `sbatch h2cmi/results/review_completion/slurm/spdim_bnci001.slurm`. No full
  W1 SPDIM sweep, Cho2017, Lee2019-MI, extra seeds, geometry stress,
  orthogonal-score implementation, or TeX edit was performed.
- P5 completion validation used `squeue` only. Final `squeue -j 888854`
  returned no job row. Stderr was empty, stdout existed, the result CSV parsed,
  all 36 expected rows were present and `status=ok`, every row carried
  `prediction_hash` and `logits_hash`, and the result CSV SHA-256 is
  `b0ccaaa05c00ca9209224a728d39bbdc71b17c7989c28673257fb89886e43a7e`.
  Machine-readable summary:
  `h2cmi/results/review_completion/spdim_bnci001_summary.json`.
- Per PM P5.1, performed CPU-only provenance reconciliation for the dirty
  launch runner diff. The reconstructed
  `git diff a749ba953b7f625cf713ab6673a569264c38af6a..6ebcb91 -- h2cmi/run_spdim_probe.py`
  hash was
  `251bd1c67b38adb777c7e9851e6f7a70c1007f0603fa3bd0b5dcdb8b0a2609da`,
  which does not match the launch-recorded
  `870ca4e40c417a0fbd80ee63e9833e3cc22bb727388fa350f7eb21d748e9ca82`.
  Applied the stop rule: BNCI001 SPDIM is marked `exploratory_only`; no result
  digest, full W1 protocol draft, GPU job, Cho2017, Lee2019-MI, extra seed,
  geometry stress, orthogonal-score, or TeX work was performed.
- Per PM P5.2A, added a strict clean-worktree guard for future official SPDIM
  probe/expansion launches and policy file
  `h2cmi/results/review_completion/SPDIM_CLEAN_RUN_POLICY.md`. The runner now
  records launch commit, `git status --porcelain`, clean-worktree status,
  runner/config checksums, external SPDIM commit, environment name, command
  line, and Slurm job id. CPU guard smoke in the current dirty worktree refused
  before dataset load/training with `git status --porcelain is nonempty`.
- Per PM P5.2B, launched the clean BNCI2014_001 seed-0 bounded rerun only after
  the P5.2A guard commit `a8b93682c152a428f9689f9941efaff486606336` had been
  pushed. The clean launch used detached worktree
  `/home/infres/yinwang/CMI_AAAI_spdim_clean_a8b9368`, with empty
  `git status --porcelain=v1 --untracked-files=all`, and submitted Slurm job
  `889192` via
  `sbatch h2cmi/results/review_completion/slurm/spdim_bnci001_clean.slurm`.
  Completion validation used `squeue` only: final `squeue -j 889192` returned
  no job row, stderr was empty, stdout existed, `spdim_bnci001_clean_results.csv`
  parsed with 36/36 `ok` rows, and the result CSV SHA-256 is
  `4b8e17542220511baddb41bdfc412dde68b38a214e813affdd7348c99d4d6338`.
  The clean rerun matches exploratory P5 on acc, bAcc, and prediction hashes for
  all 36 rows; logits byte hashes differ for all 36 rows and are disclosed in
  `spdim_bnci001_clean_compare_to_exploratory.csv`.
- Per PM P6A, prepared the official SPDIM W1 seed-0 same-split protocol and
  CPU-only dry-run gate. Added `h2cmi/run_spdim_w1_seed0.py` as a clean-guarded
  single-process W1 controller plus Slurm launcher
  `h2cmi/results/review_completion/slurm/spdim_w1_seed0.slurm`; no GPU was
  launched before the P6A commit. The dry-run used the `icml` environment and
  external SPDIM checkout
  `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`, loaded all target subjects for
  `BNCI2014_001`, `Cho2017`, and `Lee2019_MI`, instantiated official TSMNet for
  each dataset shape, and ran one CPU forward pass per dataset without target
  labels in the adaptation loader. The gate passed with expected rows
  `BNCI2014_001=36`, `Cho2017=208`, `Lee2019_MI=216`, total `460`, and
  `approve_gpu_run=true`. Cho2017's exact contiguous split yields single-class
  evaluation blocks under this W1 split; this is disclosed in
  `spdim_w1_seed0_protocol.md` and the dry-run audit rather than hidden.
- Per PM P6B, launched the approved W1 seed-0 SPDIM expansion after P6A commit
  `6a6e5b7758fe3d5130f87ea274be32ba6598dcd7` had been pushed. Initial
  monolithic Slurm job `889522` clean-launched and produced 108 `ok` rows
  (BNCI2014_001 all targets plus Cho2017 targets 1-18), then was cancelled to
  avoid a slow single-job tail and replaced by clean non-overlapping shards.
  First shard attempts `889841`-`889848` were cancelled before writing any CSV
  rows because their wrapper emitted a bad external-SHA echo into stderr.
  Corrected shard jobs `889849`-`889856` clean-launched from detached worktrees
  at the same pushed commit, with empty `git status --porcelain` blocks and
  external SPDIM SHA `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`; they completed
  Cho2017 targets 19-52 and Lee2019_MI targets 1-54.
- P6B completion validation used `squeue` only. Final `squeue` checks for
  result-carrying jobs `889522` and `889849`-`889856` returned no job row
  (`Invalid job id specified`, accepted as absent). The merged result CSV parsed
  with 460/460 `ok` rows, no duplicate keys, dataset counts
  `BNCI2014_001=36`, `Cho2017=208`, `Lee2019_MI=216`, complete prediction and
  logits hashes, and SHA-256
  `87ba93cac505e8d1d073bef67f29a4ccdd055e73185d637244ce2a3687c51698`.
  Summary/digest/audit artifacts are
  `spdim_w1_seed0_summary.json`, `spdim_w1_seed0_result_digest.md`, and
  `spdim_w1_seed0_audit.md`. No target-label leakage, official pretrained
  weight use, or third-party vendoring was detected; no seeds 1/2, full
  three-seed baseline, TeX edit, geometry stress, or orthogonal-score work was
  performed.
- Per PM P6.1, performed the CPU-only W1 split/metric impact audit. No GPU jobs
  were launched and no `sacct` calls were used. Added
  `w1_balanced_accuracy_scorer_audit.py` and generated
  `w1_split_metric_audit.{md,json}`,
  `w1_balanced_accuracy_scorer_audit.{md,json}`, and
  `w1_split_metric_impact_verdict.{md,json}`. The split audit recomputed the
  exact `contiguous_split` composition under the `icml` environment and checked
  it against the SPDIM P6 dry-run hashes/counts. Verdict: Cho2017 has
  single-class evaluation for 52/52 W1 targets; sklearn
  `balanced_accuracy_score` ignores absent classes and degenerates to ordinary
  accuracy on those rows. Affected rows are 1560 corrected REVIEW_P0 W1 raw
  rows (1404 metric rows excluding `__decomposition__`), 312 legacy W1-A rows,
  and 208 SPDIM P6 seed-0 rows. Seeds 1/2 and full three-seed SPDIM remain
  unapproved; alternative split and metric recompute are required before
  escalation.

- Per PM P6.2, completed the CPU-only W1 split repair plan and legacy-result
  quarantine. No GPU jobs, dataset reruns, seeds 1/2, full SPDIM, TeX edits,
  geometry stress, orthogonal-score work, or Slurm accounting calls were used. Added
  quarantine artifacts, valid-subset diagnostic recomputes excluding Cho2017,
  alternative split protocol/dry-run artifacts, rerun feasibility artifacts, and
  the split repair decision gate. Verdict: old W1/SPDIM remain diagnostic legacy
  only; `class_stratified_half` is the recommended replacement split candidate
  because it passes all BNCI2014_001, Cho2017, and Lee2019_MI targets, but any
  H2CMI or SPDIM rerun remains blocked pending PM approval.

- Per PM P7A, froze the CPU-only repaired W1 split manifest and H2CMI dry-run
  gate. Split family `class_stratified_half` now has manifest hash
  `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e` and expected H2CMI rows `3450`.
  No GPU jobs, SPDIM work, TeX edits, geometry stress, orthogonal-score work, or
  Slurm accounting calls were used. Dry-run verdict:
  `dryrun_pass=True`, `approve_gpu_run=True`. Independent repaired-runner
  `--dry-run` crosscheck also passed with 345 manifest units, 3450 expected rows,
  and SHA-256 `8629e2f05969e9d128677b1d740256183a57e56d1426a5b73fed8c2de610f270`.

- Per PM P7B, completed the H2CMI W1 repaired-split clean rerun. Accepted
  job IDs were `890592`, `890593`, `890594`, `890595`, `890629`, `890630`,
  and `890631`; canceled dirty/pending Lee replacements `890596`-`890598`
  are excluded. Final monitoring used `squeue` only and all accepted jobs were
  absent from the queue. The merged result CSV has `3450` rows,
  all `ok`, with no single-class eval rows and complete prediction hashes.
  Manifest hash `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e` matched every row. No SPDIM, TeX edits,
  geometry stress, or orthogonal-score work was performed.

- Per PM P8A, performed CPU-only cache hygiene and prepared the official SPDIM
  W1 repaired-split seed-0 dry-run gate. The untracked P7 training cache
  `results/h2cmi/p7_w1_repaired_bundles/` was preserved outside the repository
  at `/home/infres/yinwang/.cache/h2cmi_training_caches/p7_w1_repaired_bundles_bc61ee1`
  after recording a sha256 manifest; it was not committed and no broad
  `git clean -fd` command was used. Added clean-guarded runner
  `h2cmi/run_spdim_w1_repaired_seed0.py`, Slurm launcher
  `h2cmi/results/review_completion/slurm/spdim_w1_repaired_seed0.slurm`, P8
  protocol, and dry-run audit artifacts. The dry-run used the frozen P7
  `class_stratified_half` manifest hash
  `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`, passed
  all split/loader/model-instantiation gates, and approved only the seed-0 GPU
  run with expected rows `BNCI2014_001=36`, `Cho2017=208`, `Lee2019_MI=216`,
  total `460`. No GPU job, seeds 1/2, full SPDIM, TeX edit, geometry stress,
  orthogonal-score work, official pretrained weights, or vendored third-party
  code were used.

- Per user instruction to split P8 into four GPUs, canceled monolithic P8 job
  `891435` after it had produced 56 partial rows. Those partial rows and Slurm
  logs were moved to
  `/home/infres/yinwang/.cache/h2cmi_training_caches/p8_monolithic_891435_partial_excluded`
  and are excluded from the confirmatory merge. Added shard-aware target
  selection to `h2cmi/run_spdim_w1_repaired_seed0.py` and a four-task Slurm
  array launcher `spdim_w1_repaired_seed0_4shard.slurm`. The replacement shards
  cover BNCI2014_001 subjects 1-9, Cho2017 subjects 1-52, and Lee2019_MI
  subjects 1-54 exactly once, with expected rows `116`, `116`, `116`, and
  `112` for total `460`. Shard outputs are repository-external to preserve
  clean launch provenance. No seeds 1/2, full SPDIM, TeX edit, geometry stress,
  orthogonal-score work, or Slurm accounting calls were used.

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

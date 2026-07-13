# P13 Analyzer-Output Amendment

Status: `OUTPUT_AND_PROVENANCE_ONLY_BEFORE_AGGREGATION`  
Preceding review: `P13_ANALYZER_OUTPUT_RED_TEAM.md`

The P13 analyzer now preserves the external submission record as a checksummed result-packet snapshot and validates that record against the frozen launch commit, runner, config, manifest, clean-worktree state, expected unit count, maximum concurrency, and `squeue`-only monitoring policy.

Failed execution attempts are no longer accepted solely from metadata in the record. For each recorded attempt, the analyzer verifies the excluded raw JSON, stdout, and stderr checksums, requires a pre-result `status=failed` payload with the matching target, seed, and failure reason, and writes `fp_gem_prevalence_excluded_artifact_manifest.csv`. Pending jobs canceled with zero results must be absent from all accepted-unit provenance. The execution audit separately reports accepted result-carrying jobs, failed/excluded attempts, canceled zero-result launches, the verified post-artifact scheduler handoff, and every effective submission command.

Provenance:

- prior amended analyzer SHA-256: `93e52ca3ecb30131ecf18c91b7d7009dad6f6b48a08ccffdf73b2c7a585facdf`
- new analyzer SHA-256: `0baca1664a8c47d50a46e07705fff538fac55f1a4bc62d74b2d088c743855bc7`
- analyzer/test diff SHA-256: `39994a7304994c397495433b22c88c995c68e89dd4d3654314b4bda559229253`
- verified current excluded attempts: `6`
- verified current excluded artifact files: `18`
- P13 regression tests: `8/8 pass`

No scientific input, endpoint, comparator, bootstrap rule, or claim threshold changed, and no aggregate result was inspected before this amendment.

# P13 Raw-Schema Validation Red Team

Review time: `2026-07-14T00:05:16+02:00`  
Review state: GPU execution still active; `145/162` raw units existed. The gate read row fields for validation, but no subject aggregate, contrast, confidence interval, or method verdict had been computed or displayed.

## Findings

1. **Nonempty manifest hashes do not prove that result rows used the frozen batches.** Every result and GEM row must now equal the exact unit/q `adaptation_manifest_hash`; adaptation counts must be `[5,45]`, `[25,25]`, and `[45,5]`, with constant n=50 and balanced unchanged evaluation. Status: `MITIGATED`.

2. **Persisted prediction hashes could disagree with persisted prediction vectors.** Every 50-element prediction vector is independently rehashed as contiguous int64 using the frozen array-hash scheme. Status: `MITIGATED`.

3. **Reused q=0.5 or source-only metrics could drift while hashes still match.** Their acc, bAcc, and macro-F1 now have to equal the unchanged P12 unit metrics exactly, and every row must carry the protocol-defined result origin. Status: `MITIGATED`.

4. **Geometry diagnostics could contain self-inconsistent vectors, hashes, norms, displacement, or priors.** Log-scale and translation vectors are rehashed; norms and displacement from the q=0.5 vectors are recomputed; q=0.5 displacement must be zero; priors must normalize; and FP-GEM fitted priors must equal source priors. Status: `MITIGATED`.

5. **New-q performance values are still generated on GPU and could be wrong even when predictions are well formed.** Evaluation labels are deliberately not persisted in result JSON. Final red-team must independently reconstruct the unchanged evaluation labels from the repaired split and recompute acc/bAcc from every persisted prediction vector before accepting the endpoint packet. Status: `OPEN_UNTIL_FINAL_RED_TEAM`.

6. **New-q logits hashes cannot be independently regenerated because logits vectors were not persisted.** The final packet can verify completeness and q=0.5 exact P12 reproduction, but not independently hash new-q logits. This is a disclosed residual provenance limit, not grounds to alter or rerun the frozen experiment. Status: `RESIDUAL_LIMIT`.

7. **A post-freeze gate change must not alter statistics.** Diff review confirms changes are confined to raw internal-consistency checks and a local hash helper. Aggregation, endpoints, comparators, bootstrap, and claim rule are unchanged. All 145 currently complete units pass; the remaining 17 are missing rather than failed. Status: `MITIGATED`.

## Verdict

`PASS_FOR_STRICTER_RAW_VALIDATION_ONLY`. Analyzer SHA-256 changes from `0baca1664a8c47d50a46e07705fff538fac55f1a4bc62d74b2d088c743855bc7` to `1e02a9e467fe779feaa854c925ce0b25d2016f5974525557775c0025dc07e8e7`; the analyzer diff SHA-256 is `dbf0cc5bf301dbf7132764f1899f07194978d96e2a72bf14f999b1e6f6ddd30d`. Final analysis remains blocked on full `squeue` absence, 162/162 raw units, and independent metric recomputation.

# C84S Final-Report Red-Team

Final status: **72 / 72 PASS**.

## Frozen Identity

1. **PASS** - C84SR3 protocol commit and SHA are reported exactly.
2. **PASS** - V5 lock commit and SHA are reported exactly.
3. **PASS** - Fresh authorization commit and record SHA are reported exactly.
4. **PASS** - Authorization consumption SHA is reported exactly.
5. **PASS** - Complete C84F field SHA is unchanged.
6. **PASS** - Selection-freeze SHA is reported exactly.
7. **PASS** - Scientific result SHA is reported exactly.
8. **PASS** - Result manifest SHA is reported exactly.
9. **PASS** - Result identity SHA is reported exactly.
10. **PASS** - Historical V3/V4 attempts remain failed and non-reusable.

## Execution Reporting

11. **PASS** - Slurm job 898488 is identified.
12. **PASS** - CPU, memory, wall and GPU requests are disclosed.
13. **PASS** - `squeue` monitoring is disclosed.
14. **PASS** - The report does not claim `sacct` use.
15. **PASS** - Stage A/B/C durations and exit codes are reported.
16. **PASS** - Stage-A zero-loader replay is disclosed.
17. **PASS** - Stage-B construction-only access is disclosed.
18. **PASS** - Evaluation release after selection freeze is disclosed.
19. **PASS** - The Stage-C constant-input warning is disclosed.
20. **PASS** - No warning is misreported as a successful scientific endpoint.

## Arithmetic And Artifact Integrity

21. **PASS** - 944 contexts are reported.
22. **PASS** - 535,248 score rows are reported.
23. **PASS** - 535,248 rank rows are reported.
24. **PASS** - 4,720 fixed rows are reported.
25. **PASS** - 944 Q0 shards are reported.
26. **PASS** - 8,750,000 Q0 records are reported.
27. **PASS** - 1,093,750 Q0 sample digests are reported.
28. **PASS** - 18,432 method-context rows are reported.
29. **PASS** - All 18 result tables replay by hash and row count.
30. **PASS** - Lee B32 is reported as input-unavailable, not zero-valued.
31. **PASS** - Cho B32 remains secondary and operative.
32. **PASS** - No primary budget is omitted.

## Scientific Result Wording

33. **PASS** - The primary gate is exactly C84-D.
34. **PASS** - The label tag is exactly C84-L4.
35. **PASS** - Dataset categories are reported as Lee C / Cho A / Physionet C.
36. **PASS** - Cho U11 is identified as the only dataset-level Q1+Q2 method.
37. **PASS** - Cross-dataset A and B intersections are reported empty.
38. **PASS** - Lee COTT's positive mean and Q1 failure are both stated.
39. **PASS** - Lee COTT's worst-target floor breach is stated.
40. **PASS** - Physionet COTT's positive mean and Q1 failure are both stated.
41. **PASS** - Physionet COTT's worst-target floor breach is stated.
42. **PASS** - COTT Q2 passes are not promoted to A without Q1.
43. **PASS** - Cho COTT Q1 failure is not hidden by its Q2 pass.
44. **PASS** - Aggregate regret is not substituted for composite gates.
45. **PASS** - Top-k is not substituted for regret.
46. **PASS** - Measurement association is not substituted for Q1/Q2.

## Stability And Frontier Wording

47. **PASS** - Lee level-0 COTT Q1+Q2 pass is disclosed.
48. **PASS** - Lee level-1 COTT Q1 failure is disclosed.
49. **PASS** - LEVEL_HETEROGENEITY is explicit.
50. **PASS** - Panel/seed directional checks are separated from full passes.
51. **PASS** - LOTO same-method/category counts are exact.
52. **PASS** - LOTO passing thresholds do not erase C84-D.
53. **PASS** - Lee B* is reported absent.
54. **PASS** - Cho B*=8 is reported.
55. **PASS** - Cho level-specific FULL/4 frontier disagreement is reported.
56. **PASS** - Physionet B* is reported absent.
57. **PASS** - C84-L4 is attributed to absent dataset frontiers.
58. **PASS** - No isotonic or post-outcome frontier repair is implied.

## Protected State And Claims

59. **PASS** - Training is reported as zero.
60. **PASS** - Forward is reported as zero.
61. **PASS** - GPU is reported as zero.
62. **PASS** - Same-label-oracle access is reported as zero.
63. **PASS** - Construction and evaluation accesses are ordered correctly.
64. **PASS** - C84 is not called an exact four-class replication.
65. **PASS** - No universal zero-label impossibility claim is made.
66. **PASS** - No universal one-label sufficiency claim is made.
67. **PASS** - No universal EEG external-validity claim is made.
68. **PASS** - No deployability or causal-mechanism claim is made.
69. **PASS** - C85 and manuscript changes remain unauthorized.

## Regressions And Hygiene

70. **PASS** - Focused/C65/C23/full accepted regressions all pass with empty stderr.
71. **PASS** - Rejected regression preflights and their reason are preserved.
72. **PASS** - Git has no tracked >50 MiB file, raw EEG, weights or cache payload.

No blocking discrepancy remains between the frozen external result, the
machine-readable report, the Markdown report, project memory and handoff.

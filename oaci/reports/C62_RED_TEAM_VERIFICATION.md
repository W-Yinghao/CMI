# C62 - Red-Team Verification

All C62 red-team gates pass.

- c61_identity_replayed: PASS - C61 ladder is reproduced exactly from committed artifacts.
- full_conditional_cs_not_silently_claimed: PASS - Full sample-level conditional CS is marked unsupported.
- partition_smoothing_stable: PASS - Partition ladder ordering survives smoothing/support stress.
- endpoint_dominates_partition_and_proxy: PASS - Endpoint dominates template across binary and summary-kernel proxies.
- template_partial_below_null: PASS - Template partial signal remains below max null p95.
- endpoint_beats_null: PASS - Endpoint scalar remains above max null p95.
- template_does_not_screen_endpoint: PASS - Template does not screen off endpoint.
- source_escape_hatch_closed: PASS - No source-observable estimator escape hatch found.
- synthetic_candidate_gauge_positive: PASS - Synthetic candidate-specific gauge rows can flip ranking.
- synthetic_common_offset_negative_control: PASS - Common-offset negative control cannot flip pair ranking.
- instrumentation_not_authorized: PASS - Future instrumentation/training remains unauthorized in C62.
- forbidden_scan_passed: PASS - Forbidden affirmative claim scan passed.
- large_artifact_scan_passed: PASS - All listed artifacts are under 50MB.

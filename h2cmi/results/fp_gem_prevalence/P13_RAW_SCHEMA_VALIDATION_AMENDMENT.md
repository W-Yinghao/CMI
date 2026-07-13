# P13 Raw-Schema Validation Amendment

Status: `VALIDATION_ONLY_BEFORE_AGGREGATION`  
Preceding review: `P13_RAW_SCHEMA_VALIDATION_RED_TEAM.md`

The analyzer now binds every raw result and geometry row to the frozen dataset/target/seed/q manifest entry. It verifies adaptation/evaluation counts, checkpoint and batch hashes, 50-element prediction-vector hashes, P12-reused metrics and result origins, metric ranges, GEM vector hashes and norms, displacement from the exact q=0.5 center, normalized priors, and the fixed FP-GEM source prior.

At amendment time, all `145` completed units passed these stricter checks; the other `17` units were still running and were reported only as missing. No aggregate metric, contrast, interval, or method verdict was computed.

Provenance:

- prior analyzer SHA-256: `0baca1664a8c47d50a46e07705fff538fac55f1a4bc62d74b2d088c743855bc7`
- amended analyzer SHA-256: `1e02a9e467fe779feaa854c925ce0b25d2016f5974525557775c0025dc07e8e7`
- analyzer diff SHA-256: `dbf0cc5bf301dbf7132764f1899f07194978d96e2a72bf14f999b1e6f6ddd30d`
- runner/config/manifest/checkpoints: unchanged
- endpoint/comparator/bootstrap/claim code: unchanged

Independent recomputation of new-q acc/bAcc from evaluation labels remains a mandatory final red-team gate.

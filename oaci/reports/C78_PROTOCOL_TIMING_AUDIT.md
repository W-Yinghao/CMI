# C78 Protocol Timing Audit

- Protocol anchor: `23f549d73803bd23e435f7dae581de29bf62285f`.
- Protocol SHA-256: `ad6f4e034318b879755ca46a719d39cfd3d3c36d7ee8478771d08778a8b71afc`.
- No-auth result commit: `67bca01949c88ab58360179d19868aa78cfc93b7`.
- Exact CLI authorization received: `2026-07-10T21:44:52Z`.
- First authorized GPU submission: job `892830` (synthetic canary failed before data).
- Successful source-only training: job `892832`.
- FIELD_FROZEN materialized: `2026-07-10T22:38:39Z`.
- Post-freeze physical views materialized: `2026-07-10T22:42:55Z`.
- Instrumentation complete: `2026-07-10T22:54:20Z`.
- First target endpoint smoke collection: `2026-07-10T22:59:59Z`.
- Seed-4 access: `never`.
- Full seed-3 expansion: `not authorized`.

The target endpoint smoke occurred only after checkpoint retention and all physical instrumentation manifests were frozen. C78 remains a single-target pipeline canary.

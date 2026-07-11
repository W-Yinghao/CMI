# C79E Repair 003 - Instrumentation Process Scope

Post-C0 red-team inspection found that spawned instrumentation workers re-import
the historical C78F runtime. For primary targets they would pass the historical
target registry but bind the seed-3 C78F authorization and external root. No
Wave A/B instrumentation job had been submitted, so no such cross-seed view
access occurred.

All C79E instrumentation is therefore fixed to the existing locked CLI mode:

```text
--workers 1 --threads-per-worker 48
```

The authorized parent process retains the seed-4 protocol, lock, target registry,
checkpoint paths, and external root. This changes only process topology and
runtime, not features, data rows, forward functions, numerical tolerances,
unit identities, scientific registry, or analysis.

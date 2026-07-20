# C81R Source-Shard Schema Repair Red Team

## Result

```text
38 / 38 PASS
open blockers before reauthorization: 0
```

Selection job `894878` is retained as failed evidence. It stopped while
verifying the first strict-source shard, before a selection manifest, candidate
ranking, evaluation-label descriptor, baseline statistic, oracle access, or
target-4 primary access existed.

The root cause is exact: `c74_cache.verify_shard(required_fields=...)` interprets
`required_fields` as the complete field set, while C81 passed the three fields
it required after already confirming they were a subset. Registered shards also
contain logits, z, Wz, margins, predictions, and trial metadata, so the generic
verifier rejected a valid superset.

The repair retains the explicit three-field subset guard and invokes the full
descriptor verifier without an artificial exact subset. Hash, size, row count,
array-length consistency, unit identity, and trial alignment checks remain
active. The shared C74 verifier is unchanged.

No baseline score, representative, prior, temperature, feature layer, tie rule,
candidate universe, information view, Q1/Q2 margin, max-T family, LOTO rule,
taxonomy, or report schema changed. A synthetic registered-superset fixture
passes.

The first repaired-lock revision `6633fd7` is retained. Before reauthorization,
the red team found that the historical lock had expanded `d17ffa6` to the
nonexistent full identity `d17ffa65...`; the actual reachable commit is
`d17ffa62...`. Blob hashes were always correct. Additive correction `29f4555`
records this provenance-only defect, and final lock `bad8db4` binds only
reachable commits.

The previous authorization record is deliberately rejected by the repaired
runtime guard. No retry can start until a new direct PI authorization binds the
final repaired objects.

## Regression

```text
focused:   45 passed                         job 894892
C65-C81R: 414 passed, 1 skip, 3 deselected  job 894893
C23-C81R: 825 passed, 1 skip, 3 deselected  job 894894
full OACI: 1,749 passed, 1 skip, 3 deselected job 894895
stderr:    0 bytes for all four jobs
```

The skip is the finalized C78F guard. The three deselections remain the
historical C79P preauthorization-state tests. No C81R path was skipped.

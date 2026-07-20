# C78F Collector Repair Red-Team

Gate: **PASS**

```text
repair protocol SHA-256: 60c4c4f2a9a78e7d68af995a6319989c0d9cb46f331d9af54260f3d7e76b508e
checks: 10
failures: 0
failed collector job retained: 893052
training/forward/GPU: 0/0/0
target labels/metrics: 0/0
```

The repair is additive and changes only the compact descriptor row-key mapping
from `rows` to the frozen ABI field `row_count`. All execution-locked training
and instrumentation files remain byte-identical.

# OACI EEG-DG Project Memory Through C81R2

## Current Gate

```text
C81_SELECTION_DESCRIPTOR_REPAIR_LOCKED_READY_FOR_PI_REAUTHORIZATION
```

C81 selection is frozen. Held-evaluation scoring and the final C81 taxonomy have
not run. The same-label oracle remains closed, target 4 remains excluded, and no
training, forward/re-inference, GPU, seed 5, BNCI2014_004, active acquisition,
new method search, C82, or manuscript experiment is authorized.

## Operative Objects

```text
base protocol:                 16a0d2e
base protocol SHA-256:        cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
source-schema repair:          6371b22
source-schema repair SHA-256: ba0434b4ea7965691dafaf506547af64f851c57bdca330a0a5c88e4fa7ba1b15
descriptor repair:             5062f5a
descriptor repair SHA-256:    2acf6ecc179c739f73845d430f9eac9e9e83a83015370b1125dbe447b8b59272
repaired implementation:       225df1c
implementation SHA-256:       d5d8825e9c06994970de87728f73c6c8fef56af0cdd0f734746f1bd4863bf701
execution lock:                f82ffa4
execution-lock SHA-256:       13414dde0a88eb8a1a0810b3b36f25c718669d4cfe3178b871239eff6e292705
readiness evidence:            6118a13
```

## Preserved Attempts

`894878` failed before selection because a required-field subset was passed to
an exact-set source-shard verifier. C81R repaired this locally and retained the
failure.

After direct repaired authorization, selection job `894915` completed in
`00:06:02` on `cpu-high`, GPU 0. It froze 32 seed×target×level contexts and 19
feasible methods. An independent replay then found a second pre-evaluation ABI
defect: the generic trial-shard verifier rejected the intentionally mixed
selection shapes (`32` context rows versus `19` method IDs).

No evaluation-label descriptor was opened in either failure. Held-evaluation
statistics, oracle accesses, target-4 primary rows, training, forward,
re-inference, and GPU work are all zero.

## Frozen Selection Identity

```text
manifest self SHA-256: 4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519
manifest file SHA-256: b02cf6b80127858ef0ab57f5c6e4954715dc9d5aaea77b64cdb9bf13f09ea8a4
payload SHA-256:       1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257
payload bytes:         415,284
contexts:              32
methods:               19
selection recompute:   forbidden
```

The exact shape registry is:

```text
cell metadata:             32
candidate_global_indices:  32 x 81
method_ids:                 19
scores:                     32 x 19 x 81
selected_top10:             32 x 19 x 10
ALine diagnostics:          three arrays x 32
```

C81R2 verifies file hash, size, fields, and each shape. It does not modify score
formulas, representatives, priors, temperatures, directions, ties, methods,
candidate universe, information views, Q1/Q2 margins, max-T, LOTO, taxonomy, or
report schema. The shared C74 verifier is unchanged.

## Regression

```text
focused:    47 passed                           job 894924
C65-C81R2: 416 passed, 1 skip, 3 deselected    job 894922
C23-C81R2: 827 passed, 1 skip, 3 deselected    job 894923
full OACI: 1,751 passed, 1 skip, 3 deselected  job 894925
stderr:     empty for all four jobs
red team:   52 / 52 PASS
```

The skip is the finalized C78F guard. The three deselections are historical
C79P preauthorization-state tests. No C81R2 path was skipped.

## Scientific Base

C80E remains the latest scientific result: both full-panel seed frontiers have
`B*=1`, with source-relative regret gain but only about 3.8% top-1 localization
and 16/16 leave-one-target analyses moving the frontier. This is not universal
one-label sufficiency, low absolute regret, deployment, or external validity.

C81 has not yet produced a zero-label baseline comparison. The frozen selection
alone cannot establish C81-A/B/C/D.

## Authorization

The authorization bound at `b2f9fca` was consumed by selection job `894915` and
does not carry to lock `f82ffa4`. Policy `3d9dd76` requires only a new direct PI
statement; no token or hash recital is required. A sufficient statement is:

```text
授权 C81R2 修复后的 C81E 继续执行
```

After that statement, the server may replay the exact frozen selection, open the
physically disjoint evaluation view, run locked scoring/inference, freeze the
result, and complete C81 reporting. It must not broaden scope.

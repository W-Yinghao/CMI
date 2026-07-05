# ACAR V5 — Stage-1B Future Per-Subject Preprocessed-Window Cache (DESIGN NOTE)

```
DESIGN ONLY / FUTURE BUILD ONLY
NO CODE CHANGE
NO CURRENT RUN CHANGE
NO CURRENT PACKAGE REPLACEMENT
NO REAL DATA EXECUTION
NO CACHE CREATED
NO RERUN AUTHORIZED
```

**This is a design note, not an implementation and not an authorization.** No code in `acar/v5/` is changed by this
document. No cache directory is created. No real DEV data is read. No build is rerun. Nothing here is authorized to
execute while it is unreviewed, and nothing here is authorized to execute while Stage-1B job `881227` is running.

## 0. Relationship to the current run `881227` (NON-INVALIDATION)

This cache is **NOT** a proposal to invalidate, replace, resume, patch, or re-derive the currently running Stage-1B
build (SLURM job `881227`, implementation `c4412b40cb8218ed39c586ff2a4e48247648aa07`, run_id
`acar-v5-stage1b-c4412b4-r1`).

- If `881227` **succeeds and `admit_run(output_root, run_id)` passes**, that hash-bound package **is** the Stage-1B
  result. It stands on its own, produced with **no cache**, by the reviewed single-process build.
- The cache described here exists ONLY for **future, separately-authorized** purposes:
  1. faster **replication** of an already-accepted build (must reproduce byte-identical substrate artifacts), or
  2. **failed-run recovery** of a *future* build **after review** (never an in-place resume of `881227`), or
  3. later **separately-authorized optimized builds** whose contract is re-reviewed from scratch.
- The cache is a performance memoization of a **provably deterministic, seed-free, fold-free** preprocessing step. It
  MUST NOT change substrate semantics or substrate artifact hashes (see §4). If it ever would, it is wrong and must
  fail closed.

## 1. Motivation (evidence, not speculation)

A read-only audit of the build at `c4412b4` (branch `acar`) established:

- **Preprocessing is entirely seed-free and fold/role-free.** The training seed enters ONLY the numeric backend —
  EEGNet weight init (`torch.manual_seed`, `torch_eegnet_backend.py:44`) and mini-batch shuffle
  (`np.random.RandomState(seed).permutation`, `torch_eegnet_backend.py:67,72`). `real_mne_reader.preprocess_subject(...)`
  (`real_mne_reader.py:182`) takes no seed/fold/role and calls no RNG; the fold split
  (`splits.make_fold`, `stage1b_build.py:103`) is a pure hash of the subject universe with no seed. Therefore a
  subject's `SubjectWindows` is **byte-identical across all 3 seeds and all 5 folds**.
- **There is no window cache today.** `real_dev_reader._read_windows_with_repair` (`real_dev_reader.py:27-39`) creates a
  fresh per-call staging tempdir and re-runs the full read + repair + mne DSP on **every** call. The FIT and embedding
  views (`fit_dataset_view.py`, `embedding_dataset_view.py`) memoize nothing.
- **The redundancy is ~23×.** Per ref, Phase-B dumps the entire disease universe (`~230` PD / `~225` SCZ subjects) and
  Phase-A re-reads the `~56%` FIT subjects a second time (`~1.56·N` preprocess calls per ref). Across 3 seeds × 5 folds
  that is `~23.4·N` preprocessings of only `N` distinct subjects.
- **Read + repair + mne DSP dominates wall-clock; EEGNet training is negligible.** Proven by the ~8× SCZ-vs-PD per-ref
  gap (~16 min PD vs ~2h15m first SCZ) under an **identical** EEGNet architecture and `training_config`. The gap tracks
  cohort count × recordings × repair weight (SCZ = 4 heavy-repair cohorts), i.e. the DSP path, not the tiny CNN fit.

**Conclusion.** The single highest-value, lowest-numeric-risk lever is not parallelism; it is **eliminating the ~23×
deterministic repeated preprocessing** via a per-subject cache. Because the windows are provably invariant, this is a
pure memoization: it can only make the build faster, never change its output.

## 2. Cache scope

**Cache unit** — exactly one canonical subject:

```
disease / cohort / raw_subject_id
```

**Cache payload** — the validated, label-free preprocessing product only:

```
validated SubjectWindows (windows float32 array + channels + sfreq + window metadata)
label-free read / repair / montage provenance:
    preprocessing_config_sha256
    channel_alias / montage_completion / brainvision_read_repair / channel_name_repair applied lists + policy hashes
    raw_recording_manifest_sha256 (+ channels_tsv_sha256, repair manifest hashes where used)
NO labels
NO eligibility labels
NO target / diagnosis / group / case_control / participant_group fields
```

The cache serves **both** production read paths:

```
FIT training reads          (RealBidsDevReader.read_subject_windows)
label-free embedding reads  (WindowsOnlyReader.read_subject_windows, all-fold dump)
```

**Label values are never served by the cache.** They continue to come ONLY from
`AuthorizedFitDatasetView.read_label`. The cache is upstream of, and disjoint from, the label channel.

## 3. Cache key (content-addressed — NEVER `subject_key` alone)

A cache entry is addressable ONLY by the full content-addressed tuple below. `subject_key` alone is **forbidden** as a
key: it would let a config/policy/environment change silently serve stale windows.

```
# subject identity
subject_key
disease
cohort
raw_subject_id

# protocol / implementation binding
protocol_tag_target_sha
implementation_base_sha

# preprocessing + policy semantics (all already computed today)
preprocessing_config_sha256          # PC.config_sha256(),                 stamped at real_mne_reader.py:129
channel_alias_policy_sha256
montage_completion_policy_sha256     # PC.montage_completion_policy_sha256()
brainvision_read_repair_policy_sha256
channel_name_repair_policy_sha256

# raw inputs + repair inputs (already computed today)
raw_recording_manifest_sha256        # native path: manifest['manifest_sha256'] (real_mne_reader.py:195)
                                     # repair path: from _repair_aware_reads   (real_mne_reader.py:199)
channels_tsv_sha256                  # when a channels.tsv rename was used
repair_manifest_hashes               # marker-synth / pointer-rewrite / ordinal-rename hashes, when used

# runtime DSP fingerprint (numeric reproducibility of the mne/scipy pipeline)
mne_version
numpy_version
scipy_version
python_version
```

Explicitly **excluded from the key**: any label, any label-source output, any fold index, any seed, any split role,
any staging tempdir path (the per-call staging dir is ephemeral and, by existing design at `real_mne_reader.py:173-175`,
is deliberately kept out of provenance).

**Rationale.** Every key component except the runtime fingerprint is **already materialized** by the current reviewed
read path, so the key is derivable without new preprocessing logic. Two entries with different keys are different
artifacts; an entry whose recomputed key does not match its stored key is corrupt and MUST fail closed (§7).

## 4. Cache lifecycle (per-run scratch, NOT a cross-run public cache)

The **first** version is a **per-run scratch cache**, mirroring the Stage-1B15 repair-staging discipline. It is NOT a
persistent cross-run public store.

```
cache_root:
    explicit absolute scratch path
    outside every raw cohort tree
    outside output_root/run_id
    not a symlink
    absent OR an empty real directory at launch
    validated fail-closed BEFORE any factory/read (reuse the stage1b_repair_staging validation pattern)
    NEVER registered
    NEVER part of registry.json
    NEVER part of FINALIZED.json
```

- **On success:** the cache MAY be cleaned (it is scratch; the registered package holds the artifacts).
- **On failure:** stop and report. **No resume without review.** A failed future build does not silently reuse a
  partially-filled cache; recovery is a separately-reviewed decision.
- **Cross-run persistent reuse is a LATER, separate proposal.** It carries strictly more audit burden because it must
  bind and re-verify the environment (`mne`/`numpy`/`scipy`/`python` versions), every raw manifest, and every
  repair-policy hash across runs, and must prove no stale entry can ever be served. It is out of scope for v1.

## 5. Determinism requirement (cold-vs-cache equivalence — the load-bearing property)

The design MUST require, as a guard, that a **cache-hit** `SubjectWindows` is indistinguishable from a **cold**
`preprocess_subject(...)` result. The two must match on:

```
subject_key
channels (names + order)
sampling rate
window count
window shape
window-bytes digest (float32 array hash)
preprocessing_config_sha256
read-repair provenance
montage-completion provenance
raw_recording_manifest_sha256
```

Because the cache is strictly upstream of training and serves byte-identical windows, the **six substrate artifact
hash fields** the registry binds — `encoder_checkpoint`, `encoder_state_dict`, `source_state`,
`source_state_artifact`, `feat_dump`, and the config sidecars (per `stage1b_artifact_writer` HASH_SOURCE) — **MUST be
unchanged** between a cold build and a cache-served build. A cache implementation is **only correct if the finalized
`registry.json` / `registry_sha256` / `FINALIZED.json` are identical to the no-cache build.** The only permissible
new bytes are **non-substrate cache audit metadata written OUTSIDE the registry hash set** (e.g. a separate
cache-stats sidecar), never inside a registered artifact or the registry hash.

## 6. Label firewall (hard rules)

```
Cache FILL must never call read_subject_label (nor any label-capable reader method).
Cache payload construction must REJECT label-like fields, fail-closed, on:
    label, y, diagnosis, target, group, case_control, participant_group
The embedding view remains label-incapable after caching (no read_subject_label, no back-ref to a label-capable reader).
The FIT view may read labels separately — but a label VALUE is NEVER stored in, keyed by, or served from the cache.
```

The cache is filled through the **label-free** `WindowsOnlyReader` facade / `AuthorizedEmbeddingDatasetView` path, which
already fails closed if the underlying reader exposes `read_subject_label`. Caching MUST NOT introduce any new path by
which a label could reach the windows store.

## 7. Repair-staging interaction (do NOT bypass Stage-1B15)

The cache MUST NOT weaken or bypass the Stage-1B15 reviewed repair staging.

**Cache MISS path:**

```
goes through the SAME production RealBidsDevReader path
creates a fresh per-call repair staging dir under the validated repair_staging_root
runs the Stage-1B12/1B13/1B14 repairs EXACTLY as reviewed (marker synth / pointer rewrite / ordinal rename / montage)
computes the full content-addressed key (§3)
writes the cache payload ATOMICALLY (temp file + rename), then returns SubjectWindows
```

**Cache HIT path:**

```
does NOT recreate repaired headers
does NOT touch the raw cohort tree
does NOT create a repair staging tempdir
re-derives the content-addressed key from current config/policy/env and REQUIRES it to equal the stored key
validates the stored provenance/hash package
returns the validated SubjectWindows
```

A hit that cannot revalidate its key/provenance is a **miss-or-fail-closed**, never a silent stale serve.

## 8. Safety tests to propose (guards, not implementation)

```
test_cache_key_includes_preprocessing_policy_hashes      # key changes if any policy/config/env hash changes
test_cache_rejects_label_fields                          # payload with label/y/diagnosis/... fails closed
test_cache_hit_equals_cold_subjectwindows_digest         # cold vs hit: identical window-bytes digest + provenance
test_cache_miss_uses_repair_staging_branch               # miss goes through the reviewed repair path + staging dir
test_cache_does_not_touch_raw_tree_on_hit                # hit performs zero raw-tree / staging writes
test_cache_root_rejects_raw_tree_overlap                 # cache_root overlapping a raw cohort path fails closed
test_cache_root_rejects_output_run_root                  # cache_root overlapping output_root/run_id fails closed
test_cache_atomic_write_no_partial_hit                   # interrupted fill never yields a servable partial entry
test_cache_corrupt_payload_fails_closed                  # tampered/truncated payload -> error, not silent serve
test_cache_mne_version_mismatch_fails_closed             # stored env fingerprint != runtime -> miss-or-fail, never serve
```

(Plus the existing invariants must continue to hold under the cache: label firewall, per-ref output containment,
gate-first / fresh-run-root / all-or-none finalize, registry/FINALIZED atomicity.)

## 9. Explicit non-goals

```
Not Stage-2.
Not parallelization.
Not a SLURM-array build.
Not resume (of 881227 or any run).
Not registry modification.
Not a change to preprocessing semantics.
Not a change to EEGNet training.
Not a change to the feature-dump schema (unless separately reviewed).
Not a cross-run persistent public cache (v1 is per-run scratch only).
Not authorized while job 881227 is running.
Not authorized until this design is reviewed and a separate build is authorized.
```

## 10. Lever ranking (future build) and sequencing

```
1. Per-subject preprocessed-window cache (THIS note):
     best value / risk. Deterministic memoization -> identical substrate hashes. ~>10x by collapsing ~23x -> 1x.
2. In-run reuse of Phase-A FIT reads in Phase-B:
     smaller, local version of the same idea (removes the ~0.56N double-read). Good fallback if the full cache is
     judged too much surface area for one patch.
3. Subject-level PROCESS parallelism (intra-op threads pinned at 1):
     up to ~4x on DSP using the allocated CPUs; per-subject windows are independent -> bit-identical. Secondary.
4. SLURM-array over the 30 refs + separate gather/finalize:
     highest raw speedup, but it touches the ALL-OR-NONE gate/finalize/fresh-run-root discipline -> highest
     protocol-review burden. Must come AFTER the cache design is stable and proven.
```

**Sequencing rule (agreed): do NOT combine cache + SLURM-array in the first optimization patch.** First prove the
cache **alone** with cold-vs-hit equivalence (§5) and the §8 guards, on synthetic/fixture data, on py3.9 + py3.13.
Threading MUST stay pinned (`torch_threads=1`, `OMP/MKL=1`); raising intra-op threads changes FP reduction order and
breaks byte-reproducibility of the windows **and** the frozen encoder hash, and would also invalidate the hashed
`training_config` `config_sha256`.

**Design principle (user directive, 2026-07-05).** Do NOT submit a future substrate build as a single multi-day
single-core job (job `881227` ran ~27h on effectively one core because `torch_threads=1` and DSP `n_jobs=1`, while 4
CPUs sat mostly idle). Future builds MUST exploit the allocated cores. The ONLY reproducibility-safe way to do that is
**PROCESS-level parallelism** — fan independent subjects/refs across cores/nodes with intra-op (BLAS/OMP/torch) threads
pinned at 1 — combined with this dedup cache. Multi-core here means more *processes*, never more *threads inside a
kernel* (which breaks the frozen hash, per above). The cache alone already removes the multi-day problem (~27h → a few
hours by collapsing ~23x redundant preprocessing); process parallelism then compresses it further.

## 11. Authorization status

```
STATUS: DESIGN ONLY. UNREVIEWED. UNAUTHORIZED. UNCOMMITTED.
```

- This note changes no code and creates no cache.
- It is held **uncommitted** while job `881227` is running, so the running Stage-1B package stays cleanly associated
  with `c4412b40cb8218ed39c586ff2a4e48247648aa07`.
- After `881227` completes or stops, report the Stage-1B package status FIRST; only then decide whether to commit this
  note and whether any optimized future build is worth pursuing. A cache implementation, if pursued, is a **new,
  separately-authorized** stage (code + synthetic/fixture tests only, adversarial review, two-Python verification),
  pinned to its own reviewed implementation SHA.

# C86H — Production implementation patch: readiness object (compact)

**Status**

```text
C86H_PRODUCTION_PATCH_IMPLEMENTED (batch H1a/H1b, compact freeze, Semantics B, C86 candidate IDs,
  full inference detail, atomic terminal result, real-path enforcement)
C86H_F1F2_IS_THE_AUTHORIZED_DATA_ACCESS_STEP  (implemented as gated orchestration; not validatable in prep)
C86H_REAL_EXECUTION_NOT_AUTHORIZED  (separate 授权 C86H required) ; STOP_BEFORE_DATA_ACCESS holds
C87_NOT_AUTHORIZED ; MANUSCRIPT_WORK_NOT_AUTHORIZED
```

Mapped to the five requested production conditions. No real Brandl/ds007221 EEG/label touched.

## Condition 2 — C86-specific candidate IDs  ✔

`field_spec.c86_candidate_id` = `c86_ + SHA256(interface_id | field_training_manifest_sha256 |
panel | seed | level | regime | epoch)[:24]`; same ID for a candidate across BOTH cohorts,
disjoint from the historical `c84_` 20-channel namespace. Manifest `candidate_id_rule` updated.

## Condition 3 — production-equivalent compact H1  ✔

The ~93M-RPC per-query server is replaced by a **label-independent batch** path (the PM's
design): `batch_h1.run_h1a` generates every (target, method, chain) order from UNLABELED
probabilities; a SEALED `run_h1b_sealed` reads each target's acquisition labels/contributions
once and batch-evaluates the composite via the FROZEN `freeze_budget`. Selections are
**byte-identical** to the per-RPC C86D worker — verified: 0 mismatches across all AVAILABLE
(method,target,chain,budget) cells (`test_batch_h1_equals_per_rpc_worker`). Compact freeze:
one NPZ per (target,method) = **53×3 = 159 files** + a content-addressed H1 manifest (not
325,632 JSONs). The resource benchmark runs the REAL run_h1a/run_h1b_sealed with disk
serialization (no in-memory placeholder):

```text
per target (pool 300 = ds007221 scale, 2,048 chains, 3 methods):
  H1a order gen 203 s ; H1b sealed contribution eval 1,239 s ; total ~1,442 s
  compact freeze bytes/target ~30 MiB ; order bytes/target ~28 MiB
full campaign 53 targets:
  ~21.2 CPU-core-hours serial ; ~0.4 h wall on 53 cores (embarrassingly parallel per target)
  compact freeze storage ~1.56 GiB (159 files) + orders ~1.6 GiB  (<< 640-768 GiB scratch envelope)
  max-T Brandl 1.25 s (exhaustive 2^16) / ds007221 1.59 s (MC 65536) ; peak RAM 2.3 GiB (<< 128 envelope)
DECISION: FEASIBLE. (The compact schema keeps freeze storage ~1.6 GiB, not the tens of GiB a
  per-(target,method,chain,budget) JSON scheme would produce.)
```

## Condition 4 — Semantics B fix  ✔

ONE physical trial → ONE label → its 8 context-specific probability/contribution rows
(`field_spec._synth_labels` once per target; `_synth_probs` per context on the shared labels;
oracle + held share them). All randomness is SHA-seeded (`_sha_seed`), never Python `hash()`.
Verified: `test_semantics_b_one_label_per_physical_trial`.

## Condition 5 — complete + atomic terminal result  ✔

`held_eval.evaluate` freezes full `inference_detail` per (cohort, active method, budget):
observed t, adjusted max-T p, critical, seed + family digest + sign mode + n_signs, mean
effect, favorable fraction, worst target, 8 cell effects, all-alpha CVaR effects, LOTO, and the
mean/tail/stability qualification booleans — so the formal gate is independently re-derivable.
H4 writes `C86H_TERMINAL_RESULT.json` atomically (staging + `os.replace`), self-hashed
(`result_sha256`), carrying classification + frontier + descriptor + endpoints + full ceiling +
inference detail + H1 file hashes + bindings.

Real-path enforcement (`run_confirmation`, non-synthetic): forces `chains == range(2048)`
(rejects caller-reduced chains), binds cohort set == registered COHORTS and target count == 53,
realpaths the field root, and refuses without `授权 C86H` before opening anything.

## Condition 1 — real F1/F2  (implemented as gated orchestration; validation IS the authorized step)

`f1f2.py` binds the EXACT existing training entrypoints (`train_paired_cell`, `train_level`,
`materialize_paired_bundles`, `load_source_panel_views`, engine `train_stage1/2`) and defines
the content-addressed `real_field_manifest_schema` F2 must emit (source/target raw-file hashes,
648 weight hashes, 81 candidate IDs/context, 424 prediction contexts, split + support). It is
GATED: `f1_train_zoo` / `f2_generate_predictions` refuse without `授权 C86H`.

**Honest boundary (surfaced, not hidden):** real F1/F2 cannot be *validated* in preparation.
Discovery confirmed the entire training stack is hard-wired to **20 channels** (`in_chans=20`
in every `_model_factory`, the MOABB paradigm, shape guards, and the montage/model-init
hashes), and **ds007221 has no adapter** (OpenNeuro/NEMAR BIDS — needs mne-bids). The 11-channel
retarget deltas and the ds007221 BIDS adapter are recorded in the manifest
(`eleven_channel_retarget_requirements`) and `f1f2.ELEVEN_CH_RETARGET` / `F2_TARGET_ADAPTERS`,
but producing the 648 real weights + 424 predictions requires real EEG + GPU access — i.e. this
IS the `STOP_BEFORE_DATA_ACCESS` step. F1/F2 real training/adaptation is validated only under
`授权 C86H`, and the authorized build orchestrates the bound frozen entrypoints with the 11-ch
config; it does not fork them.

## Adversarial verification of the production patch

An independent red-team (3 skeptics) confirmed batch↔per-RPC selections byte-identical
(720/720 cells) and Semantics B / candidate IDs / real-path / F1/F2 gating all correct, and
found production defects — all now fixed and re-tested:

```text
MAJOR  verify_h1 did not reconcile the H1b freeze against the label-free H1a orders
       -> H1 verification now RECONCILES freeze.orders/q_seq/seeds == H1a orders (label-independence
          audit); H1a moved to a CAPABILITY-ISOLATED spawned process (only pool + orders_dir)
MAJOR  terminal result_sha256 never persisted + no overwrite guard
       -> digest written to a .sha256 sidecar; H4 refuses to overwrite an existing result (one-shot)
COMPLETENESS  max-T significance committed as trusted scalars
       -> inference_detail freezes the raw per-target effect_vector + common_targets so max-T is
          recomputable from the committed artifact alone
MINOR  _spawn_recv leaked the child pipe end (hang on abnormal worker death) -> child.close() + EOF->RuntimeError
MINOR  verify_h1 gaps -> q_seq in (0,1], candidate bound from K, H1 manifest loaded from disk
NIT    F2 field-key convention -> real_field_manifest_schema pins meta['dataset']==cohort interface name
```

## Tests / verification

87 c86h tests + C86D 42/42 unregressed. Batch↔per-RPC equivalence, orders-reconciliation audit,
Semantics B, compact-freeze verification, full inference detail, atomic+immutable+self-hashed
terminal result, gated refusals, real-path enforcement all covered. `verify_bindings` (V3
manifest + field/training manifest + registry tables + frozen dispatcher blobs incl. server.py
+ selection_worker.py) fail-closed OK.

# C84F Final Report Red-Team

Result: **68 / 68 PASS**.

| Category | Checks | Result |
|---|---:|---|
| authorization, protocol and execution-lock identity | 10 | PASS |
| frozen model, canary and target-input identity | 10 | PASS |
| execution order, atomic publication and field arithmetic | 12 | PASS |
| same-backend, persistence and diagnostic numerical contracts | 12 | PASS |
| target-label, training, selector, oracle and C84S isolation | 12 | PASS |
| logs, regressions, repository hygiene and claim boundary | 12 | PASS |

The audit replayed the direct authorization and consumption records, the C84FR2
repair and target-instrumentation V2 protocols, the execution lock, 38 bound
repository objects, ten protocol sidecars, 1,944 frozen model units, 7,776 model
artifacts, 2,430 dual-canary files, the 118-subject/9,621-row target registry and
all 11 rejected historical partial target objects. Jobs `896185` and `896550`
remain immutable failed attempts and contributed no target artifact to the V2
field.

The complete-field replay streamed every one of the 1,944 NPZ files and 1,944
context/digest sidecars. It verified exact manifest membership, path/size/hash
identity, all 21 registered persisted-field digests, 944 contexts, 76,464
candidate-context slices and 486 canary witnesses. The atomic complete manifest
replayed to
`cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8`.

The same-GPU/PyTorch and saved `Wz_plus_b`/logits maxima were zero; saved
softmax was `1.1920928955078125e-07`; repeat logits and z were zero. All strict
checks passed at `1e-6`. CPU PyTorch and NumPy reconstruction values were finite
and remained diagnostic only. No runtime tolerance was widened.

Protected-state checks passed with zero retraining, zero target-y operations,
zero target-label fields, zero construction/evaluation labels, zero oracle
access, zero selector scores and zero scientific statistics. C84S is false and
no C84S execution lock exists. Scheduler monitoring used `squeue`; no scheduler
terminal state was inferred. The 20 disclosed loader warnings contain no
traceback or runtime-failure marker, and no C84 job remained active at the final
scan.

The conclusion is field-generation-only. It does not establish target accuracy,
selector performance, Q1/Q2, a label-budget frontier, level effects,
cross-dataset recurrence or external validity.

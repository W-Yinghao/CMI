# C84L1C Failed Attempt 895928

## Disposition

Job `895928` consumed the scope-specific C84L1C authorization and stopped fail-closed during Lee2019_MI level-1 instrumentation. It is preserved as a failed engineering attempt and is not reusable.

The failure occurred after all three Lee training phases had completed. Seventy-three units had complete checkpoint/state/sidecar/source-audit/target-unlabeled replay. The next unit, `c84l1_3694c9bcfa6865ae595f4607e9c03a0f` (`SRC`, epoch 164, trajectory order 33), produced a CPU-versus-GPU float32 `zW+b` maximum absolute difference of `1.239776611328125e-05`, above the locked `1e-05` engineering tolerance.

The saved-softmax, repeat-logit, and repeat-z differences were all zero. This is a numerical replay blocker, not a target scientific result.

## Frozen State

- Training phases started/completed: `3 / 3`
- Checkpoints: `74`
- Optimizer states: `81`
- Complete sidecars/source-audit/target-unlabeled artifacts: `73 / 73 / 73`
- Partial-manifest SHA-256: `ba67a4a0f8a516085b3eb020c353c401c2eafdd1981eb880c5c63587ac31b091`
- Authorization-consumption SHA-256: `45a7352bc4fde96b7ae6bdd93960dbb1bec729056a7e149542cb20e92ce753d0`
- Ledger-derived runtime: `2041.986663626` seconds
- Last active scheduler observation: `squeue`, node `node08`; the job is no longer active

## Protected State

Target-y accesses, target scientific metrics, construction/evaluation view access, same-label oracle access, and target-outcome decisions all remained zero.

## Repair Boundary

The consumed authorization is not reusable. The failed external root and all partial artifacts remain preserved but cannot enter a replacement canary. Any retry requires an additive numerical repair protocol, new implementation lock, fresh content-addressed root, and fresh direct PI authorization. C84F and C84S remain unauthorized.

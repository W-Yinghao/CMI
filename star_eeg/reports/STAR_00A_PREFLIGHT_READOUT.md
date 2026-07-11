# STAR_00A Preflight Readout

This is a STAR project design and red-team preflight.
No real STAR EEG training was run.
No FACED target metric was computed.
No checkpoint was selected using target labels.
No CMI, adversary, pruning, surgery, TTA, CSP-init, or safety gate was introduced.
S2P Phase B remains independent and unchanged.
STAR_01 scientific training is not approved by this commit.

## Summary

| Field | Value |
|---|---|
| `status` | `"PASS"` |
| `dependency_commit` | `"a9134eb5eb7f8486a5e1ee41831823dab39381ed"` |
| `star_branch` | `"project/star-task-anchor"` |
| `s2p_files_modified` | `[]` |
| `h2cmi_files_modified` | `[]` |
| `oaci_files_modified` | `[]` |
| `h200_start_checkpoints_ready` | `true` |
| `checkpoint_inventory_hash` | `"89fc5d73ad6864f9f4835118100c6b0955720f1f1d76ad22a7105d01ddab87d9"` |
| `faced_split_hash` | `"9ca8d6ecd8294a3af7c4fb1044768c4dc8c3bcdde3d389ab6d1685ec5ae0f460"` |
| `anchor_manifest_preview_hash` | `"8dff4eaad88514d915aed0a0a78f0bc9a0e901a6ca513462f0762fa8a4703896"` |
| `shuffled_manifest_preview_hash` | `"03275a9fe3b74271ea970adb39cf9bc1b6a29e070bede8f79100794363ade35e"` |
| `compute_match_hash` | `"08605c5122412f2562c69c50e1ba25c18042088fd74315a5556986a6b9e50d21"` |
| `target_label_quarantine` | `"PASS"` |
| `preflight_determinism` | `"PASS"` |
| `real_training_run` | `false` |
| `target_metrics_computed` | `false` |
| `star01_approved` | `false` |

## Scope and interpretation

The preflight performed read-only repeated SHA256 checks and strict CBraMod reloads, exact split and API firewall checks, deterministic schedule/compute matching, fixed within-subject shuffled-manifest checks, active-method registry checks, and a tiny synthetic loss/gradient smoke. It did not load real EEG arrays.

H500, H1000, H2000, released, and random remain frozen descriptive references. They cannot train STAR, select a variant, tune the anchor schedule, or support an equivalence/reproduction/superiority claim.

The only primary future checkpoint is optimizer step 3750. Source-val diagnostics cannot replace it. STAR_01 remains unapproved regardless of artifact readiness.

# STAR_00 Artifact Inventory

## Read-only checkpoint inventory

The preflight inventories exactly:

- H200_s0, H200_s1
- H500_s0, H500_s1
- H1000_s0, H1000_s1
- immutable H2000_s0, H2000_s1
- released CBraMod
- random CBraMod reference configuration

`checkpoint_inventory.json` records every required field and repeats SHA256 around strict reload. H2000 resolves through the immutable read-only symlink contract. H200/H500/H1000 resolve through the completed Route B B1 result root. Released and random remain references and have no Route B training-provenance claim.

## Generated STAR_00A artifacts

All outputs are small JSON under `results/star/star00a_preflight/`:

| Artifact | Purpose |
|---|---|
| `run_manifest.json` | Scope, no-real-training assertions, synthetic smoke |
| `dependency_manifest.json` | Dependency/boundary result and authority-file hashes |
| `checkpoint_inventory.json` | Ten-object SHA/reload/provenance/role inventory |
| `faced_split_contract.json` | Exact subject firewall and split hash |
| `anchor_manifest_preview.json` | Synthetic-only source_train manifest preview |
| `shuffled_manifest_preview.json` | Synthetic-only fixed within-subject shuffle preview |
| `compute_match_contract.json` | Exact optimizer-step schedules and hash |
| `target_label_quarantine.json` | Signature and data-read firewall result |
| `no_forbidden_method_guard.json` | Active registry/import guard result |
| `preflight_summary.json` | Overall status and required summary fields |

The preview manifests contain synthetic identifiers only. They do not contain real FACED arrays, features, target labels, target distributions, or scientific metrics. The full source_train manifests are future STAR_01 launch inputs and do not exist in this commit.

`STAR_00A_PREFLIGHT_READOUT.md` is rendered from the deterministic summary after the inventory run.

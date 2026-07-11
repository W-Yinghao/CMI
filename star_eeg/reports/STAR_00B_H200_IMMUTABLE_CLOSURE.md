# STAR_00B H200 Immutable Start Closure

## Purpose

STAR_00A established stable H200 source payloads but did not make them immutable. STAR_00B copies those exact payloads without training to an external runtime root and creates committed provenance manifests only.

## Fail-closed checks

For H200_s0 and H200_s1:

1. The source path, SHA, strict reload, completed 50-epoch log, best checkpoint provenance, Route-B manifest, and source/validation disjointness must match STAR_00A.
2. Source SHA is read before copy and after destination strict reload.
3. Destination filename is `best.<full-sha256>.pth` and never overwrites different content.
4. Destination SHA equals source SHA before chmod and after strict reload.
5. Destination mode has no write bits (`0444`).
6. `best.pth` is a stable relative symlink to the SHA-named payload.
7. The STAR runner accepts only the regular SHA-named path declared as `launcher_accepted_path`; it rejects the mutable source and the convenience symlink.

The closure writes `h200_immutable_manifest.json` and `h200_immutable_go_nogo.json`. Model payloads remain outside Git. No `.pth` file is committed.

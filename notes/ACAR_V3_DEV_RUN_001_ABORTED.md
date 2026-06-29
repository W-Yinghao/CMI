# ACAR v3 — DEV run #001: OPERATIONALLY ABORTED (no scientific verdict)

**Date:** 2026-06-24 · **Status:** `NO SCIENTIFIC VERDICT / OPERATIONALLY ABORTED` (explicitly **NOT** `DEV_STOP`).

## What was attempted
The first real DEV S2/S4 gate, launched through the unique binding CLI at the DEV-design lock:

```
python -m acar.v3.run_dev_binding \
  --input-manifest <scratch>/acar_v3_dev_inputs.json \
  --output         <scratch>/dev_out
```

- Protocol commit / tag: `acar-v3-dev-design-v1 → 817b04f92d616b0b17bac223181c0f846f9209ac` (pushed).
- Env lock: `2cb61360a01af61001ac4a97e6269c16ee4d89c998122d22d557c7d7c84cab17`.
- Input manifest (built outside the repo): the 7 real cohorts — PD ds002778/ds003490/ds004584 (230 subjects), SCZ
  ds003944/ds003947/ds004000/ds004367 (225 subjects), d=16; `feat_hash_te` used as `raw_pipeline_sha256`.

## What happened
- **Preflight PASSED**: output-absent, manifest schema, `HEAD == protocol_commit`, tag → HEAD, clean worktree, per-file
  `full_dump_sha256`, env-lock verify, all five dump-derived field hashes.
- The S2/S4 bake-off then ran silently and the process was **killed before producing a verdict** (external session /
  timeout; exit code ≠ 0).

## Evidence / state
- `binding_run.log`: empty (killed during the silent gate compute, before the single final verdict print).
- `dev_out`: **never created** (no `os.rename` — atomic publication did not occur).
- `dev_out.tmp`: exists but **empty** (the atomic claim happened; no artifacts were written). Because the kill was
  external, `freeze_dev_run`'s `except` cleanup did not run, so the stale temp remains — this is the intended
  fail-closed marker (a future re-run's `os.mkdir(dev_out.tmp)` raises `FileExistsError` until it is deliberately
  cleared).

## Interpretation (per the acceptance rules)
A killed process / partial artifacts is **not** a `DEV_STOP / NO_LOCKBOX_CONSUMED` and is **not** a `SELECT`. The DEV
gate has simply **not been evaluated** on real data yet. No auto-rerun was performed. No external Arm-B endpoint or
lockbox was touched.

## To re-run (when authorized)
Run at the tagged commit, clean worktree, in a process allowed to finish (long: per-batch source adapters over ~550
eligible batches + the C1/C2/C3 + C0 fits, both diseases). Record the stale `dev_out.tmp`, then remove it (or use a
fresh output dir name); rebuild the out-of-repo input manifest; invoke the single binding CLI. The result (a real
`SELECT` + frozen artifacts, or `DEV_STOP / NO_LOCKBOX_CONSUMED`) goes into a **separate result commit** — the protocol
commit and the DEV-lock tag stay put.

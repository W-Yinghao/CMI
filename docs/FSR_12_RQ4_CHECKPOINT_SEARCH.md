# FSR_12 — RQ4 Option-A Checkpoint Search

**Project FSR — Phase 4 (Option A only, read-only).** Executes the one approved Phase-4 action: a read-only search for existing FBCSP-LGG checkpoints, plus a CPU load-test if any are found. **No inference, no latent dump, no training, no target labels.** Option B (ERM re-fit) was not approved and is not attempted.

Artifacts (this dir): `checkpoint_search_manifest.csv`, `checkpoint_search_log.txt`, `checkpoint_load_test.json`.

## Result: NOT_USABLE — no FBCSP-LGG checkpoints exist

| scope | checkpoints found |
|---|---|
| `/projects/EEG-foundation-model/yinghao` (user output dir, maxdepth 6) | **0** — no fbcsp/lgg output dir at all (only `acar_v5_*`, `oaci-*`, `cigl_bnci_readable`) |
| `yinghao/cigl_bnci_readable` | raw MNE dataset cache only, **no weights** |
| `/home/infres/yinwang` (home + worktrees, maxdepth 4) | **0** fbcsp/lgg checkpoints |
| git `project/fbcsp-lgg-*` | **0** committed checkpoints |
| root datalake full find | timed out (huge shared tree); the user output dir is the only plausible location and is empty |

**Corroboration (mechanistic, not just absence):** the FBCSP-LGG F0 pipeline persists summary JSON only; the trainer clones `best_state = backbone.state_dict()` to CPU RAM for best-epoch restore but **never `torch.save`s it** (`cmi/train/trainer.py:554`). So no weights were ever written to disk — consistent with the empty search.

The CPU load-test was therefore not run (nothing to load); torch availability was moot.

## Success criteria (all FAIL → not usable)
1. real FBCSP-LGG weights found — **NO**
2. `state_dict` keys match the current backbone — n/a (no file)
3. bindable to dataset/fold/seed/config — n/a
4. covers ≥1 complete interpretable unit — n/a
5. no target-label fit needed — n/a
6. no retraining needed — **FAILS** (retraining would be required to produce any checkpoint)

Not a single criterion is met → **NOT_USABLE**.

## Decision (per PM)
- RQ4 **remains BLOCKED**; it stays a descriptive result (spatial branch load-bearing) with the honest gap "branch-local leakage/reliance is not measured" (claim C7 `READY`).
- **Do not request an ERM re-fit** (Option B not approved). The "no usable checkpoint" finding is now the frozen provenance for the RQ4 gap and can be cited as such in the paper (§6, limitations).
- The RQ4 blocked status is itself a finding: a branch-local shortcut claim requires an instrument that does not currently exist and cannot be produced without a new training run.

## What the paper can/cannot say (unchanged)
- **Can:** "the spatial branch is load-bearing; branch-local leakage and reliance are not measured, and no frozen checkpoint exists to measure them without retraining."
- **Cannot:** "spatial leakage is harmful", "graph leakage is benign", "per-branch CMI predicts reliance."

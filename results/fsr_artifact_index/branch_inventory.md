# FSR branch inventory (Step 1 §1)

Snapshot for **Project FSR** evidence freeze. Generated from local `git` refs on 2026-07-06. `git fetch --all --prune` ran with no network change (offline/no new refs); all SHAs below are from the local clone and cross-checked against `origin/*` where a remote exists.

**Current position.** Working branch `project/functional-shortcut-reliance` (worktree `/home/infres/yinwang/CMI_AAAI_fsr`), cut from `project/cita-target-unlabeled-cmi` @ `c889730` — the branch most complete for CIGL_66/70 + CITA_03 + the CIGL_67/68/69 series + `CMI_SYNTHESIS_01`. No code merged from other branches; ledger references them by branch + SHA + path only.

## Tier-1 — FSR primary evidence sources

| branch | local_sha | remote_sha | date | worktree | tree | last_commit_subject |
|---|---|---|---|---|---|---|
| project/cita-target-unlabeled-cmi | c889730 | c889730 | 2026-07-06 | CMI_AAAI_metacmi | clean | CMI_SYNTHESIS_01 — CMI control closed (both regimes) |
| project/cigl-r123-scaffold | 1339592 | 1339592 | 2026-07-05 | CMI_AAAI_cigl_r123 | clean | CIGL_66: measurement→control gap diagnostic (Outcome A) |
| project/cigl-functional-cmi | c1d55be | c1d55be | 2026-07-06 | CMI_AAAI_fcigl | clean | CIGL_67: FCIGL raw audit .npz pruned (derived kept) |
| project/cigl-direct-reliance-cmi | e619713 | e619713 | 2026-07-06 | CMI_AAAI_dcigl | clean | CIGL_68_DIRECT_RELIANCE_CONFIRMATION: method-level NEGATIVE |
| project/metacmi-eegnet-conformer | b73686d | b73686d | 2026-07-06 | (none) | — | CIGL_70_SOURCE_ONLY_CMI_CLOSURE — freeze source-only route |
| tos | 1c65d79 | 1c65d79 | 2026-07-05 | (none; CMI_AAAI_tos on science-source-rich-v1) | — | gitignore topup_watcher.log |
| project/fbcsp-lgg-spatial-cmi-fusion | 39c245a | 39c245a | 2026-07-05 | (none) | — | CIGL_50 P6 seeds1/2: PI final decision — fragile survive |

## Tier-2 — FSR supporting / branch-locality / boundary

| branch | local_sha | remote_sha | date | worktree | tree | last_commit_subject |
|---|---|---|---|---|---|---|
| project/fbcsp-lgg-dualcmi-scaffold | eb47bd0 | eb47bd0 | 2026-07-04 | (none) | — | FBCSP-LGG F1a decoder-only + full-LOSO ERM references |
| project/fblgg-2a-bottleneck-analysis | 787fcc7 | 787fcc7 | 2026-07-02 | (none) | — | CIGL_48 P4: BNCI2014 4-class bottleneck = FBLGG feature extr |
| oaci | afc8f50 | afc8f50 | 2026-07-06 | CMI_AAAI_oaci | **dirty (4)** | C19: reword report disclaimer to avoid forbidden substr |
| acar | d287635 | d287635 | 2026-07-06 | CMI_AAAI_acar | clean | ACAR V5 closeout: Stage-2B **DEV_STOP** — engineering/proto |
| csc | 72085b7 | 7f64a49 | 2026-07-06 | CMI_AAAI_csc | clean | results: B7.1 **PROTOCOL** (pre-registered before run) |
| exp/lpc-cmi | 050d3a4 | 050d3a4 | 2026-06-21 | (none) | — | CLOSE failed LPC-CMI/CITA/gate line: full-autopsy CLOSEOUT |

## Tier-3 — background only (NOT FSR leakage/reliance evidence)

| branch | local_sha | remote_sha | date | worktree | tree | last_commit_subject |
|---|---|---|---|---|---|---|
| exp/h2cmi-wave0-mechanism | 60db118 | **(local-only, no remote)** | 2026-07-06 | CMI_AAAI_qxu | dirty (1) | h2cmi(wave0): FIX W0.2 ord_acc/recall — argmax hard preds |
| exp/h2cmi-review-p0-corrections | 5bc9bf0 | 5bc9bf0 | 2026-06-29 | (none) | — | h2cmi(review-p0): FINALIZER #4 TERMINAL results |
| main | c8fce20 | e93f348 | 2026-06-20 | (none) | — | Protocol audit P2.3 (GLS→raw sampler) + P2.5 |

## Provenance flags (must be respected when citing)

- **`csc` local is 1 commit AHEAD of `origin/csc`** (local `72085b7` vs remote `7f64a49`). The B7.1 pre-registered-protocol commit is not yet pushed → cite the local SHA and note it is unpushed.
- **`main` local is 1 commit AHEAD of `origin/main`** (`c8fce20` vs `e93f348`); also this repo's checked-out branch is `project/cigl-leakage-probes`, not main.
- **`exp/h2cmi-wave0-mechanism` is local-only** (no `origin` ref) — background tier; any citation is local-clone-only.
- **`oaci` worktree is DIRTY (4 files)** and its HEAD subject is a report-disclaimer reword (C19) — re-verify report wording against the committed tree before quoting.
- **`tos` and the FBCSP branches have no dedicated worktree** — read via `git show <sha>:<path>`, not a checkout. (`CMI_AAAI_tos` is checked out to `science-source-rich-v1` @ `f61dd90`, whose `tos_cmi/CLAIMS_LEDGER.md` is 38 lines shorter than `tos`'s — always cite `tos` @ `1c65d79`, not the worktree.)
- **`acar` HEAD = "Stage-2B DEV_STOP"** and **`csc` HEAD = "B7.1 PROTOCOL (pre-registered)"** — both confirm these lines are at protocol/closeout, not scientific-efficacy result; the ledger tags them accordingly (red-flags 9 & 10).

## GitHub raw-URL caution

The PM checklist links `raw.githubusercontent.com/.../<branch>/...` URLs. Branch-ref raw URLs are CDN-cached and can lag the true tip; all numbers in the ledger were read from local `git show <sha>:<path>`, and SHAs above are the authoritative provenance. Where a raw URL and the local `git show` disagree, the local SHA wins.

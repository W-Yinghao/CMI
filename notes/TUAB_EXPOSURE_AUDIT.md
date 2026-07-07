# TUAB EXPOSURE AUDIT (2026-06-21) — TUAB is NOT a clean pre-registered lockbox

**Verdict: `TUAB_LOCKBOX.md` cannot be honored as a pre-registered disjoint holdout. TUAB is RETRACTED as a lockbox
and must be treated as an already-examined external benchmark.** Do NOT "open" it as if clean.

## Evidence (git-verifiable on `exp/lpc-cmi`)
1. **TUAB was run BEFORE the lockbox.** The ROOT commit `fb2a878` already contains 13 TUAB result files:
   `results/TUAB_EEGNet.json`, `TUAB_EEGNet_seed{1..4}.json`, `TUAB_EEGNet_scldgn_s0.json`, `classical_TUAB.json`,
   `ladder_TUAB_seed{1,2}.json`, `rb_TUAB_seed1.json`, `rwdual_TUAB_seed1.json`, `vib_TUAB_b0p01.json`.
   `notes/TUAB_LOCKBOX.md` was created only later, in `1de7a12`. Its claim "*frozen BEFORE any TUAB result is
   computed*" is therefore false at the dataset level.
2. **The old runs were a full method comparison.** `results/TUAB_EEGNet.json` config:
   `configs=[erm:0, marginal:0.5, lpc_uniform:0.5, lpc_prior:0.3, cdann:1, dann:1]`, `protocol=loso`, EEGNet, 80
   epochs. So multiple methods (including the LPC family and adversarial baselines) were trained AND evaluated on
   TUAB recordings/subjects.
3. **TUAB numbers fed downstream claims.** `notes/calibration.md` reports `TUAB_EEGNet` ECE 29.3→24.8 / NLL
   1.679→1.200 as an LPC calibration win, and the SCPS "scorecard 4/4" in `cmi-empirical-findings` counts TUAB.
   So method framing/selection was informed by TUAB results — adaptive exposure, not a blind holdout.
4. **The lockbox spec is itself stale.** `TUAB_LOCKBOX.md` freezes the LPC nested selector (`λ∈{0,0.1,0.3}`) and
   the residual-decoder-CMI applicability gate — both now DROPPED / DIAGNOSTIC_ONLY (see `EVIDENCE_LEDGER.md`).
   Evaluating that frozen spec would test a method the project has retired.

## What the lockbox protocol does differently (not a rescue)
The lockbox prescribes **leave-one-recording-group-out** with class-spanning target batches, whereas the old runs
used **LOSO**. Different SPLIT, but the same underlying TUAB recordings/subjects were already trained on and scored.
A different partition of seen recordings is not a never-evaluated holdout.

## Required disposition (pick before any TUAB use; no method dev meanwhile)
- **(A) Demote TUAB to a normal external benchmark** — report it as already-seen, with the exposure stated; no
  "lockbox / preregistered holdout" language. Simplest and honest. **Recommended.**
- **(B) Construct a genuinely disjoint split** — ONLY if TUAB row/group/subject hashes prove a holdout of
  recording-groups that appear in NONE of the 13 historical runs (and method dev never saw their numbers). Requires
  the original TUAB cache + per-run subject/recording manifests; likely infeasible since prior runs were LOSO over
  the pool. If pursued, it needs a fresh pre-registration that freezes the CURRENT method (CITA-no-LPC, no gate, no
  LPC), not the retired one.
- **(C) Find a different, truly sealed dataset/split** for the external confirmatory test.

## Action items (archival; do not open TUAB)
1. Mark `notes/TUAB_LOCKBOX.md` SUPERSEDED (banner) — its freeze references a retired method and a false "pre-any-
   result" premise.
2. Any external write-up: TUAB is an already-examined benchmark, never "held-out / lockbox / preregistered".
3. If an external confirmatory holdout is wanted, it must be (B) with hash-proven disjointness on the CURRENT
   method, or (C). Decision deferred; TUAB stays sealed until then.

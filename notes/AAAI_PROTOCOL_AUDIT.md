# AAAI protocol/code audit — the pre-submission fix batch (started 2026-06-20)

Status snapshot at start of this batch (all pushed to `github.com/W-Yinghao/CMI`):
- **Headline (5-seed, nested no-oracle):** SCZ ERM 51.9±1.0 → CITA-nested 55.1±1.1 = **+3.1±1.3** (all 5 seeds +,
  1.4…5.1); PD 58.3±0.8 → 61.0±0.9 = **+2.7±1.0** (all 5 seeds +, 1.7…4.7). ± = seed std; **cohort-clustered
  bootstrap CI (2–4 cohorts) will be wider** — that is the CI to report.
- **Baselines (protocol-matched, 2-seed):** SPDIM **ties/slightly beats** CITA (SCZ 54.2/55.4 vs 54.0/55.8; PD
  63.6/62.3 vs 62.2/60.8); T3A ≈ native. ⟹ CITA's gain is over **ERM**, not over the strongest baseline →
  positioning is **methodological/diagnostic** (closed-form + CMI screen + lower leakage + safety gate).
- **Theory blockers fixed (commit fbc5939):** P0-2 (plug-in estimator, NOT an upper bound), P0-3 (DPI equality is
  `D⊥Y|Z`, not `Y=f(Z)`), P0-5 (GLS reference `p_d_ref=p_d`). P0-4 already handled.
- **CMI gate is load-bearing** in the confident-but-wrong regime (catches errors confidence rejection reverses on);
  score-direction wording corrected (reverse-ranking, not "blind").
- **TUAB sealed** (`notes/TUAB_LOCKBOX.md`) until this batch + freezing are done.

## The protocol/code-table batch (reviewer items), in priority order
**P1 — could move a HEADLINE number (do first):**
1. **Recording/session-grouped leakage split.** Current leakage probe uses a *random trial/window* split, which
   EEG within-recording autocorrelation lets the probe exploit → inflates measured `I(Z;D|Y)` and label
   separability. Fix: split source pool by **subject/recording** (whole recordings to train vs eval). Rerun the
   leakage numbers; compare random vs grouped. *If grouped shrinks the leakage gap, the leakage claim must be
   restated.* ← **STARTED HERE.**

**P2 — correctness (may shift numbers within noise):**
2. **Seed before backbone build.** `train_model` sets seed *after* `build_backbone` → methods may not share init;
   order-dependent. Fix: set py/np/cuda seed before build; clone one initial `state_dict` across methods.
3. **Double class-balancing.** classbal sampler + inverse-class CE weight + GLS reweight can stack. Each method
   must declare its *effective* training distribution; GLS methods use the raw sampler (or re-derive weights
   under the sampler-induced distribution).
4. **`drop_last=True`** drops rare domain×class cells (CMI is sensitive). Turn off / domain-class-aware batching.
5. **Multi-capacity leakage probes.** One MLP capacity → low leakage may be probe underfitting. Report
   linear / 2-layer MLP / strong MLP / kNN-or-HSIC; take the max detectable leakage.
6. **Signed MI/residual estimates + CIs.** Stop truncating negative estimates to 0 (upward bias); keep signed +
   CI in the statistics, truncate only in visualization.

**P3 — protocol completeness:**
7. Full-pipeline nested selection (encoder + alignment + gate selected together, not encoder-only then alignment).
8. Unified preprocessing manifest (immutable dataset hash); separate confirmatory table from exploratory results.
9. P0-5 rerun for any *marginal/dual-CMI* numbers the paper reports (headline CITA unaffected).
10. Offline / mini-batch / online-streaming TTA reported separately; never one "DG accuracy" header.

**Then:** freeze all configs → unseal TUAB once (class-spanning target batches per the lockbox).

## Decisions / findings log (appended as the batch runs)
- (start) Beginning P1.1 recording-grouped leakage split: add `--leakage_split {random,grouped}` to
  `run_scps_crossdataset`; grouped assigns whole subjects to the probe-train vs probe-eval split.

- (P2.2 ✅) seed py/np/torch/cuda BEFORE build_backbone (`_seed_all`) in config loop + nested `_train_on` — all methods/folds now share initialization (paired comparison, order-independent).
- (P2.4 ✅) `drop_last` now only when the tail batch would be size 1 (BN-safe); otherwise keep it so rare domain×class cells survive.
- NOTE: P2.2 changes initialization → the final frozen confirmatory run will re-derive the +3 numbers (robust across seeds, expected within noise). r11leak (P1.1) launched on OLD code is unaffected (it compares splits, not init).

- (P2.3 ✅) raw sampler now forced for ALL GLS methods (dualpc* AND dualc/dual+reweight) — no double-balancing. Headline (lpc_prior/erm) uses no GLS → unaffected.
- (P2.5 ✅) `leakage_probe(cap=...)` linear|mlp|strong + `--leakage_multicap` → report MAX detectable leakage (anti-underfit). r11mc (grouped split + multicap) launched.
- (P2.6) signed-MI + CIs handled at HARVEST time (report signed values + CI, truncate only in plots) — applied in the r11leak/r11mc harvesters.

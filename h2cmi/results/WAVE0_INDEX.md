# Wave 0 — index (real-EEG controlled-mechanism program)

Branch `exp/h2cmi-wave0-mechanism`, off the frozen terminal `5bc9bf0` (tag `h2cmi-review-p0-terminal`).
Two-commit discipline throughout: **result-only commit first, staged-interpretation commit second** — do
not squash. Frozen pre-registration: `WAVE0_FROZEN.md` (+ `W0.3_MECH_APPENDUM.md`, `W0.4_MECH_APPENDUM.md`,
frozen *before* their aggregates). All manuscript inserts are **STAGED, NOT APPLIED**.

| Wave | Result commit | Insert commit | Primary result | Checksums | Manuscript |
|---|---|---|---|---|---|
| W0.1 | `e75a0b0` | `5f17874` | determinism (27/27 self-replay) + admissible per-stage confusion; W2 decomposition re-confirmed to 4 dp | `B1A.sha256` | staged only |
| W0.2 | `658baf8` | `a256e2a` | FRSC not prevalence-invariant (0.08<0.42<0.80≪2.0); displacement ≠ utility | `w02_w05.sha256` | staged only |
| **W0.3** | `aa5031d` | `a1c9d07` | **metric-prior mismatch dominates W2** (oracle ρ_E wrong BA prior; transfer negligible; π_J-deviation +offsetting) | `w03.sha256` | staged only |
| W0.4 | `c79a041` | `81696c5` | larger n **exposes** (not fixes) BA prior harm (Δ P_J −0.044, sig; prior sharpening not TV convergence) | `w04.sha256` | staged only |
| W0.5 | `658baf8` | `a256e2a` | metric switch real but **specification-dependent** (V2P/FRSC sig; sleep none) | `w02_w05.sha256` | staged only |

Result docs: `W0.{2,3,4,5}_RESULTS.md`, `W0.1` result in `wave0_w2.report.json` + `WAVE0_MANUSCRIPT_NOTES.md`.
Staged inserts: `MANUSCRIPT_INSERTS_W03.md`, `MANUSCRIPT_INSERTS_W02_W05.md`, `MANUSCRIPT_INSERTS_W04.md`,
`WAVE0_MANUSCRIPT_NOTES.md` (W0.1).

## Conceptual ordering (for the rewrite)

- **W0.3** — oracle prevalence is the wrong BA decision prior (the cleanest empirical result; four-row
  mechanism table belongs in main text).
- **W0.5** — prevalence-aware decisions can help ordinary accuracy only under the right metric and
  specification.
- **W0.4** — more adaptation data does not fix the BA prior mismatch; it can expose it (appendix-first).

## Review checklist — forbidden vs required claims (run before applying any insert)

**Do NOT say:**
- π_J is the villain.
- night-to-night prevalence transfer is the W2 mechanism.
- ordinary accuracy wants prevalence.
- larger n makes π_J converge to ρ_A in TV.
- oracle ρ_E is deployable.
- sleep misspecification is causally proven by W0.5.

**Use instead:**
- metric-prior mismatch dominates W2.
- π_J deviation partially offsets oracle-prevalence harm in aggregate.
- prevalence-aware decisions can help ordinary accuracy when the deployment objective and model
  specification support them.
- W0.4 shows prior sharpening / decision-effect convergence, not TV convergence.
- ρ_E is an oracle diagnostic.
- natural sleep shows prevalence-aware decisions are not automatically decision-useful under real
  misspecification.

## Process lessons banked

Never launch a refilling babysitter without a passing probe-gate first; coverage guards use a proper count
(not `sort -un` on strings); detach embeddings before any transform fit (frozen encoder gets no backward);
single source of truth for output filenames (`wave0_fanout.output_path`), so monitors never check a stale
name; `bench_index` is never an identity (regression-gated).

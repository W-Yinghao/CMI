# Step 12 — Minimal-Information Phase Transition

Plan for the controlled `minimal_paired.py` study: how the empirical estimability of offline-TTA
harm/gain changes as minimal paired information (k target labels) is added. This is a study of the
**R1 → R2 boundary**, not an identifiability proof.

## Axes

**x-axis — minimal paired information:**
```
k target labels ∈ {0, 1, 2, 4, 8, 16, 32, 64}
```

**y-axis — estimability of harm/gain:**
- harm-sign prediction accuracy (decisive-and-correct fraction);
- target-risk / gain CI width (finite-sample, normal approx);
- abstention rate needed (fraction of repeats whose CI straddles 0 → cannot call the sign).

## Regimes compared

- **R0 source-only** — no target label; gain non-identifiable (TOS-1).
- **R1 target-unlabeled** — target X only; gain still non-identifiable (TU-2). This is **k = 0**.
- **R2 k-label target slice** — k labeled target trials; a labeled-slice risk estimate with a
  finite-sample CI, **under an iid sampling contract**. This is **k > 0**.

## Shift settings (simulator)

Four shift types with fixed true identity/adapted accuracy (gain = acc_adapt − acc_identity):
- `prior_shift_only` — TTA (prior correction) helps (gain > 0);
- `concept_shift` — TTA cannot fix concept (gain < 0, harm);
- `support_failure` — invalid transport (gain < 0, strong harm);
- `montage_transport_shift` — mild transport help (gain ≈ 0, small +).

`n_repeats = 50`, fixed seed.

## Expected reading

- **k = 0**: harm sign at chance (0.5), CI undefined, abstention 1.0 → the R1 boundary.
- **k > 0**: as k grows, CI width shrinks ~ `1/√k`, abstention falls, harm-sign accuracy rises — a
  phase transition around a small k for the strongly-harmful shifts.

## Hard boundary

The k-label slice estimates the **labeled-slice** risk. It equals full target risk only under an
explicit **iid sampling contract** (the k labels are a representative iid draw). We never write
"k labels identify full target risk" without that contract; every k > 0 record carries the
labeled-slice + sampling-contract caveat (`minimal_paired.py`).

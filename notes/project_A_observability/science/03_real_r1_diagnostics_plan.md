# Step 13 — Real R1 Diagnostics Plan

Step 12 found a **null** retrospective harm-prediction result (R0 bAcc 0.256, R1 bAcc 0.422 — both
below the 0.5 majority baseline). Possible explanations:

1. R0/R1 information is genuinely insufficient for target-harm prediction.
2. The Step-12 R1 diagnostics were too weak or partly missing.
3. Offline-TTA harm is driven by mechanisms not visible from target-unlabeled marginals.
4. The harm label is imbalanced: only 9 non-harmed cells among 54.
5. More target-unlabeled diagnostics may help empirically, but cannot make target gain identifiable.

Step 13 adds richer **label-free** R1 diagnostics to the real runner (`h2cmi/eval/harness.py`
prediction diagnostics + `h2cmi/run_real_audited.py` representation/prior diagnostics), stored in a
`r1_diagnostics` block of `raw_results.json`:

- prediction entropy / confidence (identity and adapted);
- identity-vs-adapt prediction disagreement;
- adapted pseudo-label histogram entropy and max class mass;
- target prior estimate entropy, L1 shift from uniform, L1 shift from source, max mass;
- source→target representation shift: MMD-RBF (median bandwidth), centroid distance, target→source
  kNN distance, off-source mass proxy (fraction of target beyond the 95th-percentile source→source
  nearest-neighbour distance).

**Boundary.** These are **R1** diagnostics — computed from target X / model predictions only, never
from target labels. They may be used for **retrospective empirical** harm prediction; they do **not**
identify target gain or target harm under R1 (TOS-1 / TU-2 stand). Any diagnostic that cannot be
computed is set to `null` and reason-coded in `r1_diagnostics_missing` — never silently 0.

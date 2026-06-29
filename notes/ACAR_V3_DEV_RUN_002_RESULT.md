# ACAR v3 — DEV run #002: TERMINAL RESULT = `DEV_STOP / NO_LOCKBOX_CONSUMED`

**Date:** 2026-06-29 · **Verdict:** **`DEV_STOP / NO_LOCKBOX_CONSUMED`** (no candidate passed the S2/S4 DEV gate).
This is the v3 development-stage terminal result. The held-out lockbox is **NOT** consumed; external Arm B is **NOT**
approached. The protocol commit (`817b04f`) and tag (`acar-v3-dev-design-v1`) are unchanged.

## Provenance / acceptance (all conditions met)
- SLURM job `866838` on `nodecpu05` (kernel matches the locked platform); **exit code 0**; elapsed ~3192 s (~53 min).
- `dev_out/` formed by atomic rename; **no leftover `dev_out.tmp`**; `manifest_sha256` recheck **OK**.
- `protocol_commit == 817b04f92d616b0b17bac223181c0f846f9209ac`; tag `acar-v3-dev-design-v1`; `clean_status_ok = true`;
  `env_lock_sha256 == 2cb61360…`; input-manifest bound; 7 cohorts (PD 3 + SCZ 4).
- Verdict ∈ {`SELECT`, `DEV_STOP / NO_LOCKBOX_CONSUMED`} → `DEV_STOP`.
- Full provenance manifest: `results/acar_v3_dev_run_002/DEV_STOP.json` (447 KB: per-fold FIT/CAL/EVAL hashes+counts+
  m/k/q, EVAL total/eligible/fallback, OOF digests, per-candidate diagnostics + S4 eligibility, per-cohort
  LoadedDumpManifests, best-fixed, env-lock + manifest SHA). Console + sbatch + input manifest alongside.

## Result (disease-macro S4 eligibility — why no candidate selected)
Gate thresholds: PD ≥1-action center-AUROC ≥0.60 · SCZ MAE ≤ C0 · width ≥30 % below C0 · coverage ≥0.15 ·
red_router >0 AND ≥ C0 · all q finite · S2 · dominance. C0 (v2 recipe) macro: red 0.0985, width 6.482, SCZ MAE 1.029.

| cand | eligible | red_macro | cov_macro | width_macro | PD AUROC | FAILED criteria |
|------|----------|-----------|-----------|-------------|----------|-----------------|
| C1 | ✗ | +0.1010 | 0.011 | 5.169 | 0.525 | coverage, pd_auroc, width_30pct_below_c0 |
| C2 | ✗ | +0.0016 | 0.006 | 2.490 | 0.570 | coverage, pd_auroc, red_not_below_c0, s2 |
| C3 | ✗ | +0.0652 | 0.010 | 2.350 | 0.545 | coverage, pd_auroc, red_not_below_c0 |

**Decisive, universal failures:** (1) **adaptation coverage collapses to ~0.6–1.1 %** (≪ the 15 % floor) — the
conformal `q` is so large relative to `|ΔR|` that `U_a < −δ` almost never fires, so the router abstains to identity; and
(2) **PD center-AUROC 0.525–0.570 < 0.60** — the predictor's center is a weak harm predictor on PD.

### Per-disease detail (`best_fixed = t3a` both diseases)
- **PD** (230 subj): every candidate has **coverage = 0** (the router NEVER adapts) → red ≈ 0; AUROC 0.525–0.570. PD is
  the binding constraint.
- **SCZ** (225 subj): signal is real — C1 AUROC **0.713**, C2 **0.739**, C3 **0.680** (all ≥0.60), red>0 (C1 +0.202,
  C3 +0.131) — but adaptation coverage is only **1.1–2.2 %**, far below 15 %.

## Interpretation
v3's HSCR redesign did **not** close v2's **measurement→control gap** on the DEV cohorts. It reproduces v2's pattern
under a stricter pre-registered gate: the *harm signal exists* (especially SCZ, AUROC ~0.7), but the *calibrated router
is not deployable* — coverage collapses (subject-clustered joint conformal `q` ≫ `|ΔR|`) and PD harm-prediction stays
near chance. This is a faithful negative result, exactly what the pre-registration + S2/S4 gate were built to deliver.
Consistent with v2 `MEASUREMENT_ONLY` and with the project boundary that the seven old cohorts are DEV/model-selection
only.

## Consequences (per protocol)
- v3 **stops at the development gate**. No held-out / lockbox cohort is read; external Arm B (binding G2, site-local
  coverage, harmful-rate, two-site) remains unauthorized and unexecuted.
- No auto-rerun, no threshold/seed/operating-point search to chase a pass (that would be post-hoc and is forbidden).
  Any future change is a NEW dated, separately-tagged protocol — never an in-place edit of `817b04f` or this result.
- The paper's v3 contribution is this **pre-registered negative DEV result** reinforcing the measurement→control gap
  thesis; the deployable-router claim is not made.

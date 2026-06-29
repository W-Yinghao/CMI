# ACAR v3 — CLOSEOUT

```
protocol tag   : acar-v3-dev-design-v1 @ 817b04f92d616b0b17bac223181c0f846f9209ac
result commit  : 9f4e83f   (result tag: acar-v3-dev-run002-dev-stop)
env lock       : 2cb61360a01af61001ac4a97e6269c16ee4d89c998122d22d557c7d7c84cab17
verdict        : DEV_STOP / NO_LOCKBOX_CONSUMED
external Arm B  : NOT RUN (unauthorized)
lockbox        : NOT CONSUMED
allowed future : NEW dated protocol only (e.g. acar-v4-…); never edit 817b04f or this result; no threshold/seed/δ search
```

ACAR v3 is **terminated at the development gate**. On the seven DEV cohorts (PD 230 + SCZ 225 subjects) none of the
pre-registered candidates C1/C2/C3 passed the S2/S4 admissibility gate. Decisive failures: adaptation coverage
~0.6–1.1 % (≪ the 15 % floor — the subject-clustered joint conformal `q` ≫ `|ΔR|`, so the router abstains to identity)
and PD center-AUROC 0.525–0.570 (< 0.60). SCZ shows a real harm signal (AUROC 0.68–0.74, red>0) but coverage only
1–2 %; PD never adapts (coverage 0). v3 therefore **does not close** v2's measurement→control gap. Details:
`notes/ACAR_V3_DEV_RUN_002_RESULT.md`; provenance: `results/acar_v3_dev_run_002/DEV_STOP.json`.

## Direction-2 lineage (for the manuscript Table 1)

| stage | tag / commit | status | one-line |
|-------|--------------|--------|----------|
| A0 / A0′ gate-falsification | (exp/lpc-cmi) | **DIAGNOSTIC_ONLY (closed)** | no source-free harm controller reduces deployed loss; density/CMI wrong-signed; rollback was label leakage |
| ACAR v2 | `acar-v2-protocol @ 9b2f0c1`; result `1528a94` | **MEASUREMENT_ONLY** | label-free action-conditional features predict negative transfer (G1✓); router not deployable (G2✗) |
| ACAR v3 (HSCR) | `acar-v3-dev-design-v1 @ 817b04f`; result `9f4e83f` | **DEV_STOP / NO_LOCKBOX_CONSUMED** | stricter pre-registered redesign fails the development S2/S4 gate (coverage collapse + weak PD center) |

## v3 S2/S4 result (Table 4 data — from DEV_STOP.json, disease-macro)

C0 (v2 recipe) macro: red 0.0985, width 6.482, SCZ MAE 1.029. Gate: PD AUROC ≥0.60 · SCZ MAE ≤ C0 · width ≥30 % below
C0 · coverage ≥0.15 · red >0 AND ≥ C0 · q finite · S2 · dominance.

| cand | eligible | red_macro | cov_macro | width_macro | PD AUROC | failed |
|------|----------|-----------|-----------|-------------|----------|--------|
| C1 | ✗ | +0.1010 | 0.011 | 5.169 | 0.525 | coverage, pd_auroc, width_30pct_below_c0 |
| C2 | ✗ | +0.0016 | 0.006 | 2.490 | 0.570 | coverage, pd_auroc, red_not_below_c0, s2 |
| C3 | ✗ | +0.0652 | 0.010 | 2.350 | 0.545 | coverage, pd_auroc, red_not_below_c0 |

Per-disease: PD coverage 0 (never adapts), AUROC 0.525–0.570; SCZ AUROC 0.713/0.739/0.680, red C1 +0.202 / C3 +0.131,
coverage 1–2 %. best-fixed = `t3a` both diseases.

## Manuscript plan (Direction-2 paper)

Working title: **"Predicting Negative Transfer Is Not Enough: The Measurement–Control Gap in EEG Test-Time Adaptation."**
Overleaf: `/home/infres/yinwang/AAAI_2026/nab_overleaf`. Present v3 as a **pre-registered negative development result**,
NOT an external validation study.

Sections: (1) Problem — EEG TTA can help or harm; generic shift diagnostics are not action-specific harm controllers.
(2) Estimand — `ΔR_a(B)=R_B(f_a)−R_B(f_0)`, action-conditional, paired, label-free at deployment. (3) Leakage audit —
why the A0/A0′ source-free safety gates are invalid / diagnostic-only. (4) ACAR v2 — measurement signal exists, control
fails; SCZ coverage diagnostic is "201/225=0.8933, two covered subjects short of the 203/225 pass threshold", not a
24/225 shortfall. (5) ACAR v3 — stronger HSCR design, frozen before the DEV read, C1/C2/C3 with strict S2/S4. (6) Result
— v3 stops at the DEV gate; no candidate passes; coverage collapse + weak PD center are decisive. (7) Conclusion —
unlabeled action-response observables can RANK harm, but calibrated closed-loop deployment remains unsolved under honest
subject-clustered simultaneous control.

Core tables/figures: (T1) lineage table above. (T2) v2 measurement — harm AUROCs of paired features / action
regressors. (T3) v2 control — q, abstention, coverage, G2 failure. (T4) v3 S2/S4 table above. (F5) mechanism figure —
measurement signal → risk regression → conformal joint max → abstention/control failure. Caption emphasis: *"strong SCZ
harm signal does not translate into usable adaptation coverage."*

## Future work boundary
Any continuation is a NEW research question and a NEW protocol (e.g. tag `acar-v4-dev-design-v1`), explicitly
post-DEV_STOP hypothesis generation — it must NOT reuse the v3 lockbox plan as confirmatory evidence and must NOT
re-threshold the v3 failure to compute external numbers. Candidate directions (new protocols only): (i) optimize an
abstention/constrained-routing utility directly rather than predicting `ΔR`; (ii) a hierarchical / online conformal
formulation reconciling subject-level events with batch-level utility; (iii) an information-limit argument that high
harm-ranking AUROC need not yield closed-loop risk reduction in label-free EEG TTA.

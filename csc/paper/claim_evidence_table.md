# CSC manuscript — claim → evidence contract

Every `\claimtag{...}` in `sections/*.tex` maps to a verifiable source here. A claim without a green
evidence row must be softened or cut. Numbers are traceable to committed artifacts on `origin/csc`.

| tag | claim (short) | evidence source | status |
|---|---|---|---|
| C-motivation | universal label-free concept detector is not attainable → certificate-with-abstention | Prop. 1 (csc/THEORY.md §1) | PROVEN |
| C-impossible | Z-only certificate cannot decide concept shift for ANY marginal | Prop. 1 (csc/THEORY.md §1), Fig 1 | PROVEN |
| C-routeA-neg | frozen confirmatory falsifies the Z-only dev core (both endpoints) | csc/results/confirmatory.json; notes/CSC_CONFIRMATORY_RESULT.md; tag dee8958 | VERIFIED |
| C-routeB3-pos | paired minimal-label certificate passes frozen unseen confirmatory (power 1.00; controls ≤α) | csc/results/b3_confirmatory_result.json; notes/CSC_B3_CONFIRMATORY_RESULT.md; tag 0595f64 | VERIFIED |
| C-protocol | one audit protocol yields both the negative and the positive verdict | both result JSONs + C6 notes | VERIFIED |
| C-routeA-design | Route A = source-anchored 3-state certificate (statistic + bootstrap null + support gate + atlas) | csc/THEORY.md §3–§4 | DESIGN |
| C-routeA-dev | dev core P_baseline power 0.83, 0/12 forbidden; 12 clusters → CP-UB ≈0.221 (not control) | notes/CSC_MANUSCRIPT_RESULT_MEMO.md; P1.5 map | VERIFIED |
| C-routeA-read | plausible engineered Z-only core can fail both endpoints on unseen data | Table 2 (routeA.tex) | VERIFIED |
| C-b3-idea | paired within-subject target-internal reference + few labels supplies the missing info; no source posterior | csc/mininfo/ B3 design; memory csc-concept-certificates | DESIGN |
| C-b3-test | centered ±0.5 coding required; h0 [Z,c] vs h1 [Z,c,c×Z_pc(r3)]; vote NLL diff, 3-fold | csc/mininfo/paired_conditional_test.py; paired_calibrated.py | DESIGN |
| C-b3-states | 5-state output, no COVARIATE_COMPATIBLE; fails closed | paired_calibrated.py; result JSON state set | VERIFIED |
| C-b3-envelope | primary 4 strong scenarios × m{20,30}; controls over all 6; label-noise+few-epochs pre-reg OUT of power | b3_confirmatory_manifest.json; result JSON | VERIFIED |
| C-b3-prov | base_seed 3000000, 5376 clusters, disjoint, provenance clean, red-teamed | b3_confirmatory_result.json code_provenance/seed_schedule | VERIFIED |
| C-b3-secondary | pure_conditional weak: 0.146 (m20) / 0.281 (m30), non-gating | result JSON per_cluster (secondary) | VERIFIED |
| C-b3-c6 | independent red-team reproduced C1–C5 without correction (CP |Δ| 2.8e-17) | notes/CSC_B3_CONFIRMATORY_C6_REDTEAM.md | VERIFIED |
| C-lim-synthetic | synthetic only; no real-EEG claim | scope statements throughout | STANDING |
| C-lim-envelope | development-informed envelope; noise/short-record OUT of power claim | manifest + result JSON | VERIFIED |
| C-lim-pure | pure-conditional tail weak, non-gating | result JSON secondary | VERIFIED |
| C-lim-pointwise | C2 = pointwise CP conjunction, not familywise CI | b3_confirmatory_manifest.json C2 note | VERIFIED |
| C-lim-paired | paired structure required; unpaired → abstain | Prop. 1 + method | DESIGN |

## Exact numbers (single source of truth for the tables)

### Route A confirmatory (tag dee8958, base 900000)
- dev: power 0.83 (CP-LB 0.56), forbidden 0/12 (12-cluster CP-UB ≈0.221)
- confirmatory: power 28/65 = 0.43 (CP-LB 0.326); forbidden 1/65, CP-UB 0.0709 > 0.05; headline_core_pass=false

### Route B3 confirmatory (tag 0595f64, base 3000000, 5376 clusters)
- C1: missing_pair 0/576, unequal_epochs_extreme 0/576
- C2 (kind×budget, n=288): clean m20 1/288 (0.0164), m30 3/288 (0.0267); paired_covariate m20 0 (0.0103), m30 1 (0.0164); paired_covariate_plus_label m20 1 (0.0164), m30 0 (0.0103); all others 0/288 (0.0103). worst = clean|m30 CP-up 0.0267. total control confirms 6.
- C3: max control cell 2/48 (clean|baseline|m30); 0 cells ≥6; 0 cells ≥3
- C4 (kind×budget, n=192): paired_concept & paired_concept_plus_cov, m20 & m30 all 192/192 = 1.000, CP-lo 0.9845
- C5: sampler_failures 0, boot_invalid 0, states ⊂ {NO_CONCEPT_EVIDENCE, CONCEPT_CONFIRMED, NEED_MORE_LABELS, INVALID_PAIR_STRUCTURE, UNIDENTIFIABLE}
- secondary paired_pure_conditional: m20 42/288 = 0.146 (CP-lo 0.113), m30 81/288 = 0.281 (CP-lo 0.238)
- C6: reproduced without correction; CP max |Δ| 2.8e-17; accounting 6+768+123 = 897

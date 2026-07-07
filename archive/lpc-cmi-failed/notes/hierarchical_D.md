# Hierarchical D — encoder penalty @site, decoder concept @subject (jobs hierD_*, 2026-06-13)

Decoupled the two domain variables in the cross-site runner (new `run_scps_crossdataset --dec_domain`): the
encoder penalty `I(Z;D|Y)` and encoder-leakage probe use `--domain` (here **cohort/site**), while the decoder
concept probe `I(Y;D|Z)` uses `--dec_domain` (here **subject**). Leave-one-cohort-out, ladder
`erm / lpc_prior / dualc`, 3 seeds.

| | bAcc | encLeak @site `I(Z;D_site\|Y)` | decRaw @subj `I(Y;D_subj\|Z)` | decRes @subj |
|---|---|---|---|---|
| **PD** erm | 59.8±0.5 | 0.202 | 0.120 | 0.094 |
| **PD** lpc_prior | 58.8±0.4 | **0.031** (6.5×↓) | 0.124 *(unchanged)* | 0.098 |
| **PD** dualc | 58.8±0.6 | 0.030 | 0.123 | 0.097 |
| **SCZ** erm | 52.7±0.5 | 0.446 | 0.225 | 0.168 |
| **SCZ** lpc_prior | 53.6±1.9 | **0.122** (3.7×↓) | 0.239 *(unchanged)* | 0.176 |
| **SCZ** dualc | 53.3±1.7 | 0.121 | 0.239 | 0.176 |

## Findings
1. **Encoder-at-site works**: `lpc_prior` cuts `I(Z;D_site|Y)` 4–7× → the representation becomes
   site-invariant (the right target for cross-hospital deployment), at accuracy parity.
2. **Decoder-at-subject is the `H(Y|Z)` artifact**, not concept shift: it stays ~0.12 (PD) / ~0.24 (SCZ)
   regardless of the encoder penalty (bigger for the harder near-chance SCZ). For the disease label
   `Y=g(subject)`, so `I(Y;D_subj|Z)=H(Y|Z)` = residual per-subject predictability (§1.2 degeneracy), not a
   domain-coupling the encoder term can address.
3. **Tension vs decoupling — the key structural observation.** With a *single* shared D, the encoder and
   decoder terms are coupled (tension theorem; flagship D=subject: removing leakage slightly *raises* the
   decoder CMI). With *different* D-granularities (site for encoder, subject for decoder) they are
   **essentially independent** — `I(Z;D_site|Y)→0` leaves `I(Y;D_subj|Z)` unchanged. The tension lives within
   one domain variable; it dissolves across granularities.

## Verdict on the assignment
- **encoder D = site**: correct and useful — exactly what cross-site deployment wants.
- **decoder D = subject**: for the *disease* label this is the degeneracy artifact (use decoder D = cohort to
  read true concept, which is ≈0 — the flagship `decoder_cmi_res_rw`). The decoder-at-subject term carries
  genuine concept information *only* when each subject spans both classes — the **paired med-state task**,
  where it is the confirmed subject-specific levodopa response (`I(Y;D_subj|Z)=0.05`, two-null validated,
  §3.6). So: encoder→site for invariance, decoder→cohort for concept; decoder→subject only for paired tasks.

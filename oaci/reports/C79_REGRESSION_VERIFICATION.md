# C79 Mode-R Regression Verification

Authoritative regressions ran on review implementation commit
`70c31bb7be239ee16d3f7f2730920dc1a87895f6` in `cpu-high` with 48 CPUs per job and
`/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python`.

```text
focused C79:  21 passed
C65-C79:     277 passed, 1 skipped
C23-C79:     684 passed, 1 skipped
full OACI: 1,612 passed, 1 skipped
```

The sole skip in each broad suite is registered and intentional:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

All four jobs completed with exit code `0:0`; stderr is empty in all cases.  The
stdout/stderr paths and SHA-256 values are recorded in
`c79_tables/regression_verification.csv`.

Jobs `893197`, `893199`, `893200`, and `893198` were successful pre-freeze attempts.
They are retained but superseded because the transparent pre-outcome adaptive-rule
qualification and one corresponding focused test were added before commit
`70c31bb`.  The authoritative rerun is `893206`, `893207`, `893205`, and `893204`.
No failed job or skip reason is hidden.

These regressions are CPU-only software verification.  They did not load seed-4
EEG, train, forward, re-infer, use GPU, create seed-4 artifacts, or open label views.

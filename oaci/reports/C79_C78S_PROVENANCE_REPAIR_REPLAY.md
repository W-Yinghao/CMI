# C79 Replay of the C78S Provenance Correction

## Verdict

The additive correction commit `dcd4c283573b4cdebe72c8ed3e181403232b28b7` passes `8/8`
independent checks.  It corrected an unsupported description of one intentional
pytest skip.  Slurm job `893168` established that the skip was the finalized-C78F
guard, not a C78S route-absence branch.

Changed files:

- `oaci/reports/C78S_ARTIFACT_MANIFEST.json`: `4d9f68422c34c68bcd7aefe442f7424265d528b840736dedc5b6efa48e72723c` -> `3a4eb0d42b87ec891a9bfecb92e0b4d3b9837f7cc7ea626d4f29865ce347faf1`
- `oaci/reports/C78S_FINAL_REPORT_RED_TEAM.md`: `72a97dc2f1c436a9ca6aee3625f5b516e5f696ffdb138cfd01114c3d373ef7a6` -> `0f10b08c11abb5eeb1a8d597bb7de0506354212f2294a70dcea994334fc827d3`
- `oaci/reports/C78S_REGRESSION_VERIFICATION.md`: `e67c8c90169da0247391097bd49c397ddb390fb1150548595b5fe249a1fb47d3` -> `cf1a7c66af191f8203c745a411906c7f9aaba47e48810f1e0b9db9c5e40f77e7`
- `oaci/reports/c78s_tables/final_report_red_team_checks.csv`: `f277641b84725d286516d1a745f8e2ac6c1ee5247e456db8c1a11ad133b892ef` -> `ad11c548bc8f0a894781d2014260e630d8c4a352bb2871c3193d9246d64f7ad1`
- `oaci/reports/c78s_tables/regression_verification.csv`: `34e7aa8921306ddad09371153a360313519856b46cde9740a4fc09180e7b0173` -> `fbebcd39fcdf03f1efe80cefac397594751620f34397e8aee876d256cd9db4ad`

No Python source, protocol, execution lock, C78S result, primary hypothesis table,
statistic, null, taxonomy, or outcome-dependent decision changed.  There was no
failed/replacement scientific job pair; `893168` was a dedicated provenance replay.
The correction therefore does not cause the C79 blocker.  The blocker is the C79
protocol timing and outcome-adaptive scope documented separately.

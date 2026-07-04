# CSC-realEEG-P1.3 — final pre-tag hardening + AUDIT record (NO tag, NO run)

Status: **P1.3 done + re-red-teamed + clean.** Executable-but-guarded; `--execute` fails closed without the
`csc-realeeg-v1` tag. Method locks byte-unchanged; synthetic tags `dee8958`/`0595f64` untouched. Tag + run each
need a separate reviewer go.

## The 4 reviewer blockers — all fixed
1. **[B1] sbatch shape.** The GitHub-raw "9 lines" was a rendering artifact; the actual file is a valid 67-line
   multi-line script (`bash -n` OK, 7 `#SBATCH` on their own lines, all prose `#`-commented). Made **provable**
   with `test_sbatch_wrapper_shape_is_valid_multiline_shell` (bash -n, line count, shebang, `#SBATCH --chdir`,
   no uncommented prose, freshness fields present).
2. **[B2] prereg 17-channel/FCz cleanup.** Removed every live 17-ch/FCz/`R^17` reference (§3, §10, §11 build-time
   reads, §13) → fully consistent with the frozen 16-ch `SM16_no_FCz`. Historical mentions of the 17→16
   transition retained, accurately labeled.
3. **[B3] result-payload provenance + wrapper freshness.** `execute()` payload now carries `manifest_provenance`
   (cache/bank/routeA/routeB3/engine/runner/cache/cache_metadata sha256), `frozen_refs` (expected_code_ref,
   git_head, expected_code_commit, git_status_clean, both synthetic tags, `synthetic_tags_untouched`,
   `genuine_contrast_descriptive_only`), `slurm`, `seed_schedule`. The sbatch freshness PYCHK verifies verdict +
   per_cohort + git-frozen + `expected_code_ref` + `synthetic_tags_untouched` + `base_seed==20000000` +
   `bank_manifest_sha256`.
4. **[B4] bootstrap honesty (reviewer option B).** A true subject-block cluster bound over shared-subject cohorts
   is subtle; rather than ship a mislabeled/risky estimator, the R1 bound is honestly reframed as a
   **COHORT bootstrap** (`cohort_bootstrap_upper`; old `subject_bootstrap_upper` removed). Every "subject-clustered"
   claim purged (engine docstrings + manifests + prereg); documented as a **cohort-level descriptive safety
   bound, NOT a formal subject-cluster bound** (a true subject-block bound noted as future refinement). The
   genuinely per-subject quantities (Δ_s CI, R5 leave-k-subjects-out) are unchanged and correctly labeled.

## Final re-red-team
Verdict ISSUES → the only findings were **residual "subject-clustered" strings inside the engine docstrings**
(my B4 test had grepped only the JSON manifests) + one stale "17" in §11 build-time reads. All fixed; the
reframe test now also scans `realeeg_engine.py` so the class of residue cannot recur. The red-team re-confirmed
the P1.2 fixes (V1 denominator/INCONCLUSIVE, V2 cohort bootstrap called, H1 engine-hash pin, A1 bare state) and
that execution is genuinely fail-closed (no run path; `--execute` exit 2; label_unit guardrail; gaming surfaces
all fail closed on mutation).

## Verification
JSON valid; `py_compile` OK; **dry-run 54/54 PASS**; `--execute` REFUSED **exit 2**; **tests 35/35 PASS**;
sbatch `bash -n` OK; engine re-pinned (`7ec08ad4…`).

## P1.3b docstring hotfix (reviewer's last pre-tag item)
The runner's top docstring still said "DRY-RUN ONLY / `--execute` is structurally REFUSED / implements NO path
that runs injections" — contradicting the now-real guarded `execute()`. **Fixed (docstring + usage only; no
logic/manifest/seed/cache/engine change):** "dry-run by default; guarded execute after freeze; `--execute`
disabled until checked out at `refs/tags/csc-realeeg-v1` with clean tree + matching pinned hashes; after
authorization runs the frozen bank and writes a fresh artifact; genuine contrast descriptive-only." Added
`test_runner_docstring_not_stale` (forbids the stale phrases, requires "guarded execute" + "csc-realeeg-v1").
Re-verified: py_compile OK, dry-run 54/54, `--execute` exit 2 (no tag), sbatch `bash -n` OK, **tests 36/36**.
New runner `sha256 = 7ca5906ea105944c9e8d1f048b307df1b3fe226b243d03fb785e89aaef585f93` (the runner is not
pinned in any manifest — the payload records it at run time; the engine remains the hash-pinned file).

## Next (each a separate go)
create tag `csc-realeeg-v1` (freeze) → then run the validation. Still NOT authorized here. No clinical/PD claim.

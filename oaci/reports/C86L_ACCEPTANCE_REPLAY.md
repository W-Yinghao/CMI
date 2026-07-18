# C86L — Content-Addressed Acceptance Replay (read-only)

**Gate reached**

```text
C86L_DEVELOPMENT_FIELD_CONTENT_ADDRESSED_AND_FULLY_REPLAYED_READY_FOR_C86D_PROTOCOL
```

Closes the acceptance gap from the PM's C86L verdict: content-address every input
and output artifact, replay the whole field semantically (not a spot-check), replay
Semantics B over all physical trials, and bind all frozen identities. **No field
artifact was modified.** Ran as SLURM job 902071 (cpu-high); it hashed the 45 GB of
prediction inputs.

## Result — all clean (acceptance_ok = true)

```text
INPUT REPLAY   1,944 / 1,944 prediction npz — every file SHA matches the C84F
               complete-field manifest per-unit sha256; registry + complete-field
               manifest SHAs replay; per zoo exactly 1 ERM + 40 unique OACI + 40
               unique SRC (canonical order ERM:0, OACI:1..40, SRC:1..40).
OUTPUT INVENTORY 1,891 artifacts content-addressed (944 pool + 944 contribution +
               labels.csv + context index + result manifest): path, bytes, sha256,
               schema, rows.
SEMANTIC REPLAY  944 / 944 contexts, 3,092,904 / 3,092,904 contributions:
               trial-ids + candidate-order exact; probabilities finite + normalized;
               pool carries NO label field; labels match oracle;
               correct mismatches 0; conf_bin mismatches 0;
               max |NLL err| 4.77e-7; max |confidence err| ~0; max |signed-cal err| ~0.
SEMANTICS B      4,773 / 4,773 physical trials each in exactly its 8 panel×seed×level
               contexts; label constant across all 8; each target trial counted once.
```

## Identity binding

```text
c86_effective_program_v3        c6b7e490e0f78f74f820428cee138782caff1dc0033422723593a7d8e3c5f77e
c84f_complete_field_manifest    cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8
c84_target_trial_registry       52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8
c84s_construction_view_labels   fdf36052d36ad9546cda06cbc567f68cdcced7ad08fd1311ab949471218b3134
c85u_acceptance_manifest        dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620  (identity-bound; NOT opened)
builder git blob                5b789642c69ab120ac47ab88492db2286d587abb   (oaci/active_testing/c86l_build.py)
acceptance git blob             eba4eb58f419a3cd0078b356778808c36faa0139   (oaci/active_testing/c86l_acceptance.py)
slurm script git blob           62a5750fc16a71dd21d66123b5eb98cfb3ef8f1a   (oaci/slurm_c86l_build.sh)
full artifact hashes            oaci/reports/C86L_ACCEPTANCE_MANIFEST.json  sha256 32de29ee223ccb73ea1da1c2d533f5f9215b721025f26d829df37c8c7130dacd
```

## Honest provenance boundaries (per PM)

```text
authorization: direct = PI-attested (授权 C86L);
               independently-replayable authorization message in repository = ABSENT.
split:         C86L CONSUMES the immutable C84S construction view (split_identity==construction);
               it executed no new C86_TARGET_SPLIT_V1.
isolation:     three separate filesystem DIRECTORIES; process/access-controlled isolation
               (active-client process ↔ query-server process ↔ sealed dirs) is a C86D
               requirement, NOT proven here.
```

## Status and boundary

The C86L development field is now content-addressed and fully replayed — a valid
authoritative input for C86D **protocol/client-server preparation** (GO). C86D real
active-policy execution, C86H, C87, and manuscript remain NOT authorized; C86H does
not auto-start C87. Real C86D execution requires a separate direct `授权 C86D`.

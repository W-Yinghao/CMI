# C86R2 Adult-Participant Interface and Untouched-Cohort Resolution

## Chronology

The additive C86R2 protocol was committed and pushed at
`0da4ec39f26ac4bc0d89035e9ad951f452217f05` before any additional
participant-level metadata was inspected or the public catalog screen was
expanded. Its SHA-256 is
`2e88e2fef7500b12ca8b3c5b19e6aab06df5a7f388781855b73793a1fe75df92`.

The implementation was introduced at
`bb91b82b3ac083d34c3ceb56687dea9c93fb4f76` and its catalog-count repair was
committed at `ccc1807479a2f3b3273991f6b3ca0ab5514184f1`. Before and during
C86R2, the protected counters remained zero for EEG, target labels, candidate
outputs, active acquisition, training/forward, registered synthetic results,
and GPU work.

## Adult Rule

The participant interface uses age at recording `>=18`. A deterministic subset
contains every canonical loader subject with a valid public age at recording
of at least 18 and excludes every minor or unknown-age subject. Exact loader-ID
mapping and at least 12 retained adults are mandatory. Aggregate age summaries,
consent language, and invalid year-like fields cannot establish eligibility.

## Yang And Kumar

Yang's primary descriptor reports an age range of 17-30 for the 62-person
cohort and 51 participants in the 2C task. The public NEMAR participant table
maps all 51 IDs to the same value, 29, while the paper supplement states that
age was removed during anonymization. The value is not accepted as a
subject-level age-at-recording map. All 51 2C IDs remain age unknown, so no
`Yang2025_2C_ADULT_V1` interface was created. The whole cohort remains an
age-mixed stress track. Sources: [primary descriptor](https://www.nature.com/articles/s41597-025-04826-y)
and [public NEMAR repository](https://github.com/nemarDatasets/nm000348).

Kumar's primary paper provides a mean and standard deviation but no public
all-subject adult inclusion rule. Its public participant table contains the
constant year-like value 2020 in the age column. That value is invalid as age
at recording, so all 18 IDs remain age unknown and Kumar stays an age-uncertain
stress track. Sources: [primary paper](https://academic.oup.com/pnasnexus/article/3/2/pgae076/7609232)
and [public NEMAR repository](https://github.com/nemarDatasets/nm000177).

## Catalog Replay

The search universe was frozen before screening:

| Catalog | Complete rows | Safe task/name candidates |
|---|---:|---:|
| installed MOABB imagery catalog | 53 | 53 historical audit rows |
| EEGDash public catalog | 824 | 71 |
| NEMAR public catalog | 760 | 78 |

The four NEMAR API pages and their combined 760-row projection are individually
hashed. EEGDash and NEMAR produced 149 candidate entries and 79 native datasets
after catalog-mirror deduplication. Splitting `ds007221` into its four native
task interfaces yielded 82 fully dispositioned interface rows. No eligibility
schema contains an outcome field.

## Newly Resolved Interface

The public OpenNeuro/NEMAR metadata for `ds007221` identifies four native task
interfaces. Only `OpenNeuro_ds007221_HYBRID_ADULT_V1` passes every locked rule:

```text
canonical subjects: sub-37 through sub-73
verified adults: 37 / 37
age range: 20-27
events: left_hand / right_hand
documented trials per subject: 600-640
channels: 69 native; all 11 common-interface channels present
license: CC0
historical project target/outcome access: none found
subject-list SHA-256:
  26520d5b53cfdb76915b552f4e3ac8d918deb584a430a6d03c4ac804fefa203c
```

The Graz interface fails the native binary rule because it includes feet and
rest. SSMVEP-MI includes motor-observation events and is not an exact native
binary interface. Hybrid-online has only 11 subjects. No class filtering was
used. Sources: [EEGDash field card](https://eegdash.org/api/dataset/eegdash.dataset.DS007221.html),
[NEMAR record](https://ww2.nemar.org/dataset/on007221), and
[OpenNeuro identity](https://openneuro.org/datasets/ds007221).

## Final Adult Set

Every passing interface is included. No post-screen cap was applied.

| Interface | Adult targets | Minimum trials | Role |
|---|---:|---:|---|
| `Brandl2020_CANONICAL_ADULT_V1` | 16 | 504 | primary untouched confirmation |
| `OpenNeuro_ds007221_HYBRID_ADULT_V1` | 37 | 600 | primary untouched confirmation |

The minimum two-cohort rule therefore passes with 53 target subjects. Yang and
Kumar retain their historical stress-track roles; Dreyer remains
development-only because historical project access is not certifiably absent.

## Field Replay

Both adult target interfaces share the fixed prospective
`C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3` interface with the three legacy C84
source cohorts. It uses 11 channels, 160 Hz, `[0,3)` seconds, and 4-38 Hz. The
same newly trained 648-action-unit zoo is required for both target cohorts, and
no target enters its own training.

Metadata arithmetic is:

```text
target subjects: 53
target contexts: 424
candidate-context slices: 34,344
unique candidate units: 648
training phases: 24
unit-cohort artifacts: 1,296
max-T draws per cohort: 65,536
```

These are planning values only. Future license replay, engineering canary,
candidate generation, and protected data access require later stage-specific
protocols, locks, and direct authorization.

## Effective Program

`C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V3.json` has SHA-256
`c6b7e490e0f78f74f820428cee138782caff1dc0033422723593a7d8e3c5f77e`.
It supersedes V2 only for downstream program resolution, preserves every
historical decision, and requires V3 for C86LP review. It creates no real-data
lock or authorization.

## Disposition

```text
C86_ADULT_UNTOUCHED_MULTI_COHORT_ELIGIBILITY_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW
```

C86LP, C86L, C86D, C86C/F, C86H, C87, active acquisition, real-data access,
and manuscript work remain unauthorized.

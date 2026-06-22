# H2CMI_METHOD_FREEZE

    target_only_eligibility_thread = CLOSED
    deployed_policy = metadata_only            # B2a metadata_ungated: rule table picks pooled /
                                               # canonical-CC / identity; UNSUPPORTED|UNKNOWN -> identity;
                                               # NO target-statistic eligibility gate; NO few-shot labels
    operator_set = frozen                      # {identity, pooled_empirical_diag,
                                               #  canonical = gen_oneshot_diag (frozen tie-break)}
    threshold_development = prohibited

Frozen method:  fixed-prior geometry adaptation + metadata-only operator routing + identity-by-default.

B2B conclusion, stated precisely (empirical power ceiling on the FROZEN score / calibration / sim
regime -- NOT a general impossibility theorem):
> At a route-null false-adaptation rate of ~0.10, the frozen change-of-variable evidence achieves
> only 0.15-0.16 route-level retention under the small-gain regime; therefore no threshold on this
> score can satisfy the preregistered 0.25 retention requirement.

Remaining pre-submission work = Stage V (frozen validation), NOT method search:
  V1  fresh seeds 100-119 confirmatory battery (claims 1-6); any non-replication SHRINKS the claim,
      never reopens development.
  V2  one frozen real-EEG external-validation block (BEETL/MOABB cross-dataset motor imagery):
      methods {identity, always_pooled, always_canonical_CC, metadata_only, one alignment baseline};
      routing inputs = deployable channel/layout/reference/sampling/device/task relations; target
      labels evaluation-only; unit = target dataset/site (subject nested).
Deferred (separate future work, NOT a rescue): few-labeled-target eligibility calibration
  (k in {1,2,4,8}).

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

## V1 fresh-seed confirmatory (seeds 100-119, 4900 rows, frozen analyzer)
Claims 1,2,3,6 REPLICATE (CIs exclude 0): C_prior_coupling + on cov/cov_prior/cov_cond_rot (modest,
+0.012..+0.032; smaller than the 3-seed dev); C_feedback null/neg (feedback not the harm); C_class_cond
prior+ (full +0.037/oof +0.068) and cov/rotation- (pooled wins); C_family + on rotation only
(+0.027..+0.032). Nulls exactly 0. REFINEMENT (dev did not replicate): C_responsibility is now
significant on the cov-family (+0.028..+0.048, CIs excl 0) -> oracle (labeled) responsibilities give a
modest held-out gain; orthogonal to the prior-M-step claim and does not touch the deployable
(label-free) method. b1a_confirm.report.json.

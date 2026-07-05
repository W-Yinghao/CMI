############ R0 THRESHOLD FRONTIER (allow counts /300; CP95u for nulls) ############
        tau | NULLcov NULLcl NULLlab rand |  POS POScov | NULLcov_cp95u
    0.00000 |      57     32       1    0 |   81     78 |        0.2312
    0.00114 |      57     32       1    0 |   81     78 |        0.2312
    0.00199 |      41     30       1    0 |   81     76 |        0.1736
    0.00233 |      33     25       1    0 |   77     73 |        0.1443
    0.00268 |      22     20       1    0 |   73     72 |        0.1031
    0.00311 |      15     16       1    0 |   69     68 |        0.0759
    0.00365 |      10     11       1    0 |   63     63 |        0.0559
    0.00418 |       7      7       1    0 |   61     56 |        0.0434
    0.00423 |       5      6       1    0 |   60     56 |        0.0347
    0.00537 |       1      2       0    0 |   56     48 |        0.0157
    0.00802 |       0      0       0    0 |   45     43 |        0.0099
    0.01156 |       0      0       0    0 |   34     33 |        0.0099
    0.01555 |       0      0       0    0 |   26     21 |        0.0099
    0.01824 |       0      0       0    0 |   13     14 |        0.0099
    0.02371 |       0      0       0    0 |    3      4 |        0.0099

  method_confirm baseline (/300): NULL_cov=57 NULL_cov_plus_label=32 NULL_label=1 random_label_control=0 POS_concept=81 POS_concept_plus_cov=78

############ R1-ELIGIBLE thresholds (NULLcov&NULLcl allow<=7, NULLlab&rand<=1, POSconcept>0) ############
  smallest eligible tau = 0.00418
    NULL_cov allow=7/300 (cp95u 0.0434) | NULL_cov_plus_label allow=7/300 (cp95u 0.0434)
    NULL_label=1 random=0
    POS_concept allow=61/300 (retention 75% of 81 method-confirms)
    POS_concept_plus_cov allow=56/300 (retention 72%)
    PREFERRED utility: POS_concept>=20 -> YES ; POS_cov>=15 -> YES
    thresholds meeting BOTH safety AND preferred utility: 98

  >>> R0 VIABILITY: VIABLE frontier exists (safety + some POS)

saved router_stage0_frontier.json

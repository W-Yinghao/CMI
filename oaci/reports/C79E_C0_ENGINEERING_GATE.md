# C79E C0 Engineering Gate

```text
target:                         4
seed:                           4
units:                        162 / 162
ERM / OACI / SRC:               2 / 80 / 80
checkpoint+sidecar replay:     162 / 162
strict-source rows:        746,496
target-unlabeled rows:      93,312
identity failed units:           0
max Wz/logit error:           0.0
max softmax error:            0.0
max hook-z error:             0.0
max repeat logits/z:          0.0 / 0.0
target training label reads:     0
scientific outcome reads:        0
same-label oracle access:        0
```

Measured resources:

```text
OACI/ERM job 893359: 0.551535 GPU h
SRC job 893360:      0.422568 GPU h
instrument 893447:   905.944 s CPU wall
```

Failed jobs `893354` and `893361` remain preserved with additive repairs. Both
repairs occurred without model-specific target outcomes and changed no locked
scientific object. C0 passes the engineering-only gate; it supplies no scientific
replication result.

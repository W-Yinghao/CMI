# S2P_16 - Route B Training Protocol

**Phase:** Route B B1 CBraMod 33ch budget-floor calibration.

**Launch status:** conditionally approved by PM and enabled by S2P_15 go/no-go. Training is limited to B1.

## Scientific Object

Route B asks:

> Does from-scratch CBraMod exit the frozen-transfer floor when trained at larger budgets on a pinned 33ch
> heterogeneous-reference TUEG substrate?

Allowed framing:

```text
CBraMod-only adaptive-channel high-budget calibration.
33ch fixed group-mixture sampling contract.
```

Forbidden framing:

```text
CodeBrain-compatible scaling.
CBraMod channel-invariance.
19-common-equivalent result.
subject-diversity isolation.
4000h tested.
```

## B1 Grid

```text
model = CBraMod
corpus = TUEG_processed_exact_33ch
sampling = fixed_group_mix over channel_order_hash x reference_scheme
budgets_h = {200, 500, 1000, 2000}
seeds = {0, 1}
training_runs = 8
H=4000 = held
```

The 200h point is retrained on the 33ch Route-B substrate. The old P1 200h 19-common checkpoint is historical
reference only and is not part of the Route-B curve.

## Model and Objective

Training uses the native CBraMod backbone and native mask semantics:

```text
input = B x 33 x 30 x 200
mask = generate_mask(B, 33, 30)
loss = MSE(reconstruction[mask == 1], input[mask == 1])
optimizer = AdamW
scheduler = CosineAnnealingLR
checkpoint selection = subject-disjoint pretrain-val loss
target labels used = false
```

The official CBraMod summary/ptflops 19ch reporting path is bypassed; it is a logging hardcode, not a model-forward
constraint.

## Launch Entry

Training array:

```text
s2p/slurm/route_b_b1_train_array.sbatch
```

Task manifest:

```text
results/s2p_route_b_33ch_contract/route_b_b1_training_tasks.csv
```

Training root:

```text
results/s2p_route_b_33ch_b1/
```

Each cell writes:

```text
results/s2p_route_b_33ch_b1/H{H}_s{seed}/best.pth
results/s2p_route_b_33ch_b1/H{H}_s{seed}/last.pth
results/s2p_route_b_33ch_b1/H{H}_s{seed}/train_log.jsonl
results/s2p_route_b_33ch_b1/H{H}_s{seed}/run_summary.json
```

## Downstream

Primary downstream after training:

```text
SHU-MI native32 frozen audit
```

Secondary sensitivity:

```text
SHU-MI 19-common mapped audit
```

The tiny released/random sanity in S2P_15 only authorizes the downstream path; it is not the final training result.

## Stop Rules

Stop and report if:

```text
channel/group contract artifacts change after launch
target labels appear before final downstream scoring
any training run hits unrecoverable NaN/Inf
exact window budget check fails
train/val subjects overlap
SLURM monitoring/cancel becomes unreliable
H=4000 is attempted in B1
CodeBrain is added to Route B
```

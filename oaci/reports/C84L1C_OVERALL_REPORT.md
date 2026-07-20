# C84L1C 完整报告

## 1. 执行摘要

C84L1C 是 fixed-panel level-1 source-support deletion 的授权工程 canary。
其唯一目标是验证已经前瞻锁定的 level-1 干预、训练身份、持久化工件和隔离
边界能否在 Lee2019_MI、Cho2017 和 PhysionetMI 三个真实数据接口上完整执行。

授权 replacement job `896066` 完成：

```text
datasets:                            3 / 3
training phases:                     9 / 9
candidate units:                   243 / 243
checkpoint replay:                 243 / 243
optimizer replay:                  243 / 243
sidecar replay:                    243 / 243
strict-source audit artifacts:     243 / 243
target-unlabeled artifacts:        243 / 243
```

固定删除、support gate、paired model initialization、accepted level-0 plan
replay 和全部持久化数值重放通过。所有 target-label、科学结果、outcome-driven
retention/retry 计数为零。

最终 gate：

```text
C84L1C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84FL2_REVIEW_REQUIRED
```

该 gate 只表示 level-1 工程 canary 完整通过。它不授权 C84FL2、C84F 或
C84S，也不构成 external-validity 科学证据。

---

## 2. 认识论与执行边界

C84L1C 属于：

```text
authorized engineering canary;
real source training/audit interface validation;
real target-unlabeled instrumentation validation;
fixed intervention and artifact identity replay;
future full-field reuse evidence subject to PM review and a new lock.
```

C84L1C 不属于：

```text
C84F complete field generation;
C84S scientific analysis;
target accuracy/calibration evaluation;
selector comparison;
Q1/Q2 inference;
label-budget analysis;
level-0 versus level-1 effect comparison;
external population confirmation.
```

没有 target construction/evaluation view、same-label oracle、target accuracy、
selector score、regret、Q1/Q2、label budget 或 cross-dataset scientific result
被计算或暴露。

---

## 3. 时间线和 Git 身份

```text
C84L1P readiness:              a0ec77b3a41084106713bf1f259e1daad2004607
historical authorization:      05bfca18c58b67b6cc0b7c5d57dfc7dc1036f8ea
historical failed job:         895928
C84L1R1 repair protocol:       e35ba0bfb412fbdcbc6fb127db05af1d91f51440
replacement implementation:   d0159d1b2db26d796ae3f9853329a5851aa93222
replacement execution lock:   afc5a6b5aedbb0e9d9b09acba0997657513e5268
C84L1R1 readiness handoff:     6e85428
fresh authorization:           60dd725026559f880dde71907eb69773d51961d9
result collector:              5dba965
result freeze:                 a5820eb
regression lifecycle repair:   ffdd7504d6fb0a31f8ad619f590cd7bbe4a3b4b8
```

历史失败 job `895928` 和 replacement job `896066` 是两个独立执行身份。
旧授权已消费且未复用；旧 partial artifacts 未复用。replacement 使用新
authorization、新 lock 和新的 content-addressed external root。

---

## 4. 授权、协议和执行锁

fresh authorization record：

```text
path:
  oaci/reports/C84L1C_PI_AUTHORIZATION_RECORD_V2.json

SHA-256:
  e287b40028ff9dc5373498b65f7316a443661de3e6548c23a456bedba40848fd

authorized stage:
  C84L1C engineering only

C84F / C84S:
  false / false
```

operative identities：

```text
repair protocol SHA-256:
  2e199f6f63dffd1b02c1e31102ed189e31bf6e4961465394230f8e9de1d4ddf0

canary protocol V2 SHA-256:
  6e6bcb6b60726c76c8db0afc48e954d0e4a1cf68bfd29796987bfd6828355616

execution lock V2 SHA-256:
  f9ebd88c72915bb41ba2d2d84a2a00c6748272021d48043c299bce52a1ad3813

runtime-bound objects:
  125 / 125

protocol bindings:
  5 / 5
```

additional fixed identities：

```text
20-channel montage:
  988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04

level intervention registry:
  89c4f366a222c1fe2ac31780bcbddbc9e59ff5afa4a779267abbd95429c41c17

243-unit candidate digest:
  db0c41a8caeb7d0fffd6938554c660eec36582596f12915b8b981c05bc092b95
```

---

## 5. Canary 范围

```text
datasets:
  Lee2019_MI
  Cho2017
  PhysionetMI

source panel:
  A

training seed:
  5

level:
  1

candidate zoo per dataset:
  1 ERM + 40 OACI + 40 SRC = 81

total:
  243 units / 9 training phases

canary targets:
  Lee 19
  Cho 24
  Physionet 106
```

所有模型使用 exact 20-channel physical montage、160 Hz、half-open `[0,3)`
epoch 和 20 x 480 input shape。无 interpolation、Fz substitution、zero fill
或 dataset-specific mask。

---

## 6. Fixed source-support deletion

Level 1 的身份为：

```text
C84_LEVEL1_FIXED_PANEL_LEFT_HAND_CELL_DELETION_V1
```

在 support graph 和全部 training plan materialization 之前，删除一个已注册
source-training subject 的所有 `left_hand` rows。删除不依赖 target identity、
target label、source performance 或任何 outcome。

| Dataset | Deleted cell | Rows deleted | Pre/Post trials | Retained cells | Minimum retained support |
|---|---|---:|---:|---:|---:|
| Lee2019_MI | subject 31 x left_hand | 50 | 1,200 / 1,150 | 23 / 24 | 50 |
| Cho2017 | subject 17 x left_hand | 100 | 2,440 / 2,340 | 23 / 24 | 100 |
| PhysionetMI | subject 103 x left_hand | 22 | 540 / 518 | 23 / 24 | 21 |

每个 fixed cell 在删除前存在且至少有 8 rows；删除后恰好一个
domain-by-class cell 缺失。deleted subject 的 `right_hand` support 和其他全部
observed cells 均超过注册 floor。source-audit 和 target rows 未被删除。

---

## 7. Paired training 与 level-0 replay

每个数据集的 level 0 与 level 1 使用相同 model-init seed rule、architecture、
optimizer、hyperparameters、epoch counts、checkpoint cadence 和 deterministic
settings。Level-specific plan 仅因 population/support identity 而独立 materialize。

结果：

```text
paired model-init hash:
  PASS for all 3 datasets and all 243 units

accepted level-0 plan replay:
  PASS for all 3 datasets and all 243 units

outcome-dependent seed selection:
  0
```

这验证了 deletion intervention 的工程身份，但没有比较 level-0 与 level-1 的
target performance。

---

## 8. 持久化工件和完整 gate

complete manifest：

```text
path:
  /projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v2/
  lock_f9ebd88c72915bb41ba2/C84L1C_COMPLETE_ENGINEERING_MANIFEST.json

SHA-256:
  3cf1366ccf40efc82a6bb2ffef56045e83c0f0e9670429973f23252371ad1c18
```

每个 unit 均冻结并从磁盘重载：

```text
checkpoint;
optimizer state;
engineering sidecar;
strict-source audit artifact;
target-unlabeled artifact.
```

覆盖：

```text
243 units x 5 artifact families = 1,215 successful byte/hash replays
```

外部成功根包含 `1,219` 个文件，共 `361,627,353` bytes。Git 中未提交任何
checkpoint、optimizer state、raw EEG 或 NumPy artifact。

---

## 9. 数值重放

```text
max in-memory float32 zW+b error:
  1.0967254638671875e-5

max persisted linear replay error:
  3.337860107421875e-6

locked linear tolerance:
  2e-5

max saved-softmax replay error:
  0

max repeated-logit error:
  0

max repeated-z error:
  0

locked strict tolerance:
  1e-6
```

线性重放只使用 C84L1R1 注册的 float32 reconstruction tolerance；strict
softmax/repeat checks 没有放宽。

---

## 10. Target-label 和科学结果隔离

```text
target-y access:                    0
target-label fields:                0
training target rows:               0
training target labels:             0
source-audit rows used in training: 0
target-outcome retention:           0
target-outcome retry:               0
target scientific metrics:          0
construction/evaluation/oracle:     0
```

Target loader 的结构性 label slot 未被索引、hash、repr、转换、汇总或记录。
Target artifacts 只包含 X-derived logits/probabilities/z、stable trial IDs、subject
和 session/run metadata。它们是 canary slices，不是 complete field。

---

## 11. 失败与重试治理

历史 job `895928` 在完成 73 units 后，由原 `1e-5` linear threshold 拒绝
`1.239776611328125e-5` 的 float32 reconstruction difference。该 attempt：

```text
preserved:                  true
authorization reusable:    false
partial artifacts reusable:false
target-y access:            0
scientific metrics:         0
```

C84L1R1 仅把 linear tolerance 改为 `2e-5`，并重新锁定。Job `896066` 从空的
replacement root 重训全部 243 units；没有复用 job `895928` 的 authorization
或 artifact。Replacement 本身没有 retry。

---

## 12. Scheduler、环境与资源

```text
job:                    896066
partition/node:         V100 / node42
allocation:             1 V100, 8 CPU, 64G
attempt runtime:        4355.285093888 seconds
last observed squeue:   1:12:18
application complete:   true
scheduler source:       squeue + application attempt ledger
sacct used:             false
exit code:              unavailable under squeue-only policy
```

应用完成由 complete manifest、attempt ledger final event 和全部 persisted
artifact replay 共同证明，而不是由不可用的 `sacct` 推断。

stderr 有 17 行 Cho continuous-stack edge-effect notices，均已披露为
nonblocking loader notices。Traceback/runtime-failure marker 为 0。

---

## 13. 回归验证

| Suite | Slurm job | Result | Stderr |
|---|---:|---|---:|
| focused | 896121 | 183 passed | 0 bytes |
| C65 | 896122 | 669 passed, 1 skipped, 3 deselected | 0 bytes |
| C23 | 896123 | 1,080 passed, 1 skipped, 3 deselected | 0 bytes |
| full OACI | 896124 | 2,004 passed, 1 skipped, 3 deselected | 0 bytes |

四个作业均为 `cpu-high`、48 CPU、96 GiB、GPU 0，并使用 exact C84C
environment。唯一 skip 是已 finalized 的 C78F。三个 deselections 是既有 C79
authorization-state tests，不隐藏 C84L1C 路径。C23 leading-numeric parser 已
覆盖 C34S suffix tests 和 C84L1C result-freeze tests。

回归调度同样只使用 `squeue` 监控，不调用 `sacct`。所有作业均离开队列，
stdout 有完整 pytest summary，stderr 为空。

---

## 14. 最终红队和 Git hygiene

最终红队：

```text
68 / 68 PASS
```

验证包括 authorization/lock/protocol/hash replay、全部 243 units、support
contract、paired identity、numerical replay、protected-state counters、失败重试
边界、回归、active-job audit、Git payload 和 remote identity。

在报告生成前：

```text
branch:                 oaci
HEAD == origin/oaci:    ffdd7504d6fb0a31f8ad619f590cd7bbe4a3b4b8
C84L1C active jobs:     0 by squeue
tracked forbidden data:0
largest tracked file:  21,936,073 bytes (< 50 MiB)
C84F execution lock:   absent
C84S execution lock:   absent
```

---

## 15. 复用边界和完整字段算术

经 PM review 后，可供未来 C84FL2 注册复用的 engineering objects 为：

```text
C84C level-0 model/state/source-audit units:   243
C84L1C level-1 model/state/source-audit units: 243
combined reusable units:                       486
combined reusable phases:                       18
remaining field units:                       1,458
remaining phases:                                54
```

Target-unlabeled canary artifacts 只覆盖：

```text
C84C level-0 contexts/slices: 3 / 243
C84L1C level-1 contexts/slices: 3 / 243
combined witnesses:           6 contexts / 486 slices
complete field:             944 contexts / 76,464 slices
```

这些 slices 只能作为 future complete-target instrumentation 的 subset replay
witnesses，不能描述为 complete target field。

---

## 16. 结论与下一步

C84L1C 证明以下工程命题：

> 在三个锁定数据集的 panel A / seed 5 条件下，固定 subject x left_hand
> source-support deletion 可以按注册身份完成 243-unit training、source audit、
> target-unlabeled instrumentation 和全部 persisted replay，同时保持 target-y
> 与 target scientific outcome access 为零。

它不证明 level 1 改善或损害任何 target endpoint，也不证明 selector transport
或 external validity。

下一阶段只能是 PM review 后的 C84FL2 protocol/implementation work，用于锁定
剩余 `1,458 units / 54 phases` 和 complete `76,464` target-context slices。
C84F 仍需要独立 execution lock 和新授权；C84S 需要完整字段冻结后的另一套
analysis lock 和新授权。

最终停止状态：

```text
C84L1C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84FL2_REVIEW_REQUIRED
```

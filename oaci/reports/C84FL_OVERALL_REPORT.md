# C84FL 完整报告

## 1. 执行摘要

C84FL 的任务是在已接受的 C84C 工程 canary 之后，为三数据集 fixed-zoo
完整字段实现 C84F adapter、manifest pipeline 和 scope-specific execution
lock，同时保持零新增真实数据访问和零科学结果访问。

C84FL 完成了以下 protocol/readiness 工作：

```text
C84C 结果与 manifest 精确重放；
243 个可复用 model/state/source-audit units 注册；
1,944 个完整 unit IDs 精确覆盖；
1,701 个剩余 unit IDs 和 63 个训练 phase 注册；
729 / 243 / 729 三个执行 wave 注册；
944 个 target contexts 和 76,464 个 candidate-context slices 核算；
完整 target-unlabeled artifact layout、barrier、schema、retry 和资源合同锁定；
协议在任何 C84F adapter、真实数据访问或 GPU 执行之前独立提交。
```

实现核对随后发现一个协议级 blocker：C84 协议列举 `level=[0,1]`，但没有
定义 fixed-zoo 场景下 level 1 的训练干预。现有 C84C 训练器只实现 panel A、
seed 5、level 0。历史 C78 的 level 1 依赖 target-specific
source-domain-by-class deletion；C84 又明确禁止 target-specific retraining，且
没有注册可替代的固定删除单元或 outcome-free 选择规则。

因此，剩余 1,701 units 中的 972 个 level-1 units 没有可执行的科学身份。
在这里补造规则、把 level 1 当作 level 0、删除 level 1，或只锁部分字段都会
修改已经注册的训练对象。C84FL 因而在 adapter 和 execution lock 创建前停止。

最终 gate 为：

```text
C84F_CANARY_REUSE_DATA_VIEW_IMPLEMENTATION_RESOURCE_OR_MANIFEST_RECONCILIATION_REQUIRED
```

这不是 C84C 失败，也不是资源失败。它表示完整 C84F field 暂时不能被科学地
实例化或授权。C84C 的 243 个 model/state/source-audit objects 保持有效并可在
未来 additive repair 后复用。

---

## 2. 里程碑范围与认识论状态

C84FL 属于：

```text
post-C84C engineering review;
no-new-real-data protocol and implementation readiness work;
prospective to all remaining C84 training and target instrumentation;
prospective to every C84 scientific selector/result;
engineering provenance and field-manifest design.
```

C84FL 不属于：

```text
C84F real execution;
C84S scientific analysis;
new EEG evidence;
new external-validity evidence;
new selector or hyperparameter search;
target-label analysis;
manuscript experiment.
```

C84FL 未读取 remaining-subject EEG、未读取任何 target label、未训练、未
forward、未使用 GPU、未计算 selector score，也未创建 C84F authorization
record 或 C84S execution lock。

---

## 3. Git 时间线与对象身份

### 3.1 接受的 C84C 基础

```text
C84C authorization commit:
  6949b62a51f7cd092c63be4ca24654e9ab7db068

C84C result commit:
  2f541e526deb79091ad164b0d37419941e6f662b

C84C final HEAD:
  f7bbd27579308e01ed5c0388cb728cc7417978ac

C84C result JSON SHA-256:
  bec3a8b205a3d13fdb848ce1f82f71f903d05a97f746fdae25b3b4cce40e67f0

C84C complete external manifest SHA-256:
  530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b
```

### 3.2 C84FL additive chronology

```text
f7bbd27  accepted C84C engineering base
  <
26f798e  C84F planning protocol committed before adapter implementation
  <
e141d2a  level-1 protocol/implementation blocker recorded
  <
16db04e  regression evidence and reconciliation handoff finalized
```

完整 planning protocol：

```text
path:
  oaci/reports/C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL.json

SHA-256:
  c6ab7dbed08711ceacd355183c4ad0f30d1fbef0804df86fc9159ab90327c28c

protocol commit:
  26f798ede818955927237b726e333590e80a13fa
```

该 planning protocol 锁定完整字段算术、复用边界、barrier、schema、retry、
资源上限和未来 runtime-lock 要求。它不是 C84F execution lock。协议中记录的
`C84_MULTI_DATASET_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED`
是未来成功执行的 field gate，不是当前 C84FL gate。

---

## 4. 继承的科学接口

```text
C84 external protocol V2 SHA-256:
  522e6fe8372f8c73741ed146a27068076db8c3d7087f4c4a36760fe0328b7c2f

C84 field protocol V4 SHA-256:
  eff7ebbc2e4f91830a3df1d679adfcae6eae2ab8a1e91c64ed28df7fce96aa12

C84 scientific protocol V2 SHA-256:
  dc33b22527352bd42989c26f6771b4a49dc1443d458962587ca3d70ad76dd631

C84R3 numerical repair SHA-256:
  cdbdb9a25dc29b6a37ac9eb65f130f44efa120042dfb7ddb140cf3db103ec196

20-channel montage SHA-256:
  988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04

candidate identity salt:
  C84_FIXED_ZOO_LEFT_RIGHT_20CH_V2
```

接口保持：

```text
task:          left_hand versus right_hand motor imagery
channels:      20 physical channels in one exact common order
sampling:      160 Hz
epoch:         half-open [0.0, 3.0), 480 samples
interpolation: false
zero filling:  false
target labels: unavailable during C84F
```

20-channel 顺序为：

```text
FC5 FC3 FC1 FC2 FC4 FC6
C5 C3 C1 Cz C2 C4 C6
CP5 CP3 CP1 CPz CP2 CP4 CP6
```

---

## 5. C84C 接受证据

成功 replacement job `895441` 在 V100 `node43` 上以 `0:0` 完成，运行
`01:46:19`。其工程结果为：

```text
datasets:                              3 / 3
training phases:                       9 / 9
candidate units:                     243 / 243
checkpoint/state/sidecar replay:     243 / 243
strict-source audit artifacts:       243 / 243
canary target-unlabeled artifacts:   243 / 243
```

数值重放：

```text
max persisted zW+b / linear error: 6.67572021484375e-6
locked linear tolerance:           1e-5
max saved softmax error:           0
max repeated-logit error:          0
max repeated-z error:              0
strict tolerance:                  1e-6
```

隔离计数：

```text
target-y access:                    0
target-label fields:                0
construction/evaluation view read: 0
same-label oracle access:           0
target scientific metrics:          0
training target rows/labels:        0 / 0
source-audit rows used in training: 0
target-outcome retention/retry:     0 / 0
```

失败 job `895366` 仍独立保留。它在 Lee2019_MI 阶段因原始 `1e-6`
float32 linear replay tolerance 拒绝 `2.86102294921875e-6` 差异而停止。
C84R3 只把 linear tolerance 改为 `1e-5`；softmax、repeat logits 和 repeat z
仍为 `1e-6`。成功 job 使用新 lock、新 authorization、新 external root，并未
复用失败 root 的 artifact。

---

## 6. C84C 复用边界

### 6.1 可复用对象

以下 243 units 可在未来完整 field 中按 manifest 精确重放后直接复用：

```text
candidate IDs;
model checkpoints;
optimizer states;
genealogy/state descriptors;
engineering sidecars;
strict-source audit artifacts.
```

它们对应三个完整 candidate zoos：

```text
Lee2019_MI   / panel A / seed 5 / level 0
Cho2017      / panel A / seed 5 / level 0
PhysionetMI  / panel A / seed 5 / level 0
```

### 6.2 不可作为完整字段复用的对象

C84C 的 target-unlabeled artifacts 只覆盖：

```text
Lee target 19;
Cho target 24;
Physionet target 106.
```

即：

```text
3 / 944 target contexts
243 / 76,464 candidate-context slices
```

这些 artifacts 是 canary slices，不是 243 units 的 complete-target
instrumentation。未来 C84F 必须为每个 unit 生成 one-all-target artifact，并
通过 trial ID 与数值 identity 精确重放三个 canary subsets。

---

## 7. 完整字段算术

### 7.1 Candidate field

```text
datasets:             3
source panels:        A / B
training seeds:       5 / 6
levels:               0 / 1
candidates per zoo:   81 = 1 ERM + 40 OACI + 40 SRC
zoos:                 3 x 2 x 2 x 2 = 24
training phases:      24 x 3 = 72
candidate units:      24 x 81 = 1,944
```

```text
C84C reusable:          243 units / 9 phases / 3 zoos
remaining:            1,701 units / 63 phases / 21 zoos
remaining level 0:      729 units
remaining level 1:      972 units, currently undefined
```

### 7.2 Target contexts

| Dataset | Targets | Contexts | Candidate-context slices | Canary slices | Remaining slices |
|---|---:|---:|---:|---:|---:|
| Lee2019_MI | 22 | 176 | 14,256 | 81 | 14,175 |
| Cho2017 | 20 | 160 | 12,960 | 81 | 12,879 |
| PhysionetMI | 76 | 608 | 49,248 | 81 | 49,167 |
| **Total** | **118** | **944** | **76,464** | **243** | **76,221** |

### 7.3 预注册 wave

| Wave | Scope | Units | Phases | Release basis |
|---|---|---:|---:|---|
| C84C reuse | panel A / seed 5 / level 0 | 243 | 9 | exact manifest replay |
| A | remaining panel A | 729 | 27 | engineering only |
| B0 | panel B / seed 5 / level 0 | 243 | 9 | engineering only |
| B1 | remaining panel B | 729 | 27 | engineering only |

Wave release 不得查看 target accuracy、calibration、selector score、regret、
Q1/Q2、label-budget 或任何 target-label-derived object。

---

## 8. 数据视图、freeze barrier 与 artifact 设计

### 8.1 Source views

每个 dataset/panel 固定：

```text
12 source-training subjects;
4 source-audit subjects;
source train/audit sets disjoint;
source-target overlap 0;
source labels allowed;
target rows used in training 0.
```

精确 subject IDs 位于
`c84fl_tables/source_view_contract.csv`。

### 8.2 Target-unlabeled registry

每个 target trial 只允许：

```text
dataset;
target_subject_id;
target_trial_id;
session;
run;
interface_id;
montage_sha256;
sample_rate_hz;
n_times;
finite-value flag;
target_label_field_count = 0.
```

不得包含 class label、label hash、label count 或 label-like metadata。

### 8.3 Model-field freeze

完整 target instrumentation 前必须先原子冻结：

```text
1,944 / 1,944 unique unit IDs;
72 / 72 training phases;
1,944 checkpoint replays;
1,944 optimizer replays;
1,944 sidecar replays;
1,944 strict-source audit replays;
0 training target rows/labels;
0 source-audit rows used in training;
0 target-outcome retention/retry.
```

### 8.4 Complete target instrumentation

锁定布局为：

```text
one all-target artifact per candidate unit;
1,944 artifacts total;
one target-context index per artifact family;
944 contexts;
76,464 candidate-context slices.
```

每行字段包括 unit/dataset/panel/seed/level/regime/genealogy、target subject、
trial/session/run、logits、probabilities、`z`、`Wz+b` 和 classifier parameters，
不包含 target label。

持久化重放阈值：

```text
zW+b versus logits:       <= 1e-5
saved softmax replay:     <= 1e-6
repeat logits:            <= 1e-6
repeat z:                 <= 1e-6
canary subset IDs/hashes: exact
```

---

## 9. Retry 和失败语义

```text
training failure before target instrumentation:
  same protocol/lock/rows/unit IDs/RNG only;
  failed attempt retained;
  new empty content-addressed root;
  no target value may be read.

model-field freeze failure:
  stop and reconcile;
  no target instrumentation.

target instrumentation failure:
  no retraining;
  no model-retention change;
  implementation-byte change requires additive repair and new lock.

scientific-outcome failure:
  not applicable in C84F because target labels/scientific scores are forbidden.
```

---

## 10. 资源估算

资源估算使用 C84C job `895441` 的实测值，而不是仅使用旧 safety estimate。

| Scope | Resource | Estimate |
|---|---|---:|
| C84C measured | calendar/GPU time | 1.771944 h |
| C84F remaining training | linear GPU time | 12.403611 h |
| Complete C84 training | linear GPU time | 14.175556 h |
| C84C valid external root | derived bytes | 361,267,953 |
| Complete model/state/source artifacts | projected bytes | 1,138,648,128 |
| Complete target instrumentation | projected bytes | 49,036,391,984 |
| Complete derived field | projected bytes | 50,175,040,112 |
| Raw download/cache | upper-bound bytes | 193,273,528,320 |
| Download + derived | projected bytes | 243,448,568,432 |

所有规划值均低于：

```text
GPU phase-hours:                    250 h
external download + derived data:  2 TiB
tracked Git file:                  50 MiB
```

资源本身不是当前 blocker。

---

## 11. 协议与实现核对

共执行 12 项 reconciliation checks：

```text
PASS:                         8
blocking FAIL:                2
expected nonblocking FAIL:    2
```

| ID | 检查 | 结果 | Blocking | 解释 |
|---|---|---|---|---|
| L01 | protocol enumerates levels `[0,1]` | PASS | no | arithmetic identity only |
| L02 | level-1 intervention/input rule exists | FAIL | yes | scientific training object unbound |
| L03 | training constructor accepts seed and level | FAIL | yes | canary constructor is level-0-only |
| L04 | runtime training seed parameterized | FAIL | no | full-field adapter intentionally not implemented |
| L05 | runtime covers all panel/seed/level cells | FAIL | no | canary scope filter remains intact |
| L06 | C84C is correctly level-0-only | PASS | no | no level-1 runtime evidence claimed |
| L07 | historical C78 level semantics replay | PASS | no | level 1 used target-specific deleted cell |
| L08 | C84 forbids target-specific retraining | PASS | no | historical rule cannot be imported silently |
| L09 | complete registry has 972 units/level | PASS | no | scope arithmetic exact |
| L10 | remaining registry has 729 level-0 and 972 level-1 | PASS | no | blocked scope quantified |
| L11 | C84F execution lock absent | PASS | no | correct fail-closed stop |
| L12 | C84FL protected access count is zero | PASS | no | no contamination |

L02 和 L03 是同一根因的科学合同与 executable manifestation。L04/L05
不是额外科学 blocker；它们记录在发现协议缺口后没有继续把 canary runtime
机械扩展成 full-field runtime。

---

## 12. Level-1 blocker 的根因

### 12.1 已知事实

```text
external protocol lists levels [0,1];
field arithmetic allocates 972 units to each level;
current C84C training constructor arguments are source/torch/np only;
current plan uses constant seed 5;
current canary filter selects panel A / seed 5 / level 0;
no operative C84 protocol key defines deletion, deleted_cell,
level_support or level_intervention;
historical C78 level 1 reads split["deleted_cell"];
C84 fixed-zoo protocol sets target_specific_retraining = false.
```

### 12.2 为什么不能推断 level 1

历史 C78 的 deletion cell 是 target-specific scientific object。C84 的
candidate zoo 在一个 source panel 上训练一次，然后共享给该 dataset 的所有
target subjects。如果直接复用 C78 规则，会重新引入 target-specific retraining；
如果选一个固定 cell，则必须预先定义哪个 domain/class cell、其支持条件、失败
行为和 RNG；如果把 level 1 复制为 level 0，则两个注册 levels 不再是不同干预。

这些选择会改变 source rows、support graph、training plan、unit semantics 和
科学比较对象，不能由 implementation milestone 临时决定。

### 12.3 影响范围

```text
complete level-0 units:       972
reused level-0 units:         243
remaining executable level0: 729
complete level-1 units:       972
defined level-1 units:          0
complete C84F lock possible:    no
```

即使 729 个剩余 level-0 units 可机械参数化，也不能创建声称覆盖完整 1,944-unit
field 的 execution lock。

---

## 13. 被拒绝的静默修复

C84FL 没有采用以下做法：

```text
把 level 1 当作 level 0；
从 field 删除 level 1；
只执行 972-unit level-0 partial field；
根据 target subject 重新引入 C78 target-specific deletion；
根据 source accuracy、support 或 target outcome 选择 deletion cell；
先实现 adapter、以后再决定 level-1 语义；
用成功 C84C authorization 覆盖新的 C84F scope。
```

这些做法分别会造成 intervention collapse、scope drift、incomplete manifest、
设计冲突、outcome-dependent choice、实现先于协议或 authorization scope creep。

---

## 14. Synthetic、红队和回归

### 14.1 Synthetic/contract calibration

12 个 scenario 中，11 个按合同通过；唯一预期失败为：

```text
S10 level1_intervention_bound:
  expected 1
  observed 0
  FAIL
```

其余 arithmetic、reuse、wave、context、target-label schema 和 fail-closed lock
absence scenarios 均通过。所有 synthetic rows 记录：

```text
real EEG access:       0
training/forward/GPU:  0
```

### 14.2 Slurm CPU regression

环境：

```text
Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python
partition: cpu-high
per job:   48 CPU / 96 GiB / 0 GPU
base:      e141d2a7531d15ac6a420bdb3ee9163395e57407
```

| Suite | Job | Result | stderr |
|---|---:|---|---:|
| focused C84FL/C84 | 895694 | 126 passed | 0 bytes |
| C65-C84FL | 895695 | 612 passed, 1 skipped, 3 deselected | 0 bytes |
| C23-C84FL | 895696 | 1,023 passed, 1 skipped, 3 deselected | 0 bytes |
| full OACI | 895697 | 1,947 passed, 1 skipped, 3 deselected | 0 bytes |

唯一 skip 是已完成的 C78F red-team finalization 条件；三个 deselections 是既有
C79P authorization-state tests。没有 C84FL/C84F test 被 skip 或 deselect。

最终报告补充后，本地 focused contract tests 为：

```text
15 passed
```

---

## 15. Protected state 与 Git hygiene

最终核对：

```text
C84F execution lock:          absent
C84F authorization record:    absent
C84F full-field adapter:      absent
C84S execution lock:          absent
remaining-subject EEG access: 0
target-label access:           0
C84FL training/forward:        0
C84FL GPU jobs:                0
active C84 jobs:               0
tracked raw EEG/weights/cache: 0
tracked files over 50 MiB:     0
```

OACI branch 在 C84FL handoff 时满足：

```text
HEAD == origin/oaci ==
  16db04e18d2c7c4008b12f41c4f9e8dc9acf7680
worktree clean
```

本完整报告作为后续 additive documentation commit 写入，不改变上述科学和
protected-state 结论。

---

## 16. 必需的后续 repair

PM 必须在任何 full-field adapter、execution lock 或真实 C84F access 之前，
提交一个 additive level-1 protocol。至少要绑定：

```text
1. level-1 intervention 的科学定义；
2. 每个 dataset/panel 的精确 source rows 或 deterministic cell identity；
3. domain/class support graph 与 minimum-support rule；
4. 无可用 cell 时的 fail-closed behavior；
5. seed 5/6 的 RNG 和 materialized plan derivation；
6. initialization、sampler、checkpoint cadence 和 genealogy；
7. unit-ID semantic compatibility 或必要的 prospective ID revision；
8. 与 historical C78 target-specific deletion 的相同点和不同点；
9. 该修复不使用 target labels/outcomes 的 timing audit；
10. 新 implementation、synthetic tests、runtime lock 和 fresh PI authorization。
```

修复后的合法顺序应为：

```text
C84FL blocker preserved
  < additive level-1 scientific protocol commit
  < parameterized full-field implementation
  < focused/synthetic/runtime red-team
  < new C84F execution lock
  < direct PI authorization for that unique lock
  < first remaining-subject EEG access or training
```

旧 C84C authorization 不得迁移到新 C84F lock。

---

## 17. Artifact index

主要对象：

```text
oaci/reports/C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL.json
oaci/reports/C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL.sha256
oaci/reports/C84F_PROTOCOL_TIMING_AUDIT.md
oaci/reports/C84FL_PROTOCOL_READINESS.md
oaci/reports/C84FL_FINAL_REPORT_RED_TEAM.md
oaci/reports/C84FL_REGRESSION_VERIFICATION.md
oaci/reports/C84FL_OVERALL_REPORT.md
oaci/reports/C84FL_OVERALL_REPORT.json
oaci/reports/C84FL_OVERALL_REPORT.sha256
```

关键表：

```text
c84c_result_identity_replay.csv
c84c_reusable_unit_registry.csv
c84c_canary_target_slice_registry.csv
complete_unit_registry.csv
remaining_training_registry.csv
wave_registry.csv
complete_context_arithmetic.csv
source_view_contract.csv
target_unlabeled_trial_registry_schema.csv
model_field_manifest_schema.csv
target_instrumentation_schema.csv
field_unit_descriptor_schema.csv
canary_subset_replay_contract.csv
retry_policy.csv
resource_estimate.csv
implementation_reconciliation_audit.csv
synthetic_calibration.csv
risk_register.csv
failure_reason_ledger.csv
regression_verification.csv
```

完整逐行身份、hash、schema 和 test evidence 以这些 machine-readable artifacts
为准。本报告不覆盖或修改原始 C84C/C84R3 objects。

---

## 18. 最终结论

C84FL 证明了 C84C reuse、完整字段算术、view isolation、manifest schemas、
resource envelope 和 fail-closed barrier 可以被一致地规划；它同时证明当前 C84
协议不足以实例化 972 个 level-1 units。

因此，当前唯一有效结论是：

```text
C84F_CANARY_REUSE_DATA_VIEW_IMPLEMENTATION_RESOURCE_OR_MANIFEST_RECONCILIATION_REQUIRED
```

该 gate 不授权 C84F、C84S、任何新真实数据访问、GPU 执行、target-label
provisioning 或科学结论。

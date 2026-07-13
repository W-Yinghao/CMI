# C81 整体报告

## 1. 执行摘要

C81 的目标是在冻结的 seed-3/seed-4 候选字段上，对文献中的模型选择基线进行同场比较，回答以下核心问题：

> 是否存在一个注册的零目标标签 selector，能够稳定降低 held-evaluation standardized regret，并达到或接近 C80 的 Q0 B=1 construction-label comparator？

C81 完成了协议、方法注册、synthetic calibration、冻结 selection、两次 pre-evaluation additive repair、授权绑定、审计和最终回归，但没有形成可接受的科学比较结果。

最终 gate 为：

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```

这一 gate 的精确含义是：

```text
不是“零标签方法没有效果”；
不是“严格源信息优于所有无标签方法”；
不是“一个 construction label/class 已被比较证明不可替代”；
而是 held-evaluation 结果在首个表写入前因实现层 schema defect 被阻断，
因此 C81-A/B/C/D 均不可判定。
```

C80E 仍是当前最新有效科学结果。C81 没有修改 C80 的 B*=1 结论，也没有为其增加文献基线对比证据。

---

## 2. 科学定位与认识论状态

C81 属于：

```text
existing-field comparative audit
post-C80 outcome-informed design
prospective only to the new C81 baseline computations
read-only frozen-candidate analysis
```

C81 不属于：

```text
independent confirmation
new-subject replication
new-dataset validation
external validation
new DG training campaign
open-ended baseline or feature search
deployability study
```

项目级科学对象保持为：

```text
Information-Constrained Model Selection
under Distribution Shift
```

C81 继续区分以下证据层级：

```text
reliability
!= conditional association
!= held-out prediction
!= group transport
!= low-regret actionability
!= exact-best localization
```

同时保留 C80 建立的进一步边界：

```text
source-relative regret qualification
!= low absolute regret
!= high top-1 accuracy
!= leave-target-stable label sufficiency
```

---

## 3. C80 输入背景

C81 的固定 labeled comparator 来自已接受的 C80E：

```text
B*_seed3:                         1 label/class
B*_seed4:                         1 label/class
cross-seed ordinal distance:      0
B=1 standardized regret seed3:    0.353383
B=1 standardized regret seed4:    0.373705
B=1 source-relative gain seed3:   0.426093
B=1 source-relative gain seed4:   0.423742
B=1 top-1 seed3:                  0.037842
B=1 top-1 seed4:                  0.038391
LOTO analyses moving B* to 2/4:  16 / 16
```

因此，C81 的问题不是重复估计 Q0 frontier，而是在同一字段、同一 candidate universe 和同一 evaluation object 上比较：

```text
I0   no target information
IS   strict-source information
IU   target-unlabeled outputs/geometry
ISU  source-calibrated target-unlabeled information
ILc  independent target-construction labels
IOr  descriptive evaluation ceiling only
```

C80 的 Q0 B=1 和 Q0 FULL 输出在 C81 中只能作为冻结 comparator，不能重新采样、调参或根据 C81 outcome 改写。

---

## 4. 冻结科学宇宙

```text
training seeds:          [3, 4]
primary targets:         [1, 2, 3, 5, 6, 7, 8, 9]
levels:                  [0, 1]
candidates/context:      81 = 1 ERM + 40 OACI + 40 SRC
primary contexts:        2 x 8 x 2 = 32
primary candidate units: 32 x 81 = 2,592
principal cluster:       target
seed role:               paired training factor
```

Target 4 始终是 engineering-only，并被机械排除于：

```text
primary fitting
primary selection summaries
regret/top-k estimands
null and max-T families
noninferiority tests
cross-seed/LOTO stability
final taxonomy
```

Trial rows、checkpoint candidates、ALine model pairs 和 Monte Carlo draws 均不得作为独立科学样本。

---

## 5. 方法注册与可用性

C81P 在 real outcome 之前注册了 34 个条目：

| 类别 | 注册内容 | 状态 |
|---|---|---|
| I0 controls | Uniform random、ERM anchor、final/midpoint OACI/SRC | 可执行 |
| IS | source-validation BAcc、source NLL、Source-LODO、historical F2 | 前两项可执行，后两项输入缺失 |
| IU | MSP、entropy、energy、IM、NuclearNorm、MaNo、SND | 可执行 |
| ISU | DoC、ATC、IWCV、DEV、IW-GAE、COT、COTT、ALine | DoC/ATC/COT/COTT/ALine 可执行；IWCV/DEV/IW-GAE 输入缺失 |
| ILc | frozen Q0 B=1/2/4/8/16/32/FULL | C80 固定 comparator |
| Diagnostic | Accuracy-on-the-Line | evaluation-only diagnostic |
| IOr | oracle-best denominator | ceiling only，不是 selector |

精确核算为：

```text
registered entries:                 34
feasible controls/selectors/etc.:   28
input-unavailable exclusions:        5
oracle-best denominator:             1
selection score paths in adapter:   19
primary zero-label representatives:  6
```

五个 prospectively excluded 方法为：

```text
S3  Source-LODO
S4  historical strict-source F2 selector
U8  IWCV
U9  DEV
U10 IW-GAE
```

它们被排除的原因是 frozen field 缺少 retrained source folds、per-candidate F2 descriptor、density ratios、domain discriminator 或 fitted group weights。C81 禁止为了纳入它们而训练新模型或拟合新 estimator。

### 5.1 固定 primary family representatives

| Representative | 固定方法 |
|---|---|
| R0 | Uniform random |
| R1 | ERM anchor |
| R2 | source-validation balanced accuracy |
| R3 | ATC |
| R4 | NuclearNorm |
| R5 | MaNo |
| R6 | COTT |
| R7 | SND |
| R8 | Agreement-on-the-Line |
| R9 | unavailable，无 faithful training-free importance-weighting member |
| R10 | frozen Q0 B=1 |
| R11 | frozen Q0 FULL |

注册后禁止根据 real outcome 把 secondary 方法替换为 primary representative。

---

## 6. 锁定比较问题与推断

C81 注册了五个问题：

```text
Q1: 零标签 primary representative 是否相对 strict-source comparator
    实现 material standardized-regret improvement？

Q2: 零标签 primary representative 是否在 regret 上 noninferior to Q0 B=1？

Q3: 结论是否依赖 objective：regret、top-1、top-5、top-10？

Q4: 比较是否在 seed3/seed4 和 LOTO target composition 下稳定？

Q5: 固定信息层级 I0 -> IS -> IU/ISU -> ILc 是否显示 field-specific transition？
```

关键推断规则为：

```text
material regret margin:              0.05
Q0 B=1 noninferiority margin:         0.05
principal scientific cluster:        target
exact target sign-flip vectors:      256
Q1/Q2 simultaneous family:           6 zero-label reps x 2 seeds
minimum favorable targets:           6 / 8
LOTO stability threshold:            12 / 16
alpha:                               0.05
```

Taxonomy 的优先级被锁定为：

```text
1. C81-E blocker
2. C81-D seed/target-composition heterogeneity
3. C81-A zero-label matches one-label frontier
4. C81-B zero-label improves source but not one-label frontier
5. C81-C no registered zero-label method materially improves source
```

因此，任何 protocol/input/implementation/view/dependence/provenance blocker 都优先于科学分类。

---

## 7. C81P 协议与 readiness

C81P 在没有 real C81 outcome 的状态下完成：

```text
protocol commit:          16a0d2eba4715a1cec78da6a79a182fd416a6629
protocol SHA-256:         cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
method registry SHA-256:  ef48ecf7fcc55188b78b0878d86f07f6239fe4f6c88bbc854829b3a1c7a1a120
implementation commit:    d17ffa62a63b929d36d03f74e4ce79794cd9601b
initial analysis lock:     541651c2ee3343c12d374a7322c91181a860a2c9
initial lock SHA-256:      b383707f58063c10f719194a995ab34094f6dcefe08c1e71837644db83dc94f1
readiness result:          89c6afb56bf0a386200d5ce4e54c0d14153bcde8
```

C81P evidence：

```text
C80 result hashes:           22 / 22 PASS
field/view objects:          11 / 11 PASS
target-level replay rows:   224 / 224 PASS
LOTO replay:                 16 / 16 PASS
synthetic calibration:       20 / 20 PASS
pre-execution red team:      43 / 43 PASS
final-report red team:       40 / 40 PASS
real baseline statistics:     0
evaluation-label reads:       0
same-label oracle accesses:   0
```

C81P gate：

```text
C81_AAAI_BASELINE_COMPARISON_PROTOCOL_LOCKED_READY_FOR_PI_AUTHORIZATION
```

---

## 8. 完整执行与修复时间线

### 8.1 初始 C81E preflight

PI 的直接语句 `现在我明确授权C81E` 按授权政策 commit `3d9dd76` 被接受。系统不要求 magic token 或重复 recital hash，而是将授权自动绑定到唯一 current protocol、lock 和 field/view manifest。

```text
authorization/preflight commit: 2697140
preflight gate:
  C81E_AUTHORIZATION_PROTOCOL_LOCK_AND_VIEW_PREFLIGHT_PASSED
```

Preflight 还解释了 C23 suite 相对 C80E 多出的四个测试：它们来自此前 glob 未匹配的 `test_c34s_artifact_hygiene.py`，属于 suite selector 修正，不是科学实现变化。

### 8.2 Attempt 1：source-shard schema failure

```text
attempt:                 C81E-SEL-001
Slurm job:               894878
stage:                   selection source-shard verification
status:                  FAILED_PRESERVED
selection manifests:     0
evaluation-label reads:  0
scientific statistics:   0
oracle/target4/GPU:       0 / 0 / 0
```

根因：C81 向共享 C74 verifier 传入所需字段子集；共享 verifier 将 `required_fields` 解释为完整字段集合，因此拒绝了包含额外合法字段的 registered superset shard。

C81R additive repair 保留显式 subset guard，并用完整 descriptor 做 hash、size、row-count、array-length、unit 和 trial alignment 验证；共享 C74 verifier 未修改。

```text
repair protocol:          6371b2220979b61cabfb105521036bb02f47aaea
repair protocol SHA-256:  ba0434b4ea7965691dafaf506547af64f851c57bdca330a0a5c88e4fa7ba1b15
repaired runtime guard:   570316310ccb2b0b2acb8a10952ac73431ffd2ae
final C81R lock:          bad8db494765f3f921443bf5e8cdd5db569861a9
C81R lock SHA-256:        3093201d3f2959d828cb9debb8a4aeb9252f5385b9e8f806445cff05307a8b1c
repair red team:          38 / 38 PASS
```

C81R gate：

```text
C81_SOURCE_SHARD_SCHEMA_REPAIR_LOCKED_READY_FOR_PI_REAUTHORIZATION
```

### 8.3 Attempt 2：selection 成功，descriptor replay 失败

PI 直接授权修复后的 C81E，绑定 commit `b2f9fca`。Selection job `894915` 随后成功完成：

```text
runtime:                       00:06:02
GPU:                           0
contexts:                      32
selection methods:             19
payload bytes:                 415,284
manifest self SHA-256:         4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519
payload SHA-256:               1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257
evaluation-label reads:        0
held-evaluation statistics:    0
selection recomputation:       forbidden
```

独立 freeze replay 随后发现 descriptor ABI defect：通用 verifier 假设所有 array 具有共同第一维，但合法 selection payload 同时包含：

```text
32 context rows
19 method IDs
32 x 81 candidate indices
32 x 19 x 81 scores
32 x 19 x 10 top-k selections
```

这不是 selection 科学值错误，而是对 heterogeneous registered shape 的错误验证。

C81R2 additive repair 仅增加 exact per-array shape map，保留 selection payload，禁止 recomputation，并重新锁定：

```text
descriptor repair commit:      5062f5ade0f45d6fd34f80556fb77470c2c6d717
descriptor repair SHA-256:     2acf6ecc179c739f73845d430f9eac9e9e83a83015370b1125dbe447b8b59272
repaired implementation:       225df1c2066b50abedec4bacf043f6359c715190
implementation SHA-256:        d5d8825e9c06994970de87728f73c6c8fef56af0cdd0f734746f1bd4863bf701
final C81R2 lock:              f82ffa4b147c0b1329a98649b898691cf1fdc983
C81R2 lock SHA-256:            13414dde0a88eb8a1a0810b3b36f25c718669d4cfe3178b871239eff6e292705
repair red team:                52 / 52 PASS
```

C81R2 gate：

```text
C81_SELECTION_DESCRIPTOR_REPAIR_LOCKED_READY_FOR_PI_REAUTHORIZATION
```

### 8.4 Attempt 3：authorized held-evaluation blocker

PI 直接语句 `授权 C81R2 修复后的 C81E 继续执行` 被 commit `102e466` 绑定到 C81R2 lock、field/view manifests 和冻结 selection。旧授权没有自动迁移。

```text
attempt:                       C81E-EVAL-001
Slurm job:                     894958
stage:                         held-evaluation result freeze
state / exit:                  FAILED / 1:0
runtime:                       00:00:03
GPU:                           0
evaluation views opened:       16
evaluation-label rows read:    4,746
construction views opened:     16
construction rows loaded:      4,470
contexts reached in memory:    32
method/control rows in memory: 672
scientific rows frozen:        0
```

Evaluation 只在 selection freeze 后打开，且 selection 没有重算。Same-label oracle view 未打开，target 4 未进入 primary path。

---

## 9. 最终 blocker 的技术根因

Held-evaluation job 在内存中完成 32 个 context 的 method/control row 构造，但在写入第一张表 `method_context_results.csv` 前触发 strict schema check。

Selector 和 held-evaluation ceiling row 的字段顺序为：

```text
standardized_regret
selected_utility
top1
top5
top10
coverage_top1
coverage_top5
coverage_top10
```

Analytic random-control row 的字段顺序为：

```text
standardized_regret
selected_utility
top1
coverage_top1
top5
coverage_top5
top10
coverage_top10
```

两类 row 的字段集合完全相同，差异仅为 dictionary insertion order。CSV writer 比较 `list(row)` 而不是字段集合或 canonical schema，因此抛出：

```text
C81 table schema drift
```

失败发生于 output file 打开之前：

```text
method-context rows frozen:       0
primary-comparison rows frozen:   0
Q1/Q2 executed:                   0
max-T executed:                   0
LOTO executed:                    0
nonblocker taxonomy executed:     0
```

这是一项 report-schema implementation defect，不是 endpoint 缺失、method score disagreement 或科学 negative result。

---

## 10. 为什么不能在当前协议下直接修复重跑

与前两次 failure 不同，job `894958` 已读取 evaluation outcomes。虽然这些值没有被持久化、打印或人工检查，但 locked failure policy 明确规定：

```text
post-evaluation implementation failure
=> authorization consumed
=> same-protocol patch-and-rerun forbidden
=> blocker taxonomy takes precedence
```

因此：

```text
不能把字段顺序修复视为无关紧要并静默重跑；
不能从进程内存或日志重建 672 rows；
不能重新使用 commit 102e466 的授权；
不能在原 protocol/hash/lock 下形成 C81-A/B/C/D。
```

任何未来恢复都必须由 PM/PI 单独批准新的 additive protocol、implementation lock 和授权，并明确其 post-outcome-access 状态。C81 本身不授权该动作。

---

## 11. 方法与结果核算

最终 34 个注册条目全部有明确状态：

```text
19 feasible selector methods:
  selection frozen;
  evaluation rows computed in memory but not persisted;
  no scientific result available.

B0 random and B5 held-evaluation ceiling controls:
  computed in memory;
  not persisted;
  no C81 result available.

7 Q0 comparators:
  original C80 artifacts remain valid;
  C81 comparative result blocked.

5 input-unavailable methods:
  not executed under pre-registered exclusion rules.

U16 diagnostic:
  not frozen before blocker.
```

以下 12 类 paper-ready table 均未生成，且被显式标记为 `NOT_EMITTED_BLOCKED_NO_FROZEN_SCIENTIFIC_ROWS`：

```text
primary_method_regret_table
primary_method_selected_utility_table
primary_method_source_relative_gain
primary_method_topk_table
zero_label_vs_strict_source_maxT
zero_label_vs_Q0_B1_noninferiority
seed_specific_method_results
cross_seed_stability
leave_one_target_method_stability
measurement_vs_decision_separation
target_level_catastrophic_failures
q0_budget_context
```

---

## 12. 科学结论处置

```text
Q1 zero-label versus strict source: BLOCKED
Q2 zero-label versus Q0 B=1:        BLOCKED
Q3 regret versus top-k/top-1:       BLOCKED
Q4 cross-seed and LOTO stability:   BLOCKED
Q5 information-class transition:    BLOCKED
```

因此不接受以下任何说法：

```text
零标签 selector 失败；
零标签 selector 匹配 Q0 B=1；
strict-source 是最佳零标签信息类；
一个 construction label/class 是不可替代的信息阈值；
target-unlabeled geometry 不可行动；
ATC/NuclearNorm/MaNo/COTT/SND/ALine 在该字段上成功或失败；
C81 证明了任何普遍不可能性或 deployability。
```

当前唯一合法的 C81 科学状态是：

```text
comparison unavailable under the locked C81 execution identity
```

---

## 13. 隔离、授权与范围审计

最终保护状态：

```text
target4 primary rows:              0
same-label oracle accesses:        0
training:                          0
forward:                           0
re-inference:                      0
GPU jobs:                          0
selection recomputation:           0
outcome-driven method replacement: 0
new methods/features/kernels:      0
BNCI2014_004 access:                0
seed5 access:                       0
```

需要精确区分：B5 是 held-evaluation oracle-best denominator，只在 evaluation stage 作为 descriptive ceiling 构造，且未持久化；它不是 selector，也不是 `same_label_oracle_view`。Same-label oracle route 的访问仍为 0。

授权政策按 commit `3d9dd76` 执行：PI 的直接授权语句足够，不要求 token 或重复 hash。每次 repair 后，旧授权均不自动迁移；最终授权被 job `894958` 消耗并已关闭。

---

## 14. Red-team 与回归

### 14.1 分阶段验证

```text
C81P pre-execution red team: 43 / 43 PASS
C81P final-report red team:  40 / 40 PASS
C81R repair red team:        38 / 38 PASS
C81R2 repair red team:       52 / 52 PASS
C81E scientific red team:    36 / 36 PASS
C81E final-report red team:  40 / 40 PASS
```

这些 PASS 证明 protocol replay、failure accounting、隔离和 claim boundary 一致；它们不把 blocked scientific result 变为有效 result。

### 14.2 最终回归

回归规模随 additive repair 和 blocker guard 增长，且每一阶段均保持 green：

| 阶段 | Focused | C65 slice | C23 suite | Full OACI |
|---|---:|---:|---:|---:|
| C81P | 43 | 412 + 1 skip + 3 deselected | 823 + 1 + 3 | 1,747 + 1 + 3 |
| C81R | 45 | 414 + 1 skip + 3 deselected | 825 + 1 + 3 | 1,749 + 1 + 3 |
| C81R2 | 47 | 416 + 1 skip + 3 deselected | 827 + 1 + 3 | 1,751 + 1 + 3 |
| C81E final blocker | 48 | 417 + 1 skip + 3 deselected | 828 + 1 + 3 | 1,752 + 1 + 3 |

最终回归对应 clean commit `d88e9c93c9a373c5662d9dcdc01e0c28b220335d`：

| Suite | Job | 结果 | stderr |
|---|---:|---|---:|
| focused C81E | 894970 | 48 passed | 0 bytes |
| C65-C81E | 894971 | 417 passed, 1 skipped, 3 deselected | 0 bytes |
| C23-C81E | 894972 | 828 passed, 1 skipped, 3 deselected | 0 bytes |
| full OACI | 894973 | 1,752 passed, 1 skipped, 3 deselected | 0 bytes |

唯一 skip 是已完成 C78F 的 guard；三个 deselection 是历史 C79P preauthorization-state tests。没有 C81 path 被 skip 或 deselect。

---

## 15. 提交链

核心 chronology：

```text
16a0d2e  C81P protocol lock
d17ffa6  locked baseline implementation
541651c  initial analysis lock
89c6afb  C81P readiness report
2697140  initial authorization and preflight
6371b22  C81R source-shard repair protocol
bad8db4  corrected C81R lock
4554e40  C81R readiness verification
b2f9fca  repaired direct authorization
5062f5a  C81R2 descriptor repair protocol
225df1c  heterogeneous descriptor implementation
f82ffa4  C81R2 execution lock
6118a13  C81R2 readiness verification
102e466  C81R2 direct authorization binding
8801b1c  post-evaluation blocker freeze
b4b71b9  scientific blocker red team
d88e9c9  main C81 blocker report
fc3f871  final regression/red-team/memory
5a40307  final handoff
```

所有 repair 均 additive，失败尝试被保留，没有 history rewrite 或 silent overwrite。

---

## 16. 最终项目状态

```text
C81P readiness: completed
C81R source schema repair: completed
C81R2 selection descriptor repair: completed
C81 selection freeze: completed and content-addressed
C81 held-evaluation result freeze: failed
C81 scientific comparison: unavailable
C81 final gate: C81-E_protocol_input_implementation_or_provenance_blocker
latest valid science: C80E
C82 authorization: absent
```

C81 的工程价值在于建立了一个完整、固定、可审计的 34-method comparison design，并验证 selection 可以在物理隔离的 evaluation view 之外运行。C81 的科学价值目前仅是一个严格的证据边界：由于结果未冻结，不能对任何 zero-label baseline 作成败判断。

---

## 17. 权威工件索引

协议与注册：

- [C81 protocol](C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json)
- [C81 method registry](C81_BASELINE_METHOD_REGISTRY.json)
- [C81P readiness](C81P_PROTOCOL_READINESS.md)
- [C81R source-shard repair](C81R_REPAIR_READINESS.md)
- [C81R2 descriptor repair](C81R2_REPAIR_READINESS.md)

执行与结果边界：

- [C81 authorization and preflight](C81E_AUTHORIZATION_AND_PREFLIGHT.md)
- [C81R2 authorization binding](C81E_C81R2_AUTHORIZATION_BINDING.md)
- [Machine blocker result](C81_FROZEN_FIELD_BASELINE_COMPARISON.json)
- [Final C81 report](C81_AAAI_BASELINE_COMPARISON.md)
- [Scientific red team](C81E_SCIENTIFIC_RED_TEAM.md)
- [Final-report red team](C81E_FINAL_REPORT_RED_TEAM.md)
- [Final regression](C81E_REGRESSION_VERIFICATION.md)
- [Project memory through C81E](OACI_EEG_DG_PROJECT_MEMORY_THROUGH_C81E.md)

关键表：

- [Failure and repair ledger](c81e_tables/failure_and_repair_ledger.csv)
- [Blocker evidence](c81e_tables/c81e_blocker_evidence.csv)
- [Method accounting](c81e_tables/method_failure_and_availability.csv)
- [Paper-table status](c81e_tables/paper_ready_result_table_status.csv)
- [Registry execution ledger](c81e_tables/registry_execution_ledger.csv)
- [Result artifact manifest](c81e_tables/result_artifact_manifest.csv)

---

## 18. Stop rule

C81 已结束并等待 PM review。当前状态不授权：

```text
C81 repair/re-execution
C82
new baseline or method search
active acquisition
same-label oracle
seed5
BNCI2014_004
new subjects/datasets
training or re-inference
manuscript experiments
```

任何后续动作必须使用新的、additive、scope-specific protocol 和 execution lock，并由 PI 重新直接授权。

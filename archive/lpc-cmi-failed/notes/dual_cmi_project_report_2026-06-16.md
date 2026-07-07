# Tri-CMI / Dual-CMI 项目阅读报告（2026-06-16）

## 1. 总体判断

这个项目的主线不是“泛泛做 EEG DG”，而是一个很明确的信息论框架：在零校准 cross-subject / cross-site EEG domain generalization 中，学习表示 `Z=f(X)`，让它保留任务标签 `Y`，同时尽量去掉给定标签后的域信息 `I(Z;D|Y)`。主方法是 LPC-CMI：

```text
E KL(q_psi(D | Z, Y) || pi_y(D)),  pi_y(D)=p(D|Y=y)
```

当前项目里要清楚区分三层东西：

1. **主线方法 `lpc_prior`**：只压 encoder 侧 `I(Z;D|Y)`，是目前最稳定、证据最强的部分。
2. **dual-CMI 线**：同时考虑 encoder `I(Z;D|Y)` 和 decoder `I(Y;D|Z)`。理论分解是对的，但 naive 同时最小化会在 label shift 下互相拉扯。
3. **concept-shift / predictor-stability 线**：decoder CMI 更适合做“域相关标签规则是否改变”的诊断，而不是 naive 平均准确率提升手段。CE-residual Route C 是诊断线；后续实现的 JS-consistency `dualpc` 才是当前同时控制 `P(z)` 与 `P(y|Z)` 的候选算法。

我的结论：**论文最稳的贡献应定位为 conditional leakage removal + tension theorem + concept-shift diagnostic，而不是 naive dual-CMI 的准确率提升。** 如果要把 dual 纳入主方法，应该使用后续实现的 GLS-factorized `dualpc`：用 `I_w(Z;D|Y)` 控制参考 mixture `P(z)`，用 JS consistency 控制 `P(y|Z)`。naive `dual` 更适合做 tension 反例/消融；CE-residual 版 `dualc` 更适合作为诊断/消融，而不是最终主算法。

## 2. 项目结构和关键文件

- `cmi/methods/regularizers.py`：核心 posterior-KL 估计器、`DomainPosteriors`、`dec_cmi`、`dec_cmi_residual`。
- `cmi/train/trainer.py`：所有方法的训练调度。`ALL_METHODS` 包括 `erm`、`lpc_prior`、`iib`、`dual`、`dualc`、`dualpc`、`graphcmi` 等。
- `cmi/eval/metrics.py`：分类指标、encoder leakage probe、marginal `P(z)` probe、decoder JS / residual leakage probe 和 decoder-validity 汇总。
- `cmi/run_loso.py`：LOSO 主入口，支持 `--configs dual:lam:gamma`、`dualc:lam:gamma`、`dualpc:lam:gamma`、`--label_correct`、`--reweight_dual`、`--dec_margin`。
- `cmi/run_scps_crossdataset.py`：SCPS leave-one-cohort-out，支持 encoder `D` 和 decoder `D` 分离。
- `cmi/run_glsvae.py`、`synthetic/gls_vae.py`：Route A / GLS-VAE concept-shift test。
- `synthetic/dual_cmi_v2.py`：dual-CMI 的 synthetic 验证，含 held-out CMI probe 和 exact tension sweep。
- `synthetic/dualpc_validation.py`：CPU-only production-trainer harness，用来验证 `dualpc` 的 `P(z)` / `P(y|Z)` probes、source-only selection 和 null/concept gate。
- `notes/DUAL_CMI_THEORY.md`：dual 理论主文档。
- `notes/CONCEPT_SHIFT_SECTION.md`：目前最完整的 concept-shift 诊断叙事。
- `notes/CMI_TECHNICAL_REPORT.md`：截至此前的技术总报告，但部分状态已被新结果更新。

## 3. 当前方法分辨

### 3.1 `lpc_prior`: 主线 encoder CMI

目标是压低 `I(Z;D|Y)`，即 class-conditional 的域泄漏。训练是两步交替：

1. Step A：固定 encoder，训练 `q(D|Z,Y)`。
2. Step B：更新 encoder 和分类头，优化 `CE + lambda * KL(q(D|Z,Y)||pi_y(D))`。

它的证据最强：在多数结果中，`leakage_kl` 能下降 10x 到 100x，准确率通常是 parity 或小幅波动。它是项目里最适合当“主方法”的部分。

### 3.2 `marginal` / `chain` / `lpc_uniform`

这些是关键消融：

- `marginal` 近似压 `I(Z;D)`，容易擦掉 label-relevant domain signal。
- `chain` 压 `I(Z;(D,Y))`，更容易出现 Y-erasure。
- `lpc_uniform` 用 uniform prior，能对照 CDANN 的目标，但在 `p(D|Y)` 不均衡时是错配目标。

这些消融要保留，用来证明为什么必须 conditioning on `Y`，以及为什么 `pi_y(D)` 不是装饰。

### 3.3 naive `dual`

当前 `dual` 的形式是：

```text
CE + lambda * I(Z;D|Y) + gamma * I(Y;D|Z)
```

实现上，encoder 项用 `post.reg("lpc_prior", z, y)`；decoder 项现在用单独的 `q_yz` probe 与 `h_ydz` probe 的 CE 差值 `post.dec_cmi(z,y,d)`，这比旧版把 task CE 直接拿来构造 decoder CMI 更干净。

理论上，dual 分解是正确的：

```text
I(Z;D|Y) - I(Y;D|Z) = I(Z;D) - I(Y;D)
```

但在 label shift 下，如果把 `I(Z;D|Y)` 压到 0，则 `I(Y;D|Z)` 会被 `I(Y;D)-I(Z;D)` 强制拉高。CPU 理论验证刚跑过，identity 在 3000 个随机离散 joint 上最大误差 `2.442e-15`，GLS reweight 后 tension 消失。

### 3.4 `iib`

`iib` 是 decoder-only，主要处理 `I(Y;D|Z)` 方向。它不压 encoder leakage，所以结果里经常保持高 `leakage_kl`。但在 ADFTD 这种 subject-level decoder signal 大的设置里，`iib` 的准确率最稳定，这说明 decoder 项确实有信息，但未必是我们想要的 cross-site concept shift。

### 3.5 Route B: `--reweight_dual`

Route B 试图用 GLS 权重 `w_i=pi*(y_i)/pi_{d_i}(y_i)` 同时重加权 Step A、encoder CMI 和 decoder CMI，让估计发生在 `I~(Y;D)=0` 的 reweighted measure 下。

这条线实现上是 gated 的：只对 `method == "dual"` 生效。当前代码里 encoder KL 用 `reference="marginal"`，即 reweighted 后对齐 `p(D)`；这比旧笔记里写的 uniform 更精确，因为 domain marginal 不一定 uniform。

结果看，目前 Route B **没有明显改变 naive dual 的行为**：MUMTAZ 上按预期基本 inert；ADFTD 多 seed 上 `decoder_cmi` 仍在约 `0.24-0.30`，没有解决 naive encoder 压低后 decoder signal 上升的问题。

### 3.6 Route C: `dualc`

Route C 是 CE-residual 版 dual 线，当前更适合作为 concept-shift 诊断和消融：

```text
GLS-weighted encoder CMI + gated residual decoder CMI
```

decoder residual 是：

```text
CE(h0) - CE(h)
```

其中 `h0(Y|Z,D)=u(Z)+b_D` 只允许 domain 改变 intercept / calibration，`h(Y|Z,D)` 是 full domain decoder。这个差值更接近“domain-dependent decision boundary”，比 raw `I(Y;D|Z)` 更少受 label prior / calibration 影响。`dualc` 又通过 `relu(residual - tau)` gate 避免 null 情况下乱罚。

但要注意：我刚跑的 `synthetic/route_c_positive_control.py` 显示 Route C 有 power，但 subject-degeneracy 并没有完全消失：

```text
null-prior     raw 0.109  residual 0.017
concept        raw 0.219  residual 0.218
subject-degen  raw 0.707  residual 0.307
```

所以 residual decoder CMI 对 subject-degenerate disease labels 仍不能直接当 concept shift，必须配 null gate / permutation null，并且最好在 cohort/site 级 `D` 上读。

### 3.7 `dualpc`: factorized `P(z)` + JS `P(y|Z)`

后续实现的 `dualpc` 是当前最接近“同时优化 `P(z)` 和 `P(y|Z)`”的候选主算法：

```text
CE(Y|Z) + lambda * I_w(Z;D|Y)
       + gamma * [ JS_w(h_full(Y|Z,D), h0(Y|Z,D)) - tau ]_+
```

`I_w(Z;D|Y)` 在 GLS 参考分布下控制 class-factorized reference mixture `P(z)`，避免 direct marginal `I_w(Z;D)` 擦掉标签信息；JS consistency 直接约束 full domain decoder 与 intercept-only decoder 的预测分布，从而比 CE residual training loss 更适合支撑 `P(y|Z)` stability 的主张。CE residual 仍保留为 held-out 诊断字段。

当前 CPU gate 显示：旧 `dualc`/CE-residual training 可以提高 concept positive-control 的 target bAcc，但同时恶化 `P(z)` 和 residual `P(y|Z)` probes；JS `dualpc` 在同一 gate 中小幅改善两类 probe 且不伤害 null-prior。因此论文主线应优先用 `dualpc`，把 `dualc` 放为诊断/消融。

### 3.8 Route A: GLS-VAE

Route A 用 structured latent 和 GLS decode 把 label-shift tension 结构性拿掉，然后用 `delta_d` 的 held-out ELBO gain 做 generative concept-shift test。

synthetic 上结论是“部分有效”：它能解除 tension，能让两个 leakage 同时更低，但不能“by construction” 把 `I(Z;D|Y)` 压到 0；仍需要显式 penalty。真实 EEG 上目前结果更像诊断工具而不是准确率方法：

- ADFTD: observed gain `0.221`，null `0.038±0.095`，`p=0.030` 但 `z=1.94`，按代码双条件没有 fire。
- MUMTAZ、SCZ resting、PD cohort 都 quiet。

## 4. 结果状态

### 4.1 Synthetic dual v2

`results/synthetic_dual_v2.txt` 支持三个要点：

- exact MI sweep 复现 tension：label shift 且 `alpha<1` 时，`I(Z;D|Y)=0` 仍强制 `I(Y;D|Z)>0`。
- learned tension 存在：`lam_enc 0 -> 4` 时，held-out `I(Y;D|Z)` 从 `0.162` 升到 `0.172`。
- naive `dual` 在 covariate-only 有小幅帮助，但在 concept/all-three 场景没有稳定 compounding benefit。

### 4.2 Within-dataset ladder

三组主结果的读法如下：

| 数据 | ERM | LPC-CMI | IIB | Dual | 关键读法 |
|---|---:|---:|---:|---:|---|
| ADFTD, 3 seeds, subject bAcc | 57.1 | 58.7 | **60.9** | 59.8 | decoder signal 大，但 subject-level 有退化风险 |
| MUMTAZ, 3 seeds, subject bAcc | **86.6** | 85.5 | **86.6** | 85.5 | 无 concept，CMI 主要降 leakage |
| TUAB, 2 seeds, subject bAcc | 55.6 | 57.5 | 56.9 | **58.8** | LPC/Dual 降 leakage，准确率有波动 |

leakage 方面更清楚：

- ADFTD: ERM `leakage_kl≈1.33`，LPC `≈0.17`，Dual `≈0.23`。
- MUMTAZ: ERM `≈1.53`，LPC `≈0.02`，Dual `≈0.025`。
- TUAB: ERM `≈1.48`，LPC `≈0.047`，Dual `≈0.057`。

decoder CMI 方面也符合 tension：

- ADFTD: ERM `≈0.20`，LPC `≈0.30`，Dual `≈0.275`。
- TUAB: ERM `≈0.10`，LPC/Dual `≈0.13`。
- MUMTAZ: ERM 接近 0，LPC/Dual 上升到约 `0.03`，更像压 encoder 后产生的 residual artifact。

### 4.3 Route B reweighted dual

ADFTD reweighted-dual 三个 seed 的 `dual` 结果大致是：

- subject bAcc `0.631 / 0.506 / 0.617`
- `leakage_kl≈0.23-0.24`
- `decoder_cmi≈0.238 / 0.278 / 0.301`

与 naive dual 很接近。MUMTAZ 上 `rwdual` 也基本等于 naive dual。当前证据不支持“Route B 解决了 dual 的实际优化问题”。

### 4.4 Cross-site / SCPS

新落盘的 `results/scz_resting_ladder.json` 比旧报告更重要，因为它包含四 cohort SCZ：

- ERM bAcc `50.6`，leakage `0.411`，decoder `0.0036`
- LPC bAcc `49.9`，leakage `0.128`，decoder `0.0017`
- Dual bAcc `49.9`，leakage `0.125`，decoder `0.0012`

cross-site PD/SCZ cohort seeds 的整体模式也一致：encoder leakage 可以降，cross-site decoder concept signal 接近 0，准确率基本 parity。这个结果对论文叙事很重要：它说明真实 disease cross-site 上更像 concept-shift null，而不是 dual penalty 的准确率机会。

### 4.5 Concept-shift detector

`notes/CONCEPT_SHIFT_SECTION.md` 的最新叙事是可信的：Route C 在 real-data positive control 下能 fire；真实 cross-site disease 数据上 Route C 和 Route A 都读 null；PD medication-state 是一个真正 positive case，subject-specific levodopa response 能被读到。

这个方向比 naive dual 更像论文亮点：不是“我们用 decoder CMI 提升所有任务准确率”，而是“我们能分辨 cross-site disease rule 是否真的 domain-dependent，并且能在 paired medication task 上读到 subject-specific response”。

## 5. 代码和文档风险

1. **文档里 estimator 表述不统一。** README 仍有“upper bound”表述，PROJECT_SUMMARY 又强调不是 generic upper bound，而是 Step-A 收敛时一致的 plug-in surrogate。建议统一为“consistent variational plug-in proxy；Step-A convergence 时等于 true CMI；未收敛可能 under-estimate”。

2. **Route B 文档和代码不一致。** `notes/route_B_reweighted_dual.md` 写 encoder reference 是 uniform，但当前 `trainer.py` 用 `reference="marginal"`，`regularizers.py` 注释也说 post-GLS 正确 reference 是 domain marginal `p(D)`。建议更新 Route B 文档。

3. **`--label_correct` help 可能误导。** `run_loso.py` 的 help 写“on CE + decoder CMI”，但在 naive `dual` 且未开 `--reweight_dual` 时，实际主要只影响 CE；`r_dec = post.dec_cmi(...)` 没有按 `wb` 重加权。建议修 help 或强制 `dual + label_correct` 自动转成 `reweight_dual`。

4. **decoder 辅助模型诊断仍可更细。** `stepA_dom_acc` 已补到 `dual/dualc/dualpc/dualpc_marginal`，但 decoder estimator 的可信度还依赖 `q_yz/h_ydz/h0` 拟合质量。建议继续记录 `qY_ce`、`hY_ce`、`h0_ce`、`dec_cmi_train/eval gap`。

5. **naive `dual` 的 raw `dec_cmi` 仍可为负。** evaluation 端会 `max(...,0)`，但 naive training 端 `post.dec_cmi` 不 clamp。如果 probe 噪声导致负值，loss 会奖励错误方向。`dualpc` 已避开这个问题，使用 gated JS consistency；naive `dual` 应保留为消融，或至少改成 gated/clamped 版本。

6. **`D=subject` 的 decoder CMI 需要显式标 invalid。** SCPS disease label 下 subject 往往单类，`I(Y;D_subject|Z)` 退化成 `H(Y|Z)`。后续已在 runner 中实现 `decoder_valid` / `decoder_valid_n` / `*_valid_mean` 字段；正式表格默认不把 invalid decoder split 当 concept shift。

7. **sampler-prior 一致性可以更硬。** `domainbal` 会改变 effective `p(D|Y)`，需要 `--prior effective`。现在靠 help 提醒，建议代码上 warning/error。

8. **旧报告需要归档或标日期。** `CMI_TECHNICAL_REPORT.md`、`DUAL_CMI_THEORY.md` 部分 job 状态和代码细节已被新结果更新。建议保留，但加 “superseded by ...” 指向最新结果报告。

## 6. 改进建议

### 6.1 论文/方法定位

把最终方法拆成两条清晰主线：

- **主算法**：`lpc_prior` / GraphCMI-style conditional leakage removal 是稳定基线；`dualpc` 是当前候选扩展，用 GLS-factorized `I_w(Z;D|Y)` 控制参考 `P(z)`，用 JS consistency 控制 `P(y|Z)`。
- **诊断扩展**：CE-residual Route C + null-calibrated Route A GLS-VAE，用于判断 concept shift 是否存在，并作为 `dualpc` 的 decoder-side 消融。

不要把 naive `dual` 作为“更强主算法”。目前证据更支持：naive dual 是 tension theorem 的 empirical manifestation，不是稳定提升器。

### 6.2 dual-CMI 技术路线

建议保留四个 dual 版本，但功能不同：

- `dual`：理论反例/消融，展示 label shift tension。
- `rwdual`：GLS reweight 消融；当前结果显示不够，应减少篇幅。
- `dualc`：CE-residual Route-C 诊断/消融；positive control 有 power，但 training loss 不适合作为 simultaneous `P(z)` / `P(y|Z)` 主张。
- `dualpc`：当前候选主算法；factorized GLS `P(z)` control + JS `P(y|Z)` consistency。

如果还想提升 dual 的实用性，优先做：

1. 用多 seed / 多数据集正式检验 `dualpc` 的 null safety 和 positive-control power。
2. 把 source-only guarded selector 的 penalty 权重固定成 paper protocol，避免 target-tuned `lambda/gamma/tau`。
3. encoder `D` 和 decoder `D` 支持不同粒度，默认 encoder=site/cohort，decoder=class-spanning cohort；subject 只用于 paired task。

### 6.3 实验策略

GPU 现在满，不建议提交新 GPU job。当前最有价值的 CPU/轻量工作是：

- 重新汇总已有 JSON，生成一个最新 master table，避免引用过期报告。
- 用 CPU 跑小规模 `LogCov` 或 `--max_subjects` smoke，只验证代码路径，不追求数值。
- 继续用 synthetic / tiny CPU run 验证 `dualpc` JSON 字段、selector 字段和 valid-only decoder summary，不追求性能数值。

等 GPU 空出来后，优先级建议：

1. `dualpc` vs `lpc_prior` vs `erm` 在 cross-site PD/SCZ 的多 seed 复跑，重点看 null 下不伤害；`dualc` 作为 CE-residual 消融。
2. paired PD medication-state 作为 positive case，补充 subject-level concept shift 的正式表。
3. GraphCMI 的 node/edge ablation，争取一个和 EEG 生理结构更贴的亮点图。

### 6.4 工程整理

- 新增 `scripts/summarize_dual_results.py` 或扩展 `analysis/summarize_results.py`，统一输出 method、acc、conditional KL、marginal `P(z)` KL、decoder JS/residual、valid-only decoder summary 和 selector metadata。
- 在结果 JSON 中写入 `domain_granularity`、`decoder_domain_granularity`、`domain_class_span_stats`。
- 对每个 notes 报告加日期和 supersede 关系，避免旧状态误导。
- 给 `dualpc`、`dualc` 和 `reweight_dual` 增加最小单元测试：权重为全 1 时等价、label shift toy 中 reweight 后 `I~(Y;D)` 接近 0、默认 `dualpc` tau 为 0、invalid subject domain 被标记。

## 7. 最终建议

当前项目最强、最可信的版本是：

> Tri-CMI 用 label-prior-corrected conditional MI 去除 EEG 表示中的条件域泄漏；dual-CMI 给出 encoder/decoder 两个 CMI 的精确分解和 label-shift tension；当前可投稿的 dual 扩展应是 GLS-factorized `dualpc`，同时用 source-only probes 约束 `P(z)` 和 `P(y|Z)`，而不是 naive dual loss 的平均准确率提升。

落地到论文上，建议把 naive dual 写成“为什么不能直接双压”的理论和实证证据，把 CE-residual Route C 写成 concept-shift 诊断/消融，把 `lpc_prior` / GraphCMI 和 JS-consistency `dualpc` 写成方法主线。正式投稿前仍需要多 seed 真实数据结果证明 `dualpc` 的 null safety 和 positive-control power。

## 8. 本次我做过的轻量验证

未提交 GPU/SLURM 作业。只做了 CPU 级验证和静态读取：

- `notes/theory/verify_tension.py`：通过，identity 最大误差 `2.442e-15`。
- `notes/theory/verify_resolution.py`：通过，GLS reweight 后 label shift 和两个 CMI tension 均消失，`ALL PILLAR-2 CLAIMS VERIFIED: True`。
- `synthetic/route_c_positive_control.py`：完成；concept residual 有明显 power，但 subject-degen residual 仍较大，说明 subject-level disease decoder CMI 不能直接读作 concept shift。

## 9. 追加：DualPC 优化实现后的结论

后续已实现 `dualpc` 入口，并新增 `synthetic/dualpc_validation.py` 做 CPU-only production-trainer 验证。关键结论是：

- 直接优化 marginal `I_w(Z;D)` 再加自动 GLS task CE 不稳定，已保留为 `dualpc_marginal` 负消融。
- 当前 `dualpc` 改为 factorized + JS-consistency 版本：普通 task CE + `I_w(Z;D|Y)` + gated `JS(h_full(Y|Z,D), h0(Y|Z,D))`。在 GLS 参考分布下，`I_w(Z;D|Y)->0` 可推出参考 mixture `P(z)` 对齐；decoder 侧直接让 full domain decoder 与 intercept-only decoder 的预测分布接近，从而优化 `P(y|Z)`，比旧的 CE residual training loss 更稳定。CE residual `CE(h0)-CE(h_full)` 仍作为评估诊断输出。
- 已修正 decoder gate 默认值：`dualpc` / `dualpc_marginal` 默认 `tau=0.0`，确保 JS `P(y|Z)` 项默认处于激活状态；`dualc` 仍默认 `tau=0.02`，保留 Route-C CE residual 的 null margin。显式传 `--dec_margin` 时仍覆盖默认值。CPU micro `results/dualpc_tau_default_micro.json` 验证 `dualc_tau=0.02`、`dualpc_tau=0.0`。
- CPU all-three quick 结果：`dualpc` 与 `dualc` 对齐，target bAcc 74.9，优于 ERM 72.4，接近 `lpc_prior` 75.3；`dualpc_marginal` 为 73.1。
- source-only bAcc selector 和旧 guarded-probe selector 都不能可靠发现有用的 direct marginal `dualpc` 配置。因此投稿主线应是 factorized `dualpc`，不是 direct marginal `P(z)` 惩罚；`dualc` 保留为 CE-residual 诊断/消融。
- 已修复 Step-A 诊断：`dual/dualc/dualpc/dualpc_marginal` 现在都会记录 `q(D|Z,Y)` 的 `stepA_dom_acc`，不再在 LOSO summary 里误显示为 0。
- 已跑真实数据 CPU/LogCov smoke：`BNCI2014_001`，3 subjects，2 epochs，`--device cpu`。结果保存在 `results/dualpc_losologcov_smoke.json`，证明 `dualpc` / `dualpc_marginal` 的真实数据入口、JSON 字段、probability sidecar 和 CPU-only 运行路径可用。该 smoke 太短，不作为性能证据。
- 已同步 MI Protocol-C 入口 `cmi/run_cross_dataset.py`：现在正确解析 `dualpc:lambda:gamma`、支持 `--device cpu`、并记录 `inloop_reg/inloop_dec/stepA_dom_acc`。此前这个入口会把 `dualpc:0.1:0.05` 的 `gamma` 静默丢掉。
- 已加入 decoder residual 的 source-only permutation-null 诊断：`run_loso.py` / `run_scps_crossdataset.py` 支持 `--decoder_null_perms`，输出 `decoder_cmi_res_null_q`、`decoder_cmi_res_excess`、`decoder_js_res_null_q` 和 `decoder_js_res_excess`（含 GLS reweighted 版本）。同时新增 decoder validity 元数据：只有 source probe 至少有两个 decoder domains、且每个 domain 至少跨两个 class 时，`decoder_valid=true`；summary 写入 `decoder_valid_frac`、`decoder_valid_n`、`decoder_min_domain_classes`、`decoder_single_class_frac`，并给 decoder 主指标写入 `*_valid_mean`。正式表格应优先读取 `decoder_js_res_valid_mean` / `decoder_cmi_res_valid_mean`；没有有效 fold 时写 JSON `null`，避免 invalid 数值污染聚合。这解决了报告里“只看 fixed residual、缺少 null calibration / validity flag”的一部分问题；训练时的 `dec_margin` 仍是显式超参。
- 已加入 direct `P(z)` 的 held-out marginal leakage probe：`cmi/eval/metrics.py::marginal_leakage_probe` 固定 backbone 后训练 `q(D|Z)`，runner 输出 `marginal_leakage_kl`、`marginal_leakage_kl_rw` 和 advantage 字段。3-subject CPU/LogCov smoke 保存在 `results/dualpc_pz_probe_3subj_smoke.json`：1 epoch 下 target bAcc `39.6±11.1`，conditional KL `0.642`，GLS-weighted `P(z)` KL `0.630`，`stepA_dom_acc=70.4%`。这只证明真实数据 instrumentation 非退化，不作为性能证据。
- 已把 conditional leakage probe 扩展为 GLS-weighted 版本：`leakage_probe(..., reweight=True)` 现在输出与 `dualpc` 训练目标一致的 `leakage_kl_rw`，runner summary 也记录该字段；raw `leakage_kl` 仍保留为诊断。这修正了之前 source selector 的一个语义偏差：训练侧优化的是 `I_w(Z;D|Y)`，而 selector tie-breaker 不应使用未加权 conditional KL。
- 已把 `cmi/run_lambda_select.py` 扩展为 DualPC source-only guarded selector：`--select_rule guarded_probe` 先按 source-val bAcc 过滤候选，再用 source-only probes 的 `GLS conditional KL + GLS P(z) KL + JS P(y|Z)` 作为 tie-breaker；raw conditional KL 和 CE residual `P(y|Z)` 仍作为诊断字段记录，`--dec_margins` 可同时选择 decoder gate `tau`。旧 CPU smoke 保存在 `results/dualpc_lamsel_js_guarded_smoke.json`，证明真实数据 runner 可以在 source-only setting 下记录 `select_py_js_rw` 和 `select_py_res_rw`；新 CPU smoke `results/dualpc_condrw_lamsel_valid_smoke.json` 进一步证明生产 selector 非退化写出 `select_cond_kl_rw`，且第一折 penalty 正好是 `select_cond_kl_rw + select_pz_kl_rw + select_py_js_rw`（ERM `0.00945`，DualPC `0.00518`）。这些 smoke 只有 1 epoch，不作为性能证据。
- 已继续扩展 `cmi/run_lambda_select.py`：新增 `--final_probe_epochs`，在 selected 配置和固定 ERM 都用全部 source retrain 之后，再记录一套 source-only GLS/JS probes。字段写入每个 `selection_records`：`final_selected_cond_kl_rw`、`final_selected_pz_kl_rw`、`final_selected_py_js_rw`、`final_erm_*`、`target_bacc`、`target_erm_bacc` 等。CPU smoke `results/dualpc_finalprobe_lamsel_smoke.json` 通过，4 个 folds 的 selected-candidate probe 和 final retrain probe 都有效；`analysis/dualpc_readiness.py results/dualpc_finalprobe_lamsel_smoke.json` 报告 `PASS=8`、`WARN=0`、`FAIL=0`。这些仍是 1 epoch path check，不作为性能证据。
- 已把 synthetic source-selection harness 与生产 selector 对齐：`synthetic/dualpc_validation.py --source_select --select_rule guarded_probe` 现在同样用 `GLS conditional KL + GLS P(z) KL + JS P(y|Z)`，并记录 `select_cond_kl`、`select_cond_kl_rw`、`select_pz_kl_rw`、`select_py_js_rw`、`select_py_res_rw`、`selection_probe_valid`。CPU smoke `results/dualpc_condrw_selector_smoke.json` 验证了 `cond_kl_rw` 和 `gate_cond_rw_improved` 会写入 summary；此前的 `results/dualpc_synthetic_select_aligned_smoke.json` 是旧 raw-conditional selector 的路径检查。这个结果只验证选择逻辑，不作为性能证据。
- 已新增并继续修正 `analysis/dualpc_readiness.py`，统一读取 synthetic / runner / selector JSON，输出 `cond_kl_rw`、GLS `P(z)` KL、JS `P(y|Z)`、decoder-validity 和 source selector penalty 的 readiness 表。脚本现在会内部展开 glob，会把 `source_select=true` 的 synthetic smoke 视为 selector path check 而不是 method gate，并在 final retrain probes 存在时输出 `selector_final` 行；当 runner JSON 同时含 ERM/LPC baseline 时，还会输出 `runner_compare` 行，自动记录 DualPC-vs-baseline 的 `delta_acc`、`delta_cond_kl_rw`、`delta_pz_kl_rw`、`delta_py_js_rw`，用于正式结果的 null-safety/probe 审计。当前 CPU bundle `results/dualpc_condrw_gate_js_concept_null_quick.json` + `results/dualpc_condrw_loso3_smoke.json` + `results/dualpc_finalprobe_lamsel_smoke.json` 汇总到 `results/dualpc_readiness_current.json`，脚本报告 `PASS=15`、`WARN=2`（ERM baseline 行）、`FAIL=0`。这证明当前代码路径和证据字段已经对齐，但仍不是正式多 seed real-data 性能证据。
- 已新增 `analysis/dualpc_paper_summary.py`，作为正式结果的跨文件/跨 seed 聚合层。它会读取 runner JSON，输出 method summary 与 DualPC-vs-ERM/LPC 的 `delta_acc`、`delta_cond_kl_rw`、`delta_pz_kl_rw`、`delta_py_js_rw` 聚合表；同时读取 selector JSON，聚合 final retrain probes、selected-config histogram、target bAcc 以及相对固定 ERM 的 delta。当前 smoke bundle 写到 `results/dualpc_paper_summary_current.json`，selector-final aggregate 报告 `PASS=1`、`WARN=0`、`FAIL=0`；runner comparison 为空是预期的，因为当前 `results/dualpc_condrw_loso3_smoke.json` 只含 `dualpc`，正式 protocol runner 会含 ERM/LPC baseline。
- 已新增并继续修正 `scripts/dualpc_protocol.py`，把正式 AAAI 实验协议固化成可生成的命令清单，不提交作业。`--profile paper --device cuda --seeds 0 1 2` 会生成 synthetic gate、LOSO、source-only guarded selector、SCPS PD/SCZ cohort-domain runs、最终 readiness summary、最终 paper aggregate summary 和 headline decision gate；`--profile smoke --device cpu` 只生成本地 CPU 路径检查命令。dry-run 已通过，生成的 paper 命令包含 `dualpc`、`dualc`、`dualpc_marginal`、`lpc_prior`、`erm`，使用 `--decoder_null_perms 20`、当前 `GLS conditional + GLS P(z) + JS P(y|Z)` selector protocol，并对 selector 命令开启 `--final_probe_epochs`。
- 已新增 `analysis/dualpc_decision_gate.py`，作为 readiness + paper-summary 之后的机械决策层。它只读最终 JSON，输出 `HEADLINE_READY`、`NEEDS_REVIEW`、`NOT_READY` 或 `PENDING`，规则保守：正式 comparison 缺失就是 `PENDING`，probe/accuracy gate 有 hard failure 就 `NOT_READY`，warning 留给人工 review。paper protocol 的 gate 现在要求 synthetic `null_prior`/`concept`/`all_three` 都有 main DualPC PASS，4 个正式 comparison tasks 都同时覆盖 ERM 和 `lpc_prior` baseline，并要求 2 个 LOSO dataset 都有 final-probe selector summary。当前 smoke bundle 因没有正式 runner baseline comparisons 且没有 `all_three` synthetic group，decision gate 正确返回 `PENDING`，不把 smoke 误判成 headline-ready。
- 已新增 `scripts/dualpc_slurm_plan.py` 和 `scripts/dualpc_paper_status.py`，用于把 paper protocol 转成可提交但不自动提交的 SLURM package，并在提交前/跑完后做只读状态检查。当前已生成 `scripts/dualpc_paper_tasks.tsv`、`scripts/dualpc_paper_array.slurm`、`scripts/dualpc_paper_post.slurm`：20 个 array tasks（`0-19%2`），包括 1 个 regression guard、1 个 synthetic gate，以及 seeds 0/1/2 上的 LOSO、source-only selector、SCPS PD/SCZ 任务；post 脚本跑 readiness、paper-summary 和 decision gate，其中 decision 命令使用 `--min-comparison-tasks 4 --min-selector-tasks 2 --required-baselines erm lpc_prior --required-synthetic-groups null_prior concept all_three`。`scripts/dualpc_paper_status.py` 当前报告 plan checks 全 PASS，1 个 READY no-output regression task、19 个 PENDING task JSON、3 个 PENDING post JSON，`WARN=0`、`FAIL=0`；它还会静态检查任务内容能满足 gate：synthetic 任务包含 `all_three/concept/null_prior`，runner 覆盖 `loso:BNCI2014_001`、`loso:MUMTAZ`、`scps:PD`、`scps:SCZ`，每个 runner task 含 `erm` 和 `lpc_prior`，selector 覆盖两个 LOSO dataset。这些文件尚未提交到 Slurm。
- 已新增 `scripts/dualpc_regression_checks.py`，作为无 EEG 数据依赖、CPU-only 的快速防回归脚本，并接到 `scripts/dualpc_protocol.py` 生成命令清单的第一步。它覆盖 `dualpc`/`dualpc_marginal` 的 `tau=0` 默认、`dualc` 的 `tau=0.02` 默认、GLS label-shift 权重会缩小 domain label-prior disparity、辅助 loss 在 all-ones weight 下等价、decoder validity / valid-only summary、readiness 对 source-selector/final-selector/runner_compare 的分类、paper-summary 聚合解析、decision gate 行为、protocol 命令是否包含 `--final_probe_epochs`、readiness/paper-summary/decision glob、SLURM package 生成、preflight status 解析，以及一个 tiny `train_model(..., method="dualpc")` 训练路径检查（验证 raw sampler forcing、`tau=0`、JS-side `inloop_dec_loss` 写出）。当前运行报告 `ALL PASS (11 checks)`。
- 已补一个当前 GLS-conditional 口径的 concept/null synthetic gate：`results/dualpc_condrw_gate_js_concept_null_quick.json`。在 concept DGP 中，`dualpc` 相对 ERM target bAcc `+0.046`、`cond_kl_rw -0.0046`、GLS `P(z)` KL `-0.0024`、JS `P(y|Z)` `-0.00045`，同时三类 probe 都往正确方向走；null-prior 中 target 持平，`P(z)`/JS 小幅改善，conditional KL 只有 `+0.00071` 的有限样本波动，按 null-safety tolerance 通过。这是当前代码语义下最干净的 CPU 正向证据。
- 已做一次 concept/null synthetic gate 来检查“同时优化两边”的证据。旧 `dualc`/CE-residual training 在 concept positive control 上 target bAcc 从 ERM `52.7` 升到 `63.1`，但 `P(z)` KL 从 `0.1298` 升到 `0.1559`，residual `P(y|Z)` 从 `0.0829` 升到 `0.1723`，不能支撑 simultaneous optimization 叙事。改成 JS-consistency 后，`dualpc` 在 concept 上 target bAcc `53.5`、`P(z)` KL `0.1221`、residual `P(y|Z)` `0.0801`；null-prior 上 bAcc 与 ERM 持平、两类 probe 均小幅下降。结果保存在 `results/dualpc_gate_js_concept_null_quick.json`。这支持把 JS-consistency 版 `dualpc` 作为投稿主算法，把 CE-residual 版 `dualc` 放为诊断/消融。
- JS 更新后已跑 all-three CPU smoke：`results/dualpc_js_allthree_quick.json`。DualPC-JS target bAcc `54.6` vs ERM `53.9`，`P(z)` KL `0.098` vs `0.101`，residual `P(y|Z)` 增加 `0.0045`，仍在当前 `gate_py_not_raised` 容忍线内。该 quick 训练很短，不能替代正式多 seed 结果。
- JS 更新后已补真实数据 probe 字段和 validity smoke：`results/dualpc_validity_loso_smoke.json`。2-subject/1-epoch CPU 跑通 `decoder_js_res=2.56e-4`、`decoder_js_res_rw=2.51e-4`，并记录 `inloop_dec_loss=1.0e-4`；由于每个 source fold 只有 1 个 source domain，`decoder_valid_frac=0.0`、`decoder_valid_n=0`、`decoder_js_res_valid_mean=null`，这正是预期的无效标记。`results/dualpc_validity_lamsel_smoke.json` 也验证了 source-only selector 会在 decoder split 无效时把 `selection_probe_valid=false`，不把 JS 项当作有效 tie-break。两个 smoke 都只作为字段和路径验证，不作为性能证据。
- 已补 3-subject 非退化真实数据 smoke：`results/dualpc_tau_valid_loso3_smoke.json`。默认 `dualpc` 训练记录 `train_dec_margin=0.0`，3 个 folds 全部 `decoder_valid=true`，summary 有 `decoder_valid_n=3`、`decoder_js_res_valid_mean=0.00199`、`decoder_js_res_rw_valid_mean=0.00188`、`inloop_dec_loss=4.59e-4`。这证明真实数据入口默认不会把 JS `P(y|Z)` 项 gate 掉，并且 valid-only decoder 汇总字段能非退化写出。该 smoke 仍只有 1 epoch，不作为性能证据。

详见 `notes/AAAI_DUALPC_DESIGN.md`、`analysis/dualpc_readiness.py`、`analysis/dualpc_paper_summary.py`、`analysis/dualpc_decision_gate.py`、`scripts/dualpc_protocol.py`、`scripts/dualpc_slurm_plan.py`、`scripts/dualpc_paper_status.py`、`scripts/dualpc_regression_checks.py`、`scripts/dualpc_paper_tasks.tsv`、`scripts/dualpc_paper_array.slurm`、`scripts/dualpc_paper_post.slurm`、`results/dualpc_readiness_current.json`、`results/dualpc_paper_summary_current.json`、`results/dualpc_condrw_gate_js_concept_null_quick.json`、`results/dualpc_condrw_loso3_smoke.json`、`results/dualpc_finalprobe_lamsel_smoke.json`、`results/dualpc_gate_js_concept_null_quick.json`、`results/dualpc_js_allthree_quick.json`、`results/dualpc_tau_default_micro.json`、`results/dualpc_tau_valid_loso3_smoke.json`、`results/dualpc_validity_loso_smoke.json`、`results/dualpc_validity_lamsel_smoke.json`、`results/dualpc_losologcov_smoke.json`、`results/dualpc_decoder_null_smoke.json`、`results/dualpc_pz_probe_3subj_smoke.json`、`results/dualpc_lamsel_js_guarded_smoke.json`、`results/dualpc_synthetic_select_aligned_smoke.json`、`results/dualpc_condrw_selector_smoke.json` 和 `results/dualpc_condrw_lamsel_valid_smoke.json`。

# 目标级量化失效归因与可靠恢复工程

> 工作名：RAMP-Q  
> 当前阶段：RUOD-R Gate 0 已收口；Gate 0.5 DFL 已缩至 QDQ＋Conv 六节点，NONE 成功而 OBEY 仍同类断言失败；Softmax 与末端 Reshape 均不是该局部失败的必要条件，原完整护栏仍失败（2026-10-21）  
> 当前结论：**总体方向 GO；目标级归因与恢复主线不变；“新选模算法优于传统方法”尚未成立。**

2026-10-21 最新 Conv 输出边界：保留同一概率 fixture，仅删去末端 Constant/Reshape，形成 6 节点、7 initializer 的 Q/Clip/DQ＋权重 QDQ＋Conv。NONE 约 50.381 秒构建成功，两次四维 `[1,1,4,8400]` 输出相同；在明确形状、dtype 和元素数对应后，与旧八节点三维 DFL 输出的原始数组字节也完全相同。OBEY 约 1.524 秒仍同类 `filterQDQFormats::4421`／后续 `751` 失败，无引擎、无推理。成功 Conv 实际为 INT8 输入/权重、FP32 输出，命名 tactic 不变；CPU 最大差仍约 1.91e-6。单次新 CPU 输出与旧 CPU 原始数组字节一致，NPY 形状头不同不要求文件哈希相同；没有生成概率或重跑上游。89 份原始文件已核验，局部失败不需要显式 Softmax 或末端 Reshape，但完整图唯一根因及修复仍未证明，Gate 0.5 未通过。详见[36_Gate05_DFL卷积输出边界定位.md](36_Gate05_DFL卷积输出边界定位.md)。

此前 Softmax 后切片：修正 DFL 八节点图中，NONE 约 52.228 秒成功，其两次输出及与旧九节点修正 NONE 的保存输出逐字节一致；但外部输入不同，不能据此宣称 native Softmax 概率与 CPU 概率等价。OBEY 约 1.530 秒仍同类断言失败，无引擎或推理。准备仅原 Softmax 一次 CPU 生成概率、八节点图一次 CPU 验证旧 DFL 字节等价，没有重跑 72 Conv 上游。89 份原始文件保留于[35_Gate05_DFL_Softmax邻接定位.md](35_Gate05_DFL_Softmax邻接定位.md)。

此前 DFL 原始拓扑对照：复用同一认证内部 fixture 与旧 CPU 输出，没有重跑 CPU 或 72 Conv 上游。原始 FLOAT Clip→Q/DQ 小图的 NONE 约 52.321 秒构建成功、两次输出字节一致，但相对 CPU 有 33,566/33,600 个 DFL distance 元素不同，最大绝对差约 11.95785；OBEY 约 1.509 秒触发同类断言，无引擎、无推理。两拓扑 NONE 的实际 Conv 均为 INT8 输入/权重、FP32 输出，且命名 tactic 相同；修正 NONE 的 CPU 最大差仅约 1.91e-6。四格对照说明 OBEY 局部失败不需要 code_clip 派生拓扑，而构建成功也不保证数值正确。84 份原始文件保留于[34_Gate05_DFL原始拓扑与约束对照.md](34_Gate05_DFL原始拓扑与约束对照.md)。

此前 DFL 修正拓扑上下文：精确 9 节点、7 initializer 的 Softmax→Q/Clip/DQ→Conv→Reshape 小图，在同图、同解析约束下仅切换 NONE/OBEY。NONE 约 52.212 秒构建成功并完成两次逐字节一致的执行；实际 DFL Conv 输入、权重为 INT8，输出为 FP32。OBEY 约 1.538 秒复现与完整修正 q_all 同类的 `filterQDQFormats::4421 / candidateRequirements` 断言，无引擎、无推理或原始输出。内部 logits fixture 来自精确 72 Conv 上游 CPU 一次执行，原/改 tail 各 CPU 一次、共享 fixture 输出字节一致；native 仅执行小 tail，输出是 DFL distance，不是检测框。88 份原始文件保留于[33_Gate05_DFL内部上下文定位.md](33_Gate05_DFL内部上下文定位.md)。

此前双 Conv 上下文：从实际失败的修正 q_all 源图精确提取 19 节点、20 initializer，截止第三 Conv 输入 QDQ，不包含第三 Conv。NONE/OBEY 均构建成功，构建耗时分别为 69.188/69.375 秒；两组原始输出及各自两次重复执行逐字节一致。实际两个 Conv 的输入、权重及输出均为 INT8，首 Conv 采用专用 SWISH 整数 tactic，第二 Conv 为 GENERIC 且实际融合第三处 Clip。CPU/native 的 819,200 个输出元素中仅 18 处相差 ±1 个量化码；反量化浮点最大绝对差约 1.56195259094，不能把它严格写成恰等于 scale。该上下文未复现完整 q_all 格式错误，89 份原始文件已回收核验。历史记录见[32_Gate05_双卷积上下文定位.md](32_Gate05_双卷积上下文定位.md)。

此前的单 Conv 前缀对照：固定图片、两种 Clip 拓扑各比较 NONE/OBEY，四个小引擎均构建并执行成功。每个 case 的两次原始输出逐字节一致，同拓扑 NONE/OBEY 的原始输出也逐字节一致。四者实际 Conv 输入、权重及输出均为 INT8，使用相同命名整数 tactic 和 GENERIC SiLU 融合。该首 Conv 上下文与最新双 Conv 的融合方式不同，不能互相沿用。历史记录见[31_Gate05_前缀Clip与精度约束对照.md](31_Gate05_前缀Clip与精度约束对照.md)。

2026-10-19 全 INT8 推广：新 q_all 的 88 处 code_clip 改写通过完整主辅 CPU 字节等价及独立前置审查，板端 88 处导入属性也已校正；但单次 native 构建在约 13 秒触发 TensorRT `filterQDQFormats` 断言，没有引擎和单图执行。52 份原始文件及失败终态已核验，板子保持同次启动、进程全部退出；没有更改条件重跑。此前 q_only_G1 成功保留，这份全图失败记录不因后续小切片成功而改判，Gate 0.5 仍未通过。详见[30_Gate05_全INT8修复候选验证.md](30_Gate05_全INT8修复候选验证.md)。

2026-10-19 完整参考诊断：原 q_all 的 PyTorch/ORT 护栏失败已精确复现。首卷积微小浮点差异跨过量化舍入边界；在保持原参数和量化规则的内存对照中，统一 Conv 与 SiLU 后端后，88 个卷积输出与原 ORT 逐字节一致，最终 box 差为 0、score 最大差约 1.31e-7。这定位了当前固定图的参考算术分歧，没有替换原参考或改判原失败。详见[29_Gate05_完整参考数值分歧定位.md](29_Gate05_完整参考数值分歧定位.md)。

2026-10-19 完整 q_only_G1 候选结果：保留截断语义的 code_clip 改写及 TensorRT 导入属性校正，已用于另存的完整 q_only_G1 候选。原图/改写图的 CPU 主、辅助输出在固定图片上逐位一致。Orin 完整引擎约 21 分钟构建成功，实际 6 个 INT8 输入及权重卷积具备命名整数 tactic，其余 82 个卷积满足 FP16 格式；首 Conv 使用专用 SWISH。原阈值下从零预测恢复为海参 0.657、扇贝 0.375；标注只有一个海参，不能称为两个正确目标。详见[28_Gate05_融合前缀与完整候选验证.md](28_Gate05_融合前缀与完整候选验证.md)。

此前全部 88 组冻结尺度的合成边界样本验证了 code_clip 改写的 CPU 等价性；正负 native 边界仍有 10/1811 个样本相差一个量化步长。原首层异常、13 个孤立小图及 295 份原始文件保留于[27_Gate05_首层量化异常定位.md](27_Gate05_首层量化异常定位.md)。上述完整 q_only_G1 候选新增 70 份原始文件，经独立审查和源图重审。原 PyTorch/ORT 完整数值护栏仍失败，浮点/融合对照尚不充分，没有数据集精度、目标级因果恢复或性能通过结论，Gate 0.5 仍未通过。

原始六个冻结配置的真实构建及固定单图记录均保留：FP16 对照有两个预测，五个量化配置均为零检测。完整模型修复单图成功目前仅限另存的 q_only_G1 候选；其余配置没有据此改判，q_all 与三个恢复配置的七个深度卷积严格证据缺口仍保留。原阶段记录见[26_Gate05_当前可执行工作收口.md](26_Gate05_当前可执行工作收口.md)，此前记录保留于[24](24_Gate05_Orin_INT8构建与迁移检查.md)、[25](25_Gate05_Orin_INT8完整图融合绕行检查.md)。

## 1. 这个工程研究什么

说人话就是：

> 找出哪些具体目标从FP32变成INT8后消失；再逐模块做隔离实验，判断哪些模块的量化与目标消失有关；最后从全INT8出发恢复相关模块，检查刚才消失的同一个目标能否重新出现，并统计是否误伤其他稳定目标。

核心链条：

```text
FP32能检出目标
    ↓
全W8A8后目标消失
    ↓
q_only:Gj：只量化Gj，观察该目标是否也消失
    ↓
r_only:Gj：全W8A8中只恢复Gj，观察同一目标是否回来
    ↓
统计同目标双向一致、附带漏检和高精度成本
```

混合精度恢复在本研究中有两个作用：

1. 它是验证模块归因是否可靠的干预工具；
2. 它也是未来TensorRT/Orin部署时的落地方案。

## 2. 两种干预配置是什么意思

- `q_only:Gj`：只有模块组`Gj`做W8A8 Q/DQ，其余模块保持FP32。它回答“量化这个模块是否足以让目标消失”。
- `r_only:Gj`：全模型先做W8A8，只把`Gj`恢复FP32。它回答“恢复这个模块是否能让目标回来”。
- `q_only:G1+G5`：只量化G1和G5。
- `r_only:G1+G5`：全W8A8中恢复G1和G5。

不能因为“破坏了13个、又救回13个”就说归因成立。必须检查目标ID交集，即同一个目标同时满足：

```text
q_only中消失 ∩ 全W8A8中消失 ∩ r_only中恢复
```

## 3. 证据边界

当前所有 INT8 归因与恢复结果仍来自 PyTorch Q/DQ 模拟 W8A8。2026-10-18 已完成真实 Orin FP16 引擎的构建、单图执行和层精度核查，详情见[23_Gate05_Orin_FP16首轮检查.md](23_Gate05_Orin_FP16首轮检查.md)。它尚不构成真实 INT8 恢复或数据集精度结论。现有模拟量化的范围为：

- 权重：对称逐输出通道INT8 Q/DQ；
- 激活：使用冻结校准范围的对称INT8 Q/DQ；
- 它适合做低成本机制筛查；
- 它不是TensorRT真实INT8内核；
- 它不能证明Jetson延迟、吞吐、显存、功耗或能量/帧；
- 它不能证明结果已经推广到VisDrone或其他检测器。

因此，文档中的“恢复”表示模拟量化条件下的目标级检测恢复，不等于真实部署结果。

## 4. Gate 0最终结果

### 4.1 数据

| 数据 | 图片 | 真实目标 | FP32命中 | 全W8A8新增漏检 |
|---|---:|---:|---:|---:|
| 全部screen | 500 | 2,785 | 2,293 | 165 |
| Selection | 350 | 1,920 | 1,571 | 119 |
| Holdout | 150 | 865 | 722 | 46 |

- Selection/Holdout按整张图片划分，图片重叠为0；
- 随机种子为`20260910`；
- Holdout只运行锁定的G1和G1+G5，没有在看到结果后更换候选。

### 4.2 核心候选

| 配置 | 数据 | 救回 | 附带漏检 | 净救回 | 双向同目标 | 稳定保持率 | FP32 MAC代理 |
|---|---|---:|---:|---:|---:|---:|---:|
| G1 | Selection | 94/119 | 13 | +81 | 69/119 | 99.10% | 10.32% |
| G1 | Holdout | 40/46 | 4 | +36 | 25/46 | 99.41% | 10.32% |
| G1+G5 | Selection | 100/119 | 8 | +92 | 83/119 | 99.45% | 22.35% |
| G1+G5 | Holdout | 43/46 | 5 | +38 | 29/46 | 99.26% | 22.35% |

这支持：G1的模块级破坏—恢复信号在独立Holdout上复现。它不等于G1是唯一根因，也不等于G1+G5是全局最优配置。

G1+G5相对G1在Holdout只多救回3个、净救回只增加2个，却多使用12.03个百分点FP32 MAC。因此G1是Gate 0.5 Orin真实性检查中必须保留的低成本方案。

### 4.3 同预算随机对照

| 方案 | MAC代理 | 净救回 | 净救回率 | 双向同目标 |
|---|---:|---:|---:|---:|
| G1+G5 | 22.35% | +92 | 77.31% | 83 |
| G4+G8 | 22.54% | -2 | -1.68% | 1 |
| G3+G8 | 21.08% | +7 | 5.88% | 2 |
| G2+G8 | 21.91% | +1 | 0.84% | 1 |

三个随机组合平均净救回率为1.68%。G1+G5通过预注册的1.20倍主判据和三项护栏。这个结论只表示它在当前Selection上明显优于这三个固定随机组合，不能外推为普遍优于随机选择。

### 4.4 传统基线

有标签总体敏感度排序按`q_only:Gj`造成的总体漏检数选模块，结果同样是`G1+G5`，记为`COLLISION`。因此当前没有证据证明目标级方法在“选出哪两个模块”上优于该传统规则。

无标签NMS前输出重建误差排序选出`G1+G2`：

| 方案 | MAC代理 | 救回 | 附带漏检 | 净救回 | 双向同目标 |
|---|---:|---:|---:|---:|---:|
| G1+G5 | 22.35% | 100 | 8 | +92 | 83 |
| G1+G2 | 22.97% | 93 | 6 | +87 | 73 |

G1+G5描述性多5个净救回和10个双向同目标，但两者逐目标净效用为19胜、14负、1,538相同，探索性双侧符号检验`p=0.487`。而且G1+G5是在同一Selection上选出的，所以不能声称对无标签基线取得独立显著优势。

## 5. Gate 0决策

| 问题 | 决策 |
|---|---|
| INT8是否会选择性丢失具体目标 | PASS |
| 能否定位少数相关模块 | PASS |
| 恢复相关模块能否救回同一目标 | PASS |
| 核心信号是否在独立Holdout复现 | PASS |
| 是否优于三个固定同预算随机组合 | PASS |
| 是否优于有标签总体敏感度选模 | 未成立：选择结果碰撞 |
| 是否显著优于无标签输出误差选模 | 未成立：仅有小幅描述性优势 |
| 是否已在真实TensorRT/Orin成立 | FP16 单图执行通过；INT8 失效与恢复尚未验证 |
| 是否已在VisDrone成立 | 未测试 |

最终说法统一为：

> **总体方向GO。论文继续研究目标级量化失效归因、双向恢复验证和小目标可靠性；混合精度方案是验证工具与落地结果。“新的选模算法优于传统方法”暂不作为已验证结论。**

## 6. 下一阶段

### Gate 0.5：现在做Orin真实性检查

截至 2026-10-21：六个原冻结引擎均已完成真实构建、源卷积请求格式核查与固定单图执行；另存的 q_only_G1 code_clip 候选也通过 6 INT8 / 82 FP16 的既有审计并恢复固定单图预测。另存全 q_all 修复候选仍为格式筛选失败、无引擎。最新 DFL 六节点 QDQ＋Conv 输出边界仍为 NONE 成功、OBEY 同类断言失败；NONE 的四维数组与旧八节点三维输出经形状对应后字节相同，见[36](36_Gate05_DFL卷积输出边界定位.md)。此前 Softmax 切口、原始/修正四格分别保留于[35](35_Gate05_DFL_Softmax邻接定位.md)、[34](34_Gate05_DFL原始拓扑与约束对照.md)。编译兼容与数值正确性须分别验证，取消 OBEY 不是完整图已证实的修复。原完整 PyTorch/ORT 数值护栏仍失败，原 q_all 与三个恢复配置的七个深度卷积严格整数证据缺口保留；非量化浮点及融合控制尚未完成，未运行固定 150 张图片的评估。完整 q_only_G1 候选记录见[28](28_Gate05_融合前缀与完整候选验证.md)。

下一步建议另冻结一个有限局部对照：只旁路权重 Q/DQ，让 Conv 使用同一原始 FP32 权重 initializer，保留概率输入、激活 Q/Clip/DQ、原权重值与尺度来源不变，再各做一次 NONE/OBEY。它会改变权重量化语义，必须建立自己的新 CPU 参考，不能要求与旧全量化输出字节一致，更不能称为等价修复。实际权重格式仍需 inspector 核验，不能因源 initializer 为 FP32 就推定实际计算精度；禁止重置解析约束或借用权重尺度添加输出量化。互补的激活路径旁路仅为后续备选，不自动扩成四组。当前仅有[未执行的静态建议](artifacts/gate05/20261020_dfl_conv_output_native/next_control_plan.json)，不重试完整图、启动恢复矩阵或运行数据集。

目的不是做完整SOTA部署表，而是尽早排除“PyTorch模拟现象无法迁移到TensorRT”的致命风险。

后续真实性评估仍需覆盖以下五种 engine 配置；原始配置已有工程记录，修复候选须另存并重新审计：

1. FP16；
2. 全INT8；
3. G1保持FP16、其余INT8；
4. G1+G5保持FP16、其余INT8；
5. G1+G2保持FP16、其余INT8。

完成完整图兼容性、参考可比性和数据集验收规则冻结后，再用约100–150张RUOD-R图片检查：

- TensorRT全INT8是否仍产生可匹配的新增漏检；
- 恢复G1后是否有相同目标回来；
- TensorRT层融合后，原G1–G8映射能否落实；
- 精度约束是否真的被engine采用；
- 只做初步延迟检查，不提前宣称能耗Pareto优势。

以上五种engine只验证恢复。若声称真实TensorRT上的双向归因，还需增加`q_only:G1`：仅G1为INT8、其余FP16；其他组合也需各自的破坏侧。模拟中的恢复精度是FP32，部署计划为FP16，不能视作完全相同的干预。

### Gate 1：VisDrone干净验证

在查看验证结果前冻结：数据划分、目标级公式、两类传统基线、随机对照、计算预算、候选生成和停止规则。

Gate 1回答机制是否能从水下RUOD-R迁移到非水下密集小目标场景。若不复现，就不能声称通用小目标量化机制。

### Gate 2：完整Orin部署

在论文候选方案冻结后测量真实TensorRT FP16、INT8和混合精度的端到端/纯模型延迟、P50/P95、吞吐、显存、功率、能量/帧、engine层精度和转换节点。

## 7. 运行入口

在本目录打开PowerShell：

```powershell
cd "C:\Users\lenovo\Desktop\视觉边缘研究\07_边缘小目标检测混合精度量化_工程"
```

Gate 0主要命令：

```powershell
.\scripts\run_gate0.ps1 -Phase inspect
.\scripts\run_gate0.ps1 -Phase smoke
.\scripts\run_gate0.ps1 -Phase screen
.\scripts\create_gate0_split.ps1
.\scripts\run_gate0_selection.ps1 -Configuration "q_only:G1"
.\scripts\run_gate0_selection.ps1 -Configuration "r_only:G1"
.\scripts\summarize_gate0_modules.ps1
.\scripts\run_gate0_holdout.ps1 -Configuration "q_only:G1+G5"
.\scripts\preregister_gate0_controls.ps1
.\scripts\compute_gate0_output_reconstruction.ps1
.\scripts\summarize_gate0_control_pair.ps1 -Pair "G4+G8"
.\scripts\summarize_gate0_random_controls.ps1
.\scripts\summarize_gate0_label_free_comparison.ps1
```

Gate 0已经收口。除复现或审计外，不应继续在当前Selection/Holdout试新组合。

## 8. 推荐阅读顺序

1. `README.md`：工程目的、最终结论、证据边界和文件地图；
2. `研究状态与后续计划.md`：通俗解释、当前状态和后续操作步骤；
3. `00_Gate0_最终结论.md`：Gate 0总决策；
4. `01_Gate0_无标签基线比较.md`：候选与无标签基线的最终比较；
5. `02_Gate0_随机对照判定.md`：三个固定随机对照；
6. `04_Gate0_Holdout_G1_G5验证.md`：独立Holdout证据；
7. `07_Gate0_模块排名与组合预注册.md`：G1–G8排名和候选来源；
8. `11_Gate0_数据划分检查.md`与`10_Gate0_Screen数据质量检查.md`：数据边界；
9. `artifacts/tables/gate0_final_decision.json`：机器可读最终状态。

## 9. 完整文件地图

编号表示文件最早在哪个Gate使用：`(0)`模拟量化机制筛查，`(0.5)`真实TensorRT早期真实性检查，`(1)`VisDrone通用验证，`(2)`完整Orin部署。当前项目已完成Gate 0；Gate 0.5 的诊断与原始证据在 `artifacts/gate05/`，仍未通过；Gate 1、2 尚未建立正式实验目录。

### 9.1 根目录报告

```text
README.md                                              (0) 当前总入口与最终状态
研究状态与后续计划.md                                   (0) 通俗说明、路线和操作步骤
.gitignore                                             (0) 缓存、日志和大体积产物排除规则
00_Gate0_最终结论.md                                   (0) Gate 0总审计与最终结论
01_Gate0_无标签基线比较.md                             (0) G1+G5与G1+G2比较
02_Gate0_随机对照判定.md                               (0) 随机对照总体判定
03_Gate0_无标签输出误差基线.md                         (0) 64张校准图的无标签代理排名
04_Gate0_Holdout_G1_G5验证.md                          (0) G1/G1+G5 Holdout对照
05_Gate0_Selection_G1_G5组合.md                        (0) 主组合Selection摘要
06_Gate0_Selection_G1_G3备选.md                        (0) 备组合Selection摘要
07_Gate0_模块排名与组合预注册.md                       (0) G1–G8排名、G1+G5/G1+G3预注册
08_Gate0_候选锁定与Holdout预注册.md                    (0) 锁定G1+G5及Holdout允许配置
09_Gate0_同预算对照预注册.md                           (0) 随机与传统基线的冻结方案
10_Gate0_Screen数据质量检查.md                         (0) Screen完整性、类别和风险信号检查
11_Gate0_数据划分检查.md                               (0) Selection/Holdout冻结与零重叠检查
12_Gate0_Selection_G1归因验证.md                       (0) G1双向结果
13_Gate0_Selection_G2归因验证.md                       (0) G2双向结果
14_Gate0_Selection_G3归因验证.md                       (0) G3双向结果
15_Gate0_Selection_G4归因验证.md                       (0) G4双向结果
16_Gate0_Selection_G5归因验证.md                       (0) G5双向结果
17_Gate0_Selection_G6归因验证.md                       (0) G6双向结果
18_Gate0_Selection_G7归因验证.md                       (0) G7双向结果
19_Gate0_Selection_G8归因验证.md                       (0) G8双向结果
20_Gate0_随机对照_G2_G8.md                             (0) 随机对照1
21_Gate0_随机对照_G3_G8.md                             (0) 随机对照2
22_Gate0_随机对照_G4_G8.md                             (0) 随机对照3
```

每个实际运行的`q_only`或`r_only`还对应一个归档在`archive/gate0_auto_summaries/`中的`Gate0_<subset>_<configuration>_自动摘要.md`。当前包括：

```text
Selection单组：G1–G8的q_only/r_only，共16份
Selection组合：G1+G3、G1+G5、G1+G2、G4+G8、G3+G8、G2+G8的q_only/r_only，共12份
Holdout：G1、G1+G5的q_only/r_only，共4份
```

这些自动摘要只说明一个配置；正式结论以编号报告、机器可读表格和最终审计为准。`archive/`中的文件是历史证据，不是日常入口。

### 9.2 `config/`与`splits/`

```text
config/gate0.json                                  (0) 模型、RUOD-R、阈值、种子和样本量
splits/gate0_selection_images.txt                  (0) 350张Selection绝对路径
splits/gate0_holdout_images.txt                    (0) 150张Holdout绝对路径
```

### 9.3 `scripts/`

```text
run_gate0.ps1                                      (0) inspect/smoke/screen入口
create_gate0_split.ps1                             (0) 冻结Selection/Holdout
run_gate0_selection.ps1                           (0) 运行一个Selection q_only/r_only配置
run_gate0_holdout.ps1                             (0) 运行候选锁允许的Holdout配置
summarize_gate0_modules.ps1                        (0) 汇总单模块并预注册组合
preregister_gate0_controls.ps1                     (0) 冻结同预算对照
compute_gate0_output_reconstruction.ps1            (0) 计算无标签输出误差排名
summarize_gate0_control_pair.ps1                   (0) 汇总一个随机组合的双向结果
summarize_gate0_random_controls.ps1                (0) 汇总三个随机对照并判定
summarize_gate0_label_free_comparison.ps1          (0) 比较G1+G5与G1+G2并收口
```

### 9.4 `src/`

```text
gate0.py                                           (0) 模型加载、Q/DQ、分组、推理和目标匹配
profile_predictions.py                             (0) screen数据质量与风险画像
create_gate0_split.py                              (0) 可复现数据划分
run_gate0_selection.py                             (0) Selection/Holdout单配置干预与摘要
summarize_gate0_modules.py                         (0) 单模块排名与组合预注册
preregister_gate0_controls.py                      (0) 随机池、传统基线与阈值冻结
compute_gate0_output_reconstruction.py             (0) 无标签NMS前输出重建误差
summarize_gate0_control_pair.py                    (0) 随机组合双向目标交集
summarize_gate0_random_controls.py                 (0) 随机对照独立重算与判定
summarize_gate0_label_free_comparison.py           (0) 强基线配对比较和收口决策
```

`src/__pycache__/`是Python自动缓存，不是源码或论文证据；当前已清理，运行Python后可能重新生成。

### 9.5 `tests/`

```text
test_gate0_core.py                                 (0) 分组、配置、IoU、匹配、Q/DQ关闭态测试
test_preregister_gate0_controls.py                 (0) 固定随机组合与传统基线碰撞测试
test_output_reconstruction.py                      (0) 输出误差公式和组合选择测试
```

`tests/__pycache__/`和`.pytest_cache/`是测试缓存，已从当前工程清理；最近一次可用环境测试结果为9项通过。

### 9.6 `artifacts/calibration/`

```text
screen.json                                        (0) 64张正式校准图、哈希和激活范围
```

### 9.7 `artifacts/manifests/`

```text
module_inventory.json                              (0) 88个卷积层、G1–G8和MAC代理
gate0_split.json                                   (0) 数据划分数量、路径与哈希
screen_run.json                                    (0) screen运行条件
gate0_label_free_output_reconstruction_run.json    (0) 无标签输出误差运行条件
selection_<configuration>_run.json                 (0) 每个Selection干预的运行清单
holdout_<configuration>_run.json                   (0) 每个Holdout干预的运行清单
```

配置范围就是上文全部单组、组合和Holdout配置。

### 9.8 `artifacts/predictions/`

```text
screen.csv                                         (0) 500张FP32/全W8A8逐目标记录
selection_<configuration>.csv                      (0) 每个Selection干预的1,920条目标记录
holdout_<configuration>.csv                        (0) 每个Holdout干预的865条目标记录
```

这些CSV是结果的主要可审计来源。自动摘要、双向摘要和最终JSON都应能追溯到它们。

### 9.9 `artifacts/tables/`

```text
screen_summary.json                                (0) screen摘要
screen_data_quality.json                           (0) screen数据质量
selection_module_ranking_and_pair_preregistration.json (0) 单模块排名和组合冻结
selection_candidate_lock.json                      (0) Holdout候选锁
gate0_control_preregistration.json                 (0) 同预算对照冻结
gate0_label_free_output_reconstruction.json        (0) 无标签模块排名和G1+G2选择
gate0_label_free_output_reconstruction_per_image.csv (0) 64×8逐图输出误差
selection_control_G4_plus_G8_bidirectional_summary.json (0) 随机对照1
selection_control_G3_plus_G8_bidirectional_summary.json (0) 随机对照2
selection_control_G2_plus_G8_bidirectional_summary.json (0) 随机对照3
gate0_random_control_decision.json                 (0) 三个随机对照最终判定
gate0_label_free_detection_comparison.json         (0) G1+G5/G1+G2配对比较
gate0_final_decision.json                          (0) Gate 0最终机器状态
<subset>_<configuration>_summary.json              (0) 配置级统计
<subset>_<configuration>_targets.csv               (0) 受影响目标清单
```

### 9.10 `archive/`、日志与缓存

- `archive/gate0_logs/`保存inspect、screen、Selection、Holdout和无标签代理的运行日志；
- `archive/gate0_auto_summaries/`保存每个q_only/r_only配置的自动摘要；
- `archive/gate0_smoke/`保存冒烟测试产物；
- `archive/historical_docs/`保存迁移说明和实验前草案；
- `archive/整理说明与旧文件名索引.md`提供旧名到新名的对照；历史manifest和日志中记录的旧路径保持原样，通过该索引查找；
- `.pytest_cache/`和各`__pycache__/`为可重建缓存，当前已清理；本工程自动生成的`.ultralytics/`设置目录也已清理，以后若有自定义设置应先保留；
- 日志可用于复现和排错，但不能单独作为论文结果。

这里归档的是已有运行记录，不是禁止产生新日志：重新运行脚本仍会向`logs/`写入日志，smoke可能重新产生其校准/预测产物；自动摘要的新输出统一进入`archive/gate0_auto_summaries/`。编号报告按阅读优先级排列，早期报告的阶段性状态不替代`00_`、`01_`的最终结论。

### 9.11 Gate 0.5 实机检查

- `23_Gate05_Orin_FP16首轮检查.md`：首个 FP16 配置的实机结果和限制；
- `24_Gate05_Orin_INT8构建与迁移检查.md`：早期全 INT8 构建失败、本地数值护栏失败及 G4 节点最小复现；
- `26_Gate05_当前可执行工作收口.md`：原六个冻结配置的构建、实际格式及单图执行闭环；
- `27_Gate05_首层量化异常定位.md`：首层 Clip/QDQ 数值异常与孤立算子修复；
- `28_Gate05_融合前缀与完整候选验证.md`：融合前缀、完整 q_only_G1 修复候选及证据边界；
- `29_Gate05_完整参考数值分歧定位.md`：原完整护栏复现、量化边界及完整算子后端对照；
- `30_Gate05_全INT8修复候选验证.md`：参考诊断协议、全 INT8 改写 CPU 等价及 native 格式筛选失败证据；
- `31_Gate05_前缀Clip与精度约束对照.md`：此前单 Conv 四组对照、真实 INT8/GENERIC 融合及同拓扑 NONE/OBEY 输出一致性；
- `32_Gate05_双卷积上下文定位.md`：此前双 Conv 两组对照、实际专用 SWISH/GENERIC 融合、重复及跨标志输出一致性；
- `33_Gate05_DFL内部上下文定位.md`：此前修正 DFL 9 节点局部复现，NONE 成功/OBEY 同类断言失败；
- `34_Gate05_DFL原始拓扑与约束对照.md`：此前原始 DFL 两格与旧修正两格对照，两个 OBEY 均失败、两个 NONE 构建成功但数值表现不同；
- `35_Gate05_DFL_Softmax邻接定位.md`：此前 Softmax 后八节点上下文，以及外部输入与跨阶段输出比较的界限；
- `36_Gate05_DFL卷积输出边界定位.md`：最新六节点 QDQ＋Conv、四维输出契约与旧三维原始数组字节对应，NONE 成功/OBEY 仍同类失败；
- `artifacts/gate05/20261020_prefix_flag_native/`：四组原始构建与重复输出、精确回执、根审计及独立审查；
- `artifacts/gate05/20261020_twoconv_context_native/`：双 Conv 两组原始构建、重复输出、89 文件回执、实际层审计及独立码域/进程退出复查；
- `artifacts/gate05/20261020_twoconv_context_review/next_context_static_options.json`：此前 DFL、PE 和分支/残差的静态选项；本轮只推进 DFL，PE 与分支未执行；
- `artifacts/gate05/20261020_dfl_context_probe/`：72 Conv CPU 上游来源、冻结内部 logits fixture、原/改 DFL tail 共享输入字节等价及密封清单；
- `artifacts/gate05/20261020_dfl_context_native/`：NONE 实际引擎与重复输出、OBEY 编译失败原始证据、88 文件回执及独立审查；
- `artifacts/gate05/20261020_dfl_original_native/`：原始拓扑两格、84 文件回执、四格比较、失败签名与实际层审查；其 Softmax 切口建议已由下一阶段另冻协议执行；
- `artifacts/gate05/20261020_dfl_softmax_cut_probe/`：原 Softmax 单节点、冻结概率 fixture、8 节点切片及严格两次 CPU 字节组成核验；
- `artifacts/gate05/20261020_dfl_softmax_cut_native/`：八节点原始记录、实际层/约束、NONE 保存输出和 OBEY 失败证据；其六节点建议已另冻协议执行；
- `artifacts/gate05/20261020_dfl_conv_output_probe/`：六节点精确闭包、复用概率、一次 CPU 和纯形状转换的新四维 expected 来源；
- `artifacts/gate05/20261020_dfl_conv_output_native/`：最新 89 文件回执、四维实际输出与旧三维字节对应、OBEY 失败和静态下一控制建议；
- `scripts/prepare_gate05_twoconv_context_probe.py`：精确源图闭包、两个小切片 CPU 字节等价与分阶段密封；
- `scripts/jetson_twoconv_context_probe.py`、`scripts/jetson_build_twoconv_context_probe.py`：双 Conv 有界 native 构建、解析约束记录及重复执行；
- `scripts/prepare_gate05_dfl_context_probe.py`：单次上游 CPU fixture 生成、两个 DFL tail 对照及分阶段密封；
- `scripts/jetson_dfl_context_probe.py`、`scripts/jetson_build_dfl_context_probe.py`：DFL 两组有界构建、实际标志与约束记录、成功引擎重复执行；
- `scripts/prepare_gate05_dfl_original_probe.py`：静态复用原始 DFL tail、已认证 fixture 与保存 CPU 输出，不运行新 CPU；
- `scripts/jetson_dfl_original_probe.py`、`scripts/jetson_build_dfl_original_probe.py`：原始拓扑两格有界构建、未修改 Clip 属性及实际约束记录；
- `scripts/prepare_gate05_dfl_softmax_cut_probe.py`：原 Softmax 概率生成一次、新 8 节点 CPU 一次及分阶段密封；
- `scripts/jetson_dfl_softmax_cut_probe.py`、`scripts/jetson_build_dfl_softmax_cut_probe.py`：概率输入的 8 节点有界 native 对照；
- `scripts/prepare_gate05_dfl_conv_output_probe.py`：六节点提取、唯一一次 CPU、四维 expected 纯 reshape 与分阶段密封；
- `scripts/jetson_dfl_conv_output_probe.py`、`scripts/jetson_build_dfl_conv_output_probe.py`：Conv 输出边界的六节点有界 native 对照；
- `artifacts/gate05/20261018_fp16/`：原始 engine、构建日志、资源采样、逐层精度、单图输出、回传哈希和审计摘要；
- `artifacts/gate05/20261018_int8/`：冻结量化图、两次构建失败原始证据及最小节点复现；
- `artifacts/gate05/onnx_module_map.json`：按卷积权重身份建立的 88 层 ONNX 分组映射；
- `artifacts/gate05/data_v1/`：64 张校准图片、原 Holdout 的 150 张图片及标注，只完成清单和文件准备；
- `scripts/audit_gate05_fp16.py`：离线复查回传文件、实际卷积精度、单图一致性和坐标还原；
- `scripts/jetson_monitored_build.py`、`scripts/jetson_trt_fp16_smoke.py`：板端构建监护及单图执行脚本。
- `scripts/prepare_gate05_qdq.py`、`scripts/diagnose_gate05_qdq.py`：显式量化图准备、原始数值护栏和隔离诊断；
- `scripts/jetson_build_qdq.py`、`scripts/prepare_gate05_pe_repro.py`、`scripts/jetson_pe_repro.py`：INT8 构建及失败节点最小复现；
- `scripts/audit_gate05_int8.py`：核对回传证据并生成当前实机构建审计摘要。

## 10. 维护规则

1. 小实验、代码、配置、日志、预测和实验报告放在本工程目录；
2. 论文定位、选题比较和长期路线放在`06_边缘小目标检测混合精度量化_方向`；
3. 新增实验后同步更新README文件地图和最终机器状态；
4. 历史预注册文件保留原样，不用后来的结果倒改旧门槛；
5. 不删除负结果，不只展示有利配置；
6. 不把Q/DQ模拟、ONNX导出或engine成功构建写成Jetson实测结论；
7. Gate 0已收口，任何新方法比较应进入新的干净Gate，而不是继续消耗旧Holdout。

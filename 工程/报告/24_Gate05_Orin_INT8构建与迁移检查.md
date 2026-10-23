> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate 0.5：Orin INT8 构建与迁移检查

客户端日期：2026-10-18。当前结论：**完整 INT8 图在 TensorRT 8.5.2.2 上构建失败，尚未生成可执行的完整模型引擎；本地数值迁移护栏也未通过。Gate 0.5 仍未完成。**

## 本轮范围

本轮仅检查 `q_all`：全部纳入 G1–G8 的 88 个卷积，其权重和输入激活采用显式 INT8 Q/DQ。其他网络算子允许 FP16/FP32，不能将其描述为全网络所有运算均为 INT8。本轮不运行固定 150 张图片的评估，也不运行 G1/G1+G5/G1+G2 恢复配置。

沿用 Gate 0 的模型、64 张训练校准图片产生的冻结范围和 88 层分组。没有重新校准或依据本轮结果更换图片、模块、范围。量化图来源为上一轮 FP16 使用的同一份 FP32 ONNX，输入 `[1,3,640,640]`，输出 `[1,14,8400]`。

量化设置：

- 激活为对称、逐张量、零点 0，scale=`max(冻结absmax/127,1e-12)`。
- 激活在进入 INT8 QuantizeLinear 前先 Clip 到 `[-127*scale,127*scale]`，保持原模拟的窄范围语义。
- 卷积权重为对称、逐输出通道，axis=0，scale=`max(逐通道absmax,1e-12)/127`，零点 0。
- 不额外量化卷积输出；后续卷积输入按其自身冻结范围量化。
- 显式 Q/DQ 的输入量化位置参考 [TensorRT 开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html)和 [NVIDIA 维护者对卷积输入/输出精度的说明](https://github.com/NVIDIA/TensorRT/issues/2383)。实际是否采用 INT8 内核仍由引擎逐层证据决定。

## 本地数值检查：未通过

`artifacts/gate05/20261018_int8/input_v1/protocol.json` 在运行前保存了单图数值容差。对已有 Selection 第一张 `000020.jpg`，比较 PyTorch FP32 Q/DQ 模拟与 ONNX Runtime CPU Q/DQ 执行：

| 项目 | 结果 |
|---|---|
| 全部权重和偏置与源模型核对 | 完全一致 |
| INT8 舍入与窄范围饱和独立样例 | 通过 |
| 全部原始框输出最大绝对差 | 96.570465（640 输入坐标） |
| 全部原始类别分数最大绝对差 | 0.072959 |
| 原始框数值护栏 | 未通过，33,600 个元素中 14,204 个超出预定组合容差 |
| 修改原始容差后改判通过 | 没有 |

这些最大差遍历全部 8,400 个输出位置，不等于最终检测框都偏移了该距离。

进一步隔离检查保存于 `local_divergence_diagnostic.json`、`isolated_quantizer_audit.json`：

1. 88 个量化权重张量在 PyTorch 与 ONNX Runtime 中完全一致。
2. 使用相同的输入张量分别测试 88 个激活量化器，22,835,200 个元素全部一致。
3. 第一层卷积输出已有最大约 `1.62125e-5` 的浮点后端差异；第二层量化输入有 3 个元素出现量化格差异，后续差异逐层传播。该记录支持进一步调查后端浮点计算与量化舍入的耦合，不能把后续全部差异直接视为已穷尽解释。
4. ONNX 图结构审计确认：原始非卷积节点及全部原始 initializer 保持一致；卷积只改接两个量化输入，增加 440 个量化/裁剪节点，冻结 scale 和窄范围边界准确。

相同图片的描述性 NMS 结果仍各有海参、扇贝两项检测，但海参置信度从 PyTorch 模拟的约 0.614 变为 ONNX 模拟的约 0.673；检测数相同不能替代原始数值护栏。未据此调整校准范围或选择其他图片。

## 为何继续实机诊断

在保留完整模型数值失败的前提下，另行冻结 `diagnostic_scope.json`：这里只回答该显式图能否在真实 TensorRT 8.5.2.2 上构建、88 个卷积是否采用真实 INT8 内核、固定单图能否成功执行。它不把本地数值失败改判为通过，也不直接给出迁移真实性、恢复机制或精度结论。

这一诊断没有改变量化图和校准范围。局部量化语义验证与完整模型数值检查是两项不同证据，都保留。

## 实机构建记录

第一次实机构建约 6.63 秒后失败，提示 `Int8 precision has been set ... but int8 is not configured in the builder`。这是构建参数缺失。原日志已保存于 `build_attempt1/`、`monitor_attempt1/`。

第二次仅给构建命令补上 `--int8`，保留 `--fp16 --noTF32 --memPoolSize=workspace:1024 --profilingVerbosity=detailed --buildOnly`。ONNX 图哈希保持相同，第二份输入包保留 `build_revision.json`。TensorRT 日志确认显式精度模式下不会使用 calibrator。`--int8` 开关的配置逻辑见 [TensorRT 8.5 官方样例源码](https://github.com/NVIDIA/TensorRT/blob/release/8.5/samples/common/sampleEngines.cpp)。

板端构建目录：`/home/nvidia/rampq/gate05_int8_diagnostic_20261018_v2/results/build_20261016T100429_415185Z`。

构建监护目录：`/home/nvidia/rampq/gate05_int8_monitor2_20261018_c4d8fcff`。监护温度、可用内存、内核异常及 30 分钟上限，属于尽力监护，无法保证内核挂死后自动恢复。

板端时钟仍为 2026-10-16，客户端日期为 2026-10-18。当前启动编号为 `c4d8fcff-a824-488b-b004-6ad71eeb5d95`，耗时使用单调计时。

第二次构建在监护计时 1605.26 秒（约 26 分 45 秒）后以错误退出，未达到 30 分钟停止上限。实际报错为：

```text
Could not find any implementation for node
model.10.m.0.attn.pe.conv.weight + rampq_model_10_m_0_attn_pe_conv_WeightQ
+ /model.10/m/m.0/attn/pe/conv/Conv + /model.10/m/m.0/attn/Add.
```

该节点属于 G4，是注意力 PE 分支的深度卷积及残差相加融合：输入 `[1,128,20,20]`，权重 `[128,1,3,3]`，group=128、stride=1、padding=1。不能把错误泛化为“所有 INT8 卷积均不支持”，也不能将该层恢复 FP16 后仍称为原定全 INT8 配置。

构建期间 800 条资源采样的最高温度为 54.531°C，最小可用内存约 2467.87 MiB，最大已用 swap 约 0.25 MiB。内核监护日志未捕获新事件；构建后 SSH 仍可正常执行命令。同一启动下属于明确的构建器错误，现有日志没有指向过热、内存耗尽、USB 断连或板子重启。

两次构建的 22 份原始文件已回传并按板端回执核对大小与 SHA256，保存在 `build_attempt1/`、`monitor_attempt1/`、`build_attempt2/`、`monitor_attempt2/`，回执为 `remote_receipt.json`。原始压缩包哈希为 `1af9410aa777c9866871785d73c601df30506ffa6f0240a89f31c57fa170ca86`。

## 最小复现

为区分单卷积与融合问题，提取完全相同的权重、偏置、量化范围和 128 通道卷积，完成三种构建诊断，初始每种最多 90 秒：

| 最小图 | 结果 | 耗时 |
|---|---|---:|
| 单独深度卷积 | 成功构建并反序列化 | 53.103 秒 |
| 深度卷积 + 残差相加 | 复现完整模型的同一融合错误 | 53.001 秒 |
| 深度卷积 + 残差相加，额外暴露卷积输出 | 触及初始上限，证据不足 | 90.256 秒 |
| 上述额外输出方案，仅增加一次资源上限重试 | 成功构建并反序列化；180 秒上限 | 145.510 秒 |

90 秒超时记录保留，没有改写为通过。180 秒重试只改变允许构建的时间，没有改变图、权重、校准范围或数值容差。原始结果位于 `pe_repro_results/`，重试位于 `pe_tap_retry180/`，资源预算说明为 `pe_tap_retry180_request.json`。

成功的最小图中，engine inspector 显示 `CudaDepthwiseConvolution` 使用 Int8 权重、向量化 Int8 输入、FP32 输出；残差相加为独立 PointWise 层，不再包含于卷积融合名称。该卷积层没有提供 `TacticName`，所以这里报告可见的权重和输入格式，不进一步声称 Tensor Core 指令类型或性能收益。两份小图 engine 只完成构建和反序列化，没有执行数据推理。

这组受控复现表明：相同卷积单独可构建，接上相加会触发失败，额外暴露卷积输出可以让本节点以分开的算子构建。该结论仅适用于本节点最小图，尚不能证明完整模型一定能采用该办法成功构建。最小复现额外回传的 14 份原始文件已按 `pe_repro_remote_receipt.json` 校验，本轮共核对 36 份板端文件。

## 已准备的完整图候选

`pe_tap_full_graph_candidate/model_q_all_pe_tap.onnx` 在失败的完整量化图上，仅把该 G4 卷积输出增加为第二个图输出，同时补齐形状元数据。全部算术节点和 initializer 保持逐字节一致，量化位置、范围、分组和主输出声明保持一致；结构检查已通过。候选 SHA256 为 `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`。

该完整候选尚未在板端构建或执行。额外输出会影响融合和浮点精度选择，因此下一轮还需适配额外绑定、重查所有实际层精度；后续逐目标比较应控制各 engine 的图输出结构及非量化浮点精度。不能把节点级构建成功直接上升为完整 INT8 模型已修复，也不能据此抹去原始数值迁移护栏失败。

## 结果文件与下一步

本轮证据统一位于 `artifacts/gate05/20261018_int8/`，`checkpoint.json` 保存当前机器状态，`audit_summary.json` 为核对过原始文件的机器可读摘要。`input_v1/` 保留原始失败记录，`diagnostic_bundle_v1/` 与 `diagnostic_bundle_v2/` 保存两次构建输入；不得删除第一次失败结果。

若生成完整引擎，先使用 `scripts/audit_gate05_int8_precision.py` 检查原始 Conv 身份、INT8 权重、INT8 输入格式、整数 tactic 及分组覆盖。该审计器已用既有 FP16 引擎作反例检查，88 个 FP16 卷积均不能被判为 INT8。它当前会对缺失整数 tactic 名称的层保守报需复查，后续必须依据完整 inspector 记录处理，不能为了通过而忽略未确认项。

通过实际精度审计后，才运行 `scripts/jetson_trt_int8_smoke.py` 的同一 Selection 图片执行检查。后续完整 Gate 0.5 仍需冻结数据集判据、检查三种恢复 engine、完成同一图片集的逐目标匹配，并交代本地数值迁移差异。正式延迟、FPS、功耗及双向归因尚无本轮证据。

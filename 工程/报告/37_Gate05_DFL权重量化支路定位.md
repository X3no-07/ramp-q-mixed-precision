> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate 0.5：DFL 权重量化支路定位

本轮仅旁路 DFL 的显式 WeightQ／WeightDQ，保留原概率输入、激活 Q→Clip→DQ、源权重字节和四维输出边界。新候选的 NONE、OBEY 两组均构建成功，并各执行两次，重复输出字节一致；实际 Conv 的输入、权重和输出描述均为 FP16。因此这是权重量化支路及其解析／格式上下文的局部控制结果，**不是 W8A8 修复或 Gate 通过**。

与报告 36 的六节点图不同，本候选取消了权重量化的数学操作，必须使用自己的新 CPU 参考。原完整 PyTorch／ORT 护栏仍为 `FAILED_UNCHANGED`，历史完整 corrected q_all 的 OBEY 失败不被覆盖，Gate 0.5 为 `INCOMPLETE`。另一项完整 q_all NONE 构建控制已单独冻结协议，其结果不属于本报告。

## 冻结来源与唯一变更

完整 corrected q_all 源图 SHA256 为 `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`。父六节点图 SHA256 为 `805abb78ed4e4d5abc8041b2916dadc83370a14ef1776fcc0a03f163c16bcf81`，父完成检查点 SHA256 为 `c5e4a11a37d32a6de9df22c42463de0a6a58ad4b92c97b64f4c318f71dd9947e`。

新图只删除 `rampq_model_23_dfl_conv_WeightQ` 和 `rampq_model_23_dfl_conv_WeightDQ`，将 `/model.23/dfl/conv/Conv` 的 `input[1]` 从 `rampq_model_23_dfl_conv_w_dq` 改接原 initializer `model.23.dfl.conv.weight`。恢复这一输入后，Conv 序列化字节与源节点相同；三个保留的激活节点、五个使用中 initializer 和 I/O 声明均保持来源字节。原权重 FLOAT `[1,16,1,1]` 的值仍为 0～15。旧 weight scale／zero 仅保留为来源证据，不再接入候选图，也未被挪作激活或输出尺度。

候选共 4 节点、5 个 initializer、1 个 Conv。输入仍是已有 CPU 概率 `/model.23/dfl/Softmax_output_0`，FLOAT `[1,16,4,8400]`；复用 NPY SHA256 `5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520`，未重新生成。输出仍为 `/model.23/dfl/conv/Conv_output_0`，FLOAT `[1,1,4,8400]`，表示 DFL 距离，未解码为检测框。

协议 `artifacts/gate05/20261021_dfl_weight_bypass_protocol/protocol.json` 在准备与执行前冻结，SHA256 `9b2ea26ff12ecd5a4b5a699b09c511a3ea6c2187125d037117bb85f57b7b1170`。两 case 为 `weight_bypass_none`、`weight_bypass_obey`，各最多构建一次、300 秒，batch 900 秒，成功各执行两次。共同设置为 INT8＋FP16、无 TF32／PREFER、1024 MiB workspace、avgTiming=1、DETAILED、外部 FLOAT／LINEAR；仅修正导入 Clip 属性为 ±127，不设置或 reset 层 precision／output type。

## 新候选 CPU 参考

仅执行新四节点图 CPU 一次，使用 CPUExecutionProvider、ORT_DISABLE_ALL、4 threads；未重跑 Softmax、72Conv 上游、旧 DFL 或完整检测器。新输出 33,600 个 float32 元素全部有限。新 expected NPZ 仅含 `weight_bypass_output`，来自这次候选实际输出，而不是旧六节点参考。

| 文件 | SHA256 |
|---|---|
| 新四节点 ONNX | `020dad7f70c04f495dfe1b6e62e63e925ff6468a6cde0c157b7122a38274eb41` |
| 新 CPU actual NPY | `808e4a1a17e2f9fc622d858a3dba8390e45749a72afb0ac6a8fd3d8ac413cf73` |
| 新 expected NPZ | `dff4725f1dbbb99979723b47267973f9cde5bc390db30131ad5d39361d958526` |

新候选 CPU 与保存的旧六节点 W8A8 CPU 输出有 33,592／33,600 个元素不同，最大绝对差 `0.05133557319641113`，平均绝对差 `0.019477537332845497`。这是旁路权重量化改变数值语义的直接记录；协议不要求旧新字节相等，没有采用旧参考或新增容差。

准备前后 73 份来源身份保持，新 native 工具在封包前后保持。封包 `artifacts/gate05/20261021_dfl_weight_bypass_probe.zip` 为 11,664,531 bytes，SHA256 `a86af92bdce3b72408c84f2236fe6f495baa5366d20574e97f045f1cfd1eb13b`；spec SHA256 `df7de4468cec5cb0672f89c61e63ce26decbb1d6bc52a521e453e02aad9c5b4f`，manifest SHA256 `af7e0bb5db03cd228d05df912cf4298f19ad2715a476f7c1757200fbaf1b8d3a`。

启动前独立静态审查 301 项通过，`artifacts/gate05/20261021_dfl_weight_bypass_review/pre_native_review.json` SHA256 `4f314e7457fa3a4ad572323988db23f875baebe38573d815b1b9f43e4b41c623`。板端纯标准库输入检查通过，未导入执行 runtime。

## Native 构建、实际格式与原始输出

两组各构建一次，共保存两个引擎、四次固定输入执行，没有更换设置重试。Builder API 时间和 monitor 时间只作为诊断执行记录，不代表推理性能。

| case | Builder API 秒 | 引擎 bytes | 同组两次输出 | 实际 Conv 输入／权重／输出 |
|---|---:|---:|---|---|
| weight_bypass_none | 55.092813 | 32,247 | 字节一致，全部有限 | FP16／Half×16／FP16 |
| weight_bypass_obey | 54.787531 | 30,475 | 字节一致，全部有限 | FP16／Half×16／FP16 |

NONE 引擎 SHA256 为 `3259a455db211ea42d7c8a92b801369bf1727dd99f922c1ded2a002bddf75dfd`，OBEY 引擎 SHA256 为 `6efe9963f608ee3bc3b7f57b8c9a88c650188434ecc4a8e0093ddc18fa6144e2`。两组均唯一映射到 `/model.23/dfl/conv/Conv`，kernel／stride 为 `[1,1]`、Groups=1、OutMaps=1、无 bias、Activation=NONE。

NONE 为 `CaskConvolution`，内部输入与输出为 two-wide FP16，tactic 为 `ampere_fp16x2_hcudnn_fp16x2_128x32_relu_interior_nn_v1`。OBEY 为 `CudnnConvolution`，内部输入与输出为 linear FP16，inspector 未给 `TacticName`，保留 `TacticValue=0x0000000000000038`。后者是字段缺失，不是构建或推理失败，也不补造 tactic 名。两组外部 binding 仍是 FLOAT／LINEAR，不能将外部 FLOAT 输出解释成内部 Conv 以 FP32 计算。

两组解析网络均为 6 层，实际保留 1 个 DQ precision INT8 和 1 个 Q output type INT8，无查询错误；Clip 修正前后请求一致，两 case 的请求相同。父六节点图的 2＋2 个请求随权重 Q/DQ 节点移除自然减少，并非通过手工清除约束实现。实际 flags 仅 OBEY 开关不同。INT8 builder flag、显式激活 Q/DQ 或源 FP32 initializer 均不能代替实际算子精度证据；本候选实际 Conv 为 FP16，不满足原 W8A8 计算要求。

相对本候选自己的 CPU 参考，保存数组复核如下。所有数字为描述，未定义新通过容差。

| 比较 | 不同元素／33,600 | 最大绝对差 | 平均绝对差 |
|---|---:|---:|---:|
| NONE vs 新候选 CPU | 33,594 | `0.010239601135253906` | `0.0007279089507868603` |
| OBEY vs 新候选 CPU | 33,594 | `0.00620269775390625` | `0.0004970305783319331` |
| NONE vs OBEY | 8,452 | `0.0078125` | `0.0005338795979817709` |

runner 共 123.352584 秒，monitor 共 129.53 秒。`BUILD_SCRIPT_OK` 表示按协议完成批次；本轮两组成功由各自引擎、build result 和 raw 数组单独证明。guard COMPLETE、未干预、私有组为空，没有资源停止或超时。

## 证据完整性与复读范围

原始归档 `artifacts/gate05/20261021_dfl_weight_bypass_native/native_evidence.zip` 为 12,003,589 bytes，SHA256 `5353ea2f980c5b9e0a50f63cb947c737551e5bd52978970f2f8ab282a0444c0c`。回执列出 93 份文件，另加回执自身；归档 CRC、精确文件集合、大小和 SHA256 均通过。回执 SHA256 `b490e6ed56b928d430e43076da2f6a2d53d84b3204f0483fd147246a7a019875`。

根审计 `root_native_audit.json` SHA256 `20a386e6b359608356fbd6ccf61b4d11bba1fedb56e89471ccdbd571c9a2995d`。保存证据复读文件为 `independent_native_review.json`，519 项检查通过，SHA256 `9bf0cf635c69bbf5a4d08045e645d24cc1a121cc65a56ab5ec2e81a4d2494baf`。该复读由本轮 preparer／runner／builder 的同一作者 agent 完成，已在记录中披露：它逐字核验与重算保存证据，不是外部独立作者的审稿或最终仲裁，仍需根任务或其他审查者核关键结论。复读未导入 ORT／torch／TensorRT，未新增模型 CPU／native 推理或 SSH，原始树未改。

根任务另取最新板端观察：客户端 2026-10-21 02:03:54，板端 2026-10-18 08:14:50，未校时；启动编号仍为 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`。monitor、runner、guard、两个 builder 的五个 PID（40168／40173／40179／40186／40222）均不存在，无任务进程；远端 93 份回执文件及归档仍匹配。风扇和 jtop active，NRestarts=0，可用内存 5006 MiB。非管理员功耗查询显示 15W／mode 2，同时仍有 EMC 读取权限错误，不能视为完整特权功耗核验。

该观察还确认 corrected q_only_G1 的引擎 SHA256 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`、父六节点 NONE 引擎 SHA256 `961e821b6d2965078edc8209feb2b41929012a6522df027b71a2f83834056836` 保持；父六节点 OBEY 仍无引擎，观察时历史完整 corrected q_all 仍无引擎。终态保存在本轮 `final_board_observation.json`，与本地后续完整图新控制分开记录。

## 结论与完整图后续控制

父六节点图的 OBEY 在相同局部概率输入下失败；移除显式权重 Q/DQ 的新候选两组均成功。这个变化支持继续检查权重量化支路及其关联 parser／格式选择上下文，但同时改变了数学语义、固有请求集合和实际计算路径，不能唯一归因于某个请求或节点。尤其是新实际 Conv 转为 FP16，不能把构建成功称为原 W8A8 完整模型的修复。

不再据此自动扩展下一局部 pair。完整 unchanged corrected q_all 的独立 NONE 控制已在 `artifacts/gate05/20261021_qall_none_protocol/protocol.json` 冻结，SHA256 `572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1`：原 source bundle 的 OBEY spec／合同字节保持，由新外层控制描述单次 NONE 构建；复用已有完整图 CPU 等价证据，新 CPU 为 0。该协议要求构建后先独立核实际精度，再另冻固定图像推理协议，不自动推理或进入数据集。本报告不提前宣告那项完整图控制兼容或成功。

静态库中的 SASS 指令或 kernel 字符串可用于后续规划，未经执行关联与实际路径核验，不构成此模型运行时算术通过。原完整护栏、旧 OBEY 失败、七项 depthwise 严格 INT8 tactic 缺口及 Gate `INCOMPLETE` 均保留。

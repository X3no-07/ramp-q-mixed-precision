> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate0.5：完整图 NONE 控制与验收推进

更新：2026-10-21。本文件是过程报告，Gate0.5 总状态仍为 **INCOMPLETE**。

完整 corrected q_all 已在原 Orin 同一次启动上构建成功。采用显式 Q/DQ、全局 INT8+FP16、NONE precision constraints、TF32 关闭；不设置或清除逐层 precision/output type。源模型的 88 处 code Clip 仅在原生导入时修正为 [-127,127]。权重、量化范围、组划分及旧源合同不变。

成功引擎 SHA256：`c3d9e75d61452c69b97097d1320221d5d56d7ad0e2fc1370480cf3e09eb3a41e`。构建 API 耗时约 887 秒；此项是编译时间，不是推理性能。原生 inspector 的 88 个源卷积全部唯一映射，INT8 激活输入和权重格式通过原规则。81 层具有命名整数 tactic；另外 7 层原有“缺少 tactic 名称”的记录保持原样。

随后在固定 000020 图片上执行两次完整引擎。两个输出均为有限值，两次原始数组逐字节一致，按既定 confidence=0.25/NMS IoU=0.7 产生 2 个检测。单图结果不能证明数据集精度、漏检恢复或 G1 因果关系。

这次自己的 Nsight 运行记录、源层 NVTX、CUDA API correlation、实际内核、SM87、进程加载库设备号/inode/SHA256，已与 4 个确切 mangled kernel 的静态 signed INT8 点积累加路径对应。7 个缺名层分别具有两次本次执行证据；独立审查 161 项通过。这是单独标记的运行内核与静态算术证据，不为缺名层编造 TacticName，也不声称逐指令硬件跟踪、所有内核纯 INT8 或 Tensor Core 使用。静态反汇编本身不能代替本次运行身份链。

执行正常结束，板子 boot ID 未变，监护没有干预。额外核实 profiler 创建的独立 runner 进程组也已退出；原 wrapper 组退出本身不足以证明 profiler 子进程已全部退出。后续 profiler 执行需要覆盖这一额外进程组的退出边界。

五个新的匹配对照已完成 9 次预定 CPU 准备并封装：FP16、G1 恢复、G1+G5 恢复、G1+G2 恢复及仅 G1 量化。每个对照将单独构建一次，依次运行、上限 7200 秒。全局 FP16 开启；FP16 对照 INT8 关闭，其余 INT8 开启。这一差异明确保留，不能称所有构建旗标完全一致；实际浮点路径和融合仍需审查。全局旗标和期望掩码不能证明恢复层确实采用 FP16。

后续必须完成五个实际引擎的精度/映射/融合审查、独立完整 ONNX 参考、固定 150 图/865 目标同目标比较及最终判定。G1 为预定主比较，两个组合恢复是次级结果；不得根据结果更换主比较、图像、范围或阈值。合格完整实验没有出现新增漏检或同目标恢复，应如实报告科学 FAIL；技术控制未完成应报告 INCONCLUSIVE，不能借构建成功改成 PASS。

## 后续已完成与明确范围

五个新的 NONE 对照已各构建一次成功。按原实际精度规则，FP16 对照为 88 FP16；G1 恢复为 82 INT8+6 FP16；G1+G5 为 74 INT8+14 FP16；G1+G2 为 77 INT8+11 FP16。每个 source 都按原融合和权重规则完整覆盖。前三个量化恢复配置各有 7 处缺少命名 tactic，需要其各自的新运行身份链，不能借 q_all 的 trace 自动通过。

新 NONE q_only_G1 的 6 个 INT8 层正确，但 PE Conv 实际是 FP32 输入、Float 权重和 FP32 输出，只有 81 个恢复层达到 FP16。此引擎不符合预定 82 FP16，失败报告保留。最终辅助配置事前选择此前同一完整 ONNX 文件（SHA `3f7d481fdf5268bea3d66c2176827dffaf3bd0628f5a2c412d48e3b6c53e9356`）的合格旧引擎 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`。旧引擎具有 6 INT8+82 FP16 和 6 个命名整数 tactic。选择依据是实际精度资格，不是新的 150 图检测结果；没有重新构建或重跑 CPU 模型。该辅助控制采用 PREFER 和 82 个显式 FP16 compute/output 请求，不能冒充新 NONE 成员、相同构建旗标或纯 G1 算术对照。精确选择政策另存 `auxiliary_qonly_selection_policy.json`。

新 FP16、G1、G1+G5、G1+G2 均完成各两次固定单图执行，共 8 次，无重试。每项监护正常退出，wrapper 和 profiler 独立 runner 两个进程组都已物理清空，20 个相关 PID 全部不再存在；所有 204 份回收文件与板端新鲜 SHA 和归档 receipt 一致，boot ID 未变。不能据这些单图检查宣布 150 图精度或 Gate 通过。

实际浮点元数据也已逐项检查：FP16 基线的 DFL Softmax/解码为 FP16，量化配置为 FP32；q_all 与 G1 恢复的 DFL/解码同为 FP32，但部分未改源节点的 SiLU/residual 浮点划分仍不同。原未冻结草案中更强的逐算术匹配判断另行保留。最终研究问题按原路线限定为“真实部署配置中的同目标消失/恢复存在性”，将这些编译实现差异完整披露，不主张纯 G1 算术因果或相同浮点计算。未认证的源/量化范围/权重精度/融合或身份仍是技术缺口，不会因这个范围解释而放宽。范围政策 SHA 为 `e3468871a501a77afb81d5a15facf1bda9d706adcadd65c0367e95569d759e46`；此决定发生在新的 150 图模型执行之前，不修改任何旧结果。

150 张图统一预处理已经完成，并逐文件传到板端核验；所有配置和独立 ORT 参考将复用同一 NPY。首次 OpenCV 因中文 Windows 路径无法打开第一张图片，完成数组数为 0、模型执行数为 0，失败目录保留。新单独版本仅改为读取原始字节后用同一 OpenCV 解码，其他预处理逐字节不改；新的 150 个输入已通过完整源前后 SHA、NPY 和 ZIP CRC 检查。共享输入 manifest SHA 为 `05c6509f5afcf6a715a2328aac9581ca2f9534ee5de6fcffc670a3fd745f4f80`。

历史完整 PyTorch/ORT 数值预检始终为 **FAILED_UNCHANGED**，其原始容差、记录及所有旧编译失败均保留。新独立 ONNX 部署参考不是对旧失败的重新命名。

证据入口：

- `artifacts/gate05/20261021_qall_none_native/`：成功完整构建、归档验证、原规则精度审查。
- `artifacts/gate05/20261021_qall_none_fixed_native/`：本次单图、原始数组、profile、运行加载库和独立整数算术链审查。
- `artifacts/gate05/20261021_qall_none_entry_failure/`：首次入口缺少参数的失败，未进入 TensorRT 构建；另存适配入口后在新目录构建。
- `artifacts/gate05/20261021_matched_control_family/`：五个独立准备/封装对照。
- `artifacts/gate05/20261021_matched_control_family_native_protocol/protocol.json`：五次串行构建的独立冻结合同。
- `artifacts/gate05/current_execution_state.json`：当前状态；早期过程状态已另存快照。

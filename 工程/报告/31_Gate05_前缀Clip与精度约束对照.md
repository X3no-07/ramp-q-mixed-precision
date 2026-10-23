> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate 0.5：前缀 Clip 与精度约束对照

2026-10-20，接续[全 INT8 修复候选失败](30_Gate05_全INT8修复候选验证.md)。本轮使用已执行过的单卷积前缀，区分 Clip 结构和 OBEY 标志对局部构建与数值的影响，不重跑完整模型。

**结果：四组均构建成功并执行两次；相同 Clip 结构下，开关 OBEY 的原始输出逐字节相同。完整 q_all 的格式筛选错误没有在这个单 Conv 上下文复现，因此尚不能把它归因于 OBEY。改写结构显著减小了此固定输入上的数值偏差，但仍有 7 个元素相差一个量化步长；Gate 0.5 仍未通过。**

## 冻结范围

前缀包含 11 个节点、12 个初始化器：第一处激活 Clip/Q/DQ、权重 Q/DQ、一个首 Conv、SiLU，以及第二处激活 Clip/Q/DQ。唯一输出为 `rampq_model_1_conv_a_dq`，形状 `[1,16,320,320]`。第二 Conv 不在图中，不能把它称为双卷积前缀。

输入来自原固定 `000020.jpg`；复用原图、改写图、预处理 NPY 和保存的 CPU 参考，不重新标定、训练或选择图片。原完整 PyTorch/ORT 护栏保持 `FAILED_UNCHANGED`，原全 INT8 失败和仅量化 G1 成功记录都保留。

协议 `artifacts/gate05/20261020_prefix_flag_protocol/protocol.json` 在 native 前冻结，SHA256 `c7fdcada89c685ed52f8f65d8318854982d58f52f16cfce44f3129cdd17a307d`。

| 顺序 | Clip 结构 | OBEY |
|---|---|---|
| original_none | 原浮点 Clip→Q→DQ | 关闭 |
| original_obey | 原浮点 Clip→Q→DQ | 开启 |
| code_clip_none | Q→整数码 Clip→DQ，修正导入边界 | 关闭 |
| code_clip_obey | Q→整数码 Clip→DQ，修正导入边界 | 开启 |

其余设置相同：TensorRT 8.5 Python Builder、INT8＋FP16、关闭 TF32 和 PREFER、1024 MiB workspace、avgTiming=1、详细 inspector、外部 FLOAT/LINEAR。没有层精度/输出类型 setter，也不重置解析器自动约束。每个 case 保存校正前后逐层约束及真实配置标志，不能把脚本 setter 次数当作解析网络约束总数。

每个 case 只尝试一次、构建预算 300 秒，四个 case 总监护 1500 秒。普通编译错误保存后可继续下一个冻结 case；超时、资源监护或外部中断停止后续任务，清理私有进程组。成功引擎才核验绑定、保存实际层信息并执行两次相同输入，失败引擎不执行。

## 前置核验

两份原前缀图的 SHA256 分别为：

- original：`22b1065c01792492d658596c1069aee074e054f32648fa90bf87cc39a6d11a33`。
- code_clip：`d2b8ca8105f8a05fabb7aac66c6fbab1a2558ac35bdd5ea1736ce8e0052465de`。

原预处理 NPY SHA256 为 `24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94`。独立读取核验确认两图各为单 Conv，保存的两份 CPU 期望输出逐字节一致；没有为此次准备重新执行 CPU 推理。

Windows 当前 USB 网卡是“以太网 3”，地址仍为 `192.168.55.100`；已实时恢复 USB SSH，板端启动编号仍是 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`。准备时风扇 active、jtop active 且 NRestarts=0，可用内存约 4936 MiB，无遗留构建任务。没有改变系统、功耗模式或时钟。

## 本轮结果

最终密封包采用 `artifacts/gate05/20261020_prefix_flag_probe_v2.zip`，1,071,567 bytes，SHA256 `7a6a1d88a50f9db72ac48aeed3a0c2923068031f956393b5fb5c4997f32fcc96`；spec SHA256 `66b1cf24fb5667fa92fe31c6e8ca483d63eef4495b6c4cc53a120f8ec7e28056`，manifest SHA256 `aa1959c7f4b4569f5129533fc51b3f669bb659f51d351438e0a6840547feee19`。

首版密封包在独立检查前后发现异常中断的证据文件保全还需修正，原包保留且未部署。v2 使用单独的 `runner_abort.json`，不覆盖 child 的失败原件；两个版本及源码哈希关系保存于 `20261020_prefix_flag_bundle_revision_lineage.json`，没有修改冻结的图或协议。

启动前独立审查 245 项通过，报告 `artifacts/gate05/20261020_prefix_flag_review/pre_native_review.json` SHA256 `bade9ebf0401c0b208bf1b2d9f871915bf41482bf1f42a287e6af959c0b0d10d`。板端再次核验包、监护工具和当前进程后，在新目录 `/home/nvidia/rampq/gate05_prefix_flag_20261020/gate05_prefix_flag_probe_20261020` 启动四个 case。

monitor PID 30071、私有组 30076、guard PID 30078。四组各尝试一次，均成功构建并执行两次。Builder API 和总监护用时如下，只是诊断耗时，不是推理性能测量。

| case | Builder API 秒 | 构建与两次执行 | 相对冻结 CPU 参考不同元素数 / 1,638,400 | 最大绝对差 |
|---|---:|---|---:|---:|
| original_none | 103.033 | 成功，重复输出逐字节一致 | 769,777 | 36.65729832649231 |
| original_obey | 102.935 | 成功，重复输出逐字节一致 | 769,777 | 36.65729832649231 |
| code_clip_none | 68.616 | 成功，重复输出逐字节一致 | 7 | 0.7481081485748291 |
| code_clip_obey | 68.935 | 成功，重复输出逐字节一致 | 7 | 0.7481081485748291 |

runner 共 362.849 秒，外层 monitor 共 368.42 秒，返回 `BUILD_SCRIPT_OK`。没有超时、资源停止或 guard 干预；私有组最终为空。四个 builder、runner、monitor、guard 共七个已记录进程均退出。结果全部有限。

原结构两组的 raw 输出 SHA256 均为 `27b5aa8940d2dd83edb62d270e427f82c0f8dfdd9387ab2638b6790e2b76ca6c`；改写结构两组均为 `15a6b790c26da42840120bde810c10c03e74a7bf19bb685c6552c967ebfbb2db`。这里比较的是数组连续原始字节，不是序列化 engine 的相同与否。两个结构的输出也分别与此前保存的同结构前缀 raw 逐字节一致。

两种结构的冻结 CPU 参考本来就逐字节相同。改写后的 7 处 native 差异最大值等于第二激活边界 scale `0.7481081485748291`，不能写成 native 与 CPU 完全一致，也没有据此另设通过容差。

## 实际精度、融合与解析约束

四组 inspector 均只有一个实际卷积，输入、权重和输出都是 INT8；权重数量 432、输出通道 16、group=1，与源 Conv 对应。外部输入输出仍为 FLOAT/LINEAR。

四组使用同一命名整数 tactic：

```text
sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize256x32x64_stage4_warpsize4x1x1_g1_tensor16x8x32_t1r3s3_linkable
```

激活为 `GENERIC`，逐点表达式含 5 个操作，以 Tanh 表达 SiLU；这与完整 q_only_G1 修复引擎首 Conv 的专用 `SWISH` 描述不同，不能用此前完整引擎的融合结论替代本轮实际记录。

每组解析网络均为 18 层，具有 3 个显式 INT8 precision 约束和 3 个显式 INT8 output type 约束，查询没有不支持或错误。Clip 属性校正前后，逐层约束保持相同。同拓扑 none/obey 的图哈希和解析约束相同，实际配置标志只有 `OBEY_PRECISION_CONSTRAINTS` 改变；PREFER 始终关闭。三处解析约束不是脚本新增 setter：脚本没有层精度、输出类型 setter 或 reset。

## 原始证据与终态

原始归档为 `artifacts/gate05/20261020_prefix_flag_native/native_evidence.zip`，2,219,829 bytes，SHA256 `a66da15dad97b1a807376b8281fa19457d749ae036ef6cad88b67113171cb8b6`。回执列出 107 份原始文件，另加归档中的回执元数据本身；文件清单、大小、SHA256、ZIP CRC 均核验。回执 SHA256 为 `344e92d945d4d8697f3f76c056e5941a5e0dc900688d650766710a3a221734e3`。最后通过 SSH 再次确认板端 107 份文件仍匹配，归档哈希未变。

独立 native 审查 530 项通过，见 `independent_native_review.json`，SHA256 `6e136e642346a90c0a210b098f8072791b26ab5a25863b70241fadedd5875205`。根审计 `root_native_audit.json` SHA256 为 `1351ee0ad9caa40888e716452dd0a4f767793693f9d8f14867a7a26134a45492`，同时核验实际格式、约束、raw 数组及证据前后不变。

根审计的前两次本地失败及对应源码保存在 `root_audit_attempt1/` 和 `root_audit_attempt2/`，不是 native 失败或重跑。第一次遗漏非有限元素计数字段；补齐后，第二次发现 Windows 重算与板端保存的跨结构 RMS 仅差 `2.220446049250313e-16`（`1.3701950732015873` 与 `1.370195073201587`）。原始字节、有限性、不同元素数和最大差均一致。最终审计把两端 mean/RMS 及差值并列保存，作为描述性浮点归约，不以它们的末位字节一致性作为新精度判据；原护栏和原始证据没有改变。

最终板端观察时，客户端为 2026-10-20，板端时钟仍为 2026-10-17，二者分别记录，没有校时。启动编号仍为 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`，风扇 active、jtop active 且 NRestarts=0、可用内存 5018 MiB，无本轮任务在运行。这些是本轮观察，不证明此前断连原因已消除。功耗查询显示 15W / mode 2，同时仍有非管理员读取 EMC 权限提示，未将它写成完整特权检查通过。

此前 q_only_G1 完整引擎仍为 SHA256 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`；失败的 corrected q_all 目录仍无 engine。本轮没有覆盖它们，也没有运行完整模型重试。

## 解释边界

四者都成功，说明完整 q_all 的失败未在此前缀复现；在此单 Conv 上下文，启用 OBEY 本身不足以触发该错误。不能反过来宣布完整模型可构建，也不能排除 OBEY 与其他层、分支或融合上下文的交互。

数值比较仅描述原始数组、重复一致性、实际 tactic 与融合差异，没有新设容差，不替代原完整护栏。没有检测头、GT 匹配、数据集、目标恢复或性能结论。Gate 0.5 仍为 `INCOMPLETE`。

下一步从实际失败的 corrected q_all 源图确认并冻结新的最小图上下文切片，先检查依赖、shape 与 CPU 语义，再进行一次有界 native 验证。新增第二 Conv、分支或残差都应作为新的上下文记录；错误不保证随前缀长度单调出现，不能把首个失败扩展新增的节点直接认作根因。当前不直接切换全图 NONE/PREFER、不清空解析约束、不重跑完整模型或恢复矩阵、不进入 150 图评估。

本轮闭环见 `artifacts/gate05/20261020_prefix_flag_native/checkpoint.json`，当前队列状态见 `artifacts/gate05/current_execution_state.json`。

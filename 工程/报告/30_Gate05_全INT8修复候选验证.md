> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate 0.5：全 INT8 修复候选验证

2026-10-19，接续[完整参考分歧定位](29_Gate05_完整参考数值分歧定位.md)。本轮另存的 q_all 候选通过静态及完整 CPU 改写等价检查，板端 88 处 Clip 属性也已校正；但 TensorRT 在约 13 秒出现格式筛选内部错误，未生成引擎。失败已完整回收并独立复核，没有重跑本轮全模型。原完整 PyTorch/ORT 护栏保持失败；此前 q_only_G1 的成功不被覆盖，没有启动其余恢复矩阵或数据集评估。

## 本轮参考协议

`artifacts/gate05/20261019_reference_contract/reference_contract.json` 已在本轮 native 构建前冻结，SHA256 `87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430`。独立方法审查 17 项通过，审查报告 SHA256 `8c8e46719852b18f5181edb36c3edb5be888d2063491b3076d9d8aaf6e23092d`。

协议保留四个不同角色：原 Gate 0 PyTorch Q/DQ 是历史模拟参考；原 ONNX CPU 是图改写对照；上一轮统一 Conv/SiLU 的 hybrid 只用于定位；本轮 TensorRT 是新的诊断引擎。它们不能互相冒充。协议不是原 Gate 的通过条件修改，也不把 hybrid 的收敛结果改写成原 PyTorch/ORT PASS。

本轮采用同一固定图片、冻结权重和标定、原 NMS。进入 native 前，要求静态差异范围准确、主辅 CPU 输出逐字节一致、来源审计零错误、工具输入核验及独立代码审查通过。native 后仍使用未修改的实际格式、命名整数 tactic 和融合门槛。

若实际 88 层格式、映射和融合均完整，但命名整数 tactic 证据不完整，只允许沿已有显式 `--diagnostic-formats-only` 分支执行固定图，保留严格整数证据未完成状态。新引擎的缺口必须逐项新查，不能复制旧七层结果，也不能豁免它们。

## 候选与原字节核验

父图是原 q_all＋PE 辅助输出图，SHA256 `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`。新图 795 个节点、703 个初始化器；仅改 176 个 Clip/Q 节点与 176 个 Clip bound 初始化器。88 个 Conv、所有 DQ、scale/zero、权重及 bias、其他节点和外部输入输出声明保留原字节。

原拓扑 `FLOAT Clip→Q→DQ` 改为 `Q→INT8 Clip[-127,+127]→DQ`。native Builder 还须在 ONNX 解析之后、构建之前将 88 个 Clip 的 alpha/beta 显式校正为 -127/+127，并保存修改前后的实测属性。

父 spec 的 precisionConstraints 为 `obey`，新候选保留它；88 个 INT8、0 个恢复 FP16 请求，因此没有显式 Half 层精度约束。workspace 1024 MiB、avgTiming=1 保持。新单次预算明确为 7200 秒，原父 1800 秒及此前停止记录保留；原因是旧 q_all 延长构建实际需 3276.426 秒。旧延长构建使用没有显式 precisionConstraints 的 trtexec，不能称为本轮匹配的编译对照。

首轮准备曾因浮点除法的过严静态断言停止，失败另存：22 处 `float32((scale*127)/scale)` 不恰等于 127。全部原 bounds 与 `float32(scale*±127)` 原字节一致，且量化后整数码准确为 ±127。准备器据此修正新静态检查，不改变原完整模型的任何容差。失败记录和逐项值均保留。

| 文件 | SHA256 |
|---|---|
| 候选 ZIP，10,443,020 bytes | `f7b3b24ea613265e6712b4edc265556e760de61645958dbbce159f96070b6de3` |
| model.onnx | `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee` |
| spec.json | `f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb` |
| semantic_rewrite_audit.json | `842eff15adfedd75735815cbe4040d17a645653e3b62511149afacbf8f923be9` |
| manifest.json | `f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f` |

本地 ZIP 的 21 个成员逐字节匹配目录，manifest 19 个 payload 文件 SHA 全部通过。候选包已上传到新板端目录并再核验 manifest，未覆盖旧图或旧引擎。

## CPU 完整原图与改写图：已通过本轮等价检查

使用同一固定 `000020.jpg` 的冻结预处理数组，CPUExecutionProvider、ORT_DISABLE_ALL、四线程、float32；分别执行完整原 ONNX 与改写图。

| 输出 | 形状 | 差异元素 | 原始 float32 字节 |
|---|---|---:|---|
| 主输出 output0 | [1,14,8400] | 0 | 完全相同 |
| PE 辅助输出 | [1,128,20,20] | 0 | 完全相同 |

原图主输出还与最初无 PE tap 的 ORT 保存输出逐字节一致。预处理数组 SHA256 为 `24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94`，本轮原始主辅数组包 SHA256 为 `1fe8d81720679ad47fb6f6d7d47b3019dc2cbd7fccdc72ab6683f8a5d2ce705c`，证据在 `artifacts/gate05/20261019_corrected_qall_cpu_equivalence/`。

此处通过只指图改写在该输入上的 CPU 等价性。原完整 PyTorch/ORT 数值护栏仍为 `FAILED_UNCHANGED`，不得改名或替换。

## 板端准备与当前状态

SSH 已恢复，临时密钥仍限制为电脑 USB 地址且两小时到期，未储存登录密码。板子启动编号仍为 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`；本轮准备时没有遗留任务，风扇 active，jtop active 且 NRestarts=0，可用内存约 5047 MiB。板端时间与客户端分别记录，没有改时钟、升级软件或改变 15W 模式；非管理员 EMC 权限提示保留。

新 native 根目录：`/home/nvidia/rampq/gate05_corrected_qall_20261019/gate05_corrected_qall_candidate_20261019`。独立启动前检查 304 项通过，输入/工具及 CPU 等价回执在板端再次核验后，才启动本轮唯一一次构建。

## Native 构建结果：格式筛选失败，未生成引擎

monitor PID 23569、私有构建组 23574、补充 guard PID 23575。monitor 正常观察到子进程返回码 1，状态为 `BUILD_SCRIPT_FAILED`，用时 13.07 秒；Builder 记录用时 10.5340614 秒。它不是 7200 秒超时，没有按延长预算重跑。

原始错误是：

```text
optimizer.cpp::filterQDQFormats::4421
Assertion !n->candidateRequirements.empty() failed.
All of the candidates were removed, which points to the node being incorrectly marked as an int8 node.
```

随后 `buildSerializedNetwork::751` 返回空引擎，Python 在 `assert serialized is not None` 停止。日志没有给出触发该断言的具体源节点；不能把旧七个深度卷积的严格 tactic 缺口直接当成本次根因，也不能断言已经证明是 obey 或某一处 Clip 造成。

板端解析、校正前/后的 88 份 Clip 属性记录及构建请求已保存；独立静态重审确认校正后的 alpha/beta 全部为 -127/+127。构建请求确为 INT8、FP16、OBEY，关闭 PREFER 和 TF32。记录中的“0 个显式层精度约束”仅指脚本没有调用层精度/输出类型 setter；当时没有查询解析器自身是否带入约束，不能据此断言 OBEY 没有影响。源图审计仍零错误，但这些是解析/输入证据，不是实际引擎精度通过。

优化日志将 57 个 Clip 列为独立 POINTWISE 计划层、31 个列为融合到 Conv 的计划层。尚无引擎，不能把这些日志当成实际执行精度或故障节点定位证据。

没有 `model.engine`、没有 actual precision audit、没有单图执行。现有显式 `--diagnostic-formats-only` 分支在本次不具备使用条件，未启动。目标级恢复机制本轮未被测试，不能把编译失败写成机制不复现。

## 失败证据与板端终态

失败包 `artifacts/gate05/20261019_corrected_qall_native/build_failed_evidence.zip`，10,532,288 bytes，52 份原始文件，SHA256 `027433baa5b1e18fafb72cf64b70e58e1bd2aa05b295fd8f539ae8317e1c9a99`。精确 receipt、ZIP CRC、每个文件字节和 SHA 已验证；receipt SHA256 `a1d7e456d7b005a16db01f199d32535e159799d3b60fa05b4881277c8f47fdd9`。

7 个资源样本的最高温度为 50.343°C，最低可用内存为 4070.605 MiB，交换空间使用未变。监护 reason 为 null，没有记录超时、低内存、过热或 USB 内核事件触发；kernel.log 只有日志起始头。准确结论是已记录的直接失败为 TensorRT 编译错误，这些样本不能排除所有未记录的瞬态硬件问题。

guard 为 `COMPLETE`、无干预、最终私有组为空。最后再次实时核查：三个 PID 全部消失、无残留构建/推理任务、启动编号未变、风扇 active、jtop active 且 NRestarts=0，可用内存约 5046 MiB，无引擎文件。记录在 `artifacts/gate05/20261019_corrected_qall_native/final_board_observation.json`。

根审计 `root_failed_build_audit.json` 重新核验了来源、88 处校正、失败阶段与退出。独立失败审查 252 项通过，报告 `failure_independent_review.json` SHA256 `bae8fad62b264e579b89859497ff53a2b82967fddc0b5aeb02761ea6944f2480`；原始树前后字节未变。所有新审查报告均在原始树之外。

## 下一诊断条件

在单独冻结的 parse-only 元数据检查中，没有创建 Builder 配置，没有调用构建或推理 API，只解析同一图、读取约束并对临时内存中的 88 个 Clip 校正边界。1275 个解析层的查询全部成功：

| 解析器设置的状态 | 数量及来源 |
|---|---|
| precision_is_set=true | 176 个 Dequantize 层为 INT8，另 1 个 Shape 层为 INT32 |
| output_type_is_set=true | 176 个 Quantize 输出为 INT8，另同一 Shape 输出为 INT32 |

176 对应 88 个激活及 88 个权重量化器。校正前后这两类约束各为 177，逐项元数据未变化。由此确实不能把“0 次脚本 setter”理解为解析网络无约束；但不能仅凭这些元数据判定 OBEY 就是失败原因，也不能拿 parsed 类型当实际引擎格式。

补充检查 monitor 用时 9.06 秒，guard 无干预且私有组完全为空。26 份原始文件均已回收核验，ZIP 为 122,747 bytes，SHA256 `a567b474be0e563448ce96b7b5cc0f1e9bfff1036461b9512efe9b8c1ba32d78`；文件在 `artifacts/gate05/20261019_corrected_qall_native/parser_metadata_evidence/`，与原全模型失败包分开保存。

独立补充审查 859 项通过，报告 `parser_metadata_independent_review.json` SHA256 `bb78f349eb8784fe1374f34e777d8948e2eefb1fabd8858f0d18fea1c3ac55aa`。两次任务均已退出；最后实时记录在 `final_board_observation_after_parser.json`。此前成功 q_only_G1 引擎的板端 SHA 仍为 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`，未覆盖或改动。

本协议的单次全模型尝试已经结束。下一步对比原浮点 Clip 和已校正整数码 Clip，并在 0 次手动层精度设置的同一解析上下文中区分 OBEY 与不启用 OBEY/PREFER 的影响。先固定解析 API、输入、权重和观测输出，防止同时改变多个因素。

已成功的融合前缀实际为 11 节点、**一个首 Conv**＋SiLU＋第二激活 Q/DQ 输出，不含第二 Conv。独立 builder 审查建议里的“已成功双 Conv code_clip 前缀”措辞过强，原报告保留，`next_probe_plan.json` 明确更正并重新从原 ONNX 计数。任何双 Conv 扩展都应作为新切片先验证，不能继承旧前缀成功状态。

具体后续切片仍须以本次日志和图边界选定，不能只凭最后一个打印层命名故障节点。`next_probe_plan.json` 是未执行计划，本轮未运行新的 2×2 引擎构建。

小切片若不能复现完整错误，应记为不足以定位；不能因它成功而宣称全模型可构建。若复现，必须保持原数值语义并核验实际执行格式，再评审新的全模型尝试。本轮没有偷偷调整条件重试。

## 仍未证明

没有新的完整数值迁移 PASS、严格全 INT8 PASS、数据集准确率、目标级破坏—恢复或性能结论。新 common_preparation 使用本次改写审计身份，旧 FP16 控制仅作历史来源与描述性比较，非量化浮点和融合对照尚未建立。本轮诊断和失败证据收口完成，q_all native 结果为失败/部署验证未完成；Gate 0.5 仍为 `INCOMPLETE`。

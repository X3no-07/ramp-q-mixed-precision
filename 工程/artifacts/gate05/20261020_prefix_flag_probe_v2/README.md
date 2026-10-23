# 单 Conv、两处激活边界的 2x2 前缀诊断

仅按原字节复制冻结前缀与固定 000020 输入；每图 11 个节点、12 个 initializer，
只有 `/model.0/conv/Conv` 一个 Conv，不包含第二个 Conv。唯一输出仍是
`rampq_model_1_conv_a_dq` FLOAT[1,16,320,320]，无额外中间 tap。

固定顺序 original_none、original_obey、code_clip_none、code_clip_obey。
两种拓扑与 NONE/OBEY_PRECISION_CONSTRAINTS 是仅有的两个实验因素；
code_clip 仅将两个已命名 Clip 的原生解析 alpha/beta 修正为 -127/+127。
全部配置保持 INT8+FP16、禁用 TF32、1024 MiB workspace、avgTiming=1、
DETAILED inspector 与 FLOAT/LINEAR 外部绑定。不设置任何 layer precision 或
output type，不清除或重设解析器约束；修正前后实际解析约束必须完整记录。

每 case 最多 300 秒、一次构建尝试、成功后两次固定输入重复推理；批次监护
1500 秒。编译错误留存后继续下一 case；超时或资源中止则终止整个批次，不重试。
本地准备只读取既存 CPU NPZ 验证两组预期输出原始字节一致，不执行 ORT、
Torch、CUDA、SSH、完整检测器或数据集。运行前须冻结并核验 protocol.json。

原完整 PyTorch/ORT 护栏仍是 FAILED_UNCHANGED，修正 full q_all 先前构建失败
也完整保留。本探针无论成败都不能单独确定 full q_all 的 OBEY 因果，不能作为
完整模型修复、精度/性能、严格整数算术或 Gate 0.5 通过的证据。

# 完整 q_only_G1 code_clip 候选：仅静态准备

model.onnx 来源为 exact 冻结 q_only_G1；6 个原 activation Float Clip→Q
改为 Q→INT8 Clip[-127,127]，随后原 DQ 不动。12 个 Clip/Q 节点和12个边界
initializer 是全部允许改动；新 tensor 名逐个检查唯一，量化 scale/zero、
权重及权重 Q/DQ、bias、88 个 Conv、其余节点和所有模型外部输出声明原字节不变。
原 Float bound 均验证等于 float32(scale * ±127)，zero 均为 INT8 scalar 0。
semantic_rewrite_audit.json 保存逐项改线、常量变化和全部源/新节点/常量 SHA。

spec 保留 precision_candidate_build_spec/v1、q_only_G1、6 INT8 / 82 FP16
原请求和 FLOAT 三绑定合同，prefer / 1024MiB / avg1 / 1800s。
module_map、失败 preflight、diagnostic_scope 原字节复制。源 spec/audit/manifest/
reference_protocol 以 parent_ 前缀保存；后者仍描述旧 Float Clip，不能当新协议。
field_provenance.json 逐个记录 spec/audit 顶层字段的保留、新增、更新及原/新值 SHA。

common_preparation_sha256 改为本次 rewrite audit hash；旧 preparation 身份
只保存在 parent 字段。原 matched FP16 控制尚未为本改图建立，不能沿用旧控制宣称
比较条件齐备。旧 graph_audit 里 QDQ/常量全不变的断言已改为 false并解释变更范围。

native 构建必须在同一个解析后的 network、build之前，把 clip_names 六个命名
Clip activation 的 alpha/beta 显式核对并修正为 -127/+127，记录解析前后证据。
旧 trtexec builder 不实现该校正，本包不复制为可直接运行的新构建器。
原 precision auditor 仅支持 FloatClip→Q→DQ；后续需明确支持 code Clip 拓扑
及 INT8 边界再审计；本准备器没有运行 native precision audit。

本准备器仅 CPU 静态字节差与 ONNX checker，不导入 ORT/PyTorch，不运行完整模型，
不连接设备、不运行 GPU、不改标定/阈值/权重、不做数据集。完整 CPU 数值等价尚未跑，
所有 native 构建/执行/融合精度仍待实测，原完整护栏 FAILED_UNCHANGED。
preprocessed_input.npy 为原 000020 预处理输入原字节副本；generic smoke 用原图像。

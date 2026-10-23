# 完整 corrected q_all 独立 NONE 构建控制

source_bundle/完整复制旧候选目录与所有原始字节，原spec/reference合同中的
OBEY请求不改。外层control_spec/protocol定义这次独立runtime NONE控制，
不把旧文件伪装成NONE或修复其历史失败。原helper/监护/guard/collector保持字节。

cpu_equivalence/仅复制现有原/修正完整ONNX CPU等价证据及rawNPZ；
封包不运行CPU、校准、构建或推理。原PyTorch/ORT guard仍FAILED_UNCHANGED。
本次最多一次7200秒native构建，封包阶段只运行stdlib --check-input-only。
native推理、数值迁移、准确率、性能和Gate通过不在本封包中声明。

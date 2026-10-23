# 冻结首层 Conv+SiLU 最小数值探针

从 q_only_G1 冻结图的实际首个 Conv 与 module_map 精确定位，经输入/输出依赖识别其 SiLU。
QDQ 图仅含原 Clip、activation Q/DQ、weight Q/DQ、Conv、Sigmoid、Mul 共8节点；
FP32 对照从原始 FP32 图截取对应 Conv、Sigmoid、Mul 共3节点。所有被选节点和
weights/bias/量化常量均保持各自来源中的序列化原字节。未重新校准、未改阈值。

唯一输入为 `preprocessed_input.npy`，逐字节复制先前000020固定图输入，
FLOAT[1,3,640,640]，名字 `images`。两个模型同样按顺序输出：
主输出 post_silu `/model.0/act/Mul_output_0`，辅助 pre_silu
`/model.0/conv/Conv_output_0`，均 FLOAT[1,16,320,320]。
请按tensor名字读取，不依赖原生绑定下标。精确接口、原节点/initializer SHA、
CPU期望结果 NPZ 键位在 `spec.json`。

本地只运行两个小图：CPUExecutionProvider + ORT_DISABLE_ALL。
`cpu_expected_outputs.npz` 保存四个原始 float32 数组，差值只作描述。
可选 PyTorch CPU 也只计算两个小 Conv，并分别保存 fused SiLU 和 Sigmoid*Mul；
不加载或执行完整检测器，不使用 CUDA。

暴露 pre-SiLU 输出可能改变原生融合。这只是首层定位实验，不替代完整模型数值
迁移、strict INT8/FP16格式证明、目标恢复或性能结果。原完整数值guard失败仍保留。
未运行150图评估，未建立新的PASS/FAIL门槛。原生构建及执行由独立runner完成。

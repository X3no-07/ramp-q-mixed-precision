# DFL 权重 QDQ 旁路局部对照

同修正 full q_all 源和父6节点图，仅移除 WeightQ/WeightDQ，并将
Conv.input[1] 改接原始 FP32 weight initializer；保留激活 Q/Clip/DQ
及权重原字节，得到4节点5使用中initializer。未添加 output QDQ 或新尺度。
这改变权重量化数学语义，并非与旧 W8A8 等价的修复。

复用原概率 NPY，不生成概率/不跑 Softmax、72Conv、旧 DFL 或完整检测器。
仅运行新4节点CPU一次，有限 FLOAT[1,1,4,8400] 结果独立保存为
weight_bypass_output 参考。旧6节点CPU仅描述比较，不要求字节相等。
失败保留，禁止重选/重试/seal/native。输出为DFL距离，不是解码框。

native NONE/OBEY 各构建一次300秒，batch900秒，成功各执行2次；
仅 Clip 导入属性校正为±127，不设置/重置任何层精度或输出请求，
不制造先前QDQ请求数量。源FP32权重不证明实际FP32计算，实际格式/tactic另审。
原full失败与Gate INCOMPLETE保留；本局部结果不构成完整图修复或唯一根因。

# DFL 在 Softmax 后精确裁切的局部对照

仅从修正q_all抽取原Softmax单节点、零initializer图，以及其后的8节点、
7initializer DFL切片，所有所选节点、常量和I/O声明保留源序列化字节。
原source Softmax axis=1，从已认证旧logits执行一次CPU生成概率fixture；
8节点cut再CPU执行一次，输出必须与保存的旧DFL CPU原始结果字节完全一致。
严格总两次CPU，不重跑72Conv上游、旧9节点DFL或完整检测器，不重选或重试。

native输入是FLOAT[1,16,4,8400]概率，不是logits或images。输出为
FLOAT[1,4,8400] DFL distance，不是检测框。旧expected NPZ原字节复用。
code_clip none/obey各一次300秒构建、batch900秒，成功才重复执行两次。
仅校正原命名Clip导入alpha/beta到-127/+127，不设置或重置解析精度约束；
卷积输出允许FP32，实际格式与tactic必须分开记录。

切口同时改变输入边界与native融合上下文，并使用CPU生成概率，所以结果
变化不能认定Softmax为完整图唯一原因。原完整护栏FAILED_UNCHANGED、
旧original/corrected的OBEY失败与NONE数值差异保留，Gate0.5仍未通过。

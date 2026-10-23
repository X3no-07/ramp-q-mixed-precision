# 原始 FLOAT Clip DFL 局部对照

原样复用上一DFL阶段的original tail、内部logits fixture、CPU NPZ与两份原始
CPU NPY。只静态读取验证字节，不运行新的CPU推理、上游或fixture生成。
original tail为原始完整q_all的精确9节点7initializer依赖闭包，输入为原
DFL Transpose FLOAT[1,16,4,8400]，输出为DFL distance FLOAT[1,4,8400]，
不是解码检测框；保留FLOAT Clip→Q→DQ，不校正或修改原生Clip属性。

fixture继承上一阶段修正ONNX的72Conv上游CPU一次输出，不是原始/修正上游
等价或native分布证明。原始与修正tail在该fixture上的旧CPU输出字节相同。
本轮仅original_none/original_obey各一次有界native构建，300秒/case、900秒
batch，成功才固定输入重复执行两次。保留解析器约束，不设置或重置精度、
输出类型，不要求Conv输出INT8，失败无引擎不得执行。

旧corrected DFL NONE成功/OBEY失败记录仅用于跨阶段对照，不重跑旧case。
拓扑与导入修正属于组合因素，不能拆成单独原因；局部结果不证明完整图
唯一根因、取消OBEY能修完整图或Gate通过。原完整护栏FAILED_UNCHANGED。

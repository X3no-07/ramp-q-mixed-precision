# DFL Softmax/量化/卷积/形状转换上下文

从实际失败的修正 q_all 源图提取 images 到 DFL Transpose logits 的精确祖先图，
共638节点、576 initializer、72Conv。使用原固定图片NPY，仅在CPU执行一次，
四线程、ORT_DISABLE_ALL，生成并冻结 FLOAT[1,16,4,8400] 内部fixture。
这是大段backbone/回归头的CPU诊断数据，不是native数据或新的Gate参考。

原始q_all与修正q_all的DFL tail各9节点、7 initializer；所有所选节点与常量
保留各自源图序列化字节。两tail共用上述fixture，各CPU执行一次，必须输出
FLOAT[1,4,8400]字节完全一致；这是DFL distance，不是已解码目标框。
失败不得重选fixture或重跑。没有执行完整检测器输出或数据集。

native只构建code_clip tail，none/obey各一次、300秒，batch900秒，成功后
重复执行两次。只校正一个Clip的原生导入alpha/beta为-127/+127，不设置或
清除解析器精度约束。DFL末端没有输出QDQ，不要求实际卷积输出INT8。
原完整护栏FAILED_UNCHANGED和完整修正q_all构建失败保留，本局部上下文
不能确定完整失败节点、证明完整修复或Gate 0.5通过。

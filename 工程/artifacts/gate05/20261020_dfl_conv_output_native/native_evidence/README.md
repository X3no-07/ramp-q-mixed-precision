# DFL 六节点 Conv 输出边界对照

从同一修正q_all源图提取exact6节点7initializer：Activation Q/Clip/DQ、
Weight Q/DQ和源Conv，截止原Conv_output_0 FLOAT[1,1,4,8400]。只删除父
8节点图末Constant/Reshape，所选节点、常量与源I/O声明保持原字节。
概率fixture按原NPY字节复制，不生成概率、不跑Softmax/72Conv/旧DFL。

仅执行新6节点图CPU一次；输出C顺序float32原始数组字节必须与旧
FLOAT[1,4,8400] CPU距离完全相同。旧数组纯reshape为4D生成新expectedNPZ，
保存各自shape/hash；NPY头不同不算数值差异。CPU失败保留，不重选或重试，
不启动native。输出是带单通道维的DFL distance，不是解码检测框。

native code_clip none/obey各一次300秒，batch900秒，成功后重复执行两次。
只校正原Clip导入alpha/beta为-127/+127，不设置或重置解析器精度/输出请求。
实际Conv输出允许FP32，格式与tactic必须单独核验。新输出切口改变native
边界，结果变化不能把Reshape定为完整图唯一原因。原失败和Gate未通过保留。

# 首Conv恒等Clip配对干预

clip版为上一首Conv QDQ微图的完整文件原字节副本；no_clip仅删除首Clip和两个未用
边界常量，把原ActivationQ的input[0]改为images。其余Conv、SiLU、weight/bias、
scale/zero、输出声明和顺序完全不变。反向补回三项改动可恢复整个原ModelProto字节。
所有具体改动、原/新Q节点和保留节点/常量SHA均记录在model_change_audit.json。

原固定000020输入npy逐字节复制，FLOAT images[1,3,640,640]。
原Clip上下界严格为[-1,+1]，该输入属于[0,1]，Clip前后原字节相同。
主post-SiLU和辅助pre-SiLU输出保持原名字，均FLOAT[1,16,320,320]。
CPUExecutionProvider、ORT_DISABLE_ALL各执行一个小图，先保存原始数组，再核对
两图两个输出逐元素bitexact。CPU expected NPZ键和接口在spec.json。

只为验证同native选项下去除恒等Clip是否改变这一孤立首Conv的数值，不能表述为
完整模型修复、数值迁移通过、strict算术证明或准确率/性能结果。没有重校准、
更改原标准、运行完整检测器、CUDA、SSH或150图评估。原完整数值失败仍保留。

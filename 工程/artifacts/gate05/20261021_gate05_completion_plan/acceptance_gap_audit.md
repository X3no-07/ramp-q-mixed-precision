# Gate0.5 验收缺口与闭环审计

当前仍为 INCOMPLETE。应停止把每个局部诊断收口当成 Gate 完成；闭环目标是完整可比部署、预先冻结的逐目标评估和可审计终态。科学结果可以是 PASS、FAIL 或 INCONCLUSIVE，完成任务不保证机制复现。

## 原始最小验收

原23要求 FP16、q_all、G1恢复、G1+G5恢复、G1+G2恢复五个完整 engine，同一固定集逐目标检查新增漏检、同目标救回和附带漏检，并另查 FP32→FP16。q_only:G1 只有在声称实机双向归因时才是必需的第六个配置。数据规则须在数据集输出前冻结。[23_Gate05_Orin_FP16首轮检查.md:78](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/23_Gate05_Orin_FP16首轮检查.md:78)；[研究状态与后续计划.md:167](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/研究状态与后续计划.md:167)。

原24协议的 build_pass 是88源Conv映射到真实INT8加权内核，execution_pass 是固定图真实执行且有限；明确不要求INT8检测数等于FP16。它没有定义数据集通过门槛。README9.11只是证据目录，不能把文件齐全当验收。[artifacts/gate05/20261018_int8/input_v1/protocol.json:24](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261018_int8/input_v1/protocol.json:24)；[artifacts/gate05/20261018_int8/input_v1/protocol.json:25](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261018_int8/input_v1/protocol.json:25)；[artifacts/gate05/20261018_int8/input_v1/protocol.json:26](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261018_int8/input_v1/protocol.json:26)；[README.md:383](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/README.md:383)。

## 当前真正缺口

1. 旧五个量化配置虽能运行，固定图均零检测，不能作为已修复的可信基线。只有另存 corrected q_only_G1 已恢复固定图且6INT8/82FP16完整。修正全q_all仍无engine，三个修正恢复图尚未完成。
2. 没有为新 corrected preparation 建立完整匹配的浮点/融合控制。旧FP16可复用历史来源，不能无条件宣称新家族控制已匹配。
3. 原PyTorch/ORT完整护栏失败仍保留；当前reference_contract只允许一次单图诊断，明示not_a_gate_amendment，不能自动覆盖150图验收。
4. q_all及三恢复各7层只有INT8格式和运行内核身份，缺严格整数MAC证据。未运行固定150图、未生成逐目标迁移结果，也未冻PASS/FAIL/INCONCLUSIVE。

证据：[26_Gate05_当前可执行工作收口.md:7](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/26_Gate05_当前可执行工作收口.md:7)；[artifacts/gate05/20261019_corrected_qonly_native/checkpoint.json:1](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261019_corrected_qonly_native/checkpoint.json:1)；[30_Gate05_全INT8修复候选验证.md:54](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/30_Gate05_全INT8修复候选验证.md:54)；[artifacts/gate05/20261019_reference_contract/reference_contract.json:110](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261019_reference_contract/reference_contract.json:110)；[artifacts/gate05/20261019_reference_contract/reference_contract.json:115](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261019_reference_contract/reference_contract.json:115)。

## 哪些属于附加诊断条件

“命名TacticName必须含int8/imma”等是本项目保守证据代理，不是整数计算本身的定义。不能跳过七层，但可另冻直接机器码证据通道，保留旧审计状态，不伪造TacticName。[scripts/audit_gate05_int8_precision.py:128](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/scripts/audit_gate05_int8_precision.py:128)；[artifacts/gate05/20261019_reference_contract/reference_contract.json:93](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261019_reference_contract/reference_contract.json:93)。

OBEY、每轮小图CPU字节等价及300/900秒预算是相应冻结实验的条件，不是原Gate科学要求。新独立构建可仅改变请求方式而保留真实逐层审计，不能把旧失败改判。所有Conv输出INT8、所有算子INT8、必须TensorCore、FPS/能耗优势也不是Gate0.5要求。原图没有新增Conv输出量化；性能在Gate2。[24_Gate05_Orin_INT8构建与迁移检查.md:7](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/24_Gate05_Orin_INT8构建与迁移检查.md:7)；[研究状态与后续计划.md:202](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/研究状态与后续计划.md:202)。

唯一全图错误节点定位也不是Gate原要求。修复方案若能通过来源/语义/完整实际精度/独立参考/目标级验收，可以诚实保留根因未穷尽；局部成功自身仍不算完整部署证据。

## 原护栏失败与独立参考可以并存

26已明确把“另设可对齐参考实现及独立目标级迁移验收”列为开放决策。29在固定图上证实微小Conv差跨舍入边界，hybrid收敛只是反事实，不能当独立参考。[26_Gate05_当前可执行工作收口.md:198](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/26_Gate05_当前可执行工作收口.md:198)；[29_Gate05_完整参考数值分歧定位.md:28](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/29_Gate05_完整参考数值分歧定位.md:28)；[29_Gate05_完整参考数值分歧定位.md:49](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/29_Gate05_完整参考数值分歧定位.md:49)。

正当做法是新冻部署参考协议：原guard路径/哈希/rtol/atol/FAILED_UNCHANGED全部不动；用无隐藏张量注入的完整ONNX CPU独立运行每个实际配置，权重/尺度/算子语义静态验证与跨后端数值/目标一致性分开。配置数学语义若变，例如旁路权重QDQ，必须用该候选自己的CPU参考，不能称q_all等价修复。native比较必须与同族FP16和浮点/融合控制相配；不能为覆盖已见96.57框误差而补设大容差。新协议不保证旧护栏会变PASS，也不把旧Gate0模拟参考换名。

## 七层整数MAC证据的最短路径

已有q_all7层及恢复21层的精确NVTX/inspectorJSON/同线程CUDA API/correlationId→完整mangled symbol映射，可复用而不重跑旧推理。先只读检查实装工具/库/buildID/哈希和SM，使用cuobjdump list-elf、dump-elf-symbols，再对匹配symbol dump-sass；不要假定engine.plan就是ELF。若库中无可对应cubin，记录inconclusive，才考虑新的有界module/cubin捕获。

必须把该SM/库或cubin身份/符号与运行kernel对应，追踪激活与权重的整数加载到signed8 IDP4A/IDP或IMMA/INT32累加主循环，区分FP32缩放/输出epilogue。地址计算IMAD、函数名、TacticValue或仅出现一条未执行分支的整数指令均不够。静态可达整数路径与已执行路径分开记录；若控制流不能判定，才提最小指令剖析。新的补充报告引用原audit，不覆盖它。新engine需要自己的身份链。完整七层列表和符号在JSON中。[artifacts/gate05/20261019_qall_kernel_diagnostic/analysis.json:65](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261019_qall_kernel_diagnostic/analysis.json:65)；[artifacts/gate05/20261019_restore_saved_execution_diagnostic/analysis.json:13](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261019_restore_saved_execution_diagnostic/analysis.json:13)。

官方工具说明：[CUDA11.4.4 Binary Utilities](https://docs.nvidia.com/cuda/archive/11.4.4/cuda-binary-utilities/index.html)。其支持按函数导出SASS，并区分IDP4A与IMMA整数运算；实际工具版本/架构支持仍须核实。TRT近邻归档[8.6.1开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html)解释QDQ显式表示；8.6文档不是8.5.2.2行为保证。跨后端不普遍保证逐位相同的背景见[NVIDIA量化说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)，不能据此抹去本项目原容差失败。

## 建议闭环顺序

先冻验收/参考/浮点控制和终态规则；并行做保存内核的静态SASS证据与一个有界完整corrected q_all构建控制，优先评审仅关闭OBEY/PREFER、保留QDQ parser请求/Clip校正的新全图尝试，实际精度仍严格审。它不是已经证明的全图修复。若完整数值和格式可信，再扩三恢复及匹配FP16，按声明需要复用q_only_G1，然后跑固定150图/865目标，生成逐目标CSV和终态报告。不要无限追加局部矩阵，也不能凭旁路权重的小图替代全量化模型。

数据事件沿用同类一对一匹配：F为本机FP16命中，Q为q_all命中，新增漏检F−Q，G1救回(F−Q)∩R，附带漏检(F∩Q)−R；双向另交q_only_G1漏检目标ID。原始协议没有Gate0.5净收益/稳定保持率的数字门槛，若PASS要求这些实用效果，必须先明确预注册，不能伪称继承旧规则。非空新增漏检且至少同目标救回是存在性复现的逻辑必要条件，不足以证明稳健收益或方法优越。

技术条件满足但预定复现标准失败，应报告FAIL；缺目标分母/事件太少或精度参考不成立，按预注册报INCONCLUSIVE。二者均可形成如实的阶段收口，不能写PASS。旧Holdout只作部署诊断，不能重新包装新方法独立确认集。正式FPS/功耗不在本轮验收范围。

本审计只读取文档和保存JSON；没有CPU/板端推理、SSH、改旧判据或入口文件。结构化完整证据、逐项源行和后续产物要求见[acceptance_gap_audit.json](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate05/20261021_gate05_completion_plan/acceptance_gap_audit.json)。

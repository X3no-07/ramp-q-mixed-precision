# Gate1 原始 FP32/FP16 超差独立核查

结论：冻结的 109 张 debug 图原始输出比较仍为 `FAIL_FIXED_TOLERANCES`。这次只读核查没有发现两个分支使用不同图像、不同融合方式，或 NMS 改写原始输出等可见流程错误。现有证据支持继续做分层数值诊断，尚不能把误差归因于 DFL、卷积、BN、输入舍入或某一个检测头操作。

本目录没有运行模型、读取图像或标注正文、使用 GPU，也没有修改原始门槛、原始输出或旧 Gate0.5 结论。检查的 109 个 NPZ、109 个 per_image JSON 和 5 个元数据文件，合计 223 个文件，均与原 qualification manifest 一致且读后未变。另核对了 5 个框架源码文件的记录 SHA。可复算结果在 [raw_array_diagnostics.json](raw_array_diagnostics.json)，独立 CPU 检查程序在 [inspect_saved_raw.py](inspect_saved_raw.py)。程序采用独占创建结果文件，复算时应使用新输出目录，保留本次结果。

## 可直接核验的流程观察

| 核查项 | 冻结源码/记录中的观察 | 证据范围 |
|---|---|---|
| 权重来源 | 两个分支分别 CPU 加载同一份 SHA 已验证的 `best.pt`，本次均取 `model`；保存状态浮点项为 FP16 | `runner_source.py:343-378`、`model_provenance.json`；没有在本次重新反序列化权重 |
| 有效 dtype | 同一保存 FP16 EMA 先 `.float().eval()`，再分别转 CUDA FP32/FP16；原保存 FP16 数值可精确提升至 FP32 | FP32 分支不是未损失的原始训练 FP32 EMA |
| 融合 | 两分支均保留 88 Conv2d、81 BatchNorm2d，75 个 Ultralytics Conv 包装器均保留 BN；没有调用 fuse | 这是相同未融合结构的 dtype 比较，不是 TensorRT 或 Orin 比较 |
| 输入 | 每图只解码/LetterBox 一次；同一个 RGB/CHW uint8 数组先以 FP32 除以 255，FP16 分支使用该张量 `.half()` | `runner_source.py:457-475`；输入 dtype 舍入仍是实际差异来源之一；本次没有读取图像重算像素 |
| raw 与 NMS | 先拷贝并保存 decoded raw，再比较；NMS 只接收独立 CPU FP32 clone | `runner_source.py:469-496`；NMS、原坐标还原和 GT 匹配不可能回写已保存的 raw NPZ |
| 网格/缓存 | Detect 依第一层特征形状决定是否重建网格；模型 `_apply` 同时转换 stride、anchors、strides | `head.py:207-217`、`tasks.py:289-306`；没有记录运行时 cache 具体值，故不能把缓存内容的全面正确性当作本次独立实测结果 |
| 控制 | 显式关闭 autocast、TF32、cuDNN benchmark，并启用 deterministic 算法 | `environment.json` 与 `runner_source.py:297-325`；确定性不保证不同 dtype 数值相等，也未在本次额外重跑复现 |

输入像素路径和融合结构没有可见分支分叉。dtype 对输入、参数/缓存的类型及算子执行产生有意差异；不能把这些有意差异等同于发现了实现 bug，也不能据静态审查声称排除了所有潜在 bug。

## Detect / DFL 数值路径

本次实际框架版本为 Ultralytics 8.4.142，源码 SHA 由原 `environment.json` 记录并在本次核对。路径如下：

1. `head.py:169-217`：各尺度 box head 输出按尺度拼接；DFL 后用 `dist2bbox` 解码，再乘各尺度 stride；类别 logits 走 sigmoid，最终输出 `[1,14,8400]`，前四通道是 640 输入坐标中的 xywh，后十通道是分类分数。
2. `block.py:68-81`：DFL 将回归 logits 按 4 个方向、16 个 bin 重排，在 bin 维做 softmax，再经固定 `[0,...,15]` 卷积计算期望。这里没有显式 `.float()` 数值提升。
3. `tal.py:424-439`：anchor 点使用第一层特征的 dtype 构建。640 输入下 8/16/32 stride 对应 80×80、40×40、20×20，共 8400 点。网格半整数、stride 8/16/32、DFL 整数 bin 本身可由 FP16 精确表示；这不保证其上运算结果精确。
4. `tal.py:442-452`：先算 anchor−lt、anchor+rb，再算两端均值和差，最后在 head 里乘 stride。网格单位的差异会按 stride 转成像素单位，端点相减还可能影响小宽高的相对误差。
5. `conv.py:69-78`：未融合 Conv 的路径是 Conv→BN→activation。主干、neck、head、DFL softmax/积分以及几何解码都可能贡献 dtype 差异；源码未提供把本次误差唯一归因给其中一处的证据。GPU 内部累加精度不能仅由输入/输出 dtype 推定。

NPZ 只保存两个 decoded 输出。虽然模型 forward 同时返回 head logits，runner 没有保存这些中间量；因此无法事后恢复真实的 FP32/FP16 logits、DFL 概率、DFL 距离和各算子中间值。

## 三个相对固定容差最严重的元素

排序指标为 `abs(FP16-FP32)/(atol+rtol*abs(FP32))`，完全沿用原始非对称门槛；坐标差以 640 输入像素计，anchor 索引从 0 开始。

| 图像（均为 source train/debug） | anchor / stride / 通道 | FP32 → FP16 | 绝对差 | 原固定容差 | 差/容差 | 两分支最大类别分数 |
|---|---|---:|---:|---:|---:|---:|
| `0000170_01201_d_0000001` | 8153 / 32 / h | 99.037109375 → 89.375 | 9.662109375 | 0.995185546875 | 9.708852 | 0.000326084 / 0.000324965 |
| `0000170_01201_d_0000001` | 8194 / 32 / h | 49.380584717 → 43.75 | 5.630584717 | 0.746902923584 | 7.538576 | 0.000338677 / 0.000340700 |
| `9999990_00000_d_0000024` | 8045 / 32 / w | 63.940505981 → 58.0 | 5.940505981 | 0.819702529907 | 7.247149 | 0.000277668 / 0.000280142 |

按绝对差排序的全局最大元素是 `9999990_00000_d_0000013`，anchor 8195、h 通道，193.939514160→183.5，差 10.439514160 像素；完整前三名也保存在 JSON。上述个例低于原 NMS 候选门槛，不代表其余超差点全部如此，更不能因此删除背景点把全元素 guard 改判 PASS。分层比例和 NMS/TP 统计由父任务的独立汇总单列报告。

## 最终 FP16 保存舍入这一单一解释不够

另做了一个纯 CPU 算术反事实：把已保存的 FP32 decoded raw 转成 NumPy FP16 再转 float64，直接使用原固定容差检查。109/109 图均没有 box 或 score 超差；最大 box 绝对差 0.25 像素，最大 score 绝对差 0.000244140625，最大差/容差 0.0816261264。

这说明实际差异不能由“最后把一个相同 FP32 decoded 结果保存为 FP16”这一单步舍入独自解释。它没有模拟真实 FP16 forward、输入 cast、DFL 或 decode，不能用作替代模型、替代 guard 或解码因果证明。

## 下一步最小独立诊断协议（仅提案，未执行）

目的：区分 upstream logits 差异与 DFL/解码 dtype 差异，保留现有失败结果。实验必须新建目录并在运行前冻结记录；不改变原 guard，不访问 Selection/Holdout。

1. 固定同一 checkpoint SHA、未融合结构、现有预处理和 runtime 控制。先只用本报告三个最坏元素涉及的两张 debug 图，并保留一个普通 debug 对照图；这些图是事后选取的诊断样例，不能作为新验收集或泛化证据。若准备据结果形成工程结论，再以原 109 张 debug 图验证完整范围。
2. 直接保存 FP32 与 FP16 分支各自 box logits `[1,64,8400]`、class logits `[1,10,8400]`、实际 anchors/strides、shape、输入 FP32 与 FP16 bit-pattern SHA，以及参数/浮点 buffer 值提升后的相等性记录。保留原 forward decoded 输出，核对重建的原生解码与原输出一致；记录具体误差而非默认一致。
3. 对两个冻结来源的 box logits 分别运行 FP32 解码与 FP16 解码，形成 2×2 对照。每条路径记录 logits、softmax 概率、DFL 四方向期望、网格端点、xywh、乘 stride 后像素框的 dtype/值。若使用同一来源 logits，FP16 路径仍包含该来源 logits 的 half cast，需要明确列出该步骤。
4. 以相同来源 logits、相同 anchors/strides的两种解码比较，描述 decode dtype 的干预效应；以两来源 logits 均提升至 FP32 并经同一 FP32 解码比较，描述进入解码之前已经存在的差异。class logits/sigmoid单列。不能把两种差异直接作线性可加的误差分解。
5. 若 FP32 解码干预仍不能解释差异，再冻结 head 输入特征并比较 head 或主干层级；不在本轮先启动大范围层搜索或重新训练。后续若研究 FP32 Q/DQ 机制，必须显式采用独立 FP32 研究契约，并保留“FP16资格检查失败、不是已验收部署基线”的状态。

本协议不保证某项干预会使 guard 通过。即使局部诊断或完整新对照通过，也不能覆盖本次 `FAIL_FIXED_TOLERANCES`、旧 Gate0.5 的 `FAILED_UNCHANGED`，不能据此声称 Gate1 科学验收通过、官方 VisDrone AP 等价，或 Orin/TensorRT 已验证。

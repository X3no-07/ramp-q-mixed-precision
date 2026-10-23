# Gate1 忽略区域负样本损失：缺口与最小原语

本包确认了普通训练管线的缺口，并提供独立的分类损失原语与纯 CPU 合成检验。它尚未接入数据增强、Dataset、Trainer 或真实训练；不能据此称“完整 ignore-aware 训练已解决”。旧候选、旧训练合同及全部科学门槛保持原状。

当前安装的 Ultralytics `utils/loss.py` 第 435–441 行，对全部 anchor、全部类别的分类 BCE 求和；`utils/tal.py` 第 300–307 行把未指派 anchor 的目标分数置零。目标为零仍会产生负样本损失：logit 为 0 时，每项损失为 ln(2)，对 logit 的梯度为 +0.5。只从 YOLO 正标签中删掉 ignored 项，并不能消除这项惩罚。冻结训练合同已经明确记录这一限制。

来源需区分历史固定与当前观察：旧合同分别固定了 Dataset、Augment、Tasks 等来源，而没有单独固定 `utils/loss.py`、`utils/tal.py`。本包快照与 SHA 固定的是这两个文件当前安装的字节，不能反推它们当时已经逐字冻结。`source_gap_review.json` 同时保存匹配检查、具体行号和这一限制。

原语接收最终增强图像画布内、已经裁切好的像素 XYXY ignore 矩形，以及 TAL 已产生的正 anchor 掩码。判断规则是 `x1 <= x < x2`、`y1 <= y < y2`；落在任一 ignore 内的未指派 anchor，全部类别的 BCE 置零。已指派 positive anchor 整行保留，包括该行其它类别的负项。分母仍为原 `max(target_scores.sum(), 1)`，不按 ignore 面积或剩余 anchor 数重归一化；class gain 与 batch 乘数留给调用者。输入只支持 FP32/FP64，低精度显式拒绝；非有限输入、非法几何、加权或求和溢出均拒绝。原语不修改输入。

这是新工程政策，不是官方 VisDrone 训练或 AP 定义。特别是：以 anchor 中心判断、半开边界，以及对所有 publisher-ignore 区域采用 class-agnostic 屏蔽，均是需要预先冻结、另做验证的选择。官方评测中 score=0 同类吸收、忽略区域覆盖率等语义，不能直接等同于这里的训练掩码。未来来源连接也应区分 `publisher_ignored` 与制备时非法正框的质量排除，不能把两者静默合并。当前没有读取任何真实 ignored 行来执行这种选择。

合成反例用 3 个 anchor、2 个类别：第一个是 ignore 内负样本，第二个是同一区域内的 positive，第三个是区域外负样本。普通损失为 6 ln(2)，新原语为 4 ln(2)；只清零第一个 anchor 的两个梯度，第二、三个 anchor 的全部梯度逐位相同。无 ignore 时与普通 BCE 的值和梯度逐位一致；全图均 ignore 且无 positive 时，分母为 1，损失和梯度为 0。测试还覆盖批内不同数量区域、并集、边界、正负重叠、class weights、非法输入与数值溢出。最初 44 项测试记录保留；最终硬化后的结果单独保存。

这里的“其余梯度不变”只针对同一次、固定 logits 与 assignment 的分类 logit 梯度。删除负项会改变共享网络参数的总梯度，后续训练轨迹可以改变；没有证明训练精度提升、收敛稳定或官方 AP 一致性。

真实训练连接仍需完成以下工作：

1. 给 train 样本携带独立 ignore 几何及来源身份，不把 ignore 加入 TAL 正 GT。
2. 用图像同一次采样的 resize、pad、crop、Mosaic、flip 参数变换 ignore，不能另采一份随机变换。Mosaic 当前重建字典时会丢任意 sidecar 字段；必须显式合并各来源几何。
3. ignore 仅丢弃最终画布外和空的片段；不能套用正框的面积保留率、长宽比过滤，把残留 ignore 悄悄变回负样本。若允许旋转/透视，需要 polygon 或已冻结的外包框政策，后者可能屏蔽额外背景。
4. 显式验证或拒绝 MixUp、CutMix、CopyPaste、空间 Albumentations 与多尺度训练；旧默认中前三者为 0、旋转/剪切/透视为 0，但不能把配置值当作新管线已经经过验证。
5. 在 Format、拼批和 Trainer 预处理后保留 ragged 区域及准确 batch 身份，再把最终像素坐标传给本原语。先用合成增强链验算几何和正标签不干扰，再在新合同中考虑真实训练。

全部结果只来自源码、合同元信息及新造张量。没有读取真实图像、YOLO 标签或 ignored sidecar 行，没有载入权重、使用 GPU、训练模型，没有接触 Selection 或 Holdout，也没有修改原 runner 或安装环境。

复现使用原 05 工程的现有虚拟环境：

```powershell
$env:PYTHONPATH=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\lenovo\Desktop\视觉边缘研究\05_水下小目标检测_工程\检测工程\.venv\Scripts\python.exe' -B -m pytest -q tests/test_gate1_ignore_negative_loss.py -p no:cacheprovider
```

`build_source_review.py` 只对尚不存在的结果目录生成来源快照与反例记录，不覆盖已固定证据。它不运行真实数据或网络。

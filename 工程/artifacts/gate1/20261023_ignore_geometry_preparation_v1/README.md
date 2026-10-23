# Gate1 ignore 几何与拼批前置：合成连接，尚未接真实训练

新增原语完成了独立 sidecar 的来源校验、显式 loader resize、共享 LetterBox 参数、共享翻转参数及 ragged 拼批，并将拼批结果交给已有分类损失原语做合成检查。85 项 CPU 合成测试通过。没有读取真实图像、标签或 sidecar 行，没有载入权重、训练或使用 GPU。普通 Dataset、Format、Trainer 与 criterion 的真实数据流仍未接入；不能称完整 ignore-aware 训练已经完成。

最小建议是另立一份训练合同：自建 `original → loader 实际 resize → LetterBox → flip` 链。禁用 Mosaic、MixUp、CutMix、CopyPaste、随机透视、旋转、剪切、平移、随机 scale/crop、空间 Albumentations 和多尺度训练。这里允许的是实际 loader resize 与 LetterBox 的确定性缩放；没有实现随机 scale/crop。仅把普通 `v8_transforms` 内各概率调为 0，不等于这里已经验证的显式链。这个改变会更改增强和损失政策，不能直接替代旧 30 epoch 训练，或据此重新解释旧结果。

源码发现 loader 对长边缩放时，会分别对实际宽高做 ceil。原语必须使用实际 `(resized_h, resized_w)` 相对原尺寸的两轴比例，不能把长边比例直接套给两轴。随后从图像的同一次 `LetterBox.get_params` 取 ratio 与整数 pad；翻转同样只接受图像已采样的参数，不另抽随机数。合成测试直接交叉调用当前安装的 LetterBox/RandomFlip；另 3 项对照调用实际 BaseDataset.load_image，将 `imread` 替换成纯内存造图，再按普通正框的 normalized Instances 路径做 LetterBox 和双翻转，与 ignore 的像素坐标链比较。该对照证明所测几何算法一致，不是实际数据增强连接或训练效果证据。

`score_0`、`category_0`、`category_11` 的来源原因、原 score/category、物理行号与 target 身份分别保留。原语验证原 XYWH、XYXY 与按源画布裁切的 XYXY 一致；保留任意非空的微小裁切残片，不套用正框的面积/长宽比过滤。publisher 原始宽高非正或裁切后完全为空时，整个样本显式拒绝；质量排除正框也拒绝，并要求另立 quarantine 政策，不静默变成 publisher-ignore。这意味着未来必须另审 quarantine/数据合同，不能声称现有 prepared train 的 4534 张图全部能够直接接入；本轮没有检查这些样本是否会被拒绝。

`train/` 前缀和两个 SHA256 字段仅用于身份结构与批次顺序校验。它们不是实际文件哈希核验，也不是 split membership 证明。未来 loader 仍须绑定已冻结训练 split manifest、实际核验图像和 sidecar 字节，并只加载被授权的 train 角色。现有原语不读取任何文件，也没有选择或筛除真实样本。

拼批保留每图 `[N,4]` FP32/FP64 tensor（空图必须为 `[0,4]`），逐项核对预期的图像/sidecar 身份和最终画布。坐标为像素 XYXY，半开边界与已有 loss 一致；TAL positive anchor 的所有类别保持优先，由已有 loss 原语处理。source SHA 并未做深度防伪，调用者必须提供正确身份；训练数据流中任何重排、缓存和 transfer 都仍需实际接通验证。

初次测试因新测试文件的语法错误在收集阶段停止，保留 `initial_tests.*`。第二次 65 项通过、1 项测试失败：为验证 FP32 cast 下溢而造的 `1e-50` 区域，先在 FP64 LetterBox 加 padding 时已变空，原语正确拒绝；因此改成无 pad 的画布，分别观察 cast 时的显式拒绝。之后 82 项通过，再增加 3 项 installed loader 合成对照后最终 85 项通过。未放宽非空几何要求，初次与第二次记录保留，`third_tests.*` 是 82 项中间记录。

来源见 `source_review.json`：本包固定当前 data/base.py、data/augment.py、data/dataset.py 与 utils/instance.py 快照。augment/dataset 当前 SHA 与旧训练合同相同；base/instance 未在该合同单独固定，只能称当前来源，不回溯声称旧训练已固定这两个文件。原分类损失原语 SHA 必须保持不变。

这些训练屏蔽政策不是官方 VisDrone AP 定义。没有训练有效性、官方指标、科学 Gate1 或板端部署通过的结论。接下来仍需显式 Dataset adapter、Format 包装/sidecar 保存、同批身份绑定、Trainer 预处理与 loss adapter，并先做整条纯合成 forward/backward，再用新的真实数据和训练合同开展训练。

复现（已有 05 虚拟环境，不安装或修改环境）：

```powershell
$env:PYTHONPATH=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\lenovo\Desktop\视觉边缘研究\05_水下小目标检测_工程\检测工程\.venv\Scripts\python.exe' -B -m pytest -q tests/test_gate1_ignore_geometry.py -p no:cacheprovider
```

# Gate1 浮点资格检查：源码核查与待执行合同

状态：**仅完成本地源码和元数据核查；未运行模型，未执行资格检查。** 后台 30 轮训练不受此工作影响。

本次交付包含 [待执行合同](qualification_contract_prepared.json)、[带 SHA 的源码核查](source_review.json) 和 [109 张 debug 元数据清单](debug_metadata_inventory.json)。合同尚待最终权重 SHA、资格执行脚本 SHA 和训练完成回执绑定。它不能作为浮点基线验收结果。

源码来自当前本机 `ultralytics 8.4.142`，18 个文件的逐字副本保存在 `framework_sources/`，每个文件的原绝对路径、SHA256 和行数均列于 `source_review.json`。未导入检测框架、未读图像或标注正文，未读取 Selection/testdev 图像或标注。

## 已确认的关键规则

| 规则 | 已固定的处理 | 本地源码位置（1 起始行号） |
|---|---|---|
| 保存权重不是原始 FP32 EMA | 框架把 EMA 转为 FP16 后保存；资格检查的 FP32 是该保存权重转回 float 后的计算，无法恢复训练时已经舍入掉的值。 | `engine/trainer.py:720-758`；`utils/torch_utils.py:806-829` |
| 加载最终权重 | EMA 优先，最终去优化器后可能只有 model；最终 checkpoint 的 epoch 可为 -1，选择轮次须从最终回执和训练记录证明。 | `nn/tasks.py:1938-1982`；`utils/torch_utils.py:806-829` |
| 原始输出语义 | 非 end2end 检测头在 eval 下返回 `(decoded, raw_heads)`。decoded 为 `[B,4+nc,N]`，前 4 个通道是输入像素域 xywh，后面是各类 sigmoid 概率，无独立 objectness。本轮期望 `[1,14,8400]`，必须运行时断言。 | `nn/modules/head.py:179-238` |
| 默认预测可能改变输入形状 | `Model.predict` 默认 rect=True；单图通常满足 same_shapes，LetterBox auto 会取 stride 余数，不能把请求 imgsz640 当作已验证 640×640。 | `engine/model.py:519-526`；`engine/predictor.py:212-220` |
| 手动输入与高层预测不是相同路径 | NumPy 路径做 LetterBox、BGR→RGB 和归一化；Tensor 路径不重复这些操作。手动 LetterBox Tensor 的 Results 原图尺寸也不是真实文件原图尺寸。 | `engine/predictor.py:167-220`；`models/yolo/detect/predict.py:67-68` |
| 融合会改变算术路径 | 默认 predictor 请求 fuse=True；raw load 默认 fuse=False。资格检查两路统一不融合，并只改变计算 dtype。 | `engine/predictor.py:428-438`；`nn/backends/pytorch.py:54-78` |
| 精度参数已变化 | 高层 predictor 通过 quantize==16 开 FP16；旧 half/int8 会被映射为 quantize。本次 raw 计算直接显式转换 dtype。 | `engine/predictor.py:434`；`cfg/__init__.py:594-603` |
| NMS 会修改输入 | 转置后的 view 原地覆盖前四个通道，xywh 变为 xyxy。先保存和比较 raw，再把独立 CPU FP32 clone 交给 NMS。明确导入 torchvision 固定后端。 | `utils/nms.py:89-91,96,156-170` |
| 反变换不能提前裁剪 | 默认 scale_boxes 的 xyxy 路径会 clip；ignore 栅格必须先看未裁剪的原图坐标，因此手动使用实际 left/top 和 gain 反变换。 | `utils/ops.py:122-165` |
| 奇数 padding 不对称 | LetterBox 用 round(d−0.1) / round(d+0.1) 分配边缘，保存实际整数 left/top，不能只存浮点半余量。 | `data/augment.py:1745-1765,1792-1807` |

## 本轮资格范围

只使用已经固定的 **109 张、6 个分组 debug**；这些图片之前已用于工程检查，不是新留出集。两路各 batch1，固定 640×640 LetterBox，auto=False、scaleup=True、居中、填充值114、线性插值。共享归一化后的 FP32 输入，FP16 输入仅由其转换而来。

全部 8400 个位置、全部 10 类概率均参加 raw 比较：框 `rtol=0.005, atol=0.5`，概率 `rtol=0.02, atol=0.002`。不按置信度屏蔽 raw 误差，出现超差保留失败，不自动放宽容差或更换权重。

后处理另行固定候选置信度 0.001、NMS IoU0.7、max_det1000、class-aware、best-class-only、CPU FP32。NMS 结果先恢复为未裁剪原图坐标，再交给冻结的项目评价模块：主置信度0.25、IoU0.5。这一统计是项目 target matching，**不是官方 VisDrone AP**。

GT 必须来自原始八列标注，类别保留发布者 0..11，target ID 为 `image_id:原始物理行零起始索引`，包括空白行的位置。预测的 YOLO 类别0..9显式加1映射至发布者1..10，原图 xyxy 转为 xywh。不得用 YOLO 转换标签重建 ignore 区域，也不得去掉重复原始 GT 行而改变目标身份。

## 训练事实的披露

父任务已核查训练实际 max_det 从默认300自动提高至902，源码在 `models/yolo/detect/val.py:65-97`；数据中 train 单图最大351，training_val 单图最大902。这是框架依据标注计数的行为。资格检查的1000上限是另行固定的后处理合同。

父任务也确认训练 loader 在缓存中去掉3条完全重复正标签，分别位于 `0000140_00118_d_0000002`、`9999945_00000_d_0000114`、`9999987_00000_d_0000049`，每张1条；原始准备标签字节保留。相关源码为 `data/utils.py:365-372`。本次资格评价仍按原始物理行身份记录 GT。

资格检查完成后仍须单独审查结论。有限输出、target hit 或本机 dtype 护栏通过，都不自动等于 Gate1 科学通过、官方 AP、Orin/TensorRT 验证或性能结论。旧 Gate0.5 数值护栏保持 **FAILED_UNCHANGED**。

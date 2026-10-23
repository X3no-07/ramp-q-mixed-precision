# Gate1 首轮 float 训练候选合同

状态：冻结，尚未开始训练。30 epochs 是首轮固定预算，不能据此认定已充分收敛。

YOLO11n 官方 COCO 初始权重；10 类 VisDrone；640；batch 4；workers 0；AMP 关闭；SGD，lr0 0.01，momentum 0.937，weight decay 0.0005；seed 20261022；deterministic；patience 0；最后 10 epochs 关闭 mosaic。其余默认值由 framework_defaults.json 与来源 SHA 固定，执行时完整传入并保存最终有效参数。

只复制新划分中的 train/training_val 至独立输出，逐文件核对图像与标签 SHA；不读 Selection 图像/标签，不接触 test-dev。训练内 checkpoint 选择用内置、不了解 ignored 的 mAP50-95；同分保存较晚 epoch。训练正标签省略 ignored 项，但普通 YOLO loss 不屏蔽 ignored 区域的负样本损失。

原环境不修改。自定义 trainer 在 batch 偏离 4 或观察到 loss/fitness 非有限时终止；每次保存前还检查 live/EMA 全部状态及 EMA 转 fp16 后有限，阻止框架静默重同步或夹断权重，保留失败凭证，不自动重启。框架原本的 OOM 减半与 NaN 恢复代码在 framework_sources 保存，具体行号见 implementation_evidence.json。final_eval 的额外 on_fit_epoch_end 回调单独记为最终最佳权重验证，不计作第 31 个训练 epoch。

完成仅表示训练候选产生。accepted_float_baseline_ready 仍为 false；预处理/框还原、ignore-aware 命中、FP32/FP16 数值资格检查另行开展。内置 mAP 不是官方 VisDrone AP，正式量化研究合同仍未冻结。

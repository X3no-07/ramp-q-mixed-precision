# Orin FP16 部署准备包：先检查一张图片的 PyTorch 推理

本包仅用于跑通已有 YOLO11n 模型的板端推理链路。当前脚本执行一次 CUDA FP32 图片推理，后续再单独构建 TensorRT FP16 引擎。

- 权重来自工程 config/gate0.json 指定的 best.pt，SHA256 与既有 Gate 0 运行记录逐字节核对。
- 000020.jpg 是冻结 Selection 清单的第一张图片，未经本次结果筛选。配套标签仅用于保留来源，本脚本不计算准确率。
- 不使用 Holdout，不构成新的验证集；不用于报告量化恢复、准确率、FPS、功耗或长期稳定性结论。
- 固定 batch=1、640x640 letterbox（rect=False）、conf=0.25、NMS IoU=0.7、max_det=1000、CUDA、FP32。本包的完整图像预处理和正式 Gate 0.5 评估口径仍需后续统一核对。
- 本包不包含 TensorRT 引擎，不安装依赖，不发起训练、导出或完整评估。

在 Orin 的解压目录运行：

```bash
sha256sum -c SHA256SUMS
python3 -u run_pytorch_smoke.py
```

脚本先校验输入和自身哈希，再输出当前包版本与 ONNX 模块是否存在，然后运行单张图片。结果存到 results 下带时间戳的独立目录。PYTORCH_MODEL_SMOKE_OK 表示该张图片的完整推理及结果保存通过，不要求一定检出目标。

Windows 侧仅进行了输入来源/哈希核对、Python 3.8 语法兼容检查及 ZIP CRC/内容检查；板端结果以用户实际运行输出为准。

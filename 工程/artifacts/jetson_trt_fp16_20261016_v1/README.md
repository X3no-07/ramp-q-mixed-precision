# 在 Orin 上构建第一个 TensorRT FP16 引擎

本包只推进一个配置：batch=1、输入 640x640、允许 FP16 和 FP32 回退、禁用 TF32。使用板上已经安装的 TensorRT 8.5.x / trtexec，不要求 Python ONNX 包，不安装软件，不修改功耗模式，不重新训练。

model.onnx 从 Gate 0 的原 best.pt 副本导出；源权重 SHA256 与既有运行记录一致。导出为 FP32 ONNX、opset=13、固定形状、不简化、不嵌入 NMS，FP16 优化在板上构建时完成。导出过程和版本记录见 export_validation.json；Windows 侧使用同一张既有 Selection 图片比较 PyTorch 和 ONNX Runtime CPU 原始输出，检查了输入形状、输出形状、有限值及数值误差。该检查没有访问 Holdout。

在本包目录执行：

```bash
sha256sum -c SHA256SUMS
python3 -u build_fp16.py
```

脚本先校验文件和 TensorRT 版本，再查询本机 trtexec 支持的参数。8.5 通常使用 --buildOnly。工作空间上限 1024 MiB 不是进程总内存上限。每次运行建立新的 results/build_时间戳 目录，保存环境查询、构建请求、完整日志、引擎及 engine inspector 层信息。

终端每 20 秒输出已耗时。TENSORRT_FP16_BUILD_OK 只在引擎成功构建、反序列化成功、输入输出形状正确并保存层信息后打印；不表示所有层都使用 FP16，也不表示图片推理、准确率、速度或能耗验证通过。后续应先审查层信息，再在同一图片上执行引擎并核对结果。引擎构建失败保留日志，不能作为机制不复现。

本包的板端脚本在 Windows 上仅检查 Python 3.8 语法；TensorRT 构建和运行仍以板端实际输出为准。Windows 与板端 PyTorch/Ultralytics 版本不同，export_validation.json 记录电脑版本；正式比较前仍需核查板端 FP32 基准与该导出结果的对应关系。

参考：NVIDIA TensorRT release/8.5 的 samples/common/sampleOptions.cpp；Ultralytics 本机 exporter.py 与 utils/export/engine.py。

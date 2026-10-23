"""Correct one manifest-known INT8 Clip's parsed numeric attributes, tiny probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == args.expected_sha256
    assert not args.output.exists()
    import tensorrt as trt
    assert trt.__version__.startswith('8.5.')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    importer = trt.OnnxParser(network, logger)
    assert importer.parse(args.model.read_bytes()), [str(importer.get_error(i)) for i in range(importer.num_errors)]
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    clips = [layer for layer in layers if layer.name == 'rampq_model_0_conv_Clip']
    assert len(clips) == 1 and clips[0].type == trt.LayerType.ACTIVATION
    clip = clips[0]
    clip.__class__ = trt.IActivationLayer
    assert clip.type == trt.ActivationType.CLIP
    before = {name: value if math.isfinite(value) else repr(value)
              for name, value in (('alpha', clip.alpha), ('beta', clip.beta))}
    # The source model is valid INT8 Clip[-127,127]. Explicitly recover those
    # numeric values in the TensorRT activation attributes after import.
    clip.alpha, clip.beta = -127., 127.
    after = {'alpha': clip.alpha, 'beta': clip.beta}
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
    config.avg_timing_iterations = 1
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    for index in range(network.num_inputs):
        value = network.get_input(index)
        assert value.dtype == trt.float32
        value.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    for index in range(network.num_outputs):
        value = network.get_output(index)
        assert value.dtype == trt.float32
        value.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    request = {'status': 'PARSED_INT8_CLIP_ATTRIBUTES_CORRECTED_FOR_TINY_DIAGNOSTIC',
               'model_sha256': args.expected_sha256, 'source_graph_unchanged': True,
               'clip_name': clip.name, 'before': before, 'after': after,
               'reason': 'Source INT8 constants are -127 and +127; imported FP32 attributes do not preserve these numeric values.',
               'diagnostic_only': True, 'full_model_modified': False,
               'original_complete_model_guard': 'FAILED_UNCHANGED'}
    (args.output.parent / 'importer_correction.json').write_text(json.dumps(request, indent=2) + '\n')
    started = time.monotonic()
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    args.output.write_bytes(bytes(engine))
    print(json.dumps({'status': 'CORRECTED_IMPORTER_TINY_ENGINE_SAVED', 'seconds_not_benchmark': time.monotonic() - started}))


if __name__ == '__main__':
    main()

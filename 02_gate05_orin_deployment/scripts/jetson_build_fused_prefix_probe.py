"""Build paired first-Conv/SWISH/INT8 prefixes with frozen graph lineage."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', choices=('original', 'code_clip'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads((root / 'spec.json').read_text())
    assert spec['schema'] == 'rampq.gate05.fused_prefix_probe/v1'
    variant = spec['variants'][args.variant]
    model = root / variant['model_file']
    assert sha(model) == variant['model_sha256']
    assert not args.output.exists() and root in args.output.resolve().parents
    import tensorrt as trt
    assert trt.__version__.startswith('8.5.')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    importer = trt.OnnxParser(network, logger)
    assert importer.parse(model.read_bytes()), [str(importer.get_error(i)) for i in range(importer.num_errors)]
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    names = [layer.name for layer in layers]
    assert len(names) == len(set(names))
    by_name = {layer.name: layer for layer in layers}
    assert {layer.name for layer in layers if layer.type == trt.LayerType.CONVOLUTION} == {spec['conv_name']}
    clip_attributes = []
    for name in spec['clip_names']:
        clip = by_name[name]
        assert clip.type == trt.LayerType.ACTIVATION
        clip.__class__ = trt.IActivationLayer
        assert clip.type == trt.ActivationType.CLIP
        before = {key: value if math.isfinite(value) else repr(value)
                  for key, value in (('alpha', clip.alpha), ('beta', clip.beta))}
        if args.variant == 'code_clip':
            clip.alpha, clip.beta = -127., 127.
        clip_attributes.append({'name': name, 'before': before,
                                'after': {'alpha': clip.alpha, 'beta': clip.beta},
                                'explicit_importer_correction': args.variant == 'code_clip'})
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
    config.avg_timing_iterations = 1
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    expected = {row['name']: row for row in [spec['input_binding']] + spec['output_bindings_in_order']}
    imported = []
    for is_input, count in ((True, network.num_inputs), (False, network.num_outputs)):
        for index in range(count):
            tensor = network.get_input(index) if is_input else network.get_output(index)
            assert tensor.name in expected and tensor.dtype == trt.float32
            assert list(tensor.shape) == expected[tensor.name]['shape'] and expected[tensor.name]['input'] == is_input
            tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            imported.append({'name': tensor.name, 'shape': list(tensor.shape), 'input': is_input, 'dtype': str(tensor.dtype)})
    assert len(imported) == len(expected) == 2
    request = {'status': 'FROZEN_FUSED_PREFIX_PARSED', 'variant': args.variant,
               'spec_sha256': sha(root / 'spec.json'), 'model_sha256': sha(model),
               'clip_attributes': clip_attributes, 'bindings': imported, 'conv_name': spec['conv_name'],
               'flags': ['INT8', 'FP16', 'TF32_DISABLED'], 'workspace_mib': 1024, 'avg_timing_iterations': 1,
               'source_graph_unchanged_by_native_builder': True, 'diagnostic_only': True,
               'original_complete_model_guard': 'FAILED_UNCHANGED', 'dataset_evaluation_run': False,
               'limits': 'Actual SWISH fusion and INT8 Conv output require final inspector verification; no full-model numerical pass.'}
    (args.output.parent / 'parsed_network_request.json').write_text(json.dumps(request, indent=2, allow_nan=False) + '\n')
    started = time.monotonic()
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    args.output.write_bytes(bytes(engine))
    print(json.dumps({'status': 'FUSED_PREFIX_ENGINE_SAVED', 'variant': args.variant,
                      'elapsed_seconds_not_benchmark': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()

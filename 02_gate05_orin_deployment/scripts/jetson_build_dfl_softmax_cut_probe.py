"""One frozen corrected DFL probability-input cut flag case; preserve every parser precision constraint."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

from jetson_dfl_softmax_cut_probe import require, sha, validate_inputs, write


def finite(value):
    return value if math.isfinite(value) else repr(value)


def layer_constraints(network):
    """Query every layer, including SHAPE / Q / DQ; never read unset values."""
    rows = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        row = {'index': index, 'name': layer.name, 'layer_type': str(layer.type),
               'num_inputs': layer.num_inputs, 'num_outputs': layer.num_outputs}
        try:
            is_set = bool(layer.precision_is_set)
            row['precision_is_set'] = {'supported': True, 'value': is_set}
            if is_set:
                try:
                    row['precision'] = str(layer.precision)
                except Exception as error:
                    row['precision_value_error'] = {'error_type': type(error).__name__, 'error': str(error)}
        except Exception as error:
            row['precision_is_set'] = {'supported': False, 'error_type': type(error).__name__, 'error': str(error)}
        row['output_types'] = []
        for output_index in range(layer.num_outputs):
            output = {'index': output_index}
            try:
                is_set = bool(layer.output_type_is_set(output_index))
                output['output_type_is_set'] = {'supported': True, 'value': is_set}
                if is_set:
                    try:
                        output['requested_output_type'] = str(layer.get_output_type(output_index))
                    except Exception as error:
                        output['requested_output_type_error'] = {'error_type': type(error).__name__, 'error': str(error)}
            except Exception as error:
                output['output_type_is_set'] = {'supported': False, 'error_type': type(error).__name__, 'error': str(error)}
            try:
                tensor = layer.get_output(output_index)
                output['tensor_name'] = tensor.name if tensor is not None else None
                output['tensor_dtype'] = str(tensor.dtype) if tensor is not None else None
                output['tensor_shape'] = list(tensor.shape) if tensor is not None else None
                output['is_shape_tensor'] = bool(tensor.is_shape_tensor) if tensor is not None else None
            except Exception as error:
                output['tensor_query_error'] = {'error_type': type(error).__name__, 'error': str(error)}
            row['output_types'].append(output)
        rows.append(row)
    return rows


def constraint_values(rows):
    # Wrapper casting can change the Python-visible .type property (LayerType
    # versus ActivationType). Compare the precision requests themselves only.
    values = []
    for row in rows:
        selected = {key: row[key] for key in ('index', 'name', 'precision_is_set',
                    'precision', 'precision_value_error') if key in row}
        selected['outputs'] = [{key: item[key] for key in ('index', 'output_type_is_set',
                     'requested_output_type', 'requested_output_type_error') if key in item}
                   for item in row['output_types']]
        values.append(selected)
    return values


def actual_flags(config, trt):
    flags = {}
    for name in dir(trt.BuilderFlag):
        if not name.isupper():
            continue
        try:
            flags[name] = {'supported': True, 'enabled': bool(config.get_flag(getattr(trt.BuilderFlag, name)))}
        except Exception as error:
            flags[name] = {'supported': False, 'error_type': type(error).__name__, 'error': str(error)}
    return {'raw_config_flags': int(config.flags), 'get_flag': flags}


def build(root, case, output, state):
    import tensorrt as trt
    require(trt.__version__.startswith('8.5.'), 'Expected TensorRT 8.5')
    spec = validate_inputs(root)
    row = next(item for item in spec['cases'] if item['id'] == case)
    variant = spec['variants'][row['variant']]
    model = root / variant['model_file']
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    importer = trt.OnnxParser(network, logger)
    state['phase'] = 'PARSING'
    parsed = importer.parse(model.read_bytes())
    errors = [str(importer.get_error(i)) for i in range(importer.num_errors)]
    write(output.parent / 'parser_errors.json', {'parse_returned': bool(parsed), 'errors': errors})
    require(parsed, 'ONNX parse failed: ' + repr(errors))
    before = layer_constraints(network)
    write(output.parent / 'parsed_constraints_before.json', before)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    require(len({layer.name for layer in layers}) == len(layers), 'Duplicate parser layer names')
    by_name = {layer.name: layer for layer in layers}
    require({layer.name for layer in layers if layer.type == trt.LayerType.CONVOLUTION} == set(spec['conv_names']) and
            len(spec['conv_names']) == 1, 'Wrong one DFL source Conv layer')
    require(not any(layer.type == trt.LayerType.SOFTMAX for layer in layers),
            'Frozen DFL cut must consume external probabilities and contain no native Softmax')
    clip_attributes = []
    for name in spec['clip_names']:
        clip = by_name[name]
        require(clip.type == trt.LayerType.ACTIVATION, 'Clip did not parse as activation')
        clip.__class__ = trt.IActivationLayer
        require(clip.type == trt.ActivationType.CLIP, 'Wrong parsed activation')
        attributes = {'name': name, 'before': {'alpha': finite(clip.alpha), 'beta': finite(clip.beta)},
                      'explicit_importer_correction': row['apply_native_clip_attribute_correction']}
        if row['apply_native_clip_attribute_correction']:
            clip.alpha, clip.beta = -127.0, 127.0
        attributes['after'] = {'alpha': finite(clip.alpha), 'beta': finite(clip.beta)}
        clip_attributes.append(attributes)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    if row['precision_constraint_flag'] == 'obey':
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
    config.avg_timing_iterations = 1
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    flags = actual_flags(config, trt)
    for name, expected in (('INT8', True), ('FP16', True), ('TF32', False),
                            ('PREFER_PRECISION_CONSTRAINTS', False),
                            ('OBEY_PRECISION_CONSTRAINTS', row['precision_constraint_flag'] == 'obey')):
        require(flags['get_flag'][name] == {'supported': True, 'enabled': expected}, 'Actual flag differs: ' + name)
    desired = {item['name']: item for item in [spec['input_binding']] + spec['output_bindings_in_order']}
    bindings = []
    for is_input, count in ((True, network.num_inputs), (False, network.num_outputs)):
        for index in range(count):
            tensor = network.get_input(index) if is_input else network.get_output(index)
            require(tensor.name in desired, 'Unexpected parser binding')
            target = desired[tensor.name]
            require(tensor.dtype == trt.float32 and list(tensor.shape) == target['shape'] and target['input'] == is_input,
                    'Parser binding contract differs')
            tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            bindings.append({'name': tensor.name, 'input': is_input, 'shape': list(tensor.shape),
                             'dtype': str(tensor.dtype), 'allowed_formats': int(tensor.allowed_formats)})
    require(len(bindings) == len(desired) == 2 and len({item['name'] for item in bindings}) == 2,
            'Parser binding count differs')
    after = layer_constraints(network)
    write(output.parent / 'parsed_constraints_after.json', after)
    require(constraint_values(before) == constraint_values(after), 'Parser layer precision/output constraints changed')
    request = {'status': 'DFL_SOFTMAX_CUT_REQUEST_FROZEN', 'case': row,
               'spec_sha256': sha(root / 'spec.json'), 'model_sha256': sha(model),
               'script_sha256': sha(Path(__file__).resolve()),
               'runner_script_sha256': sha(root / 'jetson_dfl_softmax_cut_probe.py'),
               'parent_spec_sha256': sha(root / 'parent_spec.json'),
               'parent_dfl_spec_sha256': sha(root / 'parent_dfl_spec.json'),
               'parent_dfl_manifest_sha256': sha(root / 'parent_dfl_manifest.json'),
               'parent_dfl_original_checkpoint_sha256': sha(root / 'parent_dfl_original_checkpoint.json'),
               'fixture_sha256': sha(root / spec['input_file']),
               'input_role': 'CPU_SOFTMAX_PROBABILITIES_NOT_LOGITS',
               'native_softmax_layer_count': 0,
               'clip_attributes': clip_attributes, 'bindings': bindings, 'flags': flags,
               'workspace_bytes_actual': int(config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)),
               'avg_timing_iterations_actual': int(config.avg_timing_iterations),
               'profiling_verbosity_actual': str(config.profiling_verbosity),
               'no_layer_precision_or_output_type_setters_or_resets': True,
               'parsed_constraints_before_after_identical': True, 'tensorrt': trt.__version__,
               'constraint_comparison_scope': 'all layer precision and output type requests; metadata retained separately',
               'diagnostic_only': True, 'dataset_evaluation_run': False,
               'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'}
    write(output.parent / 'parsed_network_request.json', request)
    state['phase'] = 'BUILDING'
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    state['build_seconds_not_benchmark'] = time.monotonic() - started
    require(serialized is not None, 'TensorRT returned no serialized engine')
    output.write_bytes(bytes(serialized))
    state['phase'] = 'INSPECTING_ENGINE'
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(output.read_bytes())
    require(engine is not None, 'Cannot deserialize serialized engine')
    inspector_raw = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    (output.parent / 'engine_layers.json').write_text(inspector_raw + '\n', encoding='utf-8')
    actual_bindings = []
    for index in range(engine.num_bindings):
        name = engine.get_binding_name(index)
        actual = {'name': name, 'input': bool(engine.binding_is_input(index)),
                  'shape': list(engine.get_binding_shape(index)), 'dtype': str(engine.get_binding_dtype(index)),
                  'format': str(engine.get_binding_format(index))}
        actual_bindings.append(actual)
    write(output.parent / 'engine_bindings.json', actual_bindings)
    require(len(actual_bindings) == len(desired) == 2 and len({item['name'] for item in actual_bindings}) == 2,
            'Actual binding count differs')
    for index, actual in enumerate(actual_bindings):
        require(actual['name'] in desired, 'Unknown actual binding')
        target = desired[actual['name']]
        require(actual['shape'] == target['shape'] and actual['input'] == target['input'] and
                engine.get_binding_dtype(index) == trt.float32 and engine.get_binding_format(index) == trt.TensorFormat.LINEAR,
                'Actual binding contract differs')
    require(sha(model) == variant['model_sha256'], 'Source graph mutated during build')
    state.update({'status': 'DFL_SOFTMAX_CUT_ENGINE_SAVED', 'phase': 'COMPLETE',
                  'engine_sha256': sha(output), 'engine_bytes': output.stat().st_size,
                  'bindings': actual_bindings, 'model_sha256_after': sha(model),
                  'builder_request_sha256': sha(output.parent / 'parsed_network_request.json'),
                  'diagnostic_only': True, 'numerical_pass_claim': False})
    write(output.parent / 'build_result.json', state)
    print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', choices=('code_clip_none', 'code_clip_obey'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(root in output.parents and output.parent.is_dir() and not output.exists(), 'Unsafe or existing output')
    state = {'case': args.case, 'phase': 'VALIDATING', 'python': sys.version}
    try:
        build(root, args.case, output, state)
    except BaseException as error:
        state.update({'status': 'DFL_SOFTMAX_CUT_BUILD_FAILED', 'error_type': type(error).__name__,
                      'error': str(error), 'traceback': traceback.format_exc(),
                      'inference_run': False, 'diagnostic_only': True,
                      'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'})
        write(output.parent / 'build_failure.json', state)
        write(output.parent / 'build_result.json', state)
        raise


if __name__ == '__main__':
    main()

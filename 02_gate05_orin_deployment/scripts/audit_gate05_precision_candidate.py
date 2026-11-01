"""Audit requested mixed precision from actual native layer formats; local, no execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def matches(name, native_name):
    return bool(re.search(re.escape(name) + r'(?=$|[\s,+|)])', native_name))


def graph_index(path):
    import onnx
    graph = onnx.load(str(path)).graph
    return {'nodes': {node.name: node for node in graph.node},
            'producers': {name: node for node in graph.node for name in node.output},
            'initializers': {value.name: value for value in graph.initializer}}


def signature(tensor):
    from onnx import numpy_helper
    array = numpy_helper.to_array(tensor)
    return {'dtype': array.dtype.str, 'shape': list(array.shape),
            'bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def activation_trace(conv, graph):
    producers, values = graph['producers'], graph['initializers']
    dq = producers[conv.input[0]]
    assert dq.op_type == 'DequantizeLinear' and len(dq.input) == 3
    quantize = producers[dq.input[0]]
    assert quantize.op_type == 'QuantizeLinear' and len(quantize.input) == 3
    clip = producers[quantize.input[0]]
    assert clip.op_type == 'Clip' and len(clip.input) == 3
    constants = {key: signature(values[name]) for key, name in
                 [('scale', quantize.input[1]), ('zero', quantize.input[2]),
                  ('dq_scale', dq.input[1]), ('dq_zero', dq.input[2]),
                  ('clip_min', clip.input[1]), ('clip_max', clip.input[2])]}
    assert constants['scale'] == constants['dq_scale'] and constants['zero'] == constants['dq_zero']
    return {'source_tensor': clip.input[0], 'constants': constants,
            'q_attributes': [value.SerializeToString().hex() for value in quantize.attribute],
            'dq_attributes': [value.SerializeToString().hex() for value in dq.attribute]}


def original_weight_name(conv, request, graph):
    if request['precision'] == 'FP16':
        assert conv.input[1] == request['weight_input']
        return conv.input[1]
    dq = graph['producers'][conv.input[1]]
    assert dq.op_type == 'DequantizeLinear'
    quantize = graph['producers'][dq.input[0]]
    assert quantize.op_type == 'QuantizeLinear' and quantize.input[0] == request['weight_input']
    return quantize.input[0]


def check_graph_requests(graph, requests, module_mapping):
    import numpy as np
    from onnx import numpy_helper
    errors = []
    nodes = graph['nodes']
    if {name for name, node in nodes.items() if node.op_type == 'Conv'} != {row['onnx_node'] for row in requests}:
        errors.append('Graph Conv inventory differs from requested source inventory')
    original = {row['onnx_node']: row for row in module_mapping}
    if len(original) != 88 or set(original) != {row['onnx_node'] for row in requests}:
        errors.append('Module map inventory differs from 88 unique requests')
    for row in requests:
        try:
            frozen = original[row['onnx_node']]
            assert row['module'] == frozen['pytorch_module'] and row['group'] == frozen['group']
            assert row['weight_input'] == frozen['weight_input']
            conv = nodes[row['onnx_node']]
            weight_name = original_weight_name(conv, row, graph)
            assert weight_name in graph['initializers']
            if row['precision'] == 'FP16':
                assert conv.input[0] == row['original_activation_input']
            else:
                trace = activation_trace(conv, graph)
                assert trace['source_tensor'] == row['original_activation_input']
                prefix = frozen['qdq_prefix']
                for field, suffix in [('scale', 'a_scale'), ('zero', 'a_zero'),
                                       ('clip_min', 'clip_min'), ('clip_max', 'clip_max')]:
                    assert trace['constants'][field] == signature(graph['initializers'][prefix + '_' + suffix])
                weight_dq = graph['producers'][conv.input[1]]
                weight_q = graph['producers'][weight_dq.input[0]]
                assert list(weight_q.input[1:]) == list(weight_dq.input[1:]) == [prefix + '_w_scale', prefix + '_w_zero']
                assert [(attribute.name, attribute.i) for attribute in weight_q.attribute] == [('axis', 0)]
                assert [(attribute.name, attribute.i) for attribute in weight_dq.attribute] == [('axis', 0)]
                expected_scale = np.asarray(frozen['activation_scale'], dtype=np.float32)
                expected_bound = np.asarray(expected_scale * np.float32(127), dtype=np.float32)
                for suffix, expected in [('a_scale', expected_scale), ('a_zero', np.asarray(0, dtype=np.int8)),
                                         ('clip_min', -expected_bound), ('clip_max', expected_bound),
                                         ('w_scale', np.asarray(frozen['weight_scale'], dtype=np.float32)),
                                         ('w_zero', np.zeros(len(frozen['weight_scale']), dtype=np.int8))]:
                    actual = numpy_helper.to_array(graph['initializers'][prefix + '_' + suffix])
                    assert actual.dtype == expected.dtype and actual.shape == expected.shape
                    assert actual.tobytes() == expected.tobytes(), suffix
        except (KeyError, IndexError, AssertionError, ValueError) as error:
            errors.append('Source request/quantizer mismatch ' + row['onnx_node'] + ': ' + str(error))
    return errors


def fusion_accounting(layer, members, graph):
    checks = {'same_group': len({row['group'] for row in members}) == 1,
              'same_requested_precision': len({row['precision'] for row in members}) == 1,
              'one_native_activation_input': len(layer.get('Inputs', [])) == 1,
              'source_graph_available': graph is not None}
    result = {'status': 'FUSION_ACCOUNTING_REQUIRED', 'checks': checks}
    if graph is None:
        return result
    named = {name for name, node in graph['nodes'].items() if node.op_type == 'Conv' and matches(name, layer['Name'])}
    checks['complete_exact_source_names'] = named == {row['onnx_node'] for row in members}
    counts, traces, errors = [], [], []
    for request in members:
        try:
            conv = graph['nodes'][request['onnx_node']]
            name = original_weight_name(conv, request, graph)
            counts.append({'onnx_node': request['onnx_node'], 'initializer': name,
                           'count': math.prod(graph['initializers'][name].dims)})
            if request['precision'] == 'INT8':
                traces.append(activation_trace(conv, graph))
            else:
                assert conv.input[0] == request['original_activation_input']
                traces.append({'source_tensor': conv.input[0]})
        except (KeyError, IndexError, AssertionError) as error:
            errors.append(request['onnx_node'] + ': ' + type(error).__name__)
    checks['all_traces_complete'] = not errors and len(counts) == len(traces) == len(members)
    checks['same_original_activation_and_quantizer_if_int8'] = (len(traces) == len(members) and
                                                               all(trace == traces[0] for trace in traces))
    total = sum(row['count'] for row in counts)
    checks['native_weight_count_equals_source_sum'] = (len(counts) == len(members) and
                                                       layer.get('Weights', {}).get('Count') == total)
    result.update(status='FUSION_ACCOUNTING_PASS' if all(checks.values()) else 'FUSION_ACCOUNTING_REQUIRED',
                  source_weight_counts=counts, source_weight_count_sum=total,
                  native_weight_count=layer.get('Weights', {}).get('Count'), traces=traces, errors=errors)
    return result


def audit_native_layers(layers, requests, graph, module_mapping):
    errors = check_graph_requests(graph, requests, module_mapping)
    assert len(requests) == len({row['onnx_node'] for row in requests}) == 88
    convs = [layer for layer in layers if 'Convolution' in layer.get('LayerType', '')]
    rows = []
    for request in requests:
        found = [layer for layer in convs if matches(request['onnx_node'], layer['Name'])]
        row = dict(request, matching_engine_layers=len(found), requested_formats_verified=False,
                   strict_int8_tactic_verified=False)
        if len(found) != 1:
            errors.append('Expected one native Conv match for ' + request['onnx_node'])
        else:
            layer = found[0]
            weight_type = layer.get('Weights', {}).get('Type', '')
            inputs = [value.get('Format/Datatype', '') for value in layer.get('Inputs', [])]
            outputs = [value.get('Format/Datatype', '') for value in layer.get('Outputs', [])]
            weight_count = layer.get('Weights', {}).get('Count')
            has_weight_count = isinstance(weight_count, int) and weight_count > 0
            activation_index = layer.get('ConvXAsActInputIdx', -1)
            activation_index = activation_index if isinstance(activation_index, int) and activation_index >= 0 else 0
            if request['precision'] == 'INT8':
                formats = (weight_type.lower() in ('int8', 'int8_t', 'signed int8', 'signed byte') and
                           len(inputs) > activation_index and 'INT8' in inputs[activation_index].upper() and has_weight_count)
            else:
                formats = (weight_type.lower() in ('half', 'float16', 'fp16') and bool(inputs) and bool(outputs) and
                           all('FP16' in value.upper() for value in inputs + outputs) and has_weight_count)
            tactic = layer.get('TacticName', '')
            integer_tactic = bool(re.search(r'i8|s8|int8|imma', tactic, re.IGNORECASE))
            row.update(engine_layer=layer['Name'], engine_layer_type=layer['LayerType'],
                       weights_type=weight_type, weights_count=weight_count, input_formats=inputs,
                       convolution_activation_input_index=activation_index,
                       output_formats=outputs, requested_formats_verified=formats,
                       tactic_name=tactic, tactic_value=layer.get('TacticValue'),
                       strict_int8_tactic_verified=request['precision'] == 'INT8' and formats and integer_tactic,
                       integer_tactic_evidence=('NOT_REQUESTED_FP16' if request['precision'] == 'FP16' else
                           'NAMED_INTEGER_TACTIC' if integer_tactic else
                           'TACTIC_NAME_NOT_EXPOSED' if not tactic else 'INTEGER_TACTIC_NOT_IDENTIFIED'))
            if not formats:
                errors.append('Requested ' + request['precision'] + ' formats incomplete for ' + request['onnx_node'])
        rows.append(row)
    mapped = {row['engine_layer'] for row in rows if 'engine_layer' in row}
    unmapped = [layer['Name'] for layer in convs if layer['Name'] not in mapped]
    errors.extend('Unmapped native Conv: ' + name for name in unmapped)
    fusions = []
    for name in sorted(mapped):
        members = [row for row in rows if row.get('engine_layer') == name]
        if len(members) == 1:
            row = members[0]
            try:
                conv = graph['nodes'][row['onnx_node']]
                weight_name = original_weight_name(conv, row, graph)
                expected_count = math.prod(graph['initializers'][weight_name].dims)
                row['source_weight_count'] = expected_count
                assert row['weights_count'] == expected_count
            except (KeyError, IndexError, AssertionError):
                row['requested_formats_verified'] = row['strict_int8_tactic_verified'] = False
                errors.append('Native weight count/source attribution mismatch: ' + name)
        if len(members) > 1:
            layer = next(layer for layer in convs if layer['Name'] == name)
            accounting = fusion_accounting(layer, members, graph)
            fusions.append({'engine_layer': name, 'source_members': [row['onnx_node'] for row in members],
                            'accounting': accounting})
            if accounting['status'] != 'FUSION_ACCOUNTING_PASS':
                errors.append('Fused source precision attribution incomplete: ' + name)
                for row in members:
                    row['requested_formats_verified'] = row['strict_int8_tactic_verified'] = False
    quantized = [row for row in rows if row['precision'] == 'INT8']
    floating = [row for row in rows if row['precision'] == 'FP16']
    formats_complete = not errors and all(row['requested_formats_verified'] for row in rows)
    int8_strict = formats_complete and all(row['strict_int8_tactic_verified'] for row in quantized)
    groups = {}
    for row in rows:
        group = groups.setdefault(row['group'], {'source_convs': 0, 'requested_int8': 0, 'requested_fp16': 0,
                                                  'verified_int8_formats': 0, 'verified_fp16_formats': 0})
        group['source_convs'] += 1
        prefix = 'int8' if row['precision'] == 'INT8' else 'fp16'
        group['requested_' + prefix] += 1
        group['verified_' + prefix + '_formats'] += int(row['requested_formats_verified'])
    return {'schema': 'rampq.gate05.precision_candidate_audit/v1',
            'status': 'REQUESTED_CONV_FORMATS_VERIFIED' if formats_complete else 'PRECISION_REVIEW_REQUIRED',
            'requested_conv_formats_complete': formats_complete, 'source_convs': 88,
            'quantized_conv_count': len(quantized), 'fp16_conv_count': len(floating),
            'verified_int8_format_convs': sum(row['requested_formats_verified'] for row in quantized),
            'verified_fp16_format_convs': sum(row['requested_formats_verified'] for row in floating),
            'strict_int8_status': ('NO_INT8_REQUESTED' if not quantized else
                                   'ALL_REQUESTED_INT8_TACTICS_VERIFIED' if int8_strict else 'INTEGER_TACTIC_REVIEW_REQUIRED'),
            'named_integer_tactic_source_convs': sum(row['strict_int8_tactic_verified'] for row in quantized),
            'source_convs_without_tactic_name': sum(row.get('integer_tactic_evidence') == 'TACTIC_NAME_NOT_EXPOSED' for row in quantized),
            'groups': groups, 'errors': errors, 'unmapped_engine_convs': unmapped,
            'mapping': rows, 'fusions': fusions,
            'scope': 'Actual requested Conv weight/input/output formats; integer tactic evidence separate. '
                     'No numerical migration, target recovery, dataset accuracy or performance claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec, build = read(args.spec), read(args.build / 'build_result.json')
    assert spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1'
    assert build['status'] == 'BUILT_AND_DESERIALIZED' and build['configuration'] == spec['configuration']
    assert sha(args.spec) == build['spec_sha256']
    assert sha(args.onnx) == spec['graph_sha256'] == build['onnx_sha256']
    assert sha(args.map) == spec['module_map_sha256'] == build['module_map_sha256']
    assert sha(args.build / 'model.engine') == build['engine_sha256']
    bundle = args.spec.parent
    manifest_path = bundle / 'manifest.json'
    assert sha(manifest_path) == build['manifest_sha256']
    for name, info in read(manifest_path)['files'].items():
        path = bundle / name
        assert path.resolve().parent == bundle.resolve()
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
    assert sha(bundle / 'graph_audit.json') == spec['graph_audit_sha256'] == build['graph_audit_sha256']
    assert sha(bundle / 'preflight.json') == spec['failed_preflight_sha256']
    assert read(bundle / 'preflight.json')['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert sha(bundle / 'diagnostic_scope.json') == spec['diagnostic_scope_sha256']
    scope = read(bundle / 'diagnostic_scope.json')
    assert scope['status'] == 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY'
    assert scope['failed_preflight_sha256'] == spec['failed_preflight_sha256']
    assert scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False
    expected = spec['expected_binding_contract']
    assert build['expected_binding_contract'] == expected and build['primary_output_name'] == spec['primary_output_name']
    key = lambda row: (row['name'], row['input'], tuple(row['shape']), row['dtype'])
    assert len({row['name'] for row in build['bindings']}) == len(build['bindings'])
    assert sorted(map(key, build['bindings'])) == sorted(map(key, expected))
    assert all(row.get('format') == 'TensorFormat.LINEAR' for row in build['bindings'])
    assert spec['diagnostic_only'] is True and build['diagnostic_only'] is True
    assert spec['original_local_preflight_status'] == build['local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    request_groups = {
        'fp16_control': set(), 'q_all': {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'},
        'r_only_G1': {'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'},
        'r_only_G1_G5': {'G2', 'G3', 'G4', 'G6', 'G7', 'G8'},
        'r_only_G1_G2': {'G3', 'G4', 'G5', 'G6', 'G7', 'G8'}, 'q_only_G1': {'G1'}}
    for row in spec['requests']:
        assert row['precision'] == ('INT8' if row['group'] in request_groups[spec['configuration']] else 'FP16')
    layer_path = args.build / 'engine_layers.json'
    report = audit_native_layers(read(layer_path)['Layers'], spec['requests'], graph_index(args.onnx), read(args.map)['mapping'])
    assert report['quantized_conv_count'] == spec['quantized_conv_count']
    assert report['fp16_conv_count'] == spec['fp16_conv_count']
    report.update(configuration=spec['configuration'], engine_sha256=build['engine_sha256'],
                  onnx_sha256=sha(args.onnx), spec_sha256=sha(args.spec), module_map_sha256=sha(args.map),
                  layer_file_sha256=sha(layer_path), build_result_sha256=sha(args.build / 'build_result.json'),
                  original_numerical_failure_retained=True,
                  nonquantized_float_control_status=spec['nonquantized_float_control_status'])
    output = args.output or args.build / 'precision_audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'mapping'}, indent=2))
    if not report['requested_conv_formats_complete']:
        raise SystemExit(2)
    if report['strict_int8_status'] == 'INTEGER_TACTIC_REVIEW_REQUIRED':
        raise SystemExit(3)


if __name__ == '__main__':
    main()

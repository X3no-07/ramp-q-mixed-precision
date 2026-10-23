"""Inspect actual INT8 Conv coverage. Non-INT8 or unmapped layers fail closed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def node_name_matches(node_name, layer_name):
    return bool(re.search(re.escape(node_name) + r'(?=$|[\s,+|)])', layer_name))


def load_graph_index(path):
    # ONNX is needed only by this local post-build audit, never by the board runner.
    import onnx
    graph = onnx.load(str(path)).graph
    return {'nodes': {node.name: node for node in graph.node},
            'producers': {name: node for node in graph.node for name in node.output},
            'initializers': {value.name: value for value in graph.initializer}}


def constant_signature(tensor):
    from onnx import numpy_helper
    value = numpy_helper.to_array(tensor)
    return {'dtype': value.dtype.str, 'shape': list(value.shape),
            'value_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def trace_activation_quantizer(conv, graph):
    producers, initializers = graph['producers'], graph['initializers']
    dq = producers[conv.input[0]]
    assert dq.op_type == 'DequantizeLinear' and len(dq.input) == 3
    quantize = producers[dq.input[0]]
    assert quantize.op_type == 'QuantizeLinear' and len(quantize.input) == 3
    clip = producers[quantize.input[0]]
    assert clip.op_type == 'Clip' and len(clip.input) == 3
    signatures = {key: constant_signature(initializers[name]) for key, name in
                  [('scale', quantize.input[1]), ('zero', quantize.input[2]),
                   ('dq_scale', dq.input[1]), ('dq_zero', dq.input[2]),
                   ('clip_min', clip.input[1]), ('clip_max', clip.input[2])]}
    return {'preclip_tensor': clip.input[0], 'constants': signatures,
            'q_dq_constants_match': signatures['scale'] == signatures['dq_scale'] and
                                    signatures['zero'] == signatures['dq_zero'],
            'q_attributes': [attribute.SerializeToString().hex() for attribute in quantize.attribute],
            'dq_attributes': [attribute.SerializeToString().hex() for attribute in dq.attribute]}


def convolution_activation_input(layer):
    """Resolve the inspector's Conv activation operand without guessing a slot.

    TensorRT's explicit -1 sentinel uses the ordinary primary input (slot 0),
    matching the native generic audit. An absent field is unambiguous only for
    a single-input convolution. A nonnegative index selects that actual input;
    residual/bias operands must never substitute for the declared activation.
    """
    inputs = layer.get('Inputs', [])
    raw = layer.get('ConvXAsActInputIdx')
    if not isinstance(inputs, list) or not inputs:
        return {'valid': False, 'index': None, 'raw_index': raw, 'evidence': 'NO_NATIVE_INPUTS'}
    if 'ConvXAsActInputIdx' not in layer:
        if len(inputs) == 1:
            return {'valid': True, 'index': 0, 'raw_index': None, 'evidence': 'ONLY_NATIVE_INPUT'}
        return {'valid': False, 'index': None, 'raw_index': None,
                'evidence': 'MISSING_INDEX_WITH_MULTIPLE_NATIVE_INPUTS'}
    if type(raw) is not int or raw < -1:
        return {'valid': False, 'index': None, 'raw_index': raw, 'evidence': 'INVALID_NATIVE_ACTIVATION_INDEX'}
    index = 0 if raw == -1 else raw
    if index >= len(inputs):
        return {'valid': False, 'index': None, 'raw_index': raw, 'evidence': 'NATIVE_ACTIVATION_INDEX_OUT_OF_RANGE'}
    return {'valid': True, 'index': index, 'raw_index': raw,
            'evidence': 'NATIVE_PRIMARY_INPUT_SENTINEL' if raw == -1 else 'EXPLICIT_NATIVE_ACTIVATION_INDEX'}


def account_fused_convs(layer, members, mapping_by_node, graph):
    """A shared descriptor covers source Convs only with explicit horizontal-fusion accounting."""
    activation = convolution_activation_input(layer)
    checks = {'source_onnx_available': graph is not None,
              'same_declared_group': len({row['group'] for row in members}) == 1,
              'one_native_activation_input': len(layer.get('Inputs', [])) == 1,
              'native_activation_index_valid': activation['valid'] and activation['index'] == 0}
    report = {'status': 'FUSION_ACCOUNTING_REQUIRED', 'checks': checks,
              'convolution_activation_input': activation,
              'source_members': [row['onnx_node'] for row in members]}
    if graph is None:
        return report
    source_names = {row['onnx_node'] for row in members}
    named_convs = {name for name, node in graph['nodes'].items()
                   if node.op_type == 'Conv' and node_name_matches(name, layer['Name'])}
    checks['exact_complete_source_member_names'] = named_convs == source_names and len(source_names) == len(members)
    report['all_graph_conv_names_in_native_layer_name'] = sorted(named_convs)
    traces, source_counts, trace_errors = [], [], []
    for row in members:
        try:
            conv = graph['nodes'][row['onnx_node']]
            assert conv.op_type == 'Conv'
            traces.append(dict(onnx_node=row['onnx_node'], **trace_activation_quantizer(conv, graph)))
            weight_name = mapping_by_node[row['onnx_node']]['weight_input']
            weight_dq = graph['producers'][conv.input[1]]
            assert weight_dq.op_type == 'DequantizeLinear'
            weight_q = graph['producers'][weight_dq.input[0]]
            assert weight_q.op_type == 'QuantizeLinear' and weight_q.input[0] == weight_name
            weight = graph['initializers'][weight_name]
            source_counts.append({'onnx_node': row['onnx_node'], 'initializer': weight_name,
                                  'count': math.prod(weight.dims)})
        except (KeyError, IndexError, AssertionError) as error:
            trace_errors.append(row['onnx_node'] + ': missing or unsupported Q/DQ source structure (' +
                                type(error).__name__ + ')')
    checks['all_activation_and_weight_qdq_traces_complete'] = not trace_errors and len(traces) == len(members)
    checks['all_q_dq_scale_zero_constants_match'] = bool(traces) and all(row['q_dq_constants_match'] for row in traces)
    checks['same_original_preclip_tensor'] = len(traces) == len(members) and len({row['preclip_tensor'] for row in traces}) == 1
    checks['byte_identical_activation_scale_zero_and_clip_bounds'] = (len(traces) == len(members) and
        all(row['constants'] == traces[0]['constants'] for row in traces))
    checks['identical_q_dq_attributes'] = (len(traces) == len(members) and
        all((row['q_attributes'], row['dq_attributes']) == (traces[0]['q_attributes'], traces[0]['dq_attributes'])
            for row in traces))
    source_count = sum(row['count'] for row in source_counts)
    checks['native_weight_count_equals_source_sum'] = (len(source_counts) == len(members) and
        layer.get('Weights', {}).get('Count') == source_count)
    report.update(status='FUSION_ACCOUNTING_PASS' if all(checks.values()) else 'FUSION_ACCOUNTING_REQUIRED',
                  source_weight_counts=source_counts, source_weight_count_sum=source_count,
                  native_weight_count=layer.get('Weights', {}).get('Count'),
                  activation_traces=traces, trace_errors=trace_errors)
    return report


def inspect_conv_precision(layer):
    """Report inspector formats separately from the stronger named-tactic test."""
    weight_type = layer.get('Weights', {}).get('Type', '')
    inputs = [t.get('Format/Datatype', '') for t in layer.get('Inputs', [])]
    outputs = [t.get('Format/Datatype', '') for t in layer.get('Outputs', [])]
    tactic = layer.get('TacticName', '')
    activation = convolution_activation_input(layer)
    weights_int8 = weight_type.lower() in ('int8', 'int8_t', 'signed int8', 'signed byte')
    activation_int8 = activation['valid'] and 'INT8' in inputs[activation['index']].upper()
    integer_tactic = bool(re.search(r'i8|s8|int8|imma', tactic, re.IGNORECASE))
    return {'engine_layer': layer['Name'], 'engine_layer_type': layer['LayerType'],
            'weights_type': weight_type, 'input_formats': inputs, 'output_formats': outputs,
            'convolution_activation_input_index': activation['index'],
            'convolution_activation_input': activation,
            'tactic': tactic, 'tactic_value': layer.get('TacticValue'),
            'int8_input_and_weight_formats': weights_int8 and activation_int8,
            'integer_tactic_evidence': 'NAMED_INTEGER_TACTIC' if integer_tactic else
                ('TACTIC_NAME_NOT_EXPOSED' if not tactic else 'INTEGER_TACTIC_NOT_IDENTIFIED'),
            'verified_int8_kernel': weights_int8 and activation_int8 and integer_tactic,
            'checks': {'weight_int8': weights_int8, 'activation_input_index_valid': activation['valid'],
                       'activation_input_int8': activation_int8,
                       'integer_tactic': integer_tactic}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--onnx', type=Path,
                        help='Exact source Q/DQ graph; mandatory for multi-source native Conv precision accounting')
    args = parser.parse_args()
    layers = json.loads(args.layers.read_text(encoding='utf-8-sig'))['Layers']
    mapping = json.loads(args.map.read_text(encoding='utf-8-sig'))['mapping']
    mapping_by_node = {item['onnx_node']: item for item in mapping}
    assert len(mapping_by_node) == len(mapping), 'Duplicate source Conv names'
    graph = load_graph_index(args.onnx) if args.onnx else None
    rows, errors = [], []
    for item in mapping:
        # Q/DQ names include the module identity too: only match actual convolutions.
        matches = [layer for layer in layers if 'Convolution' in layer.get('LayerType', '')
                   and node_name_matches(item['onnx_node'], layer['Name'])]
        row = {'module': item['pytorch_module'], 'group': item['group'], 'onnx_node': item['onnx_node'],
               'matching_engine_layers': len(matches), 'verified_int8_kernel': False,
               'int8_input_and_weight_formats': False}
        if len(matches) != 1:
            errors.append('Expected one Conv match for ' + item['pytorch_module'])
        else:
            layer = matches[0]
            # Require quantized weights, quantized activation input, and an integer tactic.
            # Float output is valid for INT8-input weighted operations.
            row.update(inspect_conv_precision(layer))
            if not row['verified_int8_kernel']:
                errors.append('INT8 kernel evidence incomplete for ' + item['pytorch_module'])
        rows.append(row)
    assert len(rows) == 88, 'Unexpected Conv inventory'
    mapped_names = {row['engine_layer'] for row in rows if 'engine_layer' in row}
    unmapped_convs = [layer['Name'] for layer in layers if 'Convolution' in layer.get('LayerType', '')
                     and layer['Name'] not in mapped_names]
    errors.extend('Unmapped engine convolution: ' + name for name in unmapped_convs)
    fusions = []
    for name in sorted(mapped_names):
        members = [row for row in rows if row.get('engine_layer') == name]
        if len(members) > 1:
            layer = next(layer for layer in layers if layer['Name'] == name)
            accounting = account_fused_convs(layer, members, mapping_by_node, graph)
            fusions.append({'engine_layer': name, 'modules': [row['module'] for row in members],
                            'groups': sorted({row['group'] for row in members}), 'accounting': accounting})
            for row in members:
                row['fusion_accounting_status'] = accounting['status']
                if accounting['status'] != 'FUSION_ACCOUNTING_PASS':
                    row['int8_input_and_weight_formats'] = False
                    row['verified_int8_kernel'] = False
            if accounting['status'] != 'FUSION_ACCOUNTING_PASS':
                errors.append('Source Conv precision cannot be attributed through native fusion: ' + name)
    group_counts = {}
    for row in rows:
        group = group_counts.setdefault(row['group'], {'total_convs': 0, 'verified_int8_convs': 0,
                                                       'convs_with_int8_input_and_weight_formats': 0})
        group['total_convs'] += 1
        group['verified_int8_convs'] += int(row['verified_int8_kernel'])
        group['convs_with_int8_input_and_weight_formats'] += int(row['int8_input_and_weight_formats'])
    format_count = sum(row['int8_input_and_weight_formats'] for row in rows)
    formats_complete = format_count == 88 and not unmapped_convs
    report = {'schema': 'rampq.gate05.int8_precision_audit/v2',
              'status': 'ALL_88_INT8_CONVS_VERIFIED' if not errors else 'PRECISION_REVIEW_REQUIRED',
              'format_status': 'ALL_88_CONVS_WITH_INT8_INPUT_AND_WEIGHT_FORMATS' if formats_complete else
                               'INT8_FORMAT_COVERAGE_INCOMPLETE',
              'all_88_int8_formats_verified': formats_complete,
              'layer_file_sha256': hashlib.sha256(args.layers.read_bytes()).hexdigest(),
              'module_map_sha256': hashlib.sha256(args.map.read_bytes()).hexdigest(),
              'source_onnx_sha256': hashlib.sha256(args.onnx.read_bytes()).hexdigest() if args.onnx else None,
              'engine_layers': len(layers), 'source_convs': len(rows),
              'verified_int8_convs': sum(row['verified_int8_kernel'] for row in rows),
              'convs_with_int8_input_and_weight_formats': format_count,
              'convs_without_tactic_name': sum(row.get('integer_tactic_evidence') == 'TACTIC_NAME_NOT_EXPOSED'
                                              for row in rows),
              'groups': group_counts, 'errors': errors, 'mapping': rows,
              'unmapped_engine_convs': unmapped_convs, 'fusions': fusions,
              'scope': 'INT8 input/weight formats and named integer tactics are separate evidence. '
                       'Missing tactic names do not establish integer arithmetic. No full numerical migration or dataset accuracy.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'mapping'}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

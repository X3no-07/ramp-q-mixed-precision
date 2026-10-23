"""Saved-only source/actual floating-path comparison for the frozen Gate0.5 family.

Reads ONNX protobufs and saved detailed TensorRT inspector JSON. It never imports
an inference runtime, executes a model/NMS, or changes a prior-stage artifact.
Missing metadata and unchanged-source floating differences remain explicit gaps.
It does not promote an observed downstream dtype change to a necessary treatment
effect merely because the node is reachable from a quantized convolution.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ('fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5',
                  'r_only_G1_G2', 'q_only_G1')
DRAFT = ROOT / 'artifacts/gate05/20261021_gate05_completion_plan/dataset_acceptance_contract_draft.json'
DRAFT_SHA = 'a848a52ad4acb5db7e5b8ad1223ff27a90ff02cf4e4123e85561f9deb7285f5e'
SHAPE_OPS = {'Reshape', 'Transpose', 'Split', 'Slice', 'Concat', 'Shape',
             'Gather', 'Unsqueeze', 'Squeeze', 'Expand', 'Identity'}
STATIC_OPS = {'Constant', 'ConstantOfShape'}
FOCUS_OPS = {'MatMul', 'Softmax', 'Add', 'Sub', 'Mul', 'Div', 'Sigmoid',
             'Relu', 'MaxPool', 'Resize', 'Pow', 'Sqrt', 'ReduceMean'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protobuf_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def write_new(path, value):
    require(path.parent.is_dir(), 'Output parent must already exist')
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def native_root(configuration):
    if configuration == 'q_all':
        return ROOT / 'artifacts/gate05/20261021_qall_none_native'
    return ROOT / 'artifacts/gate05/20261021_matched_control_family_native' / configuration


def input_identity(path):
    return {'file': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': sha(path)}


def scalar_type(value):
    text = value.lower()
    if 'int8' in text:
        return 'INT8'
    if 'fp16' in text or 'half' in text:
        return 'FP16'
    if 'fp32' in text or 'float' in text:
        return 'FP32'
    if 'int32' in text:
        return 'INT32'
    return 'UNREPORTED_OR_OTHER'


def tensor_row(tensor):
    return {'name': tensor.get('Name'), 'dimensions': tensor.get('Dimensions'),
            'format': tensor.get('Format/Datatype'),
            'scalar_type': scalar_type(tensor.get('Format/Datatype', '')),
            'location': tensor.get('Location')}


def operation_boundary(layer):
    # Actual partition endpoints, not fabricated precision for invisible internals.
    return {side.lower(): [{key: tensor_row(value)[key]
                           for key in ('dimensions', 'format', 'scalar_type')}
                          for value in layer.get(side, [])]
            for side in ('Inputs', 'Outputs')}


def is_boundary_copy(layer):
    name = layer.get('Name', '')
    return (name.startswith('Reformatting CopyNode ') or
            name.startswith('Reformatted Input Tensor ') or
            name.startswith('Reformatted Output Tensor ') or
            (layer.get('LayerType') == 'Reformat' and layer.get('Origin') == 'REFORMAT'))


def token_pattern(name):
    # Do not conflate Mul with Mul_1, Conv with Conv_output_0, or a name in a copy.
    return re.compile(r'(?<![A-Za-z0-9_/.])' + re.escape(name) + r'(?![A-Za-z0-9_/.])')


def activation_pairs(nodes):
    """Prove exact source Conv -> sigmoid(x)*x adjacency; no downstream heuristic."""
    producers = {output: node for node in nodes.values() for output in node.output}
    pairs = {}
    for node in nodes.values():
        if node.op_type != 'Mul' or len(node.input) != 2:
            continue
        for sigmoid_input, direct_input in (node.input, node.input[::-1]):
            sigmoid = producers.get(sigmoid_input)
            conv = producers.get(direct_input)
            if sigmoid is not None and sigmoid.op_type == 'Sigmoid' and list(sigmoid.input) == [direct_input] and \
                    conv is not None and conv.op_type == 'Conv':
                pairs[node.name] = conv.name
                pairs[sigmoid.name] = conv.name
    return pairs


def graph_source_audit(reference, candidate, requests):
    errors = []
    base = {node.name: node for node in reference.graph.node}
    own = {node.name: node for node in candidate.graph.node}
    base_init = {value.name: value for value in reference.graph.initializer}
    own_init = {value.name: value for value in candidate.graph.initializer}
    rows = []
    for name, original in base.items():
        actual = own.get(name)
        if actual is None:
            errors.append('Missing original source node: ' + name)
            continue
        normalized = type(actual)()
        normalized.CopyFrom(actual)
        if original.op_type == 'Conv':
            normalized.ClearField('input')
            normalized.input.extend(original.input)
        matched = normalized.SerializeToString() == original.SerializeToString()
        if not matched:
            errors.append('Unapproved source op/attribute/output change: ' + name)
        rows.append({'name': name, 'op_type': original.op_type,
                     'original_serialized_sha256': protobuf_sha(original),
                     'candidate_serialized_sha256': protobuf_sha(actual),
                     'same_except_allowed_Conv_QDQ_input_edges': matched,
                     'original_inputs': list(original.input), 'candidate_inputs': list(actual.input),
                     'outputs': list(original.output)})
    initializer_rows = []
    for name, original in base_init.items():
        value = own_init.get(name)
        same = value is not None and value.SerializeToString() == original.SerializeToString()
        if not same:
            errors.append('Original weight/bias/constant changed or missing: ' + name)
        initializer_rows.append({'name': name, 'same_serialized_bytes': same,
                                 'source_serialized_sha256': protobuf_sha(original),
                                 'candidate_serialized_sha256': protobuf_sha(value) if value is not None else None})
    for key in ('input', 'output'):
        if [value.SerializeToString() for value in getattr(reference.graph, key)] != \
                [value.SerializeToString() for value in getattr(candidate.graph, key)]:
            errors.append('Input/output declaration changed: ' + key)
    quantized = {row['onnx_node'] for row in requests if row['precision'] == 'INT8'}
    require(len(requests) == 88 and len({r['onnx_node'] for r in requests}) == 88,
            'Exactly 88 source requests required')
    quantizer_rows = []
    for request in requests:
        conv = own[request['onnx_node']]
        original = base[conv.name]
        if conv.name not in quantized:
            if conv.SerializeToString() != original.SerializeToString():
                errors.append('Restored Conv changed beyond source precision treatment: ' + conv.name)
            continue
        producers = {output: node for node in candidate.graph.node for output in node.output}
        try:
            dq = producers[conv.input[0]]
            clip = producers[dq.input[0]]
            q = producers[clip.input[0]]
            wdq = producers[conv.input[1]]
            wq = producers[wdq.input[0]]
            valid = (q.op_type == wq.op_type == 'QuantizeLinear' and clip.op_type == 'Clip' and
                     dq.op_type == wdq.op_type == 'DequantizeLinear' and
                     q.input[0] == original.input[0] and wq.input[0] == original.input[1] and
                     list(q.input[1:]) == list(dq.input[1:]) and
                     list(wq.input[1:]) == list(wdq.input[1:]))
            require(valid, 'QDQ source provenance differs')
            quantizer_rows.append({'conv': conv.name,
                'activation_nodes': [q.name, clip.name, dq.name],
                'weight_nodes': [wq.name, wdq.name],
                'source_activation': q.input[0], 'source_weight': wq.input[0],
                'initializers': {name: protobuf_sha(own_init[name])
                    for name in list(q.input[1:]) + list(clip.input[1:]) + list(wq.input[1:])}})
        except (KeyError, IndexError, ValueError) as error:
            errors.append('Cannot authenticate quantizer for ' + conv.name + ': ' + str(error))
    return {'errors': errors, 'base_node_count': len(base), 'base_initializer_count': len(base_init),
            'source_nodes': rows, 'source_initializers': initializer_rows,
            'quantizers': quantizer_rows, 'quantized_source_convs': sorted(quantized),
            'added_nodes': [{'name': n.name, 'op_type': n.op_type, 'sha256': protobuf_sha(n)}
                            for n in candidate.graph.node if n.name not in base],
            'added_initializers': {name: protobuf_sha(value) for name, value in own_init.items() if name not in base_init}}


def load_member(configuration, reference, evidence):
    import onnx
    root = native_root(configuration)
    bundle = root / 'native_evidence'
    audit_path = root / 'precision_audit.json'
    graph_path = bundle / ('source_bundle/model.onnx' if configuration == 'q_all' else 'model.onnx')
    spec_path = bundle / ('source_bundle/spec.json' if configuration == 'q_all' else 'spec.json')
    require(audit_path.is_file() and graph_path.is_file(), 'Native evidence incomplete for ' + configuration)
    builds = [p.parent for p in bundle.glob('results/*/build_result.json')]
    require(len(builds) == 1, 'Exactly one saved build required for ' + configuration)
    build = builds[0]
    audit, spec = read(audit_path), read(spec_path)
    layers_path, request_path, result_path = (build / name for name in
        ('engine_layers.json', 'build_request.json', 'build_result.json'))
    engine_path, bindings_path = build / 'model.engine', build / 'bindings.json'
    for path in (audit_path, graph_path, spec_path, layers_path, request_path, result_path, engine_path, bindings_path):
        evidence[path] = sha(path)
    require(audit['requested_conv_formats_complete'] is True and not audit['errors'] and
            not audit['unmapped_engine_convs'], 'Source Conv format audit incomplete for ' + configuration)
    for key, path in [('onnx_sha256', graph_path), ('engine_sha256', engine_path),
                      ('layer_file_sha256', layers_path), ('build_request_sha256', request_path),
                      ('build_result_sha256', result_path)]:
        require(audit[key] == evidence[path], 'Precision audit input identity mismatch: ' + key)
    request, result, inspector = read(request_path), read(result_path), read(layers_path)
    require(result['status'] == 'BUILT_AND_DESERIALIZED_BUILD_ONLY' and
            result['graph_sha256'] == evidence[graph_path] and result['engine_sha256'] == evidence[engine_path],
            'Not a successful authenticated build')
    flags = request['flags']['get_flag']
    for name, enabled in [('FP16', True), ('INT8', configuration != 'fp16_control'),
                          ('TF32', False), ('PREFER_PRECISION_CONSTRAINTS', False), ('OBEY_PRECISION_CONSTRAINTS', False)]:
        require(flags[name] == {'supported': True, 'enabled': enabled}, 'Unexpected actual flag: ' + name)
    require(request['explicit_layer_precision_constraint_count'] ==
            request['explicit_layer_output_type_constraint_count'] == request['layer_constraint_reset_count'] == 0 and
            request['parser_constraints_before_after_identical'] is True,
            'Manual source precision/output intervention or parser reset')
    requests = spec['requests']
    graph = onnx.load(str(graph_path), load_external_data=False)
    source = graph_source_audit(reference, graph, requests)
    nodes = {node.name: node for node in reference.graph.node}
    pattern = {name: token_pattern(name) for name in nodes}
    output_producer = {output: name for name, node in nodes.items() for output in node.output}
    activation_owner = activation_pairs(nodes)
    layers = inspector['Layers']
    mapped = defaultdict(list)
    observations = defaultdict(list)
    actual_rows = []
    weighted_source_by_layer = defaultdict(set)
    for row in audit['mapping']:
        weighted_source_by_layer[row['engine_layer']].add(row['onnx_node'])
    for index, layer in enumerate(layers):
        name = layer.get('Name', '')
        methods = defaultdict(set)
        if not is_boundary_copy(layer):
            for source_name, matcher in pattern.items():
                if matcher.search(name):
                    methods[source_name].add('EXACT_SOURCE_NAME_TOKEN_IN_OPERATION_LAYER')
            for tensor in layer.get('Outputs', []):
                producer = output_producer.get(tensor.get('Name'))
                if producer:
                    methods[producer].add('EXACT_SOURCE_OUTPUT_TENSOR')
            if layer.get('Activation') in ('SWISH', 'SILU'):
                conv_members = weighted_source_by_layer.get(name, set())
                for source_name, conv in activation_owner.items():
                    if conv in conv_members:
                        methods[source_name].add('EXACT_SOURCE_CONV_SILU_ADJACENCY_AND_REPORTED_ACTIVATION')
        for source_name, reasons in methods.items():
            mapped[source_name].append({'layer_index': index, 'methods': sorted(reasons)})
        for side in ('Inputs', 'Outputs'):
            for tensor in layer.get(side, []):
                observations[tensor.get('Name')].append({'layer_index': index, 'side': side, **tensor_row(tensor)})
        actual_rows.append({'layer_index': index, 'name': name, 'type': layer.get('LayerType'),
            'source_members': sorted(methods), 'weighted_source_members': sorted(weighted_source_by_layer.get(name, set())),
            'is_boundary_copy': is_boundary_copy(layer), 'boundary': operation_boundary(layer),
            'reported_compute_precision': {key: value for key, value in layer.items()
                if any(part in key.lower() for part in ('precision', 'compute', 'accumul'))},
            'tactic_name': layer.get('TacticName'), 'tactic_value': layer.get('TacticValue'),
            'raw_inspector_layer': layer})
    source_actual = {}
    for name, node in nodes.items():
        entries = mapped[name]
        source_actual[name] = {'name': name, 'op_type': node.op_type,
            'scope': 'SHAPE_OR_VIEW' if node.op_type in SHAPE_OPS else
                     'STATIC_LITERAL' if node.op_type in STATIC_OPS else
                     'WEIGHTED_SOURCE' if node.op_type == 'Conv' else 'FLOAT_ARITHMETIC',
            'mapped_layers': entries,
            'source_outputs_observed': {output: observations[output] for output in node.output},
            'source_inputs_observed': {value: observations[value] for value in node.input if value in observations},
            'source_direct_silu_conv': activation_owner.get(name),
            'internal_values_not_visible_from_fused_boundary': any(
                len(actual_rows[item['layer_index']]['source_members']) > 1 for item in entries)}
    return {'configuration': configuration, 'native_root': root.relative_to(ROOT).as_posix(),
        'identities': {str(path.relative_to(ROOT).as_posix()): expected
                       for path, expected in evidence.items() if root in path.parents},
        'source_audit': source, 'actual_flags': flags, 'raw_config_flags': request['flags']['raw_config_flags'],
        'source_requests': requests, 'source_to_actual': source_actual, 'actual_layers': actual_rows,
        'layer_type_counts_descriptive_only': dict(Counter(row['type'] for row in actual_rows)),
        'bindings': read(bindings_path), 'weighted_fusion_accounting_from_unchanged_auditor': audit['fusions'],
        'precision_audit_status': audit['status'], 'named_integer_tactic_status_not_changed': audit['strict_int8_status'],
        'source_semantics_errors': source['errors'],
        'source_weighted_formats_complete_is_not_float_comparability': True}


def mapped_boundary(member, source):
    rows = [member['actual_layers'][entry['layer_index']] for entry in source['mapped_layers']]
    # Keep all partitions without insisting that partition counts match.
    return rows, sorted({canonical(row['boundary']) for row in rows})


def compare_pair(left, right):
    changed = {a['onnx_node'] for a, b in zip(left['source_requests'], right['source_requests'])
               if a['precision'] != b['precision']}
    require([r['onnx_node'] for r in left['source_requests']] ==
            [r['onnx_node'] for r in right['source_requests']], 'Request ordering differs')
    # Remaining calibration constants must be byte-identical; removing a quantizer is allowed.
    quantizers_a = {r['conv']: r for r in left['source_audit']['quantizers']}
    quantizers_b = {r['conv']: r for r in right['source_audit']['quantizers']}
    scale_mismatches = [name for name in quantizers_a.keys() & quantizers_b.keys()
                        if quantizers_a[name] != quantizers_b[name]]
    rows, gaps = [], []
    for name, source_a in left['source_to_actual'].items():
        source_b = right['source_to_actual'][name]
        mapped_a, signature_a = mapped_boundary(left, source_a)
        mapped_b, signature_b = mapped_boundary(right, source_b)
        same_boundary = bool(mapped_a and mapped_b and signature_a == signature_b)
        implementation = {'left': [{'type': row['type'], 'source_members': row['source_members'],
            'tactic_name': row['tactic_name'], 'tactic_value': row['tactic_value'],
            'reported_compute_precision': row['reported_compute_precision']} for row in mapped_a],
            'right': [{'type': row['type'], 'source_members': row['source_members'],
            'tactic_name': row['tactic_name'], 'tactic_value': row['tactic_value'],
            'reported_compute_precision': row['reported_compute_precision']} for row in mapped_b]}
        direct = sorted({conv for row in mapped_a + mapped_b for conv in row['weighted_source_members'] if conv in changed})
        direct_silu = source_a['source_direct_silu_conv']
        if direct_silu in changed:
            direct = sorted(set(direct + [direct_silu]))
        if source_a['op_type'] == 'Conv':
            category = 'DECLARED_WEIGHTED_QUANTIZATION_OR_RESTORATION' if name in changed else 'UNCHANGED_WEIGHTED_REQUEST_SEPARATELY_AUDITED'
        elif source_a['scope'] == 'STATIC_LITERAL':
            category = 'EXACT_SOURCE_LITERAL_MAY_BE_FOLDED'
        elif same_boundary:
            category = 'SAME_REPORTED_PARTITION_FLOAT_IO_METADATA'
        elif source_a['scope'] == 'SHAPE_OR_VIEW':
            category = 'SHAPE_VIEW_STORAGE_PROPAGATION_OR_FOLD_RECORDED_NO_ARITHMETIC_EQUALITY_CLAIM'
        elif direct:
            category = 'DIRECT_CHANGED_WEIGHTED_FUSION_OR_EXACT_SILU_ADJACENCY'
        elif not mapped_a or not mapped_b:
            category = 'MAPPING_OR_FORMAT_UNAVAILABLE'
        else:
            category = 'UNCONTROLLED_UNCHANGED_SOURCE_FLOAT_PATH_DIFFERENCE'
        gap = category in ('MAPPING_OR_FORMAT_UNAVAILABLE', 'UNCONTROLLED_UNCHANGED_SOURCE_FLOAT_PATH_DIFFERENCE')
        record = {'source_node': name, 'op_type': source_a['op_type'], 'scope': source_a['scope'],
            'category': category, 'primary_comparability_gap': gap,
            'same_reported_partition_IO_metadata': same_boundary,
            'left_mapped_layers': [item['layer_index'] for item in source_a['mapped_layers']],
            'right_mapped_layers': [item['layer_index'] for item in source_b['mapped_layers']],
            'left_boundaries': [row['boundary'] for row in mapped_a],
            'right_boundaries': [row['boundary'] for row in mapped_b],
            'direct_changed_Conv_evidence': direct,
            'source_unchanged_is_not_actual_float_equality': True,
            'implementation_or_partition_differences_descriptive_only': implementation,
            'same_accumulator_or_rounding_proven': False,
            'downstream_reachability_used_as_automatic_linkage': False}
        rows.append(record)
        if gap:
            gaps.append({'source_node': name, 'op_type': source_a['op_type'], 'category': category})
    eligibility = ('SOURCE_IDENTITY_OR_REMAINING_QUANTIZER_MISMATCH' if scale_mismatches or
        left['source_semantics_errors'] or right['source_semantics_errors'] else
        'UNCONTROLLED_UNRELATED_OR_UNBRIDGED_FLOAT_PATH_DIFFERENCE' if any(
            row['category'] == 'UNCONTROLLED_UNCHANGED_SOURCE_FLOAT_PATH_DIFFERENCE' for row in rows) else
        'MAPPING_OR_FORMAT_UNAVAILABLE' if gaps else
        'MATCHED_SOURCE_AND_REPORTED_FLOAT_METADATA_WITH_DOCUMENTED_DIRECT_QUANTIZATION_LINKED_FUSION')
    return {'left': left['configuration'], 'right': right['configuration'],
        'changed_source_Conv_requests': sorted(changed), 'remaining_same_quantizer_mismatches': scale_mismatches,
        'eligibility': eligibility, 'primary_authenticity_eligible_from_this_audit': not gaps and not scale_mismatches and
            not left['source_semantics_errors'] and not right['source_semantics_errors'],
        'category_counts': dict(Counter(row['category'] for row in rows)), 'gaps': gaps,
        'source_operation_comparisons': rows,
        'tactic_fusion_layer_count_equality_is_not_a_pass_or_fail_rule': True,
        'pure_G1_causal_identification': 'NOT_ESTABLISHED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configurations', nargs='+', choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(len(args.configurations) == len(set(args.configurations)) and 'fp16_control' in args.configurations,
            'Unique members including the immutable floating source anchor required')
    require(not args.output.exists(), 'New exclusive review path required; prior snapshots are immutable')
    require(sha(DRAFT) == DRAFT_SHA, 'Original acceptance draft changed')
    import onnx
    reference_path = native_root('fp16_control') / 'native_evidence/model.onnx'
    reference = onnx.load(str(reference_path), load_external_data=False)
    evidence = {DRAFT: DRAFT_SHA, reference_path: sha(reference_path)}
    members = [load_member(cfg, reference, evidence) for cfg in args.configurations]
    pairs = [compare_pair(left, right) for left, right in itertools.combinations(members, 2)]
    require(all(sha(path) == expected for path, expected in evidence.items()), 'Saved input changed during audit')
    all_six = tuple(args.configurations) == CONFIGURATIONS
    all_pairs = all(pair['primary_authenticity_eligible_from_this_audit'] for pair in pairs)
    report = {'schema': 'rampq.gate05.float_fusion_comparability_review/v1',
        'client_UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'COMPARABILITY_GAPS_PRESERVED' if not all_pairs else
                  'PARTIAL_FAMILY_REPORTED_METADATA_CHECK_COMPLETE' if not all_six else
                  'ALL_SIX_SOURCE_AND_REPORTED_FLOAT_METADATA_CHECK_COMPLETE',
        'author_role': 'Independent saved-only float/fusion review; authored family preparation but not native builder/precision auditor',
        'audit_tool': input_identity(Path(__file__).resolve()), 'onnx_parser_version': onnx.__version__,
        'acceptance_draft_sha256': DRAFT_SHA, 'original_draft_modified': False,
        'configuration_order': args.configurations, 'all_six_members_available': all_six,
        'all_six_float_comparability_eligible': all_six and all_pairs,
        'actual_flag_policy_difference': {'fp16_control_INT8': False, 'other_five_INT8': True,
            'frozen_preparation_native_policy': 'Declared before dataset; not silently equal to draft same_flags_even_if_unused proposal',
            'same_flags_claim': False, 'same_flags_difference_alone_waived_by_this_audit': False},
        'must_match_or_be_bridged_preserved': read(DRAFT)['floating_and_fusion_comparability']['must_match_or_be_bridged'],
        'no_prespecified_float_bridge_executed_or_invented': True,
        'mapping_scope': {'exact_names': 'Token-delimited operation-layer names, excluding copy-name mentions',
            'outputs': 'Exact source output tensor when preserved',
            'implicit_activation': 'Only exact source Conv->Sigmoid/Mul adjacency plus actual reported SWISH/SILU',
            'invisible_fused_values': 'Only partition endpoints reported; no internal precision invention',
            'shape_view_storage': 'Recorded separately; never claimed identical arithmetic from names',
            'tactics': 'Raw name/ID or missing; same I/O does not establish equal accumulator or summation',
            'automatic_downstream_linkage': False},
        'members': members, 'pairwise_comparisons': pairs,
        'saved_inputs': [input_identity(path) for path in sorted(evidence, key=str)],
        'saved_inputs_unchanged_after_read': True,
        'execution': {'new_CPU_model_calls': 0, 'native_calls': 0, 'NMS_calls': 0,
            'dataset_prediction_streams_read': 0, 'SSH_calls': 0,
            'inference_runtime_imported': any(name in sys.modules for name in ('torch', 'onnxruntime', 'tensorrt'))},
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
        'target_recovery_accuracy_or_pure_G1_causal_claim': False}
    require(report['execution']['inference_runtime_imported'] is False, 'Unexpected inference runtime import')
    write_new(args.output, report)
    print(json.dumps({'status': report['status'], 'output': str(args.output.resolve()),
        'sha256': sha(args.output), 'members': len(members), 'pairs': len(pairs),
        'pair_eligibility': [{'left': p['left'], 'right': p['right'], 'eligibility': p['eligibility'],
                              'gap_count': len(p['gaps'])} for p in pairs], 'model_calls': 0, 'NMS_calls': 0}))


if __name__ == '__main__':
    main()

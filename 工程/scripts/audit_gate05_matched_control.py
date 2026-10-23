"""Audit saved matched-family source/native evidence without model execution.

Reuse the original native format, FP16 output, fusion, weight attribution and
named-integer-tactic criteria unchanged. Only new provenance and the source
code-Clip/mixed-mask interpretation differ. Exit0=source pass or complete native
formats/tactics; exit2=source/provenance/format failure; exit3=named tactic gaps.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import jetson_build_matched_control as contract
import audit_gate05_corrected_qall_candidate as corrected

base = corrected.base
BUILDER_SHA256 = '4d2ee7d283abf6cd75b6d76e58870f79f11392fa06eaef37936fab6d7bafca34'
CORRECTED_AUDITOR_SHA256 = '4a8078effd58b214874cd25ba1d0ec1f9c0e684ab39f5df707ef46556ac9912e'
BASE_AUDITOR_SHA256 = '40420902d6fb3fd5d6b6278d64bbaa4b44c839bdae69da73dbaeb3e15cf66f6a'
ORIGINAL_NATIVE_AUDIT = base.audit_native_layers
activation_trace = corrected.activation_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def tool_identity():
    return {'auditor_file': Path(__file__).name, 'auditor_sha256': sha(Path(__file__)),
            'matched_builder_contract_sha256': sha(Path(contract.__file__)),
            'corrected_activation_trace_source_sha256': sha(Path(corrected.__file__)),
            'base_native_auditor_sha256': sha(Path(base.__file__)),
            'native_function_reused': 'audit_gate05_precision_candidate.audit_native_layers',
            'native_formats_tactics_and_fusion_criteria': 'UNCHANGED_BASE_IMPLEMENTATION',
            'FP16_output_format_requirement_preserved': True,
            'SASS_evidence_not_used_to_rewrite_named_tactic_status': True}


def check_graph_requests(graph, requests, module_mapping):
    """Retain source mask/weight/scale checks; code-Clip applies only to INT8 members."""
    import numpy as np
    import onnx
    from onnx import numpy_helper
    errors = []
    nodes = graph['nodes']
    if {name for name, node in nodes.items() if node.op_type == 'Conv'} != {row['onnx_node'] for row in requests}:
        errors.append('Graph Conv inventory differs from requested source inventory')
    original = {row['onnx_node']: row for row in module_mapping}
    if len(original) != 88 or len(module_mapping) != 88 or set(original) != {row['onnx_node'] for row in requests}:
        errors.append('Module map inventory differs from 88 unique requests')
    if len(requests) != 88 or len({row['onnx_node'] for row in requests}) != 88:
        errors.append('Requested inventory differs from 88 unique requests')
    if any(row['precision'] not in ('INT8', 'FP16') for row in requests):
        errors.append('Unsupported desired source precision')
    for row in requests:
        try:
            frozen = original[row['onnx_node']]
            assert row['module'] == frozen['pytorch_module'] and row['group'] == frozen['group']
            assert row['weight_input'] == frozen['weight_input']
            conv = nodes[row['onnx_node']]
            weight_name = base.original_weight_name(conv, row, graph)
            assert weight_name in graph['initializers']
            if row['precision'] == 'FP16':
                assert conv.input[0] == row['original_activation_input']
            else:
                trace = activation_trace(conv, graph)
                assert trace['source_tensor'] == row['original_activation_input']
                prefix = frozen['qdq_prefix']
                dq = graph['producers'][conv.input[0]]
                clip = graph['producers'][dq.input[0]]
                quantize = graph['producers'][clip.input[0]]
                assert dq.name == prefix + '_ActivationDQ' and clip.name == prefix + '_Clip'
                assert quantize.name == prefix + '_ActivationQ'
                assert list(quantize.input[1:]) == list(dq.input[1:]) == [prefix + '_a_scale', prefix + '_a_zero']
                assert list(clip.input[1:]) == [prefix + '_clip_min', prefix + '_clip_max']
                for field, suffix in [('scale', 'a_scale'), ('zero', 'a_zero'),
                                       ('clip_min', 'clip_min'), ('clip_max', 'clip_max')]:
                    assert trace['constants'][field] == base.signature(graph['initializers'][prefix + '_' + suffix])
                weight_dq = graph['producers'][conv.input[1]]
                weight_q = graph['producers'][weight_dq.input[0]]
                assert weight_dq.op_type == 'DequantizeLinear' and weight_q.op_type == 'QuantizeLinear'
                assert len(weight_q.input) == len(weight_dq.input) == 3
                assert len(weight_q.output) == len(weight_dq.output) == 1
                assert list(weight_q.input[1:]) == list(weight_dq.input[1:]) == [prefix + '_w_scale', prefix + '_w_zero']
                assert [(attribute.name, attribute.i) for attribute in weight_q.attribute] == [('axis', 0)]
                assert [(attribute.name, attribute.i) for attribute in weight_dq.attribute] == [('axis', 0)]
                assert weight_q.attribute[0].type == weight_dq.attribute[0].type == onnx.AttributeProto.INT
                expected_scale = np.asarray(frozen['activation_scale'], dtype=np.float32)
                for suffix, expected in [('a_scale', expected_scale), ('a_zero', np.asarray(0, dtype=np.int8)),
                                         ('clip_min', np.asarray(-127, dtype=np.int8)), ('clip_max', np.asarray(127, dtype=np.int8)),
                                         ('w_scale', np.asarray(frozen['weight_scale'], dtype=np.float32)),
                                         ('w_zero', np.zeros(len(frozen['weight_scale']), dtype=np.int8))]:
                    actual = numpy_helper.to_array(graph['initializers'][prefix + '_' + suffix])
                    assert actual.dtype == expected.dtype and actual.shape == expected.shape
                    assert actual.tobytes() == expected.tobytes(), suffix
        except (KeyError, IndexError, AssertionError, ValueError, TypeError) as error:
            errors.append('Source request/quantizer mismatch ' + row['onnx_node'] + ': ' + str(error))
    return errors


@contextmanager
def source_trace_extension():
    original_trace, original_check = base.activation_trace, base.check_graph_requests
    assert base.audit_native_layers is ORIGINAL_NATIVE_AUDIT
    try:
        base.activation_trace = activation_trace
        base.check_graph_requests = check_graph_requests
        yield
    finally:
        base.activation_trace, base.check_graph_requests = original_trace, original_check
        assert base.audit_native_layers is ORIGINAL_NATIVE_AUDIT


def validate_source(root):
    import onnx
    assert sha(Path(contract.__file__)) == BUILDER_SHA256, 'Locked builder contract changed'
    assert sha(Path(corrected.__file__)) == CORRECTED_AUDITOR_SHA256, 'Corrected source auditor changed'
    assert sha(Path(base.__file__)) == BASE_AUDITOR_SHA256, 'Original native auditor changed'
    assert ORIGINAL_NATIVE_AUDIT.__module__ == base.__name__ and \
        Path(ORIGINAL_NATIVE_AUDIT.__code__.co_filename).resolve() == Path(base.__file__).resolve(), 'Native criterion function was replaced'
    spec, parent_spec, native_protocol = contract.validate_inputs(root)
    graph = base.graph_index(root / spec['graph_file'])
    parent_graph = base.graph_index(root / 'parent_bundle/model.onnx')
    rewrite = read(root / 'semantic_rewrite_audit.json')
    for prefix, indexed in (('candidate', graph), ('parent', parent_graph)):
        assert {name: hashlib.sha256(node.SerializeToString()).hexdigest() for name, node in indexed['nodes'].items()} == \
            rewrite[prefix + '_nodes_serialized_sha256']
        assert {name: hashlib.sha256(value.SerializeToString()).hexdigest() for name, value in indexed['initializers'].items()} == \
            rewrite[prefix + '_initializers_serialized_sha256']
    onnx.checker.check_model(str(root / spec['graph_file']), full_check=True)
    mapping = read(root / 'parent_bundle/module_map.json')['mapping']
    errors = check_graph_requests(graph, spec['requests'], mapping)
    return spec, parent_spec, native_protocol, graph, mapping, errors


def constraint_values(rows):
    return [{**{key: row[key] for key in ('index', 'name', 'precision_is_set', 'precision', 'precision_value_error') if key in row},
        'outputs': [{key: value[key] for key in ('index', 'output_type_is_set', 'requested_output_type', 'requested_output_type_error') if key in value}
                    for value in row['output_types']]} for row in rows]


def constraint_inventory(rows):
    precisions, outputs, errors = [], [], []
    for row in rows:
        name = row['name']
        state = row['precision_is_set']
        if not state.get('supported') or 'precision_value_error' in row:
            errors.append({'name': name, 'field': 'precision', 'record': row})
        if state.get('value'):
            assert 'precision' in row
            precisions.append({'name': name, 'precision': row['precision']})
        else:
            assert 'precision' not in row
        for value in row['output_types']:
            state = value['output_type_is_set']
            if not state.get('supported') or 'requested_output_type_error' in value or 'tensor_query_error' in value:
                errors.append({'name': name, 'field': 'output', 'record': value})
            if state.get('value'):
                assert 'requested_output_type' in value
                outputs.append({'name': name, 'index': value['index'], 'requested_output_type': value['requested_output_type']})
            else:
                assert 'requested_output_type' not in value
    return {'parsed_layers': len(rows), 'precision_requests': precisions, 'output_type_requests': outputs,
            'precision_request_count': len(precisions), 'output_type_request_count': len(outputs), 'query_errors': errors}


def bindings_signature(rows):
    assert len(rows) == len({row['name'] for row in rows})
    return sorted((row['name'], row['input'], tuple(row['shape']), row['dtype']) for row in rows)


def validate_native_provenance(root, build, spec):
    result = read(build / 'build_result.json')
    identity = read(build / 'tool_hashes.json')
    expected = {'builder_file': spec['native_builder_file'], 'builder_sha256': BUILDER_SHA256,
                'helper_relative_path': contract.HELPER_RELATIVE_PATH, 'helper_sha256': contract.HELPER_SHA256,
                'native_protocol_sha256': contract.PROTOCOL_SHA256,
                'family_protocol_sha256': contract.FAMILY_PROTOCOL_SHA256,
                'spec_sha256': sha(root / 'spec.json'), 'manifest_sha256': sha(root / 'manifest.json'),
                'parent_spec_sha256': sha(root / 'parent_bundle/spec.json'),
                'parent_manifest_sha256': sha(root / 'parent_bundle/manifest.json'),
                'graph_sha256': sha(root / 'model.onnx'), 'cpu_equivalence_sha256': sha(root / 'cpu_equivalence.json'),
                'cpu_expected_outputs_sha256': sha(root / 'cpu_expected_outputs.npz')}
    assert identity == expected == result['tool_identity']
    assert result['schema'] == 'rampq.gate05.matched_control_build_result/v1'
    assert result['configuration'] == spec['configuration'] and result['graph_sha256'] == spec['graph_sha256']
    assert result['legacy_parent_precision_constraints'] == result['legacy_source_precision_constraints'] == 'obey'
    assert result['runtime_precision_constraints'] == 'none' and result['separate_new_control_not_legacy_attempt'] is True
    assert result['new_CPU_execution_runs'] == result['native_inference_executions'] == 0
    assert result['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and result['gate05_status'] == 'INCOMPLETE'
    assert result['quantized_conv_count'] == spec['quantized_conv_count']
    assert result['desired_restored_fp16_conv_count'] == spec['desired_restored_fp16_conv_count']
    if not (build / 'build_request.json').exists():
        assert result['status'] == 'BUILD_OR_CONTRACT_VALIDATION_FAILED'
        return result, {}, {'status': 'FAILURE_BEFORE_FROZEN_PARSED_REQUEST'}
    request = read(build / 'build_request.json')
    assert request['schema'] == 'rampq.gate05.matched_control_build_request/v1' and request['tool_identity'] == identity
    for item in (result, request):
        assert item['configuration'] == spec['configuration'] and item['graph_sha256'] == spec['graph_sha256']
        assert item['runtime_precision_constraints'] == 'none' and item['legacy_source_precision_constraints'] == 'obey'
        assert item['maximum_native_builds'] == 1 and item['max_build_seconds'] == 7200
        assert item['preparation_CPU_model_runs'] == spec['new_CPU_execution_runs_this_member'] and item['new_CPU_execution_runs'] == 0
        assert item['desired_FP16_mask_is_not_adoption_proof'] is True
        assert item['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and item['gate05_status'] == 'INCOMPLETE'
    count = spec['quantized_conv_count']
    flags = request['flags']['get_flag']
    for name, enabled in (('INT8', count > 0), ('FP16', True), ('TF32', False),
                           ('PREFER_PRECISION_CONSTRAINTS', False), ('OBEY_PRECISION_CONSTRAINTS', False)):
        assert flags[name] == {'supported': True, 'enabled': enabled}, name
    assert request['workspace_bytes_actual'] == 1024 ** 3 and request['avg_timing_iterations_actual'] == 1
    assert request['profiling_verbosity_actual'] == 'ProfilingVerbosity.DETAILED'
    assert request['explicit_layer_precision_constraint_count'] == request['explicit_layer_output_type_constraint_count'] == \
        request['layer_constraint_reset_count'] == 0
    assert request['parser_constraints_before_after_identical'] is True and request['native_int8_QDQ_defaults_preserved'] is True
    assert request['source_conv_count'] == 88 and request['desired_source_requests'] == spec['requests']
    assert request['legacy_parent_spec_sha256'] == sha(root / 'parent_bundle/spec.json')
    before, after = read(build / 'parsed_constraints_before.json'), read(build / 'parsed_constraints_after.json')
    assert constraint_values(before) == constraint_values(after)
    inventory = constraint_inventory(before)
    assert not inventory['query_errors'], inventory['query_errors']
    clip_before = read(build / 'parsed_clip_attributes_before_correction.json')
    clip_after = read(build / 'parsed_clip_attributes_after_correction.json')
    assert clip_before['count'] == clip_after['count'] == len(clip_before['layers']) == len(clip_after['layers']) == count
    original = {row['name']: row for row in clip_before['layers']}
    corrections = {row['name']: row for row in clip_after['layers']}
    assert len(original) == len(corrections) == count and set(original) == set(corrections) == set(spec['clip_names'])
    for name, row in corrections.items():
        assert row['before'] == {key: original[name][key] for key in ('alpha', 'beta')}
        assert row['after'] == {'alpha': -127.0, 'beta': 127.0}
    assert request['explicit_clip_importer_corrections'] == clip_after['layers']
    assert bindings_signature(request['bindings_in_network_order']) == bindings_signature(spec['expected_binding_contract'])
    assert all(row['allowed_formats'] == 1 for row in request['bindings_in_network_order'])
    parsed_convs = [row['name'] for row in read(build / 'parsed_layers_before.json') if row['type'] == 'LayerType.CONVOLUTION']
    assert len(parsed_convs) == 88 and set(parsed_convs) == {row['onnx_node'] for row in spec['requests']}
    return result, request, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', '--control-root', dest='root', type=Path, required=True)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-source-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output:
        output = args.output.resolve()
        assert output.parent.is_dir() and not output.exists() and root not in output.parents, 'Choose a fresh report outside evidence'
    before = {path.relative_to(root).as_posix(): sha(path) for path in root.rglob('*') if path.is_file()}
    spec, parent, native, graph, mapping, errors = validate_source(root)
    metadata = {'audit_tool_identity': tool_identity(), 'configuration': spec['configuration'], 'source_convs': 88,
                'quantized_conv_count': spec['quantized_conv_count'], 'fp16_conv_count': spec['desired_restored_fp16_conv_count'],
                'spec_sha256': sha(root / 'spec.json'), 'source_spec_sha256': sha(root / 'spec.json'),
                'manifest_sha256': sha(root / 'manifest.json'), 'onnx_sha256': spec['graph_sha256'],
                'native_protocol_sha256': contract.PROTOCOL_SHA256, 'family_protocol_sha256': contract.FAMILY_PROTOCOL_SHA256,
                'parent_spec_sha256': sha(root / 'parent_bundle/spec.json'),
                'parent_manifest_sha256': sha(root / 'parent_bundle/manifest.json'),
                'module_map_sha256': sha(root / 'parent_bundle/module_map.json'),
                'semantic_rewrite_audit_sha256': spec['semantic_rewrite_audit_sha256'],
                'cpu_equivalence_sha256': spec['cpu_equivalence_sha256'],
                'cpu_expected_outputs_sha256': spec['cpu_expected_outputs_sha256'],
                'legacy_parent_precision_constraints': 'obey', 'runtime_precision_constraints': 'none',
                'activation_quantizer_topology': spec['activation_quantizer_topology'],
                'native_formats_tactics_and_fusion_criteria_changed': False, 'FP16_output_requirement_changed': False,
                'prior_depthwise_integer_gaps_waived': False, 'new_CPU_model_or_native_or_SSH_execution': False,
                'original_complete_model_guard': 'FAILED_UNCHANGED', 'dataset_evaluation_run': False, 'gate05_status': 'INCOMPLETE',
                'numerical_migration_pass_claim': False, 'accuracy_pass_claim': False, 'performance_pass_claim': False}
    code = 0
    if args.check_source_only:
        report = {'schema': 'rampq.gate05.matched_control_source_audit/v1',
                  'status': 'CORRECTED_SOURCE_STRUCTURE_VERIFIED' if not errors else 'SOURCE_STRUCTURE_REVIEW_REQUIRED',
                  'source_structure_errors': errors, **metadata}
        code = 2 if errors else 0
    else:
        assert args.build is not None, '--build required unless --check-source-only'
        build = args.build.resolve()
        assert root in build.parents and build.is_dir() and not errors, errors
        result, request, inventory = validate_native_provenance(root, build, spec)
        if result['status'] != 'BUILT_AND_DESERIALIZED_BUILD_ONLY':
            assert result['status'] == 'BUILD_OR_CONTRACT_VALIDATION_FAILED'
            report = {'schema': 'rampq.gate05.matched_control_precision_audit/v1',
                      'status': 'NATIVE_BUILD_FAILED_NO_ACCEPTED_ENGINE_AUDIT', 'requested_conv_formats_complete': False,
                      'strict_int8_status': 'NOT_EVALUATED_NO_SUCCESSFUL_ENGINE', 'native_build_result': result,
                      'parser_constraint_inventory': inventory, **metadata}
            code = 2
        else:
            assert sha(build / 'model.engine') == result['engine_sha256'] and \
                (build / 'model.engine').stat().st_size == result['engine_bytes'] and result['engine_execution_context_created'] is False
            bindings = read(build / 'bindings.json')['bindings']
            assert bindings == result['bindings'] and bindings_signature(bindings) == bindings_signature(spec['expected_binding_contract'])
            assert all(row['format'] == 'TensorFormat.LINEAR' for row in bindings)
            layer_path = build / 'engine_layers.json'
            inspector = read(layer_path)
            layers = inspector['Layers'] if isinstance(inspector, dict) else inspector
            convs = [row for row in layers if 'Convolution' in row.get('LayerType', '')]
            assert convs == read(build / 'conv_rows.json')
            with source_trace_extension():
                report = ORIGINAL_NATIVE_AUDIT(layers, spec['requests'], graph, mapping)
            assert report['quantized_conv_count'] == spec['quantized_conv_count'] and \
                report['fp16_conv_count'] == spec['desired_restored_fp16_conv_count']
            report.update(metadata)
            report.update(schema='rampq.gate05.matched_control_precision_audit/v1',
                          engine_sha256=result['engine_sha256'], layer_file_sha256=sha(layer_path),
                          build_result_sha256=sha(build / 'build_result.json'), build_request_sha256=sha(build / 'build_request.json'),
                          parser_constraint_inventory=inventory, actual_flags=request['flags'],
                          native_clip_correction_count=spec['quantized_conv_count'], native_build_only_no_inference=True,
                          strict_integer_gaps=[row for row in report['mapping'] if row['precision'] == 'INT8' and not row['strict_int8_tactic_verified']],
                          nonquantized_floating_layers_raw=[row for row in layers if 'Convolution' not in row.get('LayerType', '')],
                          nonquantized_floating_operation_comparability='NOT_ESTABLISHED_BY_THIS_AUDIT',
                          SASS_supplement_status='NOT_CONSUMED_BY_UNCHANGED_NAMED_TACTIC_AUDIT')
            if not report['requested_conv_formats_complete']:
                code = 2
            elif report['strict_int8_status'] == 'INTEGER_TACTIC_REVIEW_REQUIRED':
                code = 3
    assert before == {path.relative_to(root).as_posix(): sha(path) for path in root.rglob('*') if path.is_file()}, 'Input evidence changed during audit'
    report.update(client_at=datetime.now(timezone.utc).isoformat(), input_evidence_unchanged=True)
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('mapping', 'strict_integer_gaps', 'nonquantized_floating_layers_raw')},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())

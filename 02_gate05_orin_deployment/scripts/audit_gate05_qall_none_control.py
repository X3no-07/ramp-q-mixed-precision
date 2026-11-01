"""Audit the separate full corrected q_all NONE control without model execution.

Legacy source spec/reference contract remain OBEY. This new wrapper changes only
control/build provenance validation and reuses the original corrected source
tracing plus unchanged base native-format, fusion and named-tactic decisions.
Exit0=source-only pass or native formats+named tactics complete; exit2=source,
provenance or native format issue; exit3=formats complete, named-tactic gap.
Runtime SASS supplements are deliberately separate and never rewrite old audits.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import audit_gate05_corrected_qall_candidate as corrected

base = corrected.base
PROTOCOL_SHA256 = '572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1'
CORRECTED_AUDITOR_SHA256 = '4a8078effd58b214874cd25ba1d0ec1f9c0e684ab39f5df707ef46556ac9912e'
BASE_AUDITOR_SHA256 = '40420902d6fb3fd5d6b6278d64bbaa4b44c839bdae69da73dbaeb3e15cf66f6a'
HELPER_SHA256 = 'd8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f'
GRAPH_SHA256 = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
SOURCE_SPEC_SHA256 = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
SOURCE_MANIFEST_SHA256 = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
ORIGINAL_NATIVE_AUDIT = base.audit_native_layers


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def contained(root, relative):
    name = Path(relative)
    assert not name.is_absolute() and '..' not in name.parts, relative
    path = root / name
    assert root in path.resolve().parents and not path.is_symlink(), relative
    for parent in path.parents:
        if parent == root:
            break
        assert not parent.is_symlink(), relative
    return path


def verify_manifest(folder):
    manifest = read(folder / 'manifest.json')
    for name, expected in manifest['files'].items():
        path = contained(folder, name)
        assert path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    return manifest


def tool_identity():
    return {'auditor_file': Path(__file__).name, 'auditor_sha256': sha(Path(__file__)),
        'corrected_source_auditor_sha256': sha(Path(corrected.__file__)),
        'base_native_auditor_sha256': sha(Path(base.__file__)),
        'source_functions_reused': ['validate_corrected_source', 'activation_trace', 'check_graph_requests'],
        'native_function_reused': 'audit_gate05_precision_candidate.audit_native_layers',
        'native_formats_tactics_and_fusion_criteria': 'UNCHANGED_BASE_IMPLEMENTATION',
        'SASS_evidence_not_used_to_change_named_tactic_status': True}


@contextmanager
def source_trace_extension():
    # These functions only interpret the frozen Q->codeClip->DQ source. The
    # original native decision function and all its criteria remain untouched.
    original_trace, original_check = base.activation_trace, base.check_graph_requests
    assert base.audit_native_layers is ORIGINAL_NATIVE_AUDIT
    try:
        base.activation_trace = corrected.activation_trace
        base.check_graph_requests = corrected.check_graph_requests
        yield
    finally:
        base.activation_trace, base.check_graph_requests = original_trace, original_check
        assert base.audit_native_layers is ORIGINAL_NATIVE_AUDIT


def validate_control_source(root):
    assert sha(Path(corrected.__file__)) == CORRECTED_AUDITOR_SHA256
    assert sha(Path(base.__file__)) == BASE_AUDITOR_SHA256
    assert sha(root / 'protocol.json') == PROTOCOL_SHA256
    protocol = read(root / 'protocol.json')
    control = read(root / 'control_spec.json')
    assert protocol['schema'] == 'rampq.gate05.corrected_qall_none_control_protocol/v1'
    assert control['schema'] == 'rampq.gate05.corrected_qall_none_control/v1'
    assert control['protocol_sha256'] == PROTOCOL_SHA256
    assert control['runtime_precision_constraints'] == protocol['current_precision_constraints'] == 'none'
    assert control['legacy_source_precision_constraints'] == protocol['legacy_source_spec_precision_constraints'] == 'obey'
    assert control['legacy_source_metadata_rewritten'] is False and protocol['legacy_source_contract_not_rewritten'] is True
    assert control['maximum_native_builds'] == control['maximum_attempts_per_case'] == 1
    assert control['max_build_seconds'] == protocol['native_build']['max_seconds'] == 7200
    assert control['new_CPU_execution_runs_this_stage'] == protocol['new_CPU_runs'] == 0
    assert control['native_inference_executions'] == 0 and protocol['capture']['automatic_inference'] is False
    assert control['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and control['gate05_status'] == 'INCOMPLETE'
    assert control['original_tolerances_changed'] is False and control['new_tolerances_defined'] is False
    assert control['dataset_evaluation_run'] is False and control['weights_changed'] is False
    manifest = verify_manifest(root)
    assert manifest['schema'] == 'rampq.gate05.corrected_qall_none_control_bundle/v1'
    for name in ('control_spec.json', 'protocol.json', control['native_builder_file'], 'source_bundle/model.onnx',
                 'source_bundle/spec.json', 'source_bundle/manifest.json', 'cpu_equivalence/manifest.json',
                 'cpu_equivalence/analysis.json', 'cpu_equivalence/raw_outputs.npz'):
        assert name in manifest['files'], name
    builder = contained(root, control['native_builder_file'])
    assert builder.name == 'jetson_build_corrected_qall_none.py' and sha(builder) == control['native_builder_sha256']
    for name, digest in control['unchanged_helpers'].items():
        assert name in manifest['files'] and sha(contained(root, name)) == digest
    assert sha(root / 'jetson_build_precision_candidate.py') == HELPER_SHA256
    source = root / control['source_bundle_directory']
    cpu = root / control['cpu_equivalence_directory']
    assert source == root / 'source_bundle' and cpu == root / 'cpu_equivalence'
    assert sha(source / 'manifest.json') == SOURCE_MANIFEST_SHA256 == control['source_manifest_sha256']
    source_manifest = verify_manifest(source)
    assert {p.name for p in source.iterdir()} == set(source_manifest['files']) | {'manifest.json', 'SHA256SUMS'}
    assert len(list(source.iterdir())) == 21
    assert sha(source / 'spec.json') == SOURCE_SPEC_SHA256 == control['source_spec_sha256']
    assert sha(source / 'model.onnx') == GRAPH_SHA256 == control['graph_sha256']
    spec, graph, errors = corrected.validate_corrected_source(source / 'spec.json', source / 'module_map.json', source / 'model.onnx')
    assert spec['precisionConstraints'] == 'obey'  # Authentic old metadata.
    assert spec['quantized_conv_count'] == 88 and spec['fp16_conv_count'] == 0
    cpu_manifest = verify_manifest(cpu)
    assert sha(cpu / 'manifest.json') == control['cpu_equivalence_manifest_sha256'] == protocol['source_CPU_equivalence']['manifest_sha256']
    assert set(p.name for p in cpu.iterdir()) == set(cpu_manifest['files']) | {'manifest.json'} and len(list(cpu.iterdir())) == 6
    assert sha(cpu / 'analysis.json') == control['cpu_equivalence_analysis_sha256'] == protocol['source_CPU_equivalence']['analysis_sha256']
    assert sha(cpu / 'raw_outputs.npz') == control['cpu_equivalence_raw_outputs_sha256'] == protocol['source_CPU_equivalence']['raw_outputs_sha256']
    analysis = read(cpu / 'analysis.json')
    assert analysis['status'] == 'FULL_FIXED_INPUT_CPU_REWRITE_BYTE_EQUIVALENCE_VERIFIED'
    assert analysis['candidate_onnx_sha256'] == GRAPH_SHA256 and analysis['original_complete_model_pytorch_ort_guard'] == 'FAILED_UNCHANGED'
    assert analysis['numerical_migration_pass_claim'] is False
    return control, spec, graph, errors


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


def validate_native_provenance(root, build, control, spec):
    result = read(build / 'build_result.json')
    identity = read(build / 'tool_hashes.json')
    assert identity == result['tool_identity']
    expected = {'builder_file': control['native_builder_file'], 'builder_sha256': control['native_builder_sha256'],
        'helper_relative_path': 'jetson_build_precision_candidate.py', 'helper_sha256': HELPER_SHA256,
        'protocol_sha256': PROTOCOL_SHA256, 'control_spec_sha256': sha(root / 'control_spec.json'),
        'outer_manifest_sha256': sha(root / 'manifest.json'), 'legacy_source_spec_sha256': SOURCE_SPEC_SHA256,
        'legacy_source_manifest_sha256': SOURCE_MANIFEST_SHA256}
    assert identity == expected
    # A preserved parse/startup failure may precede request/Clip snapshots.
    # It still cannot be evaluated as a successful actual engine.
    if not (build / 'build_request.json').exists():
        assert result['schema'] == 'rampq.gate05.corrected_qall_none_build_result/v1'
        assert result['status'] == 'BUILD_OR_CONTRACT_VALIDATION_FAILED'
        assert result['runtime_precision_constraints'] == 'none' and result['legacy_source_precision_constraints'] == 'obey'
        assert result['graph_sha256'] == GRAPH_SHA256 and result['native_inference_executions'] == 0
        assert result['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED'
        return result, {}, {'status': 'NOT_AVAILABLE_FAILURE_BEFORE_FROZEN_PARSED_REQUEST'}
    request = read(build / 'build_request.json')
    assert identity == request['tool_identity']
    assert request['schema'] == 'rampq.gate05.corrected_qall_none_build_request/v1'
    assert result['schema'] == 'rampq.gate05.corrected_qall_none_build_result/v1'
    for item in (request, result):
        assert item['configuration'] == 'q_all' and item['graph_sha256'] == GRAPH_SHA256
        assert item['legacy_source_precision_constraints'] == 'obey' and item['runtime_precision_constraints'] == 'none'
        assert item['separate_new_control_not_legacy_attempt'] is True
        assert item['new_CPU_execution_runs'] == 0 and item['max_build_seconds'] == 7200 and item['maximum_native_builds'] == 1
        assert item['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and item['gate05_status'] == 'INCOMPLETE'
    assert result['native_inference_executions'] == 0
    flags = request['flags']['get_flag']
    for name, enabled in [('INT8', True), ('FP16', True), ('TF32', False), ('OBEY_PRECISION_CONSTRAINTS', False), ('PREFER_PRECISION_CONSTRAINTS', False)]:
        assert flags[name] == {'supported': True, 'enabled': enabled}, name
    assert request['workspace_bytes_actual'] == 1024 ** 3 and request['avg_timing_iterations_actual'] == 1
    assert request['profiling_verbosity_actual'] == 'ProfilingVerbosity.DETAILED'
    assert request['explicit_layer_precision_constraint_count'] == request['explicit_layer_output_type_constraint_count'] == request['layer_constraint_reset_count'] == 0
    assert request['parser_constraints_before_after_identical'] is True and request['native_int8_QDQ_defaults_preserved'] is True
    assert request['source_conv_count'] == 88 and request['legacy_source_requests'] == spec['requests']
    before = read(build / 'parsed_constraints_before.json')
    after = read(build / 'parsed_constraints_after.json')
    assert constraint_values(before) == constraint_values(after)
    inventory = constraint_inventory(before)
    assert not inventory['query_errors'], inventory['query_errors']
    clip_before = read(build / 'parsed_clip_attributes_before_correction.json')
    clip_after = read(build / 'parsed_clip_attributes_after_correction.json')
    assert clip_before['count'] == clip_after['count'] == len(clip_before['layers']) == len(clip_after['layers']) == 88
    original = {row['name']: row for row in clip_before['layers']}
    corrections = {row['name']: row for row in clip_after['layers']}
    assert len(original) == len(corrections) == 88 and set(original) == set(corrections) == set(spec['clip_names'])
    for name, row in corrections.items():
        assert row['before'] == {key: original[name][key] for key in ('alpha', 'beta')}
        assert row['after'] == {'alpha': -127.0, 'beta': 127.0}
    assert clip_after['layers'] == request['explicit_clip_importer_corrections']
    assert bindings_signature(request['bindings_in_network_order']) == bindings_signature(spec['expected_binding_contract'])
    assert all(row['allowed_formats'] == 1 for row in request['bindings_in_network_order'])
    parsed_convs = [row['name'] for row in read(build / 'parsed_layers_before.json') if row['type'] == 'LayerType.CONVOLUTION']
    assert len(parsed_convs) == 88 and set(parsed_convs) == {row['onnx_node'] for row in spec['requests']}
    return result, request, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-source-only', action='store_true')
    args = parser.parse_args()
    root = args.control_root.resolve()
    assert root.is_dir()
    if args.output:
        output = args.output.resolve()
        assert output.parent.is_dir() and not output.exists() and root not in output.parents
    before = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    control, spec, graph, errors = validate_control_source(root)
    metadata = {'audit_tool_identity': tool_identity(), 'configuration': 'q_all', 'source_convs': 88,
        'quantized_conv_count': 88, 'fp16_conv_count': 0, 'runtime_precision_constraints': 'none',
        'legacy_source_precision_constraints': 'obey', 'source_bundle_spec_unchanged': True,
        'control_spec_sha256': sha(root / 'control_spec.json'), 'protocol_sha256': PROTOCOL_SHA256,
        'source_manifest_sha256': SOURCE_MANIFEST_SHA256, 'source_spec_sha256': SOURCE_SPEC_SHA256,
        'onnx_sha256': GRAPH_SHA256, 'activation_quantizer_topology': corrected.TOPOLOGY,
        'native_formats_tactics_and_fusion_criteria_changed': False, 'prior_depthwise_integer_gaps_waived': False,
        'new_CPU_native_or_SSH_execution': False, 'original_complete_model_guard': 'FAILED_UNCHANGED',
        'dataset_evaluation_run': False, 'gate05_status': 'INCOMPLETE'}
    code = 0
    if args.check_source_only:
        report = {'schema': 'rampq.gate05.qall_none_source_audit/v1',
            'status': 'CORRECTED_SOURCE_STRUCTURE_VERIFIED' if not errors else 'SOURCE_STRUCTURE_REVIEW_REQUIRED',
            'source_structure_errors': errors, **metadata}
        code = 2 if errors else 0
    else:
        assert args.build is not None, '--build required unless --check-source-only'
        build = args.build.resolve()
        assert root in build.parents and build.is_dir() and not errors, errors
        result, request, inventory = validate_native_provenance(root, build, control, spec)
        if result['status'] != 'BUILT_AND_DESERIALIZED_BUILD_ONLY':
            assert result['status'] == 'BUILD_OR_CONTRACT_VALIDATION_FAILED'
            report = {'schema': 'rampq.gate05.qall_none_precision_audit/v1',
                'status': 'NATIVE_BUILD_FAILED_NO_ACTUAL_ENGINE_AUDIT', 'requested_conv_formats_complete': False,
                'strict_int8_status': 'NOT_EVALUATED_NO_SUCCESSFUL_ENGINE', 'native_build_result': result,
                'parser_constraint_inventory': inventory, **metadata}
            code = 2
        else:
            assert sha(build / 'model.engine') == result['engine_sha256']
            assert result['engine_execution_context_created'] is False
            binding_record = read(build / 'bindings.json')['bindings']
            assert binding_record == result['bindings']
            assert bindings_signature(binding_record) == bindings_signature(spec['expected_binding_contract'])
            assert all(row['format'] == 'TensorFormat.LINEAR' for row in binding_record)
            layer_path = build / 'engine_layers.json'
            inspector = read(layer_path)
            layers = inspector['Layers'] if isinstance(inspector, dict) else inspector
            convs = [row for row in layers if 'Convolution' in row.get('LayerType', '')]
            assert convs == read(build / 'conv_rows.json')
            with source_trace_extension():
                report = ORIGINAL_NATIVE_AUDIT(layers, spec['requests'], graph, base.read(root / 'source_bundle/module_map.json')['mapping'])
            assert report['quantized_conv_count'] == 88 and report['fp16_conv_count'] == 0
            report.update(metadata)
            report.update(schema='rampq.gate05.qall_none_precision_audit/v1',
                engine_sha256=result['engine_sha256'], layer_file_sha256=sha(layer_path),
                build_result_sha256=sha(build / 'build_result.json'), build_request_sha256=sha(build / 'build_request.json'),
                parser_constraint_inventory=inventory, actual_flags=request['flags'],
                native_clip_correction_count=88, native_build_only_no_inference=True,
                strict_integer_gaps=[row for row in report['mapping'] if not row['strict_int8_tactic_verified']],
                SASS_supplement_status='NOT_CONSUMED_BY_THIS_UNCHANGED_NAMED_TACTIC_AUDIT',
                numerical_migration_pass_claim=False, accuracy_pass_claim=False, performance_pass_claim=False)
            if not report['requested_conv_formats_complete']:
                code = 2
            elif report['strict_int8_status'] == 'INTEGER_TACTIC_REVIEW_REQUIRED':
                code = 3
    assert before == {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}, 'Input evidence changed during audit'
    report.update(client_at=datetime.now(timezone.utc).isoformat(), input_evidence_unchanged=True)
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('mapping', 'strict_integer_gaps')}, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())

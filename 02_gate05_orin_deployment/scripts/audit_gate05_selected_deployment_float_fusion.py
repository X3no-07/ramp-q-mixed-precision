"""Saved-only selected deployment float/fusion review with a pinned legacy auxiliary.

The core five retain their frozen NONE build policy and unchanged v2 auditor.
Only q_only_G1 uses the eligibility-selected, exact-source legacy PREFER engine.
It is explicitly a different-policy weak auxiliary, never a core primary proxy.
No model, NMS, dataset prediction, or remote connection is executed or read.
"""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import audit_gate05_float_fusion_comparability_v2 as base

ROOT = base.ROOT
BASE_SHA = 'f723a29620ba5aa18c7dd7f8e9000fbd00d405e992e5b59067c86fe06c91e523'
PLAN = ROOT / 'artifacts/gate05/20261021_gate05_completion_plan'
AUX_POLICY = PLAN / 'auxiliary_qonly_selection_policy.json'
AUX_POLICY_SHA = 'c3225af0f08c420688a63997d76855c47284e325175594f119c058b7f87414b6'
SELECTED = PLAN / 'selected_deployment_identity_table.json'
SELECTED_SHA = '542cfb974e32e81fa1acbb6c117a8902044ee574c86993135917b4eaa8800eeb'
CORE = base.CONFIGURATIONS[:-1]
require, sha, read = base.require, base.sha, base.read


def mark(path, evidence, expected=None):
    actual = sha(path)
    require(expected is None or actual == expected, 'Pinned input identity differs: ' + str(path))
    evidence[path] = actual
    return read(path) if path.suffix == '.json' else actual


def actual_mapping(reference, audit, inspector):
    """Same token/output/explicit-SWISH mapping and endpoint rule as locked v2."""
    nodes = {node.name: node for node in reference.graph.node}
    pattern = {name: base.token_pattern(name) for name in nodes}
    output_producer = {output: name for name, node in nodes.items() for output in node.output}
    activation_owner = base.activation_pairs(nodes)
    constant_only = base.constant_only_nodes(reference.graph)
    value_types = base.source_value_types(reference.graph)
    mapped, observations = defaultdict(list), defaultdict(list)
    actual_rows, weighted_source_by_layer = [], defaultdict(set)
    for row in audit['mapping']:
        weighted_source_by_layer[row['engine_layer']].add(row['onnx_node'])
    for index, layer in enumerate(inspector['Layers']):
        name, methods = layer.get('Name', ''), defaultdict(set)
        if not base.is_boundary_copy(layer):
            for source_name, matcher in pattern.items():
                if matcher.search(name):
                    methods[source_name].add('EXACT_SOURCE_NAME_TOKEN_IN_OPERATION_LAYER')
            for tensor in layer.get('Outputs', []):
                producer = output_producer.get(tensor.get('Name'))
                if producer:
                    methods[producer].add('EXACT_SOURCE_OUTPUT_TENSOR')
            if layer.get('Activation') in ('SWISH', 'SILU'):
                for source_name, conv in activation_owner.items():
                    if conv in weighted_source_by_layer.get(name, set()):
                        methods[source_name].add('EXACT_SOURCE_CONV_SILU_ADJACENCY_AND_REPORTED_ACTIVATION')
        for source_name, reasons in methods.items():
            mapped[source_name].append({'layer_index': index, 'methods': sorted(reasons)})
        for side in ('Inputs', 'Outputs'):
            for tensor in layer.get(side, []):
                observations[tensor.get('Name')].append({'layer_index': index, 'side': side, **base.tensor_row(tensor)})
        actual_rows.append({'layer_index': index, 'name': name, 'type': layer.get('LayerType'),
            'source_members': sorted(methods), 'weighted_source_members': sorted(weighted_source_by_layer.get(name, set())),
            'is_boundary_copy': base.is_boundary_copy(layer), 'boundary': base.operation_boundary(layer),
            'reported_compute_precision': {key: value for key, value in layer.items()
                if any(part in key.lower() for part in ('precision', 'compute', 'accumul'))},
            'tactic_name': layer.get('TacticName'), 'tactic_value': layer.get('TacticValue'),
            'raw_inspector_layer': layer})
    source_actual = {}
    for name, node in nodes.items():
        entries = mapped[name]
        source_actual[name] = {'name': name, 'op_type': node.op_type,
            'scope': 'INTEGER_SHAPE_ARITHMETIC' if node.op_type in ('Add', 'Sub', 'Mul', 'Div') and
                         all(value_types.get(value) == 7 for value in list(node.input) + list(node.output)) else
                     'STATIC_FOLDABLE_EXPRESSION' if name in constant_only else
                     'SHAPE_OR_VIEW' if node.op_type in base.SHAPE_OPS else
                     'STATIC_LITERAL' if node.op_type in base.STATIC_OPS else
                     'WEIGHTED_SOURCE' if node.op_type == 'Conv' else 'FLOAT_ARITHMETIC',
            'mapped_layers': entries,
            'source_outputs_observed': {output: observations[output] for output in node.output},
            'source_inputs_observed': {value: observations[value] for value in node.input if value in observations},
            'source_direct_silu_conv': activation_owner.get(name),
            'source_output_ONNX_element_types': {output: value_types.get(output) for output in node.output},
            'internal_values_not_visible_from_fused_boundary': any(
                len(actual_rows[item['layer_index']]['source_members']) > 1 for item in entries)}
    return source_actual, actual_rows


def binding_signature(rows):
    return sorted((r['name'], r['input'], tuple(r['shape']), r['dtype']) for r in rows)


def legacy_auxiliary(reference, evidence, policy, identity):
    import onnx
    bundle = ROOT / policy['local_source']
    build = ROOT / policy['local_build']
    audit_path = Path(identity['local_precision_audit'])
    paths = {'graph': bundle/'model.onnx', 'spec': bundle/'spec.json', 'manifest': bundle/'manifest.json',
        'engine': build/'model.engine', 'build_result': build/'build_result.json',
        'build_request': build/'build_request.json', 'engine_layers': build/'engine_layers.json',
        'precision_audit': audit_path}
    pins = {'graph': 'selected_graph_sha256', 'spec': 'selected_spec_sha256', 'manifest': 'selected_manifest_sha256',
        'engine': 'selected_engine_sha256', 'build_result': 'selected_build_result_sha256',
        'build_request': 'selected_build_request_sha256', 'engine_layers': 'selected_engine_layers_sha256',
        'precision_audit': 'selected_precision_audit_sha256'}
    for key, path in paths.items():
        mark(path, evidence, policy[pins[key]])
    audit, spec, manifest = (read(paths[key]) for key in ('precision_audit', 'spec', 'manifest'))
    request, result, inspector = (read(paths[key]) for key in ('build_request', 'build_result', 'engine_layers'))
    require(audit['schema'] == 'rampq.gate05.precision_candidate_audit/v1' and
        audit['requested_conv_formats_complete'] is True and not audit['errors'] and not audit['unmapped_engine_convs'] and
        audit['source_convs'] == 88 and audit['quantized_conv_count'] == audit['verified_int8_format_convs'] == 6 and
        audit['fp16_conv_count'] == audit['verified_fp16_format_convs'] == 82 and
        audit['named_integer_tactic_source_convs'] == 6 and audit['source_convs_without_tactic_name'] == 0 and
        audit['strict_int8_status'] == 'ALL_REQUESTED_INT8_TACTICS_VERIFIED',
        'Legacy auxiliary does not satisfy original 6INT8/82FP16/named-tactic rules')
    for key, filekey in [('onnx_sha256', 'graph'), ('engine_sha256', 'engine'), ('spec_sha256', 'spec'),
                          ('layer_file_sha256', 'engine_layers'), ('build_result_sha256', 'build_result')]:
        require(audit[key] == evidence[paths[filekey]], 'Legacy audit references another saved input')
    for name, row in manifest['files'].items():
        path = bundle/name
        require(path.is_file() and path.stat().st_size == row['bytes'], 'Legacy bundle file missing/wrong size')
        mark(path, evidence, row['sha256'])
    require(result['status'] == 'BUILT_AND_DESERIALIZED' and
        result['engine_sha256'] == evidence[paths['engine']] and
        result['engine_bytes'] == paths['engine'].stat().st_size and
        result['single_image_inference'] == result['dataset_evaluation'] == 'NOT_RUN',
        'Unexpected legacy saved build result')
    for row in (request, result):
        require(row['configuration'] == 'q_only_G1' and row['onnx_sha256'] == evidence[paths['graph']] and
            row['spec_sha256'] == evidence[paths['spec']] and row['manifest_sha256'] == evidence[paths['manifest']] and
            row['requested_precision_constraints_mode'] == 'prefer', 'Legacy request/result provenance differs')
    require(request['builder_flags'] == ['INT8', 'FP16', 'PREFER_PRECISION_CONSTRAINTS', 'TF32_DISABLED'] and
        request['workspace_mib'] == 1024 and request['avg_timing_iterations'] == 1,
        'Legacy PREFER policy differs from frozen auxiliary selection')
    restored = {r['onnx_node'] for r in spec['requests'] if r['precision'] == 'FP16'}
    require(len(restored) == 82 and len(request['restored_fp16_compute_and_output_constraints']) == 82 and
        set(request['restored_fp16_compute_and_output_constraints']) == restored and
        spec['restored_conv_compute_precision'] == spec['restored_conv_output_type'] == 'fp16' and
        spec['precisionConstraints'] == 'prefer', 'Legacy explicit FP16 request set differs')
    require(len(request['explicit_clip_importer_corrections']) == 6 and
        {r['name'] for r in request['explicit_clip_importer_corrections']} == set(spec['clip_names']) and
        all(r['after'] == {'alpha': -127.0, 'beta': 127.0} for r in request['explicit_clip_importer_corrections']),
        'Legacy code Clip importer correction differs')
    require(binding_signature(result['bindings']) == binding_signature(spec['expected_binding_contract']) and
        all(r['format'] == 'TensorFormat.LINEAR' for r in result['bindings']), 'Legacy IO contract differs')
    for row in audit['mapping']:
        require(row['requested_formats_verified'] is True, 'Legacy source format not verified')
        if row['precision'] == 'FP16':
            require(row['weights_type'] == 'Half' and all('FP16' in value for value in
                row['input_formats'][row['convolution_activation_input_index']:row['convolution_activation_input_index']+1] +
                row['output_formats']), 'Legacy restored source has non-FP16 actual formats')
        else:
            require(row['weights_type'] == 'Int8' and row['strict_int8_tactic_verified'] is True,
                    'Legacy INT8 actual weights/arithmetic incomplete')
    require(all(r['accounting']['status'] == 'FUSION_ACCOUNTING_PASS' and not r['accounting']['errors']
                for r in audit['fusions']), 'Legacy weighted fusion accounting incomplete')
    graph = onnx.load(str(paths['graph']), load_external_data=False)
    source = base.graph_source_audit(reference, graph, spec['requests'])
    require(not source['errors'], 'Legacy source semantics differ: '+str(source['errors']))
    environment = mark(build/'environment.json', evidence)
    parsed = mark(build/'parsed_network_validation.json', evidence)
    require(parsed['status'] == 'ALL_REQUESTED_CONV_NAMES_EXIST_IN_PARSED_NETWORK' and
        parsed['source_conv_count'] == 88 and parsed['requests'] == spec['requests'],
        'Legacy parsed source inventory differs')
    # The original saved request lists flags but did not capture config.get_flag.
    # Normalize requested values for comparison; disclose the observability limit.
    normalized_flags = {key: {'supported': True, 'enabled': enabled}
                        for key, enabled in policy['selected_flags'].items()}
    source_actual, actual_rows = actual_mapping(reference, audit, inspector)
    failed = ROOT/'artifacts/gate05/20261021_matched_control_family_native/q_only_G1/precision_audit.json'
    rejected = mark(failed, evidence, policy['rejected_new_NONE_control']['precision_audit_sha256'])
    require(rejected['requested_conv_formats_complete'] is False and
        rejected['engine_sha256'] == policy['rejected_new_NONE_control']['engine_sha256'],
        'Rejected NONE failure was altered or relabelled')
    require(sha(ROOT/'artifacts/gate05/20261021_matched_control_family_native/q_only_G1/native_evidence/model.onnx') ==
        evidence[paths['graph']], 'Selected old/new qonly source graphs are not byte-identical')
    return {'configuration': 'q_only_G1', 'native_root': bundle.parent.relative_to(ROOT).as_posix(),
        'identities': {p.relative_to(ROOT).as_posix(): expected for p, expected in evidence.items()
                       if bundle.parent in p.parents},
        'source_audit': source, 'actual_flags': normalized_flags, 'raw_config_flags': None,
        'flag_evidence_scope': 'LEGACY_SAVED_BUILD_REQUEST_FLAGS_NOT_GET_FLAG_SNAPSHOT',
        'source_requests': spec['requests'], 'source_to_actual': source_actual, 'actual_layers': actual_rows,
        'layer_type_counts_descriptive_only': dict(Counter(r['type'] for r in actual_rows)),
        'bindings': {'bindings': result['bindings']},
        'weighted_fusion_accounting_from_unchanged_auditor': audit['fusions'],
        'precision_audit_status': audit['status'], 'named_integer_tactic_status_not_changed': audit['strict_int8_status'],
        'common_build_policy': {'workspace_bytes_actual': request['workspace_mib'] * 1024**2,
            'avg_timing_iterations_actual': request['avg_timing_iterations'],
            'profiling_verbosity_actual': None, 'runtime_precision_constraints': 'prefer',
            'bindings_in_network_order': request['expected_binding_contract'],
            'explicit_layer_precision_constraint_count': 82, 'explicit_layer_output_type_constraint_count': 82,
            'layer_constraint_reset_count': None},
        'legacy_policy_observability': {'profiling_verbosity_getter_snapshot': 'NOT_SAVED',
            'config_get_flag_snapshot': 'NOT_SAVED', 'parser_constraints_before_after_snapshots': 'NOT_SAVED',
            '82_compute_and_82_output_constraints': 'EXPLICIT_SAVED_BUILD_REQUEST_AND_88_ACTUAL_FORMAT_AUDIT',
            'missing_snapshots_not_fabricated_as_NONE_equivalence': True},
        'common_runtime_environment': {key: environment[key] for key in
            ('tensorrt', 'python', 'machine', 'boot_id', 'jetson_linux', 'power_mode')},
        'source_semantics_errors': source['errors'], 'source_weighted_formats_complete_is_not_float_comparability': True,
        'role': 'AUX_DIFFERENT_CONSTRAINT_POLICY_LIMITED_WEAK_CONTROL',
        'not_core_PRIMARY_member': True, 'pure_G1_causal_control': False,
        'selected_by_format_eligibility_before_new150_outputs': True,
        'rejected_new_NONE_precision_failure_preserved': {'file': failed.relative_to(ROOT).as_posix(),
                                                        'sha256': evidence[failed]}}


def compare_selected(left, right):
    if 'q_only_G1' not in (left['configuration'], right['configuration']):
        result = base.compare_pair(left, right)
        result['comparison_role'] = 'CORE_FIVE_SAME_NONE_POLICY'
        return result
    result = base.compare_pair(left, right)
    require(not result['source_request_identity_mismatches'] and not result['remaining_same_quantizer_mismatches'] and
        not left['source_semantics_errors'] and not right['source_semantics_errors'], 'Auxiliary source/range mismatch')
    require(left['common_runtime_environment'] == right['common_runtime_environment'],
            'Auxiliary saved device/runtime/boot/power differs')
    # Preserve original strict comparison without laundering the declared policy
    # difference into identical flags; only the frozen selection policy permits
    # a limited auxiliary association, not a primary arithmetic comparison.
    result['unmodified_v2_policy_sensitive_eligibility'] = result['eligibility']
    result['comparison_role'] = 'AUX_DIFFERENT_CONSTRAINT_POLICY_LIMITED_WEAK_CONTROL'
    result['primary_authenticity_eligible_from_this_audit'] = False
    result['eligibility'] = ('AUX_MAPPING_GAP_PRESERVED' if result['gaps'] else
                             'AUX_DIFFERENT_CONSTRAINT_POLICY_LIMITED_WEAK_CONTROL')
    result['auxiliary_saved_source_and_float_metadata_accounted'] = not result['gaps']
    result['undeclared_build_policy_mismatch'] = False
    result['declared_different_build_policy'] = {'left': left['common_build_policy'],
        'right': right['common_build_policy'], 'left_flags': left['actual_flags'], 'right_flags': right['actual_flags'],
        'selection_policy_sha256': AUX_POLICY_SHA,
        'legacy_flags_are_request_not_get_flag_snapshots': True}
    result['pure_G1_causal_identification'] = 'NOT_ESTABLISHED_DIFFERENT_AUXILIARY_CONSTRAINT_POLICY'
    result['strict_unfrozen_draft_float_match']['status'] = 'GAP_PRESERVED_DIFFERENT_CONSTRAINT_POLICY'
    result['auxiliary_is_not_a_core_primary_substitute'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Exclusive new report required')
    require(sha(Path(base.__file__)) == BASE_SHA, 'Locked core float auditor changed')
    evidence = {Path(base.__file__): BASE_SHA}
    policy = mark(AUX_POLICY, evidence, AUX_POLICY_SHA)
    selected = mark(SELECTED, evidence, SELECTED_SHA)
    mark(base.DRAFT, evidence, base.DRAFT_SHA)
    mark(base.POLICY, evidence, base.POLICY_SHA)
    require(policy['status'] == 'FROZEN_BEFORE_NEW150_IMAGE_MODEL_OUTPUTS' and
        policy['new150_image_model_executions'] == 0 and policy['new150_image_outputs_read'] is False,
        'Auxiliary selection was not frozen before new model outputs')
    require([r['name'] for r in selected['members']] == list(base.CONFIGURATIONS), 'Selected family inventory differs')
    import onnx
    reference_path = base.native_root('fp16_control')/'native_evidence/model.onnx'
    mark(reference_path, evidence)
    reference = onnx.load(str(reference_path), load_external_data=False)
    members = [base.load_member(cfg, reference, evidence) for cfg in CORE]
    members.append(legacy_auxiliary(reference, evidence, policy, selected['members'][-1]))
    for member, identity in zip(members, selected['members']):
        for key, filekey in [('source_graph_sha256', 'local_model'), ('engine_sha256', 'local_engine'),
                              ('native_precision_audit_sha256', 'local_precision_audit')]:
            mark(Path(identity[filekey]), evidence, identity[key])
        for filename, key in [('build_request.json', 'build_request_sha256'), ('build_result.json', 'build_result_sha256'),
                              ('engine_layers.json', 'engine_layers_sha256')]:
            mark(Path(identity['local_build'])/filename, evidence, identity[key])
    comparisons = [compare_selected(left, right) for left, right in itertools.combinations(members, 2)]
    core_pairs = [p for p in comparisons if p['comparison_role'] == 'CORE_FIVE_SAME_NONE_POLICY']
    aux_pairs = [p for p in comparisons if p['comparison_role'] != 'CORE_FIVE_SAME_NONE_POLICY']
    core_eligible = all(p['primary_authenticity_eligible_from_this_audit'] for p in core_pairs)
    aux_accounted = all(p['auxiliary_saved_source_and_float_metadata_accounted'] for p in aux_pairs)
    require(all(sha(path) == expected for path, expected in evidence.items()), 'Saved source changed during review')
    runtime_loaded = any(name in sys.modules for name in ('torch', 'onnxruntime', 'tensorrt'))
    require(not runtime_loaded, 'Unexpected inference runtime import')
    report = {'schema': 'rampq.gate05.selected_deployment_float_fusion_review/v1',
        'status': 'CORE_FIVE_METADATA_ACCOUNTED_AUX_DIFFERENT_POLICY_LIMITED_WEAK_CONTROL'
            if core_eligible and aux_accounted else 'COMPARABILITY_GAPS_PRESERVED',
        'client_UTC': datetime.now(timezone.utc).isoformat(),
        'author_role': 'Saved-only review independently uses native auditor; authored graph preparation, not native builder or precision auditor',
        'audit_tool': base.input_identity(Path(__file__).resolve()), 'core_float_auditor_sha256': BASE_SHA,
        'source_float_interpretation_policy_sha256': base.POLICY_SHA,
        'auxiliary_selection_policy_sha256': AUX_POLICY_SHA, 'selected_identity_table_sha256': SELECTED_SHA,
        'configuration_order': list(base.CONFIGURATIONS), 'core_primary_configurations': list(CORE),
        'core_five_float_metadata_comparability_eligible': core_eligible,
        'selected_auxiliary_metadata_accounted_with_declared_policy_limit': aux_accounted,
        'all_six_identical_constraint_policy': False, 'all_six_pure_arithmetic_control': False,
        'overall_Gate_or_arithmetic_eligibility_not_decided_by_float_audit': True,
        'legacy_request_flag_snapshot_limit_retained': True,
        'strict_unfrozen_draft_result': 'Preserved per comparison and prior snapshots; no strict same-float/same-policy PASS claim',
        'no_additional_bridge_built_or_authorized': True,
        'members': members, 'pairwise_comparisons': comparisons,
        'saved_inputs': [base.input_identity(path) for path in sorted(evidence, key=str)],
        'saved_inputs_unchanged_after_read': True,
        'execution': {'new_CPU_model_calls': 0, 'native_calls': 0, 'NMS_calls': 0,
            'dataset_prediction_streams_read': 0, 'SSH_calls': 0, 'inference_runtime_imported': runtime_loaded},
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
        'pure_G1_causality_or_accuracy_or_target_recovery_claim': False}
    base.write_new(args.output, report)
    print(json.dumps({'status': report['status'], 'file': str(args.output.resolve()), 'sha256': sha(args.output),
        'core_pairs': len(core_pairs), 'aux_pairs': len(aux_pairs), 'core_eligible': core_eligible,
        'aux_metadata_accounted': aux_accounted, 'core_pair_source_errors': [p['source_request_identity_mismatches']
            for p in core_pairs], 'model_calls': 0, 'NMS_calls': 0}))


if __name__ == '__main__':
    main()

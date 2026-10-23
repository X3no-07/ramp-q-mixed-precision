"""Audit only the frozen 88-code-Clip q_all derivative (CPU, no execution).

The original auditor supplies all native formats, named-integer-tactic criteria,
fusion accounting, build lineage checks and exit codes. Only source activation
tracing and the source quantizer-bound checks are extended to Q -> INT8 Clip ->
DQ. --check-source-only checks the actual source artifact without a native build.
"""
import argparse
import hashlib
import json
from pathlib import Path

import audit_gate05_precision_candidate as base


TOPOLOGY = 'Q_INT8_CODE_CLIP_DQ'
EXPECTED_BASE_SHA256 = '40420902d6fb3fd5d6b6278d64bbaa4b44c839bdae69da73dbaeb3e15cf66f6a'
EXPECTED_GRAPH_SHA256 = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
EXPECTED_PARENT_GRAPH_SHA256 = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
EXPECTED_REWRITE_AUDIT_SHA256 = '842eff15adfedd75735815cbe4040d17a645653e3b62511149afacbf8f923be9'
EXPECTED_BUILDER_SHA256 = 'f255efb88b517dcc9537da72b76f6e4d1e71bc64e50ad13df006661821823451'
EXPECTED_BUILD_HELPER_SHA256 = 'd8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f'
EXPECTED_REFERENCE_CONTRACT_SHA256 = '87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430'
EXPECTED_SPEC_SHA256 = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
EXPECTED_MANIFEST_SHA256 = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
ORIGINAL_AUDIT_NATIVE_LAYERS = base.audit_native_layers
SOURCE_CONTEXT = {}
NATIVE_CORRECTION_CONTEXT = {}


def tool_identity():
    return {'corrected_auditor_file': Path(__file__).name,
            'corrected_auditor_sha256': base.sha(Path(__file__)),
            'base_auditor_file': Path(base.__file__).name,
            'base_auditor_sha256': base.sha(Path(base.__file__)),
            'native_formats_tactics_and_fusion_criteria': 'UNCHANGED_BASE_IMPLEMENTATION',
            'overridden_source_functions': ['activation_trace', 'check_graph_requests']}


def scalar_initializer(values, name, dtype, expected=None):
    import numpy as np
    from onnx import numpy_helper
    assert name in values, 'Initializer absent: ' + name
    value = numpy_helper.to_array(values[name])
    assert value.dtype == np.dtype(dtype) and value.ndim == 0, 'Scalar initializer dtype/shape mismatch: ' + name
    assert bool(np.isfinite(value)), 'Nonfinite scalar initializer: ' + name
    if expected is not None:
        wanted = np.asarray(expected, dtype=dtype)
        assert value.tobytes() == wanted.tobytes(), 'Scalar initializer value mismatch: ' + name
    return value


def activation_trace(conv, graph):
    """Refuse every topology except Conv <- DQ <- INT8 Clip[-127,127] <- Q."""
    import numpy as np
    producers, values = graph['producers'], graph['initializers']
    dq = producers[conv.input[0]]
    assert dq.op_type == 'DequantizeLinear' and dq.domain in ('', 'ai.onnx'), 'Expected ONNX activation DQ'
    assert len(dq.input) == 3 and len(dq.output) == 1 and not dq.attribute, 'Expected scalar activation DQ'
    clip = producers[dq.input[0]]
    assert clip.op_type == 'Clip' and clip.domain in ('', 'ai.onnx'), 'Expected code Clip immediately before DQ'
    assert len(clip.input) == 3 and len(clip.output) == 1 and not clip.attribute, 'Expected input-bound code Clip'
    quantize = producers[clip.input[0]]
    assert quantize.op_type == 'QuantizeLinear' and quantize.domain in ('', 'ai.onnx'), 'Expected Q immediately before code Clip'
    assert len(quantize.input) == 3 and len(quantize.output) == 1 and not quantize.attribute, 'Expected scalar activation Q'
    assert list(quantize.input[1:]) == list(dq.input[1:]), 'Q/DQ scale and zero names differ'
    assert conv.input[0] == dq.output[0] and dq.input[0] == clip.output[0] and clip.input[0] == quantize.output[0], \
        'Incomplete exact activation topology'
    scale = scalar_initializer(values, quantize.input[1], np.float32)
    assert float(scale) > 0, 'Activation scale must be positive'
    scalar_initializer(values, quantize.input[2], np.int8, 0)
    scalar_initializer(values, dq.input[1], np.float32)
    scalar_initializer(values, dq.input[2], np.int8, 0)
    scalar_initializer(values, clip.input[1], np.int8, -127)
    scalar_initializer(values, clip.input[2], np.int8, 127)
    constants = {key: base.signature(values[name]) for key, name in
                 [('scale', quantize.input[1]), ('zero', quantize.input[2]),
                  ('dq_scale', dq.input[1]), ('dq_zero', dq.input[2]),
                  ('clip_min', clip.input[1]), ('clip_max', clip.input[2])]}
    assert constants['scale'] == constants['dq_scale'] and constants['zero'] == constants['dq_zero']
    # Keep the trace free of node names: the original fusion accounting compares
    # source activation plus exact quantizer semantics across fused Conv members.
    return {'source_tensor': quantize.input[0], 'constants': constants,
            'q_attributes': [value.SerializeToString().hex() for value in quantize.attribute],
            'dq_attributes': [value.SerializeToString().hex() for value in dq.attribute],
            'activation_topology': TOPOLOGY}


def check_graph_requests(graph, requests, module_mapping):
    """Retain every original source-request check, changing only code Clip bounds."""
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
    if sum(row['precision'] == 'INT8' for row in requests) != 88 or sum(row['precision'] == 'FP16' for row in requests) != 0:
        errors.append('Corrected q_all requires 88 INT8 and zero FP16 requests')
    for row in requests:
        try:
            frozen = original[row['onnx_node']]
            assert row['module'] == frozen['pytorch_module'] and row['group'] == frozen['group']
            assert row['weight_input'] == frozen['weight_input']
            assert row['precision'] == 'INT8'
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


def audit_native_layers(layers, requests, graph, module_mapping):
    report = ORIGINAL_AUDIT_NATIVE_LAYERS(layers, requests, graph, module_mapping)
    report.update(activation_quantizer_topology=TOPOLOGY, audit_tool_identity=tool_identity(),
                  corrected_source_lineage=dict(SOURCE_CONTEXT),
                  native_clip_correction_evidence=dict(NATIVE_CORRECTION_CONTEXT),
                  prior_depthwise_integer_tactic_gaps_waived=False,
                  native_formats_tactics_and_fusion_criteria_changed=False)
    return report


def install_source_trace_extension():
    assert base.sha(Path(base.__file__)) == EXPECTED_BASE_SHA256, 'Original base auditor changed; review before reuse'
    base.activation_trace = activation_trace
    base.check_graph_requests = check_graph_requests
    base.audit_native_layers = audit_native_layers


def validate_corrected_source(spec_path, map_path, onnx_path):
    """Restrict this wrapper to the frozen full candidate and its rewrite audit."""
    import onnx
    assert base.sha(spec_path) == EXPECTED_SPEC_SHA256
    assert base.sha(spec_path.parent / 'manifest.json') == EXPECTED_MANIFEST_SHA256
    spec = base.read(spec_path)
    assert spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1'
    assert spec['configuration'] == 'q_all' and spec['diagnostic_only'] is True
    assert spec['activation_quantizer_topology'] == TOPOLOGY
    assert spec['requires_native_clip_importer_correction'] is True
    assert len(spec['clip_names']) == len(set(spec['clip_names'])) == 88
    assert spec['graph_sha256'] == base.sha(onnx_path) == EXPECTED_GRAPH_SHA256
    assert spec['parent_graph_sha256'] == EXPECTED_PARENT_GRAPH_SHA256
    assert spec['semantic_rewrite_audit_sha256'] == EXPECTED_REWRITE_AUDIT_SHA256
    assert spec['common_preparation_sha256'] == spec['semantic_rewrite_audit_sha256']
    assert spec['common_preparation_sha256'] != spec['parent_common_preparation_sha256']
    assert spec['native_diagnostic_reference_contract_sha256'] == EXPECTED_REFERENCE_CONTRACT_SHA256
    contract_path = spec_path.parent / 'reference_contract.json'
    assert base.sha(contract_path) == EXPECTED_REFERENCE_CONTRACT_SHA256
    contract = base.read(contract_path)
    assert contract['not_a_gate_amendment'] is True and contract['gate05_status'] == 'INCOMPLETE'
    assert contract['protected_failure']['original_complete_model_guard'] == 'FAILED_UNCHANGED'
    assert contract['protected_failure']['tolerances_changed'] is False
    assert contract['candidate_rewrite']['quantized_convs'] == contract['candidate_rewrite']['native_clip_count'] == 88
    assert contract['native_build']['attempts'] == 1 and contract['native_build']['max_seconds'] == 7200
    assert contract['native_build']['new_precisionConstraints'] == 'obey'
    assert contract['native_build']['explicit_restored_fp16_layer_constraints'] == 0
    assert spec['quantized_conv_count'] == 88 and spec['fp16_conv_count'] == 0
    assert spec['precisionConstraints'] == 'obey' and spec['workspace_mib'] == 1024
    assert spec['avg_timing'] == 1 and spec['max_build_seconds'] == 7200
    assert spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED'
    assert spec['module_map_sha256'] == base.sha(map_path)
    audit_path = spec_path.parent / 'semantic_rewrite_audit.json'
    assert base.sha(audit_path) == spec['semantic_rewrite_audit_sha256']
    rewrite = base.read(audit_path)
    assert rewrite['parent_graph_sha256'] == EXPECTED_PARENT_GRAPH_SHA256
    assert rewrite['candidate_graph_sha256'] == EXPECTED_GRAPH_SHA256
    assert rewrite['clip_names'] == spec['clip_names'] and len(rewrite['rewrites']) == 88
    graph = base.graph_index(onnx_path)
    assert {name: hashlib.sha256(node.SerializeToString()).hexdigest() for name, node in graph['nodes'].items()} == \
        rewrite['candidate_nodes_serialized_sha256']
    assert {name: hashlib.sha256(value.SerializeToString()).hexdigest() for name, value in graph['initializers'].items()} == \
        rewrite['candidate_initializers_serialized_sha256']
    onnx.checker.check_model(str(onnx_path), full_check=True)
    mapping = base.read(map_path)['mapping']
    errors = check_graph_requests(graph, spec['requests'], mapping)
    SOURCE_CONTEXT.update(onnx_sha256=base.sha(onnx_path), spec_sha256=base.sha(spec_path),
                          module_map_sha256=base.sha(map_path), parent_graph_sha256=EXPECTED_PARENT_GRAPH_SHA256,
                          semantic_rewrite_audit_sha256=EXPECTED_REWRITE_AUDIT_SHA256,
                          native_diagnostic_reference_contract_sha256=EXPECTED_REFERENCE_CONTRACT_SHA256)
    return spec, graph, errors


def validate_native_correction_evidence(build_path, spec):
    """Authenticate the new parser correction records without weakening native audit."""
    result = base.read(build_path / 'build_result.json')
    request = base.read(build_path / 'build_request.json')
    parsed = base.read(build_path / 'parsed_network_validation.json')
    before = base.read(build_path / 'parsed_clip_attributes_before_correction.json')
    after = base.read(build_path / 'parsed_clip_attributes_after_correction.json')
    identity = base.read(build_path / 'tool_hashes.json')
    assert identity == request['tool_identity'] == result['tool_identity']
    assert identity['builder_sha256'] == EXPECTED_BUILDER_SHA256
    assert identity['helper_sha256'] == EXPECTED_BUILD_HELPER_SHA256
    assert request['requested_precision_constraints_mode'] == result['requested_precision_constraints_mode'] == 'obey'
    assert request['builder_flags'] == ['INT8', 'FP16', 'OBEY_PRECISION_CONSTRAINTS',
                                         'PREFER_PRECISION_CONSTRAINTS_DISABLED', 'TF32_DISABLED']
    assert request['workspace_mib'] == 1024 and request['avg_timing_iterations'] == 1
    assert request['max_build_seconds'] == 7200 and request['build_budget_rationale']
    assert request['explicit_layer_precision_constraint_count'] == parsed['explicit_layer_precision_constraint_count'] == 0
    assert request['restored_fp16_compute_and_output_constraints'] == parsed['restored_fp16_compute_and_output_constraints'] == []
    assert request['original_complete_model_numerical_guard'] == result['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED'
    assert request['common_preparation_sha256'] == result['common_preparation_sha256'] == spec['common_preparation_sha256']
    assert request['native_diagnostic_reference_contract_sha256'] == result['native_diagnostic_reference_contract_sha256'] == EXPECTED_REFERENCE_CONTRACT_SHA256
    assert parsed['source_conv_count'] == 88 and parsed['requests'] == spec['requests']
    assert before['count'] == after['count'] == len(before['layers']) == len(after['layers']) == 88
    assert after['layers'] == request['explicit_clip_importer_corrections'] == parsed['clip_importer_corrections']
    prior = {row['name']: row for row in before['layers']}
    corrected = {row['name']: row for row in after['layers']}
    assert len(prior) == len(corrected) == 88 and set(prior) == set(corrected) == set(spec['clip_names'])
    for name, row in corrected.items():
        assert row['before'] == {key: prior[name][key] for key in ('alpha', 'beta')}
        assert row['after'] == {'alpha': -127.0, 'beta': 127.0}
    NATIVE_CORRECTION_CONTEXT.update(
        status='ALL_88_PARSED_CLIP_CORRECTIONS_RECORDED', count=88,
        precision_constraints_mode='obey', explicit_layer_precision_constraint_count=0,
        builder_sha256=identity['builder_sha256'], helper_sha256=identity['helper_sha256'],
        evidence_file_sha256={name: base.sha(build_path / name) for name in
                              ['build_request.json', 'parsed_network_validation.json',
                               'parsed_clip_attributes_before_correction.json',
                               'parsed_clip_attributes_after_correction.json', 'tool_hashes.json']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-source-only', action='store_true')
    args = parser.parse_args()
    install_source_trace_extension()
    spec, graph, errors = validate_corrected_source(args.spec, args.map, args.onnx)
    if args.check_source_only:
        report = {'schema': 'rampq.gate05.corrected_qall_precision_source_audit/v1',
                  'status': 'CORRECTED_SOURCE_STRUCTURE_VERIFIED' if not errors else 'SOURCE_STRUCTURE_REVIEW_REQUIRED',
                  'configuration': spec['configuration'], 'source_convs': 88,
                  'quantized_conv_count': 88, 'fp16_conv_count': 0, 'errors': errors,
                  'activation_quantizer_topology': TOPOLOGY, 'clip_names': spec['clip_names'],
                  'activation_traces': {row['onnx_node']: activation_trace(graph['nodes'][row['onnx_node']], graph)
                                        for row in spec['requests'] if row['precision'] == 'INT8'} if not errors else {},
                  'lineage': dict(SOURCE_CONTEXT), 'audit_tool_identity': tool_identity(),
                  'native_formats_tactics_and_fusion_criteria_changed': False,
                  'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                  'prior_depthwise_integer_tactic_gaps_waived': False,
                  'native_precision_audit_run': False, 'complete_model_inference_run': False,
                  'limits': 'Source structure only. No native formats, tactic adoption, fusion, numerical migration, accuracy or performance pass.'}
        if args.output:
            with args.output.open('x', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write('\n')
        print(json.dumps(report, indent=2))
        if errors:
            raise SystemExit(2)
        return
    assert args.build is not None, '--build required unless --check-source-only'
    assert not errors, '\n'.join(errors)
    validate_native_correction_evidence(args.build, spec)
    # Original CLI, original build/manifest lineage, original native format and
    # tactic decisions, and original exit codes are preserved by this call.
    base.main()


if __name__ == '__main__':
    main()

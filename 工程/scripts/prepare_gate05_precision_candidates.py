"""Prepare only the already planned Gate 0.5 precision graphs, without inference.

FLOAT ONNX tensors do not enforce FP16 execution. Every future engine needs
native precision constraints and a separate inspector audit.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
INT8 = ROOT / 'artifacts/gate05/20261018_int8'
CONFIGURATIONS = [
    ('fp16_control', set(), False),
    ('q_all', {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'}, False),
    ('r_only_G1', {'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'}, False),
    ('r_only_G1_G5', {'G2', 'G3', 'G4', 'G6', 'G7', 'G8'}, False),
    ('r_only_G1_G2', {'G3', 'G4', 'G5', 'G6', 'G7', 'G8'}, False),
    ('q_only_G1', {'G1'}, True),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def node_bytes(node):
    return node.SerializeToString()


def main():
    import onnx
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/gate05/20261019_precision_candidates')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing evidence; choose a fresh output directory'
    paths = {
        'source_fp32_onnx': ROOT / 'artifacts/jetson_trt_fp16_20261016_v1/model.onnx',
        'source_q_all_pe_tap_onnx': INT8 / 'pe_tap_full_graph_candidate/model_q_all_pe_tap.onnx',
        'source_candidate': INT8 / 'pe_tap_full_graph_candidate/candidate.json',
        'source_module_map': INT8 / 'input_v1/module_map.json',
        'source_calibration': INT8 / 'input_v1/calibration.json',
        'source_protocol': INT8 / 'input_v1/protocol.json',
        'source_failed_preflight': INT8 / 'input_v1/preflight.json',
        'preparation_script': Path(__file__).resolve(),
    }
    hashes_before = {name: sha(path) for name, path in paths.items()}
    candidate = json.loads(paths['source_candidate'].read_text(encoding='utf-8'))
    protocol = json.loads(paths['source_protocol'].read_text(encoding='utf-8'))
    preflight = json.loads(paths['source_failed_preflight'].read_text(encoding='utf-8'))
    mapping = json.loads(paths['source_module_map'].read_text(encoding='utf-8'))['mapping']
    assert preflight['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert hashes_before['source_fp32_onnx'] == protocol['source_onnx_sha256']
    assert hashes_before['source_q_all_pe_tap_onnx'] == candidate['candidate_sha256']
    assert hashes_before['source_calibration'] == protocol['calibration_sha256']
    assert len(mapping) == len({row['onnx_node'] for row in mapping}) == 88
    expected_groups = {'G1': 6, 'G2': 5, 'G3': 10, 'G4': 19, 'G5': 8, 'G6': 15, 'G7': 8, 'G8': 17}
    assert {group: sum(row['group'] == group for row in mapping) for group in expected_groups} == expected_groups
    original = onnx.load(str(paths['source_fp32_onnx']))
    q_all = onnx.load(str(paths['source_q_all_pe_tap_onnx']))
    source_nodes = {node.name: node for node in original.graph.node}
    quantized_nodes = {node.name: node for node in q_all.graph.node}
    source_initializers = {value.name: value for value in original.graph.initializer}
    q_all_initializers = {value.name: value for value in q_all.graph.initializer}
    assert {node.name for node in original.graph.node if node.op_type == 'Conv'} == {row['onnx_node'] for row in mapping}
    quantizer_node_names = set()
    for row in mapping:
        prefix = row['qdq_prefix']
        row['added_node_names'] = [prefix + '_' + suffix for suffix in
                                  ('Clip', 'ActivationQ', 'ActivationDQ', 'WeightQ', 'WeightDQ')]
        quantizer_node_names.update(row['added_node_names'])
        assert all(name in quantized_nodes for name in row['added_node_names'])
    added_initializer_names = set(q_all_initializers) - set(source_initializers)
    assert len(quantizer_node_names) == 440 and len(added_initializer_names) == 88 * 6
    assert set(quantized_nodes) - set(source_nodes) == quantizer_node_names
    for name, value in source_initializers.items():
        assert node_bytes(value) == node_bytes(q_all_initializers[name]), name
    expected_outputs = [node_bytes(value) for value in q_all.graph.output]
    assert [value.name for value in q_all.graph.output] == ['output0', candidate['extra_output']]
    expected_binding_contract = [
        {'name': 'images', 'input': True, 'shape': [1, 3, 640, 640], 'dtype': 'DataType.FLOAT'},
        {'name': 'output0', 'input': False, 'shape': [1, 14, 8400], 'dtype': 'DataType.FLOAT'},
        {'name': candidate['extra_output'], 'input': False,
         'shape': candidate['extra_output_shape'], 'dtype': 'DataType.FLOAT'},
    ]
    source_by_name = {row['onnx_node']: row for row in mapping}
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for configuration, quantized_groups, optional in CONFIGURATIONS:
        folder = args.output / configuration
        folder.mkdir()
        graph = copy.deepcopy(q_all)
        restored = [row for row in mapping if row['group'] not in quantized_groups]
        quantized = [row for row in mapping if row['group'] in quantized_groups]
        removed_names = {name for row in restored for name in row['added_node_names']}
        kept_nodes = []
        for node in graph.graph.node:
            if node.name in removed_names:
                continue
            row = source_by_name.get(node.name)
            if row is not None and row['group'] not in quantized_groups:
                node.input[0] = source_nodes[node.name].input[0]
                node.input[1] = source_nodes[node.name].input[1]
            kept_nodes.append(node)
        del graph.graph.node[:]
        graph.graph.node.extend(kept_nodes)
        used_inputs = {name for node in graph.graph.node for name in node.input}
        used_inputs.update(value.name for value in graph.graph.output)
        kept_initializers = [value for value in graph.graph.initializer
                             if value.name in source_initializers or value.name in used_inputs]
        del graph.graph.initializer[:]
        graph.graph.initializer.extend(kept_initializers)
        nodes = {node.name: node for node in graph.graph.node}
        initializers = {value.name: value for value in graph.graph.initializer}
        assert {node.name for node in graph.graph.node if node.op_type == 'Conv'} == set(source_by_name)
        assert len(graph.graph.node) == len(original.graph.node) + len(quantized) * 5
        assert set(initializers) - set(source_initializers) == {
            row['qdq_prefix'] + '_' + suffix for row in quantized for suffix in
            ('a_scale', 'a_zero', 'clip_min', 'clip_max', 'w_scale', 'w_zero')}
        assert [node_bytes(value) for value in graph.graph.output] == expected_outputs
        assert [node_bytes(value) for value in graph.graph.input] == [node_bytes(value) for value in q_all.graph.input]
        for name, value in source_initializers.items():
            assert node_bytes(value) == node_bytes(initializers[name]), name
        for name in set(initializers) - set(source_initializers):
            assert node_bytes(initializers[name]) == node_bytes(q_all_initializers[name]), name
        canonical_arithmetic = []
        for node in graph.graph.node:
            if node.name in quantizer_node_names:
                assert node_bytes(node) == node_bytes(quantized_nodes[node.name]), node.name
                continue
            canonical = copy.deepcopy(node)
            if canonical.op_type == 'Conv':
                # This comparison removes only intended quantization-edge rewiring.
                canonical.input[0] = source_nodes[node.name].input[0]
                canonical.input[1] = source_nodes[node.name].input[1]
            canonical_arithmetic.append(node_bytes(canonical))
        assert canonical_arithmetic == [node_bytes(node) for node in original.graph.node]
        row_audits = []
        for row in mapping:
            is_quantized = row['group'] in quantized_groups
            conv = nodes[row['onnx_node']]
            expected = quantized_nodes[conv.name] if is_quantized else source_nodes[conv.name]
            assert node_bytes(conv) == node_bytes(expected), conv.name
            assert all((name in nodes) == is_quantized for name in row['added_node_names'])
            row_audits.append({'module': row['pytorch_module'], 'onnx_node': row['onnx_node'],
                               'group': row['group'], 'quantized': is_quantized,
                               'activation_source': source_nodes[conv.name].input[0],
                               'original_weight_initializer': row['weight_input'],
                               'desired_engine_precision': 'INT8_INPUT_AND_WEIGHT' if is_quantized else 'FP16',
                               'frozen_activation_scale': row['activation_scale'] if is_quantized else None,
                               'frozen_weight_scale': row['weight_scale'] if is_quantized else None})
        onnx.checker.check_model(graph, full_check=True)
        destination = folder / 'model.onnx'
        if configuration == 'q_all':
            # Preserve the reference graph's exact file bytes, not just its semantics.
            shutil.copy2(paths['source_q_all_pe_tap_onnx'], destination)
            assert sha(destination) == hashes_before['source_q_all_pe_tap_onnx']
        else:
            onnx.save(graph, str(destination))
        group_counts = {group: {'total': count,
                               'quantized': sum(row['group'] == group for row in quantized),
                               'desired_restored_fp16': sum(row['group'] == group for row in restored)}
                        for group, count in expected_groups.items()}
        audit = {
            'status': 'GRAPH_PREPARED_AND_STRUCTURALLY_AUDITED_ONLY',
            'configuration': configuration, 'optional_bidirectional_side': optional,
            'source_conv_count': 88, 'quantized_conv_count': len(quantized),
            'desired_restored_fp16_conv_count': len(restored), 'groups': group_counts,
            'graph_sha256': sha(destination), 'common_expected_binding_contract': expected_binding_contract,
            'source_arithmetic_identical_after_removing_only_quantization_edge_rewiring': True,
            'all_source_initializer_bytes_identical': True,
            'retained_qdq_nodes_and_initializer_bytes_identical_to_frozen_reference': True,
            'removed_only_restored_qdq_clip_nodes_and_unused_added_quantizer_initializers': True,
            'common_named_main_and_auxiliary_output_declarations_byte_identical': True,
            'onnx_checker_full_check': 'PASS', 'whole_model_engine_built': False,
            'single_image_inference_run': False, 'dataset_evaluation_run': False,
            'local_numerical_migration_status': 'ORIGINAL_LOCAL_QDQ_PREFLIGHT_FAILED_RETAINED_NO_NEW_PASS',
            'planned_native_precision_control': {
                'precisionConstraints': 'obey', 'status': 'PENDING_IMPLEMENTATION_AND_NATIVE_VALIDATION',
                'restored_conv_precision_requests': {row['onnx_node']: 'fp16' for row in restored},
                'quantized_conv_precision': 'Explicit unchanged signed INT8 Q/DQ on inputs and weights',
                'graph_float_does_not_enforce_fp16': True,
                'actual_all_conv_precision_and_fusion_audit_required': True,
                'nonquantized_floating_precision_control_across_engines': 'PENDING_PROTOCOL_AND_ENGINE_VALIDATION'},
            'mapping': row_audits,
            'limits': 'Graphs only. Common output declarations do not guarantee common TensorRT fusion or floating '
                      'precision. No engine, execution, numerical pass, target rescue, bidirectional or performance claim.',
        }
        write(folder / 'graph_audit.json', audit)
        summaries.append({key: audit[key] for key in
                          ('configuration', 'optional_bidirectional_side', 'quantized_conv_count',
                           'desired_restored_fp16_conv_count', 'graph_sha256', 'status')})
    assert {name: sha(path) for name, path in paths.items()} == hashes_before
    write(args.output / 'preparation.json', {
        'schema': 'rampq.gate05.frozen_precision_candidates/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(),
        'status': 'PREPARED_ONLY_NOT_BUILT_OR_EXECUTED',
        'purpose': 'Implement the five previously frozen configurations plus explicitly optional q_only:G1 side',
        'source_paths': {name: str(path) for name, path in paths.items()},
        'source_sha256': hashes_before, 'original_sources_unchanged': True,
        'common_expected_binding_contract': expected_binding_contract,
        'configurations': summaries, 'new_calibration': False, 'dataset_evaluation_run': False,
        'original_numerical_failure_retained': preflight['status'],
        'pending_decisions': [
            'Native implementation and actual adoption of restored Conv FP16 constraints',
            'Common nonquantized floating precision/fusion controls before attributing target changes',
            'How to resolve or explicitly account for original independent-model numerical guard failure',
            'Freeze dataset PASS/FAIL/INCONCLUSIVE rules before any fixed-dataset evaluation'],
    })
    artifact_files = sorted(path for path in args.output.rglob('*') if path.is_file())
    files = {path.relative_to(args.output).as_posix(): {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in artifact_files}
    write(args.output / 'manifest.json', {'schema': 'rampq.gate05.precision_candidate_files/v1', 'files': files})
    (args.output / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n'
                                                for name, info in files.items()), encoding='utf-8')
    print(json.dumps({'status': 'GRAPH_PREPARATION_ONLY_COMPLETE', 'configurations': summaries,
                      'artifact_files': len(files), 'original_sources_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()

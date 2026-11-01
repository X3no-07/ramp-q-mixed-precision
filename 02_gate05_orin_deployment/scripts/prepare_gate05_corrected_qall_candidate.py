"""Statically prepare a full q_all candidate with 88 code-space Clips.

No ONNX Runtime, PyTorch, CUDA, full-model inference, or remote calls are used.
All parent files and previous evidence remain untouched. A future native builder
must correct the 88 imported Clip alpha/beta values before building the engine.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles/q_all'
PARENT_GRAPH_SHA256 = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
INPUT_SHA256 = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
ARCHIVE_ROOT = 'gate05_corrected_qall_candidate_20261019'
EXTENDED_MONITOR = ROOT / 'artifacts/gate05/20261019_qall_extended/post_smoke_evidence/monitor/status.json'
EXTENDED_BUILD = ROOT / 'artifacts/gate05/20261019_qall_extended/post_smoke_evidence/build/build_result.json'
REFERENCE_CONTRACT = ROOT / 'artifacts/gate05/20261019_reference_contract/reference_contract.json'
REFERENCE_CONTRACT_SHA256 = '87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430'
PREPARATION_ATTEMPT1 = ROOT / 'artifacts/gate05/20261019_corrected_qall_preparation_attempt1.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def source_hashes(paths):
    return {path.resolve().relative_to(ROOT).as_posix(): info(path) for path in sorted(set(paths))}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def field_changes(parent, candidate):
    result = {}
    for name in sorted(set(parent) | set(candidate)):
        state = 'RETAINED_PARENT_VALUE' if name in parent and name in candidate and parent[name] == candidate[name] else (
            'UPDATED_PARENT_VALUE' if name in parent and name in candidate else
            'NEW_CANDIDATE_FIELD' if name in candidate else 'REMOVED_PARENT_FIELD')
        row = {'state': state}
        if name in parent:
            row['parent_value_sha256'] = json_sha(parent[name])
        if name in candidate:
            row['candidate_value_sha256'] = json_sha(candidate[name])
        if state != 'RETAINED_PARENT_VALUE':
            if name in parent:
                row['parent_value'] = parent[name]
            if name in candidate:
                row['candidate_value'] = candidate[name]
            row['change_explanation'] = explanation(name)
        result[name] = row
    return result


def explanation(name):
    specific = {
        'max_build_seconds': 'Extended to 7200 seconds for this one q_all build; the retained prior q_all monitor allowed 7200 and its build took 3276.426 seconds. This is a runtime budget change, not a numerical tolerance change.',
        'common_preparation_sha256': 'New code-space Clip semantic rewrite audit identifies this candidate preparation; the old preparation remains separately recorded and is not a matched floating control.',
        'graph_sha256': 'Hash of the exact new graph containing only the 88 Clip/Q rewrites and 176 bound constant replacements.',
        'graph_audit_sha256': 'Hash of the updated structural audit; old graph audit is preserved unchanged under parent_.',
        'limits': 'Limits updated to state the diagnostic rewrite, pending native attribute correction, original failed guard and unmatched floating control.',
        'planned_native_precision_control': 'Retains parent obey and zero explicit Half restore requests; describes all 88 new INT8 code-space Clips and pending native adoption proof.',
        'bundle_revision': 'Distinct revision for the q_all code-space Clip candidate retaining obey with a separately extended runtime budget.',
    }
    if name in specific:
        return specific[name]
    if name.startswith('parent_') or name == 'inherited_source_arithmetic_claim_basis':
        return 'Retains exact parent lineage and hash provenance without elevating prior checks to a new candidate pass.'
    if name in ('all_source_initializer_bytes_identical', 'retained_qdq_nodes_and_initializer_bytes_identical_to_frozen_reference', 'removed_only_restored_qdq_clip_nodes_and_unused_added_quantizer_initializers'):
        return 'Withdraws the inherited all-bytes-unchanged assertion because exactly 176 INT8 bound replacements and 176 Clip/Q node rewires are now explicit.'
    if name in ('matched_fp16_control_status', 'nonquantized_float_control_status'):
        return 'No matched FP16 arithmetic/fusion control is established for the new preparation.'
    if name in ('mapping', 'clip_names', 'activation_quantizer_topology', 'native_clip_importer_correction', 'requires_native_clip_importer_correction'):
        return 'Records all 88 permitted code-space Clip rewrites and the required same-network native importer alpha/beta correction.'
    if name in ('full_model_cpu_equivalence_status', 'complete_model_cpu_inference_run', 'native_build_or_execution_run_by_preparer', 'whole_model_engine_built', 'single_image_inference_run'):
        return 'Reports static preparation scope only; no ORT/PyTorch/full-model/native inference is executed by this preparer.'
    if name in ('original_complete_model_numerical_guard', 'original_tolerances_changed', 'local_numerical_migration_status'):
        return 'Preserves the original failed full-model numerical guard and its original tolerance values; no replacement pass is declared.'
    if name in ('calibration_recomputed', 'weights_changed', 'dataset_evaluation_run'):
        return 'Explicitly records that this preparation neither recalibrates nor changes weights nor evaluates a dataset.'
    if name in ('input_file', 'input_file_sha256', 'generic_smoke_image_scope'):
        return 'Copies the unchanged fixed 000020 preprocessed input for optional later CPU comparison; generic native smoke continues to use the unchanged source image.'
    if name in ('semantic_rewrite_audit_sha256', 'all_nonbound_initializer_bytes_identical_to_parent', 'all_nodes_except_88_activation_clip_and_q_pairs_identical_to_parent', 'all_weight_qdq_and_weight_bias_bytes_identical_to_parent'):
        return 'States the exhaustive byte-level difference audit and preserves all unaffected arithmetic, quantizer, weight, bias and output bytes.'
    if name in ('common_source_provenance_scope', 'parent_common_source_provenance_fields'):
        return 'Distinguishes unchanged source provenance from the newly created preparation identity.'
    if name == 'prior_native_auditor_interface':
        return 'Records the old FloatClip-only auditor interface and the need for a new strict INT8-code-Clip source trace.'
    if name == 'build_budget_provenance':
        return 'Pins the prior retained 7200-second monitor and 3276.426-second q_all build as provenance for one extended candidate build.'
    if name == 'native_diagnostic_reference_contract_sha256':
        return 'Pins the newly frozen diagnostic reference contract, copied unchanged into this bundle; this does not replace the original numerical guard or dataset gate protocol.'
    if name == 'status':
        return 'Updates the artifact state to static code-space Clip preparation only, without claiming numeric or native success.'
    if name == 'prepared_client_at':
        return 'Records the preparation time on the client, separate from the board clock.'
    raise RuntimeError('Missing change explanation for field: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Choose a fresh output; retain earlier evidence')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    from onnx import numpy_helper

    manifest = read(PARENT / 'manifest.json')
    source_paths = [PARENT / 'manifest.json', Path(__file__).resolve()]
    for name, expected in manifest['files'].items():
        path = PARENT / name
        require(path.resolve().parent == PARENT.resolve() and not path.is_symlink() and info(path) == expected,
                'Frozen parent manifest mismatch: ' + name)
        source_paths.append(path)
    if (PARENT / 'SHA256SUMS').exists():
        source_paths.append(PARENT / 'SHA256SUMS')
    input_parent = ROOT / 'artifacts/gate05/20261019_firstconv_numeric_probe'
    source_paths.extend([input_parent / 'manifest.json', input_parent / 'preprocessed_input.npy'])
    source_paths.extend([EXTENDED_MONITOR, EXTENDED_BUILD, REFERENCE_CONTRACT, PREPARATION_ATTEMPT1])
    require(sha(REFERENCE_CONTRACT) == REFERENCE_CONTRACT_SHA256,
            'Frozen native diagnostic reference contract differs')
    require(info(input_parent / 'preprocessed_input.npy') == read(input_parent / 'manifest.json')['files']['preprocessed_input.npy'] and
            sha(input_parent / 'preprocessed_input.npy') == INPUT_SHA256, 'Frozen input bytes differ')
    before = source_hashes(source_paths)
    parent_spec = read(PARENT / 'spec.json')
    extended_monitor, extended_build = read(EXTENDED_MONITOR), read(EXTENDED_BUILD)
    require(extended_monitor['max_seconds'] == 7200 and extended_monitor['status'] == 'BUILD_SCRIPT_OK' and
            extended_build['onnx_sha256'] == PARENT_GRAPH_SHA256 and
            extended_build['status'] == 'BUILT_AND_DESERIALIZED' and extended_build['elapsed_seconds'] == 3276.426,
            'Prior extended q_all build provenance differs')
    parent_audit = read(PARENT / 'graph_audit.json')
    module_map = read(PARENT / 'module_map.json')
    require(parent_spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1' and
            parent_spec['configuration'] == 'q_all' and parent_spec['quantized_conv_count'] == 88 and
            parent_spec['fp16_conv_count'] == 0 and parent_spec['precisionConstraints'] == 'obey' and
            parent_spec['workspace_mib'] == 1024 and parent_spec['avg_timing'] == 1 and parent_spec['max_build_seconds'] == 1800,
            'Parent build interface differs')
    require(sha(PARENT / 'model.onnx') == parent_spec['graph_sha256'] == PARENT_GRAPH_SHA256,
            'Frozen parent q_all graph differs')
    for filename, field in [('graph_audit.json', 'graph_audit_sha256'), ('module_map.json', 'module_map_sha256'),
                            ('preflight.json', 'failed_preflight_sha256'), ('diagnostic_scope.json', 'diagnostic_scope_sha256')]:
        require(sha(PARENT / filename) == parent_spec[field], 'Frozen parent reference differs: ' + filename)
    require(read(PARENT / 'preflight.json')['status'] == parent_spec['original_local_preflight_status'] ==
            'LOCAL_QDQ_PREFLIGHT_FAILED', 'Original failed full-model guard must remain failed')
    source = onnx.load(str(PARENT / 'model.onnx'))
    candidate = copy.deepcopy(source)
    source_nodes = {node.name: node for node in source.graph.node}
    source_constants = {value.name: value for value in source.graph.initializer}
    producers = {name: node for node in source.graph.node for name in node.output}
    consumers = {}
    for node in source.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node.name)
    patched_nodes = {node.name: node for node in candidate.graph.node}
    patched_constants = {value.name: value for value in candidate.graph.initializer}
    mapping = {row['onnx_node']: row for row in module_map['mapping']}
    requests = parent_spec['requests']
    require(len(requests) == len(mapping) == len({row['onnx_node'] for row in requests}) == 88 and
            {node.name for node in source.graph.node if node.op_type == 'Conv'} == set(mapping),
            'Expected 88 unique original Conv identities')
    int8_requests = [row for row in requests if row['precision'] == 'INT8']
    require(len(int8_requests) == 88 and all(row['precision'] == 'INT8' for row in requests) and
            sum(row['precision'] == 'FP16' for row in requests) == 0, 'Expected unchanged 88 INT8 / 0 FP16 request partition')
    used_names = set(source_constants) | set(producers)
    used_names.update(value.name for value in source.graph.input)
    used_names.update(value.name for value in source.graph.output)
    used_names.update(value.name for value in source.graph.value_info)
    rewrites, clip_to_q, changed_nodes, changed_constants = [], {}, set(), set()
    expected_changed_nodes, expected_changed_constants = {}, {}
    for request in int8_requests:
        conv = source_nodes[request['onnx_node']]
        frozen = mapping[conv.name]
        require(request['module'] == frozen['pytorch_module'] and request['group'] == frozen['group'] and
                request['weight_input'] == frozen['weight_input'], 'Frozen module request mismatch')
        dq = producers[conv.input[0]]
        require(dq.op_type == 'DequantizeLinear' and len(dq.input) == 3, 'Frozen activation DQ missing')
        q = producers[dq.input[0]]
        require(q.op_type == 'QuantizeLinear' and len(q.input) == 3, 'Frozen activation Q missing')
        clip = producers[q.input[0]]
        require(clip.op_type == 'Clip' and len(clip.input) == 3 and clip.input[0] == request['original_activation_input'],
                'Frozen activation Float Clip source differs')
        require(not q.attribute and not dq.attribute and not clip.attribute and list(q.input[1:]) == list(dq.input[1:]),
                'Expected unchanged scalar activation Q/DQ')
        require(consumers[clip.output[0]] == [q.name] and consumers[q.output[0]] == [dq.name],
                'Unexpected side consumer would change rewrite semantics')
        scale = numpy_helper.to_array(source_constants[q.input[1]])
        zero = numpy_helper.to_array(source_constants[q.input[2]])
        lower = numpy_helper.to_array(source_constants[clip.input[1]])
        upper = numpy_helper.to_array(source_constants[clip.input[2]])
        require(scale.dtype == np.float32 and scale.ndim == 0 and float(scale) > 0 and
                zero.dtype == np.int8 and zero.ndim == 0 and int(zero) == 0 and
                scale.tobytes() == np.asarray(frozen['activation_scale'], np.float32).tobytes(),
                'Expected frozen FLOAT scalar scale and signed INT8 zero=0')
        require(lower.dtype == upper.dtype == np.float32 and lower.ndim == upper.ndim == 0 and
                np.float32(scale * np.float32(-127)).tobytes() == lower.tobytes() and
                np.float32(scale * np.float32(127)).tobytes() == upper.tobytes() and
                int(np.rint(np.float32(lower / scale))) == -127 and
                int(np.rint(np.float32(upper / scale))) == 127,
                'Parent Float Clip bounds are not exact FLOAT32 symmetric 127 times scale')
        new_tensor = q.output[0] + '_before_code_clip'
        require(new_tensor not in used_names, 'New code tensor name collision')
        used_names.add(new_tensor)
        new_q, new_clip = patched_nodes[q.name], patched_nodes[clip.name]
        new_q.input[0], new_q.output[0] = clip.input[0], new_tensor
        new_clip.input[0], new_clip.output[0] = new_tensor, q.output[0]
        expected_q, expected_clip = copy.deepcopy(q), copy.deepcopy(clip)
        expected_q.input[0], expected_q.output[0] = clip.input[0], new_tensor
        expected_clip.input[0], expected_clip.output[0] = new_tensor, q.output[0]
        expected_changed_nodes[q.name], expected_changed_nodes[clip.name] = expected_q, expected_clip
        constant_changes = []
        for name, bound in zip(clip.input[1:], (-127, 127)):
            old = source_constants[name]
            replacement = numpy_helper.from_array(np.asarray(bound, dtype=np.int8), name)
            patched_constants[name].CopyFrom(replacement)
            expected_changed_constants[name] = replacement
            changed_constants.add(name)
            constant_changes.append({'name': name, 'parent_dtype': 'float32',
                                     'parent_value': float(numpy_helper.to_array(old)), 'candidate_dtype': 'int8',
                                     'candidate_value': bound, 'parent_serialized_sha256': message_sha(old),
                                     'candidate_serialized_sha256': message_sha(replacement)})
        changed_nodes.update((clip.name, q.name))
        clip_to_q[clip.name] = q.name
        rewrites.append({'module': request['module'], 'conv_name': conv.name, 'group': request['group'],
                         'clip_name': clip.name, 'q_name': q.name, 'dq_name_unchanged': dq.name,
                         'parent_order': [clip.name, q.name, dq.name], 'candidate_order': [q.name, clip.name, dq.name],
                         'scale_initializer_unchanged': q.input[1], 'scale_float32': float(scale),
                         'zero_initializer_unchanged': q.input[2], 'zero_int8': int(zero),
                         'parent_float_clip_bounds': [float(lower), float(upper)], 'code_clip_bounds': [-127, 127],
                         'bound_scale_check': 'EXACT_FLOAT32_SIGNED_127_TIMES_SCALE',
                         'float32_bound_division': [float(np.float32(lower / scale)), float(np.float32(upper / scale))],
                         'nearest_even_quantized_bound_codes': [int(np.rint(np.float32(lower / scale))),
                                                               int(np.rint(np.float32(upper / scale)))],
                         'q_input_0': {'parent': q.input[0], 'candidate': new_q.input[0]},
                         'q_output_0': {'parent': q.output[0], 'candidate': new_q.output[0]},
                         'clip_input_0': {'parent': clip.input[0], 'candidate': new_clip.input[0]},
                         'clip_output_0': {'parent': clip.output[0], 'candidate': new_clip.output[0]},
                         'dq_input_and_output_unchanged': True, 'bound_initializer_changes': constant_changes,
                         'parent_clip_serialized_sha256': message_sha(clip), 'parent_q_serialized_sha256': message_sha(q),
                         'candidate_clip_serialized_sha256': message_sha(new_clip),
                         'candidate_q_serialized_sha256': message_sha(new_q)})
    reordered = []
    q_names = set(clip_to_q.values())
    for node in source.graph.node:
        if node.name in clip_to_q:
            reordered.extend((copy.deepcopy(patched_nodes[clip_to_q[node.name]]), copy.deepcopy(patched_nodes[node.name])))
        elif node.name not in q_names:
            reordered.append(copy.deepcopy(patched_nodes[node.name]))
    del candidate.graph.node[:]
    candidate.graph.node.extend(reordered)
    require(len(changed_nodes) == len(changed_constants) == 176 and len(rewrites) == 88,
            'Expected exactly 88 Clip/Q pairs and 176 bound initializers')
    require(len(candidate.graph.node) == len(source.graph.node) and
            len(candidate.graph.initializer) == len(source.graph.initializer), 'Graph inventory changed')
    for node in candidate.graph.node:
        expected = expected_changed_nodes.get(node.name, source_nodes[node.name])
        require(node.SerializeToString() == expected.SerializeToString(), 'Unexpected node byte difference: ' + node.name)
    for value in candidate.graph.initializer:
        expected = expected_changed_constants.get(value.name, source_constants[value.name])
        require(value.SerializeToString() == expected.SerializeToString(), 'Unexpected initializer byte difference: ' + value.name)
    # Compare every model/graph field outside the approved node/constant lists.
    old_shell, new_shell = copy.deepcopy(source), copy.deepcopy(candidate)
    for shell in (old_shell, new_shell):
        del shell.graph.node[:]
        del shell.graph.initializer[:]
    require(old_shell.SerializeToString() == new_shell.SerializeToString(), 'Non-node/non-initializer model fields changed')
    require([value.SerializeToString() for value in source.graph.output] ==
            [value.SerializeToString() for value in candidate.graph.output], 'External output declarations changed')
    require(all(node.SerializeToString() == source_nodes[node.name].SerializeToString()
                for node in candidate.graph.node if node.op_type == 'Conv'), 'An original Conv changed')
    onnx.checker.check_model(candidate, full_check=True)

    output.mkdir(parents=True, exist_ok=False)
    onnx.save(candidate, str(output / 'model.onnx'))
    for filename in ('module_map.json', 'preflight.json', 'diagnostic_scope.json'):
        shutil.copy2(PARENT / filename, output / filename)
        require((PARENT / filename).read_bytes() == (output / filename).read_bytes(), 'Retained source copy differs')
    for filename in ('spec.json', 'graph_audit.json', 'manifest.json', 'reference_protocol.json'):
        shutil.copy2(PARENT / filename, output / ('parent_' + filename))
    shutil.copy2(REFERENCE_CONTRACT, output / 'reference_contract.json')
    require((output / 'reference_contract.json').read_bytes() == REFERENCE_CONTRACT.read_bytes(),
            'Native diagnostic reference contract copy differs')
    shutil.copy2(PREPARATION_ATTEMPT1, output / 'preparation_attempt1.json')
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    shutil.copy2(input_parent / 'preprocessed_input.npy', output / 'preprocessed_input.npy')
    write(output / 'source_hashes_before.json', before)
    checks = {'exactly_88_activation_clip_q_rewrites': True, 'exactly_176_changed_nodes': True,
              'exactly_176_changed_bound_initializers': True, 'new_quantized_tensor_names_unique': True,
              'all_88_original_bounds_exact_float32_signed_127_times_scale': True,
              'all_88_original_bounds_quantize_by_nearest_even_to_signed_127': True,
              'all_88_original_activation_zero_int8_scalar_zero': True,
              'all_other_node_bytes_identical_to_parent': True, 'all_other_initializer_bytes_identical_to_parent': True,
              'all_88_conv_bytes_identical_to_parent': True, 'all_weight_qdq_and_weight_bias_bytes_identical_to_parent': True,
              'all_activation_dq_scale_zero_bytes_identical_to_parent': True,
              'all_other_model_graph_fields_and_output_declarations_identical_to_parent': True,
              'request_partition_88_int8_0_fp16_unchanged': True, 'onnx_checker_full_check_pass': True,
              'original_failed_preflight_module_map_and_scope_bytes_copied': True}
    semantic_audit = {'schema': 'rampq.gate05.corrected_qall_semantic_rewrite_audit/v1',
                      'status': 'STATIC_BYTE_REWRITE_CHECKS_PASSED_NO_INFERENCE',
                      'prepared_client_at': datetime.now().astimezone().isoformat(),
                      'parent_graph_sha256': PARENT_GRAPH_SHA256, 'candidate_graph_sha256': sha(output / 'model.onnx'),
                      'parent_spec_sha256': sha(PARENT / 'spec.json'), 'parent_manifest_sha256': sha(PARENT / 'manifest.json'),
                      'parent_common_preparation_sha256': parent_spec['common_preparation_sha256'],
                      'source_conv_count': 88, 'source_node_count': len(source.graph.node),
                      'source_initializer_count': len(source.graph.initializer),
                      'candidate_node_count': len(candidate.graph.node), 'candidate_initializer_count': len(candidate.graph.initializer),
                      'first_preparation_attempt': {'file': 'preparation_attempt1.json', 'sha256': sha(PREPARATION_ATTEMPT1),
                          'status': 'STATIC_PREPARATION_STOPPED_BEFORE_OUTPUT_CREATION',
                          'reason': 'Exact division identity was overstrict for 22 FLOAT32 scales. Product byte identity and quantized bound codes remain exact for all 88. No source value or original guard tolerance changed.'},
                      'clip_names': [row['clip_name'] for row in rewrites], 'rewrites': rewrites, 'checks': checks,
                      'parent_nodes_serialized_sha256': {node.name: message_sha(node) for node in source.graph.node},
                      'candidate_nodes_serialized_sha256': {node.name: message_sha(node) for node in candidate.graph.node},
                      'parent_initializers_serialized_sha256': {value.name: message_sha(value) for value in source.graph.initializer},
                      'candidate_initializers_serialized_sha256': {value.name: message_sha(value) for value in candidate.graph.initializer},
                      'complete_model_cpu_inference_run': False, 'native_build_or_execution_run': False, 'cuda_used': False,
                      'calibration_recomputed': False, 'weights_changed': False, 'original_tolerances_changed': False,
                      'dataset_evaluation_run': False, 'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                      'semantic_equivalence_status': 'STRUCTURAL_REWRITE_CANDIDATE_ONLY_FULL_MODEL_NUMERICS_NOT_RUN',
                      'limits': 'Static byte checks establish the exact permitted rewrite only. Full-model equivalence and native importer correction/adoption remain untested.'}
    write(output / 'semantic_rewrite_audit.json', semantic_audit)
    rewrite_sha = sha(output / 'semantic_rewrite_audit.json')
    graph_audit = copy.deepcopy(parent_audit)
    graph_audit.update(status='CODE_CLIP_GRAPH_PREPARED_AND_STRUCTURALLY_AUDITED_ONLY',
                       graph_sha256=sha(output / 'model.onnx'), parent_graph_sha256=PARENT_GRAPH_SHA256,
                       parent_graph_audit_sha256=sha(PARENT / 'graph_audit.json'),
                       common_preparation_sha256=rewrite_sha,
                       parent_common_preparation_sha256=parent_spec['common_preparation_sha256'],
                       semantic_rewrite_audit_sha256=rewrite_sha,
                       all_source_initializer_bytes_identical=False,
                       retained_qdq_nodes_and_initializer_bytes_identical_to_frozen_reference=False,
                       removed_only_restored_qdq_clip_nodes_and_unused_added_quantizer_initializers=False,
                       all_nonbound_initializer_bytes_identical_to_parent=True,
                       all_nodes_except_88_activation_clip_and_q_pairs_identical_to_parent=True,
                       all_weight_qdq_and_weight_bias_bytes_identical_to_parent=True,
                       original_complete_model_numerical_guard='FAILED_UNCHANGED',
                       whole_model_engine_built=False, single_image_inference_run=False, dataset_evaluation_run=False,
                       clip_names=[row['clip_name'] for row in rewrites],
                       local_numerical_migration_status='ORIGINAL_LOCAL_QDQ_PREFLIGHT_FAILED_RETAINED_NO_NEW_PASS',
                       matched_fp16_control_status='NOT_ESTABLISHED_FOR_NEW_CODE_CLIP_PREPARATION',
                       parent_claim_scope='The parent structural audit remains in parent_graph_audit.json. This audit compares the candidate to its exact parent, with 176 bound initializer and 176 Clip/Q node changes.',
                       inherited_source_arithmetic_claim_basis='Every non-Clip/Q parent node, including all 88 Conv, is byte-identical; parent arithmetic provenance is retained, not newly evaluated.',
                       limits='Static preparation only. Changed common_preparation identity prevents treating older FP16 controls as matched. No new CPU/native inference, precision adoption, migration, rescue, accuracy or performance pass.')
    graph_audit['planned_native_precision_control'].update(
        precisionConstraints='obey', status='PENDING_CODE_CLIP_IMPORTER_CORRECTION_BUILD_AND_NATIVE_VALIDATION',
        quantized_conv_precision='Original signed INT8 Q/DQ scales, zero points, weights and DQ; 88 activation Clips moved after Q to INT8 [-127,+127].',
        nonquantized_floating_precision_control_across_engines='NOT_ESTABLISHED_FOR_NEW_CODE_CLIP_PREPARATION')
    rewrite_by_conv = {row['conv_name']: row for row in rewrites}
    for row in graph_audit['mapping']:
        if row['onnx_node'] in rewrite_by_conv:
            rewrite = rewrite_by_conv[row['onnx_node']]
            row.update(activation_quantizer_topology='Q_INT8_CODE_CLIP_DQ',
                       code_clip_name=rewrite['clip_name'], code_clip_bounds=[-127, 127],
                       native_importer_clip_attributes_require_correction=True,
                       semantic_rewrite_audit_sha256=rewrite_sha)
    write(output / 'graph_audit.json', graph_audit)
    auditor_path = ROOT / 'scripts/audit_gate05_precision_candidate.py'
    parent_auditor_observed_sha = sha(auditor_path)
    spec = copy.deepcopy(parent_spec)
    spec.update(bundle_revision='v3_corrected_code_clip_obey_extended',
                prepared_client_at=datetime.now().astimezone().isoformat(),
                graph_sha256=sha(output / 'model.onnx'), graph_audit_sha256=sha(output / 'graph_audit.json'),
                common_preparation_sha256=rewrite_sha,
                parent_spec_sha256=sha(PARENT / 'spec.json'), parent_graph_sha256=PARENT_GRAPH_SHA256,
                parent_bundle_manifest_sha256=sha(PARENT / 'manifest.json'),
                parent_graph_audit_sha256=sha(PARENT / 'graph_audit.json'),
                parent_common_preparation_sha256=parent_spec['common_preparation_sha256'],
                semantic_rewrite_audit_sha256=rewrite_sha,
                requires_native_clip_importer_correction=True,
                clip_names=[row['clip_name'] for row in rewrites],
                native_clip_importer_correction={'activation_alpha': -127.0, 'activation_beta': 127.0,
                    'required_count': 88, 'when': 'After ONNX parsing and before engine build in the same native network.',
                    'evidence_required': 'Exact named layers and all parsed/corrected Clip attributes; assert all 88 found.',
                    'old_trtexec_builder_implements_correction': False},
                activation_quantizer_topology='Q_INT8_CODE_CLIP_DQ',
                full_model_cpu_equivalence_status='NOT_RUN_BY_PREPARER',
                original_complete_model_numerical_guard='FAILED_UNCHANGED',
                complete_model_cpu_inference_run=False, native_build_or_execution_run_by_preparer=False,
                calibration_recomputed=False, weights_changed=False, original_tolerances_changed=False,
                dataset_evaluation_run=False,
                nonquantized_float_control_status='NOT_ESTABLISHED_FOR_NEW_CODE_CLIP_PREPARATION',
                parent_common_source_provenance_fields=['common_source_fp32_onnx_sha256', 'common_source_weights_sha256',
                    'common_q_all_pe_tap_reference_sha256', 'common_output_declaration_sha256'],
                common_source_provenance_scope='Retained parent source provenance only; common_preparation identifies the new semantic rewrite audit, not the old shared preparation.',
                prior_native_auditor_interface={'observed_script_sha256': parent_auditor_observed_sha,
                    'original_trace_topology': 'FLOAT_CLIP_Q_DQ',
                    'original_checker_clip_constants_dtype': 'float32',
                    'requires_code_clip_trace_and_int8_bound_support': True,
                    'audit_run_by_preparer': False},
                input_file='preprocessed_input.npy', input_file_sha256=INPUT_SHA256,
                native_diagnostic_reference_contract_sha256=REFERENCE_CONTRACT_SHA256,
                max_build_seconds=7200,
                build_budget_provenance={
                    'parent_spec_max_build_seconds': parent_spec['max_build_seconds'],
                    'candidate_max_build_seconds': 7200, 'maximum_new_build_attempts': 1,
                    'prior_q_all_extended_monitor': EXTENDED_MONITOR.relative_to(ROOT).as_posix(),
                    'prior_q_all_extended_monitor_sha256': sha(EXTENDED_MONITOR),
                    'prior_q_all_extended_max_seconds': extended_monitor['max_seconds'],
                    'prior_q_all_build_result': EXTENDED_BUILD.relative_to(ROOT).as_posix(),
                    'prior_q_all_build_result_sha256': sha(EXTENDED_BUILD),
                    'prior_q_all_build_elapsed_seconds': extended_build['elapsed_seconds'],
                    'prior_runtime_precision_constraints_flag': 'NOT_SET_BY_OLD_Q_ALL_TRTEXEC_BUILDER',
                    'candidate_precision_constraints': 'obey',
                    'candidate_explicit_fp16_layer_requests': 0,
                    'scope': 'Budget extension only. Keep parent obey field; 88 explicit Q/DQ quantizers control INT8 adoption, with no restored Half layer requests.'},
                generic_smoke_image_scope='Use unchanged original 000020.jpg source image; this copied NPY is optional CPU comparison input only.',
                limits='Candidate prepared statically only. Native builder must correct all 88 imported Clip attributes; original trtexec builder does not implement this. Original numerical guard remains FAILED_UNCHANGED. Older shared-preparation FP16 controls are parent provenance only, not matched controls for this rewrite.')
    require(spec['requests'] == parent_spec['requests'] and spec['expected_binding_contract'] == parent_spec['expected_binding_contract'],
            'Build requests or external binding contract changed')
    for filename, field in [('module_map.json', 'module_map_sha256'), ('preflight.json', 'failed_preflight_sha256'),
                            ('diagnostic_scope.json', 'diagnostic_scope_sha256')]:
        require(sha(output / filename) == spec[field], 'Retained spec reference differs: ' + filename)
    write(output / 'spec.json', spec)
    write(output / 'field_provenance.json', {
        'schema': 'rampq.gate05.corrected_qall_field_provenance/v1',
        'parent_spec_file': 'parent_spec.json', 'parent_graph_audit_file': 'parent_graph_audit.json',
        'spec_file': 'spec.json', 'spec_sha256': sha(output / 'spec.json'),
        'graph_audit_file': 'graph_audit.json', 'graph_audit_sha256': sha(output / 'graph_audit.json'),
        'spec_top_level_fields': field_changes(parent_spec, spec),
        'graph_audit_top_level_fields': field_changes(parent_audit, graph_audit),
        'retained_files_exact_parent_bytes': ['module_map.json', 'preflight.json', 'diagnostic_scope.json'],
        'new_reference_contract_exact_source_bytes': {
            'source': REFERENCE_CONTRACT.relative_to(ROOT).as_posix(),
            'candidate_file': 'reference_contract.json', 'sha256': REFERENCE_CONTRACT_SHA256,
            'scope': 'New diagnostic contract only; not a replacement original numerical guard or dataset evaluation protocol.'},
        'parent_reference_protocol_scope': 'parent_reference_protocol.json records the old Float Clip protocol; it is not a new candidate protocol.',
        'reference_update_policy': 'No cyclic self-hashes: spec references graph and audits; manifest references spec and every payload file; SHA256SUMS additionally references manifest.'})
    after = source_hashes(source_paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'A frozen parent source changed')
    (output / 'README.md').write_text("# Complete q_all code_clip candidate: only static preparation\n\nmodel.onnx comes from the exact frozen q_all graph; the 88 original activation Float Clip→Q sequences\nbecome Q→INT8 Clip[-127,127], leaving the subsequent original DQ unchanged. The 176 Clip/Q nodes and176 bound\ninitializers are the only permitted changes. New tensor names are checked individually for uniqueness; quantization scale/zero,\nWeights and their Q/DQ, bias, 88 Conv layers, remaining nodes, and all model external outputs are kept with raw bytes unchanged.\noriginal Float bound is verified to equal float32(scale * ±127), and all zeros are INT8 scalar 0.\n22 float32(bound/scale) values deviate from integers by one ULP because of division rounding; their nearest-even\ncodes remain exactly ±127. preparation_attempt1.json retains the first preparation attempt, stopped by the additional exact-division assertion.\nsemantic_rewrite_audit.json saves per-item rewrite, constant changes, and full source/new node/constant SHA values.\n\nspec retains precision_candidate_build_spec/v1, q_all, 88 INT8 / 0 FP16\noriginal requests and FLOAT three-binding contract, with obey / 1024MiB / avg1 / 7200s.\nmodule_map, failed preflight, and diagnostic_scope are copied byte for byte. The source spec/audit/manifest/\nreference_protocol is saved with the parent_ prefix; the latter still describes the old Float Clip and is not suitable as the new protocol.\nfield_provenance.json individually records the retention, addition, update, and original/new value SHA of top-level fields in spec/audit\nreference_contract.json is a byte-for-byte copy of this round's frozen diagnostic contract, with its SHA pinned separately by spec;\nit does not replace the original numerical guardrail or dataset gate protocol and makes no new Gate PASS claim.\n\ncommon_preparation_sha256 is changed to the current rewrite-audit hash; the old preparation identity\nis retained only in parent fields. The original matched FP16 control has not been established for this modified graph; the old control cannot support claiming\nthat the comparison conditions are complete. The old graph_audit assertion that all QDQ/constants are unchanged is set to false, with the scope of changes explained.\n\nAfter parsing, native building must use the same network before build to explicitly check the alpha/beta of the 88 named\nClip activations in clip_names and correct them to -127/+127, retaining evidence before and after import correction.\nThe old trtexec builder does not implement this correction; this package does not replicate a directly runnable new builder.\nThe original precision auditor currently supports only FloatClip→Q→DQ; future work requires clear support for code Clip topology\nand INT8 boundary re-audit; this preparer has not run native precision audit.\n\nThis preparer only involves CPU static byte difference and ONNX checker, without importing ORT/PyTorch, without running the full model\ndoes not connect to a device, use a GPU, change calibration/thresholds/weights, or evaluate a dataset. Complete CPU numerical equivalence has not yet been run;\nall native builds, execution, and fused-layer precision still require measurement; the original full guardrail remains FAILED_UNCHANGED.\npreprocessed_input.npy is a original copy of the original 000020 preprocessed input raw bytes; generic smoke uses the original image.\n", encoding='utf-8')
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema': 'rampq.gate05.precision_runtime_bundle/v1',
                                    'configuration': 'q_all', 'bundle_revision': spec['bundle_revision'], 'files': files})
    checksum_files = dict(files)
    checksum_files['manifest.json'] = info(output / 'manifest.json')
    (output / 'SHA256SUMS').write_text(''.join(value['sha256'] + '  ' + name + '\n'
                                            for name, value in sorted(checksum_files.items())), encoding='utf-8')
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            require(path.is_file(), 'Unexpected output directory entry')
            archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    print(json.dumps({'status': semantic_audit['status'], 'output': str(output),
                      'archive': str(output.with_suffix('.zip')), 'archive_bytes': output.with_suffix('.zip').stat().st_size,
                      'archive_sha256': sha(output.with_suffix('.zip')), 'spec_sha256': sha(output / 'spec.json'),
                      'graph_sha256': sha(output / 'model.onnx'), 'semantic_rewrite_audit_sha256': rewrite_sha,
                      'manifest_sha256': sha(output / 'manifest.json'), 'clip_names': spec['clip_names'],
                      'source_node_count': len(source.graph.node), 'source_initializer_count': len(source.graph.initializer),
                      'all_source_hashes_unchanged': True, 'complete_model_cpu_inference_run': False,
                      'native_build_or_execution_run': False}, indent=2))


if __name__ == '__main__':
    main()

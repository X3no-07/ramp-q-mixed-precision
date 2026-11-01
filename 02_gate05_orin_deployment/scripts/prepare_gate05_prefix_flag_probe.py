"""Freeze a static one-Conv topology x builder-flag diagnostic bundle.

This copies already frozen ONNX/input/CPU bytes. It never imports ONNX Runtime
or Torch, runs a detector, connects to a board, or rebuilds old evidence.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = 'gate05_prefix_flag_probe_20261020'
PARENT = ROOT / 'artifacts/gate05/20261019_fused_prefix_probe'
ORIGINAL_SHA = '22b1065c01792492d658596c1069aee074e054f32648fa90bf87cc39a6d11a33'
CODE_CLIP_SHA = 'd2b8ca8105f8a05fabb7aac66c6fbab1a2558ac35bdd5ea1736ce8e0052465de'
INPUT_SHA = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
CPU_SHA = '720040da21943f67ed376ab433524af56249cce7e2e2f73e8dc3d7000279a59f'
PARENT_SPEC_SHA = '8eda3d9fc2b4b9e997742efe7b7e9c5599619940af0aef388bbaccc5f3eaf496'
PARENT_MANIFEST_SHA = 'f85bb5a98bfb75f5b428d6ea62927dd5e0eb878e4366dd138ddfd4cef946af05'
PROTOCOL_SHA = 'c7fdcada89c685ed52f8f65d8318854982d58f52f16cfce44f3129cdd17a307d'
PREFLIGHT_SHA = '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21'
QALL_CHECKPOINT_SHA = '66e149025bb0d777afe79a013797704c7878fdcad30140ac256aac305db48471'
CASE_ORDER = ['original_none', 'original_obey', 'code_clip_none', 'code_clip_obey']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): info(path)
            for path in sorted(set(paths))}


def check_parent_files():
    parent = PARENT.resolve()
    manifest_path = parent / 'manifest.json'
    manifest = read(manifest_path)
    require(manifest['schema'] == 'rampq.gate05.fused_prefix_probe_bundle/v1',
            'Unexpected parent manifest schema')
    paths = [manifest_path]
    for name, expected in manifest['files'].items():
        path = parent / name
        require(path.resolve().parent == parent and not path.is_symlink() and
                path.is_file() and info(path) == expected,
                'Frozen parent source differs: ' + str(path))
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/gate05/20261020_prefix_flag_probe')
    parser.add_argument('--protocol', type=Path, required=True,
                        help='Existing root-frozen stage protocol; copied byte for byte.')
    parser.add_argument('--native-runner', type=Path,
                        default=ROOT / 'scripts/jetson_prefix_flag_probe.py')
    parser.add_argument('--native-builder', type=Path,
                        default=ROOT / 'scripts/jetson_build_prefix_flag_probe.py')
    args = parser.parse_args()
    output, protocol_path = args.output.resolve(), args.protocol.resolve()
    archive_path = output.with_suffix('.zip')
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()),
            'Output must be a fresh Gate 0.5 artifact directory')
    require(not output.exists() and not archive_path.exists(),
            'Choose fresh output and ZIP paths; preserve existing evidence')
    require(protocol_path.is_relative_to(ROOT) and protocol_path.is_file() and
            not protocol_path.is_symlink(), 'Protocol must be an existing workspace file')
    protocol = read(protocol_path)
    require(isinstance(protocol, dict) and protocol and sha(protocol_path) == PROTOCOL_SHA,
            'Protocol must be the byte-exact root-frozen protocol')
    require(protocol['schema'] == 'rampq.gate05.prefix_flag_probe_protocol/v1' and
            protocol['case_order'] == CASE_ORDER and
            protocol['batch_max_seconds'] == 1500 and
            protocol['per_case_max_build_seconds'] == 300 and
            protocol['common_native_settings']['repeat_executions_if_success'] == 2,
            'Frozen protocol scope or limits differ')
    expected_cases = []
    for case_id in CASE_ORDER:
        variant = 'code_clip' if case_id.startswith('code_clip') else 'original'
        expected_cases.append({
            'id': case_id, 'variant': variant,
            'precision_constraint_flag': 'obey' if case_id.endswith('_obey') else 'none',
            'max_build_seconds': 300, 'iterations': 1,
            'apply_native_clip_attribute_correction': variant == 'code_clip'})
    require(protocol['cases'] == expected_cases, 'Frozen case definitions differ')
    native_paths = [args.native_runner.resolve(), args.native_builder.resolve()]
    require(len({path.name for path in native_paths}) == 2 and
            all(path.parent == (ROOT / 'scripts').resolve() and path.is_file() and
                not path.is_symlink() for path in native_paths),
            'Native scripts must exist separately in scripts/')

    source_paths = check_parent_files()
    require(sha(PARENT / 'spec.json') == PARENT_SPEC_SHA and
            sha(PARENT / 'manifest.json') == PARENT_MANIFEST_SHA,
            'Pinned parent spec or manifest differs')
    parent_spec = read(PARENT / 'spec.json')
    require(parent_spec['schema'] == 'rampq.gate05.fused_prefix_probe/v1' and
            parent_spec['conv_name'] == '/model.0/conv/Conv' and
            parent_spec['excluded_second_conv_name'] == '/model.1/conv/Conv',
            'Parent must be the known one-Conv prefix')
    require(list(parent_spec['variants']) == ['original', 'code_clip'] and
            all(variant['nodes'] == 11 and variant['initializers'] == 12
                for variant in parent_spec['variants'].values()),
            'Expected one Conv, 11 nodes, 12 initializers per variant')
    require(parent_spec['input_binding'] == {
        'name': 'images', 'input': True, 'shape': [1, 3, 640, 640], 'dtype': 'DataType.FLOAT'},
        'Frozen prefix input contract changed')
    require(len(parent_spec['output_bindings_in_order']) == 1 and
            parent_spec['output_bindings_in_order'][0]['name'] == 'rampq_model_1_conv_a_dq' and
            parent_spec['output_bindings_in_order'][0]['shape'] == [1, 16, 320, 320] and
            parent_spec['output_bindings_in_order'][0]['dtype'] == 'DataType.FLOAT',
            'Frozen sole prefix output contract changed')
    require(parent_spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED',
            'Original full-model guard must stay failed')
    required_parent_shas = {
        parent_spec['variants']['original']['model_file']: ORIGINAL_SHA,
        parent_spec['variants']['code_clip']['model_file']: CODE_CLIP_SHA,
        parent_spec['input_file']: INPUT_SHA,
        parent_spec['cpu_expected_outputs_file']: CPU_SHA,
        'original_failed_preflight.json': PREFLIGHT_SHA,
    }
    for name, expected in required_parent_shas.items():
        require(sha(PARENT / name) == expected, 'Pinned source hash differs: ' + name)
    qall_checkpoint_path = ROOT / 'artifacts/gate05/20261019_corrected_qall_native/checkpoint.json'
    require(sha(qall_checkpoint_path) == QALL_CHECKPOINT_SHA,
            'Retained full q_all failure checkpoint differs')
    source_paths.extend([protocol_path, qall_checkpoint_path, Path(__file__).resolve(), *native_paths])
    before = source_hashes(source_paths)

    import numpy as np
    x = np.load(PARENT / parent_spec['input_file'], allow_pickle=False)
    require(x.dtype == np.float32 and x.shape == (1, 3, 640, 640) and
            x.flags.c_contiguous and bool(np.isfinite(x).all()), 'Frozen input metadata invalid')
    with np.load(PARENT / parent_spec['cpu_expected_outputs_file'], allow_pickle=False) as arrays:
        require(set(arrays.files) == {'original_output', 'code_clip_output'},
                'Frozen CPU NPZ keys changed')
        original, code_clip = arrays['original_output'], arrays['code_clip_output']
        require(original.dtype == code_clip.dtype == np.float32 and
                original.shape == code_clip.shape == (1, 16, 320, 320) and
                bool(np.isfinite(original).all()) and bool(np.isfinite(code_clip).all()) and
                original.tobytes(order='C') == code_clip.tobytes(order='C'),
                'Frozen CPU prefix outputs are not finite, byte-identical FP32 outputs')
        output_array_sha = hashlib.sha256(original.tobytes(order='C')).hexdigest()

    output.mkdir(parents=True, exist_ok=False)
    for name in required_parent_shas:
        shutil.copy2(PARENT / name, output / name)
    shutil.copy2(PARENT / 'spec.json', output / 'parent_spec.json')
    shutil.copy2(PARENT / 'manifest.json', output / 'parent_manifest.json')
    shutil.copy2(qall_checkpoint_path, output / 'retained_qall_failure_checkpoint.json')
    shutil.copy2(protocol_path, output / 'protocol.json')
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    for path in native_paths:
        shutil.copy2(path, output / path.name)

    cases = copy.deepcopy(protocol['cases'])
    spec = copy.deepcopy(parent_spec)
    spec.update({
        'schema': 'rampq.gate05.prefix_flag_probe/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(),
        'diagnostic_only': True,
        'diagnostic_scope': 'ONE_CONV_TWO_ACTIVATION_BOUNDARIES_TOPOLOGY_X_CONSTRAINT_FLAG',
        'conv_count': 1,
        'graph_node_count_per_variant': 11,
        'graph_initializer_count_per_variant': 12,
        'parent_spec_file': 'parent_spec.json',
        'parent_spec_sha256': sha(output / 'parent_spec.json'),
        'parent_manifest_file': 'parent_manifest.json',
        'parent_manifest_sha256': sha(output / 'parent_manifest.json'),
        'protocol_file': 'protocol.json',
        'protocol_sha256': sha(output / 'protocol.json'),
        'cases': cases,
        'case_order': CASE_ORDER,
        'max_seconds': 1500,
        'max_build_seconds': 300,
        'maximum_attempts_per_case': 1,
        'maximum_native_builds': 4,
        'maximum_native_inferences': 8,
        'executions_per_successful_case': 2,
        'native_build_policy': {
            'enabled_common_flags': ['INT8', 'FP16'],
            'disabled_common_flags': ['TF32', 'PREFER_PRECISION_CONSTRAINTS'],
            'only_case_specific_builder_flag': 'OBEY_PRECISION_CONSTRAINTS',
            'workspace_mib': 1024,
            'avg_timing_iterations': 1,
            'profiling_verbosity': 'DETAILED',
            'external_binding_type': 'FLOAT',
            'external_binding_format': 'LINEAR',
            'layer_precision_setters_allowed': False,
            'layer_output_type_setters_allowed': False,
            'parser_precision_constraints_reset_allowed': False,
            'parser_output_type_constraints_reset_allowed': False,
            'record_parser_constraints_before_and_after_clip_correction': True,
            'clip_alpha_beta_correction_variants': ['code_clip'],
            'clip_correction_layer_names': copy.deepcopy(parent_spec['clip_names']),
            'clip_activation_alpha': -127.0,
            'clip_activation_beta': 127.0,
            'original_variant_parser_attributes_modified': False,
            'compilation_error_policy': 'RECORD_CASE_FAILURE_CONTINUE_NEXT_CASE_NO_RETRY',
            'timeout_or_resource_abort_policy': 'ABORT_BATCH_NO_RETRY',
            'runtime_error_policy': 'RECORD_CASE_FAILURE_CONTINUE_NEXT_CASE_NO_RETRY',
        },
        'native_runner_file': native_paths[0].name,
        'native_runner_sha256': sha(native_paths[0]),
        'native_builder_file': native_paths[1].name,
        'native_builder_sha256': sha(native_paths[1]),
        'source_second_activation_scale': 0.7481081485748291,
        'single_original_float_dq_output_no_extra_pre_or_post_taps': True,
        'cpu_expected_outputs_original_bytes_copied': True,
        'cpu_outputs_byte_exact_checked_from_existing_npz': True,
        'new_cpu_ort_or_torch_executions': 0,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
        'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
        'retained_qall_failure_checkpoint_file': 'retained_qall_failure_checkpoint.json',
        'retained_qall_failure_checkpoint_sha256': QALL_CHECKPOINT_SHA,
        'calibration_recomputed': False,
        'original_tolerances_changed': False,
        'weights_changed': False,
        'dataset_evaluation_run': False,
        'complete_detector_executions': 0,
        'native_build_or_execution_run_by_preparer': False,
        'native_fusion_check_intent': 'Observe the actual first Conv/SiLU/output-QDQ tactics independently in each case; neither SWISH nor identical tactic selection is assumed.',
        'limits': 'Static preparation of a bounded one-Conv, two-boundary, 2x2 diagnostic only. Prefix results do not establish the failed full q_all build cause, a repaired full model, strict integer arithmetic, detector accuracy, performance, or Gate 0.5 passage.',
    })
    # The inherited diagnostic hash refers to a file deliberately not recopied;
    # the full immutable parent specification retains that historical pointer.
    spec.pop('cpu_diagnostic_sha256', None)
    write(output / 'spec.json', spec)
    write(output / 'static_preparation_audit.json', {
        'schema': 'rampq.gate05.prefix_flag_probe_static_audit/v1',
        'status': 'STATIC_PREPARATION_CHECKS_PASSED',
        'parent_source_manifest_verified': True,
        'parent_model_input_and_cpu_npz_bytes_unchanged': True,
        'one_conv_two_activation_boundaries': True,
        'node_count_per_variant': 11,
        'initializer_count_per_variant': 12,
        'cpu_expected_output_shape': [1, 16, 320, 320],
        'cpu_expected_outputs_byte_exact': True,
        'cpu_expected_output_raw_array_sha256': output_array_sha,
        'new_ort_executions': 0,
        'new_torch_executions': 0,
        'native_executions': 0,
        'complete_detector_executions': 0,
        'source_hashes_before': 'source_hashes_before.json',
        'source_hashes_after': 'source_hashes_after.json',
        'original_full_guard': 'FAILED_UNCHANGED',
        'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
    })
    write(output / 'retained_failure_lineage.json', {
        'schema': 'rampq.gate05.prefix_flag_retained_failures/v1',
        'original_failed_preflight': {
            'file': 'original_failed_preflight.json', 'sha256': PREFLIGHT_SHA,
            'status': 'LOCAL_QDQ_PREFLIGHT_FAILED', 'tolerance_changed': False,
        },
        'corrected_full_qall_failed_checkpoint': {
            'source': str(qall_checkpoint_path.relative_to(ROOT)).replace('\\', '/'),
            'copied_file': 'retained_qall_failure_checkpoint.json',
            'sha256': QALL_CHECKPOINT_SHA,
            'status': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE',
            'actual_native_precision_audit': 'NOT_ELIGIBLE_NO_ENGINE',
            'native_detector_execution': 'NOT_ELIGIBLE_NO_ENGINE',
            'OBEY_causality': 'UNPROVED',
        },
    })
    write(output / 'source_hashes_before.json', before)
    after = source_hashes(source_paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'A frozen source changed while preparing')
    (output / 'README.md').write_text('# 2x2 prefix diagnosis for a single Conv with activation boundary at two locations\n\nCopy raw bytes of the frozen prefix and fixed 000020 input; one graph with 11 nodes and 12 initializers\nOnly `/model.0/conv/Conv` one Conv, without the second Conv. The unique output remains still\n`rampq_model_1_conv_a_dq FLOAT[1,16,320,320], no additional intermediate tap.\n\nfixed order: original_none, original_obey, code_clip_none, code_clip_obey\nTwo topologies and NONE/OBEY_PRECISION_CONSTRAINTS are the only two experimental factors;\ncode_clip only natively parses and adjusts the alpha/beta values of two named Clips to -127/+127.\nAll configurations remain INT8+FP16, TF32 disabled, 1024 MiB workspace, avgTiming=1\nDETAILED inspector and FLOAT/LINEAR external bindings. Do not set any layer precision or\noutput type; do not clear or reset parser constraints; ensure actual parser constraints before and after are fully recorded.\n\nEach case allows at most 300 seconds and one build attempt; after success, repeat inference twice with the fixed input; batch supervision\n1500 seconds. Proceed to the next case if a compilation error persists; terminate the entire batch upon timeout or resource cancellation, without retry.\nLocal preparation only reads existing CPU NPZ, verifies two expected output raw bytes are consistent, without executing ORT\nTorch, CUDA, SSH, the full detector, or the dataset. Freeze and verify protocol.json before execution.\n\nThe original full PyTorch/ORT guardrail remains FAILED_UNCHANGED; the earlier corrected full q_all build failure\nis also retained in full. Regardless of success or failure, this probe alone cannot establish OBEY causality in full q_all or serve as evidence for\ncomplete-model repair, accuracy/performance, strict integer arithmetic, or Gate 0.5 acceptance.\n', encoding='utf-8')
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    require(len(files) == len(list(output.iterdir())), 'Unexpected output subdirectory')
    write(output / 'manifest.json', {
        'schema': 'rampq.gate05.prefix_flag_probe_bundle/v1', 'files': files})
    checksums = dict(files)
    checksums['manifest.json'] = info(output / 'manifest.json')
    (output / 'SHA256SUMS').write_text(
        ''.join(value['sha256'] + '  ' + name + '\n' for name, value in sorted(checksums.items())),
        encoding='utf-8')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None and
                all(name.startswith(ARCHIVE_ROOT + '/') and '..' not in name.split('/')
                    for name in archive.namelist()), 'Archive path or CRC check failed')
        for path in sorted(output.iterdir()):
            require(hashlib.sha256(archive.read(ARCHIVE_ROOT + '/' + path.name)).hexdigest() == sha(path),
                    'Archive bytes differ: ' + path.name)
    print(json.dumps({
        'status': 'STATIC_PREPARATION_CHECKS_PASSED',
        'output': str(output), 'archive': str(archive_path),
        'archive_bytes': archive_path.stat().st_size, 'archive_sha256': sha(archive_path),
        'spec_sha256': sha(output / 'spec.json'),
        'protocol_sha256': sha(output / 'protocol.json'),
        'manifest_sha256': sha(output / 'manifest.json'),
        'case_order': CASE_ORDER, 'max_seconds': 1500, 'max_build_seconds': 300,
        'new_ort_or_torch_executions': 0, 'native_builds_or_executions': 0,
        'all_source_hashes_unchanged': True,
    }, indent=2))


if __name__ == '__main__':
    main()

"""Package an independent full corrected-q_all NONE control, using stdlib only.

The complete source_bundle and saved CPU-equivalence directory are copied byte
for byte. Their old OBEY metadata is never rewritten: a new outer protocol and
control_spec describe this separate NONE control. No model inference, native
build, SSH, calibration, tolerance change, or dataset evaluation is performed.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate'
CPU = ROOT / 'artifacts/gate05/20261019_corrected_qall_cpu_equivalence'
FAILED = ROOT / 'artifacts/gate05/20261019_corrected_qall_native'
REUSE_REVIEW = ROOT / 'artifacts/gate05/20261021_gate05_completion_plan/saved_full_qall_reference_reuse_review.json'
PATH_ANALYSIS = ROOT / 'artifacts/gate05/20261021_gate05_completion_plan/compiler_path_analysis.json'
PROTOCOL = ROOT / 'artifacts/gate05/20261021_qall_none_protocol/protocol.json'
OUTPUT = ROOT / 'artifacts/gate05/20261021_qall_none_control'
ARCHIVE_ROOT = 'gate05_qall_none_control_20261021'
PROTOCOL_SHA = '572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1'
GRAPH_SHA = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
SOURCE_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
SOURCE_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
INPUT_SHA = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
REWRITE_SHA = '842eff15adfedd75735815cbe4040d17a645653e3b62511149afacbf8f923be9'
CONTRACT_SHA = '87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430'
CPU_ANALYSIS_SHA = '2af9ab8e1e8b76cbafd1622d75eedd825fe6f42f2dcdcdcfe115a7110965ab87'
CPU_MANIFEST_SHA = '2d6a12602162c3fbc0c7e373d6f36bf58355c3cf5cadab98aebf3730d2a17791'
CPU_RAW_SHA = '1fe8d81720679ad47fb6f6d7d47b3019dc2cbd7fccdc72ab6683f8a5d2ce705c'
FAILED_CHECKPOINT_SHA = '66e149025bb0d777afe79a013797704c7878fdcad30140ac256aac305db48471'
REUSE_REVIEW_SHA = '1c9e21a845ae1e376611f18bb1ba1afe6daaf6e38e073fafd0597d8dadc480c2'
PATH_ANALYSIS_SHA = '78bd8f7372d92c623ff63e26646161e19aae7350bb53072a6b4149147396c306'
HELPER_PINS = {
    'jetson_build_precision_candidate.py': 'd8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f',
    'jetson_monitored_build.py': '71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
    'jetson_owned_group_guard.py': '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
    'jetson_collect_isolated_numeric_probe.py': 'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def regular_files(folder):
    files = list(folder.rglob('*'))
    require(all(not path.is_symlink() for path in files), 'Symlink in immutable source tree')
    return [path for path in files if path.is_file()]


def verify_bundle(folder, manifest_sha, has_sums=False):
    require(sha(folder / 'manifest.json') == manifest_sha, 'Frozen manifest changed: ' + str(folder))
    manifest = read(folder / 'manifest.json')
    require(isinstance(manifest['files'], dict) and manifest['files'], 'Empty frozen manifest')
    for name, expected in manifest['files'].items():
        path = (folder / name).resolve()
        require(path.parent == folder.resolve() and not path.is_symlink() and path.is_file() and
                info(path) == expected, 'Frozen manifest member differs: ' + name)
    permitted = set(manifest['files']) | {'manifest.json'} | ({'SHA256SUMS'} if has_sums else set())
    require({path.name for path in regular_files(folder)} == permitted and
            all(path.is_file() for path in folder.iterdir()), 'Unexpected source contents: ' + str(folder))
    if has_sums:
        entries = {}
        for line in (folder / 'SHA256SUMS').read_text(encoding='utf-8-sig').splitlines():
            digest, name = line.split('  ', 1)
            require(name not in entries and Path(name).name == name and len(digest) == 64, 'Malformed source SHA256SUMS')
            entries[name] = digest
        require(set(entries) == set(manifest['files']) | {'manifest.json'} and
                all(sha(folder / name) == digest for name, digest in entries.items()), 'Frozen source SHA256SUMS differs')
    return regular_files(folder)


def snapshot(paths):
    result = {}
    for path in sorted(set(paths)):
        resolved = path.resolve()
        require(resolved.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(), 'Unsafe protected source')
        result[resolved.relative_to(ROOT).as_posix()] = info(path)
    return result


def validate_saved_sources(record):
    paths = []
    for name, expected in record.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and info(path) == expected,
                'Saved CPU/preparation source no longer matches: ' + name)
        paths.append(path)
    return paths


def verify_saved_cpu():
    paths = verify_bundle(CPU, CPU_MANIFEST_SHA)
    require(sha(CPU / 'analysis.json') == CPU_ANALYSIS_SHA and sha(CPU / 'raw_outputs.npz') == CPU_RAW_SHA,
            'Saved full CPU-equivalence identities changed')
    analysis = read(CPU / 'analysis.json')
    require(analysis['schema'] == 'rampq.gate05.corrected_qall_cpu_equivalence/v1' and
            analysis['status'] == 'FULL_FIXED_INPUT_CPU_REWRITE_BYTE_EQUIVALENCE_VERIFIED' and
            analysis['candidate_onnx_sha256'] == GRAPH_SHA and analysis['candidate_spec_sha256'] == SOURCE_SPEC_SHA and
            analysis['candidate_manifest_sha256'] == SOURCE_MANIFEST_SHA and analysis['input_npy_sha256'] == INPUT_SHA and
            analysis['reference_contract_sha256'] == CONTRACT_SHA and analysis['raw_outputs_sha256'] == CPU_RAW_SHA and
            analysis['parent_onnx_sha256'] == '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d' and
            analysis['original_complete_model_pytorch_ort_guard'] == 'FAILED_UNCHANGED' and
            analysis['all_original_sources_hash_unchanged'] is True and analysis['new_tolerances_defined'] is False,
            'Saved full CPU-equivalence scope differs')
    expected_shapes = {'output0': [1, 14, 8400], '/model.10/m/m.0/attn/pe/conv/Conv_output_0': [1, 128, 20, 20]}
    require(set(analysis['source_and_candidate_outputs']) == set(expected_shapes), 'Wrong saved CPU output set')
    for name, shape in expected_shapes.items():
        row = analysis['source_and_candidate_outputs'][name]
        require(row['shape'] == shape and row['dtype'] == 'float32' and row['byte_identical'] is True and
                row['different_elements'] == 0 and row['max_abs'] == 0, 'Saved CPU equality record differs: ' + name)
    before = read(CPU / 'source_hashes_before.json')
    require(before == read(CPU / 'source_hashes_after.json'), 'Saved full CPU source hashes differ')
    paths.extend(validate_saved_sources(before))
    return paths


def validate_protocol(protocol_path):
    require(len(PROTOCOL_SHA) == 64 and sha(protocol_path) == PROTOCOL_SHA, 'Root-frozen protocol not pinned or changed')
    protocol = read(protocol_path)
    # The frozen file is the authority for remaining descriptive fields.
    require(protocol['schema'] == 'rampq.gate05.corrected_qall_none_control_protocol/v1', 'Wrong full NONE protocol')
    require(protocol['source_graph_sha256'] == GRAPH_SHA and protocol['source_spec_sha256'] == SOURCE_SPEC_SHA and
            protocol['source_manifest_sha256'] == SOURCE_MANIFEST_SHA and
            protocol['source_rewrite_audit_sha256'] == REWRITE_SHA and
            protocol['parent_failed_checkpoint_sha256'] == FAILED_CHECKPOINT_SHA and
            protocol['source_diagnostic_analysis_sha256'] == PATH_ANALYSIS_SHA and
            protocol['legacy_source_spec_precision_constraints'] == 'obey' and
            protocol['current_precision_constraints'] == 'none' and
            protocol['legacy_source_contract_not_rewritten'] is True and protocol['new_CPU_runs'] == 0,
            'Protocol full source identities or independent control scope differ')
    saved = protocol['source_CPU_equivalence']
    require(saved['directory'] == 'cpu_equivalence' and saved['analysis_sha256'] == CPU_ANALYSIS_SHA and
            saved['manifest_sha256'] == CPU_MANIFEST_SHA and saved['raw_outputs_sha256'] == CPU_RAW_SHA and
            saved['review_sha256'] == REUSE_REVIEW_SHA and saved['new_CPU_runs'] == 0 and
            protocol['native_build']['attempts'] == 1 and protocol['native_build']['max_seconds'] == 7200 and
            protocol['capture']['automatic_inference'] is False,
            'Protocol saved reference or build/inference budget differs')
    return protocol


def check_native_input(output, native_builder):
    command = [sys.executable, '-B', str(output / native_builder.name), '--root', str(output), '--check-input-only']
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding='utf-8', timeout=60)
    require(result.returncode == 0, 'Native stdlib input check failed: ' + result.stdout + result.stderr)
    parsed = json.loads(result.stdout)
    require(parsed['native_runtime_imported'] is False, 'Input check did not certify no runtime imports')
    return {'status': 'NATIVE_STDLIB_INPUT_CHECK_PASSED', 'command': command, 'returncode': result.returncode,
        'stdout': parsed, 'stderr': result.stderr, 'no_new_CPU_or_native_build_or_execution': True}


def package(output, protocol_path, native_builder, expected_builder_sha):
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Exclusive fresh destination required')
    protocol = validate_protocol(protocol_path)
    require(native_builder.name == 'jetson_build_corrected_qall_none.py' and native_builder.parent == ROOT / 'scripts' and
            native_builder.is_file() and not native_builder.is_symlink() and len(expected_builder_sha) == 64 and
            sha(native_builder) == expected_builder_sha, 'Final independent NONE builder identity required')
    source_paths = verify_bundle(SOURCE, SOURCE_MANIFEST_SHA, has_sums=True)
    source = read(SOURCE / 'spec.json')
    require(sha(SOURCE / 'spec.json') == SOURCE_SPEC_SHA and sha(SOURCE / 'model.onnx') == GRAPH_SHA and
            sha(SOURCE / 'semantic_rewrite_audit.json') == REWRITE_SHA and
            sha(SOURCE / 'preprocessed_input.npy') == INPUT_SHA and sha(SOURCE / 'reference_contract.json') == CONTRACT_SHA and
            source['configuration'] == 'q_all' and source['precisionConstraints'] == 'obey' and
            source['quantized_conv_count'] == 88 and source['fp16_conv_count'] == 0 and len(source['requests']) == 88 and
            source['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED', 'Immutable original corrected source differs')
    source_before = read(SOURCE / 'source_hashes_before.json')
    require(source_before == read(SOURCE / 'source_hashes_after.json'), 'Original preparation source hashes differ')
    source_paths.extend(validate_saved_sources(source_before))
    source_paths.extend(verify_saved_cpu())
    require(sha(FAILED / 'checkpoint.json') == FAILED_CHECKPOINT_SHA and sha(REUSE_REVIEW) == REUSE_REVIEW_SHA and
            sha(PATH_ANALYSIS) == PATH_ANALYSIS_SHA,
            'Saved failure checkpoint or independent reference-reuse review changed')
    for name, expected in HELPER_PINS.items():
        require(sha(ROOT / 'scripts' / name) == expected, 'Original helper changed: ' + name)
    extra = {FAILED / 'checkpoint.json': 'parent_failed_checkpoint.json',
        FAILED / 'failure_boundary_inventory.json': 'parent_failure_inventory.json',
        FAILED / 'stage_manifest.json': 'parent_failed_stage_manifest.json',
        REUSE_REVIEW: 'saved_full_qall_reference_reuse_review.json', PATH_ANALYSIS: 'compiler_path_analysis.json',
        protocol_path: 'protocol.json',
        native_builder: native_builder.name, Path(__file__).resolve(): Path(__file__).name,
        **{ROOT / 'scripts' / name: name for name in HELPER_PINS}}
    source_paths.extend(extra)
    before = snapshot(source_paths)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SOURCE, output / 'source_bundle')
    shutil.copytree(CPU, output / 'cpu_equivalence')
    for path, name in extra.items():
        shutil.copy2(path, output / name)
    write(output / 'source_hashes_before.json', before)
    spec = {'schema': 'rampq.gate05.corrected_qall_none_control/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
        'protocol_file': 'protocol.json', 'protocol_sha256': PROTOCOL_SHA,
        'source_bundle_directory': 'source_bundle', 'cpu_equivalence_directory': 'cpu_equivalence',
        'runtime_precision_constraints': 'none', 'legacy_source_precision_constraints': 'obey',
        'legacy_source_metadata_rewritten': False, 'source_bundle_file_set_and_all_bytes_unchanged': True,
        'maximum_native_builds': 1, 'maximum_attempts_per_case': 1, 'max_build_seconds': 7200,
        'new_CPU_execution_runs_this_stage': 0, 'native_inference_executions': 0,
        'graph_sha256': GRAPH_SHA, 'source_spec_sha256': SOURCE_SPEC_SHA, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
        'input_file': 'source_bundle/preprocessed_input.npy', 'input_file_sha256': INPUT_SHA,
        'semantic_rewrite_audit_sha256': REWRITE_SHA, 'reference_contract_sha256': CONTRACT_SHA,
        'cpu_equivalence_analysis_sha256': CPU_ANALYSIS_SHA, 'cpu_equivalence_manifest_sha256': CPU_MANIFEST_SHA,
        'cpu_equivalence_raw_outputs_sha256': CPU_RAW_SHA,
        'parent_failed_checkpoint_file': 'parent_failed_checkpoint.json', 'parent_failed_checkpoint_sha256': FAILED_CHECKPOINT_SHA,
        'saved_reference_reuse_review_file': 'saved_full_qall_reference_reuse_review.json',
        'saved_reference_reuse_review_sha256': REUSE_REVIEW_SHA,
        'source_diagnostic_analysis_file': 'compiler_path_analysis.json',
        'source_diagnostic_analysis_sha256': PATH_ANALYSIS_SHA,
        'native_builder_file': native_builder.name, 'native_builder_sha256': expected_builder_sha,
        'unchanged_helpers': dict(HELPER_PINS), 'expected_binding_contract': source['expected_binding_contract'],
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
        'new_tolerances_defined': False, 'original_tolerances_changed': False, 'weights_changed': False,
        'calibration_recomputed': False, 'dataset_evaluation_run': False,
        'complete_model_CPU_inference_run_by_packager': False, 'native_build_or_execution_run_by_packager': False,
        'limits': 'Separate runtime NONE control; prior source OBEY metadata and failed guard remain unchanged. Saved CPU rewrite equality is not PyTorch/ORT migration acceptance or native accuracy/precision proof.'}
    write(output / 'control_spec.json', spec)
    (output / 'README.md').write_text('''# 完整 corrected q_all 独立 NONE 构建控制

source_bundle/完整复制旧候选目录与所有原始字节，原spec/reference合同中的
OBEY请求不改。外层control_spec/protocol定义这次独立runtime NONE控制，
不把旧文件伪装成NONE或修复其历史失败。原helper/监护/guard/collector保持字节。

cpu_equivalence/仅复制现有原/修正完整ONNX CPU等价证据及rawNPZ；
封包不运行CPU、校准、构建或推理。原PyTorch/ORT guard仍FAILED_UNCHANGED。
本次最多一次7200秒native构建，封包阶段只运行stdlib --check-input-only。
native推理、数值迁移、准确率、性能和Gate通过不在本封包中声明。
''', encoding='utf-8')
    after = snapshot(source_paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'Historical sources or tools changed during package preparation')
    for original, copied in ((SOURCE, output / 'source_bundle'), (CPU, output / 'cpu_equivalence')):
        originals = {path.relative_to(original).as_posix(): info(path) for path in regular_files(original)}
        copies = {path.relative_to(copied).as_posix(): info(path) for path in regular_files(copied)}
        require(originals == copies, 'Original complete directory bytes differ after copy')
    # A first manifest supports the native tool's pure-stdlib validation. The
    # recorded check is then included in the final independent outer manifest.
    manifest_path = output / 'manifest.json'
    write(manifest_path, {'schema': 'rampq.gate05.corrected_qall_none_control_bundle/v1',
        'files': {path.relative_to(output).as_posix(): info(path) for path in sorted(regular_files(output))}})
    check = check_native_input(output, native_builder)
    write(output / 'native_input_check.json', check)
    final_manifest = {'schema': 'rampq.gate05.corrected_qall_none_control_bundle/v1',
        'files': {path.relative_to(output).as_posix(): info(path) for path in sorted(regular_files(output)) if path != manifest_path}}
    manifest_path.write_text(json.dumps(final_manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    # Revalidate the final manifest. No build/inference imports occur in either check.
    check_native_input(output, native_builder)
    require(snapshot(source_paths) == before, 'Protected source changed during pure-stdlib checks')
    sums = dict(final_manifest['files'])
    sums['manifest.json'] = info(manifest_path)
    (output / 'SHA256SUMS').write_text(''.join(row['sha256'] + '  ' + name + '\n'
        for name, row in sorted(sums.items())), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(regular_files(output)):
            archive.write(path, ARCHIVE_ROOT + '/' + path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP CRC check failed')
        names = {ARCHIVE_ROOT + '/' + path.relative_to(output).as_posix(): path for path in regular_files(output)}
        require(len(archive.namelist()) == len(names) and set(archive.namelist()) == set(names), 'ZIP member set differs')
        for name, path in names.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == sha(path), 'ZIP member bytes differ: ' + name)
    print(json.dumps({'status': 'CORRECTED_QALL_NONE_CONTROL_PACKAGED', 'output': str(output),
        'archive': str(archive_path), 'archive_bytes': archive_path.stat().st_size, 'archive_sha256': sha(archive_path),
        'control_spec_sha256': sha(output / 'control_spec.json'), 'manifest_sha256': sha(manifest_path),
        'protocol_sha256': PROTOCOL_SHA, 'native_builder_sha256': expected_builder_sha,
        'all_source_bytes_unchanged': True, 'new_CPU_executions': 0, 'native_build_or_inference_executions': 0}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--native-builder', type=Path, default=ROOT / 'scripts/jetson_build_corrected_qall_none.py')
    parser.add_argument('--native-builder-sha256', required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()), 'Unsafe package destination')
    existed_before = output.exists()
    try:
        package(output, args.protocol.resolve(), args.native_builder.resolve(), args.native_builder_sha256)
    except BaseException as error:
        if not existed_before and output.is_dir() and not (output / 'packaging_failure.json').exists():
            write(output / 'packaging_failure.json', {'status': 'FAILED_PRESERVE_NO_BUILD_OR_RETRY',
                'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                'new_CPU_executions': 0, 'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'})
        raise


if __name__ == '__main__':
    main()

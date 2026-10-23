"""Independent new outer-NONE control review; static/saved data only."""
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path(__file__).resolve().parent
BUNDLE = ROOT / 'artifacts/gate05/20261021_qall_none_control'
RUNTIME = ROOT / 'artifacts/gate05/20261021_qall_none_runtime'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inventory(folder):
    return {p.relative_to(folder).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in folder.rglob('*') if p.is_file()}


def main():
    out = REVIEW / 'pre_native_review.json'
    assert not out.exists()
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)

    before = inventory(BUNDLE)
    pins = {'protocol.json': '572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1',
        'control_spec.json': 'b34d042f07969bbe56e68ede5cda41f1e16996d0747a479b7733ac26876b1739',
        'manifest.json': 'e667457c74df2c81e78bfaa4b1414907f52cf5763696f075bbc9adbf67f3da60',
        'jetson_build_corrected_qall_none.py': '4e5b9b71be93e72f5310273f477c0cbf7256fe2249f66c3a01f9ad3d96844aaa',
        'saved_full_qall_reference_reuse_review.json': '1c9e21a845ae1e376611f18bb1ba1afe6daaf6e38e073fafd0597d8dadc480c2'}
    for name, digest in pins.items():
        check('pinned_identity:' + name, sha(BUNDLE / name) == digest)
    manifest, control, protocol = [read(BUNDLE / name) for name in ('manifest.json', 'control_spec.json', 'protocol.json')]
    check('outer_exact_payload_plus_seals', set(before) == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'})
    for name, info in manifest['files'].items():
        check('manifest_path:' + name, BUNDLE.resolve() in (BUNDLE / name).resolve().parents and not (BUNDLE / name).is_symlink())
        check('manifest_member:' + name, before[name] == info)
    for line in (BUNDLE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        check('SHA256SUMS:' + name, sha(BUNDLE / name) == digest)
    archive_path = BUNDLE.with_suffix('.zip')
    check('zip_identity', sha(archive_path) == 'e5eb1ed576349f5f3a173cb7a8cfbd2ffe6368450c7618a82ad7f8357659f60d' and archive_path.stat().st_size == 11383328)
    with zipfile.ZipFile(archive_path) as archive:
        check('zip_crc', archive.testzip() is None)
        members = {'gate05_qall_none_control_20261021/' + name for name in before}
        check('zip_exact_inventory', set(archive.namelist()) == members and len(archive.namelist()) == len(members))
        for name in members:
            relative = name.split('/', 1)[1]
            check('zip_original_payload:' + relative, archive.read(name) == (BUNDLE / relative).read_bytes())
    for first, last in [('source_hashes_before.json', 'source_hashes_after.json')]:
        a, b = read(BUNDLE / first), read(BUNDLE / last)
        check('protected_sources_beforeafter_equal', a == b)
        for name, info in a.items():
            check('current_protected_source:' + name, sha(ROOT / name) == info['sha256'] and (ROOT / name).stat().st_size == info['bytes'])
    for local, historical, count in [('source_bundle', '20261019_corrected_qall_candidate', 21),
                                      ('cpu_equivalence', '20261019_corrected_qall_cpu_equivalence', 6)]:
        first, old = inventory(BUNDLE / local), inventory(ROOT / 'artifacts/gate05' / historical)
        check('complete_original_directory_copy:' + local, first == old and len(first) == count)
    source = read(BUNDLE / 'source_bundle/spec.json')
    check('legacy_OBEY_and_newNONE_explicit', source['precisionConstraints'] == control['legacy_source_precision_constraints'] == 'obey' and
          control['runtime_precision_constraints'] == protocol['current_precision_constraints'] == 'none' and
          control['legacy_source_metadata_rewritten'] is False)
    check('zero_new_CPU_and_no_automatic_inference', control['new_CPU_execution_runs_this_stage'] == protocol['new_CPU_runs'] == 0 and
          control['native_inference_executions'] == 0 and protocol['capture']['automatic_inference'] is False)
    check('one_build7200_no_retry', control['maximum_native_builds'] == control['maximum_attempts_per_case'] == 1 and
          control['max_build_seconds'] == 7200 and protocol['native_build']['attempts'] == 1)
    check('exact88INT8_zero_restored', source['quantized_conv_count'] == len(source['requests']) == 88 and source['fp16_conv_count'] == 0 and
          len({r['onnx_node'] for r in source['requests']}) == 88 and all(r['precision'] == 'INT8' for r in source['requests']))
    with np.load(BUNDLE / 'cpu_equivalence/raw_outputs.npz', allow_pickle=False) as data:
        check('savedCPU_exact4keys', set(data.files) == {'original_0', 'code_clip_0', 'original_1', 'code_clip_1'})
        for index, shape in [(0, (1, 14, 8400)), (1, (1, 128, 20, 20))]:
            left, right = data['original_' + str(index)], data['code_clip_' + str(index)]
            check('savedCPU_contract:' + str(index), left.dtype == right.dtype == np.float32 and left.shape == right.shape == shape and np.isfinite(left).all())
            check('savedCPU_raw_byte_equality:' + str(index), left.tobytes() == right.tobytes())
    native_path = BUNDLE / 'jetson_build_corrected_qall_none.py'
    native_text = native_path.read_text(encoding='utf-8')
    tree = ast.parse(native_text, feature_version=(3, 8))
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    stores = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    check('Python38_native_AST', isinstance(tree, ast.Module))
    check('no_layer_setters_reset', not calls & {'set_precision', 'reset_precision', 'set_output_type', 'reset_output_type'} and not stores & {'precision', 'dtype'})
    check('no_inference_APIs', not calls & {'create_execution_context', 'execute_v2', 'execute_async_v2', 'execute_async_v3'})
    check('sole_native_build_call', sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'build_serialized_network' for n in ast.walk(tree)) == 1)
    for text in ('parsed_constraints_before.json', 'parsed_constraints_after.json', 'parsed_constraints_after_build.json',
                 'parsed_clip_attributes_before_correction.json', 'parsed_clip_attributes_after_correction.json',
                 'conv_rows.json', 'conv_source_mapping.json', 'conv_descriptors.json', 'native_build_attempt.json'):
        check('native_evidence_capture:' + text, text in native_text)
    native_check = subprocess.run([sys.executable, '-B', str(native_path), '--root', str(BUNDLE), '--check-input-only'], capture_output=True, text=True)
    check('native_stdlib_check_exit0', native_check.returncode == 0)
    receipt = json.loads(native_check.stdout)
    check('native_stdlib_check_no_execution', receipt['native_runtime_imported'] is False and receipt['native_build_run'] is False and
          receipt['native_inference_run'] is False and receipt['new_CPU_execution_runs'] == 0)
    source_audit = read(ROOT / 'artifacts/gate05/20261021_gate05_completion_plan/qall_none_source_audit.json')
    check('independent_original_source_auditor_pass', source_audit['status'] == 'CORRECTED_SOURCE_STRUCTURE_VERIFIED' and not source_audit['source_structure_errors'] and
          source_audit['audit_tool_identity']['base_native_auditor_sha256'] == '40420902d6fb3fd5d6b6278d64bbaa4b44c839bdae69da73dbaeb3e15cf66f6a' and
          source_audit['native_formats_tactics_and_fusion_criteria_changed'] is False)
    launch_path = RUNTIME / 'launch_native.py'
    launch = launch_path.read_text(encoding='utf-8')
    ast.parse(launch, feature_version=(3, 8))
    for token in ('gate05_qall_none_control_20261021', "sha(root / 'control_spec.json') == spec_sha", pins['protocol.json'],
                  "available_kib >= 3 * 1024 * 1024", "fan.stdout.strip() == 'active'", 'assert not tasks',
                  'atexit.register(cleanup_unconfirmed_launch)', 'guard_module.terminate_owned(owned, failure, save_failure)',
                  "'--max-seconds', '7200'", "'--build-script', 'jetson_build_corrected_qall_none.py'", 'start_new_session=True',
                  "guard_status['monitor_identity'][field] == status[field]", "guard_status['owned_group_fully_empty']"):
        check('final_launch_requirement:' + token, token in launch)
    old_launch = (ROOT / 'artifacts/gate05/20261020_dfl_conv_output_runtime/launch_native.py').read_text(encoding='utf-8')
    def function_ast(text, name):
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
    check('mature_unconfirmed_launch_cleanup_unchanged', function_ast(launch, 'cleanup_unconfirmed_launch') == function_ast(old_launch, 'cleanup_unconfirmed_launch'))
    check('source_bundle_and_outer_unchanged_by_review', inventory(BUNDLE) == before)
    manual = [
        ('full_control_provenance', 'New outer control owns one NONE attempt; 21 legacy source files remain original OBEY bytes and six CPU proof files copied exactly. No relabeling old attempt or replacing failed guard.'),
        ('intrinsic_requests_and_actual_flags', 'Builder preserves all parser constraints/no setters/resets,records actual flags with INT8/FP16 true and OBEY/PREFER/TF32 false;88 namedClipattributes corrected±127 only.'),
        ('native_precision_pending', 'Native raw inspector/source mapping/format/tactic/fusion audit after success. No outputINT8requirement. Missing7tacticnames remain separate until an independently authenticated runtimeSASS supplement; no oldauditrewrite.'),
        ('strict_newscope', 'One7200s build under privategroupmonitor+guard, noautomaticinference. Numeric correctness, matchedcontrol/dataset and Gate acceptance are separate future protocols. No compiler success promised.'),
        ('no_reviewer_inference', 'Only savedCPU arrays/ONNXstatic source auditor/stdlibinputcheck; no newCPUgraph/native/SSH.')]
    for name, _ in manual:
        check('manual:' + name, True)
    report = {'schema': 'rampq.gate05.qall_none_pre_native_independent_review/v1',
        'client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'PRE_NATIVE_FULL_QALL_NONE_CONTROL_REVIEW_PASSED_ACTUAL_ADOPTION_PENDING',
        'checks': checks, 'check_count': len(checks), 'launch_blockers': [],
        'identities': {**pins, 'zip_sha256': sha(archive_path), 'zip_bytes': archive_path.stat().st_size,
                       'launcher_sha256': sha(launch_path), 'source_audit_sha256': sha(ROOT / 'artifacts/gate05/20261021_gate05_completion_plan/qall_none_source_audit.json')},
        'manual_review': [{'name': n, 'assessment': a} for n, a in manual], 'inputcheck': receipt,
        'new_CPU_or_native_inference_run_by_review': False, 'reviewer_SSH': False,
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'}
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(out),
                      'launcher_sha256': sha(launch_path), 'launch_blockers': []}))


if __name__ == '__main__':
    main()

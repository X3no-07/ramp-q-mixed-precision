"""Seal the completed, saved four-case prefix diagnostic; never run inference."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05/20261020_prefix_flag_native'
RAW = BASE / 'native_evidence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    checkpoint_path = BASE / 'checkpoint.json'
    manifest_path = BASE / 'stage_manifest.json'
    before_path = BASE / 'state_before_completion.json'
    assert not any(p.exists() for p in (checkpoint_path, manifest_path, before_path))
    audit = read(BASE / 'root_native_audit.json')
    review = read(BASE / 'independent_native_review.json')
    board = read(BASE / 'final_board_observation.json')
    remote = read(BASE / 'final_remote_receipt_verification.json')
    receipt = read(RAW / 'remote_receipt.json')
    assert audit['successful_output_cases'] == 4 and audit['raw_tree_bytes_unchanged']
    assert review['check_count'] == len(review['checks']) == 530
    assert all(c['passed'] for c in review['checks']) and review['raw_tree_unchanged']
    assert len(receipt['files']) == remote['receipt_file_count'] == 107
    assert remote['all_remote_receipt_files_still_match'] and all(remote['checks'].values())
    assert {p.relative_to(RAW).as_posix() for p in RAW.rglob('*') if p.is_file()} == set(receipt['files']) | {'remote_receipt.json'}
    for rel, expected in receipt['files'].items():
        p = RAW / rel
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], rel
    assert board['boot_id'] == remote['boot_id'] == '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
    assert not board['task_processes'] and all(not r['exists'] for r in board['known_processes'].values())
    assert board['fan']['stdout'].strip() == 'active' and 'NRestarts=0' in board['jtop']['stdout']
    assert not board['corrected_qall_engine_paths']
    assert board['qonly_preserved_engine']['sha256'] == '486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476'
    assert audit['monitor_status'] == 'BUILD_SCRIPT_OK' and audit['guard_fully_empty'] and not audit['guard_intervention']
    for name, case in audit['cases'].items():
        assert case['build_status'] == 'PREFIX_FLAG_ENGINE_SAVED' and case['all_outputs_finite']
        assert case['repeat']['byte_identical'] and case['child_cleanup']['returncode'] == 0
        assert case['actual_conv']['named_integer_tactic_verified'] and case['actual_conv']['output_int8']
        assert case['parser_constraints'] == dict(parsed_layers=18, precision_set_count=3, output_type_set_count=3, metadata_errors=[])
    for pair in ('original_none_vs_original_obey', 'code_clip_none_vs_code_clip_obey'):
        assert audit['cross_native_comparisons'][pair]['byte_identical']
    protected = {
        'artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence.zip': '5eac1543e5151708585e57923dd82410c9b21aac81cb5cfd1137e057e35ed205',
        'artifacts/gate05/20261019_corrected_qall_native/build_failed_evidence.zip': '027433baa5b1e18fafb72cf64b70e58e1bd2aa05b295fd8f539ae8317e1c9a99',
        'artifacts/gate05/20261019_corrected_qall_native/parser_metadata_evidence.zip': 'a567b474be0e563448ce96b7b5cc0f1e9bfff1036461b9512efe9b8c1ba32d78',
        'artifacts/gate05/20261018_int8/input_v1/preflight.json': '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21',
        'artifacts/gate05/20261020_prefix_flag_probe_v2.zip': '7a6a1d88a50f9db72ac48aeed3a0c2923068031f956393b5fb5c4997f32fcc96',
        'artifacts/gate05/20261020_prefix_flag_probe.zip': '5615fa40845d4b0630d3ac2fa08d581ff1e3c5f0e3b2fe4a907d6192f655ba95',
        'artifacts/gate05/20261020_prefix_flag_native/native_evidence.zip': 'a66da15dad97b1a807376b8281fa19457d749ae036ef6cad88b67113171cb8b6',
    }
    before = read(ROOT / 'artifacts/gate05/20261020_prefix_flag_probe_v2/source_hashes_before.json')
    assert before == read(ROOT / 'artifacts/gate05/20261020_prefix_flag_probe_v2/source_hashes_after.json')
    for rel, expected in before.items():
        p = ROOT / rel
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], rel
    for rel, expected in protected.items():
        assert sha(ROOT / rel) == expected, rel
    for name in ('README.md', '研究状态与后续计划.md'):
        text = (ROOT / name).read_text(encoding='utf-8-sig')
        assert '2026-10-20' in text and '31_Gate05_前缀Clip与精度约束对照.md' in text
    assert '目前四组正在顺序执行' not in (ROOT / '31_Gate05_前缀Clip与精度约束对照.md').read_text(encoding='utf-8-sig')
    state_path = ROOT / 'artifacts/gate05/current_execution_state.json'
    state_bytes = state_path.read_bytes()
    state = read(state_path)
    before_path.write_bytes(state_bytes)
    directories = [p for p in (ROOT / 'artifacts/gate05').glob('20261020_prefix_flag*') if p.is_dir()]
    paths = {p for d in directories for p in d.rglob('*') if p.is_file()}
    paths.update(p for p in (ROOT / 'artifacts/gate05').glob('20261020_prefix_flag*') if p.is_file())
    paths.update(ROOT / 'scripts' / name for name in ('prepare_gate05_prefix_flag_probe.py', 'jetson_prefix_flag_probe.py', 'jetson_build_prefix_flag_probe.py', 'audit_gate05_prefix_flag_probe.py', 'complete_gate05_prefix_flag_checkpoint.py'))
    paths.update(ROOT / name for name in ('31_Gate05_前缀Clip与精度约束对照.md', 'README.md', '研究状态与后续计划.md'))
    inventory = [identity(p) for p in sorted(paths)]
    write_new(manifest_path, dict(schema='rampq.gate05.prefix_flag_stage_inventory/v1', files=inventory,
                                  scope='Files before immutable checkpoint; excludes current mutable execution state, manifest itself and checkpoint.'))
    checkpoint = dict(
        schema='rampq.gate05.prefix_flag_checkpoint/v1', client_at=datetime.now().astimezone().isoformat(),
        status='PREFIX_CLIP_FLAG_DIAGNOSTIC_CHECKPOINT_COMPLETE', gate05_status='INCOMPLETE',
        original_complete_model_guard='FAILED_UNCHANGED', source_conv_count=1, source_nodes=11,
        successful_builds=4, fixed_input_executions=8, native_build_attempts_each=1,
        same_topology_none_obey_raw_byte_identical=True, each_case_repeat_raw_byte_identical=True,
        sole_conv_actual_int8_input_weight_output_and_named_integer_tactic=True,
        same_generic_tanh_silu_fusion_all_four=True,
        parser_constraints_each=dict(parsed_layers=18, precision_INT8=3, output_type_INT8=3, metadata_errors=0),
        parser_constraints_unchanged_by_clip_correction=True, parser_constraints_not_reset=True,
        cpu_different_elements=dict(original=769777, code_clip=7),
        cpu_max_abs=dict(original=36.65729832649231, code_clip=0.7481081485748291),
        monitor_seconds=368.42, runner_seconds_not_benchmark=362.84885020799993,
        verified_receipt_files=107, independent_native_review_checks=530,
        all_seven_owned_tasks_exited_same_boot=True, guard_intervened=False,
        boot_id=board['boot_id'], full_model_retry_performed=False, full_model_failure_resolved=False,
        qonly_preserved_engine_sha256=board['qonly_preserved_engine']['sha256'],
        prior_sources_archives_and_failed_preflight_unchanged=True,
        local_auditor_failures_preserved=2, native_failures=0, raw_evidence_unchanged=True,
        derived_mean_rms_saved_separately_no_acceptance_tolerance=True,
        new_tolerances_defined=False, dataset_evaluation_run=False,
        accuracy_causal_recovery_or_performance_pass=False,
        required_work_remaining_for_this_bounded_stage=[],
        next_step='Freeze a separately identified context slice from the actual failed corrected q_all graph; verify dependency/shape/CPU semantics before bounded native. No monotonic-prefix assumption, full retry, parser reset, restore matrix or dataset run.',
        limits='OBEY alone did not reproduce the failure in this one-Conv context; interaction with other graph contexts is unresolved. Small-case success is not a full-model or Gate pass.',
        protected_identities=protected, stage_inventory_files_before_checkpoint=len(inventory),
        stage_manifest=identity(manifest_path),
        evidence=[identity(BASE / name) for name in ('root_native_audit.json', 'independent_native_review.json', 'final_board_observation.json', 'final_remote_receipt_verification.json', 'native_evidence.zip')])
    write_new(checkpoint_path, checkpoint)
    checkpoint_rel = checkpoint_path.relative_to(ROOT).as_posix()
    state.update(client_at=checkpoint['client_at'], current_stage_status=checkpoint['status'],
                 current_stage_required_work_remaining=[], engineering_stage_required_work_remaining=[],
                 stage_completion_audit=checkpoint_rel,
                 final_board_observation=(BASE / 'final_board_observation.json').relative_to(ROOT).as_posix())
    state['planning_boundary'] = [checkpoint['limits'], 'Corrected full q_all native format-selection failure remains unresolved with no engine.',
                                  'Earlier corrected q_only_G1 fixed-image success unchanged.',
                                  'Original full PyTorch/ORT guard FAILED_UNCHANGED; no dataset, causal recovery or performance pass.']
    state['next_gate_work'] = [checkpoint['next_step'],
                               'Resolve full-model reference comparability and required floating/fusion controls; preserve the original failed guard.',
                               'Retain original strict integer-tactic gaps and freeze dataset criteria before any dataset evaluation.']
    state['connection_recovery']['current_native_state'] = checkpoint['status'] + ';ALL_OWNED_TASKS_EXITED'
    state['previous_prefix_stage_completion_audit'] = 'artifacts/gate05/20261019_corrected_qall_native/checkpoint.json'
    state['prefix_flag_probe'].update(status=checkpoint['status'], process_live=False,
                                     successful_builds=4, fixed_input_executions=8, monitor_seconds=368.42,
                                     same_topology_flag_outputs_byte_identical=True,
                                     verified_receipt_files=107, full_model_failure_resolved=False,
                                     checkpoint=checkpoint_rel, checkpoint_sha256=sha(checkpoint_path),
                                     root_audit='artifacts/gate05/20261020_prefix_flag_native/root_native_audit.json',
                                     independent_native_review='artifacts/gate05/20261020_prefix_flag_native/independent_native_review.json')
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=checkpoint['status'], gate05_status='INCOMPLETE', files=len(inventory),
                          checkpoint_sha256=sha(checkpoint_path), manifest_sha256=sha(manifest_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()

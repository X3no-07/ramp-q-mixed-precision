"""Seal this two-Conv diagnostic's saved evidence and current queue state."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05/20261020_twoconv_context_native'
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
    checkpoint_path, manifest_path = BASE / 'checkpoint.json', BASE / 'stage_manifest.json'
    before_path = BASE / 'state_before_completion.json'
    assert not any(p.exists() for p in (checkpoint_path, manifest_path, before_path))
    audit = read(BASE / 'root_native_audit.json')
    review = read(BASE / 'independent_native_review.json')
    grid = read(BASE / 'independent_code_grid_and_exit_review.json')
    board = read(BASE / 'final_board_observation.json')
    receipt = read(RAW / 'remote_receipt.json')
    assert audit['successful_output_cases'] == 2 and audit['raw_tree_unchanged']
    assert review['check_count'] == len(review['checks']) == 367 and all(c['passed'] for c in review['checks'])
    assert review['raw_tree_unchanged'] and grid['raw_tree_unchanged']
    assert len(receipt['files']) == board['receipt_file_count'] == 89
    assert board['all_remote_receipt_files_still_match'] and all(board['remote_checks'].values())
    assert {p.relative_to(RAW).as_posix() for p in RAW.rglob('*') if p.is_file()} == set(receipt['files']) | {'remote_receipt.json'}
    for rel, expected in receipt['files'].items():
        p = RAW / rel
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], rel
    assert board['boot_id'] == '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
    assert len(board['known_processes']) == 5 and all(not p['exists'] for p in board['known_processes'].values())
    assert not board['task_processes'] and not board['corrected_qall_engine_paths']
    assert board['fan']['stdout'].strip() == 'active' and 'NRestarts=0' in board['jtop']['stdout']
    assert board['qonly_preserved_engine']['sha256'] == '486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476'
    assert audit['monitor_status'] == 'BUILD_SCRIPT_OK' and audit['guard_group_empty'] and not audit['guard_intervention']
    assert audit['cross_native_comparison']['byte_identical']
    for name, case in audit['cases'].items():
        assert case['build']['status'] == 'TWOCONV_CONTEXT_ENGINE_SAVED'
        assert case['repeat']['byte_identical'] and case['repeat']['all_pairs_finite'] and case['child_cleanup']['returncode'] == 0
        assert case['cpu_difference']['all_pairs_finite'] and case['cpu_difference']['different_elements'] == 18
        assert case['parser_constraints'] == dict(parsed_layers=31, precision_set_count=5, output_type_set_count=5, metadata_errors=[])
        actual = case['actual_conv_mapping']
        assert actual['actual_count'] == 2 and actual['two_distinct_source_rows']
        for row in actual['source_mappings']:
            assert row['unique'] and row['int8_input_weight'] and row['int8_output'] and row['named_integer_tactic']
        first, second = [r['actual_row'] for r in actual['source_mappings']]
        assert first['Activation'] == 'SWISH' and second['Activation'] == 'GENERIC'
        assert 'rampq_model_2_cv1_conv_Clip' in second['Name'] and len(second['PointWiseExpressionOperations']) == 7
        codes = grid['cases'][name]
        assert codes['all_values_exact_float32_dequantized_code_grid'] and codes['all_differing_codes_exactly_one_step']
        assert codes['signed_code_delta_counts'] == {'-1': 10, '0': 819182, '1': 8}
    protected = {
        'artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence.zip': '5eac1543e5151708585e57923dd82410c9b21aac81cb5cfd1137e057e35ed205',
        'artifacts/gate05/20261019_corrected_qall_native/build_failed_evidence.zip': '027433baa5b1e18fafb72cf64b70e58e1bd2aa05b295fd8f539ae8317e1c9a99',
        'artifacts/gate05/20261019_corrected_qall_native/parser_metadata_evidence.zip': 'a567b474be0e563448ce96b7b5cc0f1e9bfff1036461b9512efe9b8c1ba32d78',
        'artifacts/gate05/20261018_int8/input_v1/preflight.json': '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21',
        'artifacts/gate05/20261020_prefix_flag_native/checkpoint.json': 'fe2c0617a66dce3107be0e4ae9aed0fcdb091d50a131377110ce51d9045775c8',
        'artifacts/gate05/20261020_prefix_flag_native/native_evidence.zip': 'a66da15dad97b1a807376b8281fa19457d749ae036ef6cad88b67113171cb8b6',
        'artifacts/gate05/20261020_twoconv_context_probe.zip': '599190d18de4f57bb80911ead9e436661a3a86daa4117817044079d71c25d525',
        'artifacts/gate05/20261020_twoconv_context_native/native_evidence.zip': '6f9ed4fd0f3b21617eb48938cdf79910155a4d18bd29e49c55b168a88226594c',
    }
    source_before = read(ROOT / 'artifacts/gate05/20261020_twoconv_context_probe/source_hashes_before.json')
    assert source_before == read(ROOT / 'artifacts/gate05/20261020_twoconv_context_probe/source_hashes_after.json')
    for rel, expected in source_before.items():
        p = ROOT / rel
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], rel
    for rel, expected in protected.items():
        assert sha(ROOT / rel) == expected, rel
    for name in ('README.md', '研究状态与后续计划.md'):
        text = (ROOT / name).read_text(encoding='utf-8-sig')
        assert '32_Gate05_双卷积上下文定位.md' in text and '2026-10-20' in text
    options_path = ROOT / 'artifacts/gate05/20261020_twoconv_context_review/next_context_static_options.json'
    assert sha(options_path) == '41aaa27de8b5f275cf3632933e3d1609c03a0028b2d6906f1d8e1f4022ee6378'
    state_path = ROOT / 'artifacts/gate05/current_execution_state.json'
    state = read(state_path)
    before_path.write_bytes(state_path.read_bytes())
    directories = [p for p in (ROOT / 'artifacts/gate05').glob('20261020_twoconv_context*') if p.is_dir()]
    paths = {p for d in directories for p in d.rglob('*') if p.is_file()}
    paths.update(p for p in (ROOT / 'artifacts/gate05').glob('20261020_twoconv_context*') if p.is_file())
    paths.update(ROOT / 'scripts' / n for n in ('prepare_gate05_twoconv_context_probe.py', 'jetson_twoconv_context_probe.py', 'jetson_build_twoconv_context_probe.py', 'audit_gate05_twoconv_context_probe.py', 'complete_gate05_twoconv_context_checkpoint.py'))
    paths.update(ROOT / n for n in ('32_Gate05_双卷积上下文定位.md', 'README.md', '研究状态与后续计划.md'))
    inventory = [identity(p) for p in sorted(paths)]
    write_new(manifest_path, dict(schema='rampq.gate05.twoconv_context_stage_inventory/v1', files=inventory,
                                  scope='Files before immutable checkpoint; excludes current mutable execution state, manifest itself and checkpoint.'))
    next_step = 'Freeze one structurally distinct DFL or PE context with an authenticated internal input fixture; static options are not executed. Do not assume prefix monotonicity, reset parser constraints, retry full model/matrix or run dataset.'
    checkpoint = dict(schema='rampq.gate05.twoconv_context_checkpoint/v1', client_at=datetime.now().astimezone().isoformat(),
                      status='TWOCONV_CONTEXT_DIAGNOSTIC_CHECKPOINT_COMPLETE', gate05_status='INCOMPLETE',
                      original_complete_model_guard='FAILED_UNCHANGED', source_convs=2, source_nodes=19, source_initializers=20,
                      source_graph_sha256='755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee',
                      two_slice_CPU_executions_only=True, slice_CPU_original_corrected_outputs_byte_exact=True,
                      successful_native_builds=2, native_build_attempts_each=1, successful_fixed_input_executions=4,
                      same_topology_none_obey_raw_byte_identical=True, each_case_repeat_byte_identical=True,
                      both_conv_input_weight_output_INT8_with_named_integer_tactics=True,
                      first_conv_fusion='DEDICATED_SWISH', second_conv_fusion='GENERIC_SILU_WITH_THIRD_CLIP;7_POINTWISE_OPERATIONS',
                      first_conv_context_changes_tactic_relative_to_one_conv_prefix=True,
                      parser_constraints_each=dict(parsed_layers=31, precision_INT8=5, output_type_INT8=5, metadata_errors=0),
                      parser_constraints_unchanged_by_clip_correction=True, parser_constraints_not_reset=True,
                      native_vs_CPU_differing_elements=18, native_vs_CPU_total_elements=819200,
                      native_minus_CPU_signed_code_counts={'-1': 10, '0': 819182, '1': 8},
                      all_outputs_exact_float32_dequantized_code_grid=True, native_vs_CPU_max_abs=1.5619525909423828,
                      original_scale_float32=1.5619523525238037, float_difference_not_claimed_bit_equal_to_scale=True,
                      monitor_seconds=155.63, runner_seconds_not_benchmark=149.41414064000128,
                      verified_receipt_files=89, independent_native_review_checks=367, code_grid_and_five_PID_review_passed=True,
                      all_five_owned_tasks_exited_same_boot=True, guard_intervened=False, boot_id=board['boot_id'],
                      full_model_retry_performed=False, full_model_failure_resolved=False, earlier_qonly_success_preserved=True,
                      prior_sources_archives_and_failed_preflight_unchanged=True, raw_evidence_unchanged=True,
                      new_tolerances_defined=False, dataset_evaluation_run=False, accuracy_causal_recovery_or_performance_pass=False,
                      required_work_remaining_for_this_bounded_stage=[], next_step=next_step,
                      limits='This two-Conv context did not reproduce full q_all format failure; OBEY interactions elsewhere remain possible. Not a full-model or Gate pass.',
                      protected_identities=protected, stage_inventory_files_before_checkpoint=len(inventory), stage_manifest=identity(manifest_path),
                      evidence=[identity(BASE / n) for n in ('root_native_audit.json', 'independent_native_review.json', 'independent_code_grid_and_exit_review.json', 'final_board_observation.json', 'native_evidence.zip')] + [identity(options_path)])
    write_new(checkpoint_path, checkpoint)
    checkpoint_rel = checkpoint_path.relative_to(ROOT).as_posix()
    state.update(client_at=checkpoint['client_at'], current_stage_status=checkpoint['status'],
                 current_stage_required_work_remaining=[], engineering_stage_required_work_remaining=[],
                 stage_completion_audit=checkpoint_rel, final_board_observation=(BASE / 'final_board_observation.json').relative_to(ROOT).as_posix())
    state['planning_boundary'] = [checkpoint['limits'], 'Two-convolution extension changes first Conv fusion/tactic, so added nodes alone are not a proven cause.',
                                  'Corrected full q_all remains FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE; earlier corrected q_only_G1 success preserved.',
                                  'Original full PyTorch/ORT guard FAILED_UNCHANGED; no dataset, causal recovery or performance pass.']
    state['next_gate_work'] = [next_step, 'Resolve full reference comparability and floating/fusion controls; preserve original failed guard and strict integer-tactic gaps before dataset evaluation.']
    state['connection_recovery']['current_native_state'] = checkpoint['status'] + ';ALL_OWNED_TASKS_EXITED'
    state['previous_twoconv_stage_completion_audit'] = 'artifacts/gate05/20261020_prefix_flag_native/checkpoint.json'
    state['twoconv_context_probe'].update(status=checkpoint['status'], process_live=False, successful_builds=2, fixed_input_executions=4,
                                          monitor_seconds=155.63, same_topology_flag_outputs_byte_identical=True, verified_receipt_files=89,
                                          native_vs_CPU_differing_codes=18, full_model_failure_resolved=False,
                                          checkpoint=checkpoint_rel, checkpoint_sha256=sha(checkpoint_path),
                                          root_audit='artifacts/gate05/20261020_twoconv_context_native/root_native_audit.json',
                                          independent_native_review='artifacts/gate05/20261020_twoconv_context_native/independent_native_review.json',
                                          next_static_options=options_path.relative_to(ROOT).as_posix())
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=checkpoint['status'], gate05_status='INCOMPLETE', files=len(inventory),
                          checkpoint_sha256=sha(checkpoint_path), manifest_sha256=sha(manifest_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()

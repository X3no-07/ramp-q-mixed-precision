"""Seal original DFL controls and the saved local topology/flag comparison."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05/20261020_dfl_original_native'
RAW = BASE / 'native_evidence'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identity(p):
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))


def write_new(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    checkpoint_path, manifest_path, before_path = [BASE / n for n in ('checkpoint.json', 'stage_manifest.json', 'state_before_completion.json')]
    assert not any(p.exists() for p in (checkpoint_path, manifest_path, before_path))
    audit = read(BASE / 'root_native_audit.json')
    review = read(BASE / 'independent_native_review.json')
    supplement = read(BASE / 'independent_failure_signature_and_runtime_review.json')
    cross = read(BASE / 'topology_flag_comparison.json')
    board = read(BASE / 'final_board_observation.json')
    receipt = read(RAW / 'remote_receipt.json')
    assert audit['successful_output_cases'] == 1 and audit['raw_tree_unchanged']
    assert review['check_count'] == len(review['checks']) == 470 and all(c['passed'] for c in review['checks'])
    assert review['saved_evidence_unchanged'] and supplement['both_raw_trees_unchanged']
    assert len(receipt['files']) == board['receipt_file_count'] == 84
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
    assert board['earlier_corrected_DFL_none_engine_sha256'] == 'a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e'
    assert not board['earlier_corrected_DFL_obey_engine_present']
    assert audit['monitor_status'] == 'BUILD_SCRIPT_OK' and audit['guard_group_empty'] and not audit['guard_intervention']
    none, obey = [audit['cases'][n] for n in ('original_none', 'original_obey')]
    for case in (none, obey):
        assert case['parser_constraints'] == dict(parsed_layers=14, precision_set_count=2, output_type_set_count=2, metadata_errors=[])
        assert case['child_cleanup']['direct_python_api_child_terminated']
    assert none['build']['status'] == 'DFL_ORIGINAL_ENGINE_SAVED' and none['child_cleanup']['returncode'] == 0
    assert none['repeat']['byte_identical'] and none['repeat']['all_pairs_finite']
    assert none['cpu_difference']['all_pairs_finite'] and none['cpu_difference']['different_elements'] == 33566
    assert none['cpu_difference']['max_abs'] == 11.957853466272354
    mapping = none['actual_conv_mapping']
    assert mapping['one_unique_source_row'] and mapping['actual_count'] == 1
    row = mapping['source_mappings'][0]
    assert row['int8_input_weight'] and row['named_integer_tactic'] and not row['int8_output']
    assert row['actual_row']['Outputs'][0]['Format/Datatype'] == 'Row major linear FP32'
    assert obey['build']['status'] == 'DFL_ORIGINAL_BUILD_FAILED' and obey['child_cleanup']['returncode'] == 1
    assert not obey['engine_present'] and not obey['inference_run']
    assert cross['raw_trees_unchanged'] and cross['same_fixture_and_saved_CPU_reference']
    assert cross['same_Q_DQ_explicit_constraint_values'] and cross['all_common_config_flags_equal_except_OBEY']
    for name in ('original_obey', 'code_clip_obey'):
        assert any('filterQDQFormats::4421' in s and '!n->candidateRequirements.empty()' in s for s in cross['cases'][name]['error_lines'])
        assert not cross['cases'][name]['engine_present'] and not cross['cases'][name]['raw_output_present']
    prepared = ROOT / 'artifacts/gate05/20261020_dfl_original_probe'
    reuse = read(prepared / 'static_reuse_audit.json')
    assert reuse['new_CPU_executions'] == reuse['new_ORT_executions'] == reuse['new_Torch_executions'] == 0
    assert not reuse['new_fixture_generated'] and reuse['reused_original_code_clip_outputs_byte_identical']
    protected = {
        'artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence.zip': '5eac1543e5151708585e57923dd82410c9b21aac81cb5cfd1137e057e35ed205',
        'artifacts/gate05/20261019_corrected_qall_native/build_failed_evidence.zip': '027433baa5b1e18fafb72cf64b70e58e1bd2aa05b295fd8f539ae8317e1c9a99',
        'artifacts/gate05/20261019_corrected_qall_native/parser_metadata_evidence.zip': 'a567b474be0e563448ce96b7b5cc0f1e9bfff1036461b9512efe9b8c1ba32d78',
        'artifacts/gate05/20261018_int8/input_v1/preflight.json': '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21',
        'artifacts/gate05/20261020_dfl_context_native/checkpoint.json': '28edf81c0be1f72065a8952f84dc31330ba95709d50ed13aafc16a5bb56c602b',
        'artifacts/gate05/20261020_dfl_context_native/native_evidence.zip': 'a53750401ccdbb7c008c14e75cadec80a586a05f2baa32f5e985abdff527bd24',
        'artifacts/gate05/20261020_dfl_context_probe.zip': '05b9b50e4ddc294151121044facef5e1711ac368c793ca5232c1c1a557c42b9f',
        'artifacts/gate05/20261020_dfl_original_probe.zip': '6ea5c52c4697339692d3d5e4efac01a46d033e34855a3ed8f60e4aa1d1a22709',
        'artifacts/gate05/20261020_dfl_original_native/native_evidence.zip': '9f027ada5086843e4b967a6ab4f5a4e1d5f23bc904e30b9913e8e619db7dd96d',
    }
    sources = read(prepared / 'source_hashes_before.json')
    assert sources == read(prepared / 'source_hashes_after.json')
    for rel, expected in sources.items():
        p = ROOT / rel
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], rel
    for rel, expected in protected.items():
        assert sha(ROOT / rel) == expected, rel
    for name in ('README.md', 'docs/historical_status_and_plan.md'):
        text = (ROOT / name).read_text(encoding='utf-8-sig')
        assert 'docs/34_gate05_original_dfl_topology_and_constraint_controls.md' in text and '2026-10-20' in text
    next_path = BASE / 'next_control_plan.json'
    assert sha(next_path) == 'de9e44fcd058cd6d855174af6934f657e5c312ee5284f8463322c0c029160be2'
    state_path = ROOT / 'artifacts/gate05/current_execution_state.json'
    state = read(state_path)
    before_path.write_bytes(state_path.read_bytes())
    directories = [p for p in (ROOT / 'artifacts/gate05').glob('20261020_dfl_original*') if p.is_dir()]
    paths = {p for d in directories for p in d.rglob('*') if p.is_file()}
    paths.update(p for p in (ROOT / 'artifacts/gate05').glob('20261020_dfl_original*') if p.is_file())
    paths.update(ROOT / 'scripts' / n for n in ('prepare_gate05_dfl_original_probe.py', 'jetson_dfl_original_probe.py', 'jetson_build_dfl_original_probe.py', 'audit_gate05_dfl_original_probe.py', 'compare_gate05_dfl_topology_flag.py', 'complete_gate05_dfl_original_checkpoint.py'))
    paths.update(ROOT / n for n in ('docs/34_gate05_original_dfl_topology_and_constraint_controls.md', 'README.md', 'docs/historical_status_and_plan.md'))
    inventory = [identity(p) for p in sorted(paths)]
    write_new(manifest_path, dict(schema='rampq.gate05.dfl_original_stage_inventory/v1', files=inventory,
                                  scope='Files before immutable checkpoint; excludes current mutable execution state, manifest itself and checkpoint.'))
    next_step = 'Separately freeze a corrected DFL cut after Softmax: one exact source Softmax CPU run from saved logits creates probability fixture, one 8-node tail CPU run must match saved DFL CPU bytes; then bounded NONE/OBEY. No 72Conv upstream rerun, parser reset, full-model retry or dataset.'
    checkpoint = dict(schema='rampq.gate05.dfl_original_checkpoint/v1', client_at=datetime.now().astimezone().isoformat(),
                      status='DFL_ORIGINAL_LOCAL_FACTORIAL_CHECKPOINT_COMPLETE', gate05_status='INCOMPLETE',
                      original_complete_model_guard='FAILED_UNCHANGED', source_convs=1, source_nodes=9, source_initializers=7,
                      new_CPU_or_ORT_or_Torch_executions=0, new_fixture_generated=False,
                      native_build_attempts=2, native_build_attempts_each=1, successful_native_builds=1, failed_native_builds=1,
                      original_none_build_seconds=52.32060847998946, original_obey_build_seconds=1.5088661759946262,
                      original_none_executions=2, original_obey_executions=0, original_none_repeat_byte_identical=True,
                      original_Clip_bounds=[-0.9999907612800598, 0.9999907612800598], original_Clip_attribute_mutation=False,
                      original_none_input_weight_INT8_with_named_integer_tactic=True, original_none_conv_output='FP32',
                      original_none_cpu_differing_elements=33566, output_elements=33600, original_none_cpu_max_abs=11.957853466272354,
                      original_none_numerical_repair_claim=False, corrected_none_cpu_max_abs=1.9073486328125e-06,
                      both_topology_OBEY_cases_same_filterQDQFormats_signature=True, code_clip_derivative_required_for_this_local_failure=False,
                      both_OBEY_cases_no_engine_or_inference=True, original_vs_corrected_NONE_raw_different_elements=33566,
                      same_fixture_saved_CPU_weights_scales_and_named_Q_DQ_requests=True, only_actual_config_flag_changed='OBEY_PRECISION_CONSTRAINTS',
                      four_single_observations_across_two_stages=True, randomized_or_repeated_factorial_study=False,
                      topology_and_import_correction_remain_combined_factor=True,
                      parser_constraints_each=dict(parsed_layers=14, precision_INT8=2, output_type_INT8=2, metadata_errors=0),
                      monitor_status='BUILD_SCRIPT_OK', monitor_status_means_both_builds_passed=False,
                      monitor_seconds=69.3, runner_seconds_not_benchmark=64.02708412799984,
                      verified_receipt_files=84, independent_native_review_checks=470, raw_evidence_unchanged=True,
                      all_five_owned_tasks_exited_same_boot=True, guard_intervened=False, boot_id=board['boot_id'],
                      full_model_native_retry_performed=False, full_model_failure_resolved=False, full_model_without_OBEY_build_proven=False,
                      named_full_model_failing_node_or_unique_cause_identified=False, earlier_qonly_success_preserved=True,
                      prior_sources_archives_and_failed_preflight_unchanged=True, new_tolerances_defined=False,
                      dataset_evaluation_run=False, accuracy_causal_recovery_or_performance_pass=False,
                      required_work_remaining_for_this_bounded_stage=[], next_step=next_step,
                      limits='Both original and corrected DFL cuts fail with OBEY. NONE compilation alone does not fix original numerical error. No full-model flag workaround or unique-root-cause claim.',
                      protected_identities=protected, stage_inventory_files_before_checkpoint=len(inventory), stage_manifest=identity(manifest_path),
                      evidence=[identity(BASE / n) for n in ('root_native_audit.json', 'topology_flag_comparison.json', 'independent_native_review.json', 'independent_failure_signature_and_runtime_review.json', 'final_board_observation.json', 'native_evidence.zip', 'next_control_plan.json')])
    write_new(checkpoint_path, checkpoint)
    checkpoint_rel = checkpoint_path.relative_to(ROOT).as_posix()
    state.update(client_at=checkpoint['client_at'], current_stage_status=checkpoint['status'], current_stage_required_work_remaining=[],
                 engineering_stage_required_work_remaining=[], stage_completion_audit=checkpoint_rel,
                 final_board_observation=(BASE / 'final_board_observation.json').relative_to(ROOT).as_posix())
    state['planning_boundary'] = [checkpoint['limits'], 'Compilation compatibility and native numeric error are distinct observations; same inspector formats/tactic do not imply equal outputs.',
                                  'Original full PyTorch/ORT guard FAILED_UNCHANGED; corrected full q_all still no engine, earlier corrected q_only_G1 success preserved.',
                                  'Only corrected-CPU internal fixture tested; no native upstream distribution or dataset/causal recovery/performance pass.']
    state['next_gate_work'] = [next_step, 'Resolve full reference comparability, floating/fusion controls and prior strict integer-tactic gaps before dataset criteria and evaluation.']
    state['connection_recovery']['current_native_state'] = checkpoint['status'] + ';ALL_OWNED_TASKS_EXITED'
    state['previous_dfl_original_stage_completion_audit'] = 'artifacts/gate05/20261020_dfl_context_native/checkpoint.json'
    state['dfl_original_probe'].update(status=checkpoint['status'], process_live=False, successful_builds=1, failed_builds=1,
                                       fixed_input_executions=2, monitor_seconds=69.3, original_none_build='SUCCESS_WITH_LARGE_NUMERICAL_DIFFERENCE',
                                       original_obey_build='FAILED_SAME_FILTER_QDQ_FORMATS_ASSERTION_NO_ENGINE', verified_receipt_files=84,
                                       full_model_failure_resolved=False, checkpoint=checkpoint_rel, checkpoint_sha256=sha(checkpoint_path),
                                       root_audit='artifacts/gate05/20261020_dfl_original_native/root_native_audit.json',
                                       four_case_comparison='artifacts/gate05/20261020_dfl_original_native/topology_flag_comparison.json',
                                       independent_native_review='artifacts/gate05/20261020_dfl_original_native/independent_native_review.json',
                                       next_control_plan=next_path.relative_to(ROOT).as_posix())
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=checkpoint['status'], gate05_status='INCOMPLETE', files=len(inventory),
                          checkpoint_sha256=sha(checkpoint_path), manifest_sha256=sha(manifest_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()

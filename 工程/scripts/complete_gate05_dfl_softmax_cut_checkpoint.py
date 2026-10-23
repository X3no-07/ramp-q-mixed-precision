"""Seal the saved probability-input DFL cut diagnostic; never run inference."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native'
PREPARED = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_probe'
RAW = BASE / 'native_evidence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def write_new(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    checkpoint_path, manifest_path, before_path = [BASE / name for name in (
        'checkpoint.json', 'stage_manifest.json', 'state_before_completion.json')]
    assert not any(path.exists() for path in (checkpoint_path, manifest_path, before_path))
    audit = read(BASE / 'root_native_audit.json')
    cross = read(BASE / 'context_comparison.json')
    review = read(BASE / 'independent_native_review.json')
    board = read(BASE / 'final_board_observation.json')
    receipt = read(RAW / 'remote_receipt.json')
    cpu = read(PREPARED / 'cpu_diagnostic.json')
    graph = read(PREPARED / 'graph_identity_audit.json')
    assert review['check_count'] == len(review['checks']) == 597
    assert all(check['passed'] for check in review['checks'])
    assert review['old_new_saved_raw_unchanged'] and not review['new_CPU_OR_native_inference_by_reviewer']
    assert not review['SSH_by_reviewer'] and not review['same_native_external_inputs']
    assert review['same_CPU_reference'] and review['same_supported_common_flags_and_named_QDQ_requests']
    assert audit['successful_output_cases'] == 1 and audit['raw_tree_unchanged']
    assert len(receipt['files']) == board['receipt_file_count'] == 89
    assert board['all_remote_receipt_files_still_match'] and all(board['remote_checks'].values())
    assert {p.relative_to(RAW).as_posix() for p in RAW.rglob('*') if p.is_file()} == set(receipt['files']) | {'remote_receipt.json'}
    for relative, expected in receipt['files'].items():
        path = RAW / relative
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], relative
    assert sha(RAW / 'remote_receipt.json') == board['receipt_sha256']
    assert board['boot_id'] == '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
    assert len(board['known_processes']) == 5 and all(not p['exists'] for p in board['known_processes'].values())
    assert not board['task_processes'] and not board['corrected_qall_engine_paths']
    assert board['fan']['stdout'].strip() == 'active' and 'NRestarts=0' in board['jtop']['stdout']
    assert board['qonly_preserved_engine']['sha256'] == '486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476'
    assert board['earlier_corrected_DFL_none_engine_sha256'] == 'a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e'
    assert not board['earlier_corrected_DFL_obey_engine_present']
    assert audit['monitor_status'] == 'BUILD_SCRIPT_OK' and audit['guard_group_empty'] and not audit['guard_intervention']
    none, obey = [audit['cases'][name] for name in ('code_clip_none', 'code_clip_obey')]
    for case in (none, obey):
        assert case['parser_constraints'] == dict(parsed_layers=12, precision_set_count=2, output_type_set_count=2, metadata_errors=[])
        assert case['child_cleanup']['direct_python_api_child_terminated']
    assert none['build']['status'] == 'DFL_SOFTMAX_CUT_ENGINE_SAVED' and none['child_cleanup']['returncode'] == 0
    assert none['repeat']['byte_identical'] and none['repeat']['all_pairs_finite']
    assert none['cpu_difference']['different_elements'] == 11779
    assert none['cpu_difference']['max_abs'] == 1.9073486328125e-6
    assert none['actual_conv_mapping']['one_unique_source_row'] and none['actual_conv_mapping']['actual_count'] == 1
    row = none['actual_conv_mapping']['source_mappings'][0]
    assert row['int8_input_weight'] and row['named_integer_tactic'] and not row['int8_output']
    assert row['actual_row']['Outputs'][0]['Format/Datatype'] == 'Row major linear FP32'
    assert obey['build']['status'] == 'DFL_SOFTMAX_CUT_BUILD_FAILED' and obey['child_cleanup']['returncode'] == 1
    assert not obey['engine_present'] and not obey['inference_run']
    failure_logs = list(RAW.glob('results/*/code_clip_obey/builder.log'))
    assert len(failure_logs) == 1
    failure_text = failure_logs[0].read_text(encoding='utf-8')
    assert 'filterQDQFormats::4421' in failure_text and '!n->candidateRequirements.empty()' in failure_text
    assert 'buildSerializedNetwork::751' in failure_text
    assert cross['raw_trees_unchanged'] and cross['shared_CPU_output_reference'] and cross['CPU_composition_byte_equivalent']
    assert cross['common_named_Q_DQ_requests_equal'] and cross['common_config_flags_equal_except_OBEY']
    comparison = cross['raw_output_comparisons']['softmax_inside_none_vs_probability_input_cut_none']
    assert comparison['byte_identical'] and comparison['different_elements'] == 0
    assert not cross['new_inference']
    assert cpu['new_CPU_executions'] == 2 and cpu['generator_CPU_runs'] == cpu['cut_CPU_runs'] == 1
    assert cpu['new_72Conv_or_old_DFL_or_full_detector_runs'] == 0 and cpu['native_executions'] == 0
    assert cpu['byte_identical_saved_corrected_DFL_output'] and cpu['differing_elements'] == 0
    assert graph['all_nodes_initializers_and_IO_declarations_exact_source_bytes']
    assert graph['code_clip_cut']['nodes'] == 8 and graph['code_clip_cut']['initializers'] == 7
    assert graph['code_clip_cut']['conv_count'] == 1 and graph['Softmax_generator']['nodes'] == 1
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
        'artifacts/gate05/20261020_dfl_original_native/checkpoint.json': '6a85e00d80464f07a9ce4d88c09191f2ae3e0e79b1c7b78ff4c744cd6077e665',
        'artifacts/gate05/20261020_dfl_softmax_cut_probe.zip': 'aeff82c52ff3b2ec1d83b4b06c21f014b30699a1e428a0c5fb37bcd3f6cb25dd',
        'artifacts/gate05/20261020_dfl_softmax_cut_native/native_evidence.zip': '1b531ae8079a7563432e9702071ca8d384695f2a2624d80ab3da7798fa7215cc',
        'artifacts/gate05/20261020_dfl_softmax_cut_native/independent_native_review.json': '222617e5f6ec15c0dabb6f2e1ce697cd989359dfcf2d5914469c78f11a7326ad',
        'artifacts/gate05/20261020_dfl_softmax_cut_native/next_control_plan.json': '14d7710f373fd56a8d68ec15219e234c96a9325938e25ad703696b879160dfc8',
    }
    for prefix in ('source_hashes', 'seal_native_source_hashes'):
        sources = read(PREPARED / (prefix + '_before.json'))
        assert sources == read(PREPARED / (prefix + '_after.json'))
        for relative, expected in sources.items():
            path = ROOT / relative
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], relative
    for relative, expected in protected.items():
        assert sha(ROOT / relative) == expected, relative
    assert sha(BASE / 'native_evidence.zip') == board['archive_sha256']
    report = '35_Gate05_DFL_Softmax邻接定位.md'
    for name in ('README.md', '研究状态与后续计划.md'):
        assert report in (ROOT / name).read_text(encoding='utf-8-sig')
    report_text = (ROOT / report).read_text(encoding='utf-8-sig')
    assert '597 项全部通过' in report_text and 'INCOMPLETE' in report_text
    state_path = ROOT / 'artifacts/gate05/current_execution_state.json'
    state = read(state_path)
    assert state['dfl_softmax_cut_probe']['process_live']
    before_path.write_bytes(state_path.read_bytes())
    directories = [p for p in (ROOT / 'artifacts/gate05').glob('20261020_dfl_softmax_cut*') if p.is_dir()]
    paths = {p for directory in directories for p in directory.rglob('*') if p.is_file()}
    paths.update(p for p in (ROOT / 'artifacts/gate05').glob('20261020_dfl_softmax_cut*') if p.is_file())
    paths.update(ROOT / 'scripts' / name for name in (
        'prepare_gate05_dfl_softmax_cut_probe.py', 'jetson_dfl_softmax_cut_probe.py',
        'jetson_build_dfl_softmax_cut_probe.py', 'audit_gate05_dfl_softmax_cut_probe.py',
        'compare_gate05_dfl_softmax_context.py', 'complete_gate05_dfl_softmax_cut_checkpoint.py'))
    paths.update(ROOT / name for name in (report, 'README.md', '研究状态与后续计划.md'))
    inventory = [identity(p) for p in sorted(paths)]
    write_new(manifest_path, dict(schema='rampq.gate05.dfl_softmax_cut_stage_inventory/v1', files=inventory,
        scope='Files before immutable checkpoint; excludes mutable execution state, manifest itself and checkpoint.'))
    next_step = ('Proposed only: separately freeze a 6-node/7-initializer corrected DFL QDQ+Conv cut removing terminal Constant/Reshape, '
        'reuse probability fixture and run the new cut CPU once with raw array bytes matching saved DFL output after shape correspondence; '
        'then one NONE/OBEY attempt each. Output boundary changes. No new Softmax, 72Conv, old DFL or full-model replay, parser reset or dataset.')
    checkpoint = dict(schema='rampq.gate05.dfl_softmax_cut_checkpoint/v1', client_at=datetime.now().astimezone().isoformat(),
        status='DFL_SOFTMAX_CUT_LOCAL_FAILURE_CHECKPOINT_COMPLETE', gate05_status='INCOMPLETE',
        original_complete_model_guard='FAILED_UNCHANGED', source_convs=1, source_nodes=8, source_initializers=7,
        new_CPU_executions=2, generator_CPU_runs=1, cut_CPU_runs=1, new_72Conv_or_old_DFL_or_full_detector_runs=0,
        CPU_composition_output_byte_exact=True, CPU_probability_fixture_is_native_upstream_output=False,
        native_build_attempts=2, native_build_attempts_each=1, successful_native_builds=1, failed_native_builds=1,
        none_build_seconds=none['build']['build_seconds_not_benchmark'], obey_build_seconds=obey['build']['build_seconds_not_benchmark'],
        none_executions=2, obey_executions=0, none_repeat_byte_identical=True,
        Clip_imported_bounds_corrected_to=[-127, 127], layer_precision_or_output_type_setter_or_reset=False,
        none_input_weight_INT8_with_named_integer_tactic=True, none_conv_output='FP32',
        none_cpu_differing_elements=11779, output_elements=33600, none_cpu_max_abs=none['cpu_difference']['max_abs'],
        same_corrected_9node_NONE_final_raw_bytes=True, old_new_native_external_inputs_identical=False,
        native_internal_probabilities_or_quantized_codes_equality_proven=False,
        explicit_Softmax_required_for_this_local_failure=False, both_context_OBEY_same_4421_751_no_engine=True,
        same_saved_CPU_reference_weights_scales_and_named_Q_DQ_requests=True,
        only_actual_config_flag_changed='OBEY_PRECISION_CONSTRAINTS',
        parser_constraints_each=dict(parsed_layers=12, precision_INT8=2, output_type_INT8=2, metadata_errors=0),
        monitor_status='BUILD_SCRIPT_OK', monitor_status_means_both_builds_passed=False,
        monitor_seconds=69.31, runner_seconds_not_benchmark=64.16891516800388,
        verified_receipt_files=89, independent_native_review_checks=597, raw_evidence_unchanged=True,
        all_five_owned_tasks_exited_same_boot=True, guard_intervened=False, boot_id=board['boot_id'],
        final_board_observation_client_at=board['client_at'], final_board_observation_board_at=board['board_time'],
        power_query_privileged_full_verification=False, board_clock_corrected=False,
        full_model_native_retry_performed=False, full_model_failure_resolved=False,
        named_full_model_failing_node_or_unique_cause_identified=False, earlier_qonly_success_preserved=True,
        prior_sources_archives_and_failed_preflight_unchanged=True, new_tolerances_defined=False,
        dataset_evaluation_run=False, accuracy_causal_recovery_or_performance_pass=False,
        required_work_remaining_for_this_bounded_stage=[], next_step=next_step,
        limits='Softmax-free probability-input local cut still fails with OBEY. External inputs and fusion boundary differ; same final NONE output does not prove internal probability equivalence or unique full-model cause.',
        protected_identities=protected, stage_inventory_files_before_checkpoint=len(inventory), stage_manifest=identity(manifest_path),
        evidence=[identity(BASE / name) for name in ('root_native_audit.json', 'context_comparison.json',
            'independent_native_review.json', 'final_board_observation.json', 'native_evidence.zip', 'next_control_plan.json')])
    write_new(checkpoint_path, checkpoint)
    checkpoint_rel = checkpoint_path.relative_to(ROOT).as_posix()
    state.update(client_at=checkpoint['client_at'], current_stage_status=checkpoint['status'],
        current_stage_required_work_remaining=[], engineering_stage_required_work_remaining=[],
        gate05_status='INCOMPLETE', report=report, stage_completion_audit=checkpoint_rel,
        final_board_observation=(BASE / 'final_board_observation.json').relative_to(ROOT).as_posix())
    state['planning_boundary'] = [checkpoint['limits'],
        'Original full PyTorch/ORT guard FAILED_UNCHANGED; corrected full q_all still no engine, earlier corrected q_only_G1 success preserved.',
        'Only one saved corrected-CPU internal fixture tested; no dataset, causal recovery or performance pass.',
        'Next 6-node output-boundary control is a static proposal only; freeze its protocol separately before execution.']
    state['next_gate_work'] = [next_step, 'Resolve full reference comparability, floating/fusion controls and prior strict integer-tactic gaps before dataset criteria and evaluation.']
    state['connection_recovery']['current_native_state'] = checkpoint['status'] + ';ALL_OWNED_TASKS_EXITED'
    state['previous_dfl_softmax_cut_stage_completion_audit'] = 'artifacts/gate05/20261020_dfl_original_native/checkpoint.json'
    state['dfl_softmax_cut_probe'].update(status=checkpoint['status'], process_live=False,
        successful_builds=1, failed_builds=1, fixed_input_executions=2, monitor_seconds=69.31,
        none_build='SUCCESS_FINAL_RAW_BYTE_IDENTICAL_TO_OLD_CORRECTED_DFL_NONE',
        obey_build='FAILED_SAME_FILTER_QDQ_FORMATS_ASSERTION_NO_ENGINE', verified_receipt_files=89,
        full_model_failure_resolved=False, checkpoint=checkpoint_rel, checkpoint_sha256=sha(checkpoint_path),
        root_audit=(BASE / 'root_native_audit.json').relative_to(ROOT).as_posix(),
        context_comparison=(BASE / 'context_comparison.json').relative_to(ROOT).as_posix(),
        independent_native_review=(BASE / 'independent_native_review.json').relative_to(ROOT).as_posix(),
        next_control_plan=(BASE / 'next_control_plan.json').relative_to(ROOT).as_posix())
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=checkpoint['status'], gate05_status='INCOMPLETE', files=len(inventory),
        checkpoint_sha256=sha(checkpoint_path), manifest_sha256=sha(manifest_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Seal the completed DFL local compiler reproduction; never run inference."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05/20261020_dfl_context_native'
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
    board = read(BASE / 'final_board_observation.json')
    receipt = read(RAW / 'remote_receipt.json')
    correspondence = read(BASE / 'full_parser_constraint_correspondence.json')
    assert audit['successful_output_cases'] == 1 and audit['raw_tree_unchanged']
    assert review['check_count'] == len(review['checks']) == 309 and all(c['passed'] for c in review['checks'])
    assert review['saved_evidence_unchanged'] and review['saved_all_five_recorded_task_PIDs_exited']
    signature = review['failed_OBEY_signature_review']
    assert signature['same_optimizer_source_location_and_candidateRequirements_assertion']
    assert signature['failed_OBEY_engine_absent'] and signature['failed_OBEY_raw_output_absent']
    assert signature['same_case_source_model_and_corrected_clip_bounds']
    assert len(receipt['files']) == board['receipt_file_count'] == 88
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
    assert audit['cross_native_comparison'] == dict(status='UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED')
    none, obey = [audit['cases'][name] for name in ('code_clip_none', 'code_clip_obey')]
    for case in (none, obey):
        assert case['parser_constraints'] == dict(parsed_layers=14, precision_set_count=2, output_type_set_count=2, metadata_errors=[])
        assert case['child_cleanup']['direct_python_api_child_terminated']
    assert none['build']['status'] == 'DFL_CONTEXT_ENGINE_SAVED' and none['child_cleanup']['returncode'] == 0
    assert none['repeat']['byte_identical'] and none['repeat']['all_pairs_finite']
    assert none['cpu_difference']['all_pairs_finite'] and none['cpu_difference']['different_elements'] == 11779
    assert none['cpu_difference']['max_abs'] == 1.9073486328125e-06
    mapping = none['actual_conv_mapping']
    assert mapping['actual_count'] == 1 and mapping['one_unique_source_row']
    row = mapping['source_mappings'][0]
    assert row['unique'] and row['int8_input_weight'] and row['named_integer_tactic'] and not row['int8_output']
    assert row['actual_row']['Outputs'][0]['Format/Datatype'] == 'Row major linear FP32'
    assert row['actual_row']['TacticName'] == 'ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1'
    assert obey['build']['status'] == 'DFL_CONTEXT_BUILD_FAILED' and obey['child_cleanup']['returncode'] == 1
    assert not obey['engine_present'] and not obey['inference_run']
    assert correspondence['constraint_source_layer_count'] == 4 and not correspondence['new_inference']
    prepared = ROOT / 'artifacts/gate05/20261020_dfl_context_probe'
    cpu, fixture = read(prepared / 'cpu_diagnostic.json'), read(prepared / 'fixture_lineage.json')
    assert cpu['ancestor_cpu_executions'] == 1 and cpu['ancestor_conv_count'] == 72 and cpu['tail_cpu_executions'] == 2
    assert cpu['complete_detector_output_executions'] == 0 and cpu['both_outputs_byte_exact']
    assert fixture['ancestor_nodes'] == 638 and fixture['ancestor_initializers'] == 576
    protected = {
        'artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence.zip': '5eac1543e5151708585e57923dd82410c9b21aac81cb5cfd1137e057e35ed205',
        'artifacts/gate05/20261019_corrected_qall_native/build_failed_evidence.zip': '027433baa5b1e18fafb72cf64b70e58e1bd2aa05b295fd8f539ae8317e1c9a99',
        'artifacts/gate05/20261019_corrected_qall_native/parser_metadata_evidence.zip': 'a567b474be0e563448ce96b7b5cc0f1e9bfff1036461b9512efe9b8c1ba32d78',
        'artifacts/gate05/20261018_int8/input_v1/preflight.json': '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21',
        'artifacts/gate05/20261020_twoconv_context_native/checkpoint.json': '760cd93b257ee4473094b169b96e6c195cb4cce2d92d9fd28222a480c8fbf161',
        'artifacts/gate05/20261020_twoconv_context_native/native_evidence.zip': '6f9ed4fd0f3b21617eb48938cdf79910155a4d18bd29e49c55b168a88226594c',
        'artifacts/gate05/20261020_dfl_context_probe.zip': '05b9b50e4ddc294151121044facef5e1711ac368c793ca5232c1c1a557c42b9f',
        'artifacts/gate05/20261020_dfl_context_native/native_evidence.zip': 'a53750401ccdbb7c008c14e75cadec80a586a05f2baa32f5e985abdff527bd24',
    }
    source_before = read(prepared / 'source_hashes_before.json')
    assert source_before == read(prepared / 'source_hashes_after.json')
    for rel, expected in source_before.items():
        p = ROOT / rel
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], rel
    for rel, expected in protected.items():
        assert sha(ROOT / rel) == expected, rel
    for name in ('README.md', '研究状态与后续计划.md'):
        text = (ROOT / name).read_text(encoding='utf-8-sig')
        assert '33_Gate05_DFL内部上下文定位.md' in text and '2026-10-20' in text
    next_path = BASE / 'next_control_plan.json'
    assert sha(next_path) == '5b0469faa1f055b2e3e148c1302c65824a04f31b062abd48fbf109b70040aa7d'
    state_path = ROOT / 'artifacts/gate05/current_execution_state.json'
    state = read(state_path)
    before_path.write_bytes(state_path.read_bytes())
    directories = [p for p in (ROOT / 'artifacts/gate05').glob('20261020_dfl_context*') if p.is_dir()]
    paths = {p for d in directories for p in d.rglob('*') if p.is_file()}
    paths.update(p for p in (ROOT / 'artifacts/gate05').glob('20261020_dfl_context*') if p.is_file())
    paths.update(ROOT / 'scripts' / n for n in ('prepare_gate05_dfl_context_probe.py', 'jetson_dfl_context_probe.py', 'jetson_build_dfl_context_probe.py', 'audit_gate05_dfl_context_probe.py', 'complete_gate05_dfl_context_checkpoint.py'))
    paths.update(ROOT / n for n in ('33_Gate05_DFL内部上下文定位.md', 'README.md', '研究状态与后续计划.md'))
    inventory = [identity(p) for p in sorted(paths)]
    write_new(manifest_path, dict(schema='rampq.gate05.dfl_context_stage_inventory/v1', files=inventory,
                                  scope='Files before immutable checkpoint; excludes current mutable execution state, manifest itself and checkpoint.'))
    next_step = 'Separately freeze original FLOAT Clip DFL NONE/OBEY controls using the existing authenticated logits fixture and saved CPU outputs. Preserve parser constraints and record imported attributes; no full-model flag retry, parser reset, matrix or dataset run.'
    checkpoint = dict(schema='rampq.gate05.dfl_context_checkpoint/v1', client_at=datetime.now().astimezone().isoformat(),
                      status='DFL_CONTEXT_LOCAL_FAILURE_REPRODUCTION_CHECKPOINT_COMPLETE', gate05_status='INCOMPLETE',
                      original_complete_model_guard='FAILED_UNCHANGED', source_convs=1, source_nodes=9, source_initializers=7,
                      fixture_upstream_nodes=638, fixture_upstream_initializers=576, fixture_upstream_convs=72, fixture_upstream_CPU_runs=1,
                      tail_CPU_runs=2, tail_CPU_original_corrected_same_fixture_byte_exact=True, full_detector_output_CPU_runs=0,
                      native_build_attempts=2, native_build_attempts_each=1, successful_native_builds=1, failed_native_builds=1,
                      none_build_seconds=52.21165119999205, obey_build_seconds=1.5382017600059044,
                      none_fixed_input_executions=2, obey_fixed_input_executions=0, none_repeat_outputs_byte_exact=True,
                      none_conv_input_weight_INT8_and_named_integer_tactic=True, none_conv_output='FP32',
                      actual_INT8_output_required=False, none_cpu_differing_elements=11779, none_output_elements=33600,
                      none_cpu_max_abs=1.9073486328125e-06, none_obey_raw_pair='UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED',
                      obey_engine_absent=True, obey_inference_run=False,
                      obey_failure_signature='optimizer.cpp::filterQDQFormats::4421; !n->candidateRequirements.empty()',
                      same_optimizer_signature_as_failed_full_qall=True, named_full_model_failure_node_identified=False,
                      full_model_without_OBEY_build_proven=False, same_graph_fixture_and_parser_constraints=True,
                      only_actual_config_flag_changed='OBEY_PRECISION_CONSTRAINTS',
                      parser_constraints_each=dict(parsed_layers=14, precision_INT8=2, output_type_INT8=2, metadata_errors=0),
                      parser_constraints_not_reset=True, four_DFL_constraint_semantics_match_saved_full_metadata=True,
                      monitor_status='BUILD_SCRIPT_OK', monitor_status_means_both_builds_passed=False,
                      monitor_seconds=69.3, runner_seconds_not_benchmark=63.99363571200229,
                      verified_receipt_files=88, independent_native_review_checks=309, raw_evidence_unchanged=True,
                      all_five_owned_tasks_exited_same_boot=True, guard_intervened=False, boot_id=board['boot_id'],
                      evidence_transfer_first_attempt_reset=True, partial_transfer_preserved=True, evidence_download_retry_only=True,
                      full_model_native_retry_performed=False, full_model_failure_resolved=False, earlier_qonly_success_preserved=True,
                      prior_sources_archives_and_failed_preflight_unchanged=True, new_tolerances_defined=False,
                      dataset_evaluation_run=False, accuracy_causal_recovery_or_performance_pass=False,
                      required_work_remaining_for_this_bounded_stage=[], next_step=next_step,
                      limits='Corrected DFL cut locally reproduces full compiler signature only under OBEY in these observations. Does not identify the full failing node, establish a unique cause, or prove full-model repair.',
                      protected_identities=protected, stage_inventory_files_before_checkpoint=len(inventory), stage_manifest=identity(manifest_path),
                      evidence=[identity(BASE / n) for n in ('root_native_audit.json', 'independent_native_review.json', 'full_parser_constraint_correspondence.json', 'final_board_observation.json', 'native_evidence.zip', 'next_control_plan.json')])
    write_new(checkpoint_path, checkpoint)
    checkpoint_rel = checkpoint_path.relative_to(ROOT).as_posix()
    state.update(client_at=checkpoint['client_at'], current_stage_status=checkpoint['status'],
                 current_stage_required_work_remaining=[], engineering_stage_required_work_remaining=[],
                 stage_completion_audit=checkpoint_rel, final_board_observation=(BASE / 'final_board_observation.json').relative_to(ROOT).as_posix())
    state['planning_boundary'] = [checkpoint['limits'], 'NONE DFL input/weight INT8, output FP32, two repeated raw arrays identical; OBEY no engine and no inference.',
                                  'Original full PyTorch/ORT guard FAILED_UNCHANGED; corrected full q_all still has no engine. Earlier corrected q_only_G1 success preserved.',
                                  'Only corrected-CPU internal fixture tested; no native upstream distribution or dataset/causal recovery/performance pass.']
    state['next_gate_work'] = [next_step, 'Complete reference comparability and floating/fusion controls; retain original strict integer-tactic gaps and freeze dataset criteria before dataset evaluation.']
    state['connection_recovery']['current_native_state'] = checkpoint['status'] + ';ALL_OWNED_TASKS_EXITED'
    state['previous_dfl_stage_completion_audit'] = 'artifacts/gate05/20261020_twoconv_context_native/checkpoint.json'
    state['dfl_context_probe'].update(status=checkpoint['status'], process_live=False, successful_builds=1, failed_builds=1,
                                      fixed_input_executions=2, monitor_seconds=69.3, none_build='SUCCESS',
                                      obey_build='FAILED_SAME_FILTER_QDQ_FORMATS_ASSERTION_NO_ENGINE', verified_receipt_files=88,
                                      full_model_failure_resolved=False, checkpoint=checkpoint_rel, checkpoint_sha256=sha(checkpoint_path),
                                      root_audit='artifacts/gate05/20261020_dfl_context_native/root_native_audit.json',
                                      independent_native_review='artifacts/gate05/20261020_dfl_context_native/independent_native_review.json',
                                      next_control_plan=next_path.relative_to(ROOT).as_posix())
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=checkpoint['status'], gate05_status='INCOMPLETE', files=len(inventory),
                          checkpoint_sha256=sha(checkpoint_path), manifest_sha256=sha(manifest_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()

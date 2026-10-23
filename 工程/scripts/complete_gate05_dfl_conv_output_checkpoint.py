"""Seal saved six-node DFL outcomes; stdlib only, no inference or SSH.

This file deliberately does not predict either case's build outcome, native
numeric deltas, receipt count, or independent post-review check count.
"""
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_native'
PREPARED = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_probe'
RAW = BASE / 'native_evidence'
CASES = ('code_clip_none', 'code_clip_obey')
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
STATUS = 'DFL_CONV_OUTPUT_LOCAL_BOUNDARY_CHECKPOINT_COMPLETE'
INPUT_SHAPE = (1, 16, 4, 8400)
OUTPUT_SHAPE = (1, 1, 4, 8400)
SOURCE_CONV = '/model.23/dfl/conv/Conv'
PINS = {
    'artifacts/gate05/20261020_dfl_conv_output_probe.zip': '78008952bd6355d4a197913c669c3a7b18ff1e80b119d88789d19f53baed4ba7',
    'artifacts/gate05/20261020_dfl_conv_output_probe/spec.json': '9e9d31b55e4358d39a08a7869aab77582f094f1d6152599c727196e2b77c3897',
    'artifacts/gate05/20261020_dfl_conv_output_probe/manifest.json': '104137a621b3ea5c65a019555a1d92fc8c876fe0bf9293b5ab7cfe2e6fe71f77',
    'artifacts/gate05/20261020_dfl_conv_output_protocol/protocol.json': '8f6a55f640915b84d7300d1595ebb895092ccf0374e40a27b6aaf812f406e09b',
    'artifacts/gate05/20261020_dfl_conv_output_review/pre_native_review.json': 'a3fa1038648c560045bd79d77a5dac027d16ceb1c04bdd1ce7e486d578f97747',
    'artifacts/gate05/20261020_dfl_softmax_cut_native/checkpoint.json': '142eb219ccbeea141237cd1a89ecdb2c47c70b99d35159d9912ba8cea767c53f',
    'artifacts/gate05/20261020_dfl_softmax_cut_native/native_evidence.zip': '1b531ae8079a7563432e9702071ca8d384695f2a2624d80ab3da7798fa7215cc',
    'artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence.zip': '5eac1543e5151708585e57923dd82410c9b21aac81cb5cfd1137e057e35ed205',
    'artifacts/gate05/20261019_corrected_qall_native/build_failed_evidence.zip': '027433baa5b1e18fafb72cf64b70e58e1bd2aa05b295fd8f539ae8317e1c9a99',
    'artifacts/gate05/20261019_corrected_qall_native/parser_metadata_evidence.zip': 'a567b474be0e563448ce96b7b5cc0f1e9bfff1036461b9512efe9b8c1ba32d78',
    'artifacts/gate05/20261018_int8/input_v1/preflight.json': '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21',
    'scripts/jetson_dfl_conv_output_probe.py': 'c53b90d1b730599a3b222098286da2592cd9ecb9c0ad2453eec54c0d5dfa99e7',
    'scripts/jetson_build_dfl_conv_output_probe.py': '9f34121acf19c13a8d8501a4da62f455dddd7e8d34d2ac44d9c3779e5adc764d',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def inside(folder, relative):
    path = (folder / relative).resolve()
    assert folder.resolve() in path.parents, relative
    return path


def verify_receipt(folder):
    receipt = read(folder / 'remote_receipt.json')
    expected = receipt['files']
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    assert actual == set(expected) | {'remote_receipt.json'}
    for relative, item in expected.items():
        path = inside(folder, relative)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], relative
    return receipt


def npy_bytes(value, shape):
    """Read authenticated NPY headers/payload, without NumPy or computation."""
    handle = io.BytesIO(value)
    assert handle.read(6) == b'\x93NUMPY'
    version = tuple(handle.read(2))
    assert version in ((1, 0), (2, 0), (3, 0))
    header_size = struct.unpack('<H' if version == (1, 0) else '<I', handle.read(2 if version == (1, 0) else 4))[0]
    header = ast.literal_eval(handle.read(header_size).decode('utf-8' if version == (3, 0) else 'latin1').strip())
    assert header['descr'] == '<f4' and header['fortran_order'] is False and tuple(header['shape']) == tuple(shape)
    elements = math.prod(shape)
    payload = handle.read()
    assert len(payload) == 4 * elements
    assert all(math.isfinite(number[0]) for number in struct.iter_unpack('<f', payload))
    return payload


def npz_arrays(path, shapes):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {name + '.npy' for name in shapes}
        return {name: npy_bytes(archive.read(name + '.npy'), shape) for name, shape in shapes.items()}


def constraint_values(rows):
    return [{key: row[key] for key in ('index', 'name', 'precision_is_set', 'precision', 'precision_value_error', 'output_types') if key in row}
            for row in rows]


def passing_review(review, expected_count=None):
    checks = review['checks']
    assert isinstance(checks, list) and checks and review['check_count'] == len(checks)
    assert all(item['passed'] is True for item in checks)
    if expected_count is not None:
        assert len(checks) == expected_count
    assert not review.get('launch_blockers')
    for field in ('new_CPU_OR_native_inference_by_reviewer', 'new_CPU_or_native_inference_run_by_review',
                  'new_CPU_or_native_inference_by_reviewer', 'SSH_by_reviewer', 'new_inference'):
        if field in review:
            assert review[field] is False, field
    for field in ('old_new_saved_raw_unchanged', 'saved_evidence_unchanged', 'raw_tree_unchanged'):
        if field in review:
            assert review[field] is True, field
    return len(checks)


def inspect_case(folder, name, spec, summary, audited):
    """Require a normal terminal observation, without predicting its result."""
    build = read(folder / 'build_result.json')
    cleanup = read(folder / 'child_cleanup.json')
    request = read(folder / 'parsed_network_request.json')
    assert cleanup['direct_python_api_child_terminated'] is True and cleanup['returncode'] is not None
    assert not cleanup['signals_sent']
    assert request['case'] == next(case for case in spec['cases'] if case['id'] == name)
    assert request['spec_sha256'] == sha(PREPARED / 'spec.json')
    assert request['model_sha256'] == spec['variants']['code_clip']['model_sha256']
    flags = request['flags']['get_flag']
    for flag, enabled in (('INT8', True), ('FP16', True), ('TF32', False), ('PREFER_PRECISION_CONSTRAINTS', False),
                          ('OBEY_PRECISION_CONSTRAINTS', name.endswith('_obey'))):
        assert flags[flag] == dict(supported=True, enabled=enabled)
    assert request['workspace_bytes_actual'] == 1024 ** 3 and request['avg_timing_iterations_actual'] == 1
    assert request['profiling_verbosity_actual'] == 'ProfilingVerbosity.DETAILED'
    assert request['no_layer_precision_or_output_type_setters_or_resets'] is True
    clips = request['clip_attributes']
    assert len(clips) == 1 and clips[0]['name'] == spec['clip_names'][0]
    assert clips[0]['explicit_importer_correction'] is True and clips[0]['after'] == dict(alpha=-127.0, beta=127.0)
    before, after = read(folder / 'parsed_constraints_before.json'), read(folder / 'parsed_constraints_after.json')
    assert constraint_values(before) == constraint_values(after)
    assert audited['build'] == build and audited['child_cleanup'] == cleanup
    assert audited['parser_constraints']['metadata_errors'] == []
    output = dict(case=name, build_status=build['status'], builder_returncode=cleanup['returncode'],
                  build_seconds_not_benchmark=build['build_seconds_not_benchmark'],
                  actual_flags=flags, parser_constraints=audited['parser_constraints'],
                  all_parser_constraints_retained=True, Clip_bounds_after=[-127, 127])
    terminal = summary['cases'][name]['status']
    assert audited['status'] == terminal
    if build['status'] == 'DFL_CONV_OUTPUT_ENGINE_SAVED':
        assert terminal == 'DFL_CONV_OUTPUT_FIXED_INPUT_OUTPUTS_SAVED' and cleanup['returncode'] == 0
        assert (folder / 'model.engine').is_file() and (folder / 'raw_outputs.npz').is_file()
        result = read(folder / 'result.json')
        assert result['status'] == terminal and result['boot_id'] == BOOT
        assert sha(folder / 'model.engine') == build['engine_sha256'] == result['engine_sha256']
        assert sha(folder / 'raw_outputs.npz') == result['raw_outputs_sha256']
        arrays = npz_arrays(folder / 'raw_outputs.npz', dict(first=OUTPUT_SHAPE, repeat=OUTPUT_SHAPE))
        assert arrays['first'] == arrays['repeat']
        assert result['repeat_comparison']['byte_identical'] and result['repeat_comparison']['all_pairs_finite']
        assert audited['repeat'] == result['repeat_comparison'] or (
            audited['repeat']['byte_identical'] and audited['repeat']['different_elements'] == 0)
        assert result['stats']['shape'] == list(OUTPUT_SHAPE) and result['stats']['dtype'] == 'float32' and result['stats']['all_finite']
        bindings = read(folder / 'runtime_bindings.json')
        desired = {row['name']: row for row in [spec['input_binding']] + spec['output_bindings_in_order']}
        assert len(bindings) == 2 and {row['name'] for row in bindings} == set(desired)
        for binding in bindings:
            target = desired[binding['name']]
            assert binding['shape'] == target['shape'] and binding['input'] == target['input']
            assert binding['dtype'] == 'DataType.FLOAT' and binding['format'] == 'TensorFormat.LINEAR'
        inspector = read(folder / 'engine_layers.json')
        assert inspector == read(folder / 'runner_engine_layers.json')
        convs = [row for row in inspector['Layers'] if 'Convolution' in row.get('LayerType', '')]
        assert convs == read(folder / 'conv_rows.json')
        matched = [row for row in convs if SOURCE_CONV in row.get('Name', '')]
        assert len(convs) == len(matched) == 1
        mapping = audited['actual_conv_mapping']
        assert mapping['actual_count'] == 1 and mapping['one_unique_source_row']
        assert mapping['source_mappings'][0]['actual_row'] == matched[0]
        actual = matched[0]
        assert actual.get('Inputs') and actual.get('Outputs') and actual.get('Weights')
        # Record adopted formats and tactic as observed. Do not require INT8
        # output, a specific tactic, fixed CPU deltas, or a guessed success.
        output.update(successful_build=True, fixed_input_executions=2, engine=identity(folder / 'model.engine'),
            repeat_byte_identical=True, output_shape=list(OUTPUT_SHAPE), raw_output=identity(folder / 'raw_outputs.npz'),
            actual_convolution=actual, actual_conv_input_formats=[row.get('Format/Datatype') for row in actual['Inputs']],
            actual_weight_type=actual['Weights'].get('Type'),
            actual_conv_output_formats=[row.get('Format/Datatype') for row in actual['Outputs']],
            actual_tactic_name=actual.get('TacticName'), actual_INT8_output_required=False,
            CPU_difference_descriptive=audited['cpu_difference'])
    else:
        assert build['status'] == 'DFL_CONV_OUTPUT_BUILD_FAILED' and terminal == 'BUILD_FAILED_OBSERVATION'
        assert cleanup['returncode'] > 0 and build['inference_run'] is False
        assert not any((folder / item).exists() for item in ('model.engine', 'raw_outputs.npz', 'result.json'))
        assert audited['engine_present'] is False and audited['inference_run'] is False
        failure = read(folder / 'build_failure.json')
        assert failure == build
        log = (folder / 'builder.log').read_text(encoding='utf-8', errors='replace')
        output.update(successful_build=False, fixed_input_executions=0, engine_present=False, inference_run=False,
            failure=failure, compiler_error_lines=[line for line in log.splitlines() if '[E]' in line])
    return output, request, constraint_values(before)


def main():
    checkpoint_path = BASE / 'checkpoint.json'
    manifest_path = BASE / 'stage_manifest.json'
    before_path = BASE / 'state_before_completion.json'
    assert BASE.is_dir() and not any(path.exists() for path in (checkpoint_path, manifest_path, before_path))
    for relative, expected in PINS.items():
        assert sha(ROOT / relative) == expected, relative
    for prefix, count in (('source_hashes', 72), ('seal_native_source_hashes', 2)):
        sources = read(PREPARED / (prefix + '_before.json'))
        assert len(sources) == count and sources == read(PREPARED / (prefix + '_after.json'))
        for relative, expected in sources.items():
            path = inside(ROOT, relative)
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], relative
    spec = read(PREPARED / 'spec.json')
    assert spec['variants']['code_clip']['nodes'] == 6 and spec['variants']['code_clip']['initializers'] == 7
    assert tuple(spec['input_binding']['shape']) == INPUT_SHAPE
    assert tuple(spec['output_bindings_in_order'][0]['shape']) == OUTPUT_SHAPE
    assert spec['output_bindings_in_order'][0]['name'] == '/model.23/dfl/conv/Conv_output_0'
    pre_count = passing_review(read(ROOT / 'artifacts/gate05/20261020_dfl_conv_output_review/pre_native_review.json'), 369)
    cpu, graph = read(PREPARED / 'cpu_diagnostic.json'), read(PREPARED / 'graph_identity_audit.json')
    assert cpu['new_CPU_executions'] == 1 and cpu['native_executions'] == 0
    assert cpu['new_probability_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0
    assert cpu['C_order_array_bytes_identical_saved_reference'] and cpu['differing_elements'] == 0
    assert cpu['new_output_shape'] == list(OUTPUT_SHAPE) and cpu['old_saved_reference_shape'] == [1, 4, 8400]
    assert graph['all_selected_nodes_initializers_and_io_declarations_exact_source_bytes']
    assert graph['nodes'] == 6 and graph['initializers'] == 7 and graph['conv_count'] == 1
    old_payload = npy_bytes((PREPARED / 'saved_cpu_reference_3d.npy').read_bytes(), (1, 4, 8400))
    new_payload = npy_bytes((PREPARED / 'cpu_actual_code_clip.npy').read_bytes(), OUTPUT_SHAPE)
    expected = npz_arrays(PREPARED / spec['cpu_expected_outputs_file'], dict(code_clip_output=OUTPUT_SHAPE))
    assert old_payload == new_payload == expected['code_clip_output']
    receipt = verify_receipt(RAW)
    assert sha(RAW / 'spec.json') == sha(PREPARED / 'spec.json')
    assert sha(RAW / 'manifest.json') == sha(PREPARED / 'manifest.json')
    audit = read(BASE / 'root_native_audit.json')
    review = read(BASE / 'independent_native_review.json')
    post_count = passing_review(review)
    cross = read(BASE / 'context_comparison.json')
    assert cross['schema'] == 'rampq.gate05.dfl_saved_output_boundary_comparison/v1'
    assert cross['raw_trees_unchanged'] and not cross['new_inference']
    assert cross['same_native_probability_input_file'] and cross['CPU_reference_comparison']['byte_identical']
    assert cross['common_named_Q_DQ_requests_equal'] and cross['common_config_flags_equal_except_OBEY']
    assert cross['original_complete_model_guard'] == 'FAILED_UNCHANGED' and cross['gate05_status'] == 'INCOMPLETE'
    board = read(BASE / 'final_board_observation.json')
    assert len(receipt['files']) == board['receipt_file_count'] == audit['receipt_files']
    assert board['all_remote_receipt_files_still_match'] and all(board['remote_checks'].values())
    assert set(board['remote_checks']) == set(receipt['files'])
    assert sha(RAW / 'remote_receipt.json') == board['receipt_sha256']
    assert sha(BASE / 'native_evidence.zip') == board['archive_sha256']
    assert board['boot_id'] == BOOT and len(board['known_processes']) >= 3
    assert all(not item['exists'] for item in board['known_processes'].values())
    assert not board['task_processes'] and not board['corrected_qall_engine_paths']
    assert board['qonly_preserved_engine']['sha256'] == '486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476'
    assert board['fan']['stdout'].strip() == 'active' and 'NRestarts=0' in board['jtop']['stdout']
    assert 'ActiveState=active' in board['jtop']['stdout']
    monitor, guard = read(RAW / 'monitor/status.json'), read(RAW / 'supplemental_group_guard.json')
    assert monitor['boot_id'] == guard['boot_id'] == BOOT
    assert monitor['status'] == audit['monitor_status'] == 'BUILD_SCRIPT_OK'
    assert guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] and not guard['final_group_members']
    assert not guard['intervention_performed'] and audit['guard_group_empty'] and not audit['guard_intervention']
    assert audit['raw_tree_unchanged'] and audit['original_complete_model_guard'] == 'FAILED_UNCHANGED'
    runs = list((RAW / 'results').glob('probe_*'))
    assert len(runs) == 1
    run = runs[0]
    summary = read(run / 'summary.json')
    assert summary['status'] == 'DFL_CONV_OUTPUT_BATCH_COMPLETED' and summary['aborted'] is None
    assert tuple(summary['requested_case_order']) == CASES and set(summary['cases']) == set(CASES)
    assert summary['observed_case_count'] == 2 and summary['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED'
    cases, requests, constraints = {}, {}, {}
    for name in CASES:
        cases[name], requests[name], constraints[name] = inspect_case(run / name, name, spec, summary, audit['cases'][name])
    changed_flags = [key for key in requests[CASES[0]]['flags']['get_flag']
                     if requests[CASES[0]]['flags']['get_flag'][key] != requests[CASES[1]]['flags']['get_flag'][key]]
    assert changed_flags == ['OBEY_PRECISION_CONSTRAINTS']
    assert constraints[CASES[0]] == constraints[CASES[1]]
    successes = sum(case['successful_build'] for case in cases.values())
    assert audit['successful_output_cases'] == summary['successful_output_case_count'] == successes
    for name, case in cases.items():
        cross_case = cross['cases']['conv_output_' + name.removeprefix('code_clip_')]
        assert cross_case['build'] == audit['cases'][name]['build']
        assert cross_case['engine_present'] == case['successful_build']
    report_candidates = list(ROOT.glob('36_Gate05*.md'))
    assert len(report_candidates) == 1
    report = report_candidates[0]
    assert 'INCOMPLETE' in report.read_text(encoding='utf-8-sig')
    for name in ('README.md', '研究状态与后续计划.md'):
        assert report.name in (ROOT / name).read_text(encoding='utf-8-sig')
    # All validation precedes any new checkpoint, manifest, or mutable state.
    state_path = ROOT / 'artifacts/gate05/current_execution_state.json'
    state = read(state_path)
    assert 'dfl_conv_output_probe' in state and state.get('gate05_status') == 'INCOMPLETE'
    write_new(before_path, state)
    directories = [path for path in (ROOT / 'artifacts/gate05').glob('20261020_dfl_conv_output*') if path.is_dir()]
    paths = {path for directory in directories for path in directory.rglob('*') if path.is_file()}
    paths.update(path for path in (ROOT / 'artifacts/gate05').glob('20261020_dfl_conv_output*') if path.is_file())
    paths.update(ROOT / 'scripts' / name for name in ('prepare_gate05_dfl_conv_output_probe.py',
        'jetson_dfl_conv_output_probe.py', 'jetson_build_dfl_conv_output_probe.py',
        'audit_gate05_dfl_conv_output_probe.py', 'compare_gate05_dfl_output_boundary.py',
        'complete_gate05_dfl_conv_output_checkpoint.py'))
    paths.update((report, ROOT / 'README.md', ROOT / '研究状态与后续计划.md'))
    paths.difference_update((checkpoint_path, manifest_path, state_path))
    inventory = [identity(path) for path in sorted(paths)]
    write_new(manifest_path, dict(schema='rampq.gate05.dfl_conv_output_stage_inventory/v1', files=inventory,
        scope='Immutable pre-checkpoint inventory; excludes mutable current_execution_state.json, stage_manifest.json and checkpoint.json.'))
    next_plan = BASE / 'next_control_plan.json'
    next_identity = identity(next_plan) if next_plan.is_file() else None
    next_text = ('Further controls require a separately frozen protocol based on these recorded outcomes. '
                 'No automatic retry, parser reset, full-model replay, dataset evaluation, tolerance, weight or scale change follows from sealing.')
    checkpoint = dict(schema='rampq.gate05.dfl_conv_output_checkpoint/v1',
        client_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), status=STATUS, gate05_status='INCOMPLETE',
        original_complete_model_guard='FAILED_UNCHANGED', source_nodes=6, source_initializers=7, source_convs=1,
        new_CPU_executions=1, new_Softmax_72Conv_old_DFL_or_full_detector_CPU_runs=0,
        CPU_4D_and_saved_3D_C_order_array_bytes_equal=True, CPU_shape_correspondence=dict(new=list(OUTPUT_SHAPE), old=[1, 4, 8400]),
        input_probability_fixture_is_validated_native_upstream_output=False,
        native_build_attempts=2, native_build_attempts_each=1, successful_native_builds=successes, failed_native_builds=2 - successes,
        fixed_input_executions=sum(case['fixed_input_executions'] for case in cases.values()),
        cases=cases, pre_native_review_checks=pre_count, independent_native_review_checks=post_count,
        verified_receipt_files=len(receipt['files']), same_native_probability_fixture_as_parent=True,
        output_is_4D_DFL_Conv_distance_not_decoded_boxes=True, actual_INT8_output_required=False,
        output_boundary_changes_native_format_selection_context=True,
        cross_stage_raw_output_comparisons=cross['raw_output_comparisons'],
        cross_stage_comparison_does_not_prove_unique_removed_failing_node=True,
        monitor_status=monitor['status'], monitor_status_means_both_builds_passed=False,
        monitor_seconds=monitor['elapsed_seconds'], runner_seconds_not_benchmark=summary['elapsed_seconds_not_benchmark'],
        all_recorded_tasks_exited_same_boot=True, recorded_exit_PID_count=len(board['known_processes']),
        boot_id=BOOT, guard_intervened=False, original_sources_72_and_sealed_tools_2_unchanged=True,
        raw_evidence_unchanged=True, earlier_qonly_success_preserved=True, full_model_failure_resolved=False,
        full_model_native_retry_performed=False, named_full_model_failing_node_or_unique_cause_identified=False,
        new_tolerances_defined=False, dataset_evaluation_run=False, accuracy_causal_recovery_or_performance_pass=False,
        final_board_observation_client_at=board['client_at'], final_board_observation_board_at=board['board_time'],
        board_clock_corrected=False, power_query_privileged_full_verification=False,
        required_work_remaining_for_this_bounded_stage=[], next_step=next_text, optional_next_control_plan=next_identity,
        stage_inventory_files_before_checkpoint=len(inventory), stage_manifest=identity(manifest_path),
        protected_identities=PINS, evidence=[identity(BASE / name) for name in ('root_native_audit.json',
            'context_comparison.json', 'independent_native_review.json', 'final_board_observation.json', 'native_evidence.zip')])
    write_new(checkpoint_path, checkpoint)
    checkpoint_rel = checkpoint_path.relative_to(ROOT).as_posix()
    state.update(client_at=checkpoint['client_at'], current_stage_status=STATUS,
        current_stage_required_work_remaining=[], engineering_stage_required_work_remaining=[], gate05_status='INCOMPLETE',
        report=report.name, stage_completion_audit=checkpoint_rel,
        final_board_observation=(BASE / 'final_board_observation.json').relative_to(ROOT).as_posix())
    state['planning_boundary'] = [
        'Six-node output-boundary outcomes are recorded separately for NONE and OBEY; completion does not mean both builds passed.',
        'The external output boundary differs from the parent eight-node cut; no unique full-model failing node is established.',
        'Original full PyTorch/ORT guard FAILED_UNCHANGED; full corrected q_all remains unresolved and earlier q_only_G1 success is preserved.',
        'Any next control is a proposal until separately frozen; no parser reset, full-model retry, dataset or performance result.']
    state['next_gate_work'] = [next_text, 'Resolve full reference comparability, floating/fusion controls and prior strict integer-tactic gaps before dataset criteria and evaluation.']
    state['previous_dfl_conv_output_stage_completion_audit'] = 'artifacts/gate05/20261020_dfl_softmax_cut_native/checkpoint.json'
    state['connection_recovery']['current_native_state'] = STATUS + ';ALL_OWNED_TASKS_EXITED'
    state['dfl_conv_output_probe'].update(status=STATUS, process_live=False, native_build_attempts=2,
        successful_builds=successes, failed_builds=2 - successes, fixed_input_executions=checkpoint['fixed_input_executions'],
        case_outcomes={name: case['build_status'] for name, case in cases.items()}, monitor_seconds=monitor['elapsed_seconds'],
        verified_receipt_files=len(receipt['files']), full_model_failure_resolved=False,
        checkpoint=checkpoint_rel, checkpoint_sha256=sha(checkpoint_path),
        root_audit=(BASE / 'root_native_audit.json').relative_to(ROOT).as_posix(),
        context_comparison=(BASE / 'context_comparison.json').relative_to(ROOT).as_posix(),
        independent_native_review=(BASE / 'independent_native_review.json').relative_to(ROOT).as_posix())
    if next_identity is not None:
        state['dfl_conv_output_probe']['next_control_plan'] = next_identity['path']
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=STATUS, gate05_status='INCOMPLETE', successful_builds=successes, failed_builds=2 - successes,
        files=len(inventory), checkpoint_sha256=sha(checkpoint_path), manifest_sha256=sha(manifest_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()

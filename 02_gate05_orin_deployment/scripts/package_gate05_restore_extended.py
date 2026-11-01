"""Package one 7200-second monitored attempt of each unchanged prefer restore graph."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ORDER = ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2')
INPUTS_NAME = 'gate05_restore_extended_inputs_20261019'
QUEUE_NAME = 'gate05_restore_extended_queue_20261019'
SCRIPT_NAME = 'jetson_gate05_extended_restore_queue.py'
BASE = '/home/nvidia/rampq/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def previous_timeout(configuration):
    evidence = ROOT / 'artifacts/gate05/20261019_precision_native/prefer' / configuration / 'build_evidence'
    receipt_path = evidence / 'remote_receipt.json'
    receipt = read(receipt_path)
    assert receipt['schema'] == 'rampq.remote_file_receipt/v1'
    for filename, expected in receipt['files'].items():
        path = evidence / filename
        assert path.resolve().is_relative_to(evidence.resolve()) and not path.is_symlink(), filename
        assert path.is_file() and info(path) == expected, filename
    status_path = evidence / 'monitor/status.json'
    status = read(status_path)
    assert status['status'] == 'STOPPED_BY_MONITOR' and status['reason'] == 'Build supervision time limit reached'
    assert status['max_seconds'] == 1800 and status['elapsed_seconds'] >= 1800
    kernel = (evidence / 'monitor/kernel.log').read_text(errors='replace')
    markers = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting',
               'Out of memory:', 'oom-kill:', 'Kernel panic')
    assert not any(marker in kernel for marker in markers)
    samples = [json.loads(line) for line in (evidence / 'monitor/resources.jsonl').read_text().splitlines() if line]
    assert samples
    temperatures = [value for sample in samples for value in sample['temperatures_c'].values()]
    assert temperatures and max(temperatures) < 85
    assert not (evidence / 'build/model.engine').exists()
    return {'configuration': configuration,
            'evidence': str(evidence.relative_to(ROOT)).replace('\\', '/'),
            'receipt_sha256': sha(receipt_path), 'verified_original_file_count': len(receipt['files']),
            'monitor_status_sha256': sha(status_path), 'monitor_status': status['status'],
            'monitor_stop_reason': status['reason'], 'monitor_elapsed_seconds': status['elapsed_seconds'],
            'boot_id_of_prior_attempt': status['boot_id'],
            'max_temperature_c': max(temperatures),
            'min_available_mib': min(sample['MemAvailable_kib'] for sample in samples) / 1024,
            'no_kernel_safety_events': True, 'engine_present': False,
            'limits': 'Previous-attempt evidence only; not a current board status or a numerical/precision pass'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gate05/20261019_restore_extended_input')
    args = parser.parse_args()
    output = args.output.resolve()
    archive_path = output.with_suffix('.zip')
    assert not output.exists() and not archive_path.exists(), 'Use a fresh output and preserve prior attempts'
    source_base = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles_prefer'
    script = ROOT / 'scripts' / SCRIPT_NAME
    ast.parse(script.read_text(encoding='utf-8'), feature_version=(3, 8))
    priors = [previous_timeout(name) for name in ORDER]
    verified_sources = []
    for name in ORDER:
        source = source_base / name
        assert source.is_dir() and not source.is_symlink()
        manifest = read(source / 'manifest.json')
        assert manifest['schema'] == 'rampq.gate05.precision_runtime_bundle/v1' and manifest['configuration'] == name
        names = {path.name for path in source.iterdir() if path.is_file()}
        assert names == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'}
        assert all(path.is_file() and not path.is_symlink() for path in source.iterdir())
        for filename, expected in manifest['files'].items():
            assert Path(filename).name == filename and info(source / filename) == expected, filename
        spec = read(source / 'spec.json')
        assert spec['precisionConstraints'] == 'prefer' and spec['max_build_seconds'] == 1800
        assert spec['avg_timing'] == 1 and spec['workspace_mib'] == 1024
        assert spec['actual_precision_audit_criteria_changed'] is False
        for filename in ('jetson_build_precision_candidate.py', 'jetson_monitored_build.py'):
            ast.parse((source / filename).read_text(encoding='utf-8'), feature_version=(3, 8))
        verified_sources.append({'configuration': name, 'source': source,
                                 'files': {path.name: info(path) for path in sorted(source.iterdir())}})
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / INPUTS_NAME
    inputs.mkdir()
    shutil.copy2(script, inputs / SCRIPT_NAME)
    rows = []
    for original in verified_sources:
        name, source = original['configuration'], original['source']
        directory_name = 'gate05_precision_' + name + '_20261019_v3_extended_7200'
        destination = output / directory_name
        shutil.copytree(source, destination)
        assert all((destination / filename).read_bytes() == (source / filename).read_bytes()
                   for filename in original['files'])
        rows.append({'configuration': name, 'bundle_root': BASE + directory_name,
                     'original_client_bundle': str(source.relative_to(ROOT)).replace('\\', '/'),
                     'bundle_manifest_sha256': sha(source / 'manifest.json'),
                     'all_original_files': original['files'], 'precisionConstraints': 'prefer',
                     'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
                     'maximum_attempts': 1})
    revision = {'schema': 'rampq.gate05.extended_restore_budget_revision/v1',
                'revision': 'restore_prefer_external_7200_20261019',
                'prepared_client_at': datetime.now().astimezone().isoformat(),
                'status': 'STAGED_NOT_BUILT_OR_EXECUTED', 'configuration_order': list(ORDER),
                'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
                'maximum_attempts_per_configuration': 1, 'maximum_build_budget_sum_seconds': 21600,
                'max_monitor_and_cleanup_seconds_per_configuration': 7380,
                'original_inputs_byte_identical': True,
                'original_manifest_spec_builder_monitor_and_SHA256SUMS_unchanged': True,
                'scientific_criteria_changed': False, 'actual_precision_audit_criteria_changed': False,
                'new_calibration_or_graph_or_tactic_parameters_added': False,
                'original_numerical_guard': 'FAILED_UNCHANGED', 'dataset_evaluation_run': False,
                'time_limit_owner': 'Original jetson_monitored_build.py invoked explicitly with --max-seconds 7200',
                'budget_metadata_alone_enforces_nothing': True,
                'inherited_spec_1800_is_not_the_effective_monitor_limit': True,
                'precision_request_is_not_actual_adoption': True,
                'reason': 'The three unchanged prefer restore searches ended solely at their original 1800-second '
                          'supervision limit, without a complete engine or kernel/thermal safety event. One longer '
                          'bounded search is prepared for each, without changing model or acceptance criteria.',
                'supporting_recent_observation': 'The parent task reports that the unchanged full q_all search built '
                                                 'in 3279.35 seconds; that result is separate evidence, not a claim '
                                                 'that these mixed searches will build or adopt requested precision.',
                'prior_timeouts': priors,
                'required_pre_start': ['Use current live boot UUID, not a historical UUID',
                                       'Previous GPU work and descendants fully ended',
                                       'Fan active, >=3 GiB MemAvailable, all temperatures <85 C',
                                       'No other trtexec/build/monitor/queue/smoke and shared queue lock acquired'],
                'required_between_items': ['Terminal monitor identity and exact effective budget verified',
                                           'Only ordinary build failure or pure timeout may advance',
                                           'All recorded owned PID/start_ticks/private-session descendants ended',
                                           'Global trtexec empty and readiness checks repeated'],
                'whole_queue_stop': ['Boot ID change', 'Fan failure', 'Temperature >=85 C',
                                     'MemAvailable <384 MiB for three consecutive two-second samples',
                                     'New USB/OOM/panic kernel marker', 'Collector failure or monitor loss',
                                     'Other GPU work or descendants that cannot be confirmed ended'],
                'limits': 'Engineering build/deserialization only. Successful engines still need unchanged native '
                          'precision/tactic audits and separate fixed-image execution. No automatic retry, dataset '
                          'evaluation, target rescue, causal comparison, or performance claim.'}
    write(inputs / 'budget_revision.json', revision)
    plan = {'schema': 'rampq.gate05.extended_restore_queue_plan/v1',
            'configuration_order': list(ORDER), 'maximum_attempts_per_configuration': 1,
            'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
            'max_monitor_and_cleanup_seconds': 7380,
            'expected_boot_id_from_required_launch_argument': True,
            'scientific_criteria_changed': False, 'dataset_evaluation_run': False,
            'inputs_root': BASE + INPUTS_NAME, 'queue_run': BASE + QUEUE_NAME,
            'queue_script_sha256': sha(inputs / SCRIPT_NAME),
            'budget_revision_sha256': sha(inputs / 'budget_revision.json'), 'configurations': rows}
    write(inputs / 'plan.json', plan)
    readme = "# Three prefer recovery configurations: one extended-budget build each\n\nEngineering inputs only; the local scripts do not connect to the board, build engines, or execute image inference.\n\nOrdered as `r_only_G1 -> r_only_G1_G5 -> r_only_G1_G2`.\nEach item once, at most 7200 seconds, total build budget up to 6 hours, plus bounded cleanup time.\n\nAll original files in the three bundles are copied byte for byte, including `manifest.json`, `spec.json`, the model,\nfrozen QDQ, precision requests, builder, monitor, and `SHA256SUMS`. The original spec value\n`max_build_seconds=1800` remains unchanged; it is not the effective monitoring limit for this run.\nThe effective limit is explicitly set by the external queue passing `--max-seconds 7200` to the original monitor.\nExternal `budget_revision.json` and per-launch/request record the budget discrepancy.\nDo not modify the original spec to mask differences, nor run the builder directly to bypass monitoring.\n\nExtract all four top-level ZIP directories under `/home/nvidia/rampq/`; the three new bundle root names end in\n`_v3_extended_7200`, and both input and queue-output root directories are new.\nCheck first if the directory already exists; do not overwrite or delete previous attempts.\n\nAfter all existing GPU builds and single-image executions have ended, obtain the board's current boot UUID:\n\n```bash\ncat /proc/sys/kernel/random/boot_id\n```\n\nFirst check only the frozen files (this does not start GPU work):\n\n```bash\npython3 " + BASE + INPUTS_NAME + '/' + SCRIPT_NAME + ' --check-only\n```\n\nLaunch using the newly obtained UUID; do not copy the historical UUID. Validate the value at startup, before each item, and during runtime.\n\n```bash\npython3 -u ' + BASE + INPUTS_NAME + '/' + SCRIPT_NAME + ' --expected-boot-id LIVE_BOOT_UUID\n```\n\nFor local offline validation, run:\n\n```powershell\npython "' + str(inputs / SCRIPT_NAME) + '''" --check-only --local-archive-root "''' + str(output) + '"\n```\n\nShare a global build queue lock, rejecting other trtexec, monitor, queue, and known smoke/build jobs;\nEach item requires the fan to be active, at least 3 GiB of MemAvailable, and all temperatures below 85°C.\nThe original monitor\'s kernel/collector/temperature/low-memory-protection retains raw bytes; the new queue also repeats safety checks.\nBefore proceeding, verify the preceding item\'s final monitor PID, boot, root, and effective 7200-second budget; confirm the recorded owned\nProceed only after PID/start_ticks and private session/PGID descendants have all exited, and the global trtexec is empty.\nStandard builds failing or simple timeouts are allowed to proceed to the next item; safety events, boot changes, collector failures,\nProcess loss or failure to confirm cleanup completion will halt the entire queue. No automatic reexecution of the same configuration will be performed.\n\nA successful build does not establish actual precision adoption or successful numerical transfer. Each engine still requires the original actual-layer-format and\nINT8 tactic audits, followed by separate execution on the frozen image. The original PyTorch/ONNX numerical failures remain failures;\nThis package does not run dataset evaluation, does not alter precision or precision passing standards, and does not establish recovery, causality, or performance conclusions.\n'
    (inputs / 'README.md').write_text(readme, encoding='utf-8')
    check = subprocess.run([sys.executable, str(inputs / SCRIPT_NAME), '--check-only',
                            '--local-archive-root', str(output)], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, encoding='utf-8')
    assert check.returncode == 0, check.stdout + check.stderr
    checked = json.loads(check.stdout)
    write(inputs / 'local_check_only_result.json', checked)
    files = {str(path.relative_to(output)).replace('\\', '/'): info(path)
             for path in sorted(output.rglob('*')) if path.is_file()}
    write(inputs / 'input_receipt.json', {'schema': 'rampq.gate05.extended_restore_input_receipt/v1',
                                         'files': files, 'original_bundle_files_preserved': sum(len(row['all_original_files']) for row in rows),
                                         'native_operation_performed': False})
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(output)).replace('\\', '/'))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name, expected in files.items():
            raw = archive.read(name)
            assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'], name
    result = {'status': 'EXTENDED_RESTORE_QUEUE_PREPARED_NOT_RUN',
              'output': str(output), 'archive': str(archive_path),
              'archive_bytes': archive_path.stat().st_size, 'archive_sha256': sha(archive_path),
              'input_receipt_sha256': sha(inputs / 'input_receipt.json'),
              'plan_sha256': sha(inputs / 'plan.json'), 'queue_script_sha256': sha(inputs / SCRIPT_NAME),
              'budget_revision_sha256': sha(inputs / 'budget_revision.json'),
              'original_bundle_files_byte_identical': sum(len(row['all_original_files']) for row in rows),
              'local_check_only_status': checked['status'], 'python38_syntax_verified': True,
              'configuration_order': list(ORDER), 'inherited_spec_max_seconds': 1800,
              'effective_monitor_max_seconds': 7200, 'maximum_attempts_per_configuration': 1,
              'native_inputs_root': BASE + INPUTS_NAME, 'native_queue_run': BASE + QUEUE_NAME,
              'required_launch_command': ['python3', '-u', BASE + INPUTS_NAME + '/' + SCRIPT_NAME,
                                          '--expected-boot-id', 'LIVE_BOOT_UUID'],
              'native_operation_performed': False, 'inference_run': False, 'dataset_evaluation_run': False}
    write(output.with_suffix('.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

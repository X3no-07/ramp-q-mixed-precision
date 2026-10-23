"""Package six final frozen native raw jobs; metadata/files only, no execution.

--check-input-only validates final contracts, qualification and source identities
without creating a directory. Otherwise publish fresh six works, per-work ZIPs,
queue plan and one complete transport ZIP after exact byte revalidation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import traceback
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ORDER = ['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']
STAGE_NAME = 'gate05_dataset_native_stage_20261021'
BOARD_BASE = PurePosixPath('/home/nvidia/rampq') / STAGE_NAME
BOARD_INPUTS = '/home/nvidia/rampq/gate05_shared_inputs_20261021'
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
RAW_SHA = '11e9a09b1278fecff9f5c3f1a83011d077e88d550288503ef000c0c96012f94e'
ACCEPTANCE_SHA = '759506e78383fa262c3cb74735c24c57844d706fdb95bf95bce168d9b143f500'
TABLE_SHA = '542cfb974e32e81fa1acbb6c117a8902044ee574c86993135917b4eaa8800eeb'
RUNNER = 'run_gate05_frozen_raw_dataset_v2.py'
WRAPPER = 'jetson_run_frozen_raw_work.py'
QUEUE = 'jetson_frozen_raw_serial_queue.py'
TOOLS = {
    RUNNER: 'ef7e2780203c04442ee3378b7cea31093dafa40eb29c4cea9c012cdb0cd4d26b',
    WRAPPER: '4b3c6c5f11c4793810c272606bb1a288ab701c10a54c51c27ab8bd37a04f4329',
    QUEUE: 'ee0dee95bfac627c2ddffaad948e14fddb57f726040fc0df89d105bd465c49bc',
    'jetson_monitored_build.py': '71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
    'jetson_owned_group_guard.py': '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
    'jetson_collect_isolated_numeric_probe.py': 'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb',
}
HELPERS = {name: digest for name, digest in TOOLS.items() if name not in (RUNNER, WRAPPER, QUEUE)}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def contained(root, name):
    relative = PurePosixPath(name)
    require(str(relative) == name and not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
            'Unsafe frozen relative path: ' + name)
    path = root.joinpath(*relative.parts)
    require(root in path.resolve().parents and not path.is_symlink(), 'Frozen path escapes source root: ' + name)
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), 'Frozen path has symlink parent: ' + name)
    return path


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'No source symlinks allowed')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = info(path)
    return result


def board_path(name):
    path = PurePosixPath(name)
    require(path.is_absolute() and str(path) == name and '..' not in path.parts and
            PurePosixPath('/home/nvidia/rampq') in path.parents and BOARD_BASE not in path.parents,
            'Noncanonical or unsafe selected board source path')
    return path


def validate_inputs(contract_root):
    require(contract_root.is_dir() and not contract_root.is_symlink(), 'Final contract directory missing')
    require(sha(contract_root / 'raw_contract.json') == RAW_SHA and
            sha(contract_root / 'acceptance_contract.json') == ACCEPTANCE_SHA, 'Wrong final frozen contract pair')
    raw, acceptance = read(contract_root / 'raw_contract.json'), read(contract_root / 'acceptance_contract.json')
    receipt = read(contract_root / 'freeze_receipt.json')
    require(receipt['raw_contract_sha256'] == RAW_SHA and receipt['acceptance_contract_sha256'] == ACCEPTANCE_SHA and
            receipt['model_calls_by_freezer'] == 0 and receipt['dataset_outputs_read_by_freezer'] is False,
            'Final freeze receipt scope differs')
    before = inventory(contract_root)
    require(set(before) == set(receipt['files']) | {'freeze_receipt.json'}, 'Final contract directory contains unlisted/partial inputs')
    for name, expected in receipt['files'].items():
        require(before[name] == expected, 'Frozen receipt input changed: ' + name)
    require(raw['schema'] == 'rampq.gate05.frozen_raw_dataset_contract/v1' and
            raw['status'] == acceptance['status'] == 'FROZEN_BEFORE_DATASET_INFERENCE' and
            raw['runner_sha256'] == TOOLS[RUNNER] and list(raw['configurations']) == ORDER and
            raw['execution_order'] == ORDER and raw['native_total_model_calls'] == 900 and
            raw['maximum_attempts_per_backend_configuration'] == 1 and
            raw['warmup_calls'] == raw['extra_inference_calls'] == raw['new_builds'] == 0 and
            raw['no_NMS'] is True and raw['dataset_accuracy_evaluation_by_runner'] is False and
            raw['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and raw['gate05_status'] == 'INCOMPLETE',
            'Frozen raw scope differs')
    require(acceptance['schema'] == 'rampq.gate05.dataset_acceptance_contract/v1' and
            acceptance['raw_contract_sha256'] == RAW_SHA and acceptance['input_manifest_sha256'] == raw['input_manifest']['sha256'] and
            [row['name'] for row in acceptance['configurations']] == ORDER and acceptance['new150_model_calls_before_freeze'] == 0 and
            acceptance['protected_history']['original_complete_PyTorch_ORT_guard'] == 'FAILED_UNCHANGED', 'Acceptance pair differs')
    roles = acceptance['selected_family_roles']
    require(roles['primary_core_five_NONE'] == ORDER[:5] and roles['auxiliary_different_PREFER_policy'] == ['q_only_G1'] and
            roles['rejected_qonly_NONE_precision_failure_preserved'] is True and roles['pure_G1_arithmetic_causality_claim'] is False,
            'Primary/auxiliary or protected failure roles differ')
    table_path = contract_root / 'qualification/selected_identity_table.json'
    require(sha(table_path) == TABLE_SHA, 'Selected engine identity table changed')
    table = read(table_path)
    require([m['name'] for m in table['members']] == ORDER, 'Selected identity order changed')
    protected = {str(contract_root / name): value['sha256'] for name, value in before.items()}
    for name, digest in TOOLS.items():
        path = ROOT / 'scripts' / name
        require(sha(path) == digest, 'Pinned execution tool changed: ' + name)
        protected[str(path)] = digest
    protected[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    for member, accept in zip(table['members'], acceptance['configurations']):
        cfg = member['name']; row = raw['configurations'][cfg]; policy = row['trt']
        require(row['full_model'] is True and row['graph_sha256'] == member['source_graph_sha256'] == accept['source_graph_sha256'] and
                policy['authorized'] is True and policy['calls'] == 150 and
                policy['engine_sha256'] == member['engine_sha256'] == accept['engine_sha256'] and
                policy['attempt_state_root'] == '/home/nvidia/rampq/gate05_dataset_attempts_20261021', 'Native member identity/budget differs')
        require(row['actual_build_policy'] == member['runtime_build_policy'] and
                row['auxiliary_policy_difference_disclosed'] is (cfg == 'q_only_G1'), 'Selected runtime policy is not disclosed')
        board_path(member['board_graph']); board_path(member['board_engine'])
        for key, expected in (('local_model', member['source_graph_sha256']), ('local_engine', member['engine_sha256']),
                              ('local_precision_audit', member['native_precision_audit_sha256'])):
            path = Path(member[key]); require(path.is_file() and not path.is_symlink() and sha(path) == expected, 'Selected source changed: ' + key)
            protected[str(path.resolve())] = expected
        for backend in ('trt', 'ort'):
            for evidence in row[backend]['required_evidence']:
                path = contained(contract_root, evidence['file'])
                require(path.is_file() and info(path) == {'bytes': evidence['bytes'], 'sha256': evidence['sha256']},
                        'Qualification evidence changed: ' + evidence['file'])
    return raw, acceptance, table, before, protected


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), 'Refuse package file overwrite')
    shutil.copy2(source, destination)
    require(info(source) == info(destination), 'Copied bytes differ: ' + str(source))


def zip_exact(path, archive_root, files):
    """files maps archive-relative names to local regular source paths."""
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, archive_root + '/' + name)
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None and len(archive.namelist()) == len(files) and
                set(archive.namelist()) == {archive_root + '/' + name for name in files}, 'ZIP exact inventory differs')
        for name, source in files.items():
            require(hashlib.sha256(archive.read(archive_root + '/' + name)).hexdigest() == sha(source), 'ZIP bytes differ from sealed source')


def package(contract_root, target, raw, acceptance, table, source_before, protected):
    require(target.parent.is_dir() and not target.exists() and not target.is_symlink() and
            contract_root != target and contract_root not in target.parents and target not in contract_root.parents,
            'Fresh output must be disjoint from final contract source')
    partial = Path(tempfile.mkdtemp(prefix='.' + target.name + '.UNPUBLISHED-', dir=str(target.parent)))
    aggregate_name = STAGE_NAME + '.zip'
    aggregate_partial = partial / aggregate_name
    try:
        plan = {'schema': 'rampq.gate05.frozen_raw_serial_queue_plan/v1',
                'status': 'FROZEN_BEFORE_NATIVE_RAW_DATASET_EXECUTION', 'queue_script_sha256': TOOLS[QUEUE],
                'base': str(BOARD_BASE), 'expected_boot_id': BOOT, 'configuration_order': ORDER, 'backend': 'trt',
                'contract_sha256': RAW_SHA, 'acceptance_contract_sha256': ACCEPTANCE_SHA,
                'runner_sha256': TOOLS[RUNNER], 'wrapper_sha256': TOOLS[WRAPPER], 'helper_sha256': HELPERS,
                'model_calls_per_member': 150, 'maximum_model_calls': 900, 'warmup_calls': 0,
                'extra_inference_calls': 0, 'new_builds': 0, 'maximum_attempts': 1,
                'max_seconds_per_member': 1800, 'collection_timeout_seconds': 240, 'members': []}
        member_receipts = []
        for selected in table['members']:
            cfg = selected['name']; name = 'gate05_raw_' + cfg + '_20261021'
            work = partial / name; work.mkdir()
            for relative in source_before:
                copy_exact(contained(contract_root, relative), work.joinpath(*PurePosixPath(relative).parts))
            for filename in (RUNNER, WRAPPER, *HELPERS):
                copy_exact(ROOT / 'scripts' / filename, work / filename)
            spec = {'schema': 'rampq.gate05.raw_work_execution/v1', 'configuration': cfg, 'backend': 'trt',
                    'wrapper_sha256': TOOLS[WRAPPER], 'contract_sha256': RAW_SHA, 'acceptance_contract_sha256': ACCEPTANCE_SHA,
                    'inputs_root': BOARD_INPUTS, 'graph_file': selected['board_graph'], 'engine_file': selected['board_engine'],
                    'max_seconds': 1800, 'extra_model_calls': 0, 'maximum_model_calls': 150, 'new_builds': 0,
                    'NMS_calls': 0, 'automatic_retry': False, 'selected_build_policy': selected['runtime_build_policy'],
                    'auxiliary_policy_difference_disclosed': cfg == 'q_only_G1'}
            write(work / 'execution_spec.json', spec)
            inputs = inventory(work)
            write(work / 'manifest.json', {'schema': 'rampq.gate05.frozen_raw_work_bundle/v1',
                  'configuration': cfg, 'backend': 'trt', 'contract_sha256': RAW_SHA,
                  'acceptance_contract_sha256': ACCEPTANCE_SHA, 'files': inputs})
            require(inventory(contract_root) == source_before, 'Frozen contracts changed during packaging')
            sealed = inventory(work)
            work_zip = partial / (name + '.zip')
            zip_exact(work_zip, name, {relative: work / relative for relative in sealed})
            plan['members'].append({'configuration': cfg, 'root': str(BOARD_BASE / name),
                  'manifest_sha256': sha(work / 'manifest.json'), 'execution_spec_sha256': sha(work / 'execution_spec.json'),
                  'contract_sha256': RAW_SHA, 'archive': '/home/nvidia/rampq/gate05_raw_' + cfg + '_evidence_20261021.zip'})
            member_receipts.append({'configuration': cfg, 'local_work_directory': name, 'board_root': str(BOARD_BASE / name),
                  'source_graph_sha256': selected['source_graph_sha256'], 'engine_sha256': selected['engine_sha256'],
                  'manifest_sha256': sha(work / 'manifest.json'), 'execution_spec_sha256': sha(work / 'execution_spec.json'),
                  'work_file_count': len(sealed), 'qualification_file_count': len([n for n in sealed if n.startswith('qualification/')]),
                  'work_zip': work_zip.name, 'work_zip_identity': info(work_zip), 'exact_contract_and_qualification_bytes_preserved': True})
        copy_exact(ROOT / 'scripts' / QUEUE, partial / QUEUE)
        write(partial / 'queue_plan.json', plan)
        require(protected == {name: sha(Path(name)) for name in protected}, 'Protected input changed during packaging')
        write(partial / 'package_receipt.json', {'schema': 'rampq.gate05.frozen_raw_native_package_receipt/v1',
              'status': 'SIX_NATIVE_WORKS_SEALED_METADATA_ONLY_NO_EXECUTION', 'client_at_UTC': datetime.now(timezone.utc).isoformat(),
              'packager_sha256': sha(Path(__file__).resolve()), 'raw_contract_sha256': RAW_SHA,
              'acceptance_contract_sha256': ACCEPTANCE_SHA, 'source_contract_receipt_sha256': sha(contract_root / 'freeze_receipt.json'),
              'selected_identity_table_sha256': TABLE_SHA, 'queue_plan_sha256': sha(partial / 'queue_plan.json'),
              'queue_script_sha256': TOOLS[QUEUE], 'members': member_receipts, 'source_identities_before_and_after': protected,
              'protected_sources_unchanged': True, 'native_or_CPU_model_executions': 0, 'SSH_operations': 0,
              'GPU_runtime_imports': 0, 'dataset_outputs_read': False, 'attempt_markers_created': 0,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
              'limitations': 'Metadata seal only. Requires independent package review, board source/input checks and group-safe transport before explicit queue --execute.'})
        # Whole-stage archive excludes redundant individual work ZIPs. Its root
        # is the exact native stage basename; each work remains directly below.
        transport = {path.relative_to(partial).as_posix(): path for path in sorted(partial.rglob('*'))
                     if path.is_file() and path.suffix != '.zip'}
        write(partial / 'stage_manifest.json', {'schema': 'rampq.gate05.frozen_raw_native_transport/v1',
              'stage_root_name': STAGE_NAME, 'board_root': str(BOARD_BASE), 'queue_plan_sha256': sha(partial / 'queue_plan.json'),
              'files': {name: info(path) for name, path in transport.items()}, 'model_executions': 0})
        transport['stage_manifest.json'] = partial / 'stage_manifest.json'
        zip_exact(aggregate_partial, STAGE_NAME, transport)
        require(protected == {name: sha(Path(name)) for name in protected}, 'Protected input drift before publication')
        require(not target.exists(), 'Package destination appeared')
        partial.rename(target)
        aggregate_target = target / aggregate_name
        result = {'output': str(target), 'stage_zip': str(aggregate_target), 'stage_zip_identity': info(aggregate_target),
                  'queue_plan_sha256': sha(target / 'queue_plan.json'), 'stage_manifest_sha256': sha(target / 'stage_manifest.json'),
                  'package_receipt_sha256': sha(target / 'package_receipt.json'), 'members': member_receipts,
                  'model_executions': 0, 'SSH_operations': 0, 'dataset_outputs_read': False}
        return result
    except BaseException as error:
        if partial.is_dir():
            write(partial / 'package_failure.json', {'status': 'UNPUBLISHED_PARTIAL_NO_NATIVE_EXECUTION',
                  'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                  'target': str(target), 'model_executions': 0, 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    source = args.contract_root.resolve(); target = args.output.resolve()
    raw, acceptance, table, before, protected = validate_inputs(source)
    if args.check_input_only:
        print(json.dumps({'status': 'FINAL_CONTRACT_AND_SIX_SOURCE_IDENTITIES_VERIFIED_NO_PACKAGE_WRITTEN',
              'raw_contract_sha256': RAW_SHA, 'acceptance_contract_sha256': ACCEPTANCE_SHA,
              'configuration_order': ORDER, 'board_base': str(BOARD_BASE), 'board_inputs': BOARD_INPUTS,
              'source_contract_files': len(before), 'native_runtime_imported': False, 'model_executions': 0,
              'output_files_written': 0, 'SSH_operations': 0, 'dataset_outputs_read': False}, indent=2), flush=True)
        return
    print(json.dumps(package(source, target, raw, acceptance, table, before, protected), indent=2), flush=True)


if __name__ == '__main__':
    main()

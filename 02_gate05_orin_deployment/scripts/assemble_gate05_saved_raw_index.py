"""Authenticate completed saved jobs and publish a raw index; no inference/NMS."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RAW_SHA = '11e9a09b1278fecff9f5c3f1a83011d077e88d550288503ef000c0c96012f94e'
ACCEPT_SHA = '759506e78383fa262c3cb74735c24c57844d706fdb95bf95bce168d9b143f500'
ORDER = ['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
VERIFIED = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    data = path.read_bytes()
    remember(path, hashlib.sha256(data).hexdigest())
    return json.loads(data.decode('utf-8-sig'))


def remember(path, digest):
    key = str(path.resolve())
    require(key not in VERIFIED or VERIFIED[key] == digest, 'Input drift during saved-only verification: ' + key)
    VERIFIED[key] = digest


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    digest = h.hexdigest()
    remember(path, digest)
    return digest


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-observation', required=True, type=Path)
    parser.add_argument('--native-observation-sha256', required=True)
    args = parser.parse_args()
    implementation_sha256 = sha(Path(__file__))
    contracts = ROOT / 'artifacts/gate05/20261021_dataset_contracts'
    matrix = ROOT / 'artifacts/gate05/20261021_dataset_matrix'
    downloads = ROOT / 'artifacts/gate05/20261021_dataset_native_downloads'
    require(sha(contracts / 'raw_contract.json') == RAW_SHA and
            sha(contracts / 'acceptance_contract.json') == ACCEPT_SHA, 'Frozen contract pair changed')
    acceptance, raw = read(contracts / 'acceptance_contract.json'), read(contracts / 'raw_contract.json')
    require(str(matrix) == acceptance['postprocess_execution']['matrix_root'], 'Frozen matrix root changed')
    receipt = read(contracts / 'freeze_receipt.json')
    for name, info in receipt['files'].items():
        path = contracts / name
        require(sha(path) == info['sha256'] and path.stat().st_size == info['bytes'], 'Frozen qualification drift: ' + name)
    technical = read(contracts / 'qualification/technical_decision.json')
    require(technical['status'] == 'ALL_SELECTED_SIX_TECHNICALLY_ELIGIBLE_FOR_CONDITIONAL_DEPLOYMENT_EVALUATION',
            'Technical qualification does not authorize the selected deployment scope')
    require(sha(contracts / 'qualification/technical_decision.json') == acceptance['technical_preflight_decision_sha256'], 'Frozen technical adjudication identity differs')
    require(sha(args.native_observation) == args.native_observation_sha256, 'Final native observation identity changed')
    observed = read(args.native_observation)
    require(observed['schema'] == 'rampq.gate05.final_native_dataset_observation/v1' and
            observed['status'] == 'ALL_SIX_NATIVE150_TERMINAL_AND_FRESHLY_VERIFIED' and
            observed['boot_id'] == BOOT and observed['queue_process'] is None and
            observed['fan'] == 'active' and observed['nvinfer_sha256'] ==
            '4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca', 'Native final state is not eligible')
    native_status = observed['queue_status']
    require(native_status['status'] == 'ALL_SIX_RAW150_ATTEMPTS_TERMINAL_EMPTY_AND_COLLECTED_NO_SCORING' and
            native_status['completed_model_calls'] == 900 and native_status['new_builds'] == native_status['NMS_calls'] == 0 and
            [m['configuration'] for m in native_status['members']] == ORDER, 'Native queue is incomplete')
    ort_status = read(matrix / 'ort/status.json')
    require(ort_status['status'] == 'ALL_SIX_FULL_ORT_REFERENCES_COMPLETE_NO_SCORING' and
            ort_status['model_calls_completed'] == 900 and ort_status['NMS_calls'] == 0 and
            [m['configuration'] for m in ort_status['members']] == ORDER, 'Full CPU reference family incomplete')
    table = read(contracts / 'qualification/selected_identity_table.json')
    streams, runtime_reviews = [], []
    frozen_inputs = read(matrix / 'common_inputs/manifest.json')
    require(sha(matrix / 'common_inputs/manifest.json') == acceptance['input_manifest_sha256'], 'Common inputs changed')
    import numpy as np
    require(len(frozen_inputs['images']) == 150, 'Shared input count differs')
    for i, frozen in enumerate(frozen_inputs['images']):
        source = matrix / 'common_inputs' / frozen['input_file']
        require(frozen['image_index'] == i and sha(source) == frozen['input_sha256'] and source.stat().st_size == frozen['input_bytes'], 'Shared NPY file identity gap')
        array = np.load(source, allow_pickle=False)
        require(array.dtype == np.float32 and array.shape == (1, 3, 640, 640) and array.flags.c_contiguous and np.isfinite(array).all() and
                hashlib.sha256(array.tobytes(order='C')).hexdigest() == frozen['array_sha256'], 'Shared NPY payload identity gap')
    for member, fresh, selected in zip(native_status['members'], observed['members'], table['members']):
        cfg = selected['name']
        require(member['configuration'] == fresh['configuration'] == cfg and
                member['status'] == 'SUCCESS_RAW150_TERMINAL_EMPTY_AND_COLLECTED' and
                set(fresh['pids']) == set(fresh['process_observations']) ==
                {'monitor_pid', 'private_group', 'guard_pid', 'collector_pid', 'runner_pid'} and
                all(fresh['pids'][k] == member[k] for k in ('monitor_pid', 'private_group', 'guard_pid', 'collector_pid')) and
                fresh['all_recorded_processes_absent'] and all(p is None for p in fresh['process_observations'].values()) and fresh['wrapper_group_members'] == [] and
                fresh['source_graph_sha256'] == selected['source_graph_sha256'] and
                fresh['engine_sha256'] == selected['engine_sha256'], 'Native identity/group gap: ' + cfg)
        archive = downloads / (cfg + '.zip')
        require(sha(archive) == fresh['archive']['sha256'] == member['archive']['sha256'] and
                archive.stat().st_size == fresh['archive']['bytes'] == member['archive']['bytes'], 'Native archive changed: ' + cfg)
        verification = read(downloads / (cfg + '_verification.json'))
        require(verification['status'] == 'ARCHIVE_AND_EXACT_RECEIPT_FILE_INVENTORY_VERIFIED' and
                verification['archive_sha256'] == member['archive']['sha256'] and verification['archive_bytes'] == member['archive']['bytes'] and
                verification['verified_file_count'] == member['archive']['files'] and
                Path(verification['extracted_to']).resolve() == (matrix / 'native' / cfg).resolve(), 'Archive verification missing: ' + cfg)
        folder = matrix / 'native' / cfg
        require(sha(folder / verification['receipt_path']) == verification['receipt_sha256'], 'Extracted native receipt changed')
        for item in verification['verified_files']:
            path = folder / item['path']
            require(folder in path.resolve().parents and sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], 'Extracted native receipt file changed')
        monitor = read(folder / 'monitor/status.json')
        guard = read(folder / 'supplemental_group_guard.json')
        require(monitor['status'] == 'BUILD_SCRIPT_OK' and monitor['build_returncode'] == 0 and
                monitor.get('reason') is None and monitor['boot_id'] == BOOT and guard['status'] == 'COMPLETE' and
                monitor['root'] == member['root'] == guard['root'] and monitor['max_seconds'] == 1800 and
                monitor['build_script'] == member['root'] + '/jetson_run_frozen_raw_work.py' and
                guard['boot_id'] == BOOT and guard['owned_pgid'] == member['private_group'] == monitor['build_pid'] and
                monitor['monitor_pid'] == member['monitor_pid'] and guard['guard_pid'] == member['guard_pid'] and
                guard['leader_start_ticks'] == member['wrapper_leader_start_ticks'] and
                guard['owned_group_fully_empty'] and not guard.get('intervention_performed', False) and
                not any(a.get('dispatch') == 'SENT' for a in guard['termination_actions']), 'Native monitor/guard gap: ' + cfg)
        require(all(guard['monitor_identity'][k] == monitor[k] for k in ('root', 'monitor_pid', 'build_pid', 'max_seconds', 'build_script')),
                'Guard and terminal monitor identities disagree')
        require(sha(folder / 'monitor/status.json') == fresh['monitor_status_sha256'] and
                sha(folder / 'supplemental_group_guard.json') == fresh['guard_status_sha256'], 'Native terminal metadata drift')
        runtime = read(folder / 'raw_outputs/runtime_engine_layers.json')
        original = Path(selected['local_build']) / 'engine_layers.json'
        require(sha(original) == selected['engine_layers_sha256'] and runtime == read(original),
                'Actual deserialized engine inspector differs from qualified engine: ' + cfg)
        runtime_reviews.append({'configuration': cfg, 'runtime_engine_layers_sha256': sha(folder / 'raw_outputs/runtime_engine_layers.json'),
                                'qualified_layers_sha256': sha(original), 'parsed_layers_equal': True})
    for backend in ('trt', 'ort'):
        for cfg in ORDER:
            folder = matrix / ('native' if backend == 'trt' else 'ort') / cfg
            manifest_path = folder / ('raw_outputs/raw_manifest.json' if backend == 'trt' else 'raw_manifest.json')
            manifest = read(manifest_path)
            state, lineage = manifest['state'], manifest['lineage']
            binding = next(x for x in acceptance['configurations'] if x['name'] == cfg)
            policy = raw['configurations'][cfg][backend]
            require(manifest['status'] == 'ALL150_RAW_OUTPUTS_SERIALIZED_NO_SCORING' and
                    state['phase'] == 'COMPLETE' and state['all_outputs_finite'] and
                    state['model_calls_attempted'] == state['model_calls_completed'] == state['serialized_images'] == len(manifest['images']) == 150 and
                    manifest['NMS_run'] is False and manifest['target_matching_run'] is False and not manifest['nonfinite_images'], 'Incomplete raw stream')
            require(lineage['configuration'] == cfg and lineage['backend'] == backend and lineage['contract_sha256'] == RAW_SHA and
                    state['lineage'] == lineage and lineage['runner_sha256'] == raw['runner_sha256'] and
                    lineage['graph_sha256'] == binding['source_graph_sha256' if backend == 'trt' else 'independent_ORT_reference_graph_sha256'],
                    'Raw lineage differs from frozen graph')
            require(lineage['input_manifest_sha256'] == acceptance['input_manifest_sha256'] and
                    lineage['source_data_manifest_sha256'] == frozen_inputs['data_manifest_sha256'] ==
                    'c4dac80ad7c59524194a9501f5ff8126cc0517e84f29f4106068c0c4b27a93aa' and
                    state['warmup_calls'] == state['extra_inference_calls'] == state['builds'] == 0,
                    'Raw input provenance or extra model budget differs')
            environment = read(manifest_path.parent / 'backend_environment.json')
            require(environment == manifest['backend_environment'] and all(environment.get(k) == v for k, v in policy['environment'].items()),
                    'Actual backend environment differs from frozen runtime: ' + cfg)
            for evidence in policy['required_evidence']:
                evidence_path = (folder if backend == 'trt' else contracts) / evidence['file']
                require(sha(evidence_path) == evidence['sha256'] and evidence_path.stat().st_size == evidence['bytes'], 'Runtime qualification identity gap')
            if backend == 'trt':
                fresh = next(x for x in observed['members'] if x['configuration'] == cfg)
                require(sha(manifest_path) == fresh['raw_manifest_sha256'] and
                        sha(manifest_path.parent / 'run_state.json') == fresh['run_state_sha256'] and
                        state['boot_id'] == BOOT and lineage['engine_sha256'] == binding['engine_sha256'], 'Fresh native manifest/engine changed')
                for phase in ('before_first_model_call', 'after_last_model_call'):
                    loaded = read(manifest_path.parent / ('loaded_libraries_' + phase + '.json'))
                    maps_path = manifest_path.parent / ('maps_' + phase + '.txt')
                    sha(maps_path)
                    maps = maps_path.read_text(encoding='utf-8')
                    for required in policy['required_loaded_libraries']:
                        matches = [r for r in loaded['libraries'] if PurePosixPath(r.get('real_path', r['mapped_path'])).name == required['basename']]
                        require(len(matches) == 1, 'Missing/ambiguous actual loaded library')
                        library = matches[0]
                        require(library['hash_available'] and library['sha256'] == required['sha256'] and library['mapping_regions'],
                                'Actual loaded library digest differs')
                        for region in library['mapping_regions']:
                            device = region['device'].split(':')
                            require(int(region['inode']) == library['inode'] and int(device[0], 16) == library['device_major'] and
                                    int(device[1], 16) == library['device_minor'] and any(
                                        line.split()[0] == region['address'] and line.split()[3] == region['device'] and
                                        line.split()[4] == region['inode'] and line.rstrip().endswith(library['mapped_path'])
                                        for line in maps.splitlines() if len(line.split()) >= 6), 'Loaded library maps/device/inode do not authenticate captured binary')
            else:
                queue_member = next(x for x in ort_status['members'] if x['configuration'] == cfg)
                require(sha(manifest_path) == queue_member['raw_manifest_sha256'] and lineage['engine_sha256'] is None, 'ORT manifest changed since successful queue')
            require(len({r['raw_file'] for r in manifest['images']}) == 150, 'Duplicate saved output path')
            for row, frozen in zip(manifest['images'], frozen_inputs['images']):
                require(all(row[k] == frozen[k] for k in ('image_index', 'image', 'image_sha256', 'label', 'label_sha256',
                        'input_file', 'input_sha256', 'original_shape', 'ratio', 'left', 'top')) and
                        row['input_array_sha256'] == frozen['array_sha256'] and row['all_outputs_finite'], 'Raw input order/identity gap')
                raw_file = manifest_path.parent / row['raw_file']
                require(manifest_path.parent in raw_file.resolve().parents and sha(raw_file) == row['raw_sha256'] and
                        raw_file.stat().st_size == row['raw_bytes'], 'Raw file integrity gap')
                with np.load(raw_file, allow_pickle=False) as saved:
                    require(set(saved.files) == {'main_output', 'pe_output'}, 'Raw NPZ keys changed')
                    for output_name, npz_name in raw['output_npz_keys'].items():
                        array, stat = saved[npz_name], row['statistics'][output_name]
                        expected_shape = next(x['shape'] for x in raw['bindings'] if x['name'] == output_name)
                        require(array.dtype == np.float32 and list(array.shape) == expected_shape and np.isfinite(array).all() and
                                hashlib.sha256(array.tobytes(order='C')).hexdigest() == stat['raw_array_sha256'], 'Actual output array identity/finite gap')
            streams.append({'configuration': cfg, 'backend': backend, 'file': manifest_path.relative_to(matrix).as_posix(),
                            'sha256': sha(manifest_path)})
    protected_before_publication = dict(VERIFIED)
    for name, digest in protected_before_publication.items():
        require(sha(Path(name)) == digest, 'Saved input changed before engineering proof publication')
    proof = {'schema': 'rampq.gate05.saved_dataset_engineering_eligibility/v1', 'contract_sha256': ACCEPT_SHA,
             'status': 'ELIGIBLE', 'all_six_configuration_audits_eligible': True,
             'configuration_identities': {r['name']: {k: r[k] for k in ('source_graph_sha256', 'engine_sha256', 'build_request_sha256',
                 'native_precision_audit_sha256', 'float_fusion_comparability_audit_sha256', 'independent_ORT_reference_graph_sha256')} for r in acceptance['configurations']},
             'fresh_native_observation': {'file': str(args.native_observation.resolve()), 'sha256': args.native_observation_sha256},
             'runtime_inspector_saved_comparisons': runtime_reviews, 'native_model_calls': 900, 'independent_full_ORT_calls': 900,
             'assembler_sha256': implementation_sha256, 'protected_files_rehashed_before_publication': len(protected_before_publication),
             'model_or_NMS_calls_by_assembler': 0, 'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
             'scope': 'Conditional selected deployment comparison; auxiliary q_only has predeclared different PREFER policy; no pure G1 arithmetic causality or general numerical migration PASS.'}
    require(not (matrix / 'engineering_eligibility.json').exists() and not (matrix / 'raw_family_index.json').exists(), 'No index/proof overwrite')
    write(matrix / 'engineering_eligibility.json', proof)
    index = {'schema': 'rampq.gate05.frozen_raw_family_index/v1', 'contract_sha256': ACCEPT_SHA, 'raw_contract_sha256': RAW_SHA,
             'dataset_manifest_sha256': frozen_inputs['data_manifest_sha256'], 'inputs_directory': 'common_inputs',
             'input_manifest_file': 'manifest.json', 'input_manifest_sha256': acceptance['input_manifest_sha256'],
             'engineering_evidence': {'file': 'engineering_eligibility.json', 'sha256': sha(matrix / 'engineering_eligibility.json')}, 'streams': streams}
    write(matrix / 'raw_family_index.json', index)
    for name, digest in protected_before_publication.items():
        require(sha(Path(name)) == digest, 'Saved input changed during proof/index publication')
    print(json.dumps({'status': 'TWELVE_SAVED_RAW_STREAMS_INDEXED_NO_MODEL_OR_NMS_EXECUTION',
                      'raw_index_sha256': sha(matrix / 'raw_family_index.json'), 'engineering_evidence_sha256': sha(matrix / 'engineering_eligibility.json')}))


if __name__ == '__main__':
    main()

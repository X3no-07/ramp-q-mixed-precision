"""Read-only exact NVTX -> CUDA API -> runtime-kernel correlation on saved data.

Kernel names identify recorded functions, not their arithmetic instructions.
This script never runs an engine, modifies a profiler database, or upgrades the
retained strict INT8 result from 81/88 to 88/88.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SQLITE_SHA256 = '678b6a886207a0c89148df89117d0d30c86a24bc0689d4e5de4553df49747f29'
CAPTURE_RANGE = 'RAMPQ_QDQ_FIXED_IMAGE'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def source_hashes(paths):
    return {str(path.relative_to(ROOT)).replace('\\', '/'): {'bytes': path.stat().st_size, 'sha256': sha(path)}
            for path in sorted(set(paths), key=str)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/gate05/20261019_qall_extended/post_smoke_evidence')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gate05/20261019_qall_kernel_diagnostic')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    receipt = read(source / 'remote_receipt.json')
    original_paths = [source / 'remote_receipt.json', Path(__file__).resolve()]
    for name, info in receipt['files'].items():
        path = source / name
        require(source in path.resolve().parents and path.is_file() and
                path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Native receipt differs: ' + name)
        original_paths.append(path)
    database = source / 'smoke/profile_attempt3_fullprocess/fixed_image.sqlite'
    require(sha(database) == EXPECTED_SQLITE_SHA256, 'Unexpected frozen profiler SQLite')
    before = source_hashes(original_paths)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'source_hashes_before.json', before)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    build = read(source / 'build/build_result.json')
    audit = read(source / 'build/precision_audit.json')
    layers = read(source / 'build/engine_layers.json')['Layers']
    missing = [row for row in audit['mapping'] if row['integer_tactic_evidence'] == 'TACTIC_NAME_NOT_EXPOSED']
    targets = [row for row in layers if row.get('LayerType') == 'CudaDepthwiseConvolution']
    require(len(missing) == len(targets) == 7 and audit['convs_with_int8_input_and_weight_formats'] == 88 and
            audit['verified_int8_convs'] == 81 and audit['status'] == 'PRECISION_REVIEW_REQUIRED',
            'Original strict audit no longer matches the recorded diagnostic scope')
    require({row['engine_layer'] for row in missing} == {row['Name'] for row in targets},
            'Missing named tactics are not exactly the seven native depthwise layers')
    require(sha(source / 'build/model_q_all.engine') == build['engine_sha256'], 'Engine hash differs')
    require(sha(source / 'build/engine_layers.json') == audit['layer_file_sha256'], 'Native layer audit hash differs')
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    connection.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {'NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds', 'PROCESSES'}
        require(required <= tables, 'Required Nsight event tables are absent')
        strings = dict(connection.execute('SELECT id,value FROM StringIds'))
        ranges = []
        for event in connection.execute('SELECT rowid AS sqlite_rowid,* FROM NVTX_EVENTS'):
            row = dict(event)
            row['resolved_text'] = row['text'] if row['text'] is not None else strings.get(row['textId'])
            row['resolved_json'] = row['jsonText'] if row['jsonText'] is not None else strings.get(row['jsonTextId'])
            ranges.append(row)
        captures = [row for row in ranges if row['resolved_text'] == CAPTURE_RANGE and row['end'] is not None]
        require(len(captures) == 1, 'Expected exactly one completed fixed-image NVTX execution range')
        capture = captures[0]
        runtime = [dict(row) for row in connection.execute('SELECT rowid AS sqlite_rowid,* FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        kernel_rows = [dict(row) for row in connection.execute('SELECT rowid AS sqlite_rowid,* FROM CUPTI_ACTIVITY_KIND_KERNEL')]
        runtime_by_correlation, kernels_by_correlation = {}, {}
        for row in runtime:
            runtime_by_correlation.setdefault(row['correlationId'], []).append(row)
        for row in kernel_rows:
            kernels_by_correlation.setdefault(row['correlationId'], []).append(row)
        native_names = {row['Name'] for row in layers}
        layer_ranges = [row for row in ranges if row['resolved_text'] in native_names and row['end'] is not None]
        rows = []
        for target in targets:
            row = {'native_layer_name': target['Name'], 'native_layer_type': target['LayerType'],
                   'native_weight_format': target.get('Weights'),
                   'native_input_formats': [value.get('Format/Datatype') for value in target.get('Inputs', [])],
                   'native_output_formats': [value.get('Format/Datatype') for value in target.get('Outputs', [])],
                   'native_tactic_name': target.get('TacticName'), 'native_tactic_value': target.get('TacticValue'),
                   'runtime_kernel_identity_observed': False, 'integer_arithmetic_instruction_proven': False,
                   'strict_named_tactic_requirement_satisfied_by_trace': False,
                   'status': 'TRACE_ATTRIBUTION_INCOMPLETE', 'errors': [], 'correlations': []}
            selected = [event for event in layer_ranges if event['resolved_text'] == target['Name'] and
                        event['globalTid'] == capture['globalTid'] and capture['start'] <= event['start'] and
                        event['end'] <= capture['end']]
            row['exact_name_completed_nvtx_ranges_in_capture'] = len(selected)
            if len(selected) != 1:
                row['errors'].append('Native layer name does not identify exactly one range on the captured thread')
                rows.append(row)
                continue
            event = selected[0]
            row['nvtx'] = event
            require(event['endGlobalTid'] in (None, event['globalTid']), 'Cross-thread NVTX endpoints are unsupported')
            nvtx_layer = json.loads(event['resolved_json']) if event['resolved_json'] else None
            row['nvtx_json_matches_full_native_inspector_layer'] = nvtx_layer == target
            if nvtx_layer != target:
                row['errors'].append('Registered NVTX layer JSON differs from the frozen engine inspector')
            apis = [api for api in runtime if api['globalTid'] == event['globalTid'] and
                    event['start'] <= api['start'] and api['end'] <= event['end']]
            row['fully_contained_same_thread_runtime_api_count'] = len(apis)
            for api in apis:
                api_name = strings.get(api['nameId'])
                containing_layers = [candidate for candidate in layer_ranges if
                    candidate['globalTid'] == api['globalTid'] and candidate['start'] <= api['start'] and
                    api['end'] <= candidate['end']]
                matched_kernels = kernels_by_correlation.get(api['correlationId'], [])
                if not matched_kernels:
                    continue
                unique_api = len(runtime_by_correlation.get(api['correlationId'], [])) == 1
                exact_unique_layer = len(containing_layers) == 1 and containing_layers[0]['sqlite_rowid'] == event['sqlite_rowid']
                chain = {'runtime_api': dict(api, resolved_name=api_name),
                         'api_on_exact_nvtx_global_thread': True, 'api_fully_inside_layer_cpu_interval': True,
                         'correlation_id_unique_in_entire_runtime_table': unique_api,
                         'exactly_one_native_layer_contains_this_api': exact_unique_layer,
                         'kernel_rows_with_exact_correlation_id': []}
                if not unique_api or not exact_unique_layer or api['returnValue'] != 0:
                    row['errors'].append('Ambiguous or unsuccessful runtime launch correlation')
                for kernel in matched_kernels:
                    kernel = dict(kernel)
                    kernel['demangled_name_text'] = strings.get(kernel['demangledName'])
                    kernel['short_name_text'] = strings.get(kernel['shortName'])
                    kernel['mangled_name_text'] = strings.get(kernel['mangledName'])
                    processes = [dict(value) for value in connection.execute('SELECT * FROM PROCESSES WHERE globalPid=?',
                                                                            (kernel['globalPid'],))]
                    kernel['captured_process_rows'] = processes
                    kernel['arithmetic_instruction_evidence'] = 'NOT_PRESENT; function spelling is not disassembly or executed instruction evidence'
                    chain['kernel_rows_with_exact_correlation_id'].append(kernel)
                row['correlations'].append(chain)
            kernel_count = sum(len(chain['kernel_rows_with_exact_correlation_id']) for chain in row['correlations'])
            row['exact_correlated_kernel_count'] = kernel_count
            if not kernel_count:
                row['errors'].append('No GPU kernel linked by a unique same-thread enclosed runtime API correlation')
            if not row['errors']:
                row.update(runtime_kernel_identity_observed=True,
                           status='EXACT_LAYER_NVTX_RUNTIME_CORRELATION_TO_KERNEL_IDENTITY_OBSERVED')
            rows.append(row)
        sqlite_metadata = {'sqlite_version': sqlite3.sqlite_version,
                           'tables_and_schema': [dict(row) for row in connection.execute(
                               "SELECT name,sql FROM sqlite_master WHERE type='table' AND name IN ('NVTX_EVENTS','CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_KERNEL','StringIds','PROCESSES')")],
                           'runtime_api_rows': len(runtime), 'kernel_rows': len(kernel_rows), 'nvtx_rows': len(ranges),
                           'kernel_global_process_ids': sorted({row['globalPid'] for row in kernel_rows}),
                           'runtime_duplicate_correlation_ids': [value for value, items in runtime_by_correlation.items() if len(items) > 1]}
    finally:
        connection.close()
    observed = sum(row['runtime_kernel_identity_observed'] for row in rows)
    report = {'schema': 'rampq.gate05.depthwise_runtime_kernel_correlation/v1',
              'client_at': datetime.now().astimezone().isoformat(),
              'status': 'ALL_SEVEN_RUNTIME_KERNEL_IDENTITIES_EXACTLY_CORRELATED' if observed == 7 else 'RUNTIME_KERNEL_CORRELATION_INCOMPLETE',
              'sqlite_sha256': sha(database), 'engine_sha256': build['engine_sha256'],
              'source_receipt_sha256': sha(source / 'remote_receipt.json'), 'verified_source_receipt_files': len(receipt['files']),
              'capture_nvtx_range': capture, 'sqlite_metadata': sqlite_metadata,
              'target_depthwise_layers': 7, 'runtime_kernel_identity_observed_layers': observed,
              'evidence_layers': {'int8_input_and_weight_formats': {'source_convs': 88, 'target_depthwise_layers': 7,
                                   'source': 'Original actual engine inspector and unchanged precision audit'},
                                  'named_integer_tactics': {'source_convs': 81, 'target_depthwise_layers': 0,
                                   'missing_tactic_name_layers': 7, 'strict_audit_status_unchanged': audit['status']},
                                  'runtime_kernel_identity': {'target_depthwise_layers_observed': observed,
                                   'source': 'Exact full layer NVTX name and registered JSON, same-thread CPU API interval, unique correlationId -> GPU kernel event'},
                                  'integer_arithmetic_instructions': {'proven_layers': 0,
                                   'reason': 'Saved trace records kernel function names, launches and timings, without executed arithmetic instruction evidence. No inference from INT8-like spelling.'}},
              'layers': rows, 'strict_precision_audit_changed': False,
              'strict_named_tactic_source_convs_remain': 81,
              'new_engine_execution_performed': False, 'board_access': False,
              'performance_benchmark_claim': False, 'numerical_migration_claim': False,
              'limits': 'Runtime kernel identities are additional diagnostic evidence only. CPU NVTX intervals contain launch APIs; '
                        'GPU execution is asynchronous and need not overlap those CPU ranges. Exact correlation does not expose '
                        'arithmetic instructions or fill absent TacticName fields. Strict status remains PRECISION_REVIEW_REQUIRED.'}
    after = source_hashes(original_paths)
    require(before == after, 'A source file changed during read-only SQLite analysis')
    write(output / 'source_hashes_after.json', after)
    report['all_source_files_sha256_unchanged'] = True
    write(output / 'analysis.json', report)
    lines = ['# q_all 深度卷积的已保存运行内核诊断', '',
             '7 个原生深度卷积中，{} 个可通过完整 NVTX 层名、同线程 CUDA API 区间和唯一 correlationId 精确关联到运行内核。'.format(observed), '',
             '这是内核身份记录，不是整数算术指令证明。实际输入/权重 INT8 格式、命名 tactic、运行内核身份和算术指令证据分开记录。', '',
             '原严格结论保持：88 个卷积具备 INT8 输入/权重格式，81 个具备命名整数 tactic，7 个未暴露 tactic；严格审计仍为 PRECISION_REVIEW_REQUIRED。', '',
             '没有重新运行引擎、访问板子或修改源 SQLite/报告。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    files = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema': 'rampq.gate05.depthwise_runtime_diagnostic_files/v1', 'files': files})
    (output / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n' for name, info in files.items()), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'runtime_kernel_identity_observed_layers': observed,
                      'strict_named_tactic_source_convs_remain': 81,
                      'integer_arithmetic_instructions_proven': 0,
                      'all_source_files_sha256_unchanged': True,
                      'unique_observed_kernel_names': sorted({kernel['short_name_text'] for row in rows for chain in row['correlations']
                                                            for kernel in chain['kernel_rows_with_exact_correlation_id']})}, indent=2))


if __name__ == '__main__':
    main()

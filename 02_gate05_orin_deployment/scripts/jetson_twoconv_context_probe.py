"""Frozen corrected two-Conv context precision-flag diagnosis, Python 3.8 / TRT 8.5.

Compilation failures are observations and do not prevent the next frozen case.
Timeouts and outer interruptions abort the batch after child cleanup. This is
one fixed-image two-Conv diagnostic, not a detector accuracy or numerical pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback


SCHEMA = 'rampq.gate05.twoconv_context_probe/v1'
PARENT_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
PARENT_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
PROTOCOL_SHA = '6ba4cae53e1ca115c2b49bd7fdac91d80c15ef1cc1f854c57f01cd09ae185fa0'
CPU_OUTPUTS_SHA = 'cc980538ad73a077ecfff0798186db18f421de57a01655b571b62e224bbcbc4f'
MODEL_SHAS = {
    'code_clip': '5e6dfc0294a97b65810816c1ba80ef679b6d55851b03724be4c8de3e32c6c6fb',
}
CASES = (('code_clip_none', 'code_clip', 'none', True),
         ('code_clip_obey', 'code_clip', 'obey', True))
MAX_CHILD_SECONDS = 300
MAX_BATCH_SECONDS = 900


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def contained(root, relative):
    path = (root / relative).resolve()
    require(root in path.parents, 'Path outside frozen root: ' + str(relative))
    return path


def validate_inputs(root):
    """Stdlib only: usable on Windows without TensorRT, torch, or ONNX."""
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema'] == 'rampq.gate05.twoconv_context_probe_bundle/v1', 'Wrong manifest')
    for name, info in manifest['files'].items():
        path = contained(root, name)
        require(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'Bad manifest member: ' + name)
    for name in ('spec.json', 'parent_spec.json', 'parent_manifest.json', 'protocol.json',
                 'source_model.onnx', 'failure_inventory.json',
                 'jetson_twoconv_context_probe.py', 'jetson_build_twoconv_context_probe.py'):
        require(name in manifest['files'], 'Unsealed required file: ' + name)
    require(sha(root / 'parent_spec.json') == PARENT_SPEC_SHA, 'Parent spec changed')
    require(sha(root / 'parent_manifest.json') == PARENT_MANIFEST_SHA, 'Parent manifest changed')
    require(sha(root / 'protocol.json') == PROTOCOL_SHA, 'Frozen protocol changed')
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    spec = json.loads((root / 'spec.json').read_text(encoding='utf-8'))
    parent = json.loads((root / 'parent_spec.json').read_text(encoding='utf-8'))
    require(spec['schema'] == SCHEMA, 'Wrong spec')
    require(spec['max_seconds'] == MAX_BATCH_SECONDS and spec['max_build_seconds'] == MAX_CHILD_SECONDS,
            'Diagnostic budget changed')
    require(spec['maximum_native_builds'] == protocol['maximum_builds'] == 2 and
            spec['maximum_native_inferences'] == protocol['maximum_executions'] == 4 and
            spec['executions_per_successful_case'] == protocol['common_native_settings']['repeat_executions_if_success'] == 2,
            'Frozen build or execution count changed')
    require(spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED', 'Prior guard changed')
    require(parent['configuration'] == 'q_all' and parent['graph_sha256'] == protocol['source_graph_sha256'],
            'Wrong full failed q_all parent')
    for name, expected in (('source_model.onnx', protocol['source_graph_sha256']),
                           ('failure_inventory.json', protocol['failure_inventory_sha256'])):
        require(sha(root / name) == expected, 'Failed full graph source identity changed: ' + name)
    require(sha(root / 'parent_spec.json') == protocol['source_spec_sha256'] and
            sha(root / 'parent_manifest.json') == protocol['source_manifest_sha256'], 'Parent protocol mismatch')
    require(spec['conv_names'] == protocol['source_conv_names'] ==
            ['/model.0/conv/Conv', '/model.1/conv/Conv'], 'Wrong two source Conv names')
    require(spec['clip_names'] == protocol['clip_names'] and len(spec['clip_names']) == 3, 'Wrong three boundaries')
    require(spec['input_file_sha256'] == protocol['input_npy_sha256'], 'Input differs from protocol')
    require(spec['input_binding']['name'] == protocol['input_name'] and
            spec['input_binding']['shape'] == protocol['input_shape'] and
            spec['input_binding']['input'] is True and spec['input_binding']['dtype'] == 'DataType.FLOAT',
            'Input binding differs from protocol')
    require(len(spec['output_bindings_in_order']) == 1, 'Unexpected output tap count')
    output = spec['output_bindings_in_order'][0]
    require(output['name'] == protocol['output_name'] and output['shape'] == protocol['output_shape'] and
            output['input'] is False and output['dtype'] == 'DataType.FLOAT',
            'Output binding differs from new two-Conv protocol')
    require(spec['cpu_expected_outputs_sha256'] == CPU_OUTPUTS_SHA, 'New two-Conv CPU reference changed')
    require(set(spec['variants']) == set(MODEL_SHAS), 'Variant set changed')
    for name, expected_sha in MODEL_SHAS.items():
        variant = spec['variants'][name]
        path = contained(root, variant['model_file'])
        require(variant['model_file'] in manifest['files'] and sha(path) == expected_sha == variant['model_sha256'],
                'Frozen graph changed: ' + name)
        require(variant['nodes'] == protocol['graph_node_count'] == 19 and
                variant['initializers'] == protocol['initializer_count'] == 20, 'Two-Conv slice count changed')
        require(variant['expected_output_npz_keys'] == {protocol['output_name']: 'code_clip_output'},
                'Wrong expected two-Conv raw output key')
    for file_key, hash_key in (('input_file', 'input_file_sha256'),
                               ('cpu_expected_outputs_file', 'cpu_expected_outputs_sha256')):
        require(spec[file_key] in manifest['files'] and sha(contained(root, spec[file_key])) == spec[hash_key],
                'Frozen data changed: ' + file_key)
    require(len(spec['cases']) == len(CASES), 'Case count changed')
    require(spec['cases'] == protocol['cases'], 'Cases differ from frozen protocol')
    for row, (case, variant, flag, correction) in zip(spec['cases'], CASES):
        require(row['id'] == case and row['variant'] == variant and
                row['precision_constraint_flag'] == flag and
                row['apply_native_clip_attribute_correction'] is correction and
                row['max_build_seconds'] == MAX_CHILD_SECONDS and row['iterations'] == 1,
                'Frozen case changed: ' + case)
    return spec


def stats(array, np):
    values = array.astype(np.float64)
    finite = np.isfinite(values)
    selected = values[finite]
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'all_finite': bool(finite.all()),
            'nonfinite_count': int((~finite).sum()), 'min': float(selected.min()) if selected.size else None,
            'max': float(selected.max()) if selected.size else None,
            'raw_bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def compare(actual, expected, np):
    require(actual.shape == expected.shape, 'Comparison shape mismatch')
    valid = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual[valid].astype(np.float64) - expected[valid].astype(np.float64))
    return {'all_pairs_finite': bool(valid.all()), 'nonfinite_pair_count': int((~valid).sum()),
            'numeric_equal': bool(np.array_equal(actual, expected)),
            'byte_identical': bool(actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()),
            'different_elements': int(np.count_nonzero(actual != expected)),
            'max_abs': float(delta.max()) if delta.size else None,
            'mean_abs': float(delta.mean()) if delta.size else None,
            'rms_difference': float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
            'descriptive_only_no_new_pass_criterion': True}


def stop_child(process):
    """Child is the direct Python API builder, with no shell or grandchildren."""
    attempts = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            break
        try:
            process.send_signal(sig)
            attempts.append(int(sig))
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            break
    require(process.poll() is not None, 'Owned Python builder did not terminate')
    return {'pid': process.pid, 'signals_sent': attempts, 'returncode': process.poll(),
            'direct_python_api_child_terminated': True}


def terminate_as_interrupt(_signum, _frame):
    raise KeyboardInterrupt('Outer termination requested')


def infer_case(root, out, spec, case, expected, image, np, trt, torch):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((out / 'model.engine').read_bytes())
    require(engine is not None, 'Cannot deserialize built engine')
    inspector_raw = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    (out / 'runner_engine_layers.json').write_text(inspector_raw + '\n', encoding='utf-8')
    inspector = json.loads(inspector_raw)
    convs = [row for row in inspector.get('Layers', []) if 'Convolution' in row.get('LayerType', '')]
    write(out / 'conv_rows.json', convs)
    mapping = []
    for source in spec['conv_names']:
        indices = [index for index, row in enumerate(convs) if source in row.get('Name', '')]
        mapping.append({'source_conv_name': source, 'actual_row_indices': indices,
                        'status': 'UNIQUE_SOURCE_MAPPING' if len(indices) == 1 else
                                  ('MISSING_SOURCE_MAPPING' if not indices else 'AMBIGUOUS_SOURCE_MAPPING')})
    unique_indices = [row['actual_row_indices'][0] for row in mapping if row['status'] == 'UNIQUE_SOURCE_MAPPING']
    mapping_unambiguous = len(convs) == 2 and len(unique_indices) == 2 and len(set(unique_indices)) == 2
    if not mapping_unambiguous:
        for row in mapping:
            row['two_distinct_actual_convolutions_verified'] = False
    write(out / 'conv_source_mapping.json', {'mapping': mapping,
          'actual_convolution_row_count': len(convs), 'two_distinct_actual_convolutions_verified': mapping_unambiguous,
          'status': 'TWO_CONV_SOURCES_UNIQUELY_MAPPED' if mapping_unambiguous else 'ACTUAL_CONV_MAPPING_AMBIGUOUS_OR_MISSING'})
    # Preserve full original JSON rows, including Inputs, Weights, Outputs,
    # activation/fusion details, and TacticName; do not infer adopted precision.
    context = engine.create_execution_context()
    require(context is not None, 'Cannot create execution context')
    desired = {row['name']: row for row in [spec['input_binding']] + spec['output_bindings_in_order']}
    require(engine.num_bindings == len(desired) == 2, 'Binding count differs')
    buffers, bindings = [], []
    for index in range(engine.num_bindings):
        name = engine.get_binding_name(index)
        require(name in desired, 'Unexpected runtime binding: ' + name)
        target = desired[name]
        require(list(engine.get_binding_shape(index)) == target['shape'] and
                bool(engine.binding_is_input(index)) == target['input'] and
                engine.get_binding_dtype(index) == trt.float32 and
                engine.get_binding_format(index) == trt.TensorFormat.LINEAR,
                'Runtime binding contract differs: ' + name)
        bindings.append({'name': name, 'input': target['input'], 'shape': target['shape'],
                         'dtype': str(engine.get_binding_dtype(index)),
                         'format': str(engine.get_binding_format(index))})
        buffers.append(torch.from_numpy(image.copy()).cuda() if target['input'] else
                       torch.empty(target['shape'], dtype=torch.float32, device='cuda'))
    require(len({row['name'] for row in bindings}) == 2, 'Duplicate runtime bindings')
    write(out / 'runtime_bindings.json', bindings)
    input_index = next(i for i, row in enumerate(bindings) if row['input'])
    output_index = next(i for i, row in enumerate(bindings) if not row['input'])
    require(np.array_equal(buffers[input_index].cpu().numpy(), image), 'Input copy differs')
    saved = {}
    for repeat in range(2):
        torch.cuda.synchronize()
        require(context.execute_v2([int(value.data_ptr()) for value in buffers]), 'Execution failed')
        torch.cuda.synchronize()
        saved['first' if repeat == 0 else 'repeat'] = buffers[output_index].cpu().numpy().copy()
    np.savez_compressed(str(out / 'raw_outputs.npz'), **saved)
    variant = spec['variants'][case['variant']]
    expected_key = variant['expected_output_npz_keys'][bindings[output_index]['name']]
    result = {'status': 'TWOCONV_CONTEXT_FIXED_INPUT_OUTPUTS_SAVED', 'case': case['id'],
              'variant': case['variant'], 'precision_constraint_flag': case['precision_constraint_flag'],
              'bindings': bindings, 'stats': stats(saved['first'], np),
              'comparison_to_frozen_cpu_ort': compare(saved['first'], expected[expected_key], np),
              'repeat_comparison': compare(saved['repeat'], saved['first'], np),
              'conv_rows_file': 'conv_rows.json', 'conv_source_mapping_file': 'conv_source_mapping.json',
              'two_distinct_actual_convolutions_verified': mapping_unambiguous,
              'convolution_descriptors': [{'name': row.get('Name'),
                  'inputs': row.get('Inputs'), 'weights': row.get('Weights'),
                  'outputs': row.get('Outputs'), 'activation': row.get('Activation'),
                  'tactic_name': row.get('TacticName'), 'tactic_value': row.get('TacticValue'),
                  'actual_precision_requires_independent_review': True} for row in convs],
              'engine_sha256': sha(out / 'model.engine'), 'raw_outputs_sha256': sha(out / 'raw_outputs.npz'),
              'spec_sha256': sha(root / 'spec.json'), 'model_sha256': variant['model_sha256'],
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'tensorrt': trt.__version__,
              'torch': torch.__version__, 'diagnostic_only': True,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
              'dataset_evaluation_run': False, 'accuracy_pass_claim': False, 'performance_pass_claim': False,
              'limits': 'One fixed input, two Conv layers, three activation boundaries; no new numeric tolerance.'}
    write(out / 'result.json', result)
    output = saved['first']
    del context, buffers, engine, runtime
    torch.cuda.empty_cache()
    return result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    spec = validate_inputs(root)
    if args.check_input_only:
        print(json.dumps({'status': 'FROZEN_TWOCONV_CONTEXT_INPUTS_VERIFIED', 'spec_sha256': sha(root / 'spec.json'),
                          'case_count': len(CASES), 'native_runtime_imported': False}), flush=True)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Native Jetson required')
    import numpy as np
    import tensorrt as trt
    import torch
    require(trt.__version__.startswith('8.5.') and torch.cuda.is_available(), 'Wrong native runtime')
    image = np.load(str(root / spec['input_file']), allow_pickle=False)
    require(image.dtype == np.float32 and list(image.shape) == spec['input_binding']['shape'], 'Input contract')
    with np.load(str(root / spec['cpu_expected_outputs_file']), allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    run = root / 'results' / ('probe_' + time.strftime('%Y%m%dT%H%M%S') + '_' + str(os.getpid()))
    run.mkdir(parents=True, exist_ok=False)
    write(root / 'latest_probe.json', {'run': str(run), 'spec_sha256': sha(root / 'spec.json')})
    write(run / 'environment.json', {'python': sys.version, 'machine': platform.machine(),
                                      'tensorrt': trt.__version__, 'torch': torch.__version__,
                                      'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    started = time.monotonic()
    results, arrays = {}, {}
    aborted = None
    current = None
    try:
        for case in spec['cases']:
            current = case['id']
            require(time.monotonic() - started < MAX_BATCH_SECONDS, 'Batch budget exhausted before next case')
            out = run / current
            out.mkdir()
            command = [sys.executable, '-u', str(root / 'jetson_build_twoconv_context_probe.py'),
                       '--root', str(root), '--case', current, '--output', str(out / 'model.engine')]
            write(out / 'request.json', {'command': command, 'case': case,
                                         'spec_sha256': sha(root / 'spec.json'),
                                         'model_sha256': spec['variants'][case['variant']]['model_sha256'],
                                         'max_child_seconds': MAX_CHILD_SECONDS, 'retries': 0})
            print('TWOCONV_CONTEXT_BUILD_START ' + current, flush=True)
            process = None
            code = None
            cleanup = None
            timed_out = False
            child_phase = 'PREBUILD_INPUT_REVALIDATION'
            try:
                validate_inputs(root)
                with (out / 'builder.log').open('wb') as log:
                    child_phase = 'BUILDING'
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = process.wait(timeout=min(MAX_CHILD_SECONDS, MAX_BATCH_SECONDS - (time.monotonic() - started)))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        raise RuntimeError('Child build timeout; aborting frozen batch: ' + current)
                    finally:
                        if process is not None:
                            cleanup = stop_child(process)
                            write(out / 'child_cleanup.json', cleanup)
                child_phase = 'POSTBUILD_INPUT_REVALIDATION'
                validate_inputs(root)
                if code is not None and code < 0:
                    child_phase = 'CHILD_TERMINATED_BY_SIGNAL'
                    raise RuntimeError('Builder was terminated by signal; aborting remaining cases')
            except BaseException as error:
                failure = {'status': 'BUILD_TIMEOUT_BATCH_ABORTED' if timed_out else 'CHILD_OR_INPUT_STATE_BATCH_ABORTED',
                           'phase': child_phase,
                           'case': current, 'error_type': type(error).__name__, 'error': str(error),
                           'returncode': process.poll() if process is not None else None, 'cleanup': cleanup,
                           'inference_run': False, 'next_case_permitted': False}
                # The child may have saved its own exception on SIGINT or
                # validation failure; preserve that original record intact.
                write(out / 'runner_abort.json', failure)
                results[current] = failure
                raise
            if code != 0:
                failure = {'status': 'BUILD_FAILED_OBSERVATION', 'case': current, 'returncode': code,
                           'inference_run': False, 'next_case_permitted': True,
                           'spec_sha256': sha(root / 'spec.json'),
                           'builder_failure_file': 'build_failure.json' if (out / 'build_failure.json').is_file() else None}
                write(out / 'runner_build_failure.json', failure)
                results[current] = failure
                print('TWOCONV_CONTEXT_BUILD_FAILED_CONTINUE ' + current, flush=True)
                continue
            try:
                result, array = infer_case(root, out, spec, case, expected, image, np, trt, torch)
                results[current], arrays[current] = result, array
            except Exception as error:
                failure = {'status': 'BUILT_ENGINE_INFERENCE_OR_INSPECTION_FAILED', 'case': current,
                           'error_type': type(error).__name__, 'error': str(error),
                           'traceback': traceback.format_exc(), 'next_case_permitted': True}
                write(out / 'inference_failure.json', failure)
                results[current] = failure
            require(time.monotonic() - started <= MAX_BATCH_SECONDS, 'Batch budget exceeded')
            print('TWOCONV_CONTEXT_CASE_FINISHED ' + current, flush=True)
    except BaseException as error:
        aborted = {'case': current, 'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc()}
        raise
    finally:
        pairs = (('code_clip_none', 'code_clip_obey'),)
        comparisons = {}
        for first, second in pairs:
            comparisons[first + '_vs_' + second] = (compare(arrays[first], arrays[second], np)
                if first in arrays and second in arrays else {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'})
        write(run / 'summary.json', {'status': 'TWOCONV_CONTEXT_BATCH_ABORTED' if aborted else 'TWOCONV_CONTEXT_BATCH_COMPLETED',
                                     'cases': results, 'cross_native_comparisons': comparisons,
                                     'requested_case_order': [row[0] for row in CASES],
                                     'observed_case_count': len(results), 'successful_output_case_count': len(arrays),
                                     'elapsed_seconds_not_benchmark': time.monotonic() - started,
                                     'aborted': aborted, 'batch_completion_is_not_numerical_pass': True,
                                     'diagnostic_only': True, 'dataset_evaluation_run': False,
                                     'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'})


if __name__ == '__main__':
    main()

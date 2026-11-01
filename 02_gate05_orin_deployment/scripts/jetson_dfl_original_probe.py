"""Frozen original FLOAT-Clip DFL tail context precision-flag diagnosis, Python 3.8 / TRT 8.5.

Compilation failures are observations and do not prevent the next frozen case.
Timeouts and outer interruptions abort the batch after child cleanup. This is
one frozen internal DFL-feature diagnostic, not a detector accuracy or numerical pass.
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


SCHEMA = 'rampq.gate05.dfl_original_probe/v1'
PARENT_SPEC_SHA = '934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d'
PARENT_MANIFEST_SHA = '3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0'
PARENT_CHECKPOINT_SHA = '28edf81c0be1f72065a8952f84dc31330ba95709d50ed13aafc16a5bb56c602b'
ORIGINAL_FULL_GRAPH_SHA = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
FIXTURE_LINEAGE_SHA = 'ecd2bb92223c5e3d820cd70c0bacc5f1b9fc42b40a545f0552b0c247e45e4365'
PROTOCOL_SHA = '81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed'
CPU_OUTPUTS_SHA = 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511'
INPUT_FIXTURE_SHA = 'cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab'
MODEL_SHAS = {
    'original': 'b60be2c2d196ed1fb4e962446f374d9e53d066476172cfa1db415194b2306d90',
}
CASES = (('original_none', 'original', 'none', False),
         ('original_obey', 'original', 'obey', False))
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
    require(manifest['schema'] == 'rampq.gate05.dfl_original_probe_bundle/v1', 'Wrong manifest')
    for name, info in manifest['files'].items():
        path = contained(root, name)
        require(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'Bad manifest member: ' + name)
    for name in ('spec.json', 'parent_dfl_spec.json', 'parent_dfl_manifest.json', 'protocol.json',
                 'source_original_full_model.onnx', 'parent_dfl_checkpoint.json', 'fixture_lineage.json',
                 'jetson_dfl_original_probe.py', 'jetson_build_dfl_original_probe.py'):
        require(name in manifest['files'], 'Unsealed required file: ' + name)
    require(sha(root / 'parent_dfl_spec.json') == PARENT_SPEC_SHA, 'Parent DFL spec changed')
    require(sha(root / 'parent_dfl_manifest.json') == PARENT_MANIFEST_SHA, 'Parent DFL manifest changed')
    require(sha(root / 'parent_dfl_checkpoint.json') == PARENT_CHECKPOINT_SHA, 'Parent DFL checkpoint changed')
    require(sha(root / 'fixture_lineage.json') == FIXTURE_LINEAGE_SHA, 'Saved fixture provenance changed')
    require(sha(root / 'protocol.json') == PROTOCOL_SHA, 'Frozen protocol changed')
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    spec = json.loads((root / 'spec.json').read_text(encoding='utf-8'))
    parent = json.loads((root / 'parent_dfl_spec.json').read_text(encoding='utf-8'))
    require(spec['schema'] == SCHEMA, 'Wrong spec')
    require(spec['max_seconds'] == MAX_BATCH_SECONDS and spec['max_build_seconds'] == MAX_CHILD_SECONDS,
            'Diagnostic budget changed')
    require(spec['maximum_native_builds'] == protocol['maximum_builds'] == 2 and
            spec['maximum_native_inferences'] == protocol['maximum_executions'] == 4 and
            spec['executions_per_successful_case'] == protocol['common_native_settings']['repeat_executions_if_success'] == 2,
            'Frozen build or execution count changed')
    require(spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED', 'Prior guard changed')
    require(parent['schema'] == 'rampq.gate05.dfl_context_probe/v1' and
            parent['source_graph_sha256'] == protocol['corrected_source_graph_sha256'], 'Wrong previous DFL parent')
    require(sha(root / 'source_original_full_model.onnx') == ORIGINAL_FULL_GRAPH_SHA == protocol['source_graph_sha256'],
            'Wrong original full q_all source identity')
    require(PARENT_SPEC_SHA == protocol['parent_dfl_spec_sha256'] and
            PARENT_MANIFEST_SHA == protocol['parent_dfl_manifest_sha256'] and
            PARENT_CHECKPOINT_SHA == protocol['parent_dfl_checkpoint_sha256'], 'Parent DFL protocol mismatch')
    require(spec['new_CPU_execution_runs_this_stage'] == protocol['CPU_execution_runs_this_stage'] == 0,
            'This control must reuse saved CPU fixture and outputs without rerun')
    require(spec['conv_names'] == protocol['source_conv_names'] ==
            ['/model.23/dfl/conv/Conv'], 'Wrong one DFL source Conv name')
    require(spec['clip_names'] == protocol['clip_names'] == ['rampq_model_23_dfl_conv_Clip'],
            'Wrong DFL activation boundary')
    require(spec['input_file'] == 'dfl_transpose_fixture.npy' and spec['input_file_sha256'] == INPUT_FIXTURE_SHA,
            'Wrong DFL internal input; image or old prefix fixture is forbidden')
    require(INPUT_FIXTURE_SHA == protocol['fixture_sha256'] and CPU_OUTPUTS_SHA == protocol['cpu_expected_outputs_sha256'],
            'Fixture or CPU reference differs from frozen previous stage')
    require(spec['input_binding']['name'] == protocol['input_name'] and
            spec['input_binding']['shape'] == protocol['input_shape'] and
            spec['input_binding']['input'] is True and spec['input_binding']['dtype'] == 'DataType.FLOAT',
            'Input binding differs from protocol')
    require(len(spec['output_bindings_in_order']) == 1, 'Unexpected output tap count')
    output = spec['output_bindings_in_order'][0]
    require(output['name'] == protocol['output_name'] and output['shape'] == protocol['output_shape'] and
            output['input'] is False and output['dtype'] == 'DataType.FLOAT',
            'Output binding differs from new DFL tail protocol')
    require(spec['cpu_expected_outputs_sha256'] == CPU_OUTPUTS_SHA, 'New DFL tail CPU reference changed')
    require(set(spec['variants']) == set(MODEL_SHAS), 'Variant set changed')
    for name, expected_sha in MODEL_SHAS.items():
        variant = spec['variants'][name]
        path = contained(root, variant['model_file'])
        require(variant['model_file'] in manifest['files'] and sha(path) == expected_sha == variant['model_sha256'] == protocol['original_tail_sha256'],
                'Frozen graph changed: ' + name)
        require(variant['nodes'] == protocol['graph_node_count'] == 9 and
                variant['initializers'] == protocol['initializer_count'] == 7, 'DFL slice count changed')
        require(variant['expected_output_npz_keys'] == {protocol['output_name']: 'original_output'},
                'Wrong expected DFL tail raw output key')
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


def infer_case(root, out, spec, case, expected, feature_input, np, trt, torch):
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
    mapping_unambiguous = len(convs) == 1 and len(unique_indices) == 1 and len(set(unique_indices)) == 1
    if not mapping_unambiguous:
        for row in mapping:
            row['one_dfl_actual_convolution_uniquely_mapped'] = False
    write(out / 'conv_source_mapping.json', {'mapping': mapping,
          'actual_convolution_row_count': len(convs), 'one_dfl_actual_convolution_uniquely_mapped': mapping_unambiguous,
          'status': 'DFL_CONV_SOURCE_UNIQUELY_MAPPED' if mapping_unambiguous else 'ACTUAL_CONV_MAPPING_AMBIGUOUS_OR_MISSING',
          'int8_output_required': False, 'output_qdq_present': False})
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
        buffers.append(torch.from_numpy(feature_input.copy()).cuda() if target['input'] else
                       torch.empty(target['shape'], dtype=torch.float32, device='cuda'))
    require(len({row['name'] for row in bindings}) == 2, 'Duplicate runtime bindings')
    write(out / 'runtime_bindings.json', bindings)
    input_index = next(i for i, row in enumerate(bindings) if row['input'])
    output_index = next(i for i, row in enumerate(bindings) if not row['input'])
    require(np.array_equal(buffers[input_index].cpu().numpy(), feature_input), 'DFL feature input copy differs')
    saved = {}
    for repeat in range(2):
        torch.cuda.synchronize()
        require(context.execute_v2([int(value.data_ptr()) for value in buffers]), 'Execution failed')
        torch.cuda.synchronize()
        saved['first' if repeat == 0 else 'repeat'] = buffers[output_index].cpu().numpy().copy()
    np.savez_compressed(str(out / 'raw_outputs.npz'), **saved)
    variant = spec['variants'][case['variant']]
    expected_key = variant['expected_output_npz_keys'][bindings[output_index]['name']]
    result = {'status': 'DFL_ORIGINAL_FIXED_INPUT_OUTPUTS_SAVED', 'case': case['id'],
              'variant': case['variant'], 'precision_constraint_flag': case['precision_constraint_flag'],
              'bindings': bindings, 'stats': stats(saved['first'], np),
              'comparison_to_frozen_cpu_ort': compare(saved['first'], expected[expected_key], np),
              'repeat_comparison': compare(saved['repeat'], saved['first'], np),
              'conv_rows_file': 'conv_rows.json', 'conv_source_mapping_file': 'conv_source_mapping.json',
              'one_dfl_actual_convolution_uniquely_mapped': mapping_unambiguous,
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
              'output_role': 'DFL_DISTANCES_NOT_DECODED_DETECTION_BOXES',
              'fixture_role': 'CORRECTED_ONNX_CPU_UPSTREAM_INTERNAL_LOGITS_NOT_NATIVE_DATA',
              'actual_conv_output_int8_required': False,
              'limits': 'One fixed internal feature input, one DFL Conv and one activation boundary; no new numeric tolerance. Output is DFL distance, no output QDQ or INT8-output acceptance rule.'}
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
        print(json.dumps({'status': 'FROZEN_DFL_ORIGINAL_INPUTS_VERIFIED', 'spec_sha256': sha(root / 'spec.json'),
                          'case_count': len(CASES), 'native_runtime_imported': False}), flush=True)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Native Jetson required')
    import numpy as np
    import tensorrt as trt
    import torch
    require(trt.__version__.startswith('8.5.') and torch.cuda.is_available(), 'Wrong native runtime')
    feature_input = np.load(str(root / spec['input_file']), allow_pickle=False)
    require(feature_input.dtype == np.float32 and list(feature_input.shape) == spec['input_binding']['shape'] and
            bool(np.isfinite(feature_input).all()), 'DFL internal feature input contract')
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
            command = [sys.executable, '-u', str(root / 'jetson_build_dfl_original_probe.py'),
                       '--root', str(root), '--case', current, '--output', str(out / 'model.engine')]
            write(out / 'request.json', {'command': command, 'case': case,
                                         'spec_sha256': sha(root / 'spec.json'),
                                         'model_sha256': spec['variants'][case['variant']]['model_sha256'],
                                         'max_child_seconds': MAX_CHILD_SECONDS, 'retries': 0})
            print('DFL_ORIGINAL_BUILD_START ' + current, flush=True)
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
                print('DFL_ORIGINAL_BUILD_FAILED_CONTINUE ' + current, flush=True)
                continue
            try:
                result, array = infer_case(root, out, spec, case, expected, feature_input, np, trt, torch)
                results[current], arrays[current] = result, array
            except Exception as error:
                failure = {'status': 'BUILT_ENGINE_INFERENCE_OR_INSPECTION_FAILED', 'case': current,
                           'error_type': type(error).__name__, 'error': str(error),
                           'traceback': traceback.format_exc(), 'next_case_permitted': True}
                write(out / 'inference_failure.json', failure)
                results[current] = failure
            require(time.monotonic() - started <= MAX_BATCH_SECONDS, 'Batch budget exceeded')
            print('DFL_ORIGINAL_CASE_FINISHED ' + current, flush=True)
    except BaseException as error:
        aborted = {'case': current, 'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc()}
        raise
    finally:
        pairs = (('original_none', 'original_obey'),)
        comparisons = {}
        for first, second in pairs:
            comparisons[first + '_vs_' + second] = (compare(arrays[first], arrays[second], np)
                if first in arrays and second in arrays else {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'})
        write(run / 'summary.json', {'status': 'DFL_ORIGINAL_BATCH_ABORTED' if aborted else 'DFL_ORIGINAL_BATCH_COMPLETED',
                                     'cases': results, 'cross_native_comparisons': comparisons,
                                     'requested_case_order': [row[0] for row in CASES],
                                     'observed_case_count': len(results), 'successful_output_case_count': len(arrays),
                                     'elapsed_seconds_not_benchmark': time.monotonic() - started,
                                     'aborted': aborted, 'batch_completion_is_not_numerical_pass': True,
                                     'diagnostic_only': True, 'dataset_evaluation_run': False,
                                     'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'})


if __name__ == '__main__':
    main()

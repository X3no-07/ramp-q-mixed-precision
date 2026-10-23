"""Frozen four-node DFL weight-QDQ bypass precision-flag diagnosis, Python 3.8 / TRT 8.5.

Compilation failures are observations and do not prevent the next frozen case.
Timeouts and outer interruptions abort the batch after child cleanup. This is
one frozen CPU Softmax-probability diagnostic, not a detector accuracy or numerical pass.
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


SCHEMA = 'rampq.gate05.dfl_weight_bypass_probe/v1'
PARENT_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
PARENT_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
PARENT_CONV_OUTPUT_SPEC_SHA = '9e9d31b55e4358d39a08a7869aab77582f094f1d6152599c727196e2b77c3897'
PARENT_CONV_OUTPUT_MANIFEST_SHA = '104137a621b3ea5c65a019555a1d92fc8c876fe0bf9293b5ab7cfe2e6fe71f77'
PARENT_CONV_OUTPUT_CHECKPOINT_SHA = 'c5e4a11a37d32a6de9df22c42463de0a6a58ad4b92c97b64f4c318f71dd9947e'
PARENT_CONV_OUTPUT_GRAPH_SHA = '805abb78ed4e4d5abc8041b2916dadc83370a14ef1776fcc0a03f163c16bcf81'
PROTOCOL_SHA = '9b2ea26ff12ecd5a4b5a699b09c511a3ea6c2187125d037117bb85f57b7b1170'
CPU_OUTPUTS_SHA = 'dff4725f1dbbb99979723b47267973f9cde5bc390db30131ad5d39361d958526'
INPUT_FIXTURE_SHA = '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520'
MODEL_SHAS = {
    'weight_bypass': '020dad7f70c04f495dfe1b6e62e63e925ff6468a6cde0c157b7122a38274eb41',
}
CASES = (('weight_bypass_none', 'weight_bypass', 'none', True),
         ('weight_bypass_obey', 'weight_bypass', 'obey', True))
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
    """Stdlib-only byte verification; no runtime or inference imports."""
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema'] == 'rampq.gate05.dfl_weight_bypass_probe_bundle/v1', 'Wrong manifest')
    for name, info in manifest['files'].items():
        path = contained(root, name)
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == info['bytes'] and
                sha(path) == info['sha256'], 'Bad manifest member: ' + name)
    for name in ('spec.json', 'parent_spec.json', 'parent_manifest.json', 'protocol.json',
                 'source_model.onnx', 'failure_inventory.json', 'parent_conv_output_spec.json',
                 'parent_conv_output_manifest.json', 'parent_conv_output_checkpoint.json',
                 'parent_conv_output_model.onnx', 'fixture_lineage.json', 'graph_identity_audit.json',
                 'cpu_diagnostic.json', 'jetson_dfl_weight_bypass_probe.py',
                 'jetson_build_dfl_weight_bypass_probe.py'):
        require(name in manifest['files'], 'Unsealed required file: ' + name)
    pins = {'parent_spec.json': PARENT_SPEC_SHA, 'parent_manifest.json': PARENT_MANIFEST_SHA,
            'parent_conv_output_spec.json': PARENT_CONV_OUTPUT_SPEC_SHA,
            'parent_conv_output_manifest.json': PARENT_CONV_OUTPUT_MANIFEST_SHA,
            'parent_conv_output_checkpoint.json': PARENT_CONV_OUTPUT_CHECKPOINT_SHA,
            'parent_conv_output_model.onnx': PARENT_CONV_OUTPUT_GRAPH_SHA, 'protocol.json': PROTOCOL_SHA,
            'source_model.onnx': '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee',
            'failure_inventory.json': 'a2780d83f1de24ab166553a8a3d341f04cf3c6f74ae41cabd5885124bc81d112'}
    for name, expected in pins.items():
        require(sha(root / name) == expected, 'Frozen source changed: ' + name)
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    spec = json.loads((root / 'spec.json').read_text(encoding='utf-8'))
    parent = json.loads((root / 'parent_spec.json').read_text(encoding='utf-8'))
    require(protocol['schema'] == 'rampq.gate05.dfl_weight_bypass_protocol/v1' and spec['schema'] == SCHEMA,
            'Wrong protocol/spec schema')
    require(spec['max_seconds'] == protocol['batch_max_seconds'] == MAX_BATCH_SECONDS and
            spec['max_build_seconds'] == protocol['per_case_max_build_seconds'] == MAX_CHILD_SECONDS,
            'Diagnostic budget changed')
    require(spec['maximum_native_builds'] == protocol['maximum_builds'] == 2 and
            spec['maximum_native_inferences'] == protocol['maximum_executions'] == 4 and
            spec['executions_per_successful_case'] == protocol['common_native_settings']['repeat_executions_if_success'] == 2,
            'Frozen build/execution count changed')
    require(spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
            spec['new_CPU_execution_runs_this_stage'] == protocol['cpu_composition_check']['total_new_CPU_runs'] ==
            protocol['cpu_composition_check']['candidate_CPU_runs'] == 1 and
            spec['new_probability_fixture_generation_runs'] == 0 and
            spec['new_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0, 'Preparation/guard scope changed')
    require(parent['configuration'] == 'q_all' and parent['graph_sha256'] == protocol['source_graph_sha256'] and
            PARENT_SPEC_SHA == protocol['source_spec_sha256'] and PARENT_MANIFEST_SHA == protocol['source_manifest_sha256'],
            'Wrong full failed q_all parent')
    for field, pin in (('parent_conv_output_spec_sha256', PARENT_CONV_OUTPUT_SPEC_SHA),
                       ('parent_conv_output_manifest_sha256', PARENT_CONV_OUTPUT_MANIFEST_SHA),
                       ('parent_conv_output_checkpoint_sha256', PARENT_CONV_OUTPUT_CHECKPOINT_SHA),
                       ('parent_conv_output_graph_sha256', PARENT_CONV_OUTPUT_GRAPH_SHA)):
        require(spec[field] == protocol[field] == pin, 'Prior six-node stage differs: ' + field)
    require(INPUT_FIXTURE_SHA == protocol['saved_probability_fixture_sha256'], 'Frozen probability input differs')
    require(spec['conv_names'] == protocol['source_conv_names'] == ['/model.23/dfl/conv/Conv'] and
            spec['clip_names'] == protocol['clip_names'] == ['rampq_model_23_dfl_conv_Clip'], 'Wrong Conv/Clip boundary')
    require(spec['input_file'] == protocol['input_file'] == 'dfl_softmax_probabilities.npy' and
            spec['input_file_sha256'] == INPUT_FIXTURE_SHA and
            spec['input_binding'] == {'name': '/model.23/dfl/Softmax_output_0', 'input': True,
                                    'shape': [1, 16, 4, 8400], 'dtype': 'DataType.FLOAT'} and
            protocol['input_name'] == spec['input_binding']['name'] and
            protocol['input_shape'] == spec['input_binding']['shape'], 'Wrong frozen probability binding')
    require(len(spec['output_bindings_in_order']) == 1, 'Unexpected output taps')
    output = spec['output_bindings_in_order'][0]
    require(output['name'] == protocol['output_name'] == '/model.23/dfl/conv/Conv_output_0' and
            output['shape'] == protocol['output_shape'] == [1, 1, 4, 8400] and
            output['input'] is False and output['dtype'] == 'DataType.FLOAT', 'Wrong FLOAT4D output contract')
    require(spec['cpu_expected_outputs_sha256'] == CPU_OUTPUTS_SHA and spec['own_candidate_CPU_reference_required'] is True,
            'Own new candidate CPU reference changed')
    diagnostic = json.loads((root / 'cpu_diagnostic.json').read_text(encoding='utf-8'))
    require(diagnostic['status'] == 'OWN_ONE_RUN_CPU_REFERENCE_ESTABLISHED' and diagnostic['new_CPU_executions'] == 1 and
            diagnostic['actual_output']['shape'] == [1, 1, 4, 8400] and diagnostic['actual_output']['dtype'] == 'float32' and
            diagnostic['actual_output']['all_finite'] is True and diagnostic['expected_key'] == 'weight_bypass_output' and
            diagnostic['old_reference_adopted_as_candidate_expected'] is False and
            diagnostic['old_reference_comparison_descriptive_only']['equality_required'] is False,
            'Own-reference numerical semantics changed')
    for file_key, hash_key in (('input_file', 'input_file_sha256'),
                               ('cpu_expected_outputs_file', 'cpu_expected_outputs_sha256'),
                               ('cpu_diagnostic_file', 'cpu_diagnostic_sha256'),
                               ('fixture_lineage_file', 'fixture_lineage_sha256'),
                               ('graph_identity_audit_file', 'graph_identity_audit_sha256'),
                               ('native_runner_file', 'native_runner_sha256'),
                               ('native_builder_file', 'native_builder_sha256')):
        require(spec[file_key] in manifest['files'] and sha(contained(root, spec[file_key])) == spec[hash_key],
                'Frozen evidence changed: ' + file_key)
    require(set(spec['variants']) == set(MODEL_SHAS), 'Variant set changed')
    for name, expected_sha in MODEL_SHAS.items():
        variant = spec['variants'][name]
        require(variant['model_file'] in manifest['files'] and
                sha(contained(root, variant['model_file'])) == expected_sha == variant['model_sha256'], 'Frozen graph changed')
        require(variant['nodes'] == protocol['graph_node_count'] == 4 and
                variant['initializers'] == protocol['initializer_count'] == 5 and
                variant['expected_output_npz_keys'] == {protocol['output_name']: 'weight_bypass_output'}, 'Wrong candidate inventory/key')
    require(spec['removed_node_names'] == protocol['removed_node_names'] ==
            ['rampq_model_23_dfl_conv_WeightQ', 'rampq_model_23_dfl_conv_WeightDQ'] and
            spec['declared_node_mutation'] == protocol['declared_node_mutation'] ==
            {'node': '/model.23/dfl/conv/Conv', 'field': 'input[1]', 'old': 'rampq_model_23_dfl_conv_w_dq',
             'new': 'model.23.dfl.conv.weight'} and spec['semantic_change'] == protocol['semantic_change'],
            'Single weight-QDQ bypass definition changed')
    require(spec['cases'] == protocol['cases'] and len(spec['cases']) == len(CASES), 'Frozen cases changed')
    for row, (case, variant, flag, correction) in zip(spec['cases'], CASES):
        require(row['id'] == case and row['variant'] == variant and row['precision_constraint_flag'] == flag and
                row['apply_native_clip_attribute_correction'] is correction and
                row['max_build_seconds'] == MAX_CHILD_SECONDS and row['iterations'] == 1, 'Frozen case differs')
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
          'int8_output_required': False, 'output_qdq_present': False, 'explicit_weight_qdq_present': False,
          'source_FP32_weight_does_not_prove_actual_FP32_weight_arithmetic': True})
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
    result = {'status': 'DFL_WEIGHT_BYPASS_FIXED_INPUT_OUTPUTS_SAVED', 'case': case['id'],
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
              'output_role': 'DFL_WEIGHT_BYPASS_DISTANCES_WITH_SINGLETON_CHANNEL_NOT_DECODED_BOXES',
              'fixture_role': 'FROZEN_CPU_SOFTMAX_PROBABILITIES_NOT_LOGITS_OR_NATIVE_SOFTMAX_OUTPUT',
              'actual_conv_output_int8_required': False,
              'explicit_weight_qdq_removed': True, 'own_candidate_CPU_reference_used': True,
              'actual_weight_precision_not_inferred_from_source_dtype': True,
              'limits': 'Four-node weight-QDQ bypass changes numerical semantics and parser context; own candidate CPU reference required. Activation Q/Clip/DQ, probability input and source weight bytes retained. FP32 initializer does not prove actual FP32 arithmetic. No unique full-model cause, INT8-output acceptance rule or new tolerance.'}
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
        print(json.dumps({'status': 'FROZEN_DFL_WEIGHT_BYPASS_INPUTS_VERIFIED', 'spec_sha256': sha(root / 'spec.json'),
                          'case_count': len(CASES), 'native_runtime_imported': False}), flush=True)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Native Jetson required')
    import numpy as np
    import tensorrt as trt
    import torch
    require(trt.__version__.startswith('8.5.') and torch.cuda.is_available(), 'Wrong native runtime')
    feature_input = np.load(str(root / spec['input_file']), allow_pickle=False)
    require(feature_input.dtype == np.float32 and list(feature_input.shape) == spec['input_binding']['shape'] and
            bool(np.isfinite(feature_input).all()) and float(feature_input.min()) >= 0.0 and
            float(feature_input.max()) <= 1.0, 'DFL Softmax probability input domain contract')
    with np.load(str(root / spec['cpu_expected_outputs_file']), allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    require(set(expected) == {'weight_bypass_output'}, 'Expected NPZ must contain only the new four-dimensional reference')
    require(expected['weight_bypass_output'].dtype == np.float32 and expected['weight_bypass_output'].shape == (1, 1, 4, 8400) and
            bool(np.isfinite(expected['weight_bypass_output']).all()), 'Old three-dimensional expected array is forbidden')
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
            command = [sys.executable, '-u', str(root / 'jetson_build_dfl_weight_bypass_probe.py'),
                       '--root', str(root), '--case', current, '--output', str(out / 'model.engine')]
            write(out / 'request.json', {'command': command, 'case': case,
                                         'spec_sha256': sha(root / 'spec.json'),
                                         'model_sha256': spec['variants'][case['variant']]['model_sha256'],
                                         'max_child_seconds': MAX_CHILD_SECONDS, 'retries': 0})
            print('DFL_WEIGHT_BYPASS_BUILD_START ' + current, flush=True)
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
                print('DFL_WEIGHT_BYPASS_BUILD_FAILED_CONTINUE ' + current, flush=True)
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
            print('DFL_WEIGHT_BYPASS_CASE_FINISHED ' + current, flush=True)
    except BaseException as error:
        aborted = {'case': current, 'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc()}
        raise
    finally:
        pairs = (('weight_bypass_none', 'weight_bypass_obey'),)
        comparisons = {}
        for first, second in pairs:
            comparisons[first + '_vs_' + second] = (compare(arrays[first], arrays[second], np)
                if first in arrays and second in arrays else {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'})
        write(run / 'summary.json', {'status': 'DFL_WEIGHT_BYPASS_BATCH_ABORTED' if aborted else 'DFL_WEIGHT_BYPASS_BATCH_COMPLETED',
                                     'cases': results, 'cross_native_comparisons': comparisons,
                                     'requested_case_order': [row[0] for row in CASES],
                                     'observed_case_count': len(results), 'successful_output_case_count': len(arrays),
                                     'elapsed_seconds_not_benchmark': time.monotonic() - started,
                                     'aborted': aborted, 'batch_completion_is_not_numerical_pass': True,
                                     'diagnostic_only': True, 'dataset_evaluation_run': False,
                                     'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'})


if __name__ == '__main__':
    main()
